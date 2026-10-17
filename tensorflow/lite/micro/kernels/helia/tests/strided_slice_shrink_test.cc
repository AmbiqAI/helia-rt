/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// int8 and int16 STRIDED_SLICE through the helia kernel, compared bit for bit
// with the reference: shrunk axes in any position, empty slices, negative
// strides through masks and extreme strides. The reference overflows on an
// extreme stride from a nonzero begin, so those cases name the one input
// element they take. GCC and ATfE links wrap the entries and assert each
// invoke ran its own once, or none when empty. see AmbiqAI/helia-rt#435

#include <cstdint>
#include <cstring>
#include <limits>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/reference/strided_slice.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/strided_slice.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

enum class Leaf { kS8, kS16, kCount };

int g_leaf_calls[static_cast<int>(Leaf::kCount)] = {};

}  // namespace

#if HELIA_STRIDED_SLICE_SHRINK_LINK_WRAP
extern "C" {

#define HELIA_WRAP_SLICE(NAME, T, LEAF)                                       \
  arm_cmsis_nn_status __real_##NAME(const T*, T*, const cmsis_nn_dims* const, \
                                    const cmsis_nn_dims* const,               \
                                    const cmsis_nn_dims* const,               \
                                    const cmsis_nn_dims* const);              \
  arm_cmsis_nn_status __wrap_##NAME(                                          \
      const T* input, T* output, const cmsis_nn_dims* const input_dims,       \
      const cmsis_nn_dims* const begin_dims,                                  \
      const cmsis_nn_dims* const stride_dims,                                 \
      const cmsis_nn_dims* const output_dims) {                               \
    ++g_leaf_calls[static_cast<int>(LEAF)];                                   \
    return __real_##NAME(input, output, input_dims, begin_dims, stride_dims,  \
                         output_dims);                                        \
  }

HELIA_WRAP_SLICE(arm_strided_slice_s8, int8_t, Leaf::kS8)
HELIA_WRAP_SLICE(arm_strided_slice_s16, int16_t, Leaf::kS16)

#undef HELIA_WRAP_SLICE

}  // extern "C"
#endif  // HELIA_STRIDED_SLICE_SHRINK_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

constexpr int kMaxRank = 4;
constexpr int kMaxElements = 2 * 3 * 4;

struct Slice {
  int rank;
  int32_t shape[kMaxRank];
  int32_t begin[kMaxRank];
  int32_t end[kMaxRank];
  int32_t strides[kMaxRank];
  int begin_mask;
  int end_mask;
  int shrink_axis_mask;
};

template <typename D>
int ElementsOf(const D* dims, int rank) {
  int count = 1;
  for (int i = 0; i < rank; ++i) {
    count *= dims[i];
  }
  return count;
}

// Output dims after dropping shrunk axes, counted by stepping through each
// axis in 64 bits so they do not share the kernel's arithmetic. Clears
// *visits_any when some axis, shrunk or not, visits nothing.
int OutputShape(const Slice& s, int* out_dims, bool* visits_any) {
  using micro::strided_slice::StridedSliceEndForAxis;
  using micro::strided_slice::StridedSliceStartForAxis;
  micro::StridedSliceParams params = {};
  params.start_indices_count = s.rank;
  params.stop_indices_count = s.rank;
  params.strides_count = s.rank;
  for (int i = 0; i < s.rank; ++i) {
    params.start_indices[i] = s.begin[i];
    params.stop_indices[i] = s.end[i];
    params.strides[i] = s.strides[i];
  }
  params.begin_mask = s.begin_mask;
  params.end_mask = s.end_mask;
  params.shrink_axis_mask = s.shrink_axis_mask;
  const RuntimeShape input_shape(s.rank, s.shape);
  int out_rank = 0;
  *visits_any = true;
  for (int axis = 0; axis < s.rank; ++axis) {
    const int start = StridedSliceStartForAxis(params, input_shape, axis);
    const int stop = StridedSliceEndForAxis(params, input_shape, axis, start);
    const int64_t step = s.strides[axis];
    int count = 0;
    for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) {
      ++count;
    }
    *visits_any = *visits_any && count > 0;
    if ((s.shrink_axis_mask & (1 << axis)) == 0) {
      out_dims[out_rank++] = count;
    }
  }
  return out_rank;
}

micro::StridedSliceParams ReferenceParams(const Slice& s) {
  micro::StridedSliceParams params = {};
  params.start_indices_count = s.rank;
  params.stop_indices_count = s.rank;
  params.strides_count = s.rank;
  for (int i = 0; i < s.rank; ++i) {
    params.start_indices[i] = s.begin[i];
    params.stop_indices[i] = s.end[i];
    params.strides[i] = s.strides[i];
  }
  params.begin_mask = s.begin_mask;
  params.end_mask = s.end_mask;
  params.shrink_axis_mask = s.shrink_axis_mask;
  return params;
}

// Distinct per-element values, so a misplaced element shows.
template <typename T>
void FillInput(T* input, int count) {
  for (int i = 0; i < count; ++i) {
    input[i] = static_cast<T>(i + 1 - count / 2);
  }
}

// Runs one slice for element type T stored as TfLiteType `type` and checks it
// against the reference. `leaf` is the entry the invoke must reach.
// With `only` >= 0 the slice must take exactly input element `only`, which is
// checked without the reference.
template <typename T>
void ExpectMatchesReference(const Slice& s, TfLiteType type, Leaf leaf,
                            int only = -1) {
  static T input[kMaxElements];
  static T output[kMaxElements];
  static T expected[kMaxElements];

  int output_dims_data[1 + kMaxRank] = {};
  bool visits_any = true;
  output_dims_data[0] = OutputShape(s, &output_dims_data[1], &visits_any);
  const int input_count = ElementsOf(s.shape, s.rank);
  EXPECT_LE(input_count, kMaxElements);
  if (input_count > kMaxElements) {
    return;
  }
  FillInput<T>(input, input_count);

  int input_dims_data[1 + kMaxRank] = {s.rank};
  for (int i = 0; i < s.rank; ++i) {
    input_dims_data[1 + i] = static_cast<int>(s.shape[i]);
  }
  int index_dims_data[] = {1, s.rank};
  TfLiteTensor tensors[5] = {
      CreateTensor(input, IntArrayFromInts(input_dims_data), false, type),
      CreateTensor(s.begin, IntArrayFromInts(index_dims_data)),
      CreateTensor(s.end, IntArrayFromInts(index_dims_data)),
      CreateTensor(s.strides, IntArrayFromInts(index_dims_data)),
      CreateTensor(output, IntArrayFromInts(output_dims_data), false, type),
  };
  tensors[kStridedSliceBeginTensor].allocation_type = kTfLiteMmapRo;
  tensors[kStridedSliceEndTensor].allocation_type = kTfLiteMmapRo;
  tensors[kStridedSliceStridesTensor].allocation_type = kTfLiteMmapRo;

  TfLiteStridedSliceParams params = {};
  params.begin_mask = s.begin_mask;
  params.end_mask = s.end_mask;
  params.shrink_axis_mask = s.shrink_axis_mask;

  int inputs[] = {4, 0, 1, 2, 3};
  int outputs[] = {1, 4};
  const TFLMRegistration registration = Register_STRIDED_SLICE();
  micro::KernelRunner runner(registration, tensors, 5, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  memset(output, 0x5A, sizeof(output));
  memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
#if HELIA_STRIDED_SLICE_SHRINK_LINK_WRAP
  for (int i = 0; i < static_cast<int>(Leaf::kCount); ++i) {
    const bool expect_call = i == static_cast<int>(leaf) && visits_any;
    EXPECT_EQ(expect_call ? 1 : 0, g_leaf_calls[i]);
  }
#else
  (void)leaf;
#endif

  int32_t output_shape[kMaxRank] = {};
  for (int i = 0; i < output_dims_data[0]; ++i) {
    output_shape[i] = output_dims_data[1 + i];
  }
  memset(expected, 0x5A, sizeof(expected));
  if (only >= 0) {
    expected[0] = input[only];
  } else {
    micro::reference_ops::StridedSlice(
        ReferenceParams(s), RuntimeShape(s.rank, s.shape), input,
        RuntimeShape(output_dims_data[0], output_shape), expected);
  }
  EXPECT_EQ(0, memcmp(expected, output, sizeof(output)));
}

void ExpectAllTypesMatch(const Slice& s, int only = -1) {
  ExpectMatchesReference<int8_t>(s, kTfLiteInt8, Leaf::kS8, only);
  ExpectMatchesReference<int16_t>(s, kTfLiteInt16, Leaf::kS16, only);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ExpectAllTypesMatch;
using tflite::testing::Slice;

// A shrunk middle axis must keep its place in the heliaCORE dims: [2, 3, 4]
// sliced to row 1 of every batch is [2, 4], not the first two rows.
TEST(HeliaStridedSliceShrinkTest, ShrinkMiddleAxis) {
  ExpectAllTypesMatch(
      Slice{3, {2, 3, 4}, {0, 1, 0}, {2, 2, 4}, {1, 1, 1}, 0, 0, 0b010});
}

TEST(HeliaStridedSliceShrinkTest, ShrinkInnerAxis) {
  ExpectAllTypesMatch(
      Slice{3, {2, 3, 4}, {0, 0, 2}, {2, 3, 3}, {1, 1, 1}, 0, 0, 0b100});
}

// begin == end on one axis: nothing to copy, and no heliaCORE call.
TEST(HeliaStridedSliceShrinkTest, EmptySliceWritesNothing) {
  ExpectAllTypesMatch(Slice{2, {4, 4}, {2, 0}, {2, 4}, {1, 1}, 0, 0, 0});
}

// The per-axis element count must not overflow for extreme strides: each
// slice below takes one element.
TEST(HeliaStridedSliceShrinkTest, StrideInt32MaxTakesOneElement) {
  ExpectAllTypesMatch(Slice{
      1, {4}, {0}, {4}, {std::numeric_limits<int32_t>::max()}, 0, 0, 0});
}

TEST(HeliaStridedSliceShrinkTest, StrideInt32MinTakesOneElement) {
  ExpectAllTypesMatch(Slice{
      1, {4}, {3}, {0}, {std::numeric_limits<int32_t>::min()}, 0, 0, 0});
}

// begin + stride passes INT32_MAX: the single element is input[1], then
// input[2].
TEST(HeliaStridedSliceShrinkTest, StrideInt32MaxFromNonzeroBegin) {
  ExpectAllTypesMatch(
      Slice{1, {4}, {1}, {4}, {std::numeric_limits<int32_t>::max()}, 0, 0, 0},
      /*only=*/1);
  ExpectAllTypesMatch(Slice{1, {4}, {2}, {4},
                            {std::numeric_limits<int32_t>::max() - 1}, 0, 0, 0},
                      /*only=*/2);
}

// Negative strides through begin/end masks: the reversed input, every other
// column.
TEST(HeliaStridedSliceShrinkTest, NegativeStridesWithMasks) {
  ExpectAllTypesMatch(
      Slice{2, {3, 4}, {0, 0}, {0, 0}, {-1, -2}, 0b11, 0b11, 0});
}

// 4-D input with a shrunk leading axis, a masked axis and stride 2.
TEST(HeliaStridedSliceShrinkTest, FourDShrinkLeadingAxisWithMasks) {
  ExpectAllTypesMatch(Slice{4, {2, 2, 3, 2}, {1, 0, 0, 0}, {2, 0, 3, 2},
                            {1, 1, 2, 1}, 0b0010, 0b0010, 0b0001});
}

// Prepare checks the output shape with a float ceil, which rounds above 2^24
// elements on an axis, so a hostile model can declare an output that does not
// hold the slice. Such an axis cannot be allocated here; instead Prepare sees
// the true [2, 4] output and Invoke a [1, `elements`] one. Every type must
// fail the op without reaching heliaCORE or writing past the tensor; `leaf`
// is kCount for the types whose entries this test does not wrap.
template <typename T>
void ExpectWrongOutputSizeFails(TfLiteType type, Leaf leaf, int elements) {
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;
  T input[24];
  for (int i = 0; i < 24; ++i) {
    input[i] = static_cast<T>(i + 1);
  }
  T output[9];
  const int32_t begin[] = {0, 1, 0};
  const int32_t end[] = {2, 2, 4};
  const int32_t strides[] = {1, 1, 1};
  int input_dims[] = {3, 2, 3, 4};
  int index_dims[] = {1, 3};
  int output_dims[] = {2, 2, 4};
  TfLiteTensor tensors[5] = {
      CreateTensor(input, IntArrayFromInts(input_dims), false, type),
      CreateTensor(begin, IntArrayFromInts(index_dims)),
      CreateTensor(end, IntArrayFromInts(index_dims)),
      CreateTensor(strides, IntArrayFromInts(index_dims)),
      CreateTensor(output, IntArrayFromInts(output_dims), false, type),
  };
  for (int i = 1; i <= 3; ++i) {
    tensors[i].allocation_type = kTfLiteMmapRo;
  }
  TfLiteStridedSliceParams params = {};
  params.shrink_axis_mask = 0b010;
  int inputs[] = {4, 0, 1, 2, 3};
  int outputs[] = {1, 4};
  const TFLMRegistration registration = tflite::Register_STRIDED_SLICE();
  tflite::micro::KernelRunner runner(registration, tensors, 5,
                                     IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), &params);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  output_dims[1] = 1;
  output_dims[2] = elements;
  memset(output, 0x5A, sizeof(output));
  memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
  EXPECT_EQ(kTfLiteError, runner.Invoke());
#if HELIA_STRIDED_SLICE_SHRINK_LINK_WRAP
  for (int i = 0; i < static_cast<int>(Leaf::kCount); ++i) {
    EXPECT_EQ(0, g_leaf_calls[i]);
  }
#endif
  (void)leaf;
  const int kept = elements < 8 ? elements : 8;
  for (size_t i = kept * sizeof(T); i < sizeof(output); ++i) {
    EXPECT_EQ(0x5A, reinterpret_cast<const uint8_t*>(output)[i]);
  }
}

TEST(HeliaStridedSliceShrinkTest, OutputShorterThanSliceFailsInvoke) {
  ExpectWrongOutputSizeFails<int8_t>(kTfLiteInt8, Leaf::kS8, 7);
  ExpectWrongOutputSizeFails<int16_t>(kTfLiteInt16, Leaf::kS16, 7);
  ExpectWrongOutputSizeFails<float>(kTfLiteFloat32, Leaf::kCount, 7);
  ExpectWrongOutputSizeFails<int32_t>(kTfLiteInt32, Leaf::kCount, 7);
  ExpectWrongOutputSizeFails<bool>(kTfLiteBool, Leaf::kCount, 7);
}

// The check is for equality, so a larger output fails too.
TEST(HeliaStridedSliceShrinkTest, OutputLargerThanSliceFailsInvoke) {
  ExpectWrongOutputSizeFails<int8_t>(kTfLiteInt8, Leaf::kS8, 9);
  ExpectWrongOutputSizeFails<float>(kTfLiteFloat32, Leaf::kCount, 9);
}

TF_LITE_MICRO_TESTS_MAIN
