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

// STRIDED_SLICE through the helia kernel, compared bit for bit with the
// reference for float32, int8, int16, int32 and, on fp16 builds, float16. The
// slices cover shrunk inner axes, negative strides, begin/end masks and an
// empty result. GCC and ATfE links wrap the heliaCORE entries and assert each
// invoke ran exactly the one for its type, or none for an empty result.

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

enum class Leaf { kF32, kS8, kS16, kS32, kF16, kCount };

int g_leaf_calls[static_cast<int>(Leaf::kCount)] = {};

}  // namespace

#if HELIA_STRIDED_SLICE_LINK_WRAP
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

HELIA_WRAP_SLICE(arm_strided_slice_f32, float32_t, Leaf::kF32)
HELIA_WRAP_SLICE(arm_strided_slice_s8, int8_t, Leaf::kS8)
HELIA_WRAP_SLICE(arm_strided_slice_s16, int16_t, Leaf::kS16)
HELIA_WRAP_SLICE(arm_strided_slice_s32, int32_t, Leaf::kS32)
#if ARM_NN_ENABLE_F16
HELIA_WRAP_SLICE(arm_strided_slice_f16, float16_t, Leaf::kF16)
#endif

#undef HELIA_WRAP_SLICE

}  // extern "C"
#endif  // HELIA_STRIDED_SLICE_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

constexpr int kMaxRank = 4;
constexpr int kMaxElements = 2 * 5 * 6 * 3;

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

// Output dims after dropping shrunk axes, as StridedSlicePrepare checks them.
int OutputShape(const Slice& s, int* out_dims) {
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
  for (int axis = 0; axis < s.rank; ++axis) {
    const int start = StridedSliceStartForAxis(params, input_shape, axis);
    const int stop = StridedSliceEndForAxis(params, input_shape, axis, start);
    const int step = s.strides[axis];
    int count = step > 0 ? (stop - start + step - 1) / step
                         : (start - stop - step - 1) / -step;
    count = count < 0 ? 0 : count;
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

// Distinct per-element bit patterns; float32 lanes include a NaN with a
// payload and -0.0f, which a copy must carry through unchanged.
template <typename T>
void FillInput(T* input, int count) {
  for (int i = 0; i < count; ++i) {
    input[i] = static_cast<T>(i + 1 - count / 2);
  }
}

template <>
void FillInput<float>(float* input, int count) {
  for (int i = 0; i < count; ++i) {
    input[i] = 0.25f * static_cast<float>(i) - 7.0f;
  }
  const uint32_t nan_bits = 0xFFC12345u;
  if (count > 3) {
    std::memcpy(&input[3], &nan_bits, sizeof(nan_bits));
  }
  if (count > 5) {
    input[5] = -0.0f;
  }
}

#if ARM_NN_ENABLE_F16
template <>
void FillInput<uint16_t>(uint16_t* input, int count) {
  for (int i = 0; i < count; ++i) {
    // Finite binary16 patterns of both signs.
    input[i] = static_cast<uint16_t>((i % 2 ? 0x8000u : 0u) | (0x3C00u + i));
  }
  if (count > 3) {
    input[3] = 0xFE01u;  // NaN with a payload
  }
}
#endif

// Runs one slice for element type T stored as TfLiteType `type` and checks it
// against the reference. `leaf` is the entry the invoke must reach.
template <typename T>
void ExpectMatchesReference(const Slice& s, TfLiteType type, Leaf leaf) {
  static T input[kMaxElements];
  static T output[kMaxElements];
  static T expected[kMaxElements];

  int output_dims_data[1 + kMaxRank] = {};
  output_dims_data[0] = OutputShape(s, &output_dims_data[1]);
  const int input_count = ElementsOf(s.shape, s.rank);
  const int output_count =
      ElementsOf(&output_dims_data[1], output_dims_data[0]);
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
#if HELIA_STRIDED_SLICE_LINK_WRAP
  for (int i = 0; i < static_cast<int>(Leaf::kCount); ++i) {
    const bool expect_call = i == static_cast<int>(leaf) && output_count > 0;
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
  micro::reference_ops::StridedSlice(
      ReferenceParams(s), RuntimeShape(s.rank, s.shape), input,
      RuntimeShape(output_dims_data[0], output_shape), expected);
  EXPECT_EQ(0, memcmp(expected, output, sizeof(output)));
}

void ExpectAllTypesMatch(const Slice& s) {
  ExpectMatchesReference<float>(s, kTfLiteFloat32, Leaf::kF32);
  ExpectMatchesReference<int8_t>(s, kTfLiteInt8, Leaf::kS8);
  ExpectMatchesReference<int16_t>(s, kTfLiteInt16, Leaf::kS16);
  ExpectMatchesReference<int32_t>(s, kTfLiteInt32, Leaf::kS32);
#if ARM_NN_ENABLE_F16
  ExpectMatchesReference<uint16_t>(s, kTfLiteFloat16, Leaf::kF16);
#endif
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ExpectAllTypesMatch;
using tflite::testing::Slice;

// A shrunk middle axis must keep its place in the heliaCORE dims: [2, 3, 4]
// sliced to row 1 of every batch is [2, 4], not the first two rows.
TEST(HeliaStridedSliceRouteTest, ShrinkMiddleAxis) {
  ExpectAllTypesMatch(
      Slice{3, {2, 3, 4}, {0, 1, 0}, {2, 2, 4}, {1, 1, 1}, 0, 0, 0b010});
}

TEST(HeliaStridedSliceRouteTest, ShrinkInnerAxis) {
  ExpectAllTypesMatch(
      Slice{3, {2, 3, 4}, {0, 0, 2}, {2, 3, 3}, {1, 1, 1}, 0, 0, 0b100});
}

TEST(HeliaStridedSliceRouteTest, NegativeStrides) {
  ExpectAllTypesMatch(
      Slice{3, {3, 4, 5}, {2, 0, 4}, {0, 4, 0}, {-1, 2, -2}, 0, 0, 0});
}

TEST(HeliaStridedSliceRouteTest, Rank4MasksAndStrideThree) {
  ExpectAllTypesMatch(Slice{4,
                            {2, 5, 6, 3},
                            {1, 3, 1, 0},
                            {2, 5, 0, 3},
                            {1, 2, 3, 1},
                            0b0010,
                            0b0100,
                            0});
}

TEST(HeliaStridedSliceRouteTest, Rank1NegativeIndices) {
  ExpectAllTypesMatch(Slice{1, {10}, {-7}, {-1}, {2}, 0, 0, 0});
}

TEST(HeliaStridedSliceRouteTest, ShrinkEveryAxisToScalar) {
  ExpectAllTypesMatch(Slice{2, {3, 4}, {1, 2}, {2, 3}, {1, 1}, 0, 0, 0b11});
}

// begin == end on one axis: nothing to copy, and no heliaCORE call.
TEST(HeliaStridedSliceRouteTest, EmptySliceWritesNothing) {
  ExpectAllTypesMatch(Slice{2, {4, 4}, {2, 0}, {2, 4}, {1, 1}, 0, 0, 0});
}

// Float16 has no reference kernel, so a build without the heliaCORE fp16 API
// must reject it when the model is prepared rather than when it runs.
TEST(HeliaStridedSliceRouteTest, Float16PrepareNeedsTheFp16Api) {
  uint16_t input[4] = {0x3C00, 0x4000, 0x4200, 0x4400};
  uint16_t output[2] = {};
  const int32_t begin[] = {1};
  const int32_t end[] = {3};
  const int32_t strides[] = {1};
  int input_dims[] = {1, 4};
  int index_dims[] = {1, 1};
  int output_dims[] = {1, 2};
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;
  TfLiteTensor tensors[5] = {
      CreateTensor(input, IntArrayFromInts(input_dims), false, kTfLiteFloat16),
      CreateTensor(begin, IntArrayFromInts(index_dims)),
      CreateTensor(end, IntArrayFromInts(index_dims)),
      CreateTensor(strides, IntArrayFromInts(index_dims)),
      CreateTensor(output, IntArrayFromInts(output_dims), false,
                   kTfLiteFloat16),
  };
  for (int i = 1; i <= 3; ++i) {
    tensors[i].allocation_type = kTfLiteMmapRo;
  }
  TfLiteStridedSliceParams params = {};
  int inputs[] = {4, 0, 1, 2, 3};
  int outputs[] = {1, 4};
  const TFLMRegistration registration = tflite::Register_STRIDED_SLICE();
  tflite::micro::KernelRunner runner(registration, tensors, 5,
                                     IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), &params);
#if ARM_NN_ENABLE_F16
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  EXPECT_EQ(0x4000, output[0]);
  EXPECT_EQ(0x4200, output[1]);
#else
  EXPECT_EQ(kTfLiteError, runner.InitAndPrepare());
#endif
}

TF_LITE_MICRO_TESTS_MAIN
