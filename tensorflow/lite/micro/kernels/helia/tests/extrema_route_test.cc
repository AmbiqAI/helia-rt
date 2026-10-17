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

// ARG_MAX/ARG_MIN (float32, int8) and REDUCE_MAX/REDUCE_MIN through the helia
// kernels. ARG layers run twice in the same binary: at rank <= 4 they take
// heliaCORE, and padded to rank 5 they take the helia reference path; the two
// agree on every input, NaN included (AmbiqAI/helia-rt#359).
//
// Float32 REDUCE_MAX/REDUCE_MIN run arm_reduce_max/min_tflite_f32 at rank <= 4
// and the reference at rank 5; the two agree bit for bit, and the tests pin
// the TFLite results for NaN, infinities, empty windows and an empty axis
// list. Float16 runs the _tflite_f16 entries with the same semantics. GCC and
// ATfE links wrap the heliaCORE entries to prove each route.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

enum class Leaf {
  kArgMaxF32,
  kArgMinF32,
  kArgMaxS8,
  kArgMinS8,
  kReduceMaxTfliteF32,
  kReduceMinTfliteF32,
  kReduceMaxTfliteF16,
  kReduceMinTfliteF16,
  kReduceMaxS8,
  kReduceMaxS16,
  kCount
};

int g_leaf_calls[static_cast<int>(Leaf::kCount)] = {};

int CoreCalls() {
  int calls = 0;
  for (int count : g_leaf_calls) {
    calls += count;
  }
  return calls;
}

}  // namespace

#if HELIA_EXTREMA_LINK_WRAP
extern "C" {

#define HELIA_WRAP_ARG(NAME, T, LEAF)                                      \
  arm_cmsis_nn_status __real_##NAME(const T*, const cmsis_nn_dims*,        \
                                    int32_t, int32_t*);                    \
  arm_cmsis_nn_status __wrap_##NAME(const T* input,                        \
                                    const cmsis_nn_dims* input_dims,       \
                                    int32_t axis, int32_t* output) {       \
    ++g_leaf_calls[static_cast<int>(LEAF)];                                \
    return __real_##NAME(input, input_dims, axis, output);                 \
  }

#define HELIA_WRAP_REDUCE(NAME, T, LEAF)                                       \
  arm_cmsis_nn_status __real_##NAME(const T*, const cmsis_nn_dims*,            \
                                    const cmsis_nn_dims*, T*,                  \
                                    const cmsis_nn_dims*);                     \
  arm_cmsis_nn_status __wrap_##NAME(                                           \
      const T* input, const cmsis_nn_dims* input_dims,                         \
      const cmsis_nn_dims* axis_dims, T* output,                               \
      const cmsis_nn_dims* output_dims) {                                      \
    ++g_leaf_calls[static_cast<int>(LEAF)];                                    \
    return __real_##NAME(input, input_dims, axis_dims, output, output_dims);   \
  }

HELIA_WRAP_ARG(arm_argmax_f32, float32_t, Leaf::kArgMaxF32)
HELIA_WRAP_ARG(arm_argmin_f32, float32_t, Leaf::kArgMinF32)
HELIA_WRAP_ARG(arm_argmax_s8, int8_t, Leaf::kArgMaxS8)
HELIA_WRAP_ARG(arm_argmin_s8, int8_t, Leaf::kArgMinS8)
HELIA_WRAP_REDUCE(arm_reduce_max_tflite_f32, float32_t,
                  Leaf::kReduceMaxTfliteF32)
HELIA_WRAP_REDUCE(arm_reduce_min_tflite_f32, float32_t,
                  Leaf::kReduceMinTfliteF32)
HELIA_WRAP_REDUCE(arm_reduce_max_s8, int8_t, Leaf::kReduceMaxS8)
HELIA_WRAP_REDUCE(arm_reduce_max_s16, int16_t, Leaf::kReduceMaxS16)
#if ARM_NN_ENABLE_F16
HELIA_WRAP_REDUCE(arm_reduce_max_tflite_f16, float16_t,
                  Leaf::kReduceMaxTfliteF16)
HELIA_WRAP_REDUCE(arm_reduce_min_tflite_f16, float16_t,
                  Leaf::kReduceMinTfliteF16)
#endif

#undef HELIA_WRAP_REDUCE
#undef HELIA_WRAP_ARG

}  // extern "C"
#endif  // HELIA_EXTREMA_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

constexpr int kMaxElements = 2 * 3 * 4 * 5;
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kLowest = std::numeric_limits<float>::lowest();
constexpr float kMax = std::numeric_limits<float>::max();

uint32_t Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float FromBits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// One layer at a logical rank, and the same layer padded with leading unit
// dimensions to rank 5, which the helia kernels leave on their reference path.
struct Layer {
  int rank;
  int dims[5];
  int axis_count;
  int32_t axis[4];
};

void Promote(const Layer& in, Layer* out) {
  const int pad = 5 - in.rank;
  out->rank = 5;
  for (int i = 0; i < pad; ++i) {
    out->dims[i] = 1;
  }
  for (int i = 0; i < in.rank; ++i) {
    out->dims[pad + i] = in.dims[i];
  }
  out->axis_count = in.axis_count;
  for (int i = 0; i < in.axis_count; ++i) {
    out->axis[i] = in.axis[i] < 0 ? in.axis[i] : in.axis[i] + pad;
  }
}

int Normalized(int axis, int rank) { return axis < 0 ? axis + rank : axis; }

bool Reduced(const Layer& l, int dim) {
  for (int i = 0; i < l.axis_count; ++i) {
    if (Normalized(l.axis[i], l.rank) == dim) {
      return true;
    }
  }
  return false;
}

int Elements(const Layer& l) {
  int count = 1;
  for (int i = 0; i < l.rank; ++i) {
    count *= l.dims[i];
  }
  return count;
}

// ---- ARG_MAX / ARG_MIN ------------------------------------------------------

template <typename T>
TfLiteStatus RunArg(const Layer& l, bool is_max, TfLiteType type,
                    const T* input, int32_t* output, float scale = 1.0f) {
  int input_dims[6] = {l.rank};
  int output_dims[6] = {l.rank - 1};
  const int axis = Normalized(l.axis[0], l.rank);
  for (int i = 0, o = 0; i < l.rank; ++i) {
    input_dims[1 + i] = l.dims[i];
    if (i != axis) {
      output_dims[1 + o++] = l.dims[i];
    }
  }
  int axis_dims[] = {1, 1};
  TfLiteTensor tensors[3] = {
      type == kTfLiteInt8
          ? CreateQuantizedTensor(reinterpret_cast<const int8_t*>(input),
                                  IntArrayFromInts(input_dims), scale, 0)
          : CreateTensor(input, IntArrayFromInts(input_dims), false, type),
      CreateTensor(l.axis, IntArrayFromInts(axis_dims)),
      CreateTensor(output, IntArrayFromInts(output_dims)),
  };
  tensors[1].allocation_type = kTfLiteMmapRo;
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration =
      is_max ? Register_ARG_MAX() : Register_ARG_MIN();
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), nullptr);
  TF_LITE_ENSURE_STATUS(runner.InitAndPrepare());
  return runner.Invoke();
}

// Runs the layer on heliaCORE and on the reference and expects equal indices.
template <typename T>
void ExpectArgMatchesReference(const Layer& l, TfLiteType type,
                               const T* input, Leaf max_leaf, Leaf min_leaf) {
  Layer promoted;
  Promote(l, &promoted);
  const int output_count = Elements(l) / l.dims[Normalized(l.axis[0], l.rank)];
  for (int is_max = 0; is_max < 2; ++is_max) {
    int32_t core[kMaxElements];
    int32_t reference[kMaxElements];
    memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
    EXPECT_EQ(kTfLiteOk, RunArg(l, is_max, type, input, core));
#if HELIA_EXTREMA_LINK_WRAP
    EXPECT_EQ(1, g_leaf_calls[static_cast<int>(is_max ? max_leaf : min_leaf)]);
#else
    (void)max_leaf;
    (void)min_leaf;
#endif
    memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
    EXPECT_EQ(kTfLiteOk, RunArg(promoted, is_max, type, input, reference));
    EXPECT_EQ(0, CoreCalls());
    for (int i = 0; i < output_count; ++i) {
      EXPECT_EQ(reference[i], core[i]);
    }
  }
}

// Ties between equal values and between zero signs, infinities, subnormals,
// and NaNs (with payload and sign) at the first, last and inner positions.
void FillFloatSpecials(float* input, int count, bool with_nan) {
  uint32_t seed = 359;
  for (int i = 0; i < count; ++i) {
    seed = seed * 1103515245u + 12345u;
    input[i] = 0.5f * static_cast<float>(static_cast<int>((seed >> 16) % 17) - 8);
  }
  const float specials[] = {-0.0f, 0.0f, kInf, -kInf, 1e-40f, -1e-40f};
  for (int i = 0; i < 6; ++i) {
    input[(i * 37 + 5) % count] = specials[i];
  }
  if (with_nan) {
    input[0] = FromBits(0x7FC00123u);
    input[count - 1] = FromBits(0xFFC00042u);
    input[count / 2] = kNaN;
  }
}

// ---- REDUCE_MAX / REDUCE_MIN ------------------------------------------------

template <typename T>
TfLiteStatus RunReduce(const Layer& l, bool is_max, bool keep_dims,
                       TfLiteType type, const T* input, T* output,
                       float input_scale = 1.0f, float output_scale = 1.0f) {
  int input_dims[6] = {l.rank};
  int output_dims[6] = {0};
  for (int i = 0; i < l.rank; ++i) {
    input_dims[1 + i] = l.dims[i];
    if (keep_dims || !Reduced(l, i)) {
      output_dims[1 + output_dims[0]++] = Reduced(l, i) ? 1 : l.dims[i];
    }
  }
  int axis_dims[] = {1, l.axis_count};
  TfLiteTensor tensors[3] = {
      type == kTfLiteInt8
          ? CreateQuantizedTensor(reinterpret_cast<const int8_t*>(input),
                                  IntArrayFromInts(input_dims), input_scale, 0)
          : CreateTensor(input, IntArrayFromInts(input_dims), false, type),
      CreateTensor(l.axis, IntArrayFromInts(axis_dims)),
      type == kTfLiteInt8
          ? CreateQuantizedTensor(reinterpret_cast<int8_t*>(output),
                                  IntArrayFromInts(output_dims), output_scale,
                                  0)
          : CreateTensor(output, IntArrayFromInts(output_dims), false, type),
  };
  tensors[1].allocation_type = kTfLiteMmapRo;
  TfLiteReducerParams params = {};
  params.keep_dims = keep_dims;
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration =
      is_max ? Register_REDUCE_MAX() : Register_REDUCE_MIN();
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  TF_LITE_ENSURE_STATUS(runner.InitAndPrepare());
  return runner.Invoke();
}

int ReducedOutputCount(const Layer& l) {
  int count = 1;
  for (int i = 0; i < l.rank; ++i) {
    count *= Reduced(l, i) ? 1 : l.dims[i];
  }
  return count;
}

// Calls of the float32 _tflite entry for REDUCE_MAX (is_max) or REDUCE_MIN.
void ExpectTfliteF32Calls(bool is_max, int calls) {
#if HELIA_EXTREMA_LINK_WRAP
  EXPECT_EQ(calls, g_leaf_calls[static_cast<int>(
                       is_max ? Leaf::kReduceMaxTfliteF32
                              : Leaf::kReduceMinTfliteF32)]);
  EXPECT_EQ(calls, CoreCalls());
#else
  (void)is_max;
  (void)calls;
#endif
}

// Float32 REDUCE at the layer's rank, on heliaCORE, and padded to rank 5, on
// the reference: the two must agree bit for bit.
void ExpectReduceF32MatchesReference(const Layer& l, const float* input,
                                     bool keep_dims) {
  Layer promoted;
  Promote(l, &promoted);
  const int output_count = ReducedOutputCount(l);
  for (int is_max = 0; is_max < 2; ++is_max) {
    float at_rank[kMaxElements];
    float padded[kMaxElements];
    memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
    EXPECT_EQ(kTfLiteOk,
              RunReduce(l, is_max, keep_dims, kTfLiteFloat32, input, at_rank));
    ExpectTfliteF32Calls(is_max, 1);
    memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
    EXPECT_EQ(kTfLiteOk, RunReduce(promoted, is_max, keep_dims, kTfLiteFloat32,
                                   input, padded));
    ExpectTfliteF32Calls(is_max, 0);
    for (int i = 0; i < output_count; ++i) {
      EXPECT_EQ(Bits(padded[i]), Bits(at_rank[i]));
    }
  }
}

// One float32 REDUCE at the layer's rank, which must run the _tflite entry.
void ReduceF32(const Layer& l, bool is_max, const float* input,
               float* output) {
  memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
  EXPECT_EQ(kTfLiteOk, RunReduce(l, is_max, /*keep_dims=*/false,
                                 kTfLiteFloat32, input, output));
  ExpectTfliteF32Calls(is_max, 1);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::Bits;
using tflite::testing::kInf;
using tflite::testing::kLowest;
using tflite::testing::kMax;
using tflite::testing::kMaxElements;
using tflite::testing::Layer;

TEST(HeliaExtremaRouteTest, ArgFloat32MatchesReferenceOnEveryAxis) {
  float input[kMaxElements];
  tflite::testing::FillFloatSpecials(input, kMaxElements, /*with_nan=*/true);
  for (int axis = -1; axis < 4; ++axis) {
    tflite::testing::ExpectArgMatchesReference(
        Layer{4, {2, 3, 4, 5}, 1, {axis}}, kTfLiteFloat32, input,
        Leaf::kArgMaxF32, Leaf::kArgMinF32);
  }
}

// A line of NaNs returns index 0; a NaN never displaces a number.
TEST(HeliaExtremaRouteTest, ArgFloat32NaNLines) {
  const float n = tflite::testing::kNaN;
  const float input[12] = {n, n, n, n, n, 2.0f, n, 1.0f, -0.0f, 0.0f, n, -0.0f};
  tflite::testing::ExpectArgMatchesReference(Layer{2, {3, 4}, 1, {1}},
                                             kTfLiteFloat32, input,
                                             Leaf::kArgMaxF32, Leaf::kArgMinF32);
  int32_t index[3];
  EXPECT_EQ(kTfLiteOk, tflite::testing::RunArg(Layer{2, {3, 4}, 1, {1}}, true,
                                               kTfLiteFloat32, input, index));
  EXPECT_EQ(0, index[0]);
  EXPECT_EQ(1, index[1]);
  EXPECT_EQ(0, index[2]);
}

TEST(HeliaExtremaRouteTest, ArgInt8MatchesReferenceWithTies) {
  int8_t input[kMaxElements];
  for (int i = 0; i < kMaxElements; ++i) {
    input[i] = static_cast<int8_t>((i * 7) % 11 - 5);
  }
  input[3] = -128;
  input[50] = 127;
  for (int axis = 0; axis < 4; ++axis) {
    tflite::testing::ExpectArgMatchesReference(
        Layer{4, {2, 3, 4, 5}, 1, {axis}}, kTfLiteInt8, input, Leaf::kArgMaxS8,
        Leaf::kArgMinS8);
  }
}

// Ties, zero signs, subnormals, infinities and NaNs, over axis masks and both
// keep_dims settings.
TEST(HeliaExtremaRouteTest, ReduceFloat32MatchesReference) {
  float input[kMaxElements];
  tflite::testing::FillFloatSpecials(input, kMaxElements, /*with_nan=*/true);
  const Layer layers[] = {
      {4, {2, 3, 4, 5}, 2, {1, 2}},  {4, {2, 3, 4, 5}, 1, {3}},
      {4, {2, 3, 4, 5}, 1, {-4}},    {3, {6, 4, 5}, 2, {0, 2}},
      {2, {24, 5}, 1, {1}},          {4, {2, 3, 4, 5}, 4, {0, 1, 2, 3}},
  };
  for (const Layer& l : layers) {
    tflite::testing::ExpectReduceF32MatchesReference(l, input,
                                                     /*keep_dims=*/false);
    tflite::testing::ExpectReduceF32MatchesReference(l, input,
                                                     /*keep_dims=*/true);
  }
}

// A NaN never wins: a window gives its numeric extremum, or -FLT_MAX (max) /
// +FLT_MAX (min) when it holds only NaNs.
TEST(HeliaExtremaRouteTest, ReduceFloat32SkipsNaN) {
  const float n = tflite::testing::kNaN;
  const float input[6] = {1.0f, n, 3.0f, n, n, n};
  float output[2];
  for (int is_max = 0; is_max < 2; ++is_max) {
    tflite::testing::ReduceF32(Layer{2, {2, 3}, 1, {1}}, is_max, input, output);
    EXPECT_EQ(Bits(is_max ? 3.0f : 1.0f), Bits(output[0]));
    EXPECT_EQ(Bits(is_max ? kLowest : kMax), Bits(output[1]));
  }
}

// The reduction starts from -FLT_MAX (+FLT_MAX), so REDUCE_MAX over -Inf only
// gives -FLT_MAX and REDUCE_MIN over +Inf only gives +FLT_MAX.
TEST(HeliaExtremaRouteTest, ReduceFloat32InfinityWindowGivesStartValue) {
  const float input_max[3] = {-kInf, -kInf, -kInf};
  const float input_min[3] = {kInf, kInf, kInf};
  float output[1];
  tflite::testing::ReduceF32(Layer{1, {3}, 1, {0}}, true, input_max, output);
  EXPECT_EQ(Bits(kLowest), Bits(output[0]));
  tflite::testing::ReduceF32(Layer{1, {3}, 1, {0}}, false, input_min, output);
  EXPECT_EQ(Bits(kMax), Bits(output[0]));
}

TEST(HeliaExtremaRouteTest, ReduceFloat32EmptyDomainGivesStartValue) {
  const float input[1] = {0.0f};
  float output[6];
  for (int is_max = 0; is_max < 2; ++is_max) {
    tflite::testing::ReduceF32(Layer{3, {2, 0, 3}, 1, {1}}, is_max, input,
                               output);
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(Bits(is_max ? kLowest : kMax), Bits(output[i]));
    }
  }
}

// With no axes each element is folded into the start value on its own, so a
// NaN and the start side's infinity become -FLT_MAX (+FLT_MAX).
TEST(HeliaExtremaRouteTest, ReduceFloat32EmptyAxisListFoldsEachElement) {
  const float input[4] = {tflite::testing::FromBits(0x7FC00123u), -kInf, kInf,
                          2.0f};
  float output[4];
  for (int is_max = 0; is_max < 2; ++is_max) {
    tflite::testing::ReduceF32(Layer{1, {4}, 0, {}}, is_max, input, output);
    EXPECT_EQ(Bits(is_max ? kLowest : kMax), Bits(output[0]));
    EXPECT_EQ(Bits(is_max ? kLowest : -kInf), Bits(output[1]));
    EXPECT_EQ(Bits(is_max ? kInf : kMax), Bits(output[2]));
    EXPECT_EQ(Bits(2.0f), Bits(output[3]));
  }
}

// The int8 kernels copy values without requantizing, so REDUCE_MAX/MIN
// requires the output to share the input's scale on the heliaCORE path too.
TEST(HeliaExtremaRouteTest, ReduceInt8RejectsRescaledOutput) {
  const int8_t input[6] = {1, 2, 3, 4, 5, 6};
  int8_t output[2];
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunReduce(Layer{2, {2, 3}, 1, {1}}, true, false,
                                       kTfLiteInt8, input, output, 0.5f, 0.25f));
  EXPECT_EQ(kTfLiteOk,
            tflite::testing::RunReduce(Layer{2, {2, 3}, 1, {1}}, true, false,
                                       kTfLiteInt8, input, output, 0.5f, 0.5f));
  EXPECT_EQ(3, output[0]);
  EXPECT_EQ(6, output[1]);
}

// Float16 REDUCE_MAX/MIN have no reference kernel: fp16 builds run the
// _tflite_f16 entries, with float32's TFLite semantics at the float16 start
// values -/+65504, and other builds reject float16 when the model is prepared.
TEST(HeliaExtremaRouteTest, ReduceFloat16) {
  // 1.0, -2.0, 0.5 | NaN, 4.0, -0.0 | -Inf, -0.0, 0.0 | 6.0, 6.0, -6.0 |
  // NaN, NaN, NaN | -Inf, -Inf, -Inf
  const uint16_t input[18] = {0x3C00, 0xC000, 0x3800, 0x7E55, 0x4400, 0x8000,
                              0xFC00, 0x8000, 0x0000, 0x4600, 0x4600, 0xC600,
                              0x7E01, 0xFE02, 0x7C03, 0xFC00, 0xFC00, 0xFC00};
  uint16_t output[18] = {};
  const Layer l{2, {6, 3}, 1, {1}};
#if ARM_NN_ENABLE_F16
  for (int is_max = 0; is_max < 2; ++is_max) {
    memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
    EXPECT_EQ(kTfLiteOk, tflite::testing::RunReduce(
                             l, is_max, false, kTfLiteFloat16, input, output));
#if HELIA_EXTREMA_LINK_WRAP
    EXPECT_EQ(1, g_leaf_calls[static_cast<int>(
                     is_max ? Leaf::kReduceMaxTfliteF16
                            : Leaf::kReduceMinTfliteF16)]);
#endif
    EXPECT_EQ(is_max ? 0x3C00 : 0xC000, output[0]);
    // A NaN never wins.
    EXPECT_EQ(is_max ? 0x4400 : 0x8000, output[1]);
    // Equal values keep the first in row-major order, zero signs included.
    EXPECT_EQ(is_max ? 0x8000 : 0xFC00, output[2]);
    EXPECT_EQ(is_max ? 0x4600 : 0xC600, output[3]);
    // A window of NaNs, or of the start side's infinity, gives -/+65504.
    EXPECT_EQ(is_max ? 0xFBFF : 0x7BFF, output[4]);
    EXPECT_EQ(is_max ? 0xFBFF : 0xFC00, output[5]);
  }
  // An empty axis list folds each element into the start value.
  const Layer no_axes{1, {6}, 0, {}};
  const uint16_t row[6] = {0x7E55, 0xFC00, 0x7C00, 0x4000, 0x8000, 0xFE01};
  for (int is_max = 0; is_max < 2; ++is_max) {
    EXPECT_EQ(kTfLiteOk, tflite::testing::RunReduce(
                             no_axes, is_max, false, kTfLiteFloat16, row,
                             output));
    EXPECT_EQ(is_max ? 0xFBFF : 0x7BFF, output[0]);
    EXPECT_EQ(is_max ? 0xFBFF : 0xFC00, output[1]);
    EXPECT_EQ(is_max ? 0x7C00 : 0x7BFF, output[2]);
    EXPECT_EQ(0x4000, output[3]);
    EXPECT_EQ(0x8000, output[4]);
    EXPECT_EQ(is_max ? 0xFBFF : 0x7BFF, output[5]);
  }
#else
  for (int is_max = 0; is_max < 2; ++is_max) {
    EXPECT_EQ(kTfLiteError, tflite::testing::RunReduce(
                                l, is_max, false, kTfLiteFloat16, input, output));
  }
#endif
}

// Prepare never compares the output tensor with the reduced shape. A model can
// declare a [2, 3] -> axis `axis` reduction's output as `output_dims`; Prepare
// must accept it and Invoke must fail without reaching heliaCORE or writing
// past the tensor. See also reduce_output_size_test.cc.
template <typename T>
void ExpectReduceMaxPreparesThenFailsInvoke(TfLiteType type, int32_t axis_value,
                                           int* output_dims, int capacity) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;
  const T input[6] = {1, 2, 3, 4, 5, 6};
  T output[8];
  const int32_t axis[1] = {axis_value};
  int input_dims[] = {2, 2, 3};
  int axis_dims[] = {1, 1};
  TfLiteTensor tensors[3];
  if constexpr (std::is_same_v<T, uint16_t>) {
    tensors[0] =
        CreateTensor(input, IntArrayFromInts(input_dims), false, type);
    tensors[2] =
        CreateTensor(output, IntArrayFromInts(output_dims), false, type);
  } else {
    tensors[0] =
        CreateQuantizedTensor(input, IntArrayFromInts(input_dims), 0.5f, 0);
    tensors[2] =
        CreateQuantizedTensor(output, IntArrayFromInts(output_dims), 0.5f, 0);
  }
  tensors[1] = CreateTensor(axis, IntArrayFromInts(axis_dims));
  tensors[1].allocation_type = kTfLiteMmapRo;
  TfLiteReducerParams params = {};
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = tflite::Register_REDUCE_MAX();
  tflite::micro::KernelRunner runner(registration, tensors, 3,
                                     IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), &params);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  memset(output, 0x5A, sizeof(output));
  memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
  EXPECT_EQ(kTfLiteError, runner.Invoke());
  EXPECT_EQ(0, CoreCalls());
  for (size_t i = 0; i < sizeof(T) * capacity; ++i) {
    EXPECT_EQ(0x5A, reinterpret_cast<const uint8_t*>(output)[i]);
  }
}

TEST(HeliaExtremaRouteTest, ReduceMaxShortOutputPreparesThenFailsInvoke) {
  int output_dims[] = {1, 1};
  ExpectReduceMaxPreparesThenFailsInvoke<int8_t>(kTfLiteInt8, 1, output_dims,
                                                 8);
  ExpectReduceMaxPreparesThenFailsInvoke<int16_t>(kTfLiteInt16, 1, output_dims,
                                                  8);
#if ARM_NN_ENABLE_F16
  ExpectReduceMaxPreparesThenFailsInvoke<uint16_t>(kTfLiteFloat16, 1,
                                                   output_dims, 8);
#endif
}

// An axis outside the input rank fails, as on the reference and float16
// routes, rather than being skipped and copying the input.
TEST(HeliaExtremaRouteTest, ReduceMaxIntAxisOutOfRangeFailsInvoke) {
  int output_dims[] = {2, 2, 3};
  ExpectReduceMaxPreparesThenFailsInvoke<int8_t>(kTfLiteInt8, 7, output_dims,
                                                 8);
  ExpectReduceMaxPreparesThenFailsInvoke<int16_t>(kTfLiteInt16, 7, output_dims,
                                                  8);
}

TF_LITE_MICRO_TESTS_MAIN
