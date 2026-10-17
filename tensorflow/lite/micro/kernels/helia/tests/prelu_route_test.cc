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

// PRELU through the helia kernel. int8 and float32 compare bit for bit with
// the TFLM reference (reference_ops::BroadcastPrelu4DSlow and
// BroadcastPrelu4DSlowFloat) for alpha of the input's shape, a scalar, per
// channel and a general broadcast, all on heliaCORE, and for an alpha that
// widens the output, which stays on the reference. int16 stays on the
// reference: TFLM's int16 PRELU takes an int8 alpha, arm_prelu_s16 an int16
// one. float16 (M55) has no reference kernel and is checked against the
// formula. GCC and ATfE links wrap the heliaCORE entries and assert which ran.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/prelu.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/prelu.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

#if ARM_NN_ENABLE_F32 || ARM_NN_ENABLE_F16
#include "Include/arm_nnfunctions_flt.h"
#endif

namespace {

enum class Leaf { kS8, kS16, kF32, kF16, kCount };

int g_leaf_calls[static_cast<int>(Leaf::kCount)] = {};

#if HELIA_PRELU_LINK_WRAP
void CountCall(Leaf leaf) { ++g_leaf_calls[static_cast<int>(leaf)]; }
#endif

}  // namespace

#if HELIA_PRELU_LINK_WRAP
extern "C" {

#define HELIA_PRELU_Q_ARGS(T)                                                \
  const cmsis_nn_dims *di, const T *in, const cmsis_nn_dims *da,             \
      const T *alpha, const int32_t io, const int32_t ao, const int32_t oo,   \
      const int32_t m1, const int32_t s1, const int32_t m2, const int32_t s2, \
      const cmsis_nn_dims *dout, T *out
#define HELIA_PRELU_Q_FORWARD di, in, da, alpha, io, ao, oo, m1, s1, m2, s2, dout, out

arm_cmsis_nn_status __real_arm_prelu_s8(HELIA_PRELU_Q_ARGS(int8_t));
arm_cmsis_nn_status __wrap_arm_prelu_s8(HELIA_PRELU_Q_ARGS(int8_t)) {
  CountCall(Leaf::kS8);
  return __real_arm_prelu_s8(HELIA_PRELU_Q_FORWARD);
}
arm_cmsis_nn_status __real_arm_prelu_s16(HELIA_PRELU_Q_ARGS(int16_t));
arm_cmsis_nn_status __wrap_arm_prelu_s16(HELIA_PRELU_Q_ARGS(int16_t)) {
  CountCall(Leaf::kS16);
  return __real_arm_prelu_s16(HELIA_PRELU_Q_FORWARD);
}

#undef HELIA_PRELU_Q_FORWARD
#undef HELIA_PRELU_Q_ARGS

#if ARM_NN_ENABLE_F32
arm_cmsis_nn_status __real_arm_prelu_f32(const cmsis_nn_dims*,
                                         const float32_t*,
                                         const cmsis_nn_dims*,
                                         const float32_t*,
                                         const cmsis_nn_dims*, float32_t*);
arm_cmsis_nn_status __wrap_arm_prelu_f32(const cmsis_nn_dims* di,
                                         const float32_t* in,
                                         const cmsis_nn_dims* da,
                                         const float32_t* alpha,
                                         const cmsis_nn_dims* dout,
                                         float32_t* out) {
  CountCall(Leaf::kF32);
  return __real_arm_prelu_f32(di, in, da, alpha, dout, out);
}
#endif

#if ARM_NN_ENABLE_F16
arm_cmsis_nn_status __real_arm_prelu_f16(const cmsis_nn_dims*,
                                         const float16_t*,
                                         const cmsis_nn_dims*,
                                         const float16_t*,
                                         const cmsis_nn_dims*, float16_t*);
arm_cmsis_nn_status __wrap_arm_prelu_f16(const cmsis_nn_dims* di,
                                         const float16_t* in,
                                         const cmsis_nn_dims* da,
                                         const float16_t* alpha,
                                         const cmsis_nn_dims* dout,
                                         float16_t* out) {
  CountCall(Leaf::kF16);
  return __real_arm_prelu_f16(di, in, da, alpha, dout, out);
}
#endif

}  // extern "C"
#endif  // HELIA_PRELU_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

#if ARM_NN_ENABLE_F32
constexpr bool kHeliaF32 = true;
#else
constexpr bool kHeliaF32 = false;
#endif

// Expected calls of `leaf` for one invoke.
void ExpectCalls(Leaf leaf, int calls) {
#if HELIA_PRELU_LINK_WRAP
  EXPECT_EQ(calls, g_leaf_calls[static_cast<int>(leaf)]);
#else
  (void)leaf;
  (void)calls;
#endif
}

constexpr int kMaxRank = 4;
constexpr int kMaxElements = 3 * 4 * 5 * 6;

struct Shape {
  int rank;
  int dims[kMaxRank];
  int Count() const {
    int count = 1;
    for (int i = 0; i < rank; ++i) count *= dims[i];
    return count;
  }
  RuntimeShape Runtime() const {
    RuntimeShape shape(rank);
    for (int i = 0; i < rank; ++i) shape.SetDim(i, dims[i]);
    return shape;
  }
};

uint32_t g_seed = 6028;
int NextInt(int lo, int hi) {
  g_seed = g_seed * 1103515245u + 12345u;
  return lo + static_cast<int>((g_seed >> 16) % (hi - lo + 1));
}

Shape BroadcastShape(const Shape& a, const Shape& b) {
  Shape out = {a.rank > b.rank ? a.rank : b.rank, {}};
  for (int i = 0; i < out.rank; ++i) {
    const int ai = i - (out.rank - a.rank);
    const int bi = i - (out.rank - b.rank);
    const int da = ai >= 0 ? a.dims[ai] : 1;
    const int db = bi >= 0 ? b.dims[bi] : 1;
    out.dims[i] = da > db ? da : db;
  }
  return out;
}

// float16_t has no TfLiteType mapping in the test helpers.
template <typename T>
constexpr TfLiteType FloatTensorType() {
  return std::is_same<T, float>::value ? kTfLiteFloat32 : kTfLiteFloat16;
}

template <typename T, typename U>
TfLiteStatus Run(const Shape& si, const T* in, const Shape& sa, const U* alpha,
                 const Shape& so, T* out, float in_scale, int in_zp,
                 float alpha_scale, int alpha_zp, float out_scale,
                 int out_zp) {
  int di[kMaxRank + 1] = {si.rank};
  int da[kMaxRank + 1] = {sa.rank};
  int dout[kMaxRank + 1] = {so.rank};
  for (int i = 0; i < si.rank; ++i) di[i + 1] = si.dims[i];
  for (int i = 0; i < sa.rank; ++i) da[i + 1] = sa.dims[i];
  for (int i = 0; i < so.rank; ++i) dout[i + 1] = so.dims[i];
  TfLiteTensor tensors[3];
  if constexpr (std::numeric_limits<T>::is_integer) {
    tensors[0] =
        CreateQuantizedTensor(in, IntArrayFromInts(di), in_scale, in_zp);
    tensors[1] = CreateQuantizedTensor(alpha, IntArrayFromInts(da),
                                       alpha_scale, alpha_zp);
    tensors[2] =
        CreateQuantizedTensor(out, IntArrayFromInts(dout), out_scale, out_zp);
  } else {
    tensors[0] =
        CreateTensor(in, IntArrayFromInts(di), false, FloatTensorType<T>());
    tensors[1] =
        CreateTensor(alpha, IntArrayFromInts(da), false, FloatTensorType<U>());
    tensors[2] =
        CreateTensor(out, IntArrayFromInts(dout), false, FloatTensorType<T>());
  }
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = Register_PRELU();
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), nullptr);
  const TfLiteStatus prepare = runner.InitAndPrepare();
  if (prepare != kTfLiteOk) {
    return prepare;
  }
  memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
  return runner.Invoke();
}

// T input and output, U alpha: int8/int8 or int16/int8, as TFLM types them.
template <typename T, typename U>
void ExpectQuantizedMatches(const Shape& si, const Shape& sa, Leaf leaf,
                            int calls) {
  static T in[kMaxElements];
  static U alpha[kMaxElements];
  static T out[kMaxElements];
  static T expected[kMaxElements];
  constexpr bool kIs16 = sizeof(T) == 2;
  const Shape so = BroadcastShape(si, sa);
  const int lo = std::numeric_limits<T>::min();
  const int hi = std::numeric_limits<T>::max();
  for (int i = 0; i < si.Count(); ++i) in[i] = static_cast<T>(NextInt(lo, hi));
  for (int i = 0; i < sa.Count(); ++i) {
    alpha[i] = static_cast<U>(NextInt(-128, 127));
  }
  // The extremes reach saturation on both branches.
  in[0] = static_cast<T>(lo);
  in[1] = static_cast<T>(hi);
  const float in_scale = kIs16 ? 1.0f / 1024 : 0.04f;
  const int in_zp = kIs16 ? 0 : -3;
  const float alpha_scale = 0.01f;
  const int alpha_zp = 5;
  const float out_scale = kIs16 ? 1.0f / 1500 : 0.03f;
  const int out_zp = kIs16 ? 0 : 4;
  EXPECT_EQ(kTfLiteOk, Run(si, in, sa, alpha, so, out, in_scale, in_zp,
                           alpha_scale, alpha_zp, out_scale, out_zp));
  ExpectCalls(leaf, calls);

  PreluParams params = {};
  // As CalculatePreluParams() derives them.
  QuantizeMultiplier(
      static_cast<double>(in_scale) / static_cast<double>(out_scale),
      &params.output_multiplier_1, &params.output_shift_1);
  QuantizeMultiplier(static_cast<double>(in_scale) *
                         static_cast<double>(alpha_scale) /
                         static_cast<double>(out_scale),
                     &params.output_multiplier_2, &params.output_shift_2);
  params.input_offset = -in_zp;
  params.alpha_offset = -alpha_zp;
  params.output_offset = out_zp;
  micro::reference_ops::BroadcastPrelu4DSlow(params, si.Runtime(), in,
                                             sa.Runtime(), alpha, so.Runtime(),
                                             expected);
  int mismatches = 0;
  for (int i = 0; i < so.Count(); ++i) {
    mismatches += out[i] != expected[i];
  }
  EXPECT_EQ(0, mismatches);
}

uint32_t Bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool SameFloat(float a, float b) {
  return (std::isnan(a) && std::isnan(b)) || Bits(a) == Bits(b);
}

void ExpectFloat32Matches(const Shape& si, const Shape& sa, bool helia) {
  static float in[kMaxElements];
  static float alpha[kMaxElements];
  static float out[kMaxElements];
  static float expected[kMaxElements];
  const Shape so = BroadcastShape(si, sa);
  for (int i = 0; i < si.Count(); ++i) in[i] = NextInt(-4000, 4000) / 64.0f;
  for (int i = 0; i < sa.Count(); ++i) alpha[i] = NextInt(-64, 64) / 32.0f;
  EXPECT_EQ(kTfLiteOk, Run(si, in, sa, alpha, so, out, 0, 0, 0, 0, 0, 0));
  ExpectCalls(Leaf::kF32, helia && kHeliaF32 ? 1 : 0);
  micro::BroadcastPrelu4DSlowFloat(si.Runtime(), in, sa.Runtime(), alpha,
                                   so.Runtime(), expected);
  int mismatches = 0;
  for (int i = 0; i < so.Count(); ++i) {
    mismatches += !SameFloat(out[i], expected[i]);
  }
  EXPECT_EQ(0, mismatches);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ExpectFloat32Matches;
using tflite::testing::ExpectQuantizedMatches;
using tflite::testing::Shape;

TEST(HeliaPreluRouteTest, Int8AlphaSameShape) {
  ExpectQuantizedMatches<int8_t, int8_t>(Shape{4, {1, 4, 5, 6}},
                                         Shape{4, {1, 4, 5, 6}}, Leaf::kS8, 1);
}

TEST(HeliaPreluRouteTest, Int8ScalarAlpha) {
  ExpectQuantizedMatches<int8_t, int8_t>(Shape{4, {2, 3, 5, 6}}, Shape{1, {1}},
                                         Leaf::kS8, 1);
}

TEST(HeliaPreluRouteTest, Int8PerChannelAlpha) {
  ExpectQuantizedMatches<int8_t, int8_t>(Shape{4, {2, 3, 5, 6}}, Shape{1, {6}},
                                         Leaf::kS8, 1);
}

TEST(HeliaPreluRouteTest, Int8GeneralBroadcastAlpha) {
  ExpectQuantizedMatches<int8_t, int8_t>(
      Shape{4, {2, 4, 5, 6}}, Shape{3, {4, 1, 6}}, Leaf::kS8, 1);
}

// An alpha extent above the input's widens the output, which heliaCORE does
// not take.
TEST(HeliaPreluRouteTest, Int8AlphaWideningOutputIsReference) {
  ExpectQuantizedMatches<int8_t, int8_t>(
      Shape{4, {1, 1, 5, 6}}, Shape{4, {3, 1, 1, 6}}, Leaf::kS8, 0);
}

TEST(HeliaPreluRouteTest, Int16IsReference) {
  ExpectQuantizedMatches<int16_t, int8_t>(Shape{4, {2, 3, 5, 6}},
                                          Shape{1, {6}}, Leaf::kS16, 0);
}

TEST(HeliaPreluRouteTest, Float32PerChannelAlpha) {
  ExpectFloat32Matches(Shape{4, {2, 3, 5, 6}}, Shape{1, {6}}, true);
}

TEST(HeliaPreluRouteTest, Float32GeneralBroadcastAlpha) {
  ExpectFloat32Matches(Shape{4, {2, 4, 5, 6}}, Shape{3, {4, 1, 6}}, true);
}

TEST(HeliaPreluRouteTest, Float32AlphaWideningOutputIsReference) {
  ExpectFloat32Matches(Shape{4, {1, 1, 5, 6}}, Shape{4, {3, 1, 1, 6}}, false);
}

// The reference's x >= 0 ? x : x * alpha bit for bit: NaN input gives NaN, a
// NaN alpha reaches only negative inputs, -0 and +0 pass through with their
// sign, -Inf * 0 is NaN, and positive subnormals pass through.
TEST(HeliaPreluRouteTest, Float32EdgeValues) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float denorm = std::numeric_limits<float>::denorm_min();
  float in[] = {nan, -2.0f, 2.0f, -0.0f, 0.0f, -inf, -inf, inf, 8 * denorm,
                -3.0f};
  float alpha[] = {0.5f, nan, nan, 0.5f, 0.5f, 0.25f, 0.0f, 0.5f, 0.5f, 0.5f};
  float out[10] = {};
  float expected[10] = {};
  const Shape shape{1, {10}};
  EXPECT_EQ(kTfLiteOk, tflite::testing::Run(shape, in, shape, alpha, shape, out,
                                            0, 0, 0, 0, 0, 0));
  tflite::testing::ExpectCalls(Leaf::kF32, tflite::testing::kHeliaF32 ? 1 : 0);
  tflite::micro::BroadcastPrelu4DSlowFloat(shape.Runtime(), in,
                                           shape.Runtime(), alpha,
                                           shape.Runtime(), expected);
  for (int i = 0; i < 10; ++i) {
    EXPECT_TRUE(tflite::testing::SameFloat(out[i], expected[i]));
  }
}

// CHARACTERIZATION: a negative subnormal input. The reference and the scalar
// heliaCORE leg compute x * alpha. The MVE float32 leg compares with
// vcmpgeq_n_f32, which flushes the subnormal to -0 and finds it >= 0, so it
// returns x unchanged.
// TODO(AmbiqAI/ns-cmsis-nn#873): expect x * alpha on every leg once fixed.
TEST(HeliaPreluRouteTest, Float32NegativeSubnormalInput) {
  const float x = -8 * std::numeric_limits<float>::denorm_min();
  float in[] = {x, x, x, x};
  float alpha[] = {0.25f, 0.25f, 0.25f, 0.25f};
  float out[4] = {};
  float expected[4] = {};
  const Shape shape{1, {4}};
  EXPECT_EQ(kTfLiteOk, tflite::testing::Run(shape, in, shape, alpha, shape, out,
                                            0, 0, 0, 0, 0, 0));
  tflite::micro::BroadcastPrelu4DSlowFloat(shape.Runtime(), in,
                                           shape.Runtime(), alpha,
                                           shape.Runtime(), expected);
#if ARM_NN_ENABLE_F32 && defined(ARM_MATH_MVEF) && \
    !defined(ARM_MATH_AUTOVECTORIZE)
  const float pinned = x;
#else
  const float pinned = expected[0];
#endif
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(tflite::testing::Bits(pinned), tflite::testing::Bits(out[i]));
  }
}

#if ARM_NN_ENABLE_F16

TEST(HeliaPreluRouteTest, Float16PerChannelAlpha) {
  float16_t in[3 * 5 * 6];
  float16_t alpha[6];
  float16_t out[3 * 5 * 6] = {};
  for (int i = 0; i < 3 * 5 * 6; ++i) {
    in[i] = static_cast<float16_t>(tflite::testing::NextInt(-512, 512) / 64.0f);
  }
  for (int i = 0; i < 6; ++i) {
    alpha[i] = static_cast<float16_t>(tflite::testing::NextInt(-32, 32) / 16.0f);
  }
  EXPECT_EQ(kTfLiteOk,
            tflite::testing::Run(Shape{3, {3, 5, 6}}, in, Shape{1, {6}}, alpha,
                                 Shape{3, {3, 5, 6}}, out, 0, 0, 0, 0, 0, 0));
  tflite::testing::ExpectCalls(Leaf::kF16, 1);
  int mismatches = 0;
  for (int i = 0; i < 3 * 5 * 6; ++i) {
    // Selects stay in float: a scalar float16 conditional select ICEs GCC
    // (PR target/118460).
    const float x = static_cast<float>(in[i]);
    const float negative = static_cast<float>(
        static_cast<float16_t>(x * static_cast<float>(alpha[i % 6])));
    const float expected = x >= 0.0f ? x : negative;
    mismatches += static_cast<float>(out[i]) != expected;
  }
  EXPECT_EQ(0, mismatches);
}

TEST(HeliaPreluRouteTest, Float16AlphaWideningOutputRejectedAtPrepare) {
  float16_t in[30] = {};
  float16_t alpha[18] = {};
  float16_t out[90] = {};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::Run(Shape{4, {1, 1, 5, 6}}, in,
                                 Shape{4, {3, 1, 1, 6}}, alpha,
                                 Shape{4, {3, 1, 5, 6}}, out, 0, 0, 0, 0, 0,
                                 0));
}

// NaN input gives NaN, signed zeros pass through, -Inf * 0 is NaN, and unlike
// the MVE float32 leg a negative subnormal gives x * alpha.
TEST(HeliaPreluRouteTest, Float16EdgeValues) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float sub = -8 * 5.9604644775390625e-08f;  // -8 * 2^-24
  const float xs[] = {nan, -0.0f, 0.0f, -inf, sub, -3.0f};
  const float as[] = {0.5f, 0.5f, 0.5f, 0.0f, 0.25f, 0.5f};
  float16_t in[6];
  float16_t alpha[6];
  float16_t out[6] = {};
  for (int i = 0; i < 6; ++i) {
    in[i] = static_cast<float16_t>(xs[i]);
    alpha[i] = static_cast<float16_t>(as[i]);
  }
  const Shape shape{1, {6}};
  EXPECT_EQ(kTfLiteOk, tflite::testing::Run(shape, in, shape, alpha, shape, out,
                                            0, 0, 0, 0, 0, 0));
  EXPECT_TRUE(std::isnan(static_cast<float>(out[0])));
  EXPECT_TRUE(static_cast<float>(out[1]) == 0.0f &&
              std::signbit(static_cast<float>(out[1])));
  EXPECT_TRUE(static_cast<float>(out[2]) == 0.0f &&
              !std::signbit(static_cast<float>(out[2])));
  EXPECT_TRUE(std::isnan(static_cast<float>(out[3])));
  EXPECT_EQ(sub * 0.25f, static_cast<float>(out[4]));
  EXPECT_EQ(-1.5f, static_cast<float>(out[5]));
}

#elif defined(__ARM_FEATURE_MVE) && ((__ARM_FEATURE_MVE) & 2)

TEST(HeliaPreluRouteTest, Float16CoverageMustNotSilentlyDisappear) {
  FAIL(
      "ARM_NN_ENABLE_F16 is not defined on a build with MVE floating point. "
      "The float16 PRELU coverage silently compiled out.");
}

#endif  // ARM_NN_ENABLE_F16

TF_LITE_MICRO_TESTS_MAIN
