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

// float DIV through the helia kernel. float32 compares bit for bit with the
// reference formula clamp(a / b) on equal shapes and 4-D broadcasts, where
// heliaCORE runs, and on rank-5 broadcasts, which stay on the reference; the
// edge cases cover NaN, infinities, signed zeros, division by zero, overflow
// and gradual underflow through the activation clamp. float16 (M55) has no
// reference kernel, so it is checked against the float32 quotient rounded to
// float16, the heliaCORE contract. GCC and ATfE links wrap the heliaCORE
// entries and assert which ran.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/helia/helia_float_common.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

#if ARM_NN_ENABLE_F32 || ARM_NN_ENABLE_F16
#include "Include/arm_nnfunctions_flt.h"
#endif

namespace {

enum class Leaf { kFlatF32, kWalkF32, kFlatF16, kWalkF16, kCount };

int g_leaf_calls[static_cast<int>(Leaf::kCount)] = {};

#if HELIA_DIV_LINK_WRAP
void CountCall(Leaf leaf) { ++g_leaf_calls[static_cast<int>(leaf)]; }
#endif

}  // namespace

#if HELIA_DIV_LINK_WRAP
extern "C" {

#if ARM_NN_ENABLE_F32
arm_cmsis_nn_status __real_arm_elementwise_div_f32(const float32_t*,
                                                   const float32_t*, float32_t*,
                                                   float32_t, float32_t,
                                                   int32_t);
arm_cmsis_nn_status __wrap_arm_elementwise_div_f32(const float32_t* a,
                                                   const float32_t* b,
                                                   float32_t* out, float32_t lo,
                                                   float32_t hi, int32_t n) {
  CountCall(Leaf::kFlatF32);
  return __real_arm_elementwise_div_f32(a, b, out, lo, hi, n);
}
arm_cmsis_nn_status __real_arm_elementwise_div_broadcast_f32(
    const float32_t*, const cmsis_nn_dims*, const float32_t*,
    const cmsis_nn_dims*, float32_t*, const cmsis_nn_dims*, float32_t,
    float32_t);
arm_cmsis_nn_status __wrap_arm_elementwise_div_broadcast_f32(
    const float32_t* a, const cmsis_nn_dims* da, const float32_t* b,
    const cmsis_nn_dims* db, float32_t* out, const cmsis_nn_dims* dout,
    float32_t lo, float32_t hi) {
  CountCall(Leaf::kWalkF32);
  return __real_arm_elementwise_div_broadcast_f32(a, da, b, db, out, dout, lo,
                                                  hi);
}
#endif

#if ARM_NN_ENABLE_F16
arm_cmsis_nn_status __real_arm_elementwise_div_f16(const float16_t*,
                                                   const float16_t*, float16_t*,
                                                   float16_t, float16_t,
                                                   int32_t);
arm_cmsis_nn_status __wrap_arm_elementwise_div_f16(const float16_t* a,
                                                   const float16_t* b,
                                                   float16_t* out, float16_t lo,
                                                   float16_t hi, int32_t n) {
  CountCall(Leaf::kFlatF16);
  return __real_arm_elementwise_div_f16(a, b, out, lo, hi, n);
}
arm_cmsis_nn_status __real_arm_elementwise_div_broadcast_f16(
    const float16_t*, const cmsis_nn_dims*, const float16_t*,
    const cmsis_nn_dims*, float16_t*, const cmsis_nn_dims*, float16_t,
    float16_t);
arm_cmsis_nn_status __wrap_arm_elementwise_div_broadcast_f16(
    const float16_t* a, const cmsis_nn_dims* da, const float16_t* b,
    const cmsis_nn_dims* db, float16_t* out, const cmsis_nn_dims* dout,
    float16_t lo, float16_t hi) {
  CountCall(Leaf::kWalkF16);
  return __real_arm_elementwise_div_broadcast_f16(a, da, b, db, out, dout, lo,
                                                  hi);
}
#endif

}  // extern "C"
#endif  // HELIA_DIV_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

enum class Route { kFlat, kWalk, kReference };

#if ARM_NN_ENABLE_F32
constexpr Route kFlatF32Route = Route::kFlat;
constexpr Route kWalkF32Route = Route::kWalk;
#else
constexpr Route kFlatF32Route = Route::kReference;
constexpr Route kWalkF32Route = Route::kReference;
#endif

void ExpectRoute(Leaf flat, Leaf walk, Route route) {
#if HELIA_DIV_LINK_WRAP
  const int flat_calls = g_leaf_calls[static_cast<int>(flat)];
  const int walk_calls = g_leaf_calls[static_cast<int>(walk)];
  switch (route) {
    case Route::kFlat:
      EXPECT_GE(flat_calls, 1);
      EXPECT_EQ(0, walk_calls);
      break;
    case Route::kWalk:
      EXPECT_EQ(1, walk_calls);
      break;
    case Route::kReference:
      EXPECT_EQ(0, flat_calls);
      EXPECT_EQ(0, walk_calls);
      break;
  }
#else
  (void)flat;
  (void)walk;
  (void)route;
#endif
}

constexpr int kMaxRank = 5;
constexpr int kMaxElements = 2 * 3 * 4 * 5 * 6;

struct Shape {
  int rank;
  int dims[kMaxRank];
  int Count() const {
    int count = 1;
    for (int i = 0; i < rank; ++i) count *= dims[i];
    return count;
  }
};

uint32_t g_seed = 1265;
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

int BroadcastIndex(const Shape& in, const Shape& out, int flat) {
  int index = 0;
  int stride = 1;
  for (int i = out.rank - 1; i >= 0; --i) {
    const int coord = flat % out.dims[i];
    flat /= out.dims[i];
    const int ii = i - (out.rank - in.rank);
    if (ii >= 0) {
      index += (in.dims[ii] == 1 ? 0 : coord) * stride;
      stride *= in.dims[ii];
    }
  }
  return index;
}

// float16_t has no TfLiteType mapping in the test helpers.
template <typename T>
constexpr TfLiteType FloatTensorType() {
  return std::is_same<T, float>::value ? kTfLiteFloat32 : kTfLiteFloat16;
}

template <typename T>
TfLiteStatus Run(const Shape& s1, const T* in1, const Shape& s2, const T* in2,
                 const Shape& so, T* out, TfLiteFusedActivation activation) {
  int d1[kMaxRank + 1] = {s1.rank};
  int d2[kMaxRank + 1] = {s2.rank};
  int dout[kMaxRank + 1] = {so.rank};
  for (int i = 0; i < s1.rank; ++i) d1[i + 1] = s1.dims[i];
  for (int i = 0; i < s2.rank; ++i) d2[i + 1] = s2.dims[i];
  for (int i = 0; i < so.rank; ++i) dout[i + 1] = so.dims[i];
  TfLiteTensor tensors[3] = {
      CreateTensor(in1, IntArrayFromInts(d1), false, FloatTensorType<T>()),
      CreateTensor(in2, IntArrayFromInts(d2), false, FloatTensorType<T>()),
      CreateTensor(out, IntArrayFromInts(dout), false, FloatTensorType<T>()),
  };
  TfLiteDivParams params = {activation};
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = Register_DIV();
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  const TfLiteStatus prepare = runner.InitAndPrepare();
  if (prepare != kTfLiteOk) {
    return prepare;
  }
  memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
  return runner.Invoke();
}

uint32_t Bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float ReferenceDiv(float a, float b, TfLiteFusedActivation activation) {
  float lo;
  float hi;
  micro::CalculateActivationRange(activation, &lo, &hi);
  return micro::ActivationFunctionWithMinMax(a / b, lo, hi);
}

// Same bits, or both NaN (payload not asserted).
bool SameFloat(float a, float b) {
  return (std::isnan(a) && std::isnan(b)) || Bits(a) == Bits(b);
}

void ExpectFloat32Matches(const Shape& s1, const Shape& s2, Route route,
                          TfLiteFusedActivation activation) {
  static float in1[kMaxElements];
  static float in2[kMaxElements];
  static float out[kMaxElements];
  const Shape so = BroadcastShape(s1, s2);
  for (int i = 0; i < s1.Count(); ++i) in1[i] = NextInt(-4000, 4000) / 64.0f;
  for (int i = 0; i < s2.Count(); ++i) {
    const int value = NextInt(-400, 400);
    in2[i] = (value == 0 ? 7 : value) / 32.0f;
  }
  EXPECT_EQ(kTfLiteOk, Run(s1, in1, s2, in2, so, out, activation));
  ExpectRoute(Leaf::kFlatF32, Leaf::kWalkF32, route);
  int mismatches = 0;
  for (int i = 0; i < so.Count(); ++i) {
    mismatches += !SameFloat(out[i],
                             ReferenceDiv(in1[BroadcastIndex(s1, so, i)],
                                          in2[BroadcastIndex(s2, so, i)],
                                          activation));
  }
  EXPECT_EQ(0, mismatches);
}

constexpr int kEdgeCount = 12;

// Numerator/denominator pairs whose IEEE-754 quotient is NaN, infinite,
// signed zero, subnormal or overflowing, plus one finite control.
void EdgeOperands(float* num, float* den) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float in1[kEdgeCount] = {nan,   1.0f,  0.0f,  inf,   1.0f,   -1.0f,
                                 1.0f,  -1.0f, -0.0f, 1e38f, 1e-30f, 6.0f};
  const float in2[kEdgeCount] = {2.0f,  -nan,  0.0f,  -inf,  0.0f,  0.0f,
                                 -inf,  inf,   3.0f,  1e-10f, 1e10f, 3.0f};
  memcpy(num, in1, sizeof(in1));
  memcpy(den, in2, sizeof(in2));
}

void ExpectFloat32EdgesMatchReference(TfLiteFusedActivation activation) {
  float in1[kEdgeCount];
  float in2[kEdgeCount];
  float out[kEdgeCount] = {};
  EdgeOperands(in1, in2);
  const Shape shape{1, {kEdgeCount}};
  EXPECT_EQ(kTfLiteOk, Run(shape, in1, shape, in2, shape, out, activation));
  ExpectRoute(Leaf::kFlatF32, Leaf::kWalkF32, kFlatF32Route);
  for (int i = 0; i < kEdgeCount; ++i) {
    const float expected = ReferenceDiv(in1[i], in2[i], activation);
    if (!SameFloat(out[i], expected)) {
      MicroPrintf("DIV f32 edge %d: %e / %e -> 0x%08x, reference 0x%08x", i,
                  static_cast<double>(in1[i]), static_cast<double>(in2[i]),
                  static_cast<unsigned>(Bits(out[i])),
                  static_cast<unsigned>(Bits(expected)));
    }
    EXPECT_TRUE(SameFloat(out[i], expected));
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ExpectFloat32EdgesMatchReference;
using tflite::testing::ExpectFloat32Matches;
using tflite::testing::kFlatF32Route;
using tflite::testing::kWalkF32Route;
using tflite::testing::Route;
using tflite::testing::Shape;

TEST(HeliaDivRouteTest, Float32EqualShapes) {
  ExpectFloat32Matches(Shape{3, {4, 5, 7}}, Shape{3, {4, 5, 7}},
                       kFlatF32Route, kTfLiteActNone);
}

TEST(HeliaDivRouteTest, Float32EqualShapesRankFiveRelu6) {
  ExpectFloat32Matches(Shape{5, {2, 3, 2, 2, 5}}, Shape{5, {2, 3, 2, 2, 5}},
                       kFlatF32Route, kTfLiteActRelu6);
}

TEST(HeliaDivRouteTest, Float32ScalarDenominator) {
  ExpectFloat32Matches(Shape{4, {1, 3, 5, 6}}, Shape{1, {1}}, kWalkF32Route,
                       kTfLiteActNone);
}

TEST(HeliaDivRouteTest, Float32ScalarNumeratorRelu) {
  ExpectFloat32Matches(Shape{1, {1}}, Shape{3, {4, 5, 3}}, kWalkF32Route,
                       kTfLiteActRelu);
}

TEST(HeliaDivRouteTest, Float32GeneralBroadcast) {
  ExpectFloat32Matches(Shape{4, {2, 1, 5, 6}}, Shape{3, {4, 1, 6}},
                       kWalkF32Route, kTfLiteActReluN1To1);
}

// Rank-5 broadcasts have no NHWC spelling and stay on the reference.
TEST(HeliaDivRouteTest, Float32RankFiveBroadcastIsReference) {
  ExpectFloat32Matches(Shape{5, {2, 3, 1, 4, 5}}, Shape{2, {4, 1}},
                       Route::kReference, kTfLiteActNone);
}

// The heliaCORE quotient and clamp give the reference's result bit for bit on
// every non-NaN edge: +/-FLT_MAX for a division by zero or an overflow (the
// unbounded clamp is +/-FLT_MAX), signed zeros, and a gradual-underflow
// subnormal. NaN cases are NaN on both (payload not asserted).
TEST(HeliaDivRouteTest, Float32EdgeValuesNoActivation) {
  ExpectFloat32EdgesMatchReference(kTfLiteActNone);
}

TEST(HeliaDivRouteTest, Float32EdgeValuesRelu) {
  ExpectFloat32EdgesMatchReference(kTfLiteActRelu);
}

#if ARM_NN_ENABLE_F16

namespace {

// The float32 quotient rounded to float16, then clamped. Selects stay in
// float: a scalar float16 conditional select ICEs GCC (PR target/118460).
float ExpectedDivF16(float16_t a, float16_t b,
                     TfLiteFusedActivation activation) {
  float lo;
  float hi;
  tflite::micro::CalculateActivationRange(activation, &lo, &hi);
  const float q = static_cast<float>(static_cast<float16_t>(
      static_cast<float>(a) / static_cast<float>(b)));
  const float lo16 =
      static_cast<float>(tflite::micro::HeliaFloat16ActivationBound(lo));
  const float hi16 =
      static_cast<float>(tflite::micro::HeliaFloat16ActivationBound(hi));
  if (std::isnan(q)) return q;
  if (q < lo16) return lo16;
  if (q > hi16) return hi16;
  return q;
}

void ExpectFloat16Matches(const Shape& s1, const Shape& s2, Route route,
                          TfLiteFusedActivation activation) {
  static float16_t in1[tflite::testing::kMaxElements];
  static float16_t in2[tflite::testing::kMaxElements];
  static float16_t out[tflite::testing::kMaxElements];
  const Shape so = tflite::testing::BroadcastShape(s1, s2);
  for (int i = 0; i < s1.Count(); ++i) {
    in1[i] = static_cast<float16_t>(tflite::testing::NextInt(-4000, 4000) /
                                    64.0f);
  }
  for (int i = 0; i < s2.Count(); ++i) {
    const int value = tflite::testing::NextInt(-400, 400);
    in2[i] = static_cast<float16_t>((value == 0 ? 7 : value) / 32.0f);
  }
  EXPECT_EQ(kTfLiteOk,
            tflite::testing::Run(s1, in1, s2, in2, so, out, activation));
  tflite::testing::ExpectRoute(Leaf::kFlatF16, Leaf::kWalkF16, route);
  int mismatches = 0;
  for (int i = 0; i < so.Count(); ++i) {
    const float expected = ExpectedDivF16(
        in1[tflite::testing::BroadcastIndex(s1, so, i)],
        in2[tflite::testing::BroadcastIndex(s2, so, i)], activation);
    mismatches += static_cast<float>(out[i]) != expected;
  }
  EXPECT_EQ(0, mismatches);
}

}  // namespace

TEST(HeliaDivRouteTest, Float16EqualShapes) {
  ExpectFloat16Matches(Shape{3, {4, 5, 7}}, Shape{3, {4, 5, 7}}, Route::kFlat,
                       kTfLiteActNone);
}

TEST(HeliaDivRouteTest, Float16GeneralBroadcastRelu6) {
  ExpectFloat16Matches(Shape{4, {2, 1, 5, 6}}, Shape{3, {4, 1, 6}},
                       Route::kWalk, kTfLiteActRelu6);
}

TEST(HeliaDivRouteTest, Float16RankFiveBroadcastRejectedAtPrepare) {
  float16_t in1[2 * 3 * 4 * 5] = {};
  float16_t in2[4] = {};
  float16_t out[2 * 3 * 4 * 5] = {};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::Run(Shape{5, {2, 3, 1, 4, 5}}, in1,
                                 Shape{2, {4, 1}}, in2,
                                 Shape{5, {2, 3, 1, 4, 5}}, out,
                                 kTfLiteActNone));
}

// NaN cases are NaN; a division by zero or an overflow saturates to the
// finite float16 bound the unbounded clamp maps to (+/-65504), where float32
// saturates to +/-FLT_MAX; signed zeros follow IEEE division.
TEST(HeliaDivRouteTest, Float16EdgeValues) {
  float num[tflite::testing::kEdgeCount];
  float den[tflite::testing::kEdgeCount];
  tflite::testing::EdgeOperands(num, den);
  // float16 has no 1e38 or 1e-30: use an overflowing and an underflowing
  // quotient within its range instead.
  num[9] = 60000.0f;
  den[9] = 0.25f;
  num[10] = 0.0001f;
  den[10] = 1000.0f;
  float16_t in1[tflite::testing::kEdgeCount];
  float16_t in2[tflite::testing::kEdgeCount];
  float16_t out[tflite::testing::kEdgeCount] = {};
  for (int i = 0; i < tflite::testing::kEdgeCount; ++i) {
    in1[i] = static_cast<float16_t>(num[i]);
    in2[i] = static_cast<float16_t>(den[i]);
  }
  const Shape shape{1, {tflite::testing::kEdgeCount}};
  EXPECT_EQ(kTfLiteOk, tflite::testing::Run(shape, in1, shape, in2, shape, out,
                                            kTfLiteActNone));
  for (int i = 0; i < tflite::testing::kEdgeCount; ++i) {
    const float want = ExpectedDivF16(in1[i], in2[i], kTfLiteActNone);
    const float got = static_cast<float>(out[i]);
    const bool same = (std::isnan(got) && std::isnan(want)) ||
                      (got == want && std::signbit(got) == std::signbit(want));
    if (!same) {
      MicroPrintf("DIV f16 edge %d: got %e, expected %e", i,
                  static_cast<double>(got), static_cast<double>(want));
    }
    EXPECT_TRUE(same);
  }
}

#elif defined(__ARM_FEATURE_MVE) && ((__ARM_FEATURE_MVE) & 2)

TEST(HeliaDivRouteTest, Float16CoverageMustNotSilentlyDisappear) {
  FAIL(
      "ARM_NN_ENABLE_F16 is not defined on a build with MVE floating point. "
      "The float16 DIV coverage silently compiled out.");
}

#endif  // ARM_NN_ENABLE_F16

TF_LITE_MICRO_TESTS_MAIN
