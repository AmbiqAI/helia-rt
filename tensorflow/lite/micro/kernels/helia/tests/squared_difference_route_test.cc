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

// SQUARED_DIFFERENCE through the helia kernel. int8 and int16 compare bit for
// bit with the TFLM quantized formula (kernels/squared_difference.cc) on equal
// shapes at any rank and on 4-D broadcasts; float32 compares with (a - b)^2 on
// equal shapes, where heliaCORE runs, and on broadcasts, which stay on the
// reference. GCC and ATfE links wrap the heliaCORE entries and assert which
// ran. The float edge cases assert IEEE results; float16 (M55) has no
// reference kernel, so it is checked against the formula evaluated in float16.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

#if ARM_NN_ENABLE_F32 || ARM_NN_ENABLE_F16
#include "Include/arm_nnfunctions_flt.h"
#endif

namespace {

enum class Leaf { kFlatS8, kWalkS8, kFlatS16, kWalkS16, kFlatF32, kFlatF16,
                  kCount };

int g_leaf_calls[static_cast<int>(Leaf::kCount)] = {};

#if HELIA_SQDIFF_LINK_WRAP
void CountCall(Leaf leaf) { ++g_leaf_calls[static_cast<int>(leaf)]; }
#endif

}  // namespace

#if HELIA_SQDIFF_LINK_WRAP
extern "C" {

#define HELIA_SQDIFF_FLAT_ARGS(T)                                           \
  const T *a, const T *b, const int32_t o1, const int32_t m1,               \
      const int32_t s1, const int32_t o2, const int32_t m2, const int32_t s2, \
      const int32_t ls, T *out, const int32_t oo, const int32_t om,          \
      const int32_t os, const int32_t lo, const int32_t hi, const int32_t n
#define HELIA_SQDIFF_FLAT_FORWARD a, b, o1, m1, s1, o2, m2, s2, ls, out, oo, om, os, lo, hi, n
#define HELIA_SQDIFF_WALK_ARGS(T)                                            \
  const T *a, const cmsis_nn_dims *da, const T *b, const cmsis_nn_dims *db,  \
      const int32_t o1, const int32_t m1, const int32_t s1, const int32_t o2, \
      const int32_t m2, const int32_t s2, const int32_t ls, T *out,           \
      const cmsis_nn_dims *dout, const int32_t oo, const int32_t om,          \
      const int32_t os, const int32_t lo, const int32_t hi
#define HELIA_SQDIFF_WALK_FORWARD \
  a, da, b, db, o1, m1, s1, o2, m2, s2, ls, out, dout, oo, om, os, lo, hi

arm_cmsis_nn_status __real_arm_elementwise_squared_difference_s8(
    HELIA_SQDIFF_FLAT_ARGS(int8_t));
arm_cmsis_nn_status __wrap_arm_elementwise_squared_difference_s8(
    HELIA_SQDIFF_FLAT_ARGS(int8_t)) {
  CountCall(Leaf::kFlatS8);
  return __real_arm_elementwise_squared_difference_s8(HELIA_SQDIFF_FLAT_FORWARD);
}
arm_cmsis_nn_status __real_arm_elementwise_squared_difference_s16(
    HELIA_SQDIFF_FLAT_ARGS(int16_t));
arm_cmsis_nn_status __wrap_arm_elementwise_squared_difference_s16(
    HELIA_SQDIFF_FLAT_ARGS(int16_t)) {
  CountCall(Leaf::kFlatS16);
  return __real_arm_elementwise_squared_difference_s16(
      HELIA_SQDIFF_FLAT_FORWARD);
}
arm_cmsis_nn_status __real_arm_squared_difference_s8(
    HELIA_SQDIFF_WALK_ARGS(int8_t));
arm_cmsis_nn_status __wrap_arm_squared_difference_s8(
    HELIA_SQDIFF_WALK_ARGS(int8_t)) {
  CountCall(Leaf::kWalkS8);
  return __real_arm_squared_difference_s8(HELIA_SQDIFF_WALK_FORWARD);
}
arm_cmsis_nn_status __real_arm_squared_difference_s16(
    HELIA_SQDIFF_WALK_ARGS(int16_t));
arm_cmsis_nn_status __wrap_arm_squared_difference_s16(
    HELIA_SQDIFF_WALK_ARGS(int16_t)) {
  CountCall(Leaf::kWalkS16);
  return __real_arm_squared_difference_s16(HELIA_SQDIFF_WALK_FORWARD);
}

#if ARM_NN_ENABLE_F32
arm_cmsis_nn_status __real_arm_elementwise_squared_difference_f32(
    const float32_t*, const float32_t*, float32_t*, int32_t);
arm_cmsis_nn_status __wrap_arm_elementwise_squared_difference_f32(
    const float32_t* a, const float32_t* b, float32_t* out, int32_t n) {
  CountCall(Leaf::kFlatF32);
  return __real_arm_elementwise_squared_difference_f32(a, b, out, n);
}
#endif

#if ARM_NN_ENABLE_F16
arm_cmsis_nn_status __real_arm_elementwise_squared_difference_f16(
    const float16_t*, const float16_t*, float16_t*, int32_t);
arm_cmsis_nn_status __wrap_arm_elementwise_squared_difference_f16(
    const float16_t* a, const float16_t* b, float16_t* out, int32_t n) {
  CountCall(Leaf::kFlatF16);
  return __real_arm_elementwise_squared_difference_f16(a, b, out, n);
}
#endif

#undef HELIA_SQDIFF_WALK_FORWARD
#undef HELIA_SQDIFF_WALK_ARGS
#undef HELIA_SQDIFF_FLAT_FORWARD
#undef HELIA_SQDIFF_FLAT_ARGS

}  // extern "C"
#endif  // HELIA_SQDIFF_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

enum class Route { kFlat, kWalk, kReference };

// MVE float builds keep float32 on the reference; see squared_difference.cc.
#if ARM_NN_ENABLE_F32 && \
    !(defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE))
constexpr Route kHeliaFloat32Route = Route::kFlat;
#else
constexpr Route kHeliaFloat32Route = Route::kReference;
#endif

void ExpectRoute(Leaf flat, Leaf walk, Route route) {
#if HELIA_SQDIFF_LINK_WRAP
  const int flat_calls = g_leaf_calls[static_cast<int>(flat)];
  const int walk_calls =
      walk == Leaf::kCount ? 0 : g_leaf_calls[static_cast<int>(walk)];
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

uint32_t g_seed = 4325;
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

// Flat index into `in` of output element `flat` under right-aligned
// broadcasting.
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
void Run(const Shape& s1, const T* in1, const Shape& s2, const T* in2,
         const Shape& so, T* out, float scale1, int zp1, float scale2, int zp2,
         float scale_out, int zp_out, TfLiteStatus expected_prepare) {
  int d1[kMaxRank + 1] = {s1.rank};
  int d2[kMaxRank + 1] = {s2.rank};
  int dout[kMaxRank + 1] = {so.rank};
  for (int i = 0; i < s1.rank; ++i) d1[i + 1] = s1.dims[i];
  for (int i = 0; i < s2.rank; ++i) d2[i + 1] = s2.dims[i];
  for (int i = 0; i < so.rank; ++i) dout[i + 1] = so.dims[i];
  TfLiteTensor tensors[3];
  if constexpr (std::numeric_limits<T>::is_integer) {
    tensors[0] = CreateQuantizedTensor(in1, IntArrayFromInts(d1), scale1, zp1);
    tensors[1] = CreateQuantizedTensor(in2, IntArrayFromInts(d2), scale2, zp2);
    tensors[2] =
        CreateQuantizedTensor(out, IntArrayFromInts(dout), scale_out, zp_out);
  } else {
    tensors[0] =
        CreateTensor(in1, IntArrayFromInts(d1), false, FloatTensorType<T>());
    tensors[1] =
        CreateTensor(in2, IntArrayFromInts(d2), false, FloatTensorType<T>());
    tensors[2] =
        CreateTensor(out, IntArrayFromInts(dout), false, FloatTensorType<T>());
  }
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = Register_SQUARED_DIFFERENCE();
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), nullptr);
  EXPECT_EQ(expected_prepare, runner.InitAndPrepare());
  if (expected_prepare != kTfLiteOk) {
    return;
  }
  memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

// The TFLM quantized formula, with the parameters its Prepare derives.
template <typename T>
T ReferenceQuantized(T x, T y, float s1, int zp1, float s2, int zp2,
                     float s_out, int zp_out) {
  const int left_shift = sizeof(T) == 1 ? 7 : 0;
  const double twice_max = 2.0 * static_cast<double>(s1 > s2 ? s1 : s2);
  int32_t m1, m2, mo;
  int sh1, sh2, sho;
  QuantizeMultiplierSmallerThanOneExp(static_cast<double>(s1) / twice_max, &m1,
                                      &sh1);
  QuantizeMultiplierSmallerThanOneExp(static_cast<double>(s2) / twice_max, &m2,
                                      &sh2);
  QuantizeMultiplier(twice_max * twice_max /
                         static_cast<double>((1 << left_shift * 2) * s_out),
                     &mo, &sho);
  const int32_t a = (x - zp1) * (1 << left_shift);
  const int32_t b = (y - zp2) * (1 << left_shift);
  const int32_t sa = MultiplyByQuantizedMultiplierSmallerThanOneExp(a, m1, sh1);
  const int32_t sb = MultiplyByQuantizedMultiplierSmallerThanOneExp(b, m2, sh2);
  const int32_t diff = sa - sb;
  int32_t raw = MultiplyByQuantizedMultiplier(diff * diff, mo, sho) + zp_out;
  raw = raw < std::numeric_limits<T>::min() ? std::numeric_limits<T>::min()
                                            : raw;
  raw = raw > std::numeric_limits<T>::max() ? std::numeric_limits<T>::max()
                                            : raw;
  return static_cast<T>(raw);
}

template <typename T>
void ExpectQuantizedMatches(const Shape& s1, const Shape& s2, Route route) {
  static T in1[kMaxElements];
  static T in2[kMaxElements];
  static T out[kMaxElements];
  constexpr bool kIs16 = sizeof(T) == 2;
  const Shape so = BroadcastShape(s1, s2);
  const int lo = kIs16 ? -32768 : -128;
  const int hi = kIs16 ? 32767 : 127;
  for (int i = 0; i < s1.Count(); ++i) in1[i] = static_cast<T>(NextInt(lo, hi));
  for (int i = 0; i < s2.Count(); ++i) in2[i] = static_cast<T>(NextInt(lo, hi));
  // Extremes on both sides exercise saturation and the widest differences.
  in1[0] = static_cast<T>(hi);
  in2[0] = static_cast<T>(lo);
  const float s_1 = kIs16 ? 1.0f / 512 : 0.05f;
  const float s_2 = kIs16 ? 1.0f / 256 : 0.08f;
  const float s_out = kIs16 ? 1.0f / 4 : 0.6f;
  const int zp1 = kIs16 ? 0 : 7;
  const int zp2 = kIs16 ? 0 : -11;
  const int zp_out = kIs16 ? 0 : -128;
  Run(s1, in1, s2, in2, so, out, s_1, zp1, s_2, zp2, s_out, zp_out, kTfLiteOk);
  ExpectRoute(kIs16 ? Leaf::kFlatS16 : Leaf::kFlatS8,
              kIs16 ? Leaf::kWalkS16 : Leaf::kWalkS8, route);
  int mismatches = 0;
  for (int i = 0; i < so.Count(); ++i) {
    const T expected = ReferenceQuantized<T>(in1[BroadcastIndex(s1, so, i)],
                                             in2[BroadcastIndex(s2, so, i)],
                                             s_1, zp1, s_2, zp2, s_out, zp_out);
    mismatches += out[i] != expected;
  }
  EXPECT_EQ(0, mismatches);
}

uint32_t Bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void ExpectFloat32Matches(const Shape& s1, const Shape& s2, Route route) {
  static float in1[kMaxElements];
  static float in2[kMaxElements];
  static float out[kMaxElements];
  const Shape so = BroadcastShape(s1, s2);
  for (int i = 0; i < s1.Count(); ++i) in1[i] = NextInt(-4000, 4000) / 128.0f;
  for (int i = 0; i < s2.Count(); ++i) in2[i] = NextInt(-4000, 4000) / 128.0f;
  Run(s1, in1, s2, in2, so, out, 0, 0, 0, 0, 0, 0, kTfLiteOk);
  ExpectRoute(Leaf::kFlatF32, Leaf::kCount, route);
  int mismatches = 0;
  for (int i = 0; i < so.Count(); ++i) {
    const float d =
        in1[BroadcastIndex(s1, so, i)] - in2[BroadcastIndex(s2, so, i)];
    mismatches += Bits(out[i]) != Bits(d * d);
  }
  EXPECT_EQ(0, mismatches);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ExpectFloat32Matches;
using tflite::testing::ExpectQuantizedMatches;
using tflite::testing::Route;
using tflite::testing::Shape;

TEST(HeliaSquaredDifferenceRouteTest, Int8EqualShapes) {
  ExpectQuantizedMatches<int8_t>(Shape{2, {7, 9}}, Shape{2, {7, 9}},
                                 Route::kFlat);
}

// Rank 5 has no NHWC spelling, but equal shapes still take the flat kernel.
TEST(HeliaSquaredDifferenceRouteTest, Int8EqualShapesRankFive) {
  ExpectQuantizedMatches<int8_t>(Shape{5, {2, 3, 2, 2, 5}},
                                 Shape{5, {2, 3, 2, 2, 5}}, Route::kFlat);
}

TEST(HeliaSquaredDifferenceRouteTest, Int8ScalarSecondOperand) {
  ExpectQuantizedMatches<int8_t>(Shape{4, {1, 3, 5, 6}}, Shape{1, {1}},
                                 Route::kWalk);
}

TEST(HeliaSquaredDifferenceRouteTest, Int8ScalarFirstOperand) {
  ExpectQuantizedMatches<int8_t>(Shape{1, {1}}, Shape{3, {4, 5, 3}},
                                 Route::kWalk);
}

TEST(HeliaSquaredDifferenceRouteTest, Int8GeneralBroadcast) {
  ExpectQuantizedMatches<int8_t>(Shape{4, {2, 1, 5, 6}}, Shape{3, {4, 1, 6}},
                                 Route::kWalk);
}

TEST(HeliaSquaredDifferenceRouteTest, Int16EqualShapes) {
  ExpectQuantizedMatches<int16_t>(Shape{3, {3, 5, 7}}, Shape{3, {3, 5, 7}},
                                  Route::kFlat);
}

TEST(HeliaSquaredDifferenceRouteTest, Int16PerChannelBroadcast) {
  ExpectQuantizedMatches<int16_t>(Shape{4, {1, 4, 5, 6}}, Shape{1, {6}},
                                  Route::kWalk);
}

TEST(HeliaSquaredDifferenceRouteTest, Int16GeneralBroadcast) {
  ExpectQuantizedMatches<int16_t>(Shape{4, {2, 3, 1, 5}},
                                  Shape{4, {1, 1, 4, 5}}, Route::kWalk);
}

TEST(HeliaSquaredDifferenceRouteTest, Float32EqualShapes) {
  ExpectFloat32Matches(Shape{3, {4, 5, 7}}, Shape{3, {4, 5, 7}},
                       tflite::testing::kHeliaFloat32Route);
}

// heliaCORE has no float32 broadcast squared difference.
TEST(HeliaSquaredDifferenceRouteTest, Float32BroadcastIsReference) {
  ExpectFloat32Matches(Shape{4, {2, 3, 4, 5}}, Shape{1, {5}},
                       Route::kReference);
}

// IEEE-754 results, which the reference kernel's (a - b) * (a - b) gives too:
// NaN in either operand and Inf - Inf give NaN (payload not asserted), an
// infinite or overflowing difference gives +Inf, and equal operands give +0.
TEST(HeliaSquaredDifferenceRouteTest, Float32EdgeValues) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float denorm = std::numeric_limits<float>::denorm_min();
  float in1[] = {nan, 1.0f, inf, inf, -0.0f, 3e38f, denorm, 3.0f};
  float in2[] = {1.0f, -nan, inf, -inf, 0.0f, -3e38f, 0.0f, 1.0f};
  float out[8] = {};
  const Shape shape{1, {8}};
  tflite::testing::Run(shape, in1, shape, in2, shape, out, 0, 0, 0, 0, 0, 0,
                       kTfLiteOk);
  tflite::testing::ExpectRoute(Leaf::kFlatF32, Leaf::kCount,
                               tflite::testing::kHeliaFloat32Route);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_TRUE(std::isnan(out[1]));
  EXPECT_TRUE(std::isnan(out[2]));
  EXPECT_EQ(tflite::testing::Bits(inf), tflite::testing::Bits(out[3]));
  EXPECT_EQ(tflite::testing::Bits(0.0f), tflite::testing::Bits(out[4]));
  EXPECT_EQ(tflite::testing::Bits(inf), tflite::testing::Bits(out[5]));
  // denorm_min^2 underflows to +0 with or without subnormal flushing.
  EXPECT_EQ(tflite::testing::Bits(0.0f), tflite::testing::Bits(out[6]));
  EXPECT_EQ(tflite::testing::Bits(4.0f), tflite::testing::Bits(out[7]));
}

// Normal operands whose squared difference is subnormal keep gradual
// underflow. The expected bits come from double arithmetic, which no float32
// flush mode reaches.
// CHARACTERIZATION on clang builds with MVE float: clang may auto-vectorize
// the reference kernel's loop with MVE, which flushes subnormal results, so
// each element there is the IEEE result or +0; the observed bits print.
TEST(HeliaSquaredDifferenceRouteTest, Float32SubnormalResultsMatchReference) {
  float in1[] = {1e-20f, 1e-20f, 1e-19f, 3e-20f, -2e-20f, 1e-22f, 1e-23f, 3.0f};
  float in2[] = {0.0f, -1e-20f, 0.0f, 1e-20f, 0.0f, 0.0f, 0.0f, 1.0f};
  float out[8] = {};
  const Shape shape{1, {8}};
  tflite::testing::Run(shape, in1, shape, in2, shape, out, 0, 0, 0, 0, 0, 0,
                       kTfLiteOk);
  tflite::testing::ExpectRoute(Leaf::kFlatF32, Leaf::kCount,
                               tflite::testing::kHeliaFloat32Route);
#if defined(__clang__) && defined(__ARM_FEATURE_MVE) && \
    ((__ARM_FEATURE_MVE) & 2)
  constexpr bool kFlushAllowed = true;
#else
  constexpr bool kFlushAllowed = false;
#endif
  int mismatches = 0;
  for (int i = 0; i < 8; ++i) {
    const double d = static_cast<double>(in1[i] - in2[i]);
    const float expected = static_cast<float>(d * d);
    const bool flushed = kFlushAllowed &&
                         std::fpclassify(expected) == FP_SUBNORMAL &&
                         tflite::testing::Bits(out[i]) == 0u;
    if (flushed) {
      MicroPrintf("SQUARED_DIFFERENCE f32 subnormal %d: flushed to +0", i);
    } else if (tflite::testing::Bits(out[i]) !=
               tflite::testing::Bits(expected)) {
      MicroPrintf("SQUARED_DIFFERENCE f32 subnormal %d: 0x%08x, expected 0x%08x",
                  i, static_cast<unsigned>(tflite::testing::Bits(out[i])),
                  static_cast<unsigned>(tflite::testing::Bits(expected)));
      ++mismatches;
    }
  }
  EXPECT_EQ(0, mismatches);
}

#if ARM_NN_ENABLE_F16

TEST(HeliaSquaredDifferenceRouteTest, Float16EqualShapes) {
  float16_t in1[24];
  float16_t in2[24];
  float16_t out[24] = {};
  for (int i = 0; i < 24; ++i) {
    in1[i] = static_cast<float16_t>(tflite::testing::NextInt(-512, 512) / 64.0f);
    in2[i] = static_cast<float16_t>(tflite::testing::NextInt(-512, 512) / 64.0f);
  }
  const Shape shape{3, {2, 3, 4}};
  tflite::testing::Run(shape, in1, shape, in2, shape, out, 0, 0, 0, 0, 0, 0,
                       kTfLiteOk);
  tflite::testing::ExpectRoute(Leaf::kFlatF16, Leaf::kCount, Route::kFlat);
  int mismatches = 0;
  for (int i = 0; i < 24; ++i) {
    // Inputs on a 1/64 grid within +/-8 subtract exactly in float.
    const float16_t d = static_cast<float16_t>(static_cast<float>(in1[i]) -
                                               static_cast<float>(in2[i]));
    const float16_t expected =
        static_cast<float16_t>(static_cast<float>(d) * static_cast<float>(d));
    mismatches += static_cast<float>(out[i]) != static_cast<float>(expected);
  }
  EXPECT_EQ(0, mismatches);
}

TEST(HeliaSquaredDifferenceRouteTest, Float16BroadcastRejectedAtPrepare) {
  float16_t in1[6] = {};
  float16_t in2[1] = {};
  float16_t out[6] = {};
  tflite::testing::Run(Shape{2, {2, 3}}, in1, Shape{1, {1}}, in2,
                       Shape{2, {2, 3}}, out, 0, 0, 0, 0, 0, 0, kTfLiteError);
}

// There is no float16 reference; these are the IEEE-754 results: NaN for NaN
// and Inf - Inf, +Inf for an overflowing difference, and 2^-24 (the smallest
// float16 subnormal) for (2^-12)^2, which the MVE float16 leg keeps rather
// than flushing.
TEST(HeliaSquaredDifferenceRouteTest, Float16EdgeValues) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  float16_t in1[] = {static_cast<float16_t>(nan), static_cast<float16_t>(inf),
                     static_cast<float16_t>(60000.0f),
                     static_cast<float16_t>(0.000244140625f),
                     static_cast<float16_t>(3.0f)};
  float16_t in2[] = {static_cast<float16_t>(1.0f), static_cast<float16_t>(inf),
                     static_cast<float16_t>(-60000.0f),
                     static_cast<float16_t>(0.0f),
                     static_cast<float16_t>(1.0f)};
  float16_t out[5] = {};
  const Shape shape{1, {5}};
  tflite::testing::Run(shape, in1, shape, in2, shape, out, 0, 0, 0, 0, 0, 0,
                       kTfLiteOk);
  EXPECT_TRUE(std::isnan(static_cast<float>(out[0])));
  EXPECT_TRUE(std::isnan(static_cast<float>(out[1])));
  EXPECT_TRUE(std::isinf(static_cast<float>(out[2])));
  EXPECT_TRUE(static_cast<float>(out[2]) > 0.0f);
  EXPECT_EQ(5.9604644775390625e-08f, static_cast<float>(out[3]));
  EXPECT_EQ(4.0f, static_cast<float>(out[4]));
}

#elif defined(__ARM_FEATURE_MVE) && ((__ARM_FEATURE_MVE) & 2)

TEST(HeliaSquaredDifferenceRouteTest, Float16CoverageMustNotSilentlyDisappear) {
  FAIL(
      "ARM_NN_ENABLE_F16 is not defined on a build with MVE floating point. "
      "The float16 SQUARED_DIFFERENCE coverage silently compiled out.");
}

#endif  // ARM_NN_ENABLE_F16

TF_LITE_MICRO_TESTS_MAIN
