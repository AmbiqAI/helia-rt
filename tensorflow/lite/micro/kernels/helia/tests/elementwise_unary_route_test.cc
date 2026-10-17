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

// ABS, NEG, SQRT and RSQRT through the helia kernels against the reference
// they replace (std::abs, unary minus, std::sqrt, 1.f / std::sqrt, and the
// quantized ABS formula). float32 ABS and NEG and quantized ABS are compared
// bit for bit on every input, NaN payloads included. SQRT and RSQRT are
// compared bit for bit on finite values, zeros, +Inf and subnormals; for
// negative and NaN inputs only the NaN class has to agree, and the heliaCORE
// bit patterns are pinned as CHARACTERIZATION. float16 has no reference and is
// asserted against the heliaCORE contract. GCC and ATfE links wrap every entry
// to prove the route and inject a failing status.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

enum class Entry {
  kAbsF32,
  kAbsS8,
  kAbsS16,
  kAbsF16,
  kSqrtF32,
  kRsqrtF32,
  kNegF32,
  kNegF16,
  kCount,
};

int g_calls[static_cast<int>(Entry::kCount)] = {};
// The wrapped entry that returns ARM_CMSIS_NN_ARG_ERROR without running.
Entry g_fail = Entry::kCount;

int Calls(Entry entry) { return g_calls[static_cast<int>(entry)]; }

}  // namespace

#if HELIA_UNARY_LINK_WRAP
#define HELIA_UNARY_WRAP(name, type, entry)                                 \
  extern "C" arm_cmsis_nn_status __real_##name(const type*, type*,          \
                                               int32_t);                    \
  extern "C" arm_cmsis_nn_status __wrap_##name(const type* input,           \
                                               type* output, int32_t n) {   \
    ++g_calls[static_cast<int>(entry)];                                     \
    if (g_fail == entry) {                                                  \
      return ARM_CMSIS_NN_ARG_ERROR;                                        \
    }                                                                       \
    return __real_##name(input, output, n);                                 \
  }

HELIA_UNARY_WRAP(arm_nn_abs_f32, float32_t, Entry::kAbsF32)
HELIA_UNARY_WRAP(arm_nn_sqrt_f32, float32_t, Entry::kSqrtF32)
HELIA_UNARY_WRAP(arm_rsqrt_f32, float32_t, Entry::kRsqrtF32)
HELIA_UNARY_WRAP(arm_nn_neg_f32, float32_t, Entry::kNegF32)
#if ARM_NN_ENABLE_F16
HELIA_UNARY_WRAP(arm_nn_abs_f16, float16_t, Entry::kAbsF16)
HELIA_UNARY_WRAP(arm_nn_neg_f16, float16_t, Entry::kNegF16)
#endif
#undef HELIA_UNARY_WRAP

#define HELIA_ABS_Q_WRAP(name, type, entry)                                  \
  extern "C" arm_cmsis_nn_status __real_##name(                              \
      const type*, int32_t, type*, int32_t, int32_t, int32_t, bool, int32_t, \
      int32_t, int32_t);                                                     \
  extern "C" arm_cmsis_nn_status __wrap_##name(                              \
      const type* input, int32_t input_offset, type* output,                 \
      int32_t out_offset, int32_t out_mult, int32_t out_shift,               \
      bool needs_rescale, int32_t act_min, int32_t act_max, int32_t n) {     \
    ++g_calls[static_cast<int>(entry)];                                      \
    if (g_fail == entry) {                                                   \
      return ARM_CMSIS_NN_ARG_ERROR;                                         \
    }                                                                        \
    return __real_##name(input, input_offset, output, out_offset, out_mult,  \
                         out_shift, needs_rescale, act_min, act_max, n);     \
  }

HELIA_ABS_Q_WRAP(arm_abs_s8, int8_t, Entry::kAbsS8)
HELIA_ABS_Q_WRAP(arm_abs_s16, int16_t, Entry::kAbsS16)
#undef HELIA_ABS_Q_WRAP
#endif  // HELIA_UNARY_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

#if HELIA_UNARY_LINK_WRAP
constexpr bool kWrapped = true;
#else
constexpr bool kWrapped = false;
#endif

// Every helia make build enables the float32 API; float16 only on M55.
#if ARM_NN_ENABLE_F16
constexpr bool kF16 = true;
#else
constexpr bool kF16 = false;
#endif

constexpr int kMaxCount = 256;

uint32_t Bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float FromBits(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t g_seed = 433;
uint32_t NextU32() {
  g_seed = g_seed * 1103515245u + 12345u;
  return g_seed;
}

// Random normal float32 values in roughly [2^-30, 2^30], either sign.
float RandomNormal(bool positive) {
  const uint32_t exponent = 97 + (NextU32() >> 8) % 60;
  const uint32_t mantissa = (NextU32() >> 4) & 0x7FFFFF;
  const uint32_t sign = positive ? 0 : (NextU32() >> 20) & 1;
  return FromBits((sign << 31) | (exponent << 23) | mantissa);
}

// Bit patterns every float32 case runs: NaN with payloads, a signaling NaN,
// infinities, signed zeros, subnormals and the finite extremes.
constexpr uint32_t kSpecialF32[] = {
    0x7FC00001u, 0xFFC00123u, 0x7F800001u, 0x7F800000u, 0xFF800000u,
    0x00000000u, 0x80000000u, 0x00000001u, 0x807FFFFFu, 0x7F7FFFFFu,
    0xFF7FFFFFu, 0x00800000u,
};
constexpr int kSpecialF32Count = sizeof(kSpecialF32) / sizeof(kSpecialF32[0]);

template <typename T>
TfLiteStatus RunUnary(const TFLMRegistration& registration, T* input,
                      T* output, int count, TfLiteType type,
                      bool quantized = false, float input_scale = 0.f,
                      int input_zero_point = 0, float output_scale = 0.f,
                      int output_zero_point = 0) {
  int dims[] = {1, count};
  TfLiteTensor tensors[2] = {
      CreateTensor(input, IntArrayFromInts(dims), false, type),
      CreateTensor(output, IntArrayFromInts(dims), false, type),
  };
  float input_scales[] = {1, input_scale};
  int input_zero_points[] = {1, input_zero_point};
  float output_scales[] = {1, output_scale};
  int output_zero_points[] = {1, output_zero_point};
  TfLiteAffineQuantization input_quant = {FloatArrayFromFloats(input_scales),
                                          IntArrayFromInts(input_zero_points),
                                          0};
  TfLiteAffineQuantization output_quant = {
      FloatArrayFromFloats(output_scales), IntArrayFromInts(output_zero_points),
      0};
  if (quantized) {
    tensors[0].params = {input_scale, input_zero_point};
    tensors[0].quantization = {kTfLiteAffineQuantization, &input_quant};
    tensors[1].params = {output_scale, output_zero_point};
    tensors[1].quantization = {kTfLiteAffineQuantization, &output_quant};
  }
  int inputs[] = {1, 0};
  int outputs[] = {1, 1};
  micro::KernelRunner runner(registration, tensors, 2,
                             IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), nullptr);
  const TfLiteStatus prepare = runner.InitAndPrepare();
  if (prepare != kTfLiteOk) {
    return prepare;
  }
  memset(g_calls, 0, sizeof(g_calls));
  return runner.Invoke();
}

void ExpectCalls(Entry entry, int expected) {
  if (kWrapped) {
    EXPECT_EQ(expected, Calls(entry));
  }
}

// Fills `input` with the special patterns followed by random normals.
int FillFloat32(float* input, bool positive_normals) {
  int n = 0;
  for (int i = 0; i < kSpecialF32Count; ++i) {
    input[n++] = FromBits(kSpecialF32[i]);
  }
  while (n < 61) {  // Not a multiple of the four-lane MVE loop.
    input[n++] = RandomNormal(positive_normals);
  }
  return n;
}

void ExpectFloat32BitExact(const TFLMRegistration& registration, Entry entry,
                           float reference(float), uint32_t contract(uint32_t)) {
  float input[kMaxCount];
  float output[kMaxCount];
  const int n = FillFloat32(input, /*positive_normals=*/false);
  EXPECT_EQ(kTfLiteOk,
            RunUnary(registration, input, output, n, kTfLiteFloat32));
  ExpectCalls(entry, 1);
  int reference_mismatches = 0;
  int contract_mismatches = 0;
  for (int i = 0; i < n; ++i) {
    reference_mismatches += Bits(output[i]) != Bits(reference(input[i]));
    contract_mismatches += Bits(output[i]) != contract(Bits(input[i]));
  }
  EXPECT_EQ(0, reference_mismatches);
  EXPECT_EQ(0, contract_mismatches);
}

// SQRT and RSQRT: bit-exact with the reference on everything except negative
// and NaN inputs, where both must produce a NaN. A negative subnormal is the
// exception on the reference side: under FPSCR.FZ it reads as -0, so the
// reference returns -0 (SQRT) or -Inf (RSQRT) where heliaCORE, which decides
// on the bit pattern, returns NaN. The characterization test logs it.
void ExpectSqrtFamilyMatchesReference(const TFLMRegistration& registration,
                                      Entry entry, float reference(float)) {
  float input[kMaxCount];
  float output[kMaxCount];
  const int n = FillFloat32(input, /*positive_normals=*/true);
  EXPECT_EQ(kTfLiteOk,
            RunUnary(registration, input, output, n, kTfLiteFloat32));
  ExpectCalls(entry, 1);
  int mismatches = 0;
  int nan_class_mismatches = 0;
  for (int i = 0; i < n; ++i) {
    const float expected = reference(input[i]);
    const uint32_t bits = Bits(input[i]);
    const bool negative_nonzero = (bits >> 31) != 0 && (bits << 1) != 0;
    const bool negative_subnormal =
        negative_nonzero && (bits & 0x7F800000u) == 0;
    if (std::isnan(input[i]) || negative_nonzero) {
      nan_class_mismatches += !std::isnan(output[i]);
      if (!negative_subnormal) {
        nan_class_mismatches += !std::isnan(expected);
      }
    } else {
      mismatches += Bits(output[i]) != Bits(expected);
    }
  }
  EXPECT_EQ(0, mismatches);
  EXPECT_EQ(0, nan_class_mismatches);
}

// CHARACTERIZATION of the heliaCORE NaN patterns for SQRT and RSQRT: a
// negative input or -Inf gives the positive quiet NaN 0x7FC00000, and a NaN
// input keeps its sign and payload with the quiet bit set. The reference
// returns whatever NaN the FPU produces (an x86 host gives 0xFFC00000 for a
// negative input; Arm FPUs give the default NaN or the quieted input, by
// FPSCR.DN). The reference bits are logged next to each pinned value.
void ExpectSqrtFamilyNanPatterns(const TFLMRegistration& registration,
                                 Entry entry, float reference(float),
                                 const char* label) {
  float input[] = {-1.0f,
                   -FromBits(0x00000001u),
                   FromBits(0xFF800000u),
                   -FLT_MAX,
                   FromBits(0x7FC00001u),
                   FromBits(0xFFC00123u),
                   FromBits(0x7F800001u)};
  const uint32_t expected[] = {0x7FC00000u, 0x7FC00000u, 0x7FC00000u,
                               0x7FC00000u, 0x7FC00001u, 0xFFC00123u,
                               0x7FC00001u};
  constexpr int n = sizeof(input) / sizeof(input[0]);
  float output[n];
  EXPECT_EQ(kTfLiteOk,
            RunUnary(registration, input, output, n, kTfLiteFloat32));
  ExpectCalls(entry, 1);
  for (int i = 0; i < n; ++i) {
    const uint32_t ref = Bits(reference(input[i]));
    if (ref != expected[i]) {
      MicroPrintf("%s: input 0x%x reference 0x%x heliaCORE 0x%x", label,
                  static_cast<unsigned>(Bits(input[i])),
                  static_cast<unsigned>(ref),
                  static_cast<unsigned>(Bits(output[i])));
    }
    EXPECT_EQ(expected[i], Bits(output[i]));
  }
}

// TFLite's quantized ABS: |x - zp_in|, requantized when the scales differ,
// plus zp_out, clamped to the type range.
template <typename T>
T ReferenceAbsQuantized(T x, float input_scale, int input_zero_point,
                        float output_scale, int output_zero_point) {
  const int32_t value = std::abs(static_cast<int32_t>(x) - input_zero_point);
  int32_t output = value + output_zero_point;
  if (input_scale != output_scale) {
    int32_t multiplier;
    int shift;
    QuantizeMultiplier(static_cast<double>(input_scale / output_scale),
                       &multiplier, &shift);
    output = MultiplyByQuantizedMultiplier(value, multiplier, shift) +
             output_zero_point;
  }
  output = std::max<int32_t>(output, std::numeric_limits<T>::min());
  output = std::min<int32_t>(output, std::numeric_limits<T>::max());
  return static_cast<T>(output);
}

template <typename T>
void ExpectAbsQuantizedBitExact(const T* input, int n, float input_scale,
                                int input_zero_point, float output_scale,
                                int output_zero_point, Entry entry) {
  T in[kMaxCount];
  T out[kMaxCount];
  memcpy(in, input, n * sizeof(T));
  const TfLiteType type = sizeof(T) == 1 ? kTfLiteInt8 : kTfLiteInt16;
  EXPECT_EQ(kTfLiteOk, RunUnary(Register_ABS(), in, out, n, type,
                                /*quantized=*/true, input_scale,
                                input_zero_point, output_scale,
                                output_zero_point));
  ExpectCalls(entry, 1);
  int mismatches = 0;
  for (int i = 0; i < n; ++i) {
    mismatches += out[i] != ReferenceAbsQuantized<T>(in[i], input_scale,
                                                     input_zero_point,
                                                     output_scale,
                                                     output_zero_point);
  }
  EXPECT_EQ(0, mismatches);
}

void ExpectAbsInt8(float input_scale, int input_zero_point, float output_scale,
                   int output_zero_point) {
  int8_t input[256];
  for (int i = 0; i < 256; ++i) {
    input[i] = static_cast<int8_t>(i - 128);
  }
  ExpectAbsQuantizedBitExact<int8_t>(input, 256, input_scale, input_zero_point,
                                     output_scale, output_zero_point,
                                     Entry::kAbsS8);
}

void ExpectAbsInt16(float input_scale, float output_scale) {
  int16_t input[kMaxCount];
  int n = 0;
  input[n++] = -32768;
  input[n++] = -32767;
  input[n++] = 32767;
  input[n++] = 0;
  input[n++] = -1;
  for (int v = -32768 + 97; v < 32768 && n < 255; v += 257) {
    input[n++] = static_cast<int16_t>(v);
  }
  ExpectAbsQuantizedBitExact<int16_t>(input, n, input_scale, 0, output_scale, 0,
                                      Entry::kAbsS16);
}

// float16 patterns as raw bits: NaN payloads, signaling NaN, infinities,
// signed zeros, subnormals, finite extremes and a few normals.
constexpr uint16_t kSpecialF16[] = {
    0x7E01u, 0xFE23u, 0x7C01u, 0x7C00u, 0xFC00u, 0x0000u, 0x8000u, 0x0001u,
    0x83FFu, 0x7BFFu, 0xFBFFu, 0x3C00u, 0xBC00u, 0x4248u, 0xC248u, 0x1234u,
    0x9234u,
};
constexpr int kSpecialF16Count = sizeof(kSpecialF16) / sizeof(kSpecialF16[0]);

void ExpectFloat16Contract(const TFLMRegistration& registration, Entry entry,
                           uint16_t contract(uint16_t)) {
  uint16_t input[kSpecialF16Count];
  uint16_t output[kSpecialF16Count];
  memcpy(input, kSpecialF16, sizeof(input));
  const TfLiteStatus status = RunUnary(registration, input, output,
                                       kSpecialF16Count, kTfLiteFloat16);
  if (!kF16) {
    // No reference path exists for float16, so Prepare rejects it.
    EXPECT_EQ(kTfLiteError, status);
    return;
  }
  EXPECT_EQ(kTfLiteOk, status);
  ExpectCalls(entry, 1);
  int mismatches = 0;
  for (int i = 0; i < kSpecialF16Count; ++i) {
    mismatches += output[i] != contract(input[i]);
  }
  EXPECT_EQ(0, mismatches);
}

void ExpectFailingEntryFailsInvoke(const TFLMRegistration& registration,
                                   Entry entry, TfLiteType type) {
  if (!kWrapped || (!kF16 && type == kTfLiteFloat16)) {
    return;
  }
  uint8_t input[8 * sizeof(float)] = {};
  uint8_t output[8 * sizeof(float)] = {};
  g_fail = entry;
  TfLiteStatus status = kTfLiteOk;
  switch (type) {
    case kTfLiteFloat32:
      status = RunUnary(registration, reinterpret_cast<float*>(input),
                        reinterpret_cast<float*>(output), 8, type);
      break;
    case kTfLiteFloat16:
      status = RunUnary(registration, reinterpret_cast<uint16_t*>(input),
                        reinterpret_cast<uint16_t*>(output), 8, type);
      break;
    case kTfLiteInt8:
      status = RunUnary(registration, reinterpret_cast<int8_t*>(input),
                        reinterpret_cast<int8_t*>(output), 8, type,
                        /*quantized=*/true, 0.5f, 0, 0.25f, 0);
      break;
    default:
      status = RunUnary(registration, reinterpret_cast<int16_t*>(input),
                        reinterpret_cast<int16_t*>(output), 8, type,
                        /*quantized=*/true, 0.5f, 0, 0.25f, 0);
      break;
  }
  g_fail = Entry::kCount;
  EXPECT_EQ(kTfLiteError, status);
  ExpectCalls(entry, 1);
}

float ReferenceAbs(float x) { return std::abs(x); }
float ReferenceNeg(float x) { return -x; }
float ReferenceSqrt(float x) { return std::sqrt(x); }
float ReferenceRsqrt(float x) { return 1.f / std::sqrt(x); }
uint32_t ContractAbsF32(uint32_t bits) { return bits & 0x7FFFFFFFu; }
uint32_t ContractNegF32(uint32_t bits) { return bits ^ 0x80000000u; }
uint16_t ContractAbsF16(uint16_t bits) {
  return static_cast<uint16_t>(bits & 0x7FFFu);
}
uint16_t ContractNegF16(uint16_t bits) {
  return static_cast<uint16_t>(bits ^ 0x8000u);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using namespace tflite::testing;  // NOLINT

TEST(HeliaElementwiseUnaryRouteTest, AbsFloat32MatchesReferenceBitForBit) {
  ExpectFloat32BitExact(tflite::Register_ABS(), Entry::kAbsF32, ReferenceAbs,
                        ContractAbsF32);
}

TEST(HeliaElementwiseUnaryRouteTest, NegFloat32MatchesReferenceBitForBit) {
  ExpectFloat32BitExact(tflite::Register_NEG(), Entry::kNegF32, ReferenceNeg,
                        ContractNegF32);
}

TEST(HeliaElementwiseUnaryRouteTest, SqrtFloat32MatchesReference) {
  ExpectSqrtFamilyMatchesReference(tflite::Register_SQRT(), Entry::kSqrtF32,
                                   ReferenceSqrt);
}

TEST(HeliaElementwiseUnaryRouteTest, RsqrtFloat32MatchesReference) {
  ExpectSqrtFamilyMatchesReference(tflite::Register_RSQRT(), Entry::kRsqrtF32,
                                   ReferenceRsqrt);
}

TEST(HeliaElementwiseUnaryRouteTest, SqrtFloat32NanPatternsCharacterized) {
  ExpectSqrtFamilyNanPatterns(tflite::Register_SQRT(), Entry::kSqrtF32,
                              ReferenceSqrt, "SQRT");
}

TEST(HeliaElementwiseUnaryRouteTest, RsqrtFloat32NanPatternsCharacterized) {
  ExpectSqrtFamilyNanPatterns(tflite::Register_RSQRT(), Entry::kRsqrtF32,
                              ReferenceRsqrt, "RSQRT");
}

TEST(HeliaElementwiseUnaryRouteTest, AbsInt8SameScaleMatchesReference) {
  ExpectAbsInt8(0.5f, 5, 0.5f, -7);
}

TEST(HeliaElementwiseUnaryRouteTest, AbsInt8RescaleMatchesReference) {
  ExpectAbsInt8(0.3f, 12, 0.17f, -20);
}

// Zero points at opposite extremes: |x - zp| reaches 255 and saturates.
TEST(HeliaElementwiseUnaryRouteTest, AbsInt8SaturationMatchesReference) {
  ExpectAbsInt8(0.1f, -128, 0.9f, 127);
  ExpectAbsInt8(1.0f, 127, 0.01f, -128);
}

TEST(HeliaElementwiseUnaryRouteTest, AbsInt16SameScaleMatchesReference) {
  ExpectAbsInt16(1.0f / 1024, 1.0f / 1024);
}

TEST(HeliaElementwiseUnaryRouteTest, AbsInt16RescaleMatchesReference) {
  ExpectAbsInt16(1.0f / 1024, 1.0f / 2048);
  ExpectAbsInt16(1.0f / 1024, 1.0f / 333);
}

// Unquantized int16 ABS stays on std::abs, which wraps -32768 to itself;
// arm_abs_s16 would clamp it to 32767.
TEST(HeliaElementwiseUnaryRouteTest, AbsInt16UnquantizedStaysOnReference) {
  int16_t input[] = {-32768, -32767, -1, 0, 1, 32767};
  int16_t output[6];
  EXPECT_EQ(kTfLiteOk, RunUnary(tflite::Register_ABS(), input, output, 6,
                                kTfLiteInt16));
  ExpectCalls(Entry::kAbsS16, 0);
  const int16_t expected[] = {-32768, 32767, 1, 0, 1, 32767};
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(expected[i], output[i]);
  }
}

TEST(HeliaElementwiseUnaryRouteTest, AbsFloat16ClearsSignBit) {
  ExpectFloat16Contract(tflite::Register_ABS(), Entry::kAbsF16,
                        ContractAbsF16);
}

TEST(HeliaElementwiseUnaryRouteTest, NegFloat16TogglesSignBit) {
  ExpectFloat16Contract(tflite::Register_NEG(), Entry::kNegF16,
                        ContractNegF16);
}

// heliaCORE rejects a zero count, so an empty tensor never reaches it.
TEST(HeliaElementwiseUnaryRouteTest, EmptyTensorSkipsEntry) {
  float input[1] = {-1.0f};
  float output[1] = {0.0f};
  EXPECT_EQ(kTfLiteOk,
            RunUnary(tflite::Register_ABS(), input, output, 0, kTfLiteFloat32));
  ExpectCalls(Entry::kAbsF32, 0);
  EXPECT_EQ(0.0f, output[0]);
}

TEST(HeliaElementwiseUnaryRouteTest, FailingEntryFailsInvoke) {
  ExpectFailingEntryFailsInvoke(tflite::Register_ABS(), Entry::kAbsF32,
                                kTfLiteFloat32);
  ExpectFailingEntryFailsInvoke(tflite::Register_ABS(), Entry::kAbsS8,
                                kTfLiteInt8);
  ExpectFailingEntryFailsInvoke(tflite::Register_ABS(), Entry::kAbsS16,
                                kTfLiteInt16);
  ExpectFailingEntryFailsInvoke(tflite::Register_ABS(), Entry::kAbsF16,
                                kTfLiteFloat16);
  ExpectFailingEntryFailsInvoke(tflite::Register_SQRT(), Entry::kSqrtF32,
                                kTfLiteFloat32);
  ExpectFailingEntryFailsInvoke(tflite::Register_RSQRT(), Entry::kRsqrtF32,
                                kTfLiteFloat32);
  ExpectFailingEntryFailsInvoke(tflite::Register_NEG(), Entry::kNegF32,
                                kTfLiteFloat32);
  ExpectFailingEntryFailsInvoke(tflite::Register_NEG(), Entry::kNegF16,
                                kTfLiteFloat16);
}

TF_LITE_MICRO_TESTS_MAIN
