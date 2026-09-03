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

// Edge-of-range (NaN, +/-Inf) coverage for the helia float TANH and LOGISTIC
// kernels. The shared kernels/tanh_test.cc and logistic_test.cc feed only
// finite values, so nothing else in the tree observes this.
// see AmbiqAI/helia-rt#227
//
// NaN behaviour splits by path, and the tests below split the same way:
//
//   TANH float32, scalar leg (cortex-m3, cortex-m4+fp)   -> NaN propagates.
//       CONTRACT. arm_nn_tanh_scalar_ref_f32 carries an explicit NaN guard,
//       which -ffinite-math-only would delete; helia builds ns-cmsis-nn at -O3
//       for gcc and ATfE, and armclang appends -ffp-mode=full after -Ofast, so
//       the guard holds there too. see AmbiqAI/helia-rt#230
//   TANH float32/float16, MVE leg (cortex-m55)           -> saturation bound.
//       CHARACTERIZATION. Documented and deliberate upstream: vminnmq is IEEE
//       minNum, so a qNaN lane is replaced by the table bound.
//       see AmbiqAI/ns-cmsis-nn#382, AmbiqAI/ns-cmsis-nn#388
//   LOGISTIC float32/float16                             -> saturation bound.
//       CHARACTERIZATION. There is no MVE sigmoid helper for either precision,
//       so LOGISTIC is always the scalar path; its exp input clamp flushes NaN
//       to +80 on purpose.
//       see AmbiqAI/ns-cmsis-nn#382, AmbiqAI/ns-cmsis-nn#388
//
// The characterization cases assert the behaviour CLASS (finite, correct sign,
// at the saturation bound) rather than a literal, because the literal moves
// when the upstream table window changes. Each logs what it observed, so the
// CI record carries the concrete number for whichever pin ran. A failure means
// upstream changed a documented behaviour: re-read the helper comments before
// touching the assertion, and convert the case back to a contract assertion if
// NaN propagation has been restored.
//
// The +/-Inf assertions are contract assertions on every path:
// tanh(+/-Inf) = +/-1, logistic(+Inf) = 1, logistic(-Inf) = 0.
//
// kernels/helia/tanh.cc and logistic.cc fall back to reference_ops::* and
// still return kTfLiteOk when the heliaCORE entry point declines, and the
// reference kernels propagate NaN, so a tightening of argument validation
// would flip these tests GREEN while the code under test never ran. The
// *PathIsReachable tests call arm_nn_activation_f32/f16 directly to pin that
// dispatch precondition; the link probe is a different guard, proving only
// that the symbol exists. see AmbiqAI/helia-rt#234

#include <cmath>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

#if ARM_NN_ENABLE_F32 || ARM_NN_ENABLE_F16
#include "arm_nnfunctions_flt.h"
#endif

// Is the heliaCORE optimized float32 path compiled in at all? On a host or
// reference build it is not, and the TFLM reference kernels run instead --
// those propagate NaN correctly, so the expectations differ.
#if ARM_NN_ENABLE_F32
#define HELIA_TEST_OPTIMIZED_F32 1
#else
#define HELIA_TEST_OPTIMIZED_F32 0
#endif

// Does this build select the MVE (vector) activation helpers? heliaCORE gates
// them on ARM_MATH_MVEF / ARM_MATH_MVE_FLOAT16, which track the compiler's
// __ARM_FEATURE_MVE; bit 1 is MVE floating point. The ATfE legs build
// cortex-m55 with +nomve and do not set it. see AmbiqAI/helia-rt#225
#if defined(__ARM_FEATURE_MVE) && ((__ARM_FEATURE_MVE) & 2)
#define HELIA_TEST_MVE_FLOAT 1
#else
#define HELIA_TEST_MVE_FLOAT 0
#endif

namespace tflite {
namespace testing {
namespace {

enum class Activation { kTanh, kLogistic };

TFLMRegistration ActivationRegistration(Activation activation) {
  return activation == Activation::kTanh ? Register_TANH()
                                         : Register_LOGISTIC();
}

// Runs the registered kernel over a 1-D tensor of `count` elements.
template <typename T>
void RunActivation(Activation activation, TfLiteType tensor_type,
                   const T* input, T* output, int count) {
  int dims_data[] = {2, 1, count};
  TfLiteTensor tensors[] = {
      CreateTensor(input, IntArrayFromInts(dims_data), false, tensor_type),
      CreateTensor(output, IntArrayFromInts(dims_data), false, tensor_type),
  };
  int inputs_array_data[] = {1, 0};
  int outputs_array_data[] = {1, 1};

  // Bound to a named local, not passed as a temporary: the registration has
  // to outlive the runner. see AmbiqAI/helia-rt#248
  const TFLMRegistration registration = ActivationRegistration(activation);
  micro::KernelRunner runner(registration, tensors, 2,
                             IntArrayFromInts(inputs_array_data),
                             IntArrayFromInts(outputs_array_data), nullptr);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

// Non-finite probe inputs. The trailing finite value keeps a normal element in
// the same tensor, so a kernel that bails out early on the non-finite lanes is
// still caught by the finite one.
constexpr int kNonFiniteCount = 4;

double ReferenceLogistic(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Shared saturation-magnitude floor. tanh(5) == 0.99991 and
// logistic(5) == 0.99331, so 0.99 is a safe floor for both.
constexpr float kSaturationFloor = 0.99f;

// Upper slack differs by precision: float32 resolves ~6e-8 near 1.0, so a few
// ULP is all that is warranted; float16 resolves ~4.9e-4 and needs more room.
constexpr float kSaturationCeilF32 = 1.0000005f;  // ~8 float32 ULP
constexpr float kLogisticFloorF32 = -2e-7f;       // ~3 float32 ULP below 0

// Asserts the "NaN was mapped to the saturation bound" behavior class without
// pinning a literal, which moves with the upstream table window. Logs the
// observed value so the CI record carries the concrete number.
void ExpectTanhNanCharacterized(float observed, const char* label) {
  MicroPrintf("%s: tanh(NaN) observed = %f", label,
              static_cast<double>(observed));
  EXPECT_FALSE(std::isnan(observed));
  EXPECT_TRUE(std::isfinite(observed));
  // Documented as NEGATIVE: Armv8.1-M defines VCMP `lt` as the logical inverse
  // of `ge`, so it is true for unordered operands and the vnegq_m negation
  // predicate fires on every NaN lane.
  EXPECT_LT(observed, 0.0f);
  EXPECT_GE(-observed, kSaturationFloor);
  EXPECT_LE(-observed, kSaturationCeilF32);
}

void ExpectLogisticNanCharacterized(float observed, const char* label) {
  MicroPrintf("%s: logistic(NaN) observed = %f", label,
              static_cast<double>(observed));
  EXPECT_FALSE(std::isnan(observed));
  EXPECT_TRUE(std::isfinite(observed));
  // The sigmoid helpers route NaN through arm_nn_softmax_exp_lut_f32, whose
  // input clamp deliberately flushes NaN to +80, giving sigmoid(80) == 1.
  EXPECT_GE(observed, kSaturationFloor);
  EXPECT_LE(observed, kSaturationCeilF32);
}

// Large-input band. Only sign and saturation are asserted here.
constexpr int kLargeCount = 8;
constexpr float kLargeInputs[kLargeCount] = {-8.0f, -7.0f, -6.0f, -5.0f,
                                             5.0f,  6.0f,  7.0f,  8.0f};

#if ARM_NN_ENABLE_F16
// Stable golden window: |x| <= 4. The heliaCORE table behavior outside this
// band has changed across releases, so exact goldens are asserted only inside
// it. Float32 goldens over the whole [-8, 8] sweep already live in the shared
// kernels/tanh_test.cc, so this window is only used by the float16 cases.
constexpr int kWindowCount = 17;
constexpr float kWindowInputs[kWindowCount] = {
    -4.0f, -3.5f, -3.0f, -2.5f, -2.0f, -1.5f, -1.0f, -0.5f, 0.0f,
    0.5f,  1.0f,  1.5f,  2.0f,  2.5f,  3.0f,  3.5f,  4.0f};

constexpr float kSaturationCeilF16 = 1.0009f;  // ~2 float16 ULP
constexpr float kLogisticFloorF16 = -0.0009f;

// Float16 golden tolerance: half precision has an 11-bit significand, so one
// ULP just below 1.0 is 2^-11 == 4.88e-4. 3e-3 is about six ULP at that
// magnitude, which covers the heliaCORE table interpolation error plus the
// round trip through float16 storage without admitting a wrong result.
constexpr float kFloat16ActivationTolerance = 3e-3f;

// Much looser than the tolerance above because heliaCORE computes float16 tanh
// two ways and the goldens hold for both: the MVE LUT (~1 ULP) and a scalar
// Pade form whose error sets this bound. A "the curve is roughly right" check,
// not a defect detector. see AmbiqAI/ns-cmsis-nn#407
constexpr float kFloat16TanhGoldenTolerance = 3e-2f;
#endif  // ARM_NN_ENABLE_F16

}  // namespace
}  // namespace testing
}  // namespace tflite

// ---------------------------------------------------------------------------
// Optimized-path reachability. See "Proving the optimized kernel actually ran"
// in the file header: without these, a silent fallback to reference_ops would
// make the tests below pass while the code under test never executed.
// ---------------------------------------------------------------------------

TEST(HeliaFloatActivationEdgeTest, OptimizedFloat32PathIsReachable) {
#if HELIA_TEST_OPTIMIZED_F32
  // Exactly the element counts the tests in this file use. 17 is deliberately
  // not a multiple of the MVE vector width, which is the case a future
  // argument-validation change would most plausibly reject.
  const int sizes[] = {tflite::testing::kNonFiniteCount,
                       tflite::testing::kLargeCount, 17};
  float input[17] = {};
  float output[17] = {};
  for (int i = 0; i < 17; ++i) {
    input[i] = 0.25f * static_cast<float>(i) - 2.0f;
  }
  for (int s = 0; s < 3; ++s) {
    EXPECT_EQ(ARM_CMSIS_NN_SUCCESS,
              arm_nn_activation_f32(input, output, sizes[s],
                                    ARM_NN_FLT_ACT_TANH, 0.0f));
    EXPECT_EQ(ARM_CMSIS_NN_SUCCESS,
              arm_nn_activation_f32(input, output, sizes[s],
                                    ARM_NN_FLT_ACT_SIGMOID, 0.0f));
  }
#else
  MicroPrintf(
      "ARM_NN_ENABLE_F32 unset: heliaCORE float32 not compiled in; the TFLM "
      "reference kernels are under test here.");
#endif
}

TEST(HeliaFloatActivationEdgeTest, TanhFloat32NanBehavior) {
  const float input[tflite::testing::kNonFiniteCount] = {NAN, INFINITY,
                                                         -INFINITY, 0.5f};
  float output[tflite::testing::kNonFiniteCount] = {};
  tflite::testing::RunActivation(tflite::testing::Activation::kTanh,
                                 kTfLiteFloat32, input, output,
                                 tflite::testing::kNonFiniteCount);

#if HELIA_TEST_OPTIMIZED_F32 && HELIA_TEST_MVE_FLOAT
  // CHARACTERIZATION: the MVE leg maps NaN to the negated saturation bound by
  // design. Not a defect; see the file header.
  tflite::testing::ExpectTanhNanCharacterized(output[0], "f32/MVE");
#else
  // CONTRACT: the scalar leg propagates NaN, and so does the TFLM reference
  // kernel used on host builds.
  EXPECT_TRUE(std::isnan(output[0]));
#endif

  // Contract on every path.
  EXPECT_NEAR(1.0f, output[1], 1e-6f);
  EXPECT_NEAR(-1.0f, output[2], 1e-6f);
  EXPECT_NEAR(std::tanh(0.5f), output[3], 1e-5f);
}

TEST(HeliaFloatActivationEdgeTest, LogisticFloat32NanBehavior) {
  const float input[tflite::testing::kNonFiniteCount] = {NAN, INFINITY,
                                                         -INFINITY, 0.5f};
  float output[tflite::testing::kNonFiniteCount] = {};
  tflite::testing::RunActivation(tflite::testing::Activation::kLogistic,
                                 kTfLiteFloat32, input, output,
                                 tflite::testing::kNonFiniteCount);

#if HELIA_TEST_OPTIMIZED_F32
  // CHARACTERIZATION: there is no MVE sigmoid helper, so this is the scalar
  // path on every target. Its exp input clamp flushes NaN to +80 on purpose,
  // giving sigmoid(80) == 1.
  // AmbiqAI/ns-cmsis-nn#388 excludes SIGMOID.
  tflite::testing::ExpectLogisticNanCharacterized(output[0], "f32");
#else
  EXPECT_TRUE(std::isnan(output[0]));
#endif

  EXPECT_NEAR(1.0f, output[1], 1e-6f);
  EXPECT_NEAR(0.0f, output[2], 1e-6f);
  EXPECT_NEAR(static_cast<float>(tflite::testing::ReferenceLogistic(0.5)),
              output[3], 1e-5f);
}

TEST(HeliaFloatActivationEdgeTest, TanhFloat32LargeInputsSaturate) {
  float output[tflite::testing::kLargeCount] = {};
  tflite::testing::RunActivation(tflite::testing::Activation::kTanh,
                                 kTfLiteFloat32, tflite::testing::kLargeInputs,
                                 output, tflite::testing::kLargeCount);

  for (int i = 0; i < tflite::testing::kLargeCount; ++i) {
    const float signed_output =
        tflite::testing::kLargeInputs[i] < 0.0f ? -output[i] : output[i];
    EXPECT_GE(signed_output, tflite::testing::kSaturationFloor);
    EXPECT_LE(signed_output, tflite::testing::kSaturationCeilF32);
  }
}

TEST(HeliaFloatActivationEdgeTest, LogisticFloat32LargeInputsSaturate) {
  float output[tflite::testing::kLargeCount] = {};
  tflite::testing::RunActivation(
      tflite::testing::Activation::kLogistic, kTfLiteFloat32,
      tflite::testing::kLargeInputs, output, tflite::testing::kLargeCount);

  for (int i = 0; i < tflite::testing::kLargeCount; ++i) {
    if (tflite::testing::kLargeInputs[i] < 0.0f) {
      EXPECT_GE(output[i], tflite::testing::kLogisticFloorF32);
      EXPECT_LE(output[i], 1.0f - tflite::testing::kSaturationFloor);
    } else {
      EXPECT_GE(output[i], tflite::testing::kSaturationFloor);
      EXPECT_LE(output[i], tflite::testing::kSaturationCeilF32);
    }
  }
}

#if ARM_NN_ENABLE_F16
TEST(HeliaFloatActivationEdgeTest, OptimizedFloat16PathIsReachable) {
  const int sizes[] = {tflite::testing::kNonFiniteCount,
                       tflite::testing::kLargeCount,
                       tflite::testing::kWindowCount};
  float16_t input[tflite::testing::kWindowCount] = {};
  float16_t output[tflite::testing::kWindowCount] = {};
  for (int i = 0; i < tflite::testing::kWindowCount; ++i) {
    input[i] = static_cast<float16_t>(0.25f * static_cast<float>(i) - 2.0f);
  }
  for (int s = 0; s < 3; ++s) {
    EXPECT_EQ(ARM_CMSIS_NN_SUCCESS,
              arm_nn_activation_f16(input, output, sizes[s],
                                    ARM_NN_FLT_ACT_TANH, 0.0f));
    EXPECT_EQ(ARM_CMSIS_NN_SUCCESS,
              arm_nn_activation_f16(input, output, sizes[s],
                                    ARM_NN_FLT_ACT_SIGMOID, 0.0f));
  }
}

TEST(HeliaFloatActivationEdgeTest, TanhFloat16NanBehavior) {
  const float16_t input[tflite::testing::kNonFiniteCount] = {
      static_cast<float16_t>(NAN), static_cast<float16_t>(INFINITY),
      static_cast<float16_t>(-INFINITY), static_cast<float16_t>(0.5f)};
  float16_t output[tflite::testing::kNonFiniteCount] = {};
  tflite::testing::RunActivation(tflite::testing::Activation::kTanh,
                                 kTfLiteFloat16, input, output,
                                 tflite::testing::kNonFiniteCount);

#if HELIA_TEST_MVE_FLOAT
  // CHARACTERIZATION: arm_nn_vtanh_lut_direct_mve_f16 has the same
  // vminnmq/vnegq_m structure as the float32 MVE helper, over a narrower
  // table window, so the expected magnitude is the float16 window's bound.
  tflite::testing::ExpectTanhNanCharacterized(
      static_cast<float>(output[0]), "f16/MVE");
#else
  // CONTRACT: arm_nn_tanh_scalar_ref_f16 is a pure rational evaluation, so a
  // NaN flows through the arithmetic untouched.
  EXPECT_TRUE(std::isnan(static_cast<float>(output[0])));
#endif

  EXPECT_NEAR(1.0f, static_cast<float>(output[1]),
              tflite::testing::kFloat16ActivationTolerance);
  EXPECT_NEAR(-1.0f, static_cast<float>(output[2]),
              tflite::testing::kFloat16ActivationTolerance);
  EXPECT_NEAR(std::tanh(0.5f), static_cast<float>(output[3]),
              tflite::testing::kFloat16ActivationTolerance);
}

TEST(HeliaFloatActivationEdgeTest, LogisticFloat16NanBehavior) {
  const float16_t input[tflite::testing::kNonFiniteCount] = {
      static_cast<float16_t>(NAN), static_cast<float16_t>(INFINITY),
      static_cast<float16_t>(-INFINITY), static_cast<float16_t>(0.5f)};
  float16_t output[tflite::testing::kNonFiniteCount] = {};
  tflite::testing::RunActivation(tflite::testing::Activation::kLogistic,
                                 kTfLiteFloat16, input, output,
                                 tflite::testing::kNonFiniteCount);

  // CHARACTERIZATION, same reason as float32: there is no MVE float16 sigmoid
  // either, so this is always arm_nn_sigmoid_scalar_f16, which routes through
  // arm_nn_softmax_exp_scalar_f16, whose clamp flushes NaN.
  tflite::testing::ExpectLogisticNanCharacterized(
      static_cast<float>(output[0]), "f16");

  EXPECT_NEAR(1.0f, static_cast<float>(output[1]),
              tflite::testing::kFloat16ActivationTolerance);
  EXPECT_NEAR(0.0f, static_cast<float>(output[2]),
              tflite::testing::kFloat16ActivationTolerance);
  EXPECT_NEAR(static_cast<float>(tflite::testing::ReferenceLogistic(0.5)),
              static_cast<float>(output[3]),
              tflite::testing::kFloat16ActivationTolerance);
}

TEST(HeliaFloatActivationEdgeTest, TanhFloat16MatchesGoldensInsideWindow) {
  float16_t input[tflite::testing::kWindowCount] = {};
  float16_t output[tflite::testing::kWindowCount] = {};
  for (int i = 0; i < tflite::testing::kWindowCount; ++i) {
    input[i] = static_cast<float16_t>(tflite::testing::kWindowInputs[i]);
  }
  tflite::testing::RunActivation(tflite::testing::Activation::kTanh,
                                 kTfLiteFloat16, input, output,
                                 tflite::testing::kWindowCount);

  for (int i = 0; i < tflite::testing::kWindowCount; ++i) {
    const float golden = static_cast<float>(static_cast<float16_t>(
        std::tanh(static_cast<double>(tflite::testing::kWindowInputs[i]))));
    EXPECT_NEAR(golden, static_cast<float>(output[i]),
                tflite::testing::kFloat16TanhGoldenTolerance);
  }
}

TEST(HeliaFloatActivationEdgeTest, LogisticFloat16MatchesGoldensInsideWindow) {
  float16_t input[tflite::testing::kWindowCount] = {};
  float16_t output[tflite::testing::kWindowCount] = {};
  for (int i = 0; i < tflite::testing::kWindowCount; ++i) {
    input[i] = static_cast<float16_t>(tflite::testing::kWindowInputs[i]);
  }
  tflite::testing::RunActivation(tflite::testing::Activation::kLogistic,
                                 kTfLiteFloat16, input, output,
                                 tflite::testing::kWindowCount);

  for (int i = 0; i < tflite::testing::kWindowCount; ++i) {
    const float golden = static_cast<float>(
        static_cast<float16_t>(tflite::testing::ReferenceLogistic(
            static_cast<double>(tflite::testing::kWindowInputs[i]))));
    EXPECT_NEAR(golden, static_cast<float>(output[i]),
                tflite::testing::kFloat16ActivationTolerance);
  }
}

TEST(HeliaFloatActivationEdgeTest, TanhFloat16LargeInputsSaturate) {
  float16_t input[tflite::testing::kLargeCount] = {};
  float16_t output[tflite::testing::kLargeCount] = {};
  for (int i = 0; i < tflite::testing::kLargeCount; ++i) {
    input[i] = static_cast<float16_t>(tflite::testing::kLargeInputs[i]);
  }
  tflite::testing::RunActivation(tflite::testing::Activation::kTanh,
                                 kTfLiteFloat16, input, output,
                                 tflite::testing::kLargeCount);

  for (int i = 0; i < tflite::testing::kLargeCount; ++i) {
    const float value = static_cast<float>(output[i]);
    const float signed_output =
        tflite::testing::kLargeInputs[i] < 0.0f ? -value : value;
    EXPECT_GE(signed_output, tflite::testing::kSaturationFloor);
    EXPECT_LE(signed_output, tflite::testing::kSaturationCeilF16);
  }
}

TEST(HeliaFloatActivationEdgeTest, LogisticFloat16LargeInputsSaturate) {
  float16_t input[tflite::testing::kLargeCount] = {};
  float16_t output[tflite::testing::kLargeCount] = {};
  for (int i = 0; i < tflite::testing::kLargeCount; ++i) {
    input[i] = static_cast<float16_t>(tflite::testing::kLargeInputs[i]);
  }
  tflite::testing::RunActivation(tflite::testing::Activation::kLogistic,
                                 kTfLiteFloat16, input, output,
                                 tflite::testing::kLargeCount);

  for (int i = 0; i < tflite::testing::kLargeCount; ++i) {
    const float value = static_cast<float>(output[i]);
    if (tflite::testing::kLargeInputs[i] < 0.0f) {
      EXPECT_GE(value, tflite::testing::kLogisticFloorF16);
      EXPECT_LE(value, 1.0f - tflite::testing::kSaturationFloor);
    } else {
      EXPECT_GE(value, tflite::testing::kSaturationFloor);
      EXPECT_LE(value, tflite::testing::kSaturationCeilF16);
    }
  }
}

#elif defined(__ARM_FEATURE_MVE) && ((__ARM_FEATURE_MVE) & 2)

// helia.inc defines ARM_NN_ENABLE_F16 for TARGET_ARCH=cortex-m55 only. If that
// match drifts, every float16 case here would vanish silently and the leg
// would still print ALL TESTS PASSED, so turn the compile-out into a loud
// failure on a build that has MVE floating point.
// see AmbiqAI/helia-rt#231, AmbiqAI/helia-rt#256
//
// Known gap: the ATfE legs build cortex-m55 with +nomve, so __ARM_FEATURE_MVE
// is unset there and this guard cannot fire. Acceptable: without MVE there is
// no body/tail split to protect. see AmbiqAI/helia-rt#225
TEST(HeliaFloatActivationEdgeTest, Float16CoverageMustNotSilentlyDisappear) {
  FAIL(
      "ARM_NN_ENABLE_F16 is not defined on a build with MVE floating point. "
      "The float16 activation coverage silently compiled out. Check "
      "ext_libs/helia.inc's TARGET_ARCH match and the ARM_NN_ENABLE_F16 macro "
      "name.");
}

#endif  // ARM_NN_ENABLE_F16

TF_LITE_MICRO_TESTS_MAIN
