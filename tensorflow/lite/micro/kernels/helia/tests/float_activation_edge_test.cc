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

// Edge-of-range coverage for the helia float TANH and LOGISTIC kernels
// (AmbiqAI/helia-rt#227, ns#314 / ns#303 defect classes).
//
// What the shared upstream tests already execute, and why it is not enough:
//   * kernels/tanh_test.cc sweeps float32 tanh over [-8, 8] with goldens, and
//     has one four-point float16 golden case ({-2, 0.5, 2, 4}).
//   * kernels/logistic_test.cc has float32 and float16 golden cases.
//   * Neither file feeds a non-finite value to any float kernel, so nothing in
//     the tree observes NaN or +/-Inf propagation through the heliaCORE
//     activation entry points. ns#314 is exactly that class of defect, and it
//     manifests on the armclang libraries because of the fast-math setting
//     tracked in helia-rt#228.
//
// This file adds:
//   1. NaN and +/-Inf propagation cases for float32 and float16 TANH and
//      LOGISTIC. NaN in must give NaN out; tanh(+/-Inf) must be +/-1;
//      logistic(+Inf) must be 1 and logistic(-Inf) must be 0.
//   2. A float16 golden sweep over the stable part of the activation table
//      window, |x| <= 4.
//   3. Separate large-input cases for |x| in [4, 8] that assert only sign and
//      saturation, never an exact value. ns#303 widened the heliaCORE table
//      window in v7.30.0, so the precise values in that band legitimately
//      change with the pin; sign and saturation do not.
//
// Note on the gcc / ATfE legs: those libraries are built at -O3 without
// fast-math, so the NaN cases can pass there even on ns-cmsis-nn v7.29.2.
// They are still the guard for the armclang path (helia-rt#228) and for any
// future backend change.

#include <cmath>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

#if ARM_NN_ENABLE_F16
#include "arm_nnfunctions_flt.h"
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

  // micro::KernelRunner keeps a `const TFLMRegistration&`, so the registration
  // has to outlive the runner: bind it to a named local rather than passing a
  // temporary.
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

#if ARM_NN_ENABLE_F16
// Stable golden window: |x| <= 4. ns#303 changed heliaCORE table behavior
// outside this band in v7.30.0, so exact goldens are only asserted inside it.
// Float32 goldens over the whole [-8, 8] sweep already live in the shared
// kernels/tanh_test.cc, so this window is only used by the float16 cases.
constexpr int kWindowCount = 17;
constexpr float kWindowInputs[kWindowCount] = {
    -4.0f, -3.5f, -3.0f, -2.5f, -2.0f, -1.5f, -1.0f, -0.5f, 0.0f,
    0.5f,  1.0f,  1.5f,  2.0f,  2.5f,  3.0f,  3.5f,  4.0f};
#endif  // ARM_NN_ENABLE_F16

// Large-input band. Only sign and saturation are asserted here.
constexpr int kLargeCount = 8;
constexpr float kLargeInputs[kLargeCount] = {-8.0f, -7.0f, -6.0f, -5.0f,
                                             5.0f,  6.0f,  7.0f,  8.0f};

#if ARM_NN_ENABLE_F16
// Float16 golden tolerance: half precision has an 11-bit significand, so one
// ULP just below 1.0 is 2^-11 == 4.88e-4. 3e-3 is about six ULP at that
// magnitude, which covers the heliaCORE table interpolation error plus the
// round trip through float16 storage without admitting a wrong result.
constexpr float kFloat16ActivationTolerance = 3e-3f;

// Float16 TANH golden tolerance is deliberately much looser than the tolerance
// above, because heliaCORE computes float16 tanh two different ways and the
// goldens have to hold for both:
//   * MVE builds route TANH through arm_nn_vtanh_lut_direct_mve_f16, whose
//     LUT256 table IS the golden curve, so the error there is ~1 ULP.
//   * Scalar builds (any non-MVE float16 target, e.g. ATfE's +nomve) use
//     arm_nn_tanh_scalar_ref_f16, a Pade form x(27+x^2)/(27+9x^2) with
//     coefficients {3, 27, 9}. That approximation deviates from true tanh by up
//     to 2.34e-2 at x = +/-1.5 (also 2.00e-2 at +/-2.0, 1.61e-2 at +/-1.0):
//     12 of the 17 window points below exceed 3e-3.
// 3e-2 is therefore set by the scalar rational path, not by rounding. This
// assertion is a "the curve is roughly right" check; it is NOT a defect
// detector, and it must not be read as one.
constexpr float kFloat16TanhGoldenTolerance = 3e-2f;
#endif  // ARM_NN_ENABLE_F16

// Saturation band for |x| >= 5. tanh(5) == 0.99991 and logistic(5) == 0.99331,
// so 0.99 is a safe floor for both; the small upper slack absorbs a table
// value that rounds a hair above the mathematical limit in float16.
constexpr float kSaturationFloor = 0.99f;
constexpr float kSaturationCeil = 1.0009f;

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(HeliaFloatActivationEdgeTest, TanhFloat32PropagatesNanAndInf) {
  const float input[tflite::testing::kNonFiniteCount] = {
      NAN, INFINITY, -INFINITY, 0.5f};
  float output[tflite::testing::kNonFiniteCount] = {};
  tflite::testing::RunActivation(tflite::testing::Activation::kTanh,
                                 kTfLiteFloat32, input, output,
                                 tflite::testing::kNonFiniteCount);

  EXPECT_TRUE(std::isnan(output[0]));
  EXPECT_NEAR(1.0f, output[1], 1e-6f);
  EXPECT_NEAR(-1.0f, output[2], 1e-6f);
  EXPECT_NEAR(std::tanh(0.5f), output[3], 1e-5f);
}

TEST(HeliaFloatActivationEdgeTest, LogisticFloat32PropagatesNanAndInf) {
  const float input[tflite::testing::kNonFiniteCount] = {
      NAN, INFINITY, -INFINITY, 0.5f};
  float output[tflite::testing::kNonFiniteCount] = {};
  tflite::testing::RunActivation(tflite::testing::Activation::kLogistic,
                                 kTfLiteFloat32, input, output,
                                 tflite::testing::kNonFiniteCount);

  EXPECT_TRUE(std::isnan(output[0]));
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
    EXPECT_LE(signed_output, tflite::testing::kSaturationCeil);
  }
}

TEST(HeliaFloatActivationEdgeTest, LogisticFloat32LargeInputsSaturate) {
  float output[tflite::testing::kLargeCount] = {};
  tflite::testing::RunActivation(
      tflite::testing::Activation::kLogistic, kTfLiteFloat32,
      tflite::testing::kLargeInputs, output, tflite::testing::kLargeCount);

  for (int i = 0; i < tflite::testing::kLargeCount; ++i) {
    if (tflite::testing::kLargeInputs[i] < 0.0f) {
      EXPECT_GE(output[i], -0.0009f);
      EXPECT_LE(output[i], 1.0f - tflite::testing::kSaturationFloor);
    } else {
      EXPECT_GE(output[i], tflite::testing::kSaturationFloor);
      EXPECT_LE(output[i], tflite::testing::kSaturationCeil);
    }
  }
}

#if ARM_NN_ENABLE_F16
TEST(HeliaFloatActivationEdgeTest, TanhFloat16PropagatesNanAndInf) {
  const float16_t input[tflite::testing::kNonFiniteCount] = {
      static_cast<float16_t>(NAN), static_cast<float16_t>(INFINITY),
      static_cast<float16_t>(-INFINITY), static_cast<float16_t>(0.5f)};
  float16_t output[tflite::testing::kNonFiniteCount] = {};
  tflite::testing::RunActivation(tflite::testing::Activation::kTanh,
                                 kTfLiteFloat16, input, output,
                                 tflite::testing::kNonFiniteCount);

  EXPECT_TRUE(std::isnan(static_cast<float>(output[0])));
  EXPECT_NEAR(1.0f, static_cast<float>(output[1]),
              tflite::testing::kFloat16ActivationTolerance);
  EXPECT_NEAR(-1.0f, static_cast<float>(output[2]),
              tflite::testing::kFloat16ActivationTolerance);
  EXPECT_NEAR(std::tanh(0.5f), static_cast<float>(output[3]),
              tflite::testing::kFloat16ActivationTolerance);
}

TEST(HeliaFloatActivationEdgeTest, LogisticFloat16PropagatesNanAndInf) {
  const float16_t input[tflite::testing::kNonFiniteCount] = {
      static_cast<float16_t>(NAN), static_cast<float16_t>(INFINITY),
      static_cast<float16_t>(-INFINITY), static_cast<float16_t>(0.5f)};
  float16_t output[tflite::testing::kNonFiniteCount] = {};
  tflite::testing::RunActivation(tflite::testing::Activation::kLogistic,
                                 kTfLiteFloat16, input, output,
                                 tflite::testing::kNonFiniteCount);

  EXPECT_TRUE(std::isnan(static_cast<float>(output[0])));
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
    EXPECT_LE(signed_output, tflite::testing::kSaturationCeil);
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
      EXPECT_GE(value, -0.0009f);
      EXPECT_LE(value, 1.0f - tflite::testing::kSaturationFloor);
    } else {
      EXPECT_GE(value, tflite::testing::kSaturationFloor);
      EXPECT_LE(value, tflite::testing::kSaturationCeil);
    }
  }
}
#endif  // ARM_NN_ENABLE_F16

TF_LITE_MICRO_TESTS_MAIN
