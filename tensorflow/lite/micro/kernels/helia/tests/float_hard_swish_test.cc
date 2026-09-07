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

// Float HARD_SWISH coverage for the heliaCORE dispatch in
// kernels/helia/hard_swish.cc. see AmbiqAI/helia-rt#275
//
// arm_hard_swish_f32/_f16 evaluate x * clamp(fma(x, 1/6, 0.5), 0, 1); the TFLM
// reference evaluates x * clamp(x + 3, 0, 6) / 6. The two agree to within
// float rounding, so the assertions below are against the reference with a
// derived bound rather than against goldens.
//
// The element count is deliberately not a multiple of four so that an MVE
// build exercises both the vector body and the predicated tail.

#include "tensorflow/lite/kernels/internal/reference/hard_swish.h"

#include <algorithm>
#include <cmath>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

#if ARM_NN_ENABLE_F32 || ARM_NN_ENABLE_F16
#include "arm_nnfunctions_flt.h"
#endif

namespace tflite {
namespace testing {
namespace {

constexpr int kCount = 21;

// Sweeps [-4, 4] and lands exactly on the three breakpoints of the piecewise
// definition, -3, 0 and 3, plus the neighbours of the two knots where the
// kernel's gate and the reference's clamp round in opposite directions.
constexpr float kInput[kCount] = {
    -4.0f,      -3.5f,      -3.0f,      -2.9999f, -2.5f, -2.0f, -1.0f,
    -0.5f,      -0.015625f, 0.0f,       0.015625f, 0.5f, 1.0f,  2.0f,
    2.5f,       2.9999f,    3.0f,       3.0001f,  3.5f,  3.75f, 4.0f};

// |kernel - reference| over [-4, 4], where |output| <= 4. Each side applies a
// handful of correctly rounded float32 operations to the same input, so the
// spread is a small multiple of ulp(4) = 2^-22 ~ 2.4e-07; eight of those is
// 1.9e-06. Derived from the rounding, not measured: a failure here is a
// heliaCORE finding for AmbiqAI/ns-cmsis-nn#428, not a bound to widen.
constexpr float kFloat32Tolerance = 2e-6f;

#if ARM_NN_ENABLE_F16
// Same argument in float16: ulp(4) = 2^-8 = 3.90625e-03, and the MVE leg
// rounds the gate and the product separately, so it can sit two ulp from a
// singly rounded result, plus half an ulp for narrowing the reference.
constexpr float kFloat16Tolerance = 3.0f * 3.90625e-03f;
#endif

template <typename T>
void RunHardSwish(const T* input, T* output, TfLiteType tensor_type) {
  int dims_data[] = {2, 1, kCount};
  TfLiteTensor tensors[] = {
      CreateTensor(input, IntArrayFromInts(dims_data), false, tensor_type),
      CreateTensor(output, IntArrayFromInts(dims_data), false, tensor_type),
  };
  int inputs_array_data[] = {1, 0};
  int outputs_array_data[] = {1, 1};

  micro::KernelRunner runner(tflite::Register_HARD_SWISH(), tensors, 2,
                             IntArrayFromInts(inputs_array_data),
                             IntArrayFromInts(outputs_array_data), nullptr);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

}  // namespace
}  // namespace testing
}  // namespace tflite

// kernels/helia/hard_swish.cc falls back to reference_ops::HardSwish and still
// returns kTfLiteOk when the heliaCORE entry point declines, so the comparison
// below would pass with the optimized path never taken. Assert the dispatch
// precondition directly. see AmbiqAI/helia-rt#234
TEST(HeliaFloatHardSwishTest, OptimizedFloat32PathIsReachable) {
#if ARM_NN_ENABLE_F32
  float out[tflite::testing::kCount] = {};
  EXPECT_EQ(ARM_CMSIS_NN_SUCCESS,
            arm_hard_swish_f32(tflite::testing::kInput, out,
                               tflite::testing::kCount));
#else
  MicroPrintf(
      "ARM_NN_ENABLE_F32 unset: heliaCORE float32 not compiled in; the TFLM "
      "reference kernel is under test here.");
#endif
}

TEST(HeliaFloatHardSwishTest, Float32MatchesReference) {
  float output[tflite::testing::kCount] = {};
  float expected[tflite::testing::kCount] = {};

  const int32_t shape_data[] = {1, tflite::testing::kCount};
  const tflite::RuntimeShape shape(2, shape_data);
  tflite::reference_ops::HardSwish<float>(shape, tflite::testing::kInput, shape,
                                          expected);

  tflite::testing::RunHardSwish(tflite::testing::kInput, output,
                                kTfLiteFloat32);

  for (int i = 0; i < tflite::testing::kCount; ++i) {
    EXPECT_NEAR(expected[i], output[i], tflite::testing::kFloat32Tolerance);
  }
}

TEST(HeliaFloatHardSwishTest, Float32SaturatedRegionsAreExact) {
  float output[tflite::testing::kCount] = {};
  tflite::testing::RunHardSwish(tflite::testing::kInput, output,
                                kTfLiteFloat32);

  for (int i = 0; i < tflite::testing::kCount; ++i) {
    const float in = tflite::testing::kInput[i];
    if (in <= -3.0f) {
      EXPECT_EQ(0.0f, output[i]);
    } else if (in >= 3.0f) {
      EXPECT_EQ(in, output[i]);
    }
  }
}

#if ARM_NN_ENABLE_F16
TEST(HeliaFloatHardSwishTest, Float16MatchesReference) {
  float16_t input[tflite::testing::kCount] = {};
  float16_t output[tflite::testing::kCount] = {};
  for (int i = 0; i < tflite::testing::kCount; ++i) {
    input[i] = static_cast<float16_t>(tflite::testing::kInput[i]);
  }

  tflite::testing::RunHardSwish(input, output, kTfLiteFloat16);

  for (int i = 0; i < tflite::testing::kCount; ++i) {
    // The reference expression of reference_ops::HardSwish, evaluated in
    // float32 on the narrowed input.
    const float in = static_cast<float>(input[i]);
    const float expected =
        in * std::min(6.0f, std::max(0.0f, in + 3.0f)) / 6.0f;
    EXPECT_NEAR(expected, static_cast<float>(output[i]),
                tflite::testing::kFloat16Tolerance);
  }
}
#elif defined(__ARM_FEATURE_MVE) && ((__ARM_FEATURE_MVE) & 2)

// helia.inc defines ARM_NN_ENABLE_F16 for TARGET_ARCH=cortex-m55 only, and a
// silent compile-out would drop the case while the leg still reported success.
// see AmbiqAI/helia-rt#231
TEST(HeliaFloatHardSwishTest, Float16CoverageMustNotSilentlyDisappear) {
  FAIL(
      "ARM_NN_ENABLE_F16 is not defined on a build with MVE floating point. "
      "The float16 HARD_SWISH coverage silently compiled out.");
}

#endif  // ARM_NN_ENABLE_F16

TF_LITE_MICRO_TESTS_MAIN
