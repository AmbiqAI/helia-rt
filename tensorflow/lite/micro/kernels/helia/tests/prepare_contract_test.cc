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

// Prepare/Eval contracts of the helia kernels: the elementwise ops reject an
// unsupported dtype at Prepare, and float16 MAXIMUM/MINIMUM treat an empty
// output as a no-op like the integer types do. see AmbiqAI/helia-rt#342

#include <cstdint>

#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

// Prepares a one-input elementwise node whose input and output share `type`.
template <typename T>
TfLiteStatus PrepareUnary(const TFLMRegistration& registration, T* input,
                          T* output, TfLiteType type) {
  int dims_data[] = {1, 4};
  TfLiteTensor tensors[] = {
      CreateTensor(input, IntArrayFromInts(dims_data), false, type),
      CreateTensor(output, IntArrayFromInts(dims_data), false, type),
  };
  int inputs[] = {1, 0};
  int outputs[] = {1, 1};
  micro::KernelRunner runner(registration, tensors, 2, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), nullptr);
  const TfLiteStatus status = runner.InitAndPrepare();
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
  return status;
}

// Runs float16 MAXIMUM or MINIMUM on an empty output of the given shape and
// checks nothing was written.
TfLiteStatus InvokeEmptyFloat16(const TFLMRegistration& registration,
                                int* dims_data) {
  constexpr uint16_t kSentinel = 0x7e00;
  uint16_t input1[1] = {};
  uint16_t input2[1] = {};
  uint16_t output[1] = {kSentinel};
  TfLiteTensor tensors[] = {
      CreateTensor(input1, IntArrayFromInts(dims_data), false, kTfLiteFloat16),
      CreateTensor(input2, IntArrayFromInts(dims_data), false, kTfLiteFloat16),
      CreateTensor(output, IntArrayFromInts(dims_data), false, kTfLiteFloat16),
  };
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), nullptr);
  TfLiteStatus status = runner.InitAndPrepare();
  if (status == kTfLiteOk) {
    status = runner.Invoke();
  }
  EXPECT_EQ(kSentinel, output[0]);
  return status;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(HeliaPrepareContractTest, NumericOpsRejectInt8AtPrepare) {
  using tflite::testing::PrepareUnary;
  int8_t input[4] = {};
  int8_t output[4] = {};
  EXPECT_EQ(kTfLiteError,
            PrepareUnary(tflite::Register_SIN(), input, output, kTfLiteInt8));
  EXPECT_EQ(kTfLiteError,
            PrepareUnary(tflite::Register_COS(), input, output, kTfLiteInt8));
  EXPECT_EQ(kTfLiteError,
            PrepareUnary(tflite::Register_LOG(), input, output, kTfLiteInt8));
  EXPECT_EQ(kTfLiteError, PrepareUnary(tflite::Register_SQUARE(), input,
                                       output, kTfLiteInt8));
}

TEST(HeliaPrepareContractTest, LogicalNotRejectsFloatAtPrepare) {
  float input[4] = {};
  float output[4] = {};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareUnary(tflite::Register_LOGICAL_NOT(), input,
                                          output, kTfLiteFloat32));
}

TEST(HeliaPrepareContractTest, AbsRejectsInt32AtPrepare) {
  int32_t input[4] = {};
  int32_t output[4] = {};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareUnary(tflite::Register_ABS(), input, output,
                                          kTfLiteInt32));
  uint16_t half_input[4] = {};
  uint16_t half_output[4] = {};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareUnary(tflite::Register_ABS(), half_input,
                                          half_output, kTfLiteFloat16));
}

TEST(HeliaPrepareContractTest, SupportedTypesStillPrepare) {
  using tflite::testing::PrepareUnary;
  float input[4] = {};
  float output[4] = {};
  EXPECT_EQ(kTfLiteOk,
            PrepareUnary(tflite::Register_SIN(), input, output, kTfLiteFloat32));
  bool bool_input[4] = {};
  bool bool_output[4] = {};
  EXPECT_EQ(kTfLiteOk, PrepareUnary(tflite::Register_LOGICAL_NOT(), bool_input,
                                    bool_output, kTfLiteBool));
  EXPECT_EQ(kTfLiteOk,
            PrepareUnary(tflite::Register_ABS(), input, output, kTfLiteFloat32));
}

TEST(HeliaPrepareContractTest, Float16MaximumMinimumEmptyOutput) {
  using tflite::testing::InvokeEmptyFloat16;
  int rank2[] = {2, 2, 0};
  int rank5[] = {5, 1, 1, 1, 2, 0};
#if ARM_NN_ENABLE_F16
  constexpr TfLiteStatus kExpected = kTfLiteOk;
#else
  constexpr TfLiteStatus kExpected = kTfLiteError;
#endif
  EXPECT_EQ(kExpected, InvokeEmptyFloat16(tflite::Register_MAXIMUM(), rank2));
  EXPECT_EQ(kExpected, InvokeEmptyFloat16(tflite::Register_MINIMUM(), rank2));
  EXPECT_EQ(kExpected, InvokeEmptyFloat16(tflite::Register_MAXIMUM(), rank5));
  EXPECT_EQ(kExpected, InvokeEmptyFloat16(tflite::Register_MINIMUM(), rank5));
}

#if !ARM_NN_ENABLE_F16 && defined(__ARM_FEATURE_MVE) && \
    ((__ARM_FEATURE_MVE) & 2)
// helia.inc defines ARM_NN_ENABLE_F16 for TARGET_ARCH=cortex-m55 only, and a
// silent compile-out would flip the float16 case to its rejection branch.
// see AmbiqAI/helia-rt#231
TEST(HeliaPrepareContractTest, Float16CoverageMustNotSilentlyDisappear) {
  FAIL(
      "ARM_NN_ENABLE_F16 is not defined on a build with MVE floating point. "
      "The float16 MAXIMUM/MINIMUM coverage silently compiled out.");
}
#endif

TF_LITE_MICRO_TESTS_MAIN
