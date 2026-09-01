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

// Helia-specific coverage for the QUANTIZE zero_point range guard.
//
// helia's quantize kernel (kernels/helia/quantize_common.cc) dispatches
// float->int8 and float->int16 to arm_quantize_f32_s8()/arm_quantize_f32_s16().
// Those kernels reject an out-of-range zero_point and write no output, so a
// discarded status would leave the output tensor holding whatever the arena
// contained while the op still reported success. helia therefore (a) rejects
// an output zero_point that is not representable in the output element type
// during Prepare, so the failure surfaces at AllocateTensors(), and (b) checks
// the kernel status during Eval.
//
// The upstream reference kernel has no such guard, so this test lives under
// kernels/helia/tests/ and is wired in via ext_libs/helia_tests.inc, which is
// only included when OPTIMIZED_KERNEL_DIR=helia.

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

// Runs QUANTIZE's Init+Prepare over a float input and a quantized output whose
// zero_point is supplied by the caller, and returns the Prepare status.
template <typename T>
TfLiteStatus PrepareQuantizeWithOutputZeroPoint(float scale, int zero_point,
                                                T* output_data) {
  int dims_data[] = {2, 1, 4};
  TfLiteIntArray* dims = IntArrayFromInts(dims_data);
  const float input_data[] = {0.0f, 1.0f, 2.0f, 3.0f};

  TfLiteTensor output_tensor =
      CreateQuantizedTensor(output_data, dims, scale, zero_point);

  TfLiteAffineQuantization quant;
  float scales[] = {1, scale};
  int zero_points[] = {1, zero_point};
  quant.scale = FloatArrayFromFloats(scales);
  quant.zero_point = IntArrayFromInts(zero_points);
  output_tensor.quantization = {kTfLiteAffineQuantization, &quant};

  constexpr int tensors_size = 2;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, dims),
      output_tensor,
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = Register_QUANTIZE();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);
  return runner.InitAndPrepare();
}

}  // namespace
}  // namespace testing
}  // namespace tflite

// A zero_point outside the output element type's range cannot come from a
// valid TFLite model, and it is exactly what the heliaCORE quantize kernels
// reject at run time. Prepare must fail so the error surfaces at
// AllocateTensors() rather than producing an untouched output tensor mid
// inference.
TEST(HeliaQuantizeTest, PrepareRejectsOutOfRangeInt8ZeroPoint) {
  int8_t output_data[4] = {0};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareQuantizeWithOutputZeroPoint(
                /*scale=*/0.5f, /*zero_point=*/300, output_data));
  // Below the range as well as above it, so that dropping either half of the
  // comparison is caught.
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareQuantizeWithOutputZeroPoint(
                /*scale=*/0.5f, /*zero_point=*/-200, output_data));
}

TEST(HeliaQuantizeTest, PrepareRejectsOutOfRangeInt16ZeroPoint) {
  int16_t output_data[4] = {0};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareQuantizeWithOutputZeroPoint(
                /*scale=*/0.5f, /*zero_point=*/40000, output_data));
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareQuantizeWithOutputZeroPoint(
                /*scale=*/0.5f, /*zero_point=*/-40000, output_data));
}

// uint8 is the asymmetric case: the guard has to use [0, 255], not a signed
// range, so a negative zero_point is rejected where int8 would accept it.
TEST(HeliaQuantizeTest, PrepareRejectsOutOfRangeUInt8ZeroPoint) {
  uint8_t output_data[4] = {0};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareQuantizeWithOutputZeroPoint(
                /*scale=*/0.5f, /*zero_point=*/300, output_data));
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareQuantizeWithOutputZeroPoint(
                /*scale=*/0.5f, /*zero_point=*/-1, output_data));
}

// The guard must not reject anything a valid model can express: the extremes
// of each output type's range are legal zero_points.
TEST(HeliaQuantizeTest, PrepareAcceptsInRangeZeroPoints) {
  int8_t int8_output_data[4] = {0};
  EXPECT_EQ(kTfLiteOk,
            tflite::testing::PrepareQuantizeWithOutputZeroPoint(
                /*scale=*/0.5f, /*zero_point=*/-128, int8_output_data));
  EXPECT_EQ(kTfLiteOk,
            tflite::testing::PrepareQuantizeWithOutputZeroPoint(
                /*scale=*/0.5f, /*zero_point=*/127, int8_output_data));

  int16_t int16_output_data[4] = {0};
  EXPECT_EQ(kTfLiteOk,
            tflite::testing::PrepareQuantizeWithOutputZeroPoint(
                /*scale=*/0.5f, /*zero_point=*/-32768, int16_output_data));
  EXPECT_EQ(kTfLiteOk,
            tflite::testing::PrepareQuantizeWithOutputZeroPoint(
                /*scale=*/0.5f, /*zero_point=*/32767, int16_output_data));

  uint8_t uint8_output_data[4] = {0};
  EXPECT_EQ(kTfLiteOk,
            tflite::testing::PrepareQuantizeWithOutputZeroPoint(
                /*scale=*/0.5f, /*zero_point=*/0, uint8_output_data));
  EXPECT_EQ(kTfLiteOk,
            tflite::testing::PrepareQuantizeWithOutputZeroPoint(
                /*scale=*/0.5f, /*zero_point=*/255, uint8_output_data));
}

TF_LITE_MICRO_TESTS_MAIN
