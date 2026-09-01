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

// Coverage for a 3-input (bias omitted) int16 TRANSPOSE_CONV.
//
// bias is optional for TRANSPOSE_CONV, and the kernel's Eval path already
// guards on `NumInputs(node) == 4`. Prepare did not: CalculateOpData() read
// `bias->type` unconditionally on the int16 branch, so a legal 3-input int16
// model dereferenced null inside AllocateTensors(). The upstream test helper
// hardcodes 4 inputs (see the "TODO(b/358151309): support optional bias
// tensor" in kernels/transpose_conv_test.cc), which is why the case was never
// exercised.
//
// This lives under kernels/helia/tests/ rather than in the upstream test file
// because the fix is helia-side; adding it upstream would fail the reference
// builds that still carry the defect. Wired in via ext_libs/helia_tests.inc,
// which is only included when OPTIMIZED_KERNEL_DIR=helia.

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

// Transpose conv uses TfLiteConvParams.
const TfLiteConvParams kConvParams = {
    kTfLitePaddingSame,  // padding
    1,                   // stride_width
    1,                   // stride_height
    kTfLiteActNone,
    1,  // dilation_width_factor
    1,  // dilation_height_factor
    kTfLiteNoType};

constexpr int kElements = 4;

int kInputShape[] = {4, 1, 2, 2, 1};
int kFilterShape[] = {4, 1, 2, 2, 1};
int kOutputShape[] = {4, 1, 2, 2, 1};

const float kInputData[kElements] = {1.0f, 2.0f, 3.0f, 4.0f};
const float kFilterData[kElements] = {1.0f, 0.0f, 0.0f, 1.0f};

}  // namespace
}  // namespace testing
}  // namespace tflite

// The assertion is that Prepare and Invoke complete at all: before the fix
// this dereferenced a null bias tensor inside Prepare.
TEST(HeliaTransposeConvTest, Int16WithoutBiasPreparesAndInvokes) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateSymmetricPerChannelQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;

  TfLiteIntArray* input_dims =
      IntArrayFromInts(tflite::testing::kInputShape);
  TfLiteIntArray* filter_dims =
      IntArrayFromInts(tflite::testing::kFilterShape);
  TfLiteIntArray* output_dims =
      IntArrayFromInts(tflite::testing::kOutputShape);

  int16_t input_quantized[tflite::testing::kElements];
  int8_t filter_quantized[tflite::testing::kElements];
  int16_t output_data[tflite::testing::kElements];

  int filter_zero_points[2];
  float filter_scales[2];
  TfLiteAffineQuantization filter_quant;
  TfLiteTensor filter_tensor = CreateSymmetricPerChannelQuantizedTensor(
      tflite::testing::kFilterData, filter_quantized, filter_dims,
      filter_scales, filter_zero_points, &filter_quant,
      /*quantized_dimension=*/0);

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims =
      IntArrayFromInts(output_shape_dims_data);

  // Three inputs only: output_shape, filter, input. No bias tensor.
  constexpr int tensors_size = 4;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims),
      filter_tensor,
      CreateQuantizedTensor(tflite::testing::kInputData, input_quantized,
                            input_dims, /*scale=*/1.0f / 32.0f,
                            /*zero_point=*/0),
      CreateQuantizedTensor(output_data, output_dims, /*scale=*/1.0f / 32.0f,
                            /*zero_point=*/0),
  };

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_TRANSPOSE_CONV();
  tflite::micro::KernelRunner runner(registration, tensors, tensors_size,
                                     inputs_array, outputs_array,
                                     &tflite::testing::kConvParams);

  const char* init_data =
      reinterpret_cast<const char*>(&tflite::testing::kConvParams);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare(init_data));
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

TF_LITE_MICRO_TESTS_MAIN
