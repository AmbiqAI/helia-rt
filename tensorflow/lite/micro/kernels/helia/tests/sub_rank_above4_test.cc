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

// int16 SUB above rank 4, which the helia kernel sends to the reference
// broadcast. The scales are not powers of two apart, so a power-of-two-only
// routine fails. int16 SUB requires zero point 0. Kept out of the shared
// sub_test.cc because other backends cap SUB below rank 6.
// see AmbiqAI/helia-rt#334

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

// Quantized outputs must match the golden exactly.
void TestSubQuantized(int* input1_dims_data, const float* input1_data,
                      int16_t* input1_quantized, float input1_scale,
                      int input1_zero_point, int* input2_dims_data,
                      const float* input2_data, int16_t* input2_quantized,
                      float input2_scale, int input2_zero_point,
                      int* output_dims_data, const float* golden,
                      int16_t* golden_quantized, float output_scale,
                      int output_zero_point, TfLiteFusedActivation activation,
                      int16_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int kTensorsSize = 3;
  TfLiteTensor tensors[kTensorsSize] = {
      CreateQuantizedTensor(input1_data, input1_quantized, input1_dims,
                            input1_scale, input1_zero_point),
      CreateQuantizedTensor(input2_data, input2_quantized, input2_dims,
                            input2_scale, input2_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };
  const int output_size = ElementCount(*output_dims);
  tflite::Quantize(golden, golden_quantized, output_size, output_scale,
                   output_zero_point);

  TfLiteSubParams builtin_data;
  builtin_data.activation = activation;
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_SUB();
  micro::KernelRunner runner(registration, tensors, kTensorsSize, inputs_array,
                             outputs_array, &builtin_data);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  for (int i = 0; i < output_size; ++i) {
    EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(SubRankAbove4Test, QuantizedSubRankAbove4Int16) {
  const float scales[] = {0.06, 0.04, 0.1};
  const int zero_points[] = {0, 0, 0};
  const int output_dims_count = 6;

  constexpr int num_shapes = 2;
  constexpr int max_shape_size = 7;
  int test_shapes[num_shapes][max_shape_size] = {
      {5, 1, 2, 1, 3, 1},
      {6, 1, 1, 2, 1, 3, 1},
  };

  const float input1_values[] = {0.6, 1.2, -0.6, 0.3, 2.4, 0.0};
  const float input2_values[] = {0.4, 0.2, 0.4, 0.8, -1.6, 1.2};
  const float golden_values[] = {0.2, 1.0, -1.0, -0.5, 4.0, -1.2};

  int16_t input1_quantized[output_dims_count];
  int16_t input2_quantized[output_dims_count];
  int16_t golden_quantized[output_dims_count];
  int16_t output[output_dims_count];

  for (int i = 0; i < num_shapes; i++) {
    tflite::testing::TestSubQuantized(
        test_shapes[i], input1_values, input1_quantized, scales[0],
        zero_points[0], test_shapes[i], input2_values, input2_quantized,
        scales[1], zero_points[1], test_shapes[i], golden_values,
        golden_quantized, scales[2], zero_points[2], kTfLiteActNone, output);
  }
}

TEST(SubRankAbove4Test, QuantizedSubRankAbove4BroadcastInt16) {
  const float scales[] = {0.06, 0.04, 0.1};
  const int zero_points[] = {0, 0, 0};
  const int output_dims_count = 6;

  int input1_shape[] = {5, 1, 2, 1, 3, 1};
  int input2_shape[] = {1, 1};
  const float input1_values[] = {0.6, 1.2, -0.6, 0.3, 2.4, 0.0};
  const float input2_values[] = {0.4};
  const float golden_values[] = {0.2, 0.8, -1.0, -0.1, 2.0, -0.4};

  int16_t input1_quantized[output_dims_count];
  int16_t input2_quantized[1];
  int16_t golden_quantized[output_dims_count];
  int16_t output[output_dims_count];

  tflite::testing::TestSubQuantized(
      input1_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      input2_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      input1_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActNone, output);
}

TEST(SubRankAbove4Test, QuantizedSubRankAbove4SecondInputInt16) {
  const float scales[] = {0.04, 0.06, 0.1};
  const int zero_points[] = {0, 0, 0};
  const int output_dims_count = 6;

  int input1_shape[] = {1, 1};
  int input2_shape[] = {5, 1, 2, 1, 3, 1};
  const float input1_values[] = {0.4};
  const float input2_values[] = {0.6, 1.2, -0.6, 0.3, 2.4, 0.0};
  const float golden_values[] = {-0.2, -0.8, 1.0, 0.1, -2.0, 0.4};

  int16_t input1_quantized[1];
  int16_t input2_quantized[output_dims_count];
  int16_t golden_quantized[output_dims_count];
  int16_t output[output_dims_count];

  tflite::testing::TestSubQuantized(
      input1_shape, input1_values, input1_quantized, scales[0], zero_points[0],
      input2_shape, input2_values, input2_quantized, scales[1], zero_points[1],
      input2_shape, golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActNone, output);
}

TEST(SubRankAbove4Test, QuantizedSubRankAbove4ActivationInt16) {
  const float scales[] = {0.06, 0.04, 0.1};
  const int zero_points[] = {0, 0, 0};
  const int output_dims_count = 6;

  int shape[] = {5, 1, 2, 1, 3, 1};
  const float input1_values[] = {0.6, 1.2, -0.6, 0.3, 2.4, 0.0};
  const float input2_values[] = {0.4, 0.2, 0.4, 0.8, -1.6, 1.2};
  const float golden_values[] = {0.2, 1.0, -1.0, -0.5, 1.0, -1.0};

  int16_t input1_quantized[output_dims_count];
  int16_t input2_quantized[output_dims_count];
  int16_t golden_quantized[output_dims_count];
  int16_t output[output_dims_count];

  tflite::testing::TestSubQuantized(
      shape, input1_values, input1_quantized, scales[0], zero_points[0], shape,
      input2_values, input2_quantized, scales[1], zero_points[1], shape,
      golden_values, golden_quantized, scales[2], zero_points[2],
      kTfLiteActReluN1To1, output);
}

TF_LITE_MICRO_TESTS_MAIN
