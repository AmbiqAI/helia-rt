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

// Helia-specific coverage for the int16 path of HARD_SWISH. The upstream
// TFLM HARD_SWISH kernel (kernels/hard_swish.cc) only supports float and
// int8; helia's optimized kernel (kernels/helia/hard_swish.cc) adds int16
// support. This test exercises that path and lives outside the upstream
// kernels/ directory so it does not introduce drift in the upstream test
// file. It is wired into the build via ext_libs/helia_tests.inc, which is
// only included when OPTIMIZED_KERNEL_DIR=helia.

#include <algorithm>
#include <limits>
#include <random>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

void GenerateUniformRandomVector(int size, float min, float max,
                                 std::minstd_rand* random_engine,
                                 float* result) {
  for (int i = 0; i < size; i++) {
    float random_value_scaled_0_1 =
        (*random_engine)() *
        (1.0f / static_cast<float>(std::minstd_rand::modulus));
    result[i] = min + (max - min) * random_value_scaled_0_1;
  }
}

void EvalTestReferenceHardSwish(int size, float* input, float* result) {
  for (int i = 0; i < size; i++) {
    const float in = input[i];
    result[i] = in * std::min(6.0f, std::max(0.0f, in + 3)) * (1.0f / 6.0f);
  }
}

// int16 variant of TestHardSwishQuantized. Uses symmetric quantization
// (zero_point == 0) per the int16 contract, and a tolerance scaled to the
// 16-bit dynamic range.
void TestHardSwishQuantizedInt16(int size, const int16_t* output_data,
                                 int16_t* input_data_quantized,
                                 float* dequantized_output, float input_min,
                                 float input_max, float output_min,
                                 float output_max,
                                 std::minstd_rand* random_engine,
                                 float* float_input_values,
                                 float* float_ref_output_values) {
  int input_dims_data[] = {2, 1, size};
  int output_dims_data[] = {2, 1, size};

  const float input_scale = SymmetricScaleFromMinMax<int16_t>(input_min, input_max);
  const int input_zero_point = 0;
  const float output_scale =
      SymmetricScaleFromMinMax<int16_t>(output_min, output_max);
  const int output_zero_point = 0;

  // Tolerance: ~4 quantization steps over the larger of input/output range,
  // matching helia's empirically-validated bound for 16x8 hard-swish.
  const float kTolerance =
      std::max(input_max - input_min, output_max - output_min) *
      (4.0f / 32767.f);

  // For symmetric int16 quantization the representable output band is the
  // full [-output_max, output_max-eps] window. Clip the float reference to
  // that band so the dequantized fixed-point result has a fair golden.
  const float exp_clip_min =
      (std::numeric_limits<int16_t>::min() - output_zero_point) * output_scale;
  const float exp_clip_max =
      (std::numeric_limits<int16_t>::max() - output_zero_point) * output_scale;

  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_elements_count = ElementCount(*output_dims);

  EXPECT_EQ(output_elements_count, size);

  GenerateUniformRandomVector(size, input_min, input_max, random_engine,
                              float_input_values);
  EvalTestReferenceHardSwish(size, float_input_values, float_ref_output_values);
  for (int i = 0; i < size; i++) {
    float val = float_ref_output_values[i];
    float_ref_output_values[i] = std::min(exp_clip_max, std::max(exp_clip_min, val));
  }

  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(float_input_values, input_data_quantized,
                            input_dims, input_scale, input_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_HARD_SWISH();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  Dequantize<int16_t>(output_data, output_elements_count, output_scale,
                      output_zero_point, dequantized_output);

  for (int i = 0; i < output_elements_count; ++i) {
    EXPECT_NEAR(float_ref_output_values[i], dequantized_output[i], kTolerance);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(HeliaHardSwishTest, SimpleHardSwishTestInt16) {
  std::minstd_rand random_engine;
  constexpr int pairs = 4, one_pair = 2;
  constexpr int size = 101;
  constexpr float minmax_pairs[pairs][one_pair] = {
      {0.f, 1.f}, {-2.f, 1.f}, {-5.f, 10.f}, {-40.f, 60.f}};
  int16_t output_data[size] = {0};
  int16_t input_data_quantized[size] = {0};
  float dequantized_output[size] = {0.f};
  float input_values[size] = {0.f};
  float output_values[size] = {0.f};

  for (int x = 0; x < pairs; x++) {
    for (int y = 0; y < pairs; y++) {
      float input_min = minmax_pairs[x][0];
      float input_max = minmax_pairs[x][1];
      float output_min = minmax_pairs[y][0];
      float output_max = minmax_pairs[y][1];

      tflite::testing::TestHardSwishQuantizedInt16(
          size, output_data, input_data_quantized, dequantized_output,
          input_min, input_max, output_min, output_max, &random_engine,
          input_values, output_values);
    }
  }
}

TF_LITE_MICRO_TESTS_MAIN
