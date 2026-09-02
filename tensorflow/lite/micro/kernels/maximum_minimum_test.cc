/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

#if ARM_NN_ENABLE_F16
#include "arm_nnfunctions_flt.h"
#endif

namespace tflite {
namespace testing {
namespace {

void TestMaxMinFloat(const TFLMRegistration& registration,
                     int* input1_dims_data, const float* input1_data,
                     int* input2_dims_data, const float* input2_data,
                     const float* expected_output_data, int* output_dims_data,
                     float* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    EXPECT_NEAR(expected_output_data[i], output_data[i], 1e-5f);
  }
}

void TestMaxMinQuantized(const TFLMRegistration& registration,
                         int* input1_dims_data, const int8_t* input1_data,
                         float const input1_scale, const int input1_zero_point,
                         int* input2_dims_data, const int8_t* input2_data,
                         const float input2_scale, const int input2_zero_point,
                         const int8_t* expected_output_data,
                         const float output_scale, const int output_zero_point,
                         int* output_dims_data, int8_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_dims, input1_scale,
                            input1_zero_point),
      CreateQuantizedTensor(input2_data, input2_dims, input2_scale,
                            input2_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

void TestMaxMinQuantizedInt16(
    const TFLMRegistration& registration, int* input1_dims_data,
    const int16_t* input1_data, float const input1_scale,
    const int input1_zero_point, int* input2_dims_data,
    const int16_t* input2_data, const float input2_scale,
    const int input2_zero_point, const int16_t* expected_output_data,
    const float output_scale, const int output_zero_point,
    int* output_dims_data, int16_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_dims, input1_scale,
                            input1_zero_point),
      CreateQuantizedTensor(input2_data, input2_dims, input2_scale,
                            input2_zero_point),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point),
  };

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}
void TestMaxMinQuantizedInt32(const TFLMRegistration& registration,
                              int* input1_dims_data, const int32_t* input1_data,
                              int* input2_dims_data, const int32_t* input2_data,
                              const int32_t* expected_output_data,
                              int* output_dims_data, int32_t* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             /*builtin_data=*/nullptr);

  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(MaximumMinimumTest, FloatTest) {
  int dims[] = {3, 3, 1, 2};
  const float data1[] = {1.0, 0.0, -1.0, 11.0, -2.0, -1.44};
  const float data2[] = {-1.0, 0.0, 1.0, 12.0, -3.0, -1.43};
  const float golden_max[] = {1.0, 0.0, 1.0, 12.0, -2.0, -1.43};
  const float golden_min[] = {-1.0, 0.0, -1.0, 11.0, -3.0, -1.44};
  float output_data[6];

  tflite::testing::TestMaxMinFloat(tflite::Register_MAXIMUM(), dims, data1,
                                   dims, data2, golden_max, dims, output_data);

  tflite::testing::TestMaxMinFloat(tflite::Register_MINIMUM(), dims, data1,
                                   dims, data2, golden_min, dims, output_data);
}

// Rank-5 tensors exceed what 4-D optimized kernels can describe; a backend
// whose dims mapping collapses rank >= 5 shapes to a scalar computes only
// one element and leaves the rest stale. The reference loop must serve this.
TEST(MaximumMinimumTest, FloatRank5Test) {
  int dims[] = {5, 1, 1, 1, 2, 3};
  const float data1[] = {1.0, -2.0, 3.0, 4.0, 5.0, -6.0};
  const float data2[] = {2.0, 3.0, -1.0, 5.0, -4.0, 6.0};
  const float golden_max[] = {2.0, 3.0, 3.0, 5.0, 5.0, 6.0};
  const float golden_min[] = {1.0, -2.0, -1.0, 4.0, -4.0, -6.0};
  float output_data[6];

  tflite::testing::TestMaxMinFloat(tflite::Register_MAXIMUM(), dims, data1,
                                   dims, data2, golden_max, dims, output_data);

  tflite::testing::TestMaxMinFloat(tflite::Register_MINIMUM(), dims, data1,
                                   dims, data2, golden_min, dims, output_data);
}

TEST(MaximumMinimumTest, Int8Test) {
  int dims[] = {3, 3, 1, 2};
  const int8_t data1[] = {1, 0, 2, 11, 2, 23};
  const int8_t data2[] = {0, 0, 1, 12, 127, 1};
  const int8_t golden_max[] = {1, 0, 2, 12, 127, 23};
  const int8_t golden_min[] = {0, 0, 1, 11, 2, 1};

  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;

  int8_t output_data[6];

  tflite::testing::TestMaxMinQuantized(
      tflite::Register_MAXIMUM(), dims, data1, input_scale, input_zero_point,
      dims, data2, input_scale, input_zero_point, golden_max, output_scale,
      output_zero_point, dims, output_data);

  tflite::testing::TestMaxMinQuantized(
      tflite::Register_MINIMUM(), dims, data1, input_scale, input_zero_point,
      dims, data2, input_scale, input_zero_point, golden_min, output_scale,
      output_zero_point, dims, output_data);
}

// Rank-5 counterpart of FloatRank5Test: dims mappings limited to 4-D collapse
// rank >= 5 shapes to a scalar and compute only one element, so the int8 path
// must fall back to the reference loop here.
TEST(MaximumMinimumTest, Int8Rank5Test) {
  int dims[] = {5, 1, 1, 1, 2, 3};
  const int8_t data1[] = {1, -2, 3, 4, 5, -6};
  const int8_t data2[] = {2, 3, -1, 5, -4, 6};
  const int8_t golden_max[] = {2, 3, 3, 5, 5, 6};
  const int8_t golden_min[] = {1, -2, -1, 4, -4, -6};

  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;

  int8_t output_data[6];

  tflite::testing::TestMaxMinQuantized(
      tflite::Register_MAXIMUM(), dims, data1, input_scale, input_zero_point,
      dims, data2, input_scale, input_zero_point, golden_max, output_scale,
      output_zero_point, dims, output_data);

  tflite::testing::TestMaxMinQuantized(
      tflite::Register_MINIMUM(), dims, data1, input_scale, input_zero_point,
      dims, data2, input_scale, input_zero_point, golden_min, output_scale,
      output_zero_point, dims, output_data);
}

TEST(MaximumMinimumTest, Int16Test) {
  int dims[] = {3, 3, 1, 2};
  const int16_t data1[] = {-30, 0, 2124, -123, -32768, 26236};
  const int16_t data2[] = {24, 0, 1, -4256, 32767, -577};
  const int16_t golden_max[] = {24, 0, 2124, -123, 32767, 26236};
  const int16_t golden_min[] = {-30, 0, 1, -4256, -32768, -577};

  const float input_scale = 1.0;
  const int input_zero_point = 0;
  const float output_scale = 1.0;
  const int output_zero_point = 0;

  int16_t output_data[6];

  tflite::testing::TestMaxMinQuantizedInt16(
      tflite::Register_MAXIMUM(), dims, data1, input_scale, input_zero_point,
      dims, data2, input_scale, input_zero_point, golden_max, output_scale,
      output_zero_point, dims, output_data);

  tflite::testing::TestMaxMinQuantizedInt16(
      tflite::Register_MINIMUM(), dims, data1, input_scale, input_zero_point,
      dims, data2, input_scale, input_zero_point, golden_min, output_scale,
      output_zero_point, dims, output_data);
}

TEST(MaximumMinimumTest, FloatWithBroadcastTest) {
  int dims[] = {3, 3, 1, 2};
  int dims_scalar[] = {1, 2};
  const float data1[] = {1.0, 0.0, -1.0, -2.0, -1.44, 11.0};
  const float data2[] = {0.5, 2.0};
  const float golden_max[] = {1.0, 2.0, 0.5, 2.0, 0.5, 11.0};
  const float golden_min[] = {0.5, 0.0, -1.0, -2.0, -1.44, 2.0};
  float output_data[6];

  tflite::testing::TestMaxMinFloat(tflite::Register_MAXIMUM(), dims, data1,
                                   dims_scalar, data2, golden_max, dims,
                                   output_data);

  tflite::testing::TestMaxMinFloat(tflite::Register_MINIMUM(), dims, data1,
                                   dims_scalar, data2, golden_min, dims,
                                   output_data);
}

TEST(MaximumMinimumTest, Int32WithBroadcastTest) {
  int dims[] = {3, 3, 1, 2};
  int dims_scalar[] = {1, 1};
  const int32_t data1[] = {1, 0, -1, -2, 3, 11};
  const int32_t data2[] = {2};
  const int32_t golden_max[] = {2, 2, 2, 2, 3, 11};
  const int32_t golden_min[] = {1, 0, -1, -2, 2, 2};
  int32_t output_data[6];

  tflite::testing::TestMaxMinQuantizedInt32(tflite::Register_MAXIMUM(), dims,
                                            data1, dims_scalar, data2,
                                            golden_max, dims, output_data);

  tflite::testing::TestMaxMinQuantizedInt32(tflite::Register_MINIMUM(), dims,
                                            data1, dims_scalar, data2,
                                            golden_min, dims, output_data);
}

#if ARM_NN_ENABLE_F16
namespace tflite {
namespace testing {
TEST(MaximumMinimumTest, Float16MultiElementGolden) {
  int dims[] = {2, 2, 3};
  float16_t data1[] = {1, -2, 3, 4, 5, -6};
  float16_t data2[] = {2, 3, -1, 5, -4, 6};
  const float golden_max[] = {2, 3, 3, 5, 5, 6};
  const float golden_min[] = {1, -2, -1, 4, -4, -6};
  float16_t output_data[6] = {};

  TfLiteTensor max_tensors[] = {
      CreateTensor(data1, IntArrayFromInts(dims), false, kTfLiteFloat16),
      CreateTensor(data2, IntArrayFromInts(dims), false, kTfLiteFloat16),
      CreateTensor(output_data, IntArrayFromInts(dims), false, kTfLiteFloat16),
  };
  int inputs_array_data[] = {2, 0, 1};
  int outputs_array_data[] = {1, 2};
  const TFLMRegistration max_registration = Register_MAXIMUM();
  micro::KernelRunner max_runner(max_registration, max_tensors, 3,
                                 IntArrayFromInts(inputs_array_data),
                                 IntArrayFromInts(outputs_array_data),
                                 nullptr);
  EXPECT_EQ(kTfLiteOk, max_runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, max_runner.Invoke());
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(golden_max[i], static_cast<float>(output_data[i]), 1e-3f);
  }

  TfLiteTensor min_tensors[] = {
      CreateTensor(data1, IntArrayFromInts(dims), false, kTfLiteFloat16),
      CreateTensor(data2, IntArrayFromInts(dims), false, kTfLiteFloat16),
      CreateTensor(output_data, IntArrayFromInts(dims), false, kTfLiteFloat16),
  };
  const TFLMRegistration min_registration = Register_MINIMUM();
  micro::KernelRunner min_runner(min_registration, min_tensors, 3,
                                 IntArrayFromInts(inputs_array_data),
                                 IntArrayFromInts(outputs_array_data),
                                 nullptr);
  EXPECT_EQ(kTfLiteOk, min_runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, min_runner.Invoke());
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(golden_min[i], static_cast<float>(output_data[i]), 1e-3f);
  }
}
}  // namespace testing
}  // namespace tflite
#endif

TF_LITE_MICRO_TESTS_MAIN
