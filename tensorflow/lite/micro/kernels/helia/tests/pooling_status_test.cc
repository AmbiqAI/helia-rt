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

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

constexpr TfLitePoolParams kPoolParams = {kTfLitePaddingValid, 1, 1, 1, 1,
                                          kTfLiteActNone,      {}};

template <typename T>
void ExpectBothBatches(const TFLMRegistration& registration) {
  int input_dims_data[] = {4, 2, 1, 1, 1};
  int output_dims_data[] = {4, 2, 1, 1, 1};
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  const T input_data[] = {static_cast<T>(5), static_cast<T>(9)};
  T output_data[] = {static_cast<T>(-77), static_cast<T>(-77)};
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input_data, input_dims, /*scale=*/1.0f,
                            /*zero_point=*/0),
      CreateQuantizedTensor(output_data, output_dims, /*scale=*/1.0f,
                            /*zero_point=*/0),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, /*tensors_size=*/2,
                             inputs_array, outputs_array, &kPoolParams);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  EXPECT_EQ(input_data[0], output_data[0]);
  EXPECT_EQ(input_data[1], output_data[1]);
}

template <typename T>
void ExpectZeroBatchRejected(const TFLMRegistration& registration) {
  int input_dims_data[] = {4, 1, 1, 1, 1};
  int output_dims_data[] = {4, 1, 1, 1, 1};
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  const T input_data[] = {static_cast<T>(5)};
  T output_data[] = {static_cast<T>(-77)};
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input_data, input_dims, /*scale=*/1.0f,
                            /*zero_point=*/0),
      CreateQuantizedTensor(output_data, output_dims, /*scale=*/1.0f,
                            /*zero_point=*/0),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, /*tensors_size=*/2,
                             inputs_array, outputs_array, &kPoolParams);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());

  input_dims->data[0] = 0;
  output_dims->data[0] = 0;
  EXPECT_EQ(kTfLiteError, runner.Invoke());
  EXPECT_EQ(static_cast<T>(-77), output_data[0]);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(HeliaPoolingStatusTest, AverageGenericInt8ProcessesBothBatches) {
  tflite::testing::ExpectBothBatches<int8_t>(
      tflite::Register_AVERAGE_POOL_2D());
}

TEST(HeliaPoolingStatusTest, AverageInt16ProcessesBothBatches) {
  tflite::testing::ExpectBothBatches<int16_t>(
      tflite::Register_AVERAGE_POOL_2D_INT16());
}

TEST(HeliaPoolingStatusTest, MaxGenericInt8ProcessesBothBatches) {
  tflite::testing::ExpectBothBatches<int8_t>(tflite::Register_MAX_POOL_2D());
}

TEST(HeliaPoolingStatusTest, MaxInt16ProcessesBothBatches) {
  tflite::testing::ExpectBothBatches<int16_t>(
      tflite::Register_MAX_POOL_2D_INT16());
}

TEST(HeliaPoolingStatusTest, AverageGenericRejectsZeroBatch) {
  tflite::testing::ExpectZeroBatchRejected<int8_t>(
      tflite::Register_AVERAGE_POOL_2D());
}

TEST(HeliaPoolingStatusTest, AverageInt8RejectsZeroBatch) {
  tflite::testing::ExpectZeroBatchRejected<int8_t>(
      tflite::Register_AVERAGE_POOL_2D_INT8());
}

TEST(HeliaPoolingStatusTest, AverageInt16RejectsZeroBatch) {
  tflite::testing::ExpectZeroBatchRejected<int16_t>(
      tflite::Register_AVERAGE_POOL_2D_INT16());
}

TEST(HeliaPoolingStatusTest, MaxGenericRejectsZeroBatch) {
  tflite::testing::ExpectZeroBatchRejected<int8_t>(
      tflite::Register_MAX_POOL_2D());
}

TEST(HeliaPoolingStatusTest, MaxInt8RejectsZeroBatch) {
  tflite::testing::ExpectZeroBatchRejected<int8_t>(
      tflite::Register_MAX_POOL_2D_INT8());
}

TEST(HeliaPoolingStatusTest, MaxInt16RejectsZeroBatch) {
  tflite::testing::ExpectZeroBatchRejected<int16_t>(
      tflite::Register_MAX_POOL_2D_INT16());
}

TF_LITE_MICRO_TESTS_MAIN
