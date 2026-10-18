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

// MEAN, SUM, REDUCE_MIN/MAX and REDUCE_ALL Prepare must reject a node with a
// missing input, axis or output, the wrong input count, a non-int32 axis, or
// (REDUCE_ALL) a non-bool input with kTfLiteError, before using the missing
// tensor, and must release every temporary on that return.

#include <cstdint>
#include <type_traits>

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

// Tensor 0 is a [2, 3] input, 1 an int32 axis {1}, 2 the [2] output and 3 a
// float32 axis {1}. `inputs` and `outputs` are TfLiteIntArray-shaped slot
// lists into them; -1 marks a missing tensor.
template <typename T>
TfLiteStatus PrepareReduce(const TFLMRegistration& registration, int* inputs,
                           int* outputs, bool* released) {
  static T input[6] = {};
  static T output[2] = {};
  static const int32_t axis[] = {1};
  static const float float_axis[] = {1.0f};
  int input_dims[] = {2, 2, 3};
  int axis_dims[] = {1, 1};
  int output_dims[] = {1, 2};
  TfLiteTensor tensors[4];
  if constexpr (std::is_same_v<T, bool>) {
    tensors[0] = CreateTensor(input, IntArrayFromInts(input_dims));
    tensors[2] = CreateTensor(output, IntArrayFromInts(output_dims));
  } else {
    tensors[0] =
        CreateQuantizedTensor(input, IntArrayFromInts(input_dims), 0.5f, 0);
    tensors[2] =
        CreateQuantizedTensor(output, IntArrayFromInts(output_dims), 0.5f, 0);
  }
  tensors[1] = CreateTensor(axis, IntArrayFromInts(axis_dims));
  tensors[1].allocation_type = kTfLiteMmapRo;
  tensors[3] = CreateTensor(float_axis, IntArrayFromInts(axis_dims));
  tensors[3].allocation_type = kTfLiteMmapRo;
  TfLiteReducerParams params = {};
  micro::KernelRunner runner(registration, tensors, 4, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  const TfLiteStatus status = runner.InitAndPrepare();
  *released = runner.ValidateTempBufferDeallocated();
  return status;
}

template <typename T>
void ExpectRejected(const TFLMRegistration& registration, int* inputs,
                    int* outputs) {
  bool released = false;
  EXPECT_EQ(kTfLiteError,
            PrepareReduce<T>(registration, inputs, outputs, &released));
  EXPECT_TRUE(released);
}

// Every op on `inputs` and `outputs`: int8 for MEAN, SUM and REDUCE_MIN/MAX,
// bool for REDUCE_ALL.
void ExpectAllRejected(int* inputs, int* outputs) {
  ExpectRejected<int8_t>(Register_MEAN(), inputs, outputs);
  ExpectRejected<int8_t>(Register_SUM(), inputs, outputs);
  ExpectRejected<int8_t>(Register_REDUCE_MIN(), inputs, outputs);
  ExpectRejected<int8_t>(Register_REDUCE_MAX(), inputs, outputs);
  ExpectRejected<bool>(Register_REDUCE_ALL(), inputs, outputs);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ExpectAllRejected;
using tflite::testing::ExpectRejected;
using tflite::testing::PrepareReduce;

// The node the rejection cases alter prepares on every op.
TEST(HeliaReducePrepareTest, ValidNodesPrepare) {
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  bool released = false;
  EXPECT_EQ(kTfLiteOk, PrepareReduce<int8_t>(tflite::Register_MEAN(), inputs,
                                             outputs, &released));
  EXPECT_TRUE(released);
  EXPECT_EQ(kTfLiteOk, PrepareReduce<int8_t>(tflite::Register_SUM(), inputs,
                                             outputs, &released));
  EXPECT_TRUE(released);
  EXPECT_EQ(kTfLiteOk, PrepareReduce<int8_t>(tflite::Register_REDUCE_MIN(),
                                             inputs, outputs, &released));
  EXPECT_TRUE(released);
  EXPECT_EQ(kTfLiteOk, PrepareReduce<int8_t>(tflite::Register_REDUCE_MAX(),
                                             inputs, outputs, &released));
  EXPECT_TRUE(released);
  EXPECT_EQ(kTfLiteOk, PrepareReduce<bool>(tflite::Register_REDUCE_ALL(),
                                           inputs, outputs, &released));
  EXPECT_TRUE(released);
}

TEST(HeliaReducePrepareTest, MissingInputRejected) {
  int inputs[] = {2, -1, 1};
  int outputs[] = {1, 2};
  ExpectAllRejected(inputs, outputs);
}

TEST(HeliaReducePrepareTest, MissingAxisRejected) {
  int inputs[] = {2, 0, -1};
  int outputs[] = {1, 2};
  ExpectAllRejected(inputs, outputs);
}

TEST(HeliaReducePrepareTest, MissingOutputRejected) {
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, -1};
  ExpectAllRejected(inputs, outputs);
}

TEST(HeliaReducePrepareTest, WrongInputCountRejected) {
  int inputs[] = {3, 0, 1, 1};
  int outputs[] = {1, 2};
  ExpectAllRejected(inputs, outputs);
}

TEST(HeliaReducePrepareTest, NonInt32AxisRejected) {
  int inputs[] = {2, 0, 3};
  int outputs[] = {1, 2};
  ExpectAllRejected(inputs, outputs);
}

TEST(HeliaReducePrepareTest, ReduceAllNonBoolInputRejected) {
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  ExpectRejected<int8_t>(tflite::Register_REDUCE_ALL(), inputs, outputs);
}

TF_LITE_MICRO_TESTS_MAIN
