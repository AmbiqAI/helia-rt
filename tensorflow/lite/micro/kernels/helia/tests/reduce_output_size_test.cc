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

// Prepare never compares a reduction's output tensor with the shape the
// reduction writes, and both heliaCORE and the reference reducers write by the
// input shape. Every REDUCE op and route must therefore fail at Invoke when the
// output does not hold exactly the reduced elements, without writing past it:
// MEAN, SUM, REDUCE_MIN/MAX and REDUCE_ALL, heliaCORE and reference routes,
// rank 0 through rank 6. A matching output still succeeds.

#include <cstdint>
#include <cstring>
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

constexpr int kCapacity = 8;
constexpr uint8_t kCanary = 0x5A;

// Prepares and invokes `registration` on `input` (dims `input_dims`, axes
// `axis`) with an output declared as `output_dims`, and returns the invoke
// status. Prepare must succeed. `written` is the number of leading output
// elements the call may write; every byte past them must keep the canary.
template <typename T>
TfLiteStatus RunReduce(const TFLMRegistration& registration, TfLiteType type,
                       int* input_dims, const int32_t* axis, int axis_count,
                       int* output_dims, int written) {
  static T input[kCapacity * 4];
  static T output[kCapacity];
  for (int i = 0; i < kCapacity * 4; ++i) {
    input[i] = static_cast<T>(i % 5 + 1);
  }
  int axis_dims[] = {1, axis_count};
  TfLiteTensor tensors[3];
  if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>) {
    tensors[0] = CreateQuantizedTensor(input, IntArrayFromInts(input_dims),
                                       0.5f, 0);
    tensors[2] = CreateQuantizedTensor(output, IntArrayFromInts(output_dims),
                                       0.5f, 0);
  } else {
    tensors[0] =
        CreateTensor(input, IntArrayFromInts(input_dims), false, type);
    tensors[2] =
        CreateTensor(output, IntArrayFromInts(output_dims), false, type);
  }
  tensors[1] = CreateTensor(axis, IntArrayFromInts(axis_dims));
  tensors[1].allocation_type = kTfLiteMmapRo;
  TfLiteReducerParams params = {};
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  memset(output, kCanary, sizeof(output));
  const TfLiteStatus status = runner.Invoke();
  for (size_t i = written * sizeof(T); i < sizeof(output); ++i) {
    EXPECT_EQ(kCanary, reinterpret_cast<const uint8_t*>(output)[i]);
  }
  return status;
}

// [2, 3] reduced over axis 1 holds two elements; an output declared [1] or
// [3] must fail without a write, and [2] must succeed.
template <typename T>
void ExpectRank2SizeRule(const TFLMRegistration& registration,
                         TfLiteType type) {
  const int32_t axis[] = {1};
  int input_dims[] = {2, 2, 3};
  int short_dims[] = {1, 1};
  int long_dims[] = {1, 3};
  int exact_dims[] = {1, 2};
  EXPECT_EQ(kTfLiteError, RunReduce<T>(registration, type, input_dims, axis,
                                       1, short_dims, 0));
  EXPECT_EQ(kTfLiteError, RunReduce<T>(registration, type, input_dims, axis,
                                       1, long_dims, 0));
  EXPECT_EQ(kTfLiteOk, RunReduce<T>(registration, type, input_dims, axis, 1,
                                    exact_dims, 2));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ExpectRank2SizeRule;
using tflite::testing::RunReduce;

// MEAN int8 runs heliaCORE: a one-element output for a two-element
// reduction used to take a write past its end.
TEST(HeliaReduceOutputSizeTest, MeanInt8) {
  ExpectRank2SizeRule<int8_t>(tflite::Register_MEAN(), kTfLiteInt8);
}

TEST(HeliaReduceOutputSizeTest, MeanInt16AndFloat32) {
  ExpectRank2SizeRule<int16_t>(tflite::Register_MEAN(), kTfLiteInt16);
  ExpectRank2SizeRule<float>(tflite::Register_MEAN(), kTfLiteFloat32);
}

TEST(HeliaReduceOutputSizeTest, SumInt8AndFloat32) {
  ExpectRank2SizeRule<int8_t>(tflite::Register_SUM(), kTfLiteInt8);
  ExpectRank2SizeRule<float>(tflite::Register_SUM(), kTfLiteFloat32);
}

// float32 REDUCE_MIN/MAX run heliaCORE's _tflite entries at rank <= 4.
TEST(HeliaReduceOutputSizeTest, ReduceMinMaxFloat32) {
  ExpectRank2SizeRule<float>(tflite::Register_REDUCE_MIN(), kTfLiteFloat32);
  ExpectRank2SizeRule<float>(tflite::Register_REDUCE_MAX(), kTfLiteFloat32);
}

TEST(HeliaReduceOutputSizeTest, ReduceAllBool) {
  ExpectRank2SizeRule<bool>(tflite::Register_REDUCE_ALL(), kTfLiteBool);
}

// Rank 5 takes the reference route for every type.
TEST(HeliaReduceOutputSizeTest, Rank5ReferenceRoute) {
  const int32_t axis[] = {4};
  int input_dims[] = {5, 1, 1, 1, 2, 3};
  int short_dims[] = {1, 1};
  int exact_dims[] = {4, 1, 1, 1, 2};
  EXPECT_EQ(kTfLiteError,
            RunReduce<int8_t>(tflite::Register_REDUCE_MIN(), kTfLiteInt8,
                              input_dims, axis, 1, short_dims, 0));
  EXPECT_EQ(kTfLiteError,
            RunReduce<float>(tflite::Register_MEAN(), kTfLiteFloat32,
                             input_dims, axis, 1, short_dims, 0));
  EXPECT_EQ(kTfLiteOk,
            RunReduce<int8_t>(tflite::Register_REDUCE_MIN(), kTfLiteInt8,
                              input_dims, axis, 1, exact_dims, 2));
}

// REDUCE_MIN/MAX and REDUCE_ALL size their reference scratch by the input
// rank, so rank 6 still runs; MEAN and SUM iterate with five-axis arrays and
// fail above rank 5.
TEST(HeliaReduceOutputSizeTest, Rank6ReferenceRoute) {
  const int32_t axis[] = {5};
  int input_dims[] = {6, 1, 1, 1, 1, 2, 3};
  int short_dims[] = {1, 1};
  int exact_dims[] = {5, 1, 1, 1, 1, 2};
  EXPECT_EQ(kTfLiteOk,
            RunReduce<int8_t>(tflite::Register_REDUCE_MIN(), kTfLiteInt8,
                              input_dims, axis, 1, exact_dims, 2));
  EXPECT_EQ(kTfLiteOk,
            RunReduce<float>(tflite::Register_REDUCE_MAX(), kTfLiteFloat32,
                             input_dims, axis, 1, exact_dims, 2));
  EXPECT_EQ(kTfLiteOk,
            RunReduce<bool>(tflite::Register_REDUCE_ALL(), kTfLiteBool,
                            input_dims, axis, 1, exact_dims, 2));
  EXPECT_EQ(kTfLiteError,
            RunReduce<float>(tflite::Register_REDUCE_MAX(), kTfLiteFloat32,
                             input_dims, axis, 1, short_dims, 0));
  EXPECT_EQ(kTfLiteError,
            RunReduce<float>(tflite::Register_SUM(), kTfLiteFloat32, input_dims,
                             axis, 1, exact_dims, 0));
}

// The reference ignores the axes of a scalar input, so axis 0 or -1 reduces
// nothing and writes the one output element.
TEST(HeliaReduceOutputSizeTest, ScalarInputIgnoresAxes) {
  int scalar_dims[] = {0};
  int pair_dims[] = {1, 2};
  for (const int32_t a : {0, -1}) {
    const int32_t axis[] = {a};
    EXPECT_EQ(kTfLiteOk,
              RunReduce<float>(tflite::Register_REDUCE_MIN(), kTfLiteFloat32,
                               scalar_dims, axis, 1, scalar_dims, 1));
    EXPECT_EQ(kTfLiteOk,
              RunReduce<int8_t>(tflite::Register_REDUCE_MAX(), kTfLiteInt8,
                                scalar_dims, axis, 1, scalar_dims, 1));
    EXPECT_EQ(kTfLiteOk,
              RunReduce<float>(tflite::Register_SUM(), kTfLiteFloat32,
                               scalar_dims, axis, 1, scalar_dims, 1));
    EXPECT_EQ(kTfLiteOk,
              RunReduce<bool>(tflite::Register_REDUCE_ALL(), kTfLiteBool,
                              scalar_dims, axis, 1, scalar_dims, 1));
    EXPECT_EQ(kTfLiteError,
              RunReduce<float>(tflite::Register_REDUCE_MIN(), kTfLiteFloat32,
                               scalar_dims, axis, 1, pair_dims, 0));
  }
}

// float16 MEAN/SUM/REDUCE_MIN/MAX run heliaCORE on fp16 builds; elsewhere
// Prepare rejects float16, which RunReduce asserts against, so only fp16
// builds run them.
TEST(HeliaReduceOutputSizeTest, Float16) {
#if ARM_NN_ENABLE_F16
  ExpectRank2SizeRule<uint16_t>(tflite::Register_MEAN(), kTfLiteFloat16);
  ExpectRank2SizeRule<uint16_t>(tflite::Register_SUM(), kTfLiteFloat16);
  ExpectRank2SizeRule<uint16_t>(tflite::Register_REDUCE_MAX(),
                                kTfLiteFloat16);
#endif
}

// A reduced axis of extent 0 still fills the kept elements with the start
// value: the size rule counts kept axes only.
TEST(HeliaReduceOutputSizeTest, EmptyReducedAxisStillSucceeds) {
  const int32_t axis[] = {1};
  int input_dims[] = {3, 2, 0, 3};
  int output_dims[] = {2, 2, 3};
  EXPECT_EQ(kTfLiteOk,
            RunReduce<float>(tflite::Register_REDUCE_MAX(), kTfLiteFloat32,
                             input_dims, axis, 1, output_dims, 6));
}

TF_LITE_MICRO_TESTS_MAIN
