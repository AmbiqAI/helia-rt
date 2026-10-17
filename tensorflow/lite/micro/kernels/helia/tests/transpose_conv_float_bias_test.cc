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

// Float TRANSPOSE_CONV Prepare must reject a bias whose type differs from the
// input's: the float kernels read output-channel many elements of the input
// type from it. see AmbiqAI/helia-rt#443

#include <cstdint>

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/transpose_conv.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kInC = 4;
constexpr int kOutC = 8;
constexpr int kInputCount = 4 * 4 * kInC;
constexpr int kFilterCount = kOutC * 3 * 3 * kInC;
constexpr int kOutputCount = 4 * 4 * kOutC;

// Prepares (and, when it succeeds, invokes) a 1x4x4x4 -> 1x4x4x8 SAME,
// stride 1 transpose convolution whose input, filter and output are
// `io_type` and whose bias is `bias_tensor`.
struct RunResult {
  TfLiteStatus prepare;
  TfLiteStatus invoke;
};

// Prepares the layer and, when that succeeds, invokes it.
RunResult RunWithBias(TfLiteType io_type, TfLiteTensor bias_tensor) {
  static uint32_t input[kInputCount];
  static uint32_t filter[kFilterCount];
  static uint32_t output[kOutputCount];
  for (int i = 0; i < kInputCount; ++i) {
    input[i] = 0;
  }
  for (int i = 0; i < kFilterCount; ++i) {
    filter[i] = 0;
  }

  int output_shape_dims[] = {1, 0};
  int32_t* output_shape = nullptr;
  int input_dims[] = {4, 1, 4, 4, kInC};
  int filter_dims[] = {4, kOutC, 3, 3, kInC};
  int output_dims[] = {4, 1, 4, 4, kOutC};
  TfLiteTensor tensors[5] = {
      CreateTensor(output_shape, IntArrayFromInts(output_shape_dims)),
      CreateTensor(filter, IntArrayFromInts(filter_dims), false, io_type),
      CreateTensor(input, IntArrayFromInts(input_dims), false, io_type),
      bias_tensor,
      CreateTensor(output, IntArrayFromInts(output_dims), false, io_type),
  };

  TfLiteTransposeConvParams params = {};
  params.padding = kTfLitePaddingSame;
  params.stride_width = 1;
  params.stride_height = 1;
  params.activation = kTfLiteActNone;

  int inputs[] = {4, 0, 1, 2, 3};
  int outputs[] = {1, 4};
  const TFLMRegistration registration = Register_TRANSPOSE_CONV();
  micro::KernelRunner runner(registration, tensors, 5, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  const TfLiteStatus prepare =
      runner.InitAndPrepare(reinterpret_cast<const char*>(&params));
  return {prepare, prepare == kTfLiteOk ? runner.Invoke() : kTfLiteError};
}

int g_bias_dims[] = {1, kOutC};

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::CreateTensor;
using tflite::testing::g_bias_dims;
using tflite::testing::IntArrayFromInts;
using tflite::testing::RunWithBias;

// The control: a float32 bias prepares and runs.
TEST(HeliaTransposeConvFloatBiasTest, Float32BiasPrepares) {
  static float bias[8] = {};
  const auto result = RunWithBias(
      kTfLiteFloat32, CreateTensor(bias, IntArrayFromInts(g_bias_dims)));
  EXPECT_EQ(kTfLiteOk, result.prepare);
  EXPECT_EQ(kTfLiteOk, result.invoke);
}

// 16 bytes of int16 bias, where the float32 kernel reads 32.
TEST(HeliaTransposeConvFloatBiasTest, Float32WithInt16BiasFailsPrepare) {
  static int16_t bias[8] = {};
  EXPECT_EQ(kTfLiteError,
            RunWithBias(kTfLiteFloat32,
                        CreateTensor(bias, IntArrayFromInts(g_bias_dims)))
                .prepare);
}

// Float16 builds reject the float32 bias; builds without the fp16 API reject
// float16 itself.
TEST(HeliaTransposeConvFloatBiasTest, Float16WithFloat32BiasFailsPrepare) {
  static float bias[8] = {};
  EXPECT_EQ(kTfLiteError,
            RunWithBias(kTfLiteFloat16,
                        CreateTensor(bias, IntArrayFromInts(g_bias_dims)))
                .prepare);
}

// A bias of shape {-2, 2147483644} has -4294967288 elements, which an int
// truncates to the 8 output channels; the count must be compared whole.
TEST(HeliaTransposeConvFloatBiasTest, WrappingBiasCountFailsPrepare) {
  static float bias[8] = {};
  TfLiteTensor bias_tensor = CreateTensor(bias, IntArrayFromInts(g_bias_dims));
  static int wrapping_dims[] = {2, -2, 2147483644};
  bias_tensor.dims = IntArrayFromInts(wrapping_dims);
  EXPECT_EQ(kTfLiteError, RunWithBias(kTfLiteFloat32, bias_tensor).prepare);
}

TF_LITE_MICRO_TESTS_MAIN
