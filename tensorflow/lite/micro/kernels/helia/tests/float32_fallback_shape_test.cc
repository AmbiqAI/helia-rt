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

// float32 HARD_SWISH and FULLY_CONNECTED take the TFLM reference when
// heliaCORE's float32 API is not compiled in (HELIA_ENABLE_F32=0) and when
// heliaCORE declines. The reference sizes its writes from other tensors than
// heliaCORE does, so a mismatched output must fail Invoke on either route
// without a write past the output.

#include <cstdint>
#include <cstring>

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/reference/hard_swish.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kCapacity = 8;
constexpr uint8_t kCanary = 0x5A;

// Runs HARD_SWISH on an 8-element input with an output declared
// `output_dims`; `written` leading output elements may change.
TfLiteStatus RunHardSwish(int* input_dims, int* output_dims, int written,
                          float* output) {
  static float input[kCapacity];
  for (int i = 0; i < kCapacity; ++i) {
    input[i] = static_cast<float>(i) - 4.5f;
  }
  TfLiteTensor tensors[2] = {
      CreateTensor(input, IntArrayFromInts(input_dims)),
      CreateTensor(output, IntArrayFromInts(output_dims)),
  };
  int inputs[] = {1, 0};
  int outputs[] = {1, 1};
  const TFLMRegistration registration = Register_HARD_SWISH();
  micro::KernelRunner runner(registration, tensors, 2, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), nullptr);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  memset(output, kCanary, kCapacity * sizeof(float));
  const TfLiteStatus status = runner.Invoke();
  for (size_t i = written * sizeof(float); i < kCapacity * sizeof(float); ++i) {
    EXPECT_EQ(kCanary, reinterpret_cast<const uint8_t*>(output)[i]);
  }
  return status;
}

// Runs FULLY_CONNECTED on a [1, 2] input, a filter of `filter_rows` x 2 and
// no bias, into an output declared `output_dims`; `written` leading output
// elements may change.
TfLiteStatus RunFullyConnected(int filter_rows, int* output_dims, int written,
                               float* output) {
  static const float input[2] = {1.0f, 2.0f};
  static const float filter[2 * kCapacity] = {1, 0, 0, 1, 1, 1, 2, 2,
                                              3, 3, 4, 4, 5, 5, 6, 6};
  int input_dims[] = {2, 1, 2};
  int filter_dims[] = {2, filter_rows, 2};
  TfLiteTensor tensors[3] = {
      CreateTensor(input, IntArrayFromInts(input_dims)),
      CreateTensor(filter, IntArrayFromInts(filter_dims)),
      CreateTensor(output, IntArrayFromInts(output_dims)),
  };
  tensors[1].allocation_type = kTfLiteMmapRo;
  TfLiteFullyConnectedParams params = {};
  params.activation = kTfLiteActNone;
  int inputs[] = {3, 0, 1, kTfLiteOptionalTensor};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = Register_FULLY_CONNECTED();
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  memset(output, kCanary, kCapacity * sizeof(float));
  const TfLiteStatus status = runner.Invoke();
  for (size_t i = written * sizeof(float); i < kCapacity * sizeof(float); ++i) {
    EXPECT_EQ(kCanary, reinterpret_cast<const uint8_t*>(output)[i]);
  }
  return status;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::kCapacity;

TEST(HeliaFloat32FallbackShapeTest, HardSwishShortOutputFailsInvoke) {
  float output[kCapacity];
  int input_dims[] = {1, 8};
  int output_dims[] = {1, 2};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunHardSwish(input_dims, output_dims, 0, output));
}

// An output of the input's count in another shape takes the flat count.
TEST(HeliaFloat32FallbackShapeTest, HardSwishReshapedOutputMatchesReference) {
  float output[kCapacity];
  int input_dims[] = {2, 2, 4};
  int output_dims[] = {1, 8};
  EXPECT_EQ(kTfLiteOk,
            tflite::testing::RunHardSwish(input_dims, output_dims, 8, output));
  float input[kCapacity];
  float expected[kCapacity];
  for (int i = 0; i < kCapacity; ++i) {
    input[i] = static_cast<float>(i) - 4.5f;
  }
  const int32_t count = kCapacity;
  const tflite::RuntimeShape shape(1, &count);
  tflite::micro::reference_ops::HardSwish<float>(shape, input, shape, expected);
  for (int i = 0; i < kCapacity; ++i) {
    EXPECT_NEAR(expected[i], output[i], 1e-6f);
  }
}

// Filter rows set the reference's output depth; a filter with more rows
// than the output's last extent would write past the output.
TEST(HeliaFloat32FallbackShapeTest,
     FullyConnectedFilterRowsMustMatchOutputDepth) {
  float output[kCapacity];
  int output_dims[] = {2, 1, 2};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunFullyConnected(4, output_dims, 0, output));
}

TEST(HeliaFloat32FallbackShapeTest, FullyConnectedMatchingShapesRun) {
  float output[kCapacity];
  int output_dims[] = {2, 1, 2};
  EXPECT_EQ(kTfLiteOk,
            tflite::testing::RunFullyConnected(2, output_dims, 2, output));
  // Filter rows {1, 0} and {0, 1} pass the input through.
  EXPECT_EQ(1.0f, output[0]);
  EXPECT_EQ(2.0f, output[1]);
}

TF_LITE_MICRO_TESTS_MAIN
