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

// Per-channel DEQUANTIZE gives TensorFlow Lite's per-channel results for int8
// and uint8 on the first, a middle and the last axis, rejects int16 and
// inconsistent per-channel parameters in Prepare, and leaves per-tensor input
// on arm_dequantize_s8_f32. The goldens are exact in float32. GCC and ATfE
// links wrap the heliaCORE entry to prove the route. see AmbiqAI/helia-rt#498

#include <cstdint>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

int g_s8_calls = 0;

}  // namespace

#if HELIA_DEQUANTIZE_PER_CHANNEL_LINK_WRAP
extern "C" {

arm_cmsis_nn_status __real_arm_dequantize_s8_f32(const int8_t*, float*, int32_t,
                                                 int32_t, float);
arm_cmsis_nn_status __wrap_arm_dequantize_s8_f32(const int8_t* input,
                                                 float* output,
                                                 int32_t block_size,
                                                 int32_t zero_point,
                                                 float scale) {
  ++g_s8_calls;
  return __real_arm_dequantize_s8_f32(input, output, block_size, zero_point,
                                      scale);
}

}  // extern "C"
#endif  // HELIA_DEQUANTIZE_PER_CHANNEL_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

constexpr int kMaxChannels = 4;

// FloatArrayFromFloats and IntArrayFromInts overwrite the leading count in
// place, so each array keeps its own storage here.
struct Quantization {
  Quantization(int channels, const float* channel_scales,
               const int* channel_zero_points, int quantized_dimension) {
    scales[0] = static_cast<float>(channels);
    zero_points[0] = channels;
    for (int c = 0; c < channels; ++c) {
      scales[c + 1] = channel_scales[c];
      zero_points[c + 1] = channel_zero_points[c];
    }
    params.scale = FloatArrayFromFloats(scales);
    params.zero_point = IntArrayFromInts(zero_points);
    params.quantized_dimension = quantized_dimension;
  }
  float scales[kMaxChannels + 1];
  int zero_points[kMaxChannels + 1];
  TfLiteAffineQuantization params;
};

// As the allocator fills a tensor: params from channel 0, plus the arrays.
template <typename T>
TfLiteTensor QuantizedInput(T* data, TfLiteIntArray* dims, TfLiteType type,
                            Quantization* quant) {
  TfLiteTensor input = CreateTensor(data, dims, false, type);
  input.params = {quant->params.scale->data[0],
                  quant->params.zero_point->data[0]};
  input.quantization = {kTfLiteAffineQuantization, &quant->params};
  return input;
}

struct Result {
  TfLiteStatus prepare;
  TfLiteStatus invoke;
  int s8_calls;
};

Result Run(TfLiteTensor input, TfLiteIntArray* output_dims, float* output) {
  TfLiteTensor tensors[2] = {input, CreateTensor(output, output_dims)};
  int inputs[] = {1, 0};
  int outputs[] = {1, 1};
  const TFLMRegistration registration = Register_DEQUANTIZE();
  micro::KernelRunner runner(registration, tensors, 2, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), nullptr);
  Result result = {runner.InitAndPrepare(), kTfLiteError, 0};
  if (result.prepare == kTfLiteOk) {
    g_s8_calls = 0;
    result.invoke = runner.Invoke();
    result.s8_calls = g_s8_calls;
  }
  return result;
}

// int8 [2, 3], one scale per row: the shape from AmbiqAI/helia-rt#498.
TfLiteStatus PrepareInt8Axis0(Quantization* quant, int* dims) {
  int8_t input[6] = {-4, 2, 10, 3, -1, 7};
  float output[6];
  return Run(QuantizedInput(input, IntArrayFromInts(dims), kTfLiteInt8, quant),
             IntArrayFromInts(dims), output)
      .prepare;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::IntArrayFromInts;
using tflite::testing::Quantization;
using tflite::testing::QuantizedInput;
using tflite::testing::Result;
using tflite::testing::Run;

TEST(HeliaDequantizePerChannelTest, Int8FirstAxisUsesEachChannelsScale) {
  // Channel 0's parameters for row 1 would give 1.5, -0.5, 3.5.
  int dims[] = {2, 2, 3};
  const float scales[] = {0.5f, 2.0f};
  const int zero_points[] = {0, 1};
  Quantization quant(2, scales, zero_points, 0);
  int8_t input[6] = {-4, 2, 10, 3, -1, 7};
  float output[6] = {};
  const Result result =
      Run(QuantizedInput(input, IntArrayFromInts(dims), kTfLiteInt8, &quant),
          IntArrayFromInts(dims), output);
  EXPECT_EQ(kTfLiteOk, result.prepare);
  EXPECT_EQ(kTfLiteOk, result.invoke);
  const float expected[6] = {-2.0f, 1.0f, 5.0f, 4.0f, -4.0f, 12.0f};
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(expected[i], output[i]);
  }
  EXPECT_EQ(0, result.s8_calls);
}

TEST(HeliaDequantizePerChannelTest, Int8LastAxis) {
  int dims[] = {3, 2, 2, 3};
  const float scales[] = {0.25f, 1.0f, 4.0f};
  const int zero_points[] = {-1, 0, 2};
  Quantization quant(3, scales, zero_points, 2);
  int8_t input[12];
  for (int i = 0; i < 12; ++i) input[i] = static_cast<int8_t>(i - 6);
  float output[12] = {};
  const Result result =
      Run(QuantizedInput(input, IntArrayFromInts(dims), kTfLiteInt8, &quant),
          IntArrayFromInts(dims), output);
  EXPECT_EQ(kTfLiteOk, result.prepare);
  EXPECT_EQ(kTfLiteOk, result.invoke);
  const float expected[12] = {-1.25f, -5.0f, -24.0f, -0.5f, -2.0f, -12.0f,
                              0.25f,  1.0f,  0.0f,   1.0f,  4.0f,  12.0f};
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(expected[i], output[i]);
  }
}

TEST(HeliaDequantizePerChannelTest, UInt8MiddleAxis) {
  int dims[] = {3, 2, 3, 2};
  const float scales[] = {0.5f, 0.125f, 2.0f};
  const int zero_points[] = {128, 100, 10};
  Quantization quant(3, scales, zero_points, 1);
  uint8_t input[12] = {128, 130, 90, 100, 10, 20, 0, 255, 104, 96, 12, 9};
  float output[12] = {};
  const Result result =
      Run(QuantizedInput(input, IntArrayFromInts(dims), kTfLiteUInt8, &quant),
          IntArrayFromInts(dims), output);
  EXPECT_EQ(kTfLiteOk, result.prepare);
  EXPECT_EQ(kTfLiteOk, result.invoke);
  const float expected[12] = {0.0f,   1.0f,  -1.25f, 0.0f,  0.0f, 20.0f,
                              -64.0f, 63.5f, 0.5f,   -0.5f, 4.0f, -2.0f};
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(expected[i], output[i]);
  }
}

TEST(HeliaDequantizePerChannelTest, Int16PerChannelIsRejected) {
  int dims[] = {2, 2, 3};
  const float scales[] = {0.5f, 2.0f};
  const int zero_points[] = {0, 0};
  Quantization quant(2, scales, zero_points, 0);
  int16_t input[6] = {-4, 2, 10, 3, -1, 7};
  float output[6];
  EXPECT_EQ(kTfLiteError, Run(QuantizedInput(input, IntArrayFromInts(dims),
                                             kTfLiteInt16, &quant),
                              IntArrayFromInts(dims), output)
                              .prepare);
}

TEST(HeliaDequantizePerChannelTest, PerTensorInt8StaysOnHeliaCore) {
  int dims[] = {2, 2, 3};
  const float scales[] = {0.5f};
  const int zero_points[] = {3};
  Quantization quant(1, scales, zero_points, 0);
  int8_t input[6] = {-4, 2, 10, 3, -1, 7};
  float output[6] = {};
  const Result result =
      Run(QuantizedInput(input, IntArrayFromInts(dims), kTfLiteInt8, &quant),
          IntArrayFromInts(dims), output);
  EXPECT_EQ(kTfLiteOk, result.prepare);
  EXPECT_EQ(kTfLiteOk, result.invoke);
  const float expected[6] = {-3.5f, -0.5f, 3.5f, 0.0f, -2.0f, 2.0f};
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(expected[i], output[i]);
  }
#if HELIA_DEQUANTIZE_PER_CHANNEL_LINK_WRAP
  EXPECT_EQ(1, result.s8_calls);
#endif
}

TEST(HeliaDequantizePerChannelTest, MismatchedParameterCountsAreRejected) {
  int dims[] = {2, 2, 3};
  // Two scales, one zero point.
  const float scales[] = {0.5f, 2.0f};
  const int zero_points[] = {0, 1};
  Quantization quant(2, scales, zero_points, 0);
  quant.params.zero_point->size = 1;
  EXPECT_EQ(kTfLiteError, tflite::testing::PrepareInt8Axis0(&quant, dims));
  // Three scales for a quantized dimension of 2.
  const float three_scales[] = {0.5f, 2.0f, 1.0f};
  const int three_zero_points[] = {0, 1, 0};
  Quantization three(3, three_scales, three_zero_points, 0);
  EXPECT_EQ(kTfLiteError, tflite::testing::PrepareInt8Axis0(&three, dims));
}

TEST(HeliaDequantizePerChannelTest, QuantizedDimensionOutOfRangeIsRejected) {
  int dims[] = {2, 2, 3};
  const float scales[] = {0.5f, 2.0f};
  const int zero_points[] = {0, 1};
  Quantization past_rank(2, scales, zero_points, 2);
  EXPECT_EQ(kTfLiteError, tflite::testing::PrepareInt8Axis0(&past_rank, dims));
  Quantization negative(2, scales, zero_points, -1);
  EXPECT_EQ(kTfLiteError, tflite::testing::PrepareInt8Axis0(&negative, dims));
}

TEST(HeliaDequantizePerChannelTest, OutputShapeMismatchIsRejected) {
  int input_dims[] = {2, 2, 3};
  int output_dims[] = {2, 2, 2};
  const float scales[] = {0.5f, 2.0f};
  const int zero_points[] = {0, 1};
  Quantization quant(2, scales, zero_points, 0);
  int8_t input[6] = {-4, 2, 10, 3, -1, 7};
  float output[6];
  EXPECT_EQ(kTfLiteError,
            Run(QuantizedInput(input, IntArrayFromInts(input_dims), kTfLiteInt8,
                               &quant),
                IntArrayFromInts(output_dims), output)
                .prepare);
}

namespace tflite {
namespace testing {
namespace {

TfLiteStatus PrepareChannels(int channels) {
  constexpr int kChannels = 4096;
  static float scales[kChannels + 1];
  static int zero_points[kChannels + 1];
  static int8_t input[kChannels];
  static float output[kChannels];
  scales[0] = static_cast<float>(channels);
  zero_points[0] = channels;
  for (int c = 1; c <= channels; ++c) {
    scales[c] = 1.0f;
    zero_points[c] = 0;
  }
  TfLiteAffineQuantization params;
  params.scale = FloatArrayFromFloats(scales);
  params.zero_point = IntArrayFromInts(zero_points);
  params.quantized_dimension = 0;
  int dims[] = {1, channels};
  TfLiteTensor tensor =
      CreateTensor(input, IntArrayFromInts(dims), false, kTfLiteInt8);
  tensor.params = {1.0f, 0};
  tensor.quantization = {kTfLiteAffineQuantization, &params};
  return Run(tensor, IntArrayFromInts(dims), output).prepare;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

// KernelRunner's 10000-byte arena holds neither 4096-entry copy, and holds the
// 1500-entry scales but not the zero points after them.
TEST(HeliaDequantizePerChannelTest, PersistentCopyFailureFailsPrepare) {
  EXPECT_EQ(kTfLiteError, tflite::testing::PrepareChannels(4096));
  EXPECT_EQ(kTfLiteError, tflite::testing::PrepareChannels(1500));
}

TF_LITE_MICRO_TESTS_MAIN
