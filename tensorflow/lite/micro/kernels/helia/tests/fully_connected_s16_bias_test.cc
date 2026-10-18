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

// 16x8 FULLY_CONNECTED through the helia kernel, compared bit for bit with the
// reference integer implementation, for int64, int32 and absent biases under
// per-tensor and per-channel filter quantization. The converter emits an
// int32 bias when FullyConnectedOptions.quantized_bias_type asks for one.
// Any other bias type must fail Prepare. see AmbiqAI/helia-rt#470

#include <cstdint>
#include <cstring>

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

enum class Bias { kInt64, kInt32, kNone, kInt16, kFloat32 };

constexpr int kBatches = 3;
constexpr int kAccumDepth = 24;
constexpr int kOutputDepth = 7;
constexpr int kOutputCount = kBatches * kOutputDepth;

uint32_t g_seed = 4129;
int NextInt(int lo, int hi) {
  g_seed = g_seed * 1103515245u + 12345u;
  return lo + static_cast<int>((g_seed >> 16) % (hi - lo + 1));
}

void ExpectMatchesReference(Bias bias_type, bool per_channel,
                            TFLMRegistration registration) {
  int16_t input[kBatches * kAccumDepth];
  int8_t filter[kOutputDepth * kAccumDepth];
  int64_t bias64[kOutputDepth];
  int32_t bias32[kOutputDepth];
  int16_t bias16[kOutputDepth];
  float bias_f[kOutputDepth];
  // One extra element catches a write past the output.
  int16_t output[kOutputCount + 1];
  int16_t expected[kOutputCount];

  for (int16_t& v : input) v = static_cast<int16_t>(NextInt(-4000, 4000));
  for (int8_t& v : filter) v = static_cast<int8_t>(NextInt(-127, 127));
  for (int c = 0; c < kOutputDepth; ++c) {
    bias32[c] = NextInt(-(1 << 20), 1 << 20);
    bias64[c] = bias32[c];
    bias16[c] = static_cast<int16_t>(bias32[c] >> 8);
    bias_f[c] = static_cast<float>(bias32[c]);
  }

  const float input_scale = 1.0f / 1024;
  const float output_scale = 1.0f / 16;
  float filter_scales[kOutputDepth + 1] = {
      static_cast<float>(per_channel ? kOutputDepth : 1)};
  int filter_zero_points[kOutputDepth + 1] = {per_channel ? kOutputDepth : 1};
  int32_t multipliers[kOutputDepth];
  int shifts[kOutputDepth];
  for (int c = 0; c < kOutputDepth; ++c) {
    filter_scales[c + 1] = per_channel ? 0.002f * (1 + (c % 5)) : 0.003f;
    filter_zero_points[c + 1] = 0;
    QuantizeMultiplier(static_cast<double>(input_scale) *
                           static_cast<double>(filter_scales[c + 1]) /
                           static_cast<double>(output_scale),
                       &multipliers[c], &shifts[c]);
  }

  int input_dims[] = {2, kBatches, kAccumDepth};
  int filter_dims[] = {2, kOutputDepth, kAccumDepth};
  int bias_dims[] = {1, kOutputDepth};
  int output_dims[] = {2, kBatches, kOutputDepth};
  TfLiteAffineQuantization filter_quant = {FloatArrayFromFloats(filter_scales),
                                           IntArrayFromInts(filter_zero_points),
                                           0};
  TfLiteTensor bias_tensor;
  switch (bias_type) {
    case Bias::kInt32:
      bias_tensor = CreateTensor(bias32, IntArrayFromInts(bias_dims));
      break;
    case Bias::kInt16:
      bias_tensor = CreateTensor(bias16, IntArrayFromInts(bias_dims));
      break;
    case Bias::kFloat32:
      bias_tensor = CreateTensor(bias_f, IntArrayFromInts(bias_dims));
      break;
    default:
      bias_tensor = CreateTensor(bias64, IntArrayFromInts(bias_dims));
      break;
  }
  TfLiteTensor tensors[4] = {
      CreateQuantizedTensor(input, IntArrayFromInts(input_dims), input_scale,
                            0),
      CreateTensor(filter, IntArrayFromInts(filter_dims)),
      bias_tensor,
      CreateQuantizedTensor(output, IntArrayFromInts(output_dims), output_scale,
                            0),
  };
  tensors[1].quantization = {kTfLiteAffineQuantization, &filter_quant};
  tensors[1].params = {filter_scales[1], 0};
  tensors[2].params = {input_scale * filter_scales[1], 0};

  TfLiteFullyConnectedParams params = {};
  params.activation = kTfLiteActNone;
  params.weights_format = kTfLiteFullyConnectedWeightsFormatDefault;
  params.quantized_bias_type =
      bias_type == Bias::kInt32 ? kTfLiteInt32 : kTfLiteNoType;

  const bool has_bias = bias_type != Bias::kNone;
  int inputs_with_bias[] = {3, 0, 1, 2};
  int inputs_without_bias[] = {3, 0, 1, -1};
  int outputs[] = {1, 3};
  micro::KernelRunner runner(
      registration, tensors, 4,
      IntArrayFromInts(has_bias ? inputs_with_bias : inputs_without_bias),
      IntArrayFromInts(outputs), &params);
  const TfLiteStatus prepare = runner.InitAndPrepare();
  if (bias_type == Bias::kInt16 || bias_type == Bias::kFloat32) {
    EXPECT_EQ(kTfLiteError, prepare);
    // InitAndPrepare() checks this only when Prepare succeeds.
    EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
    return;
  }
  EXPECT_EQ(kTfLiteOk, prepare);
  memset(output, 0x55, sizeof(output));
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  FullyConnectedParams op_params = {};
  op_params.input_offset = 0;
  op_params.weights_offset = 0;
  op_params.output_offset = 0;
  op_params.output_multiplier = multipliers[0];
  op_params.output_shift = shifts[0];
  op_params.quantized_activation_min = -32768;
  op_params.quantized_activation_max = 32767;
  const int32_t in_shape[] = {kBatches, kAccumDepth};
  const int32_t f_shape[] = {kOutputDepth, kAccumDepth};
  const int32_t b_shape[] = {kOutputDepth};
  const int32_t o_shape[] = {kBatches, kOutputDepth};
  const RuntimeShape input_shape(2, in_shape);
  const RuntimeShape filter_shape(2, f_shape);
  const RuntimeShape bias_shape(1, b_shape);
  const RuntimeShape output_shape(2, o_shape);
  if (bias_type == Bias::kInt32) {
    per_channel
        ? micro::reference_integer_ops::FullyConnectedPerChannel(
              op_params, multipliers, shifts, input_shape, input, filter_shape,
              filter, bias_shape, bias32, output_shape, expected)
        : micro::reference_integer_ops::FullyConnected(
              op_params, input_shape, input, filter_shape, filter, bias_shape,
              bias32, output_shape, expected);
  } else {
    const int64_t* bias = has_bias ? bias64 : nullptr;
    per_channel
        ? micro::reference_integer_ops::FullyConnectedPerChannel(
              op_params, multipliers, shifts, input_shape, input, filter_shape,
              filter, bias_shape, bias, output_shape, expected)
        : micro::reference_integer_ops::FullyConnected(
              op_params, input_shape, input, filter_shape, filter, bias_shape,
              bias, output_shape, expected);
  }

  int mismatches = 0;
  for (int i = 0; i < kOutputCount; ++i) {
    mismatches += output[i] != expected[i];
  }
  EXPECT_EQ(0, mismatches);
  EXPECT_EQ(0x5555, static_cast<uint16_t>(output[kOutputCount]));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::Bias;
using tflite::testing::ExpectMatchesReference;

TEST(HeliaFullyConnectedS16BiasTest, Int32BiasPerTensor) {
  ExpectMatchesReference(Bias::kInt32, false,
                         tflite::Register_FULLY_CONNECTED());
}

TEST(HeliaFullyConnectedS16BiasTest, Int32BiasPerChannel) {
  ExpectMatchesReference(Bias::kInt32, true,
                         tflite::Register_FULLY_CONNECTED());
}

TEST(HeliaFullyConnectedS16BiasTest, Int32BiasInt16Registration) {
  ExpectMatchesReference(Bias::kInt32, true,
                         tflite::Register_FULLY_CONNECTED_INT16());
}

TEST(HeliaFullyConnectedS16BiasTest, Int64BiasPerTensor) {
  ExpectMatchesReference(Bias::kInt64, false,
                         tflite::Register_FULLY_CONNECTED());
}

TEST(HeliaFullyConnectedS16BiasTest, Int64BiasPerChannel) {
  ExpectMatchesReference(Bias::kInt64, true,
                         tflite::Register_FULLY_CONNECTED());
}

TEST(HeliaFullyConnectedS16BiasTest, NoBiasPerChannel) {
  ExpectMatchesReference(Bias::kNone, true, tflite::Register_FULLY_CONNECTED());
}

TEST(HeliaFullyConnectedS16BiasTest, Int16BiasFailsPrepare) {
  ExpectMatchesReference(Bias::kInt16, false,
                         tflite::Register_FULLY_CONNECTED());
}

TEST(HeliaFullyConnectedS16BiasTest, Float32BiasFailsPrepare) {
  ExpectMatchesReference(Bias::kFloat32, true,
                         tflite::Register_FULLY_CONNECTED());
}

TF_LITE_MICRO_TESTS_MAIN
