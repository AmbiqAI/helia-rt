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

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

#ifndef HELIA_FC_LINK_WRAP
#define HELIA_FC_LINK_WRAP 0
#endif

namespace {

enum class Scenario {
  kPerChannel,
  kPerTensor,
};

TfLiteStatus (*g_fully_connected_prepare)(TfLiteContext*, TfLiteNode*);

TfLiteStatus PrepareAfterFourScratchReservations(TfLiteContext* context,
                                                 TfLiteNode* node) {
  // The adapter requires its per-tensor quantization scratch index to be >= 4.
  for (int i = 0; i < 4; ++i) {
    int scratch_index;
    TF_LITE_ENSURE_STATUS(
        context->RequestScratchBufferInArena(context, 1, &scratch_index));
  }
  return g_fully_connected_prepare(context, node);
}

#if HELIA_FC_LINK_WRAP
struct LinkState {
  Scenario scenario;
  int size_calls;
  int compute_calls;
  bool context_ok;
};

LinkState g_link_state;

void ResetLinkState(Scenario scenario) {
  g_link_state = {scenario, 0, 0, true};
}

extern "C" {

int32_t __real_arm_convolve_1x1_s8_fast_get_buffer_size(
    const cmsis_nn_dims* input_dims);
int32_t __wrap_arm_convolve_1x1_s8_fast_get_buffer_size(
    const cmsis_nn_dims* input_dims) {
  ++g_link_state.size_calls;
  if (g_link_state.scenario == Scenario::kPerTensor) {
    return 16;
  }
  return __real_arm_convolve_1x1_s8_fast_get_buffer_size(input_dims);
}

arm_cmsis_nn_status __real_arm_convolve_1x1_s8_fast(
    const cmsis_nn_context*, const cmsis_nn_context*,
    const cmsis_nn_conv_params*, const cmsis_nn_per_channel_quant_params*,
    const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*, const int8_t*,
    const cmsis_nn_dims*, const int32_t*, const cmsis_nn_dims*, int8_t*);
arm_cmsis_nn_status __wrap_arm_convolve_1x1_s8_fast(
    const cmsis_nn_context* context,
    const cmsis_nn_context* output_channel_context,
    const cmsis_nn_conv_params* conv_params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int8_t* input,
    const cmsis_nn_dims* filter_dims, const int8_t* filter,
    const cmsis_nn_dims* bias_dims, const int32_t* bias,
    const cmsis_nn_dims* output_dims, int8_t* output) {
  ++g_link_state.compute_calls;
  bool context_ok = context != nullptr && context->size == 0;
  if (g_link_state.scenario == Scenario::kPerChannel) {
    context_ok =
        context_ok && context->buf == nullptr && g_link_state.size_calls == 0;
  } else {
    context_ok =
        context_ok && context->buf != nullptr && g_link_state.size_calls == 1;
  }
  g_link_state.context_ok = g_link_state.context_ok && context_ok;
  if (!context_ok) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  return __real_arm_convolve_1x1_s8_fast(
      context, output_channel_context, conv_params, quant_params, input_dims,
      input, filter_dims, filter, bias_dims, bias, output_dims, output);
}

}  // extern "C"
#endif  // HELIA_FC_LINK_WRAP

struct RunResult {
  TfLiteStatus prepare;
  TfLiteStatus invoke;
  int8_t output[3];
};

RunResult RunFullyConnected(Scenario scenario) {
  int input_dims_data[] = {4, 1, 1, 1, 4};
  int filter_dims_data[] = {2, 3, 4};
  int bias_dims_data[] = {1, 3};
  int output_dims_data[] = {4, 1, 1, 1, 3};
  int8_t input[] = {1, -2, 3, 4};
  int8_t filter[] = {
      1, 2, -1, 1, -2, 0, 1, 3, 1, -1, 1, -1,
  };
  int32_t bias[] = {1, -3, 2};
  int8_t output[] = {0, 0, 0};

  TfLiteTensor filter_tensor;
  TfLiteTensor bias_tensor;
  TfLiteAffineQuantization filter_quantization = {};
  TfLiteAffineQuantization bias_quantization = {};
  float filter_scales_data[] = {3, 1.0f, 1.0f, 1.0f};
  int filter_zero_points_data[] = {3, 0, 0, 0};
  float bias_scales_data[] = {3, 1.0f, 1.0f, 1.0f};
  int bias_zero_points_data[] = {3, 0, 0, 0};

  if (scenario == Scenario::kPerChannel) {
    TfLiteFloatArray* filter_scales =
        tflite::testing::FloatArrayFromFloats(filter_scales_data);
    filter_tensor = tflite::testing::CreatePerChannelQuantizedTensor(
        filter, tflite::testing::IntArrayFromInts(filter_dims_data),
        filter_scales,
        tflite::testing::IntArrayFromInts(filter_zero_points_data),
        &filter_quantization, 0);
    bias_tensor = tflite::testing::CreatePerChannelQuantizedBiasTensor(
        bias, tflite::testing::IntArrayFromInts(bias_dims_data), 1.0f,
        filter_scales, tflite::testing::FloatArrayFromFloats(bias_scales_data),
        tflite::testing::IntArrayFromInts(bias_zero_points_data),
        &bias_quantization, 0);
  } else {
    filter_tensor = tflite::testing::CreateQuantizedTensor(
        filter, tflite::testing::IntArrayFromInts(filter_dims_data), 1.0f);
    bias_tensor = tflite::testing::CreateQuantizedTensor(
        bias, tflite::testing::IntArrayFromInts(bias_dims_data), 1.0f);
  }
  filter_tensor.allocation_type = kTfLiteMmapRo;
  bias_tensor.allocation_type = kTfLiteMmapRo;

  TfLiteTensor tensors[] = {
      tflite::testing::CreateQuantizedTensor(
          input, tflite::testing::IntArrayFromInts(input_dims_data), 1.0f),
      filter_tensor,
      bias_tensor,
      tflite::testing::CreateQuantizedTensor(
          output, tflite::testing::IntArrayFromInts(output_dims_data), 1.0f),
  };
  int inputs_data[] = {3, 0, 1, 2};
  int outputs_data[] = {1, 3};
  TfLiteFullyConnectedParams params = {};
  params.activation = kTfLiteActNone;
  params.weights_format = kTfLiteFullyConnectedWeightsFormatDefault;

#if HELIA_FC_LINK_WRAP
  ResetLinkState(scenario);
#endif
  TFLMRegistration registration = tflite::Register_FULLY_CONNECTED();
  if (scenario == Scenario::kPerTensor) {
    g_fully_connected_prepare = registration.prepare;
    registration.prepare = PrepareAfterFourScratchReservations;
  }
  tflite::micro::KernelRunner runner(
      registration, tensors, 4, tflite::testing::IntArrayFromInts(inputs_data),
      tflite::testing::IntArrayFromInts(outputs_data), &params);
  const TfLiteStatus prepare = runner.InitAndPrepare();
  const TfLiteStatus invoke =
      prepare == kTfLiteOk ? runner.Invoke() : kTfLiteError;
  return {prepare, invoke, {output[0], output[1], output[2]}};
}

void ExpectValid(Scenario scenario) {
  const RunResult result = RunFullyConnected(scenario);
  EXPECT_EQ(kTfLiteOk, result.prepare);
  EXPECT_EQ(kTfLiteOk, result.invoke);
  const int8_t expected[] = {-1, 10, 4};
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(expected[i], result.output[i]);
  }
#if HELIA_FC_LINK_WRAP
  EXPECT_EQ(1, g_link_state.compute_calls);
  EXPECT_TRUE(g_link_state.context_ok);
  EXPECT_EQ(scenario == Scenario::kPerTensor ? 1 : 0, g_link_state.size_calls);
#endif
}

}  // namespace

TEST(HeliaFullyConnectedContextTest, PerChannelUsesNullActivationBuffer) {
  ExpectValid(Scenario::kPerChannel);
}

TEST(HeliaFullyConnectedContextTest, PerTensorForwardsActivationBuffer) {
  ExpectValid(Scenario::kPerTensor);
}

TF_LITE_MICRO_TESTS_MAIN
