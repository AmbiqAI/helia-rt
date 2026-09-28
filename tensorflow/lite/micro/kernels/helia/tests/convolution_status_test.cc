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
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

enum class ComputeRoute {
  kConvS8,
  kConvS4,
  kConvS16,
  kDepthwiseS8,
  kDepthwiseS4,
  kDepthwiseS16,
  kCount,
};

enum class QueryRoute {
  kConvS8,
  kConvS4,
  kConvS16,
  kDepthwiseS8,
  kDepthwiseS4,
  kDepthwiseS16,
  kCount,
};

enum class PrecomputeRoute {
  kConv,
  kDepthwise,
  kCount,
};

constexpr int kSyntheticActivationBytes = 64;
constexpr int kSyntheticWeightSumBytes = sizeof(int32_t);
constexpr int kOutputSentinel = -77;

#if HELIA_CONV_LINK_WRAP

struct LinkState {
  QueryRoute query_override = QueryRoute::kCount;
  int32_t query_override_result = 0;
  bool override_weight_query = false;
  int32_t weight_query_result = 0;
  ComputeRoute compute_error = ComputeRoute::kCount;
  PrecomputeRoute precompute_error = PrecomputeRoute::kCount;
  int query_calls[static_cast<int>(QueryRoute::kCount)] = {};
  int32_t query_results[static_cast<int>(QueryRoute::kCount)] = {};
  int compute_calls[static_cast<int>(ComputeRoute::kCount)] = {};
  int32_t compute_context_sizes[static_cast<int>(ComputeRoute::kCount)] = {};
  bool compute_context_buffers[static_cast<int>(ComputeRoute::kCount)] = {};
  int weight_query_calls = 0;
  int precompute_calls[static_cast<int>(PrecomputeRoute::kCount)] = {};
  int32_t weight_context_sizes[static_cast<int>(PrecomputeRoute::kCount)] = {};
  bool weight_context_buffers[static_cast<int>(PrecomputeRoute::kCount)] = {};
};

LinkState g_link_state;

int Index(QueryRoute route) { return static_cast<int>(route); }
int Index(ComputeRoute route) { return static_cast<int>(route); }
int Index(PrecomputeRoute route) { return static_cast<int>(route); }

void ResetLinkState() { g_link_state = LinkState{}; }

int32_t RecordQuery(QueryRoute route, int32_t actual) {
  ++g_link_state.query_calls[Index(route)];
  int32_t result = actual;
  if (g_link_state.query_override == route) {
    result = g_link_state.query_override_result < 0 || actual == 0
                 ? g_link_state.query_override_result
                 : actual;
  }
  g_link_state.query_results[Index(route)] = result;
  return result;
}

bool RecordCompute(ComputeRoute route, const cmsis_nn_context* context,
                   const cmsis_nn_context* weight_context = nullptr) {
  ++g_link_state.compute_calls[Index(route)];
  g_link_state.compute_context_sizes[Index(route)] =
      context == nullptr ? -1 : context->size;
  g_link_state.compute_context_buffers[Index(route)] =
      context != nullptr && context->buf != nullptr;
  if (weight_context != nullptr) {
    const PrecomputeRoute precompute_route = route == ComputeRoute::kConvS8
                                                 ? PrecomputeRoute::kConv
                                                 : PrecomputeRoute::kDepthwise;
    g_link_state.weight_context_sizes[Index(precompute_route)] =
        weight_context->size;
    g_link_state.weight_context_buffers[Index(precompute_route)] =
        weight_context->buf != nullptr;
  }
  return g_link_state.compute_error == route;
}

extern "C" {

int32_t __real_arm_convolve_wrapper_s8_get_buffer_size(
    const cmsis_nn_conv_params*, const cmsis_nn_dims*, const cmsis_nn_dims*,
    const cmsis_nn_dims*);
int32_t __wrap_arm_convolve_wrapper_s8_get_buffer_size(
    const cmsis_nn_conv_params* params, const cmsis_nn_dims* input_dims,
    const cmsis_nn_dims* filter_dims, const cmsis_nn_dims* output_dims) {
  return RecordQuery(QueryRoute::kConvS8,
                     __real_arm_convolve_wrapper_s8_get_buffer_size(
                         params, input_dims, filter_dims, output_dims));
}

int32_t __real_arm_convolve_wrapper_s4_get_buffer_size(
    const cmsis_nn_conv_params*, const cmsis_nn_dims*, const cmsis_nn_dims*,
    const cmsis_nn_dims*);
int32_t __wrap_arm_convolve_wrapper_s4_get_buffer_size(
    const cmsis_nn_conv_params* params, const cmsis_nn_dims* input_dims,
    const cmsis_nn_dims* filter_dims, const cmsis_nn_dims* output_dims) {
  return RecordQuery(QueryRoute::kConvS4,
                     __real_arm_convolve_wrapper_s4_get_buffer_size(
                         params, input_dims, filter_dims, output_dims));
}

int32_t __real_arm_convolve_wrapper_s16_get_buffer_size(
    const cmsis_nn_conv_params*, const cmsis_nn_dims*, const cmsis_nn_dims*,
    const cmsis_nn_dims*);
int32_t __wrap_arm_convolve_wrapper_s16_get_buffer_size(
    const cmsis_nn_conv_params* params, const cmsis_nn_dims* input_dims,
    const cmsis_nn_dims* filter_dims, const cmsis_nn_dims* output_dims) {
  return RecordQuery(QueryRoute::kConvS16,
                     __real_arm_convolve_wrapper_s16_get_buffer_size(
                         params, input_dims, filter_dims, output_dims));
}

int32_t __real_arm_depthwise_conv_wrapper_s8_get_buffer_size(
    const cmsis_nn_dw_conv_params*, const cmsis_nn_dims*, const cmsis_nn_dims*,
    const cmsis_nn_dims*);
int32_t __wrap_arm_depthwise_conv_wrapper_s8_get_buffer_size(
    const cmsis_nn_dw_conv_params* params, const cmsis_nn_dims* input_dims,
    const cmsis_nn_dims* filter_dims, const cmsis_nn_dims* output_dims) {
  return RecordQuery(QueryRoute::kDepthwiseS8,
                     __real_arm_depthwise_conv_wrapper_s8_get_buffer_size(
                         params, input_dims, filter_dims, output_dims));
}

int32_t __real_arm_depthwise_conv_wrapper_s4_get_buffer_size(
    const cmsis_nn_dw_conv_params*, const cmsis_nn_dims*, const cmsis_nn_dims*,
    const cmsis_nn_dims*);
int32_t __wrap_arm_depthwise_conv_wrapper_s4_get_buffer_size(
    const cmsis_nn_dw_conv_params* params, const cmsis_nn_dims* input_dims,
    const cmsis_nn_dims* filter_dims, const cmsis_nn_dims* output_dims) {
  return RecordQuery(QueryRoute::kDepthwiseS4,
                     __real_arm_depthwise_conv_wrapper_s4_get_buffer_size(
                         params, input_dims, filter_dims, output_dims));
}

int32_t __real_arm_depthwise_conv_wrapper_s16_get_buffer_size(
    const cmsis_nn_dw_conv_params*, const cmsis_nn_dims*, const cmsis_nn_dims*,
    const cmsis_nn_dims*);
int32_t __wrap_arm_depthwise_conv_wrapper_s16_get_buffer_size(
    const cmsis_nn_dw_conv_params* params, const cmsis_nn_dims* input_dims,
    const cmsis_nn_dims* filter_dims, const cmsis_nn_dims* output_dims) {
  return RecordQuery(QueryRoute::kDepthwiseS16,
                     __real_arm_depthwise_conv_wrapper_s16_get_buffer_size(
                         params, input_dims, filter_dims, output_dims));
}

int32_t __real_arm_convolve_s8_get_weights_sum_size(const cmsis_nn_dims*);
int32_t __wrap_arm_convolve_s8_get_weights_sum_size(
    const cmsis_nn_dims* output_dims) {
  ++g_link_state.weight_query_calls;
  const int32_t actual =
      __real_arm_convolve_s8_get_weights_sum_size(output_dims);
  int32_t result = actual;
  if (g_link_state.override_weight_query) {
    result = g_link_state.weight_query_result < 0 || actual == 0
                 ? g_link_state.weight_query_result
                 : actual;
  }
  g_link_state.weight_query_result = result;
  return result;
}

arm_cmsis_nn_status __real_arm_convolve_weight_sum(int32_t*, const int8_t*,
                                                   const cmsis_nn_dims*,
                                                   const cmsis_nn_dims*,
                                                   const cmsis_nn_dims*,
                                                   int32_t, const int32_t*);
arm_cmsis_nn_status __wrap_arm_convolve_weight_sum(
    int32_t* sums, const int8_t* filter, const cmsis_nn_dims* input_dims,
    const cmsis_nn_dims* filter_dims, const cmsis_nn_dims* output_dims,
    int32_t input_offset, const int32_t* bias) {
  ++g_link_state.precompute_calls[Index(PrecomputeRoute::kConv)];
  if (g_link_state.precompute_error == PrecomputeRoute::kConv) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  const arm_cmsis_nn_status status = __real_arm_convolve_weight_sum(
      sums, filter, input_dims, filter_dims, output_dims, input_offset, bias);
  if (status == ARM_CMSIS_NN_NO_IMPL_ERROR) {
    if (sums != nullptr) {
      sums[0] = input_offset * filter[0] + (bias == nullptr ? 0 : bias[0]);
    }
    return ARM_CMSIS_NN_SUCCESS;
  }
  return status;
}

arm_cmsis_nn_status __real_arm_depthwise_convolve_weight_sum(
    int32_t*, int8_t*, const int8_t*, const cmsis_nn_dw_conv_params*,
    const cmsis_nn_dims*, const cmsis_nn_dims*, const cmsis_nn_dims*, int32_t,
    const int32_t*);
arm_cmsis_nn_status __wrap_arm_depthwise_convolve_weight_sum(
    int32_t* sums, int8_t* scratch, const int8_t* filter,
    const cmsis_nn_dw_conv_params* params, const cmsis_nn_dims* input_dims,
    const cmsis_nn_dims* filter_dims, const cmsis_nn_dims* output_dims,
    int32_t input_offset, const int32_t* bias) {
  ++g_link_state.precompute_calls[Index(PrecomputeRoute::kDepthwise)];
  if (g_link_state.precompute_error == PrecomputeRoute::kDepthwise) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  const arm_cmsis_nn_status status = __real_arm_depthwise_convolve_weight_sum(
      sums, scratch, filter, params, input_dims, filter_dims, output_dims,
      input_offset, bias);
  if (status == ARM_CMSIS_NN_NO_IMPL_ERROR) {
    if (sums != nullptr) {
      sums[0] = input_offset * filter[0] + (bias == nullptr ? 0 : bias[0]);
    }
    return ARM_CMSIS_NN_SUCCESS;
  }
  return status;
}

arm_cmsis_nn_status __real_arm_convolve_wrapper_s8(
    const cmsis_nn_context*, const cmsis_nn_context*,
    const cmsis_nn_conv_params*, const cmsis_nn_per_channel_quant_params*,
    const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*, const int8_t*,
    const cmsis_nn_dims*, const int32_t*, const cmsis_nn_dims*, int8_t*);
arm_cmsis_nn_status __wrap_arm_convolve_wrapper_s8(
    const cmsis_nn_context* context, const cmsis_nn_context* weight_context,
    const cmsis_nn_conv_params* params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int8_t* input,
    const cmsis_nn_dims* filter_dims, const int8_t* filter,
    const cmsis_nn_dims* bias_dims, const int32_t* bias,
    const cmsis_nn_dims* output_dims, int8_t* output) {
  if (RecordCompute(ComputeRoute::kConvS8, context, weight_context)) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  return __real_arm_convolve_wrapper_s8(
      context, weight_context, params, quant_params, input_dims, input,
      filter_dims, filter, bias_dims, bias, output_dims, output);
}

arm_cmsis_nn_status __real_arm_convolve_wrapper_s4(
    const cmsis_nn_context*, const cmsis_nn_conv_params*,
    const cmsis_nn_per_channel_quant_params*, const cmsis_nn_dims*,
    const int8_t*, const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*,
    const int32_t*, const cmsis_nn_dims*, int8_t*);
arm_cmsis_nn_status __wrap_arm_convolve_wrapper_s4(
    const cmsis_nn_context* context, const cmsis_nn_conv_params* params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int8_t* input,
    const cmsis_nn_dims* filter_dims, const int8_t* filter,
    const cmsis_nn_dims* bias_dims, const int32_t* bias,
    const cmsis_nn_dims* output_dims, int8_t* output) {
  if (RecordCompute(ComputeRoute::kConvS4, context)) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  return __real_arm_convolve_wrapper_s4(context, params, quant_params,
                                        input_dims, input, filter_dims, filter,
                                        bias_dims, bias, output_dims, output);
}

arm_cmsis_nn_status __real_arm_convolve_wrapper_s16(
    const cmsis_nn_context*, const cmsis_nn_conv_params*,
    const cmsis_nn_per_channel_quant_params*, const cmsis_nn_dims*,
    const int16_t*, const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*,
    const cmsis_nn_bias_data*, const cmsis_nn_dims*, int16_t*);
arm_cmsis_nn_status __wrap_arm_convolve_wrapper_s16(
    const cmsis_nn_context* context, const cmsis_nn_conv_params* params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int16_t* input,
    const cmsis_nn_dims* filter_dims, const int8_t* filter,
    const cmsis_nn_dims* bias_dims, const cmsis_nn_bias_data* bias,
    const cmsis_nn_dims* output_dims, int16_t* output) {
  if (RecordCompute(ComputeRoute::kConvS16, context)) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  return __real_arm_convolve_wrapper_s16(context, params, quant_params,
                                         input_dims, input, filter_dims, filter,
                                         bias_dims, bias, output_dims, output);
}

arm_cmsis_nn_status __real_arm_depthwise_conv_wrapper_s8(
    const cmsis_nn_context*, const cmsis_nn_context*,
    const cmsis_nn_dw_conv_params*, const cmsis_nn_per_channel_quant_params*,
    const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*, const int8_t*,
    const cmsis_nn_dims*, const int32_t*, const cmsis_nn_dims*, int8_t*);
arm_cmsis_nn_status __wrap_arm_depthwise_conv_wrapper_s8(
    const cmsis_nn_context* context, const cmsis_nn_context* weight_context,
    const cmsis_nn_dw_conv_params* params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int8_t* input,
    const cmsis_nn_dims* filter_dims, const int8_t* filter,
    const cmsis_nn_dims* bias_dims, const int32_t* bias,
    const cmsis_nn_dims* output_dims, int8_t* output) {
  if (RecordCompute(ComputeRoute::kDepthwiseS8, context, weight_context)) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  return __real_arm_depthwise_conv_wrapper_s8(
      context, weight_context, params, quant_params, input_dims, input,
      filter_dims, filter, bias_dims, bias, output_dims, output);
}

arm_cmsis_nn_status __real_arm_depthwise_conv_wrapper_s4(
    const cmsis_nn_context*, const cmsis_nn_dw_conv_params*,
    const cmsis_nn_per_channel_quant_params*, const cmsis_nn_dims*,
    const int8_t*, const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*,
    const int32_t*, const cmsis_nn_dims*, int8_t*);
arm_cmsis_nn_status __wrap_arm_depthwise_conv_wrapper_s4(
    const cmsis_nn_context* context, const cmsis_nn_dw_conv_params* params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int8_t* input,
    const cmsis_nn_dims* filter_dims, const int8_t* filter,
    const cmsis_nn_dims* bias_dims, const int32_t* bias,
    const cmsis_nn_dims* output_dims, int8_t* output) {
  if (RecordCompute(ComputeRoute::kDepthwiseS4, context)) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  return __real_arm_depthwise_conv_wrapper_s4(
      context, params, quant_params, input_dims, input, filter_dims, filter,
      bias_dims, bias, output_dims, output);
}

arm_cmsis_nn_status __real_arm_depthwise_conv_wrapper_s16(
    const cmsis_nn_context*, const cmsis_nn_dw_conv_params*,
    const cmsis_nn_per_channel_quant_params*, const cmsis_nn_dims*,
    const int16_t*, const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*,
    const int64_t*, const cmsis_nn_dims*, int16_t*);
arm_cmsis_nn_status __wrap_arm_depthwise_conv_wrapper_s16(
    const cmsis_nn_context* context, const cmsis_nn_dw_conv_params* params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int16_t* input,
    const cmsis_nn_dims* filter_dims, const int8_t* filter,
    const cmsis_nn_dims* bias_dims, const int64_t* bias,
    const cmsis_nn_dims* output_dims, int16_t* output) {
  if (RecordCompute(ComputeRoute::kDepthwiseS16, context)) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  return __real_arm_depthwise_conv_wrapper_s16(
      context, params, quant_params, input_dims, input, filter_dims, filter,
      bias_dims, bias, output_dims, output);
}

}  // extern "C"
#endif  // HELIA_CONV_LINK_WRAP

struct RunResult {
  TfLiteStatus prepare;
  TfLiteStatus invoke;
  int output;
};

RunResult RunInt8(bool depthwise, bool int4,
                  const TFLMRegistration& registration) {
  int input_dims_data[] = {4, 1, 1, 1, 1};
  int filter_dims_data[] = {4, 1, 1, 1, 1};
  int bias_dims_data[] = {1, 1};
  int output_dims_data[] = {4, 1, 1, 1, 1};
  float filter_scales_data[] = {1, 1.0f};
  int filter_zero_points_data[] = {1, 0};
  int8_t input[] = {3};
  int8_t filter[] = {2};
  int32_t bias[] = {1};
  int8_t output[] = {kOutputSentinel};

  TfLiteAffineQuantization filter_quantization = {};
  TfLiteTensor filter_tensor = tflite::testing::CreatePerChannelQuantizedTensor(
      filter, tflite::testing::IntArrayFromInts(filter_dims_data),
      tflite::testing::FloatArrayFromFloats(filter_scales_data),
      tflite::testing::IntArrayFromInts(filter_zero_points_data),
      &filter_quantization,
      depthwise ? tflite::kDepthwiseConvQuantizedDimension
                : tflite::kConvQuantizedDimension,
      false, int4 ? kTfLiteInt4 : kTfLiteInt8);

  TfLiteTensor tensors[] = {
      tflite::testing::CreateQuantizedTensor(
          input, tflite::testing::IntArrayFromInts(input_dims_data), 1.0f),
      filter_tensor,
      tflite::testing::CreateQuantizedTensor(
          bias, tflite::testing::IntArrayFromInts(bias_dims_data), 1.0f),
      tflite::testing::CreateQuantizedTensor(
          output, tflite::testing::IntArrayFromInts(output_dims_data), 1.0f),
  };
  int inputs_data[] = {3, 0, 1, 2};
  int outputs_data[] = {1, 3};

  TfLiteConvParams conv_params = {};
  conv_params.padding = kTfLitePaddingValid;
  conv_params.stride_width = 1;
  conv_params.stride_height = 1;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;

  TfLiteDepthwiseConvParams depthwise_params = {};
  depthwise_params.padding = kTfLitePaddingValid;
  depthwise_params.stride_width = 1;
  depthwise_params.stride_height = 1;
  depthwise_params.depth_multiplier = 1;
  depthwise_params.activation = kTfLiteActNone;
  depthwise_params.dilation_width_factor = 1;
  depthwise_params.dilation_height_factor = 1;

  tflite::micro::KernelRunner runner(
      registration, tensors, 4, tflite::testing::IntArrayFromInts(inputs_data),
      tflite::testing::IntArrayFromInts(outputs_data),
      depthwise ? static_cast<const void*>(&depthwise_params)
                : static_cast<const void*>(&conv_params));
  const TfLiteStatus prepare = runner.InitAndPrepare();
  const TfLiteStatus invoke =
      prepare == kTfLiteOk ? runner.Invoke() : kTfLiteError;
  return {prepare, invoke, output[0]};
}

RunResult RunInt16(bool depthwise, const TFLMRegistration& registration) {
  int input_dims_data[] = {4, 1, 1, 1, 1};
  int filter_dims_data[] = {4, 1, 1, 1, 1};
  int bias_dims_data[] = {1, 1};
  int output_dims_data[] = {4, 1, 1, 1, 1};
  float filter_scales_data[] = {1, 1.0f};
  int filter_zero_points_data[] = {1, 0};
  int16_t input[] = {3};
  int8_t filter[] = {2};
  int64_t bias[] = {1};
  int16_t output[] = {kOutputSentinel};

  TfLiteAffineQuantization filter_quantization = {};
  TfLiteTensor filter_tensor = tflite::testing::CreatePerChannelQuantizedTensor(
      filter, tflite::testing::IntArrayFromInts(filter_dims_data),
      tflite::testing::FloatArrayFromFloats(filter_scales_data),
      tflite::testing::IntArrayFromInts(filter_zero_points_data),
      &filter_quantization,
      depthwise ? tflite::kDepthwiseConvQuantizedDimension
                : tflite::kConvQuantizedDimension);

  TfLiteTensor tensors[] = {
      tflite::testing::CreateQuantizedTensor(
          input, tflite::testing::IntArrayFromInts(input_dims_data), 1.0f),
      filter_tensor,
      tflite::testing::CreateQuantizedTensor(
          bias, tflite::testing::IntArrayFromInts(bias_dims_data), 1.0f),
      tflite::testing::CreateQuantizedTensor(
          output, tflite::testing::IntArrayFromInts(output_dims_data), 1.0f),
  };
  int inputs_data[] = {3, 0, 1, 2};
  int outputs_data[] = {1, 3};

  TfLiteConvParams conv_params = {};
  conv_params.padding = kTfLitePaddingValid;
  conv_params.stride_width = 1;
  conv_params.stride_height = 1;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;

  TfLiteDepthwiseConvParams depthwise_params = {};
  depthwise_params.padding = kTfLitePaddingValid;
  depthwise_params.stride_width = 1;
  depthwise_params.stride_height = 1;
  depthwise_params.depth_multiplier = 1;
  depthwise_params.activation = kTfLiteActNone;
  depthwise_params.dilation_width_factor = 1;
  depthwise_params.dilation_height_factor = 1;

  tflite::micro::KernelRunner runner(
      registration, tensors, 4, tflite::testing::IntArrayFromInts(inputs_data),
      tflite::testing::IntArrayFromInts(outputs_data),
      depthwise ? static_cast<const void*>(&depthwise_params)
                : static_cast<const void*>(&conv_params));
  const TfLiteStatus prepare = runner.InitAndPrepare();
  const TfLiteStatus invoke =
      prepare == kTfLiteOk ? runner.Invoke() : kTfLiteError;
  return {prepare, invoke, output[0]};
}

RunResult Run(ComputeRoute route, const TFLMRegistration& registration) {
  switch (route) {
    case ComputeRoute::kConvS8:
      return RunInt8(false, false, registration);
    case ComputeRoute::kConvS4:
      return RunInt8(false, true, registration);
    case ComputeRoute::kConvS16:
      return RunInt16(false, registration);
    case ComputeRoute::kDepthwiseS8:
      return RunInt8(true, false, registration);
    case ComputeRoute::kDepthwiseS4:
      return RunInt8(true, true, registration);
    case ComputeRoute::kDepthwiseS16:
      return RunInt16(true, registration);
    case ComputeRoute::kCount:
      break;
  }
  return {kTfLiteError, kTfLiteError, kOutputSentinel};
}

QueryRoute QueryFor(ComputeRoute route) {
  switch (route) {
    case ComputeRoute::kConvS8:
      return QueryRoute::kConvS8;
    case ComputeRoute::kConvS4:
      return QueryRoute::kConvS4;
    case ComputeRoute::kConvS16:
      return QueryRoute::kConvS16;
    case ComputeRoute::kDepthwiseS8:
      return QueryRoute::kDepthwiseS8;
    case ComputeRoute::kDepthwiseS4:
      return QueryRoute::kDepthwiseS4;
    case ComputeRoute::kDepthwiseS16:
      return QueryRoute::kDepthwiseS16;
    case ComputeRoute::kCount:
      return QueryRoute::kCount;
  }
  return QueryRoute::kCount;
}

bool IsS8(ComputeRoute route) {
  return route == ComputeRoute::kConvS8 || route == ComputeRoute::kDepthwiseS8;
}

PrecomputeRoute PrecomputeFor(ComputeRoute route) {
  return route == ComputeRoute::kConvS8 ? PrecomputeRoute::kConv
                                        : PrecomputeRoute::kDepthwise;
}

void ExpectValid(const RunResult& result) {
  EXPECT_EQ(kTfLiteOk, result.prepare);
  EXPECT_EQ(kTfLiteOk, result.invoke);
  EXPECT_EQ(7, result.output);
}

void ExpectContexts(ComputeRoute route) {
#if HELIA_CONV_LINK_WRAP
  EXPECT_EQ(1, g_link_state.compute_calls[Index(route)]);
  const QueryRoute query = QueryFor(route);
  if (query == QueryRoute::kCount) {
    EXPECT_EQ(0, g_link_state.compute_context_sizes[Index(route)]);
    EXPECT_FALSE(g_link_state.compute_context_buffers[Index(route)]);
  } else {
    EXPECT_EQ(1, g_link_state.query_calls[Index(query)]);
    const int32_t size = g_link_state.query_results[Index(query)];
    EXPECT_TRUE(size >= 0);
    EXPECT_EQ(size, g_link_state.compute_context_sizes[Index(route)]);
    EXPECT_EQ(size > 0, g_link_state.compute_context_buffers[Index(route)]);
  }
  if (IsS8(route)) {
    const PrecomputeRoute precompute = PrecomputeFor(route);
    EXPECT_EQ(1, g_link_state.weight_query_calls);
    const int32_t size = g_link_state.weight_query_result;
    EXPECT_TRUE(size >= 0);
    EXPECT_EQ(size > 0 ? 1 : 0,
              g_link_state.precompute_calls[Index(precompute)]);
    EXPECT_EQ(size, g_link_state.weight_context_sizes[Index(precompute)]);
    EXPECT_EQ(size > 0, g_link_state.weight_context_buffers[Index(precompute)]);
  }
#else
  (void)route;
#endif
}

void SetPositiveQueries(ComputeRoute route) {
#if HELIA_CONV_LINK_WRAP
  const QueryRoute query = QueryFor(route);
  if (query != QueryRoute::kCount) {
    g_link_state.query_override = query;
    g_link_state.query_override_result = kSyntheticActivationBytes;
  }
  if (IsS8(route)) {
    g_link_state.override_weight_query = true;
    g_link_state.weight_query_result = kSyntheticWeightSumBytes;
  }
#else
  (void)route;
#endif
}

void ExpectComputeContract(ComputeRoute route,
                           const TFLMRegistration& registration) {
#if HELIA_CONV_LINK_WRAP
  ResetLinkState();
  SetPositiveQueries(route);
#endif
  ExpectValid(Run(route, registration));

#if HELIA_CONV_LINK_WRAP
  ExpectContexts(route);

  ResetLinkState();
  SetPositiveQueries(route);
  g_link_state.compute_error = route;
  const RunResult error_result = Run(route, registration);
  EXPECT_EQ(kTfLiteOk, error_result.prepare);
  EXPECT_EQ(kTfLiteError, error_result.invoke);
  EXPECT_EQ(kOutputSentinel, error_result.output);
  EXPECT_EQ(1, g_link_state.compute_calls[Index(route)]);
#endif
}

void ExpectPrecomputeContract(ComputeRoute route,
                              const TFLMRegistration& registration) {
#if HELIA_CONV_LINK_WRAP
  ResetLinkState();
  SetPositiveQueries(route);
  ExpectValid(Run(route, registration));
  ExpectContexts(route);

  ResetLinkState();
  SetPositiveQueries(route);
  const PrecomputeRoute precompute = PrecomputeFor(route);
  g_link_state.precompute_error = precompute;
  const RunResult result = Run(route, registration);
#if defined(CONV_KERNEL_OPTIMIZED_FOR_SPEED)
  EXPECT_EQ(kTfLiteError, result.prepare);
#else
  EXPECT_EQ(kTfLiteOk, result.prepare);
  EXPECT_EQ(kTfLiteError, result.invoke);
#endif
  EXPECT_EQ(kOutputSentinel, result.output);
  EXPECT_EQ(1, g_link_state.precompute_calls[Index(precompute)]);
  EXPECT_EQ(0, g_link_state.compute_calls[Index(route)]);
#else
  ExpectValid(Run(route, registration));
#endif
}

void ExpectActivationQueryError(ComputeRoute route,
                                const TFLMRegistration& registration) {
#if HELIA_CONV_LINK_WRAP
  ResetLinkState();
#endif
  ExpectValid(Run(route, registration));
#if HELIA_CONV_LINK_WRAP
  ExpectContexts(route);
  ResetLinkState();
  g_link_state.query_override = QueryFor(route);
  g_link_state.query_override_result = -1;
  const RunResult result = Run(route, registration);
  EXPECT_EQ(kTfLiteError, result.prepare);
  EXPECT_EQ(kOutputSentinel, result.output);
  EXPECT_EQ(1, g_link_state.query_calls[Index(QueryFor(route))]);
  EXPECT_EQ(0, g_link_state.compute_calls[Index(route)]);
#endif
}

void ExpectWeightQueryError(ComputeRoute route,
                            const TFLMRegistration& registration) {
#if HELIA_CONV_LINK_WRAP
  ResetLinkState();
#endif
  ExpectValid(Run(route, registration));
#if HELIA_CONV_LINK_WRAP
  ExpectContexts(route);
  ResetLinkState();
  g_link_state.override_weight_query = true;
  g_link_state.weight_query_result = -1;
  const RunResult result = Run(route, registration);
  EXPECT_EQ(kTfLiteError, result.prepare);
  EXPECT_EQ(kOutputSentinel, result.output);
  EXPECT_EQ(1, g_link_state.weight_query_calls);
  EXPECT_EQ(0, g_link_state.compute_calls[Index(route)]);
#endif
}

}  // namespace

TEST(HeliaConvolutionStatusTest, ConvS8GenericComputeContract) {
  ExpectComputeContract(ComputeRoute::kConvS8, tflite::Register_CONV_2D());
}

TEST(HeliaConvolutionStatusTest, ConvS8TypedComputeContract) {
  ExpectComputeContract(ComputeRoute::kConvS8, tflite::Register_CONV_2D_INT8());
}

TEST(HeliaConvolutionStatusTest, ConvS4GenericComputeContract) {
  ExpectComputeContract(ComputeRoute::kConvS4, tflite::Register_CONV_2D());
}

TEST(HeliaConvolutionStatusTest, ConvS4TypedComputeContract) {
  ExpectComputeContract(ComputeRoute::kConvS4, tflite::Register_CONV_2D_INT4());
}

TEST(HeliaConvolutionStatusTest, ConvS16GenericComputeContract) {
  ExpectComputeContract(ComputeRoute::kConvS16, tflite::Register_CONV_2D());
}

TEST(HeliaConvolutionStatusTest, ConvS16TypedComputeContract) {
  ExpectComputeContract(ComputeRoute::kConvS16,
                        tflite::Register_CONV_2D_INT16());
}

TEST(HeliaConvolutionStatusTest, DepthwiseS8GenericComputeContract) {
  ExpectComputeContract(ComputeRoute::kDepthwiseS8,
                        tflite::Register_DEPTHWISE_CONV_2D());
}

TEST(HeliaConvolutionStatusTest, DepthwiseS8TypedComputeContract) {
  ExpectComputeContract(ComputeRoute::kDepthwiseS8,
                        tflite::Register_DEPTHWISE_CONV_2D_INT8());
}

TEST(HeliaConvolutionStatusTest, DepthwiseS4GenericComputeContract) {
  ExpectComputeContract(ComputeRoute::kDepthwiseS4,
                        tflite::Register_DEPTHWISE_CONV_2D());
}

TEST(HeliaConvolutionStatusTest, DepthwiseS4TypedComputeContract) {
  ExpectComputeContract(ComputeRoute::kDepthwiseS4,
                        tflite::Register_DEPTHWISE_CONV_2D_INT4());
}

TEST(HeliaConvolutionStatusTest, DepthwiseS16GenericComputeContract) {
  ExpectComputeContract(ComputeRoute::kDepthwiseS16,
                        tflite::Register_DEPTHWISE_CONV_2D());
}

TEST(HeliaConvolutionStatusTest, DepthwiseS16TypedComputeContract) {
  ExpectComputeContract(ComputeRoute::kDepthwiseS16,
                        tflite::Register_DEPTHWISE_CONV_2D_INT16());
}

TEST(HeliaConvolutionStatusTest, ConvS8GenericPrecomputeStatus) {
  ExpectPrecomputeContract(ComputeRoute::kConvS8, tflite::Register_CONV_2D());
}

TEST(HeliaConvolutionStatusTest, ConvS8TypedPrecomputeStatus) {
  ExpectPrecomputeContract(ComputeRoute::kConvS8,
                           tflite::Register_CONV_2D_INT8());
}

TEST(HeliaConvolutionStatusTest, DepthwiseS8GenericPrecomputeStatus) {
  ExpectPrecomputeContract(ComputeRoute::kDepthwiseS8,
                           tflite::Register_DEPTHWISE_CONV_2D());
}

TEST(HeliaConvolutionStatusTest, DepthwiseS8TypedPrecomputeStatus) {
  ExpectPrecomputeContract(ComputeRoute::kDepthwiseS8,
                           tflite::Register_DEPTHWISE_CONV_2D_INT8());
}

TEST(HeliaConvolutionStatusTest, ConvS8RejectsNegativeActivationSize) {
  ExpectActivationQueryError(ComputeRoute::kConvS8, tflite::Register_CONV_2D());
}

TEST(HeliaConvolutionStatusTest, ConvS4RejectsNegativeActivationSize) {
  ExpectActivationQueryError(ComputeRoute::kConvS4, tflite::Register_CONV_2D());
}

TEST(HeliaConvolutionStatusTest, ConvS16RejectsNegativeActivationSize) {
  ExpectActivationQueryError(ComputeRoute::kConvS16,
                             tflite::Register_CONV_2D());
}

TEST(HeliaConvolutionStatusTest, DepthwiseS8RejectsNegativeActivationSize) {
  ExpectActivationQueryError(ComputeRoute::kDepthwiseS8,
                             tflite::Register_DEPTHWISE_CONV_2D());
}

TEST(HeliaConvolutionStatusTest, DepthwiseS4RejectsNegativeActivationSize) {
  ExpectActivationQueryError(ComputeRoute::kDepthwiseS4,
                             tflite::Register_DEPTHWISE_CONV_2D());
}

TEST(HeliaConvolutionStatusTest, DepthwiseS16RejectsNegativeActivationSize) {
  ExpectActivationQueryError(ComputeRoute::kDepthwiseS16,
                             tflite::Register_DEPTHWISE_CONV_2D_INT16());
}

TEST(HeliaConvolutionStatusTest, ConvS8RejectsNegativeWeightSize) {
  ExpectWeightQueryError(ComputeRoute::kConvS8, tflite::Register_CONV_2D());
}

TEST(HeliaConvolutionStatusTest, DepthwiseS8RejectsNegativeWeightSize) {
  ExpectWeightQueryError(ComputeRoute::kDepthwiseS8,
                         tflite::Register_DEPTHWISE_CONV_2D());
}

TF_LITE_MICRO_TESTS_MAIN
