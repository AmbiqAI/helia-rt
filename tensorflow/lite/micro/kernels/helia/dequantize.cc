/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "Include/arm_nnsupportfunctions.h"
#include "Include/arm_nnfunctions.h"

#include "tensorflow/lite/micro/kernels/internal/reference/dequantize.h"

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/dequantize.h"
#include "tensorflow/lite/micro/kernels/helia/helia_element_count.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/quantize.h"
#include "tensorflow/lite/micro/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace micro {

namespace {

// Persistent per-channel parameters: the input's quantization arrays do not
// outlive Prepare. The input is viewed as [outer_size, num_channels,
// inner_size] around its quantized dimension.
struct PerChannelParams {
  const float* scales;
  const int32_t* zero_points;
  int32_t num_channels;
  int32_t outer_size;
  int32_t inner_size;
};

// DequantizePrepare reads node->user_data as a DequantizeOpData, so `base`
// must stay the first member.
struct HeliaDequantizeOpData {
  DequantizeOpData base;
  // nullptr for a per-tensor input.
  const PerChannelParams* per_channel;
};
static_assert(std::is_standard_layout<HeliaDequantizeOpData>::value,
              "HeliaDequantizeOpData must be standard layout");
static_assert(offsetof(HeliaDequantizeOpData, base) == 0,
              "DequantizePrepare needs base at offset 0");

// Per-channel means more than one scale.
bool IsQuantizedPerChannel(const TfLiteTensor* input) {
  if (input->quantization.type != kTfLiteAffineQuantization ||
      input->quantization.params == nullptr) {
    return false;
  }
  const auto* params =
      static_cast<const TfLiteAffineQuantization*>(input->quantization.params);
  return params->scale != nullptr && params->scale->size > 1;
}

// Accepts a per-channel int8 or uint8 input and rejects every other type.
// see AmbiqAI/helia-rt#498
TfLiteStatus PreparePerChannel(TfLiteContext* context,
                               const TfLiteTensor* input,
                               const TfLiteTensor* output,
                               HeliaDequantizeOpData* data) {
  if (input->type != kTfLiteInt8 && input->type != kTfLiteUInt8) {
    MicroPrintf("DEQUANTIZE: per-channel %s input is not supported.",
                TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }
  const auto* params =
      static_cast<const TfLiteAffineQuantization*>(input->quantization.params);
  const int32_t num_channels = params->scale->size;
  if (params->zero_point == nullptr ||
      params->zero_point->size != num_channels) {
    MicroPrintf("DEQUANTIZE: %d scales need as many zero points.",
                static_cast<int>(num_channels));
    return kTfLiteError;
  }
  TF_LITE_ENSURE(context, input->dims != nullptr);
  const int32_t axis = params->quantized_dimension;
  if (axis < 0 || axis >= input->dims->size ||
      input->dims->data[axis] != num_channels) {
    MicroPrintf("DEQUANTIZE: %d scales do not match quantized dimension %d.",
                static_cast<int>(num_channels), static_cast<int>(axis));
    return kTfLiteError;
  }
  int32_t count;
  TF_LITE_ENSURE_OK(context, HeliaCheckedSameShapeElementCount(
                                 context, input->dims, output->dims, &count));

  // One persistent block: the parameters, the scales, then the zero points.
  static_assert(sizeof(float) == sizeof(int32_t), "one bound covers both");
  static_assert(sizeof(PerChannelParams) % alignof(float) == 0,
                "the scales follow the parameters");
  TF_LITE_ENSURE(
      context, static_cast<size_t>(num_channels) <=
                   (SIZE_MAX - sizeof(PerChannelParams)) / (2 * sizeof(float)));
  const size_t bytes = sizeof(PerChannelParams) +
                       2 * static_cast<size_t>(num_channels) * sizeof(float);
  void* block = context->AllocatePersistentBuffer(context, bytes);
  TF_LITE_ENSURE(context, block != nullptr);
  auto* per_channel = static_cast<PerChannelParams*>(block);
  float* scales = reinterpret_cast<float*>(per_channel + 1);
  int32_t* zero_points = reinterpret_cast<int32_t*>(scales + num_channels);
  for (int32_t c = 0; c < num_channels; ++c) {
    scales[c] = params->scale->data[c];
    zero_points[c] = params->zero_point->data[c];
  }

  // A zero extent elsewhere could make a partial product overflow; an empty
  // tensor needs no loop.
  int32_t outer_size = 0;
  int32_t inner_size = 0;
  if (count > 0) {
    outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= input->dims->data[i];
    inner_size = count / (outer_size * num_channels);
  }

  per_channel->scales = scales;
  per_channel->zero_points = zero_points;
  per_channel->num_channels = num_channels;
  per_channel->outer_size = outer_size;
  per_channel->inner_size = inner_size;
  data->per_channel = per_channel;
  return kTfLiteOk;
}

TfLiteStatus HeliaDequantizePrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->user_data != nullptr);
  auto* data = static_cast<HeliaDequantizeOpData*>(node->user_data);
  data->per_channel = nullptr;
  TF_LITE_ENSURE_OK(context, DequantizePrepare(context, node));

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  if (output == nullptr) {
    micro_context->DeallocateTempTfLiteTensor(input);
    return kTfLiteError;
  }
  TfLiteStatus status = kTfLiteOk;
  if (IsQuantizedPerChannel(input)) {
    status = PreparePerChannel(context, input, output, data);
  }
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return status;
}

// For any 8-bit q, q - zero_point fits int32 when |zero_point| is at most
// this. A channel outside it subtracts in int64, so the difference is exact
// before its one rounding to float.
constexpr int32_t kInt32ExactZeroPoint = INT32_MAX - 255;

template <typename Difference, typename T>
void DequantizeSlice(const T* input, float* output, int32_t size, float scale,
                     int32_t zero_point) {
  for (int32_t i = 0; i < size; ++i) {
    output[i] = scale * static_cast<float>(static_cast<Difference>(input[i]) -
                                           zero_point);
  }
}

// out = scale[c] * float(q - zero_point[c]) for channel c.
template <typename T>
void PerChannelDequantize(const PerChannelParams& params, const T* input,
                          float* output) {
  int32_t offset = 0;
  for (int32_t outer = 0; outer < params.outer_size; ++outer) {
    for (int32_t c = 0; c < params.num_channels; ++c) {
      const float scale = params.scales[c];
      const int32_t zero_point = params.zero_points[c];
      if (zero_point >= -kInt32ExactZeroPoint &&
          zero_point <= kInt32ExactZeroPoint) {
        DequantizeSlice<int32_t>(input + offset, output + offset,
                                 params.inner_size, scale, zero_point);
      } else {
        DequantizeSlice<int64_t>(input + offset, output + offset,
                                 params.inner_size, scale, zero_point);
      }
      offset += params.inner_size;
    }
  }
}

}  // namespace

void* DequantizeInit(TfLiteContext* context, const char* buffer,
                     size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(HeliaDequantizeOpData));
}

TfLiteStatus DequantizeEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto* helia_data =
      static_cast<const HeliaDequantizeOpData*>(node->user_data);
  const DequantizeOpData* data = &helia_data->base;

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  // Output type ensured to be kTfLiteFloat32 at the Prepare stage
  TFLITE_DCHECK(output->type == kTfLiteFloat32);

  if (helia_data->per_channel != nullptr) {
    switch (input->type) {
      case kTfLiteInt8:
        PerChannelDequantize(*helia_data->per_channel,
                             tflite::micro::GetTensorData<int8_t>(input),
                             tflite::micro::GetTensorData<float>(output));
        return kTfLiteOk;
      case kTfLiteUInt8:
        PerChannelDequantize(*helia_data->per_channel,
                             tflite::micro::GetTensorData<uint8_t>(input),
                             tflite::micro::GetTensorData<float>(output));
        return kTfLiteOk;
      default:
        MicroPrintf("DEQUANTIZE: per-channel %s input is not supported.",
                    TfLiteTypeGetName(input->type));
        return kTfLiteError;
    }
  }

  switch (input->type) {
    case kTfLiteInt8: {
      const arm_cmsis_nn_status status = arm_dequantize_s8_f32(
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorData<float>(output),
        ElementCount(*input->dims),
        data->quantization_params.zero_point,
        data->quantization_params.scale
      );
      if (status != ARM_CMSIS_NN_SUCCESS) {
        MicroPrintf("DEQUANTIZE: arm_dequantize_s8_f32 failed (%d).",
                    static_cast<int>(status));
        return kTfLiteError;
      }
      break;
    }
    case kTfLiteInt16: {
      const arm_cmsis_nn_status status = arm_dequantize_s16_f32(
        tflite::micro::GetTensorData<int16_t>(input),
        tflite::micro::GetTensorData<float>(output),
        ElementCount(*input->dims),
        data->quantization_params.zero_point,
        data->quantization_params.scale
      );
      if (status != ARM_CMSIS_NN_SUCCESS) {
        MicroPrintf("DEQUANTIZE: arm_dequantize_s16_f32 failed (%d).",
                    static_cast<int>(status));
        return kTfLiteError;
      }
      break;
    }
    case kTfLiteUInt8:
      reference_ops::Dequantize(data->quantization_params,
                                tflite::micro::GetTensorShape(input),
                                tflite::micro::GetTensorData<uint8_t>(input),
                                tflite::micro::GetTensorShape(output),
                                tflite::micro::GetTensorData<float>(output));
      break;
    // Storage widening, not float16 arithmetic, so it is not gated on
    // kHeliaFloat16Enabled; see AmbiqAI/helia-rt#255.
    case kTfLiteFloat16: {
      static_assert(sizeof(TfLiteFloat16) == sizeof(uint16_t),
                    "TfLiteFloat16 must be the raw binary16 bits");
      const arm_cmsis_nn_status status = arm_dequantize_f16_bits_f32(
          reinterpret_cast<const uint16_t*>(
              tflite::micro::GetTensorData<TfLiteFloat16>(input)),
          tflite::micro::GetTensorData<float>(output),
          ElementCount(*input->dims));
      if (status != ARM_CMSIS_NN_SUCCESS) {
        MicroPrintf("DEQUANTIZE: arm_dequantize_f16_bits_f32 failed (%d).",
                    static_cast<int>(status));
        return kTfLiteError;
      }
      break;
    }
    default:
      MicroPrintf("Input %s, output %s not supported.",
                  TfLiteTypeGetName(input->type),
                  TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

TFLMRegistration Register_DEQUANTIZE() {
  return tflite::micro::RegisterOp(DequantizeInit, HeliaDequantizePrepare,
                                   DequantizeEval);
}

}  // namespace micro
}  // namespace tflite
