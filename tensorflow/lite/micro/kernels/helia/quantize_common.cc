/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "Include/arm_nnsupportfunctions.h"
#include "Include/arm_nnfunctions.h"

#include <limits>
#include <vector>
#include <algorithm>


#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/quantize.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/quantize.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

TfLiteStatus PrepareQuantizeReference(TfLiteContext* context,
                                      TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  auto* data = static_cast<OpDataQuantizeReference*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  // TODO(b/128934713): Add support for fixed-point per-channel quantization.
  // Currently this only support affine per-layer quantization.
  TF_LITE_ENSURE_EQ(context, output->quantization.type,
                    kTfLiteAffineQuantization);
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(output->quantization.params);
  TF_LITE_ENSURE(context, affine_quantization);
  TF_LITE_ENSURE(context, affine_quantization->scale);
  TF_LITE_ENSURE(context, affine_quantization->scale->size == 1);

  TF_LITE_ENSURE(
      context, input->type == kTfLiteFloat32 || input->type == kTfLiteInt32 ||
                   input->type == kTfLiteInt16 || input->type == kTfLiteInt8 ||
                   input->type == kTfLiteUInt8);
  TF_LITE_ENSURE(context, output->type == kTfLiteInt8 ||
                              output->type == kTfLiteInt16 ||
                              output->type == kTfLiteInt32 ||
                              output->type == kTfLiteUInt8);

  if ((input->type == kTfLiteInt16 && output->type == kTfLiteInt8) ||
      (input->type == kTfLiteInt8 && output->type == kTfLiteInt8) ||
      (input->type == kTfLiteInt8 && output->type == kTfLiteUInt8) ||
      (input->type == kTfLiteUInt8 && output->type == kTfLiteInt8) ||
      (input->type == kTfLiteInt8 && output->type == kTfLiteInt16) ||
      (input->type == kTfLiteInt8 && output->type == kTfLiteInt32) ||
      (input->type == kTfLiteInt16 && output->type == kTfLiteInt16) ||
      (input->type == kTfLiteInt16 && output->type == kTfLiteInt32) ||
      (input->type == kTfLiteInt32 && output->type == kTfLiteInt8) ||
      (input->type == kTfLiteInt32 && output->type == kTfLiteInt16)) {
    double effective_scale = static_cast<double>(input->params.scale) /
                             static_cast<double>(output->params.scale);

    QuantizeMultiplier(effective_scale, &data->requantize_output_multiplier,
                       &data->requantize_output_shift);
  }

  // heliaCORE's arm_quantize_f32_s8()/arm_quantize_f32_s16() reject an
  // out-of-range zero_point at run time (ARG_ERROR, and no output written).
  // The exact range they enforce is not part of the public contract of the
  // pinned version, so rather than duplicate their validation this checks only
  // the invariant TFLite itself guarantees: a quantized tensor's zero_point
  // must be representable in that tensor's element type. That is a superset of
  // any narrower kernel rule, so it cannot reject a model the kernel would
  // have accepted, and it surfaces the failure at AllocateTensors() instead of
  // mid-Invoke. The Eval-side status check stays authoritative.
  int32_t zero_point_min = std::numeric_limits<int32_t>::min();
  int32_t zero_point_max = std::numeric_limits<int32_t>::max();
  switch (output->type) {
    case kTfLiteInt8:
      zero_point_min = std::numeric_limits<int8_t>::min();
      zero_point_max = std::numeric_limits<int8_t>::max();
      break;
    case kTfLiteInt16:
      zero_point_min = std::numeric_limits<int16_t>::min();
      zero_point_max = std::numeric_limits<int16_t>::max();
      break;
    case kTfLiteUInt8:
      zero_point_min = std::numeric_limits<uint8_t>::min();
      zero_point_max = std::numeric_limits<uint8_t>::max();
      break;
    default:
      break;
  }
  if (output->params.zero_point < zero_point_min ||
      output->params.zero_point > zero_point_max) {
    MicroPrintf("Quantize: output zero_point %d is out of range for %s.",
                static_cast<int>(output->params.zero_point),
                TfLiteTypeGetName(output->type));
    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(output);
    return kTfLiteError;
  }

  data->quantization_params.zero_point = output->params.zero_point;
  data->quantization_params.scale = static_cast<double>(output->params.scale);

  data->input_zero_point = input->params.zero_point;

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizeReference(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  auto* data = static_cast<OpDataQuantizeReference*>(node->user_data);

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  if (input->type == kTfLiteFloat32) {
    switch (output->type) {
      case kTfLiteInt8: {
        // An ARG_ERROR here (out-of-range zero_point) means the kernel wrote
        // no output at all, so the status must not be discarded.
        const arm_cmsis_nn_status status = arm_quantize_f32_s8(
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorData<int8_t>(output),
          ElementCount(*input->dims),
          data->quantization_params.zero_point,
          data->quantization_params.scale
        );
        if (status != ARM_CMSIS_NN_SUCCESS) {
          MicroPrintf(
              "Quantize: arm_quantize_f32_s8 failed (%d) for zero_point %d.",
              static_cast<int>(status),
              static_cast<int>(data->quantization_params.zero_point));
          return kTfLiteError;
        }
        break;
      }
      case kTfLiteInt16: {
        const arm_cmsis_nn_status status = arm_quantize_f32_s16(
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorData<int16_t>(output),
          ElementCount(*input->dims),
          data->quantization_params.zero_point,
          data->quantization_params.scale
        );
        if (status != ARM_CMSIS_NN_SUCCESS) {
          MicroPrintf(
              "Quantize: arm_quantize_f32_s16 failed (%d) for zero_point %d.",
              static_cast<int>(status),
              static_cast<int>(data->quantization_params.zero_point));
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
  } else if (input->type == kTfLiteInt32) {
    size_t size = ElementCount(*input->dims);
    switch (output->type) {
      case kTfLiteInt8:
        reference_ops::Requantize(
            tflite::micro::GetTensorData<int32_t>(input), size,
            data->requantize_output_multiplier, data->requantize_output_shift,
            data->input_zero_point, data->quantization_params.zero_point,
            tflite::micro::GetTensorData<int8_t>(output));
        break;
      case kTfLiteInt16:
        reference_ops::Requantize(
            tflite::micro::GetTensorData<int32_t>(input), size,
            data->requantize_output_multiplier, data->requantize_output_shift,
            data->input_zero_point, data->quantization_params.zero_point,
            tflite::micro::GetTensorData<int16_t>(output));
        break;
      default:
        MicroPrintf("Input %s, output %s not supported.",
                    TfLiteTypeGetName(input->type),
                    TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else if (input->type == kTfLiteInt16) {
    size_t size = ElementCount(*input->dims);
    switch (output->type) {
      case kTfLiteInt8:
        reference_ops::Requantize(
            tflite::micro::GetTensorData<int16_t>(input), size,
            data->requantize_output_multiplier, data->requantize_output_shift,
            data->input_zero_point, data->quantization_params.zero_point,
            tflite::micro::GetTensorData<int8_t>(output));
        break;
      case kTfLiteInt16: {
        const arm_cmsis_nn_status status = arm_requantize_s16_s16(
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorData<int16_t>(output),
          size,
          data->requantize_output_multiplier,
          data->requantize_output_shift,
          data->input_zero_point,
          data->quantization_params.zero_point
        );
        if (status != ARM_CMSIS_NN_SUCCESS) {
          MicroPrintf(
              "Quantize: arm_requantize_s16_s16 failed (%d) for zero_point %d.",
              static_cast<int>(status),
              static_cast<int>(data->quantization_params.zero_point));
          return kTfLiteError;
        }
        break;
      }
      case kTfLiteInt32:
        reference_ops::Requantize(
            tflite::micro::GetTensorData<int16_t>(input), size,
            data->requantize_output_multiplier, data->requantize_output_shift,
            data->input_zero_point, data->quantization_params.zero_point,
            tflite::micro::GetTensorData<int32_t>(output));
        return kTfLiteOk;
      default:
        MicroPrintf("Input %s, output %s not supported.",
                    TfLiteTypeGetName(input->type),
                    TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else if (input->type == kTfLiteInt8) {
    // Int8 to Int8 requantization, required if the input and output tensors
    // have different scales and/or zero points.
    size_t size = ElementCount(*input->dims);
    switch (output->type) {
      case kTfLiteInt8: {
        const arm_cmsis_nn_status status = arm_requantize_s8_s8(
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorData<int8_t>(output),
          size,
          data->requantize_output_multiplier,
          data->requantize_output_shift,
          data->input_zero_point,
          data->quantization_params.zero_point
        );
        if (status != ARM_CMSIS_NN_SUCCESS) {
          MicroPrintf(
              "Quantize: arm_requantize_s8_s8 failed (%d) for zero_point %d.",
              static_cast<int>(status),
              static_cast<int>(data->quantization_params.zero_point));
          return kTfLiteError;
        }
        break;
      }
      case kTfLiteUInt8:
        reference_ops::Requantize(
            tflite::micro::GetTensorData<int8_t>(input), size,
            data->requantize_output_multiplier, data->requantize_output_shift,
            data->input_zero_point, data->quantization_params.zero_point,
            tflite::micro::GetTensorData<uint8_t>(output));
        break;
      case kTfLiteInt16:
        reference_ops::Requantize(
            tflite::micro::GetTensorData<int8_t>(input), size,
            data->requantize_output_multiplier, data->requantize_output_shift,
            data->input_zero_point, data->quantization_params.zero_point,
            tflite::micro::GetTensorData<int16_t>(output));
        break;
      case kTfLiteInt32:
        reference_ops::Requantize(
            tflite::micro::GetTensorData<int8_t>(input), size,
            data->requantize_output_multiplier, data->requantize_output_shift,
            data->input_zero_point, data->quantization_params.zero_point,
            tflite::micro::GetTensorData<int32_t>(output));
        break;
      default:
        MicroPrintf("Input %s, output %s not supported.",
                    TfLiteTypeGetName(input->type),
                    TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else if (input->type == kTfLiteUInt8) {
    size_t size = ElementCount(*input->dims);
    switch (output->type) {
      case kTfLiteInt8:
        reference_ops::Requantize(
            tflite::micro::GetTensorData<uint8_t>(input), size,
            data->requantize_output_multiplier, data->requantize_output_shift,
            data->input_zero_point, data->quantization_params.zero_point,
            tflite::micro::GetTensorData<int8_t>(output));
        break;
      default:
        MicroPrintf("Input %s, output %s not supported.",
                    TfLiteTypeGetName(input->type),
                    TfLiteTypeGetName(output->type));
        return kTfLiteError;
    }
  } else {
    MicroPrintf("Input %s, output %s not supported.",
                TfLiteTypeGetName(input->type),
                TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace tflite
