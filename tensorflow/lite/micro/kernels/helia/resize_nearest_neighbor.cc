/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/internal/reference/resize_nearest_neighbor.h"

#include <cstdint>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/helia/helia_float_common.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/op_macros.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

constexpr int kInputTensor = 0;
constexpr int kSizeTensor = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  // True when the layer runs on the heliaCORE entry for its type.
  bool use_helia;
  int scratch_index;
  int32_t scratch_size;
};

void* ResizeNearestNeighborInit(TfLiteContext* context, const char* buffer,
                                size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

bool HasHeliaEntry(TfLiteType type) {
  switch (type) {
    case kTfLiteInt8:
    case kTfLiteInt16:
      return true;
    case kTfLiteFloat32:
      return kHeliaFloat32Enabled;
    case kTfLiteFloat16:
      return kHeliaFloat16Enabled;
    default:
      return false;
  }
}

// Scratch bytes for the heliaCORE entry, or -1. The f32/f16 headers name a
// sizer; the s8/s16 headers state the same (height + width) int32_t count.
int32_t HeliaScratchSize(TfLiteType type, const cmsis_nn_dims& output_dims) {
#if ARM_NN_ENABLE_F32
  if (type == kTfLiteFloat32) {
    return arm_resize_nearest_neighbor_f32_get_buffer_size(&output_dims);
  }
#endif
#if ARM_NN_ENABLE_F16
  if (type == kTfLiteFloat16) {
    return arm_resize_nearest_neighbor_f16_get_buffer_size(&output_dims);
  }
#endif
  const int64_t bytes =
      (static_cast<int64_t>(output_dims.h) + output_dims.w) *
      static_cast<int64_t>(sizeof(int32_t));
  return bytes <= INT32_MAX ? static_cast<int32_t>(bytes) : -1;
}

TfLiteStatus ResizeNearestNeighborPrepare(TfLiteContext* context,
                                          TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TfLiteTensor* size =
      micro_context->AllocateTempInputTensor(node, kSizeTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);

  // Our current implementations rely on the input being 4D,
  // and the size being 1D tensor with exactly 2 elements.
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(size), 1);
  TF_LITE_ENSURE_EQ(context, size->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, size->dims->data[0], 2);

  output->type = input->type;

  TF_LITE_ENSURE_MSG(context, IsConstantTensor(size),
                     "Non-constant >size< tensor is not supported");
  // Float16 has no reference kernel to fall back to.
  TF_LITE_ENSURE_MSG(context,
                     input->type != kTfLiteFloat16 || kHeliaFloat16Enabled,
                     "Float16 RESIZE_NEAREST_NEIGHBOR requires heliaCORE "
                     "float16 support.");

  OpData* data = static_cast<OpData*>(node->user_data);
  data->use_helia = false;
  data->scratch_index = -1;
  data->scratch_size = 0;

  // The entries reject a layer whose output shape is not
  // [N, size[0], size[1], C]; such a layer keeps the reference path.
  const int32_t* size_data = GetTensorData<int32_t>(size);
  const cmsis_nn_dims output_dims = {
      output->dims->size == 4 ? output->dims->data[0] : 0,
      output->dims->size == 4 ? output->dims->data[1] : 0,
      output->dims->size == 4 ? output->dims->data[2] : 0,
      output->dims->size == 4 ? output->dims->data[3] : 0};
  const bool shapes_match =
      HasHeliaEntry(input->type) && size_data != nullptr &&
      output->dims->size == 4 && input->dims->data[0] > 0 &&
      input->dims->data[1] > 0 && input->dims->data[2] > 0 &&
      input->dims->data[3] > 0 && output_dims.n == input->dims->data[0] &&
      output_dims.c == input->dims->data[3] && size_data[0] > 0 &&
      size_data[1] > 0 && output_dims.h == size_data[0] &&
      output_dims.w == size_data[1];
  if (shapes_match) {
    const int32_t scratch_size = HeliaScratchSize(input->type, output_dims);
    if (scratch_size > 0) {
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
          context, scratch_size, &data->scratch_index));
      data->scratch_size = scratch_size;
      data->use_helia = true;
    }
  }
  TF_LITE_ENSURE_MSG(context,
                     input->type != kTfLiteFloat16 || data->use_helia,
                     "Float16 RESIZE_NEAREST_NEIGHBOR requires an output "
                     "shape of [N, size[0], size[1], C].");

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(size);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus EvalHelia(TfLiteContext* context, const OpData& data,
                       const TfLiteResizeNearestNeighborParams& params,
                       const TfLiteEvalTensor* input,
                       const TfLiteEvalTensor* size, TfLiteEvalTensor* output) {
  // The entry reads input elements of the output's type.
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  const cmsis_nn_context ctx = {
      context->GetScratchBuffer(context, data.scratch_index),
      data.scratch_size};
  const cmsis_nn_resize_params resize_params = {params.align_corners,
                                                params.half_pixel_centers};
  const cmsis_nn_dims input_dims = {
      input->dims->data[0], input->dims->data[1], input->dims->data[2],
      input->dims->data[3]};
  const cmsis_nn_dims size_dims = {1, 1, 1, 2};
  const cmsis_nn_dims output_dims = {
      output->dims->data[0], output->dims->data[1], output->dims->data[2],
      output->dims->data[3]};
  const int32_t* size_data = tflite::micro::GetTensorData<int32_t>(size);

  arm_cmsis_nn_status status = ARM_CMSIS_NN_ARG_ERROR;
  const char* entry = "";
  switch (output->type) {
    case kTfLiteInt8:
      entry = "arm_resize_nearest_neighbor_s8";
      status = arm_resize_nearest_neighbor_s8(
          &ctx, &resize_params, &input_dims,
          tflite::micro::GetTensorData<int8_t>(input), &size_dims, size_data,
          &output_dims, tflite::micro::GetTensorData<int8_t>(output));
      break;
    case kTfLiteInt16:
      entry = "arm_resize_nearest_neighbor_s16";
      status = arm_resize_nearest_neighbor_s16(
          &ctx, &resize_params, &input_dims,
          tflite::micro::GetTensorData<int16_t>(input), &size_dims, size_data,
          &output_dims, tflite::micro::GetTensorData<int16_t>(output));
      break;
#if ARM_NN_ENABLE_F32
    case kTfLiteFloat32:
      entry = "arm_resize_nearest_neighbor_f32";
      status = arm_resize_nearest_neighbor_f32(
          &ctx, &resize_params, &input_dims,
          tflite::micro::GetTensorData<float>(input), &size_dims, size_data,
          &output_dims, tflite::micro::GetTensorData<float>(output));
      break;
#endif
#if ARM_NN_ENABLE_F16
    case kTfLiteFloat16:
      entry = "arm_resize_nearest_neighbor_f16";
      status = arm_resize_nearest_neighbor_f16(
          &ctx, &resize_params, &input_dims,
          tflite::micro::GetTensorData<float16_t>(input), &size_dims,
          size_data, &output_dims,
          tflite::micro::GetTensorData<float16_t>(output));
      break;
#endif
    default:
      MicroPrintf("RESIZE_NEAREST_NEIGHBOR: type %s is not supported.",
                  TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
  if (status != ARM_CMSIS_NN_SUCCESS) {
    MicroPrintf("RESIZE_NEAREST_NEIGHBOR: %s failed (%d).", entry,
                static_cast<int>(status));
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus ResizeNearestNeighborEval(TfLiteContext* context,
                                       TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteResizeNearestNeighborParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* size =
      tflite::micro::GetEvalInput(context, node, kSizeTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *static_cast<const OpData*>(node->user_data);
  if (data.use_helia) {
    return EvalHelia(context, data, *params, input, size, output);
  }

  tflite::micro::ResizeNearestNeighborParams op_params;
  op_params.align_corners = params->align_corners;
  op_params.half_pixel_centers = params->half_pixel_centers;

  if (output->type == kTfLiteFloat32) {
    reference_ops::ResizeNearestNeighbor(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int32_t>(input),
        tflite::micro::GetTensorShape(size),
        tflite::micro::GetTensorData<int32_t>(size),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int32_t>(output));
  } else if (output->type == kTfLiteInt8) {
    reference_ops::ResizeNearestNeighbor(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(size),
        tflite::micro::GetTensorData<int32_t>(size),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
  } else if (output->type == kTfLiteInt16) {
    reference_ops::ResizeNearestNeighbor(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int16_t>(input),
        tflite::micro::GetTensorShape(size),
        tflite::micro::GetTensorData<int32_t>(size),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int16_t>(output));
  } else {
    MicroPrintf("Output tensor type %s (%d) not supported.",
                TfLiteTypeGetName(output->type), output->type);

    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_RESIZE_NEAREST_NEIGHBOR() {
  return tflite::micro::RegisterOp(ResizeNearestNeighborInit,
                                   ResizeNearestNeighborPrepare,
                                   ResizeNearestNeighborEval);
}

}  // namespace micro
}  // namespace tflite
