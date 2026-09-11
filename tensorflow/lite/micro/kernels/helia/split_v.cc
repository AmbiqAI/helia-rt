/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstring>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/helia/helia_data_movement.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kSizesTensor = 1;
constexpr int kAxisTensor = 2;

TfLiteStatus CheckConstantInput(TfLiteContext* context, TfLiteNode* node,
                                int index) {
  auto* micro_context = GetMicroContext(context);
  auto* tensor = micro_context->AllocateTempInputTensor(node, index);
  TF_LITE_ENSURE(context, tensor != nullptr);
  const bool constant = IsConstantTensor(tensor);
  micro_context->DeallocateTempTfLiteTensor(tensor);
  TF_LITE_ENSURE_MSG(context, constant,
                     "SPLIT_V axis and size_splits must be constant");
  return kTfLiteOk;
}

TfLiteStatus SplitVPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE(context, NumOutputs(node) > 0);
  TF_LITE_ENSURE_OK(context, CheckConstantInput(context, node, kAxisTensor));
  TF_LITE_ENSURE_OK(context, CheckConstantInput(context, node, kSizesTensor));
  const auto* input = micro::GetEvalInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input->type == kTfLiteInt8 ||
                              input->type == kTfLiteInt16 ||
                              input->type == kTfLiteInt32 ||
                              IsHeliaDataMovementFloat(input->type));
  TF_LITE_ENSURE_OK(context, CheckHeliaDataMovementType(context, input->type));
  auto* data = static_cast<HeliaDataMovementData*>(node->user_data);
  TF_LITE_ENSURE(context, data != nullptr);
  data->count = NumOutputs(node);
  const auto* params =
      static_cast<const TfLiteSplitVParams*>(node->builtin_data);
  if (params != nullptr) {
    TF_LITE_ENSURE_EQ(context, params->num_splits, data->count);
  }
  // The shared shape check conservatively budgets four bytes for integers.
  TF_LITE_ENSURE_OK(context,
                    HeliaDataMovementElements(context, input->dims, input->type,
                                              &data->elements));
  const int rank = input->dims->size;
  TF_LITE_ENSURE(context, rank > 0);
  const auto* axis = micro::GetEvalInput(context, node, kAxisTensor);
  TF_LITE_ENSURE_EQ(context, axis->type, kTfLiteInt32);
  int32_t axis_elements;
  TF_LITE_ENSURE_OK(context,
                    HeliaDataMovementElements(context, axis->dims, kTfLiteInt32,
                                              &axis_elements));
  TF_LITE_ENSURE_EQ(context, axis_elements, 1);
  TF_LITE_ENSURE(context, axis->data.raw != nullptr);
  data->axis = micro::GetTensorData<int32_t>(axis)[0];
  if (data->axis < 0) data->axis += rank;
  TF_LITE_ENSURE(context, data->axis >= 0 && data->axis < rank);
  const auto* sizes = micro::GetEvalInput(context, node, kSizesTensor);
  TF_LITE_ENSURE_EQ(context, sizes->type, kTfLiteInt32);
  TF_LITE_ENSURE(context, sizes->dims != nullptr);
  TF_LITE_ENSURE_EQ(context, sizes->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, sizes->dims->data[0], data->count);
  TF_LITE_ENSURE(context, sizes->data.raw != nullptr);
  const auto* lengths = micro::GetTensorData<int32_t>(sizes);
  const int32_t axis_length = input->dims->data[data->axis];
  int inferred = -1;
  int64_t known = 0;
  for (int i = 0; i < data->count; ++i) {
    if (lengths[i] == -1) {
      TF_LITE_ENSURE_MSG(context, inferred == -1,
                         "SPLIT_V permits only one inferred size");
      inferred = i;
    } else {
      TF_LITE_ENSURE(context, lengths[i] >= 0);
      known += lengths[i];
      TF_LITE_ENSURE(context, known <= axis_length);
    }
  }
  TF_LITE_ENSURE(context, inferred >= 0 || known == axis_length);
  for (int i = 0; i < data->count; ++i) {
    const auto* output = micro::GetEvalOutput(context, node, i);
    TF_LITE_ENSURE_EQ(context, output->type, input->type);
    TF_LITE_ENSURE(context, output->dims != nullptr);
    TF_LITE_ENSURE_EQ(context, output->dims->size, rank);
    const int32_t length = i == inferred ? axis_length - known : lengths[i];
    for (int d = 0; d < rank; ++d) {
      TF_LITE_ENSURE_EQ(context, output->dims->data[d],
                        d == data->axis ? length : input->dims->data[d]);
    }
  }
  if (data->elements == 0) return kTfLiteOk;
  TF_LITE_ENSURE(context,
                 static_cast<size_t>(data->count) <=
                     std::numeric_limits<size_t>::max() / sizeof(int32_t));
  data->split_sizes = static_cast<int32_t*>(context->AllocatePersistentBuffer(
      context, static_cast<size_t>(data->count) * sizeof(int32_t)));
  TF_LITE_ENSURE(context, data->split_sizes != nullptr);
  for (int i = 0; i < data->count; ++i) {
    data->split_sizes[i] = i == inferred ? axis_length - known : lengths[i];
  }
  TF_LITE_ENSURE_OK(context,
                    HeliaDataMovementShape(context, input->dims, data));
  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16 ||
      (input->type == kTfLiteFloat32 && kHeliaFloat32Enabled) ||
      input->type == kTfLiteFloat16) {
    TF_LITE_ENSURE_OK(context, HeliaDataMovementPointers(context, data));
  }
  return kTfLiteOk;
}

template <typename T, typename Kernel>
TfLiteStatus SplitVNative(TfLiteContext* context, TfLiteNode* node,
                          const TfLiteEvalTensor* input,
                          const HeliaDataMovementData* data, Kernel kernel) {
  auto** outputs = static_cast<T**>(
      context->GetScratchBuffer(context, data->pointer_scratch));
  TF_LITE_ENSURE(context, outputs != nullptr);
  T empty_output = {};
  for (int i = 0; i < data->count; ++i) {
    // Native split requires a valid pointer even for a zero-length piece.
    outputs[i] =
        data->split_sizes[i] == 0
            ? &empty_output
            : micro::GetTensorData<T>(micro::GetEvalOutput(context, node, i));
    TF_LITE_ENSURE(context, outputs[i] != nullptr);
  }
  const auto status =
      kernel(micro::GetTensorData<T>(input), input->dims->size, data->shape,
             data->axis, data->count, data->split_sizes, outputs);
  TF_LITE_ENSURE_EQ(context, status, ARM_CMSIS_NN_SUCCESS);
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus SplitVReference(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteEvalTensor* input,
                             const HeliaDataMovementData* data) {
  int32_t outer = 1;
  int32_t inner = 1;
  for (int d = 0; d < data->axis; ++d) outer *= data->shape[d];
  for (int d = data->axis + 1; d < input->dims->size; ++d)
    inner *= data->shape[d];
  const T* source = micro::GetTensorData<T>(input);
  for (int k = 0; k < outer; ++k) {
    for (int i = 0; i < data->count; ++i) {
      const int32_t count = data->split_sizes[i] * inner;
      if (count == 0) continue;
      auto* output =
          micro::GetTensorData<T>(micro::GetEvalOutput(context, node, i));
      TF_LITE_ENSURE(context, output != nullptr);
      memcpy(output + k * count, source,
             static_cast<size_t>(count) * sizeof(T));
      source += count;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus SplitVEval(TfLiteContext* context, TfLiteNode* node) {
  const auto* data = static_cast<const HeliaDataMovementData*>(node->user_data);
  if (data->elements == 0) return kTfLiteOk;
  const auto* input = micro::GetEvalInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input->data.raw != nullptr);
  switch (input->type) {
    case kTfLiteInt8:
      return SplitVNative<int8_t>(context, node, input, data, arm_split_s8);
    case kTfLiteInt16:
      return SplitVNative<int16_t>(context, node, input, data, arm_split_s16);
    case kTfLiteInt32:
      return SplitVReference<int32_t>(context, node, input, data);
    case kTfLiteFloat32:
#if ARM_NN_ENABLE_F32
      return SplitVNative<float>(context, node, input, data, arm_split_f32);
#else
      return SplitVReference<float>(context, node, input, data);
#endif
#if ARM_NN_ENABLE_F16
    case kTfLiteFloat16:
      return SplitVNative<float16_t>(context, node, input, data, arm_split_f16);
#endif
    default:
      return kTfLiteError;
  }
}

}  // namespace

TFLMRegistration Register_SPLIT_V() {
  return micro::RegisterOp(InitHeliaDataMovement, SplitVPrepare, SplitVEval);
}

}  // namespace tflite
