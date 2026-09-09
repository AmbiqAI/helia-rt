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

#include "tensorflow/lite/kernels/internal/reference/fill.h"

#include <stdint.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "Include/arm_nnsupportfunctions.h"
#include "tensorflow/lite/micro/kernels/helia/helia_data_movement.h"

namespace tflite {

namespace {

template <typename T>
TfLiteStatus EnsureEqImpl(TfLiteContext* context, const TfLiteIntArray* array,
                          const TfLiteTensor* tensor) {
  for (int i = 0; i < array->size; ++i) {
    TF_LITE_ENSURE_EQ(context, array->data[i], GetTensorData<T>(tensor)[i]);
  }
  return kTfLiteOk;
}

// Ensure the equality of an int array and a tensor, which must be
// one-dimensional and of an integer type.
TfLiteStatus EnsureEq(TfLiteContext* context, const TfLiteIntArray* array,
                      const TfLiteTensor* tensor) {
  TF_LITE_ENSURE(context, tensor->dims != nullptr);
  TF_LITE_ENSURE_EQ(context, NumDimensions(tensor), 1);
  const auto tensor_len = tensor->dims->data[0];
  TF_LITE_ENSURE_EQ(context, array->size, tensor_len);
  TF_LITE_ENSURE(context, tensor_len == 0 || tensor->data.raw != nullptr);

  switch (tensor->type) {
    case kTfLiteInt8:
      return EnsureEqImpl<int8_t>(context, array, tensor);
    case kTfLiteInt16:
      return EnsureEqImpl<int16_t>(context, array, tensor);
    case kTfLiteInt32:
      return EnsureEqImpl<int32_t>(context, array, tensor);
    case kTfLiteInt64:
      return EnsureEqImpl<int64_t>(context, array, tensor);
    default:
      MicroPrintf("cannot compare int array to tensor of type %d.",
                  tensor->type);
      return kTfLiteError;
  }
}

constexpr int kDimsTensor = 0;
constexpr int kValueTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus FillPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const auto* value = micro::GetEvalInput(context, node, kValueTensor);
  const auto* output = micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, value->dims != nullptr && output->dims != nullptr);
  TF_LITE_ENSURE_EQ(context, value->dims->size, 0);
  TF_LITE_ENSURE_EQ(context, value->type, output->type);
  if (IsHeliaDataMovementFloat(value->type)) {
    TF_LITE_ENSURE_OK(context, CheckHeliaDataMovementType(context, value->type));
    auto* data = static_cast<HeliaDataMovementData*>(node->user_data);
    TF_LITE_ENSURE(context, data != nullptr);
    TF_LITE_ENSURE_OK(context, HeliaDataMovementElements(
                                   context, output->dims, value->type,
                                   &data->elements));
  }
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* dims = micro_context->AllocateTempInputTensor(node, kDimsTensor);
  TF_LITE_ENSURE(context, dims != nullptr);
  const bool constant_dims = IsConstantTensor(dims);
  const TfLiteStatus shape_status =
      constant_dims ? EnsureEq(context, output->dims, dims) : kTfLiteError;
  micro_context->DeallocateTempTfLiteTensor(dims);
  TF_LITE_ENSURE_MSG(context, constant_dims,
                     "Non-constant >dims< tensor is not supported");
  TF_LITE_ENSURE_OK(context, shape_status);
  return kTfLiteOk;
}

// Generic fallback (used for float, int32, etc.)
template <typename T>
inline void FillImpl(const TfLiteEvalTensor* value, TfLiteEvalTensor* output) {
  reference_ops::Fill(
      micro::GetTensorShape(value), micro::GetTensorData<T>(value),
      micro::GetTensorShape(output), micro::GetTensorData<T>(output));
}

template <>
inline void FillImpl<int8_t>(const TfLiteEvalTensor* value,
                             TfLiteEvalTensor* output) {
  const RuntimeShape& value_shape = micro::GetTensorShape(value);
  TFLITE_DCHECK_EQ(value_shape.DimensionsCount(), 0);
  const int8_t v = *micro::GetTensorData<int8_t>(value);

  int8_t* out_ptr = micro::GetTensorData<int8_t>(output);
  const uint32_t n = static_cast<uint32_t>(micro::GetTensorShape(output).FlatSize());
  arm_memset_s8(out_ptr, v, n);
}

template <>
inline void FillImpl<int16_t>(const TfLiteEvalTensor* value,
                              TfLiteEvalTensor* output) {
  const RuntimeShape& value_shape = micro::GetTensorShape(value);
  TFLITE_DCHECK_EQ(value_shape.DimensionsCount(), 0);
  const int16_t v = *micro::GetTensorData<int16_t>(value);

  int16_t* out_ptr = micro::GetTensorData<int16_t>(output);
  const uint32_t n = static_cast<uint32_t>(micro::GetTensorShape(output).FlatSize());
  arm_memset_s16(out_ptr, v, n);
}

TfLiteStatus FillEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* value =
      micro::GetEvalInput(context, node, kValueTensor);
  TfLiteEvalTensor* output = micro::GetEvalOutput(context, node, kOutputTensor);

  switch (value->type) {
    case kTfLiteFloat32: {
      const auto* data = static_cast<const HeliaDataMovementData*>(node->user_data);
      if (data->elements == 0) return kTfLiteOk;
      TF_LITE_ENSURE(context, value->data.raw != nullptr);
#if ARM_NN_ENABLE_F32
      const auto status = arm_nn_fill_f32(
          *micro::GetTensorData<float>(value), micro::GetTensorData<float>(output),
          data->elements);
      TF_LITE_ENSURE_EQ(context, status, ARM_CMSIS_NN_SUCCESS);
#else
      RuntimeShape flat_shape(1);
      flat_shape.SetDim(0, data->elements);
      reference_ops::Fill(RuntimeShape(), micro::GetTensorData<float>(value),
                          flat_shape, micro::GetTensorData<float>(output));
#endif
      return kTfLiteOk;
    }
    case kTfLiteFloat16: {
      const auto* data = static_cast<const HeliaDataMovementData*>(node->user_data);
      if (data->elements == 0) return kTfLiteOk;
      TF_LITE_ENSURE(context, value->data.raw != nullptr);
#if ARM_NN_ENABLE_F16
      const auto status = arm_nn_fill_f16(
          *micro::GetTensorData<float16_t>(value), micro::GetTensorData<float16_t>(output),
          data->elements);
      TF_LITE_ENSURE_EQ(context, status, ARM_CMSIS_NN_SUCCESS);
#else
      return kTfLiteError;
#endif
      return kTfLiteOk;
    }
    case kTfLiteInt32:
      FillImpl<int32_t>(value, output);
      break;
    case kTfLiteInt8:
      FillImpl<int8_t>(value, output);
      break;
    case kTfLiteInt16:
      FillImpl<int16_t>(value, output);
      break;
    default:
      MicroPrintf("Fill does not support type '%s'.",
                  TfLiteTypeGetName(value->type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_FILL() {
  return tflite::micro::RegisterOp(InitHeliaDataMovement, FillPrepare, FillEval);
}

}  // namespace tflite
