/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/arg_min_max.h"

#include <climits>
#include <cstddef>
#include <cstdint>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/comparisons.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/helia/helia_float_common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

constexpr int kInputTensor = 0;
constexpr int kAxis = 1;
constexpr int kOutputTensor = 0;
constexpr int kCoreRank = 4;

class ScopedTempTensor {
 public:
  ScopedTempTensor(MicroContext* context, TfLiteTensor* tensor)
      : context_(context), tensor_(tensor) {}
  ~ScopedTempTensor() {
    if (tensor_ != nullptr) {
      context_->DeallocateTempTfLiteTensor(tensor_);
    }
  }

 private:
  MicroContext* context_;
  TfLiteTensor* tensor_;
};

struct CoreArgShape {
  cmsis_nn_dims input_dims;
  int32_t axis;
  size_t input_count;
  size_t output_count;
};

struct OpData {
  size_t input_capacity;
  size_t output_capacity;
  bool prepared_float16;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  OpData* data = static_cast<OpData*>(
      context->AllocatePersistentBuffer(context, sizeof(OpData)));
  if (data != nullptr) {
    data->input_capacity = 0;
    data->output_capacity = 0;
    data->prepared_float16 = false;
  }
  return data;
}

bool CountFits(const TfLiteIntArray* dims, size_t element_size, size_t* count) {
  if (dims == nullptr || dims->size < 0 || element_size == 0 ||
      count == nullptr) {
    return false;
  }
  for (int i = 0; i < dims->size; ++i) {
    if (dims->data[i] < 0) {
      return false;
    }
  }
  for (int i = 0; i < dims->size; ++i) {
    if (dims->data[i] == 0) {
      *count = 0;
      return true;
    }
  }
  size_t elements = 1;
  const size_t limit = INT32_MAX / element_size;
  for (int i = 0; i < dims->size; ++i) {
    const size_t extent = static_cast<size_t>(dims->data[i]);
    if (extent > limit / elements) {
      return false;
    }
    elements *= extent;
  }
  *count = elements;
  return true;
}

bool ValidateCoreArgShape(const TfLiteIntArray* input_dims,
                          const TfLiteIntArray* axis_dims,
                          const int32_t* axis_data,
                          const TfLiteIntArray* output_dims,
                          CoreArgShape* core_shape) {
  if (input_dims == nullptr || axis_dims == nullptr || axis_data == nullptr ||
      output_dims == nullptr || core_shape == nullptr || input_dims->size < 1 ||
      input_dims->size > kCoreRank ||
      output_dims->size != input_dims->size - 1) {
    return false;
  }

  size_t axis_count;
  if (!CountFits(axis_dims, sizeof(*axis_data), &axis_count) ||
      axis_count != 1) {
    return false;
  }

  int axis = axis_data[0];
  if (axis < 0) {
    axis += input_dims->size;
  }
  if (axis < 0 || axis >= input_dims->size) {
    return false;
  }

  for (int i = 0; i < input_dims->size; ++i) {
    if (input_dims->data[i] < 0) {
      return false;
    }
  }
  if (input_dims->data[axis] == 0) {
    return false;
  }
  for (int input_index = 0, output_index = 0; input_index < input_dims->size;
       ++input_index) {
    if (input_index == axis) {
      continue;
    }
    if (output_dims->data[output_index++] != input_dims->data[input_index]) {
      return false;
    }
  }

  if (!CountFits(input_dims, sizeof(uint16_t), &core_shape->input_count) ||
      !CountFits(output_dims, sizeof(int32_t), &core_shape->output_count)) {
    return false;
  }

  int32_t extended_dims[kCoreRank] = {1, 1, 1, 1};
  const int padding = kCoreRank - input_dims->size;
  for (int i = 0; i < input_dims->size; ++i) {
    extended_dims[padding + i] = input_dims->data[i];
  }
  core_shape->input_dims = {extended_dims[0], extended_dims[1],
                            extended_dims[2], extended_dims[3]};
  core_shape->axis = axis + padding;
  return true;
}

TfLiteStatus ArgMinMaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpData* data = static_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, data != nullptr);
  data->prepared_float16 = false;

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  ScopedTempTensor input_guard(micro_context, input);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* axis = micro_context->AllocateTempInputTensor(node, kAxis);
  ScopedTempTensor axis_guard(micro_context, axis);
  TF_LITE_ENSURE(context, axis != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  ScopedTempTensor output_guard(micro_context, output);
  TF_LITE_ENSURE(context, output != nullptr);

  if (input->type != kTfLiteFloat16) {
    return kTfLiteOk;
  }
  TF_LITE_ENSURE(context, kHeliaFloat16Enabled);
  TF_LITE_ENSURE_TYPES_EQ(context, axis->type, kTfLiteInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt32);

  CoreArgShape core_shape;
  TF_LITE_ENSURE(context,
                 ValidateCoreArgShape(input->dims, axis->dims, axis->data.i32,
                                      output->dims, &core_shape));
  data->input_capacity = core_shape.input_count;
  data->output_capacity = core_shape.output_count;
  data->prepared_float16 = true;
  return kTfLiteOk;
}

template <typename T1, typename T2, typename T3>
inline void ArgMinMaxHelper(const RuntimeShape& input1_shape,
                            const T1* input1_data, const T3* input2_data,
                            const RuntimeShape& output_shape, T2* output_data,
                            bool is_arg_max) {
  // Use Greater/Less from comparisons.h (formerly from kernels/micro_utils.h
  // which was deprecated). Same as gtl::Greater but used here to reduce
  // dependencies and binary size for micro environment.
  if (is_arg_max) {
    reference_ops::ArgMinMax(input1_shape, input1_data, input2_data,
                             output_shape, output_data,
                             reference_ops::GreaterFn<T1>);
  } else {
    reference_ops::ArgMinMax(input1_shape, input1_data, input2_data,
                             output_shape, output_data,
                             reference_ops::LessFn<T1>);
  }
}

TfLiteStatus EvalReference(TfLiteContext* context,
                           const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* axis,
                           TfLiteEvalTensor* output, bool is_arg_max) {
#define TF_LITE_ARG_MIN_MAX(data_type, axis_type, output_type)       \
  ArgMinMaxHelper(tflite::micro::GetTensorShape(input),              \
                  tflite::micro::GetTensorData<data_type>(input),    \
                  tflite::micro::GetTensorData<axis_type>(axis),     \
                  tflite::micro::GetTensorShape(output),             \
                  tflite::micro::GetTensorData<output_type>(output), \
                  is_arg_max)
  if (axis->type == kTfLiteInt32) {
    if (output->type == kTfLiteInt32) {
      switch (input->type) {
        case kTfLiteFloat32:
          TF_LITE_ARG_MIN_MAX(float, int32_t, int32_t);
          break;
        case kTfLiteInt8:
          TF_LITE_ARG_MIN_MAX(int8_t, int32_t, int32_t);
          break;
        default:
          MicroPrintf(
              "Only float32, uint8_t and int8_t are "
              "supported currently, got %s.",
              TfLiteTypeGetName(input->type));
          return kTfLiteError;
      }
    } else {
      MicroPrintf("Only int32_t are supported currently, got %s.",
                  TfLiteTypeGetName(output->type));
      return kTfLiteError;
    }
  } else {
    MicroPrintf("Only int32_t are supported currently, got %s.",
                TfLiteTypeGetName(axis->type));
    return kTfLiteError;
  }

#undef TF_LITE_ARG_MIN_MAX

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node, bool is_arg_max) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* axis =
      tflite::micro::GetEvalInput(context, node, kAxis);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  if (input->type != kTfLiteFloat16) {
    return EvalReference(context, input, axis, output, is_arg_max);
  }
  TF_LITE_ENSURE_TYPES_EQ(context, axis->type, kTfLiteInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt32);

#if ARM_NN_ENABLE_F16
  const OpData* data = static_cast<const OpData*>(node->user_data);
  TF_LITE_ENSURE(context, data != nullptr && data->prepared_float16);
  const int32_t* axis_data = tflite::micro::GetTensorData<int32_t>(axis);
  CoreArgShape core_shape;
  TF_LITE_ENSURE(context,
                 ValidateCoreArgShape(input->dims, axis->dims, axis_data,
                                      output->dims, &core_shape));
  TF_LITE_ENSURE(context, core_shape.input_count <= data->input_capacity &&
                              core_shape.output_count <= data->output_capacity);
  const float16_t* input_data = tflite::micro::GetTensorData<float16_t>(input);
  int32_t* output_data = tflite::micro::GetTensorData<int32_t>(output);
  TF_LITE_ENSURE(context,
                 (core_shape.input_count == 0 || input_data != nullptr) &&
                     (core_shape.output_count == 0 || output_data != nullptr));
  const arm_cmsis_nn_status status =
      is_arg_max ? arm_argmax_f16(input_data, &core_shape.input_dims,
                                  core_shape.axis, output_data)
                 : arm_argmin_f16(input_data, &core_shape.input_dims,
                                  core_shape.axis, output_data);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    MicroPrintf("ARG_%s failed (%d).", is_arg_max ? "MAX" : "MIN", status);
    return kTfLiteError;
  }
  return kTfLiteOk;
#else
  return kTfLiteError;
#endif
}

TfLiteStatus ArgMinEval(TfLiteContext* context, TfLiteNode* node) {
  return Eval(context, node, false);
}

TfLiteStatus ArgMaxEval(TfLiteContext* context, TfLiteNode* node) {
  return Eval(context, node, true);
}

}  // namespace

TFLMRegistration Register_ARG_MAX() {
  return tflite::micro::RegisterOp(Init, ArgMinMaxPrepare, ArgMaxEval);
}

TFLMRegistration Register_ARG_MIN() {
  return tflite::micro::RegisterOp(Init, ArgMinMaxPrepare, ArgMinEval);
}

}  // namespace tflite
