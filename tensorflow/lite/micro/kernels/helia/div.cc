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

#include "tensorflow/lite/micro/kernels/internal/reference/div.h"

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/helia/helia_broadcast.h"
#include "tensorflow/lite/micro/kernels/helia/helia_float_common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpDataDiv {
  // Parameters used in the quantized paths where the output is 8bit
  int32_t input1_zero_point;
  int32_t input2_zero_point;
  int32_t output_zero_point;
  int32_t output_activation_min;
  int32_t output_activation_max;

  // Parameters used in all quantized paths
  int32_t output_multiplier;
  int output_shift;

  HeliaBroadcastClass broadcast_class;
};

// Releases a Prepare temporary on every return.
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

TfLiteStatus CalculateOpDataDiv(TfLiteContext* context, TfLiteTensor* input1,
                                TfLiteTensor* input2, TfLiteTensor* output,
                                TfLiteDivParams* params, OpDataDiv* data) {
  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);
  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, output->type);

  if (output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
    const double real_multiplier = static_cast<double>(
        input1->params.scale / (input2->params.scale * output->params.scale));
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
    data->input1_zero_point = input1->params.zero_point;
    data->input2_zero_point = input2->params.zero_point;
    data->output_zero_point = output->params.zero_point;
  }

  return kTfLiteOk;
}

void* DivInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataDiv));
}

TfLiteStatus DivPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input1 =
      micro_context->AllocateTempInputTensor(node, kInputTensor1);
  ScopedTempTensor input1_guard(micro_context, input1);
  TF_LITE_ENSURE(context, input1 != nullptr);
  TfLiteTensor* input2 =
      micro_context->AllocateTempInputTensor(node, kInputTensor2);
  ScopedTempTensor input2_guard(micro_context, input2);
  TF_LITE_ENSURE(context, input2 != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  ScopedTempTensor output_guard(micro_context, output);
  TF_LITE_ENSURE(context, output != nullptr);

  OpDataDiv* data = static_cast<OpDataDiv*>(node->user_data);
  auto* params = reinterpret_cast<TfLiteDivParams*>(node->builtin_data);

  TF_LITE_ENSURE_STATUS(
      CalculateOpDataDiv(context, input1, input2, output, params, data));
  data->broadcast_class = HeliaClassifyBroadcast(
      GetTensorShape(input1), GetTensorShape(input2), GetTensorShape(output));

  // float16 has no reference kernel, so a shape pair the heliaCORE broadcast
  // walk cannot express fails here rather than at Invoke time.
  if (output->type == kTfLiteFloat16) {
    TF_LITE_ENSURE_MSG(context, kHeliaFloat16Enabled,
                       "Float16 DIV requires the fp16 API.");
    TF_LITE_ENSURE_MSG(
        context, data->broadcast_class != HeliaBroadcastClass::kUnsupported,
        "Float16 DIV does not support these input shapes.");
  }

  if (output->type == kTfLiteInt32) {
    // Only support int32 unquantized DIV for now.
    TF_LITE_ENSURE_EQ(context, input1->quantization.type,
                      kTfLiteNoQuantization);
    TF_LITE_ENSURE_EQ(context, input2->quantization.type,
                      kTfLiteNoQuantization);
  }
  return kTfLiteOk;
}

template <typename T>
void EvalDiv(TfLiteContext* context, TfLiteNode* node, TfLiteDivParams* params,
             const OpDataDiv* data, const TfLiteEvalTensor* input1,
             const TfLiteEvalTensor* input2, TfLiteEvalTensor* output) {
  tflite::micro::ArithmeticParams op_params = {};

#define TF_LITE_DIV(type, opname, data_type)                           \
  data_type output_activation_min, output_activation_max;              \
  CalculateActivationRange(params->activation, &output_activation_min, \
                           &output_activation_max);                    \
  SetActivationParams(output_activation_min, output_activation_max,    \
                      &op_params);                                     \
  type::opname(op_params, tflite::micro::GetTensorShape(input1),       \
               tflite::micro::GetTensorData<data_type>(input1),        \
               tflite::micro::GetTensorShape(input2),                  \
               tflite::micro::GetTensorData<data_type>(input2),        \
               tflite::micro::GetTensorShape(output),                  \
               tflite::micro::GetTensorData<data_type>(output))

  bool requires_broadcast = reference_ops::ProcessBroadcastShapes(
      tflite::micro::GetTensorShape(input1),
      tflite::micro::GetTensorShape(input2), &op_params);

  if (requires_broadcast) {
    TF_LITE_DIV(reference_ops, BroadcastDivSlow, T);
  } else {
    TF_LITE_DIV(reference_ops, Div, T);
  }
#undef TF_LITE_DIV
}

#if ARM_NN_ENABLE_F32
// Runs float32 DIV on heliaCORE for the shapes it takes and sets `handled`;
// other shapes stay unhandled for the reference. A failing entry is an error.
TfLiteStatus EvalDivFloat32Helia(TfLiteDivParams* params, const OpDataDiv* data,
                                 const TfLiteEvalTensor* input1,
                                 const TfLiteEvalTensor* input2,
                                 TfLiteEvalTensor* output, bool* handled) {
  const int size = tflite::micro::GetTensorShape(output).FlatSize();
  *handled = data->broadcast_class == HeliaBroadcastClass::kSameShape ||
             HeliaBroadcastNeedsWalk(data->broadcast_class);
  // heliaCORE rejects zero-size dims; an empty output has nothing to write.
  if (!*handled || size == 0) {
    return kTfLiteOk;
  }
  float activation_min;
  float activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);
  arm_cmsis_nn_status status;
  const char* entry;
  if (data->broadcast_class == HeliaBroadcastClass::kSameShape) {
    entry = "arm_elementwise_div_f32";
    status =
        arm_elementwise_div_f32(tflite::micro::GetTensorData<float>(input1),
                                tflite::micro::GetTensorData<float>(input2),
                                tflite::micro::GetTensorData<float>(output),
                                activation_min, activation_max, size);
  } else {
    entry = "arm_elementwise_div_broadcast_f32";
    const cmsis_nn_dims input1_dims =
        HeliaNhwcDims(tflite::micro::GetTensorShape(input1));
    const cmsis_nn_dims input2_dims =
        HeliaNhwcDims(tflite::micro::GetTensorShape(input2));
    const cmsis_nn_dims output_dims =
        HeliaNhwcDims(tflite::micro::GetTensorShape(output));
    status = arm_elementwise_div_broadcast_f32(
        tflite::micro::GetTensorData<float>(input1), &input1_dims,
        tflite::micro::GetTensorData<float>(input2), &input2_dims,
        tflite::micro::GetTensorData<float>(output), &output_dims,
        activation_min, activation_max);
  }
  if (status != ARM_CMSIS_NN_SUCCESS) {
    MicroPrintf("DIV: %s failed (%d).", entry, static_cast<int>(status));
    return kTfLiteError;
  }
  return kTfLiteOk;
}
#endif

TfLiteStatus EvalDivFloat16(TfLiteDivParams* params, const OpDataDiv* data,
                            const TfLiteEvalTensor* input1,
                            const TfLiteEvalTensor* input2,
                            TfLiteEvalTensor* output) {
#if ARM_NN_ENABLE_F16
  // heliaCORE rejects zero-size dims; an empty output has nothing to write.
  if (tflite::micro::GetTensorShape(output).FlatSize() == 0) {
    return kTfLiteOk;
  }
  float activation_min;
  float activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);
  const float16_t activation_min_f16 =
      HeliaFloat16ActivationBound(activation_min);
  const float16_t activation_max_f16 =
      HeliaFloat16ActivationBound(activation_max);
  arm_cmsis_nn_status status = ARM_CMSIS_NN_ARG_ERROR;
  if (data->broadcast_class == HeliaBroadcastClass::kSameShape) {
    status = arm_elementwise_div_f16(
        tflite::micro::GetTensorData<float16_t>(input1),
        tflite::micro::GetTensorData<float16_t>(input2),
        tflite::micro::GetTensorData<float16_t>(output), activation_min_f16,
        activation_max_f16, tflite::micro::GetTensorShape(output).FlatSize());
  } else if (HeliaBroadcastNeedsWalk(data->broadcast_class)) {
    const cmsis_nn_dims input1_dims =
        HeliaNhwcDims(tflite::micro::GetTensorShape(input1));
    const cmsis_nn_dims input2_dims =
        HeliaNhwcDims(tflite::micro::GetTensorShape(input2));
    const cmsis_nn_dims output_dims =
        HeliaNhwcDims(tflite::micro::GetTensorShape(output));
    status = arm_elementwise_div_broadcast_f16(
        tflite::micro::GetTensorData<float16_t>(input1), &input1_dims,
        tflite::micro::GetTensorData<float16_t>(input2), &input2_dims,
        tflite::micro::GetTensorData<float16_t>(output), &output_dims,
        activation_min_f16, activation_max_f16);
  }
  if (status != ARM_CMSIS_NN_SUCCESS) {
    MicroPrintf("Float16 DIV: heliaCORE kernel failed (%d).",
                static_cast<int>(status));
    return kTfLiteError;
  }
  return kTfLiteOk;
#else
  (void)params;
  (void)data;
  (void)input1;
  (void)input2;
  (void)output;
  return kTfLiteError;
#endif
}

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteDivParams* params, const OpDataDiv* data,
                           const TfLiteEvalTensor* input1,
                           const TfLiteEvalTensor* input2,
                           TfLiteEvalTensor* output) {
  tflite::micro::ArithmeticParams op_params = {};

#define TF_LITE_DIV(type, opname, dtype)                         \
  type::opname(op_params, tflite::micro::GetTensorShape(input1), \
               tflite::micro::GetTensorData<dtype>(input1),      \
               tflite::micro::GetTensorShape(input2),            \
               tflite::micro::GetTensorData<dtype>(input2),      \
               tflite::micro::GetTensorShape(output),            \
               tflite::micro::GetTensorData<dtype>(output))

  if (input1->type == kTfLiteInt8 && input2->type == kTfLiteInt8 &&
      output->type == kTfLiteInt8) {
    SetActivationParams(data->output_activation_min,
                        data->output_activation_max, &op_params);
    op_params.input1_offset = -data->input1_zero_point;
    op_params.input2_offset = -data->input2_zero_point;
    op_params.output_offset = data->output_zero_point;
    op_params.output_multiplier = data->output_multiplier;
    op_params.output_shift = data->output_shift;

    bool requires_broadcast = reference_ops::ProcessBroadcastShapes(
        tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorShape(input2), &op_params);

    if (requires_broadcast) {
      TF_LITE_DIV(reference_ops, BroadcastDivSlow, int8_t);
    } else {
      TF_LITE_DIV(reference_ops, Div, int8_t);
    }
#undef TF_LITE_DIV
  } else {
    MicroPrintf("Unsupported combination of input and output types in DIV.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus DivEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = static_cast<TfLiteDivParams*>(node->builtin_data);
  TFLITE_DCHECK(node->user_data != nullptr);
  auto* data = static_cast<OpDataDiv*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  if (output->type == kTfLiteFloat32) {
#if ARM_NN_ENABLE_F32
    bool handled = false;
    TF_LITE_ENSURE_OK(context, EvalDivFloat32Helia(params, data, input1, input2,
                                                   output, &handled));
    if (handled) {
      return kTfLiteOk;
    }
#endif
    EvalDiv<float>(context, node, params, data, input1, input2, output);
  } else if (output->type == kTfLiteFloat16) {
    return EvalDivFloat16(params, data, input1, input2, output);
  } else if (output->type == kTfLiteInt32) {
    EvalDiv<int32_t>(context, node, params, data, input1, input2, output);
  } else if (output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_OK(context, EvalQuantized(context, node, params, data,
                                             input1, input2, output));
  } else {
    MicroPrintf(
        "DIV only supports FLOAT32, FLOAT16, INT32, quantized INT8 "
        "now, got type %s (%d).",
        TfLiteTypeGetName(output->type), output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_DIV() {
  return tflite::micro::RegisterOp(DivInit, DivPrepare, DivEval);
}

}  // namespace micro
}  // namespace tflite
