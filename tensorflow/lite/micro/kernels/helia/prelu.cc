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

#include "tensorflow/lite/micro/kernels/internal/reference/prelu.h"

#include <cstdint>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/helia/helia_broadcast.h"
#include "tensorflow/lite/micro/kernels/helia/helia_float_common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/prelu.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

// PreluPrepare() writes the PreluParams at the start of user_data.
struct OpDataPrelu {
  PreluParams params;
  // heliaCORE PRELU takes 4-D NHWC dims, an output shaped like the input, and
  // an alpha that broadcasts into the input.
  bool helia_shapes;
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

bool HeliaPreluShapes(const RuntimeShape& input_shape,
                      const RuntimeShape& alpha_shape,
                      const RuntimeShape& output_shape) {
  if (input_shape.DimensionsCount() > 4 || alpha_shape.DimensionsCount() > 4 ||
      output_shape.DimensionsCount() > 4 || output_shape.FlatSize() == 0) {
    return false;
  }
  const cmsis_nn_dims input = HeliaNhwcDims(input_shape);
  const cmsis_nn_dims alpha = HeliaNhwcDims(alpha_shape);
  const cmsis_nn_dims output = HeliaNhwcDims(output_shape);
  const int32_t in[4] = {input.n, input.h, input.w, input.c};
  const int32_t al[4] = {alpha.n, alpha.h, alpha.w, alpha.c};
  const int32_t out[4] = {output.n, output.h, output.w, output.c};
  for (int i = 0; i < 4; ++i) {
    if (out[i] != in[i] || al[i] < 1 || (al[i] != in[i] && al[i] != 1)) {
      return false;
    }
  }
  return true;
}

TfLiteStatus HeliaPreluPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, PreluPrepare(context, node));
  OpDataPrelu* data = static_cast<OpDataPrelu*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  ScopedTempTensor input_guard(micro_context, input);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* alpha = micro_context->AllocateTempInputTensor(node, 1);
  ScopedTempTensor alpha_guard(micro_context, alpha);
  TF_LITE_ENSURE(context, alpha != nullptr);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  ScopedTempTensor output_guard(micro_context, output);
  TF_LITE_ENSURE(context, output != nullptr);

  // Eval writes through the input's type, and the float kernels read alpha as
  // it too.
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);
  if (input->type == kTfLiteFloat32) {
    TF_LITE_ENSURE_TYPES_EQ(context, alpha->type, kTfLiteFloat32);
  }

  data->helia_shapes =
      HeliaPreluShapes(GetTensorShape(input), GetTensorShape(alpha),
                       GetTensorShape(output));

  // float16 has no reference kernel. An empty output is a no-op at Eval.
  if (input->type == kTfLiteFloat16) {
    TF_LITE_ENSURE_MSG(context, kHeliaFloat16Enabled,
                       "Float16 PRELU requires the fp16 API.");
    TF_LITE_ENSURE_TYPES_EQ(context, alpha->type, kTfLiteFloat16);
    TF_LITE_ENSURE_MSG(context,
                       data->helia_shapes || NumElements(output) == 0,
                       "Float16 PRELU does not support these shapes.");
  }
  return kTfLiteOk;
}

// Callers pass only shapes HeliaPreluShapes() accepted.
struct HeliaPreluDims {
  HeliaPreluDims(const TfLiteEvalTensor* input,
                 const TfLiteEvalTensor* alpha,
                 const TfLiteEvalTensor* output)
      : input(HeliaNhwcDims(tflite::micro::GetTensorShape(input))),
        alpha(HeliaNhwcDims(tflite::micro::GetTensorShape(alpha))),
        output(HeliaNhwcDims(tflite::micro::GetTensorShape(output))) {}
  const cmsis_nn_dims input;
  const cmsis_nn_dims alpha;
  const cmsis_nn_dims output;
};

}  // namespace

void* PreluInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataPrelu));
}

TfLiteStatus PreluEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataPrelu& data = *(static_cast<const OpDataPrelu*>(node->user_data));
  const PreluParams& params = data.params;

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* alpha = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  switch (input->type) {
#if ARM_NN_ENABLE_F16
    case kTfLiteFloat16: {
      if (tflite::micro::GetTensorShape(output).FlatSize() == 0) {
        return kTfLiteOk;
      }
      const HeliaPreluDims dims(input, alpha, output);
      const arm_cmsis_nn_status status = arm_prelu_f16(
          &dims.input, tflite::micro::GetTensorData<float16_t>(input),
          &dims.alpha, tflite::micro::GetTensorData<float16_t>(alpha),
          &dims.output, tflite::micro::GetTensorData<float16_t>(output));
      if (status != ARM_CMSIS_NN_SUCCESS) {
        MicroPrintf("PRELU: arm_prelu_f16 failed (%d).",
                    static_cast<int>(status));
        return kTfLiteError;
      }
      return kTfLiteOk;
    } break;
#endif
    case kTfLiteFloat32: {
#if ARM_NN_ENABLE_F32
      if (data.helia_shapes) {
        const HeliaPreluDims dims(input, alpha, output);
        const arm_cmsis_nn_status status = arm_prelu_f32(
            &dims.input, tflite::micro::GetTensorData<float>(input),
            &dims.alpha, tflite::micro::GetTensorData<float>(alpha),
            &dims.output, tflite::micro::GetTensorData<float>(output));
        if (status != ARM_CMSIS_NN_SUCCESS) {
          MicroPrintf("PRELU: arm_prelu_f32 failed (%d).",
                      static_cast<int>(status));
          return kTfLiteError;
        }
        return kTfLiteOk;
      }
#endif
      BroadcastPrelu4DSlowFloat(tflite::micro::GetTensorShape(input),
                                tflite::micro::GetTensorData<float>(input),
                                tflite::micro::GetTensorShape(alpha),
                                tflite::micro::GetTensorData<float>(alpha),
                                tflite::micro::GetTensorShape(output),
                                tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteInt8: {
      if (data.helia_shapes) {
        const HeliaPreluDims dims(input, alpha, output);
        const arm_cmsis_nn_status status = arm_prelu_s8(
            &dims.input, tflite::micro::GetTensorData<int8_t>(input),
            &dims.alpha, tflite::micro::GetTensorData<int8_t>(alpha),
            params.input_offset, params.alpha_offset, params.output_offset,
            params.output_multiplier_1, params.output_shift_1,
            params.output_multiplier_2, params.output_shift_2, &dims.output,
            tflite::micro::GetTensorData<int8_t>(output));
        if (status != ARM_CMSIS_NN_SUCCESS) {
          MicroPrintf("PRELU: arm_prelu_s8 failed (%d).",
                      static_cast<int>(status));
          return kTfLiteError;
        }
        return kTfLiteOk;
      }
      reference_ops::BroadcastPrelu4DSlow(
          params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(alpha),
          tflite::micro::GetTensorData<int8_t>(alpha),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteInt16: {
      reference_ops::BroadcastPrelu4DSlow(
          params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorShape(alpha),
          tflite::micro::GetTensorData<int8_t>(alpha),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
      return kTfLiteOk;
    } break;
    default:
      MicroPrintf("Input type '%s' is not supported.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

TFLMRegistration Register_PRELU() {
  return tflite::micro::RegisterOp(PreluInit, HeliaPreluPrepare, PreluEval);
}

}  // namespace micro
}  // namespace tflite
