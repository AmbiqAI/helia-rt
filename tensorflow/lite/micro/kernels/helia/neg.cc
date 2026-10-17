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

#include "tensorflow/lite/micro/kernels/internal/reference/neg.h"

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/helia/helia_float_common.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace micro {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

// float16 has no reference path, so it is rejected here on builds without
// ARM_NN_ENABLE_F16. arm_nn_neg_f16 writes as many elements as the input
// holds, so a float16 output must have the input's shape.
TfLiteStatus NegPrepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  if (output == nullptr) {
    micro_context->DeallocateTempTfLiteTensor(input);
    return kTfLiteError;
  }
  const TfLiteType type = input->type;
  const bool same_shape = TfLiteIntArrayEqual(input->dims, output->dims);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  TF_LITE_ENSURE_MSG(context,
                     type != kTfLiteFloat16 || kHeliaFloat16Enabled,
                     "Float16 NEG requires ARM_NN_ENABLE_F16.");
  TF_LITE_ENSURE_MSG(context, type != kTfLiteFloat16 || same_shape,
                     "Float16 NEG requires matching input and output shapes.");
  return kTfLiteOk;
}

// Runs arm_nn_neg_f32/f16, a sign-bit toggle like reference_ops::Negate.
template <typename T>
TfLiteStatus EvalHeliaNeg(TfLiteContext* context, const TfLiteEvalTensor* input,
                          TfLiteEvalTensor* output,
                          arm_cmsis_nn_status (*entry)(const T*, T*, int32_t),
                          const char* entry_name) {
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  const int count = ElementCount(*input->dims);
  if (count == 0) {
    return kTfLiteOk;
  }
  const arm_cmsis_nn_status status =
      entry(tflite::micro::GetTensorData<T>(input),
            tflite::micro::GetTensorData<T>(output), count);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    MicroPrintf("NEG: %s failed (%d).", entry_name, static_cast<int>(status));
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus NegEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  switch (input->type) {
    // TODO(wangtz): handle for kTfLiteInt8
    case kTfLiteFloat32:
#if ARM_NN_ENABLE_F32
      return EvalHeliaNeg<float>(context, input, output, arm_nn_neg_f32,
                                 "arm_nn_neg_f32");
#else
      reference_ops::Negate(tflite::micro::GetTensorShape(input),
                            tflite::micro::GetTensorData<float>(input),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<float>(output));
      break;
#endif
#if ARM_NN_ENABLE_F16
    case kTfLiteFloat16:
      TF_LITE_ENSURE_EQ(context, ElementCount(*input->dims),
                        ElementCount(*output->dims));
      return EvalHeliaNeg<float16_t>(context, input, output, arm_nn_neg_f16,
                                     "arm_nn_neg_f16");
#endif
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_NEG() {
  return tflite::micro::RegisterOp(nullptr, NegPrepare, NegEval);
}

}  // namespace micro
}  // namespace tflite
