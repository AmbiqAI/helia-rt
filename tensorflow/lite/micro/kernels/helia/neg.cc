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
#include "tensorflow/lite/micro/kernels/helia/helia_element_count.h"
#include "tensorflow/lite/micro/kernels/helia/helia_float_common.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

// float16 has no reference path, so it is rejected here on builds without
// ARM_NN_ENABLE_F16, and needs an output of the input's shape, as the other
// float16 elementwise kernels do.
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
  TfLiteStatus status = kTfLiteOk;
  if (input->type == kTfLiteFloat16) {
    if (!kHeliaFloat16Enabled) {
      MicroPrintf("Float16 NEG requires ARM_NN_ENABLE_F16.");
      status = kTfLiteError;
    } else {
      int32_t count;
      status = HeliaCheckedSameShapeElementCount(context, input->dims,
                                                 output->dims, &count);
    }
  }
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return status;
}

// Runs arm_nn_neg_f32/f16, a sign-bit toggle like reference_ops::Negate. The
// entry writes as many elements as it reads, so the output must hold the
// input's count.
template <typename T>
TfLiteStatus EvalHeliaNeg(TfLiteContext* context, const TfLiteEvalTensor* input,
                          TfLiteEvalTensor* output,
                          arm_cmsis_nn_status (*entry)(const T*, T*, int32_t),
                          const char* entry_name) {
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  int32_t count;
  TF_LITE_ENSURE_OK(context, HeliaCheckedMatchingElementCount(
                                 context, input->dims, output->dims, &count));
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
    case kTfLiteFloat16: {
      int32_t count;
      TF_LITE_ENSURE_OK(context, HeliaCheckedSameShapeElementCount(
                                     context, input->dims, output->dims,
                                     &count));
      return EvalHeliaNeg<float16_t>(context, input, output, arm_nn_neg_f16,
                                     "arm_nn_neg_f16");
    }
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
