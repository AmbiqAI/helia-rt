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
#include "tensorflow/lite/micro/kernels/internal/reference/strided_slice.h"

#include <cstdint>
#include <cstring>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/helia/helia_float_common.h"
#include "tensorflow/lite/micro/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/strided_slice.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {

namespace {

// Fills the 4-D heliaCORE operands. Each output dim counts the elements the
// reference loop visits on that axis, so a shrunk axis keeps its position.
// Returns false when some axis visits nothing, i.e. the output is empty.
bool ArmPopulateCommonParams(const tflite::StridedSliceParams& op_params,
                             const RuntimeShape& unextended_input_shape,
                             cmsis_nn_dims* const input_dims,
                             cmsis_nn_dims* const output_dims,
                             cmsis_nn_dims* const begin_dims,
                             cmsis_nn_dims* const stride_dims) {
  using ::tflite::strided_slice::StridedSliceEndForAxis;
  using ::tflite::strided_slice::StridedSliceStartForAxis;

  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  const RuntimeShape input_shape =
      RuntimeShape::ExtendedShape(4, unextended_input_shape);
  tflite::StridedSliceParams params_copy = op_params;
  ::tflite::strided_slice::StridedSlicePadIndices(&params_copy, 4);

  int32_t* const in[4] = {&input_dims->n, &input_dims->h, &input_dims->w,
                          &input_dims->c};
  int32_t* const out[4] = {&output_dims->n, &output_dims->h,
                           &output_dims->w, &output_dims->c};
  int32_t* const begin[4] = {&begin_dims->n, &begin_dims->h, &begin_dims->w,
                             &begin_dims->c};
  int32_t* const stride[4] = {&stride_dims->n, &stride_dims->h,
                              &stride_dims->w, &stride_dims->c};
  for (int axis = 0; axis < 4; ++axis) {
    const int32_t start =
        StridedSliceStartForAxis(params_copy, input_shape, axis);
    const int32_t stop =
        StridedSliceEndForAxis(params_copy, input_shape, axis, start);
    const int32_t step = params_copy.strides[axis];
    const int32_t count = step > 0 ? (stop - start + step - 1) / step
                                   : (start - stop - step - 1) / -step;
    if (count <= 0) {
      return false;
    }
    *in[axis] = input_shape.Dims(axis);
    *out[axis] = count;
    *begin[axis] = start;
    *stride[axis] = step;
  }
  return true;
}

template <typename T>
TfLiteStatus EvalArm(const tflite::StridedSliceParams& op_params,
                     const TfLiteEvalTensor* input, TfLiteEvalTensor* output,
                     arm_cmsis_nn_status (*slice)(
                         const T*, T*, const cmsis_nn_dims* const,
                         const cmsis_nn_dims* const,
                         const cmsis_nn_dims* const,
                         const cmsis_nn_dims* const),
                     const char* entry) {
  cmsis_nn_dims input_dims;
  cmsis_nn_dims output_dims;
  cmsis_nn_dims begin_dims;
  cmsis_nn_dims stride_dims;
  if (!ArmPopulateCommonParams(op_params, tflite::micro::GetTensorShape(input),
                               &input_dims, &output_dims, &begin_dims,
                               &stride_dims)) {
    return kTfLiteOk;
  }
  const arm_cmsis_nn_status status =
      slice(tflite::micro::GetTensorData<T>(input),
            tflite::micro::GetTensorData<T>(output), &input_dims, &begin_dims,
            &stride_dims, &output_dims);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    MicroPrintf("STRIDED_SLICE: %s failed (%d).", entry,
                static_cast<int>(status));
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus StridedSliceHeliaPrepare(TfLiteContext* context,
                                      TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, StridedSlicePrepare(context, node));
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kStridedSliceOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  const bool supported = kHeliaFloat16Enabled || output->type != kTfLiteFloat16;
  micro_context->DeallocateTempTfLiteTensor(output);
  TF_LITE_ENSURE_MSG(context, supported,
                     "Float16 STRIDED_SLICE requires the heliaCORE fp16 API.");
  return kTfLiteOk;
}

TfLiteStatus StridedSliceEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const StridedSliceParams& op_params =
      *(static_cast<const StridedSliceParams*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kStridedSliceInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kStridedSliceOutputTensor);

  switch (output->type) {
    case kTfLiteFloat32:
#if ARM_NN_ENABLE_F32
      return EvalArm<float32_t>(op_params, input, output,
                                arm_strided_slice_f32, "arm_strided_slice_f32");
#else
      reference_ops::StridedSlice(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
#endif
#if ARM_NN_ENABLE_F16
    case kTfLiteFloat16:
      return EvalArm<float16_t>(op_params, input, output,
                                arm_strided_slice_f16, "arm_strided_slice_f16");
#endif
    case kTfLiteInt8:
      return EvalArm<int8_t>(op_params, input, output, arm_strided_slice_s8,
                             "arm_strided_slice_s8");
    case kTfLiteInt16:
      return EvalArm<int16_t>(op_params, input, output, arm_strided_slice_s16,
                              "arm_strided_slice_s16");
    case kTfLiteInt32:
      return EvalArm<int32_t>(op_params, input, output, arm_strided_slice_s32,
                              "arm_strided_slice_s32");
    case kTfLiteBool:
      reference_ops::StridedSlice(
        op_params,
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<bool>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<bool>(output)
      );
      break;
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_STRIDED_SLICE() {
  return tflite::micro::RegisterOp(StridedSliceInit, StridedSliceHeliaPrepare,
                                   StridedSliceEval);
}

}  // namespace micro
}  // namespace tflite
