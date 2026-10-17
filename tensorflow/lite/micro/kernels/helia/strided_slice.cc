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
    // start and stop lie in [-1, dim], so the span fits int32_t; the stride
    // magnitude of INT32_MIN needs uint32_t.
    const int32_t span = step > 0 ? stop - start : start - stop;
    if (span <= 0) {
      return false;
    }
    const uint32_t magnitude = step > 0 ? static_cast<uint32_t>(step)
                                        : 0u - static_cast<uint32_t>(step);
    const uint32_t count = (static_cast<uint32_t>(span) - 1u) / magnitude + 1u;
    *in[axis] = input_shape.Dims(axis);
    *out[axis] = static_cast<int32_t>(count);
    *begin[axis] = start;
    // heliaCORE forms begin + stride * count in int32_t. An axis that takes one
    // element never steps, so give it stride 1; a longer one has |stride| <
    // dim, which keeps that sum under 2 * dim, in range because Eval rejects
    // axes above 2^30.
    *stride[axis] = count == 1u ? 1 : step;
  }
  return true;
}

// Number of elements the slice takes, or of the tensor; saturates above
// INT32_MAX so a product of absurd dims cannot wrap into a match.
template <typename D>
uint64_t SaturatingProduct(const D* dims, int count) {
  uint64_t elements = 1;
  for (int i = 0; i < count; ++i) {
    if (dims[i] <= 0) {
      return 0;
    }
    elements *= static_cast<uint64_t>(dims[i]);
    if (elements > static_cast<uint64_t>(INT32_MAX)) {
      return static_cast<uint64_t>(INT32_MAX) + 1;
    }
  }
  return elements;
}

template <typename T>
TfLiteStatus EvalArm(const TfLiteEvalTensor* input, TfLiteEvalTensor* output,
                     const cmsis_nn_dims& input_dims,
                     const cmsis_nn_dims& output_dims,
                     const cmsis_nn_dims& begin_dims,
                     const cmsis_nn_dims& stride_dims,
                     arm_cmsis_nn_status (*slice)(
                         const T*, T*, const cmsis_nn_dims* const,
                         const cmsis_nn_dims* const,
                         const cmsis_nn_dims* const,
                         const cmsis_nn_dims* const),
                     const char* entry) {
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

TfLiteStatus StridedSliceEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const StridedSliceParams& op_params =
      *(static_cast<const StridedSliceParams*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kStridedSliceInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kStridedSliceOutputTensor);

  // The copy reads input elements of the output's type.
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  switch (output->type) {
    case kTfLiteFloat32:
    case kTfLiteInt8:
    case kTfLiteInt16:
    case kTfLiteInt32:
    case kTfLiteBool:
      break;
    default:
      MicroPrintf("Type %s (%d) not supported.",
                  TfLiteTypeGetName(output->type), output->type);
      return kTfLiteError;
  }

  // heliaCORE's offsets and the reference's indices are int32_t: each axis
  // must stay at most 2^30 so begin + stride * count fits, and the input
  // below 2^31 elements so element offsets fit.
  constexpr int32_t kMaxAxis = int32_t{1} << 30;
  for (int i = 0; i < input->dims->size; ++i) {
    if (input->dims->data[i] < 0 || input->dims->data[i] > kMaxAxis) {
      MicroPrintf("STRIDED_SLICE: input axis %d is negative or above 2^30.", i);
      return kTfLiteError;
    }
  }
  if (SaturatingProduct(input->dims->data, input->dims->size) >
      static_cast<uint64_t>(INT32_MAX)) {
    MicroPrintf("STRIDED_SLICE: the input exceeds INT32_MAX elements.");
    return kTfLiteError;
  }

  // Prepare checks only the output shape, which it derives with a float ceil
  // that rounds above 2^24 elements on an axis, so check for every type that
  // the output holds exactly the elements the slice writes. A slice that takes
  // nothing writes nothing, as in the reference, even when a shrunk axis
  // leaves a one-element output.
  cmsis_nn_dims input_dims = {};
  cmsis_nn_dims output_dims = {};
  cmsis_nn_dims begin_dims = {};
  cmsis_nn_dims stride_dims = {};
  if (!ArmPopulateCommonParams(op_params, tflite::micro::GetTensorShape(input),
                               &input_dims, &output_dims, &begin_dims,
                               &stride_dims)) {
    return kTfLiteOk;
  }
  const int32_t counts[4] = {output_dims.n, output_dims.h, output_dims.w,
                             output_dims.c};
  // The input bound keeps the slice below 2^31 elements, so neither count
  // saturates and they compare exactly.
  if (SaturatingProduct(counts, 4) !=
      SaturatingProduct(output->dims->data, output->dims->size)) {
    MicroPrintf("STRIDED_SLICE: the output size does not match the slice.");
    return kTfLiteError;
  }

  switch (output->type) {
    case kTfLiteFloat32:
      reference_ops::StridedSlice(
        op_params,
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<float>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<float>(output)
      );
      break;
    case kTfLiteInt8:
      return EvalArm<int8_t>(input, output, input_dims, output_dims,
                             begin_dims, stride_dims, arm_strided_slice_s8,
                             "arm_strided_slice_s8");
    case kTfLiteInt16:
      return EvalArm<int16_t>(input, output, input_dims, output_dims,
                              begin_dims, stride_dims, arm_strided_slice_s16,
                              "arm_strided_slice_s16");
    case kTfLiteInt32:
      reference_ops::StridedSlice(
        op_params,
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int32_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int32_t>(output)
      );
      break;
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
  return tflite::micro::RegisterOp(StridedSliceInit, StridedSlicePrepare, StridedSliceEval);
}

}  // namespace micro
}  // namespace tflite
