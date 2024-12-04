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

#include "tensorflow/lite/kernels/internal/reference/leaky_relu.h"

#include "Include/arm_nnsupportfunctions.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/leaky_relu.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

void leaky_relu_s8(
    const LeakyReluOpData& data,
    const RuntimeShape& input_shape,
    const int8_t* input_data,
    const RuntimeShape& output_shape,
    int8_t* output_data
  ) {

  const int32_t input_offset = data.input_zero_point;
  const int32_t output_offset = data.output_zero_point;
  const int32_t output_multiplier_alpha = data.output_multiplier_alpha;
  const int32_t output_shift_alpha = data.output_shift_alpha;
  const int32_t output_multiplier_identity = data.output_multiplier_identity;
  const int32_t output_shift_identity = data.output_shift_identity;

  int32_t flat_size = MatchingFlatSize(input_shape, output_shape);
  const int32_t quantized_min = std::numeric_limits<int8_t>::min();
  const int32_t quantized_max = std::numeric_limits<int8_t>::max();

#if defined(ARM_MATH_MVEI)
  // Perform 4 operations in parallel
  uint32_t blkCnt = (flat_size + 3) / 4;

  #ifdef CMSIS_NN_USE_SINGLE_ROUNDING
    const int32_t right_shift_alpha = MIN(-1, output_shift_alpha);
    const int32_t left_shift_alpha = output_shift_alpha - right_shift_alpha;
    const int32_t right_shift_identity = MIN(-1, output_shift_identity);
    const int32_t left_shift_identity = output_shift_identity - right_shift_identity;
  #else
    const int32_t left_shift_alpha = LEFT_SHIFT(output_shift_alpha);
    const int32_t right_shift_alpha = -RIGHT_SHIFT(output_shift_alpha);
    const int32_t left_shift_identity = LEFT_SHIFT(output_shift_identity);
    const int32_t right_shift_identity = -RIGHT_SHIFT(output_shift_identity);
  #endif

  while (blkCnt > 0U) {
    mve_pred16_t val_pred = vctp32q((uint32_t)flat_size);
    // Load values in 32-bit registers
    int32x4_t res = vldrbq_z_s32(input_data, val_pred);
    // Subtract the input offset
    res = vsubq_s32(res, vdupq_n_s32(input_offset));
    // For values < 0, apply alpha otherwise apply identity
    mve_pred16_t alpha_pred = vcmpltq_n_s32(res, 0);
    int32x4_t left_shift_ident_dup = vdupq_n_s32(left_shift_identity);
    int32x4_t left_shift_dup = vdupq_m_n_s32(left_shift_ident_dup, left_shift_alpha, alpha_pred);
    int32x4_t right_shift_ident_dup = vdupq_n_s32(right_shift_identity);
    int32x4_t right_shift_dup = vdupq_m_n_s32(right_shift_ident_dup, right_shift_alpha, alpha_pred);
    int32x4_t mult_ident_dup = vdupq_n_s32(output_multiplier_identity);
    int32x4_t mult_dup = vdupq_m_n_s32(mult_ident_dup, output_multiplier_alpha, alpha_pred);
  #ifdef CMSIS_NN_USE_SINGLE_ROUNDING
    res = vqdmulhq_s32(vshlq_s32(res, left_shift_dup), mult_dup);
    res = vrshlq_s32(res, right_shift_dup);
  #else
    res = vqrdmulhq_s32(vshlq_s32(res, left_shift_dup), mult_dup);
    int32x4_t fixup = vshrq_n_s32(vandq_s32(res, right_shift_dup), 31);
    int32x4_t fixed_up_dividend = vqaddq_s32(res, fixup);
    res = vrshlq_s32(fixed_up_dividend, right_shift_dup);
  #endif
    // Add the output offset
    res = vaddq_n_s32(res, output_offset);
    // Clamp the result
    res = vmaxq_s32(res, vdupq_n_s32(quantized_min));
    res = vminq_s32(res, vdupq_n_s32(quantized_max));
    // Store the result
    vstrbq_p_s32(output_data, res, val_pred);
    // Increment pointers
    input_data += 4;
    output_data += 4;
    blkCnt -= 1;
    flat_size -= 4;
  }

  // while (blkCnt > 0U) {
  //   mve_pred16_t val_pred = vctp32q((uint32_t)flat_size);
  //   // Load values in 32bit registers
  //   res = vldrbq_z_s32(input_data, val_pred);
  //   // Subtract the input offset
  //   res = vsubq_s32(res, vdupq_n_s32(input_offset));
  //   // For values < 0, apply the leaky relu
  //   mve_pred16_t neg_pred = vcmpltq_n_s32(res, 0);
  //   res = arm_requantize_mve_pred(res, output_multiplier_alpha, output_shift_alpha, neg_pred);
  //   // For values >= 0, apply the identity
  //   mve_pred16_t pos_pred = vcmpgeq_n_s32(res, 0);
  //   res = arm_requantize_mve_pred(res, output_multiplier_identity, output_shift_identity, pos_pred);
  //   // Add the output offset
  //   res = vaddq_n_s32(res, output_offset);
  //   // Clamp the result
  //   res = vmaxq_s32(res, vdupq_n_s32(quantized_min));
  //   res = vminq_s32(res, vdupq_n_s32(quantized_max));
  //   // Store the result
  //   vstrbq_p_s32(output_data, res, val_pred);
  //   // Increment pointers
  //   input_data += 4;
  //   output_data += 4;
  //   blkCnt -= 1;
  //   flat_size -= 4;
  // }
#else

  int32_t val;
  for (int i = 0; i < flat_size; ++i) {
    const int32_t input = input_data[i] - op_params.input_offset;
    if (input >= 0) {
      val = arm_nn_requantize(input, op_params.output_multiplier_identity, op_params.output_shift_identity);
    } else {
      val = arm_nn_requantize(input, op_params.output_multiplier_alpha, op_params.output_shift_alpha);
    }
    val += op_params.output_offset;
    val = std::min(quantized_max, std::max(quantized_min, val));
    output_data[i] = static_cast<int8_t>(val);
  }

#endif

}


template <typename T>
void QuantizeLeakyRelu(const LeakyReluOpData& data,
                       const TfLiteEvalTensor* input,
                       TfLiteEvalTensor* output) {
  LeakyReluParams op_params = {};

  op_params.input_offset = data.input_zero_point;
  op_params.output_offset = data.output_zero_point;
  op_params.output_multiplier_alpha = data.output_multiplier_alpha;
  op_params.output_shift_alpha = data.output_shift_alpha;
  op_params.output_multiplier_identity = data.output_multiplier_identity;
  op_params.output_shift_identity = data.output_shift_identity;
  reference_ops::QuantizeLeakyRelu(op_params,
                                   tflite::micro::GetTensorShape(input),
                                   tflite::micro::GetTensorData<T>(input),
                                   tflite::micro::GetTensorShape(output),
                                   tflite::micro::GetTensorData<T>(output));
}

void* LeakyReluInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(LeakyReluOpData));
}

TfLiteStatus LeakyReluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  const LeakyReluOpData& data = *static_cast<LeakyReluOpData*>(node->user_data);

  switch (input->type) {
    case kTfLiteFloat32: {
      LeakyReluParams op_params = {};
      const auto* params =
          static_cast<TfLiteLeakyReluParams*>(node->builtin_data);

      op_params.alpha = params->alpha;
      reference_ops::LeakyRelu(op_params, tflite::micro::GetTensorShape(input),
                               tflite::micro::GetTensorData<float>(input),
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteInt8: {
      leaky_relu_s8(
        data,
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output)
      );
      // QuantizeLeakyRelu<int8_t>(data, input, output);
      return kTfLiteOk;
    } break;
    case kTfLiteInt16: {
      QuantizeLeakyRelu<int16_t>(data, input, output);
      return kTfLiteOk;
    } break;
    default:
      MicroPrintf("Only float32, int8 are supported by LEAKY_RELU, got %s.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }

  return kTfLiteError;
}

TFLMRegistration Register_LEAKY_RELU() {
  return tflite::micro::RegisterOp(LeakyReluInit, LeakyReluPrepare,
                                   LeakyReluEval);
}

}  // namespace tflite
