/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

// Integer version of unidirectional sequence LSTM. Only the standard LSTM
// (defined in the keras LSTM layer, e.g., no peephole etc.) is supported here.
// Currently used by the 8 bits activation case only, except for fallbacks.

#include <algorithm>
#include <limits>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/lstm_eval.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/kernels/micro_tensor_utils.h"
namespace tflite {

namespace {

struct OpData {
  OpDataLSTM params_ref;                 // Used for fallback implementation
  cmsis_nn_lstm_params params_cmsis_nn;  // Used for  CMSIS-NN implementation
#if ARM_NN_ENABLE_F32
  cmsis_nn_lstm_params_f32 params_cmsis_nn_f32;
#endif
#if ARM_NN_ENABLE_F16
  cmsis_nn_lstm_params_f16 params_cmsis_nn_f16;
#endif
};

#if ARM_NN_ENABLE_F32 || ARM_NN_ENABLE_F16
bool IsSupportedFloatLstm(const LSTMKernelContents& content,
                          TfLiteType activation_type) {
  // The Helia float kernel implements the standard (Keras) LSTM with all four
  // gates. Every input-to-gate and recurrent-to-gate weight matrix (including
  // the input gate, i.e. non-CIFG) and every gate bias must be present and
  // match the activation type.
  auto is_present = [&](int i) {
    const TfLiteEvalTensor* tensor = content.GetInternalTensor(i);
    return tensor != nullptr && tensor->type == activation_type;
  };
  for (int i = kLstmInputToInputWeightsTensor;
       i <= kLstmRecurrentToOutputWeightsTensor; ++i) {
    if (!is_present(i)) return false;
  }
  for (int i = kLstmInputGateBiasTensor; i <= kLstmOutputGateBiasTensor; ++i) {
    if (!is_present(i)) return false;
  }
  // The Helia float kernel does not implement optional LSTM variants
  // (peephole connections, projection, or layer normalization); reject any
  // model that provides those tensors.
  for (int i = kLstmCellToInputWeightsTensor;
       i <= kLstmCellToOutputWeightsTensor; ++i) {
    if (content.GetInternalTensor(i) != nullptr) return false;
  }
  for (int i = kLstmProjectionWeightsTensor; i <= kLstmProjectionBiasTensor;
       ++i) {
    if (content.GetInternalTensor(i) != nullptr) return false;
  }
  for (int i = kLstmInputLayerNormCoefficientsTensor;
       i <= kLstmOutputLayerNormCoefficientsTensor; ++i) {
    if (content.GetInternalTensor(i) != nullptr) return false;
  }
  return true;
}

#if NS_CMSIS_NN_VERSION < 7029000
template <typename T>
bool IsZeroInitialState(const TfLiteEvalTensor* hidden,
                        const TfLiteEvalTensor* cell, int count) {
  const T* hidden_data = tflite::micro::GetTensorData<T>(hidden);
  const T* cell_data = tflite::micro::GetTensorData<T>(cell);
  for (int i = 0; i < count; ++i) {
    if (hidden_data[i] != static_cast<T>(0) ||
        cell_data[i] != static_cast<T>(0)) {
      return false;
    }
  }
  return true;
}
#endif
#endif

// Number of state-sized scratch buffers required by the optimized quantized
// LSTM path. From ns-cmsis-nn v7.29.0 onwards the cell state lives in the
// TFLite variable tensor, so only the two temporary gate buffers are needed.
#if NS_CMSIS_NN_VERSION >= 7029000
constexpr size_t kCmsisNnQuantizedScratchBuffers = 2;
#else
constexpr size_t kCmsisNnQuantizedScratchBuffers = 3;
#endif

LSTMBuffers<int16_t> CMSIS_NN_CreateLSTMBuffers(TfLiteContext* context,
                                                const int* buffer_indices) {
  LSTMBuffers<int16_t> buffers = {};
  buffers.buffer0 = reinterpret_cast<int16_t*>(
      context->GetScratchBuffer(context, buffer_indices[0]));
  buffers.buffer1 = reinterpret_cast<int16_t*>(
      context->GetScratchBuffer(context, buffer_indices[1]));
#if NS_CMSIS_NN_VERSION < 7029000
  buffers.buffer2 = reinterpret_cast<int16_t*>(
      context->GetScratchBuffer(context, buffer_indices[2]));
#endif

  return buffers;
}

void CMSIS_NN_VectorSum(int32_t* kernel_sum, const int32_t size1,
                        const int32_t size2, const int8_t* weights,
                        const int32_t offset, const int32_t* biases) {
  arm_vector_sum_s8(kernel_sum, size1, size2, weights, offset, 0, biases);
}

void CMSIS_NN_VectorSum(int64_t* kernel_sum, const int32_t size1,
                        const int32_t size2, const int8_t* weights,
                        const int32_t offset, const int64_t* biases) {
  arm_vector_sum_s8_s64(kernel_sum, size1, size2, weights, offset, biases);
}

template <typename BiasType>
TfLiteStatus CMSIS_NN_PortOpData(TfLiteContext* context, OpDataLSTM* params_ref,
                                 const LSTMKernelContents& kernel_content,
                                 cmsis_nn_lstm_params* params_cmsis_nn) {
  // Unwrap pointers
  const BiasType* input_gate_bias =
      tflite::micro::GetOptionalTensorData<BiasType>(
          kernel_content.GetInternalTensor(tflite::kLstmInputGateBiasTensor));
  const BiasType* forget_gate_bias =
      tflite::micro::GetOptionalTensorData<BiasType>(
          kernel_content.GetInternalTensor(tflite::kLstmForgetGateBiasTensor));
  const BiasType* cell_gate_bias =
      tflite::micro::GetOptionalTensorData<BiasType>(
          kernel_content.GetInternalTensor(tflite::kLstmCellGateBiasTensor));
  const BiasType* output_gate_bias =
      tflite::micro::GetOptionalTensorData<BiasType>(
          kernel_content.GetInternalTensor(tflite::kLstmOutputGateBiasTensor));

  const int8_t* input_to_input_weights =
      tflite::micro::GetOptionalTensorData<int8_t>(
          kernel_content.GetInternalTensor(
              tflite::kLstmInputToInputWeightsTensor));
  const int8_t* input_to_forget_weights =
      tflite::micro::GetOptionalTensorData<int8_t>(
          kernel_content.GetInternalTensor(
              tflite::kLstmInputToForgetWeightsTensor));
  const int8_t* input_to_cell_weights =
      tflite::micro::GetOptionalTensorData<int8_t>(
          kernel_content.GetInternalTensor(
              tflite::kLstmInputToCellWeightsTensor));
  const int8_t* input_to_output_weights =
      tflite::micro::GetOptionalTensorData<int8_t>(
          kernel_content.GetInternalTensor(
              tflite::kLstmInputToOutputWeightsTensor));

  const int8_t* recurrent_to_input_weights =
      tflite::micro::GetOptionalTensorData<int8_t>(
          kernel_content.GetInternalTensor(
              tflite::kLstmRecurrentToInputWeightsTensor));
  const int8_t* recurrent_to_forget_weights =
      tflite::micro::GetOptionalTensorData<int8_t>(
          kernel_content.GetInternalTensor(
              tflite::kLstmRecurrentToForgetWeightsTensor));
  const int8_t* recurrent_to_cell_weights =
      tflite::micro::GetOptionalTensorData<int8_t>(
          kernel_content.GetInternalTensor(
              tflite::kLstmRecurrentToCellWeightsTensor));
  const int8_t* recurrent_to_output_weights =
      tflite::micro::GetOptionalTensorData<int8_t>(
          kernel_content.GetInternalTensor(
              tflite::kLstmRecurrentToOutputWeightsTensor));

  int32_t size_data = params_ref->size_info.input_dimension;
  int32_t size_hidden = params_ref->size_info.state_dimension;

  BiasType* input_data_kernel_sum{
      static_cast<BiasType*>(context->AllocatePersistentBuffer(
          context, size_hidden * sizeof(BiasType)))};
  BiasType* forget_data_kernel_sum{
      static_cast<BiasType*>(context->AllocatePersistentBuffer(
          context, size_hidden * sizeof(BiasType)))};
  BiasType* cell_data_kernel_sum{
      static_cast<BiasType*>(context->AllocatePersistentBuffer(
          context, size_hidden * sizeof(BiasType)))};
  BiasType* output_data_kernel_sum{
      static_cast<BiasType*>(context->AllocatePersistentBuffer(
          context, size_hidden * sizeof(BiasType)))};

  BiasType* input_hidden_kernel_sum{
      static_cast<BiasType*>(context->AllocatePersistentBuffer(
          context, size_hidden * sizeof(BiasType)))};
  BiasType* forget_hidden_kernel_sum{
      static_cast<BiasType*>(context->AllocatePersistentBuffer(
          context, size_hidden * sizeof(BiasType)))};
  BiasType* cell_hidden_kernel_sum = {
      static_cast<BiasType*>(context->AllocatePersistentBuffer(
          context, size_hidden * sizeof(BiasType)))};
  BiasType* output_hidden_kernel_sum = {
      static_cast<BiasType*>(context->AllocatePersistentBuffer(
          context, size_hidden * sizeof(BiasType)))};

  // Compute effective biases
  CMSIS_NN_VectorSum(
      input_data_kernel_sum, size_data, size_hidden, input_to_input_weights,
      params_ref->input_gate_parameters.input_fc_params.input_offset,
      input_gate_bias);

  CMSIS_NN_VectorSum(
      forget_data_kernel_sum, size_data, size_hidden, input_to_forget_weights,
      params_ref->forget_gate_parameters.input_fc_params.input_offset,
      forget_gate_bias);

  CMSIS_NN_VectorSum(
      cell_data_kernel_sum, size_data, size_hidden, input_to_cell_weights,
      params_ref->cell_gate_parameters.input_fc_params.input_offset,
      cell_gate_bias);

  CMSIS_NN_VectorSum(
      output_data_kernel_sum, size_data, size_hidden, input_to_output_weights,
      params_ref->output_gate_parameters.input_fc_params.input_offset,
      output_gate_bias);

  CMSIS_NN_VectorSum(
      input_hidden_kernel_sum, size_hidden, size_hidden,
      recurrent_to_input_weights,
      -params_ref->inter_gate_parameters.output_mul_params.output_offset,
      nullptr);

  CMSIS_NN_VectorSum(
      forget_hidden_kernel_sum, size_hidden, size_hidden,
      recurrent_to_forget_weights,
      -params_ref->inter_gate_parameters.output_mul_params.output_offset,
      nullptr);

  CMSIS_NN_VectorSum(
      cell_hidden_kernel_sum, size_hidden, size_hidden,
      recurrent_to_cell_weights,
      -params_ref->inter_gate_parameters.output_mul_params.output_offset,
      nullptr);

  CMSIS_NN_VectorSum(
      output_hidden_kernel_sum, size_hidden, size_hidden,
      recurrent_to_output_weights,
      -params_ref->inter_gate_parameters.output_mul_params.output_offset,
      nullptr);

  // Create input gate parameters
  cmsis_nn_lstm_gate gate_input{
      params_ref->input_gate_parameters.input_fc_params.output_multiplier,
      params_ref->input_gate_parameters.input_fc_params.output_shift,
      input_to_input_weights,
      input_data_kernel_sum,
      params_ref->input_gate_parameters.recurrent_fc_params.output_multiplier,
      params_ref->input_gate_parameters.recurrent_fc_params.output_shift,
      recurrent_to_input_weights,
      input_hidden_kernel_sum,
      input_gate_bias,
      ARM_SIGMOID};

  // Create forget gate parameters
  cmsis_nn_lstm_gate gate_forget{
      params_ref->forget_gate_parameters.input_fc_params.output_multiplier,
      params_ref->forget_gate_parameters.input_fc_params.output_shift,
      input_to_forget_weights,
      forget_data_kernel_sum,
      params_ref->forget_gate_parameters.recurrent_fc_params.output_multiplier,
      params_ref->forget_gate_parameters.recurrent_fc_params.output_shift,
      recurrent_to_forget_weights,
      forget_hidden_kernel_sum,
      forget_gate_bias,
      ARM_SIGMOID};

  auto cell_gate_nonlinear_type =
      (params_ref->cell_gate_nonlinear_type == kTfLiteActTanh) ? ARM_TANH
                                                               : ARM_SIGMOID;
  // Create cell gate parameters
  cmsis_nn_lstm_gate gate_cell{
      params_ref->cell_gate_parameters.input_fc_params.output_multiplier,
      params_ref->cell_gate_parameters.input_fc_params.output_shift,
      input_to_cell_weights,
      cell_data_kernel_sum,
      params_ref->cell_gate_parameters.recurrent_fc_params.output_multiplier,
      params_ref->cell_gate_parameters.recurrent_fc_params.output_shift,
      recurrent_to_cell_weights,
      cell_hidden_kernel_sum,
      cell_gate_bias,
      cell_gate_nonlinear_type};

  // Create output gate parameters
  cmsis_nn_lstm_gate gate_output{
      params_ref->output_gate_parameters.input_fc_params.output_multiplier,
      params_ref->output_gate_parameters.input_fc_params.output_shift,
      input_to_output_weights,
      output_data_kernel_sum,
      params_ref->output_gate_parameters.recurrent_fc_params.output_multiplier,
      params_ref->output_gate_parameters.recurrent_fc_params.output_shift,
      recurrent_to_output_weights,
      output_hidden_kernel_sum,
      output_gate_bias,
      ARM_SIGMOID};

  // Create the complete lstm data struct
  *params_cmsis_nn = {
      params_ref->size_info.time_major,
      params_ref->size_info.batch_size,
      params_ref->size_info.time_steps,
      params_ref->size_info.input_dimension,
      params_ref->size_info.state_dimension,
      params_ref->forget_gate_parameters.input_fc_params.input_offset,
      params_ref->inter_gate_parameters.forget_cell_mul_params
          .output_multiplier,
      params_ref->inter_gate_parameters.forget_cell_mul_params.output_shift,
      params_ref->inter_gate_parameters.input_mul_params.output_multiplier,
      params_ref->inter_gate_parameters.input_mul_params.output_shift,
      params_ref->cell_state_info.quantized_cell_clip,
      params_ref->cell_state_info.cell_state_scale_power,
      params_ref->inter_gate_parameters.output_mul_params.output_multiplier,
      params_ref->inter_gate_parameters.output_mul_params.output_shift,
      params_ref->inter_gate_parameters.output_mul_params.output_offset,
      gate_forget,
      gate_input,
      gate_cell,
      gate_output};

  return kTfLiteOk;
}

TfLiteStatus CMSIS_NN_EvalInteger8x8_16Lstm(
    const OpData& op_data, LSTMKernelContents& kernel_content,
    const LSTMBuffers<int16_t>& buffers) {
  TFLITE_DCHECK(
      kernel_content.GetInternalTensor(tflite::kLstmInputTensor)->dims->size >=
          2 &&
      kernel_content.GetInternalTensor(tflite::kLstmInputTensor)->dims->size <=
          3);

  const int8_t* input = tflite::micro::GetOptionalTensorData<int8_t>(
      kernel_content.GetInternalTensor(tflite::kLstmInputTensor));
  int8_t* output =
      tflite::micro::GetTensorData<int8_t>(kernel_content.output_tensor);

  // Create lstm buffer struct.
  cmsis_nn_lstm_context cmsis_buffers = {};
  cmsis_buffers.temp1 = reinterpret_cast<int16_t*>(buffers.buffer0);
  cmsis_buffers.temp2 = reinterpret_cast<int16_t*>(buffers.buffer1);
#if NS_CMSIS_NN_VERSION >= 7029000
  // A non-NULL hidden_state makes arm_lstm_unidirectional_s8 stateful: it
  // consumes the incoming hidden/cell state from the TFLite variable tensors
  // and writes the final state back into them.
  cmsis_buffers.cell_state =
      tflite::micro::GetTensorData<int16_t>(kernel_content.CellStateTensor());
  cmsis_buffers.hidden_state =
      tflite::micro::GetTensorData<int8_t>(kernel_content.HiddenStateTensor());
#else
  // Older ns-cmsis-nn revisions do not persist the final hidden state, so run
  // the kernel stateless (NULL hidden_state) with a scratch cell state.
  cmsis_buffers.cell_state = reinterpret_cast<int16_t*>(buffers.buffer2);
#endif

  if (arm_lstm_unidirectional_s8(input, output, &op_data.params_cmsis_nn,
                                 &cmsis_buffers) != ARM_CMSIS_NN_SUCCESS) {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus CMSIS_NN_EvalInteger16x8_16Lstm(
    const OpData& op_data, LSTMKernelContents& kernel_content,
    const LSTMBuffers<int16_t>& buffers) {
  TFLITE_DCHECK(
      kernel_content.GetInternalTensor(tflite::kLstmInputTensor)->dims->size >=
          2 &&
      kernel_content.GetInternalTensor(tflite::kLstmInputTensor)->dims->size <=
          3);

  const int16_t* input = tflite::micro::GetOptionalTensorData<int16_t>(
      kernel_content.GetInternalTensor(tflite::kLstmInputTensor));
  int16_t* output =
      tflite::micro::GetTensorData<int16_t>(kernel_content.output_tensor);

  // Create lstm buffer struct.
  cmsis_nn_lstm_context cmsis_buffers = {};
  cmsis_buffers.temp1 = reinterpret_cast<int16_t*>(buffers.buffer0);
  cmsis_buffers.temp2 = reinterpret_cast<int16_t*>(buffers.buffer1);
#if NS_CMSIS_NN_VERSION >= 7029000
  // A non-NULL hidden_state makes arm_lstm_unidirectional_s16 stateful: it
  // consumes the incoming hidden/cell state from the TFLite variable tensors
  // and writes the final state back into them.
  cmsis_buffers.cell_state =
      tflite::micro::GetTensorData<int16_t>(kernel_content.CellStateTensor());
  cmsis_buffers.hidden_state =
      tflite::micro::GetTensorData<int16_t>(kernel_content.HiddenStateTensor());
#else
  // Older ns-cmsis-nn revisions do not persist the final hidden state, so run
  // the kernel stateless (NULL hidden_state) with a scratch cell state.
  cmsis_buffers.cell_state = reinterpret_cast<int16_t*>(buffers.buffer2);
#endif

  if (arm_lstm_unidirectional_s16(input, output, &op_data.params_cmsis_nn,
                                  &cmsis_buffers) != ARM_CMSIS_NN_SUCCESS) {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

#if ARM_NN_ENABLE_F32
TfLiteStatus CMSIS_NN_EvalFloat32(
    OpData* op_data, LSTMKernelContents& content,
    const LSTMBuffers<float>& buffers) {
  const auto activation = op_data->params_ref.cell_gate_nonlinear_type;
  if ((activation != kTfLiteActTanh && activation != kTfLiteActSigmoid) ||
      !IsSupportedFloatLstm(content, kTfLiteFloat32)) {
    return kTfLiteError;
  }
#if NS_CMSIS_NN_VERSION < 7029000
  const int state_count = op_data->params_ref.size_info.batch_size *
                          op_data->params_ref.size_info.state_dimension;
  if (!IsZeroInitialState<float>(content.HiddenStateTensor(),
                                 content.CellStateTensor(), state_count)) {
    return kTfLiteError;
  }
#endif
  const auto gate = [&](int iw, int hw, int b, arm_nn_activation_type_flt a) {
    return cmsis_nn_lstm_gate_f32{
        tflite::micro::GetTensorData<float>(content.GetInternalTensor(iw)),
        tflite::micro::GetTensorData<float>(content.GetInternalTensor(hw)),
        tflite::micro::GetOptionalTensorData<float>(
            content.GetInternalTensor(b)),
        a};
  };
  op_data->params_cmsis_nn_f32 = {
      op_data->params_ref.size_info.time_major,
      op_data->params_ref.size_info.batch_size,
      op_data->params_ref.size_info.time_steps,
      op_data->params_ref.size_info.input_dimension,
      op_data->params_ref.size_info.state_dimension,
      op_data->params_ref.cell_state_info.cell_clip,
      gate(kLstmInputToForgetWeightsTensor, kLstmRecurrentToForgetWeightsTensor,
           kLstmForgetGateBiasTensor, ARM_NN_FLT_ACT_SIGMOID),
      gate(kLstmInputToInputWeightsTensor, kLstmRecurrentToInputWeightsTensor,
           kLstmInputGateBiasTensor, ARM_NN_FLT_ACT_SIGMOID),
      gate(kLstmInputToCellWeightsTensor, kLstmRecurrentToCellWeightsTensor,
           kLstmCellGateBiasTensor,
           activation == kTfLiteActTanh ? ARM_NN_FLT_ACT_TANH
                                        : ARM_NN_FLT_ACT_SIGMOID),
      gate(kLstmInputToOutputWeightsTensor, kLstmRecurrentToOutputWeightsTensor,
           kLstmOutputGateBiasTensor, ARM_NN_FLT_ACT_SIGMOID)};
  cmsis_nn_lstm_context_f32 cmsis_buffers = {};
  cmsis_buffers.temp1 = buffers.buffer0;
  cmsis_buffers.temp2 = buffers.buffer1;
#if NS_CMSIS_NN_VERSION >= 7029000
  cmsis_buffers.cell_state =
      tflite::micro::GetTensorData<float>(content.CellStateTensor());
  cmsis_buffers.hidden_state =
      tflite::micro::GetTensorData<float>(content.HiddenStateTensor());
#else
  cmsis_buffers.cell_state = buffers.buffer2;
#endif
  const auto* input =
      tflite::micro::GetTensorData<float>(
          content.GetInternalTensor(kLstmInputTensor));
  auto* output = tflite::micro::GetTensorData<float>(content.output_tensor);
  if (arm_lstm_unidirectional_f32(input, output, &op_data->params_cmsis_nn_f32,
                                  &cmsis_buffers) != ARM_CMSIS_NN_SUCCESS) {
    return kTfLiteError;
  }
#if NS_CMSIS_NN_VERSION < 7029000
  auto* hidden = tflite::micro::GetTensorData<float>(content.HiddenStateTensor());
  auto* cell = tflite::micro::GetTensorData<float>(content.CellStateTensor());
  const int batch = op_data->params_ref.size_info.batch_size;
  const int time = op_data->params_ref.size_info.time_steps;
  const int hidden_size = op_data->params_ref.size_info.state_dimension;
  for (int b = 0; b < batch; ++b) {
    const int src = op_data->params_ref.size_info.time_major
                        ? (time - 1) * batch * hidden_size + b * hidden_size
                        : (b * time + time - 1) * hidden_size;
    for (int h = 0; h < hidden_size; ++h) {
      hidden[b * hidden_size + h] = output[src + h];
      cell[b * hidden_size + h] = buffers.buffer2[b * hidden_size + h];
    }
  }
#endif
  return kTfLiteOk;
}
#endif

#if ARM_NN_ENABLE_F16
TfLiteStatus CMSIS_NN_EvalFloat16(
    OpData* op_data, LSTMKernelContents& content,
    const LSTMBuffers<float16_t>& buffers) {
  const auto activation = op_data->params_ref.cell_gate_nonlinear_type;
  if ((activation != kTfLiteActTanh && activation != kTfLiteActSigmoid) ||
      !IsSupportedFloatLstm(content, kTfLiteFloat16)) {
    return kTfLiteError;
  }
#if NS_CMSIS_NN_VERSION < 7029000
  const int state_count = op_data->params_ref.size_info.batch_size *
                          op_data->params_ref.size_info.state_dimension;
  if (!IsZeroInitialState<float16_t>(content.HiddenStateTensor(),
                                     content.CellStateTensor(), state_count)) {
    MicroPrintf(
        "Helia float16 LSTM requires zero initial hidden/cell state; "
        "stateful float16 LSTM requires ns-cmsis-nn v7.29.0 or newer.");
    return kTfLiteError;
  }
#endif
  const auto gate = [&](int iw, int hw, int b, arm_nn_activation_type_flt a) {
    return cmsis_nn_lstm_gate_f16{
        tflite::micro::GetTensorData<float16_t>(content.GetInternalTensor(iw)),
        tflite::micro::GetTensorData<float16_t>(content.GetInternalTensor(hw)),
        tflite::micro::GetOptionalTensorData<float16_t>(
            content.GetInternalTensor(b)),
        a};
  };
  op_data->params_cmsis_nn_f16 = {
      op_data->params_ref.size_info.time_major,
      op_data->params_ref.size_info.batch_size,
      op_data->params_ref.size_info.time_steps,
      op_data->params_ref.size_info.input_dimension,
      op_data->params_ref.size_info.state_dimension,
      static_cast<float16_t>(op_data->params_ref.cell_state_info.cell_clip),
      gate(kLstmInputToForgetWeightsTensor, kLstmRecurrentToForgetWeightsTensor,
           kLstmForgetGateBiasTensor, ARM_NN_FLT_ACT_SIGMOID),
      gate(kLstmInputToInputWeightsTensor, kLstmRecurrentToInputWeightsTensor,
           kLstmInputGateBiasTensor, ARM_NN_FLT_ACT_SIGMOID),
      gate(kLstmInputToCellWeightsTensor, kLstmRecurrentToCellWeightsTensor,
           kLstmCellGateBiasTensor,
           activation == kTfLiteActTanh ? ARM_NN_FLT_ACT_TANH
                                        : ARM_NN_FLT_ACT_SIGMOID),
      gate(kLstmInputToOutputWeightsTensor, kLstmRecurrentToOutputWeightsTensor,
           kLstmOutputGateBiasTensor, ARM_NN_FLT_ACT_SIGMOID)};
  cmsis_nn_lstm_context_f16 cmsis_buffers = {};
  cmsis_buffers.temp1 = buffers.buffer0;
  cmsis_buffers.temp2 = buffers.buffer1;
#if NS_CMSIS_NN_VERSION >= 7029000
  cmsis_buffers.cell_state =
      tflite::micro::GetTensorData<float16_t>(content.CellStateTensor());
  cmsis_buffers.hidden_state =
      tflite::micro::GetTensorData<float16_t>(content.HiddenStateTensor());
#else
  cmsis_buffers.cell_state = buffers.buffer2;
#endif
  const auto* input =
      tflite::micro::GetTensorData<float16_t>(
          content.GetInternalTensor(kLstmInputTensor));
  auto* output =
      tflite::micro::GetTensorData<float16_t>(content.output_tensor);
  if (arm_lstm_unidirectional_f16(input, output, &op_data->params_cmsis_nn_f16,
                                  &cmsis_buffers) != ARM_CMSIS_NN_SUCCESS) {
    return kTfLiteError;
  }
#if NS_CMSIS_NN_VERSION < 7029000
  auto* hidden =
      tflite::micro::GetTensorData<float16_t>(content.HiddenStateTensor());
  auto* cell =
      tflite::micro::GetTensorData<float16_t>(content.CellStateTensor());
  const int batch = op_data->params_ref.size_info.batch_size;
  const int time = op_data->params_ref.size_info.time_steps;
  const int hidden_size = op_data->params_ref.size_info.state_dimension;
  for (int b = 0; b < batch; ++b) {
    const int src = op_data->params_ref.size_info.time_major
                        ? (time - 1) * batch * hidden_size + b * hidden_size
                        : (b * time + time - 1) * hidden_size;
    for (int h = 0; h < hidden_size; ++h) {
      hidden[b * hidden_size + h] = output[src + h];
      cell[b * hidden_size + h] = buffers.buffer2[b * hidden_size + h];
    }
  }
#endif
  return kTfLiteOk;
}
#endif

/*Kernel functions*/
void* UnidirectionalSequenceLstmInit(TfLiteContext* context, const char* buffer,
                                     size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus UnidirectionalSequenceLstmPrepare(TfLiteContext* context,
                                               TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 24);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  TFLITE_DCHECK(node->user_data != nullptr);

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  OpDataLSTM* op_data_lstm = &op_data->params_ref;

  const auto* builtin_data =
      static_cast<TfLiteUnidirectionalSequenceLSTMParams*>(node->builtin_data);
  // All TempTfLiteTensors will be deallocated through the destructor.
  LstmTensors lstm_tensors(context, node);
  TF_LITE_ENSURE_OK(context, lstm_tensors.ValidateTensorStatus(context));

  op_data_lstm->cell_gate_nonlinear_type = builtin_data->activation;
  op_data_lstm->size_info =
      CreateLstmSizeInfo(builtin_data->time_major,
                         lstm_tensors.GetInternalTensor(kLstmInputTensor)->dims,
                         lstm_tensors.HiddenStateTensor()->dims);

  const TfLiteTensor* input = lstm_tensors.GetInternalTensor(kLstmInputTensor);
  const auto activation_type = input->type;

  TF_LITE_ENSURE_OK(context, ValidateTensorSize(context, lstm_tensors,
                                                op_data_lstm->size_info));

  auto cell_state_type =
      lstm_tensors.GetInternalTensor(kLstmCellStateTensor)->type;
  if (cell_state_type == kTfLiteFloat32
#if ARM_NN_ENABLE_F16
      || cell_state_type == kTfLiteFloat16
#endif
  ) {
    op_data_lstm->cell_state_info =
        CreateLstmCellStateInfoFloat(builtin_data->cell_clip);
    TF_LITE_ENSURE_OK(context, PrepareGateParametersFloat(context, lstm_tensors,
                                                          op_data_lstm));
  } else if (cell_state_type == kTfLiteInt16) {
    op_data_lstm->cell_state_info = CreateLstmCellStateInfo(
        lstm_tensors.CellStateTensor()->params.scale, builtin_data->cell_clip);
    TF_LITE_ENSURE_OK(context, PrepareGateParametersInteger(
                                   context, lstm_tensors, op_data_lstm));
  } else {
    MicroPrintf(
        "Cell state type %s (%d) not supported. The quantized Unidirectional "
        "Sequence LSTM Op only support int16 cell state",
        TfLiteTypeGetName(cell_state_type), cell_state_type);
    return kTfLiteError;
  }

  size_t number_of_buffers;
  if (activation_type == kTfLiteInt8 && cell_state_type == kTfLiteInt16) {
    auto kernel_content = CreateLSTMKernelContent(context, node);
    number_of_buffers = kCmsisNnQuantizedScratchBuffers;
    CMSIS_NN_PortOpData<int32_t>(context, op_data_lstm, kernel_content,
                                 &op_data->params_cmsis_nn);
  } else if (activation_type == kTfLiteInt16 &&
             cell_state_type == kTfLiteInt16) {
    auto kernel_content = CreateLSTMKernelContent(context, node);
    number_of_buffers = kCmsisNnQuantizedScratchBuffers;
    CMSIS_NN_PortOpData<int64_t>(context, op_data_lstm, kernel_content,
                                 &op_data->params_cmsis_nn);
  } else {
    number_of_buffers = 4;
  }

  for (size_t i = 0; i < number_of_buffers; i++) {
    TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                   context,
                                   op_data_lstm->size_info.batch_size *
                                       op_data_lstm->size_info.state_dimension *
                                       TfLiteTypeGetSize(cell_state_type),
                                   &(op_data_lstm->buffer_indices[i])));
  }

  return kTfLiteOk;
}

TfLiteStatus UnidirectionalSequenceLstmEval(TfLiteContext* context,
                                            TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  OpData& op_data = *reinterpret_cast<OpData*>(node->user_data);
  const OpDataLSTM& op_data_lstm = op_data.params_ref;

  auto kernel_content = CreateLSTMKernelContent(context, node);

  const auto activation_type =
      kernel_content.internal_tensors[kLstmInputTensor]->type;
  const auto weight_type =
      kernel_content.internal_tensors[kLstmInputToInputWeightsTensor]->type;

  switch (activation_type) {
    case kTfLiteFloat32: {
#if ARM_NN_ENABLE_F32
      LSTMBuffers<float> buffers =
          CreateLSTMBuffers<float>(context, op_data_lstm.buffer_indices);
      if (CMSIS_NN_EvalFloat32(&op_data, kernel_content, buffers) !=
          kTfLiteOk) {
        EvalLstm<float, float, float, float>(op_data_lstm, kernel_content,
                                             buffers);
      }
#else
      LSTMBuffers<float> buffers =
          CreateLSTMBuffers<float>(context, op_data_lstm.buffer_indices);
      EvalLstm<float, float, float, float>(op_data_lstm, kernel_content,
                                           buffers);
#endif
      break;
    }
    case kTfLiteFloat16: {
#if ARM_NN_ENABLE_F16
      LSTMBuffers<float16_t> buffers =
          CreateLSTMBuffers<float16_t>(context, op_data_lstm.buffer_indices);
      if (CMSIS_NN_EvalFloat16(&op_data, kernel_content, buffers) !=
          kTfLiteOk) {
        MicroPrintf("Unsupported float16 Unidirectional Sequence LSTM "
                    "configuration.");
        return kTfLiteError;
      }
#else
      MicroPrintf("Float16 LSTM support is disabled.");
      return kTfLiteError;
#endif
      break;
    }
    case kTfLiteInt8: {
      switch (weight_type) {
        case kTfLiteInt8: {
          // 8(activation)x8(weight)->16(cell) LSTM with 32 bits bias
          LSTMBuffers<int16_t> buffers =
              CMSIS_NN_CreateLSTMBuffers(context, op_data_lstm.buffer_indices);
          TF_LITE_ENSURE_OK(context, CMSIS_NN_EvalInteger8x8_16Lstm(
                                         op_data, kernel_content, buffers));
          break;
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(weight_type), activation_type);
          return kTfLiteError;
        }
      }
      break;
    }
    case kTfLiteInt16: {
      switch (weight_type) {
        case kTfLiteInt8: {
          // 16(activation)x8(weight)->16(cell) LSTM with 64 bits bias
          LSTMBuffers<int16_t> buffers =
              CMSIS_NN_CreateLSTMBuffers(context, op_data_lstm.buffer_indices);
          TF_LITE_ENSURE_OK(context, CMSIS_NN_EvalInteger16x8_16Lstm(
                                         op_data, kernel_content, buffers));
          break;
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(weight_type), weight_type);
          return kTfLiteError;
        }
      }
      break;
    }
    default: {
      MicroPrintf("Input type %s (%d) not supported.",
                  TfLiteTypeGetName(activation_type), activation_type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus UnidirectionalSequenceLstmEvalInt8(TfLiteContext* context,
                                                TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& op_data = *reinterpret_cast<const OpData*>(node->user_data);
  const OpDataLSTM& op_data_lstm = op_data.params_ref;
  auto kernel_content = CreateLSTMKernelContent(context, node);
  const auto activation_type =
      kernel_content.internal_tensors[kLstmInputTensor]->type;
  const auto weight_type =
      kernel_content.internal_tensors[kLstmInputToInputWeightsTensor]->type;

  TFLITE_DCHECK(weight_type == kTfLiteInt16 &&
                "Only int16 filter type supported.");

  if (activation_type == kTfLiteInt8) {
    LSTMBuffers<int16_t> buffers =
        CMSIS_NN_CreateLSTMBuffers(context, op_data_lstm.buffer_indices);

    return CMSIS_NN_EvalInteger8x8_16Lstm(op_data, kernel_content, buffers);
  } else {
    MicroPrintf("Input type %s (%d) not supported.",
                TfLiteTypeGetName(activation_type), activation_type);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus UnidirectionalSequenceLstmEvalInt16(TfLiteContext* context,
                                                 TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& op_data = *reinterpret_cast<const OpData*>(node->user_data);
  const OpDataLSTM& op_data_lstm = op_data.params_ref;
  auto kernel_content = CreateLSTMKernelContent(context, node);
  const auto activation_type =
      kernel_content.internal_tensors[kLstmInputTensor]->type;
  const auto weight_type =
      kernel_content.internal_tensors[kLstmInputToInputWeightsTensor]->type;

  TFLITE_DCHECK(weight_type == kTfLiteInt16 &&
                "Only int16 filter type supported.");

  if (activation_type == kTfLiteInt16) {
    LSTMBuffers<int16_t> buffers =
        CMSIS_NN_CreateLSTMBuffers(context, op_data_lstm.buffer_indices);

    return CMSIS_NN_EvalInteger16x8_16Lstm(op_data, kernel_content, buffers);
  } else {
    MicroPrintf("Input type %s (%d) not supported.",
                TfLiteTypeGetName(activation_type), activation_type);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM() {
  return tflite::micro::RegisterOp(UnidirectionalSequenceLstmInit,
                                   UnidirectionalSequenceLstmPrepare,
                                   UnidirectionalSequenceLstmEval);
}

TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM_INT8() {
  return tflite::micro::RegisterOp(UnidirectionalSequenceLstmInit,
                                   UnidirectionalSequenceLstmPrepare,
                                   UnidirectionalSequenceLstmEvalInt8);
}

TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM_INT16() {
  return tflite::micro::RegisterOp(UnidirectionalSequenceLstmInit,
                                   UnidirectionalSequenceLstmPrepare,
                                   UnidirectionalSequenceLstmEvalInt16);
}

}  // namespace tflite
