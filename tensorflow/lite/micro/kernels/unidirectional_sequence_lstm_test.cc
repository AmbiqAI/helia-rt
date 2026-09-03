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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/testdata/lstm_test_data.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

#if ARM_NN_ENABLE_F16
#include "arm_nnfunctions_flt.h"
#endif

// The stateful quantized-LSTM assertions below need the persistent-state
// contract, which upstream CMSIS-NN lacks and ns-cmsis-nn added in v7.28.0.
// Keep this threshold in sync with the quantized NS_CMSIS_NN_VERSION fences in
// kernels/helia/unidirectional_sequence_lstm.cc.
#if defined(HELIA)
#include "Include/arm_nn_types.h"  // NS_CMSIS_NN_VERSION
#endif

#if !defined(CMSIS_NN) ||                                  \
    (defined(HELIA) && defined(NS_CMSIS_NN_VERSION) &&     \
     NS_CMSIS_NN_VERSION >= 7028000)
#define TFLM_LSTM_QUANTIZED_STATEFUL 1
#endif

namespace tflite {
namespace testing {
namespace {

constexpr int kLstmMaxNumInputOutputTensors = 24 + 1;

// Set weights and biases to be const-tensors
void SetConstTensors(TfLiteTensor* tensors) {
  for (size_t i = 1; i < 9; i++) {
    // weights
    tensors[i].allocation_type = kTfLiteMmapRo;
  }
  for (size_t i = 12; i < 16; i++) {
    // biases
    tensors[i].allocation_type = kTfLiteMmapRo;
  }
}

// Validate the output result array with golden values
template <typename T>
void ValidateResultGoldens(const T* golden, const T* output_data,
                           const int output_len, const float tolerance) {
  for (int i = 0; i < output_len; ++i) {
    EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
}

// Golden values for a second invocation of the 2x3x2x2 model, i.e. the model
// continuing from the hidden/cell state left behind by the first invocation.
constexpr float kExpectedSecondOutput[] = {
    0.61399786f, 0.61399787f, 0.62385699f, 0.62385699f,
    0.62659836f, 0.62659836f, 0.33596584f, 0.33545765f,
    0.61567060f, 0.61567062f, 0.56908714f, 0.56908710f};
constexpr float kExpectedSecondHidden[] = {0.62659836f, 0.62659836f,
                                           0.56908714f, 0.56908710f};
constexpr float kExpectedSecondCell[] = {0.94111480f, 0.94111480f, 0.88564131f,
                                         0.88564121f};

#ifdef TFLM_LSTM_QUANTIZED_STATEFUL
// Goldens for an invocation seeded with a NON-ZERO cell state and a ZERO hidden
// state, i.e. hidden_state = {0,0,0,0} and
// cell_state = Get2X2LstmEvalCheckData().expected_cell_state.
//
// Why this case exists: every other test in this file leaves the incoming cell
// state effectively unobservable.  The 2x2 fixture deliberately uses "negative
// large weights for forget gate to make it really forget"
// ({-10,-10,-20,-20} in lstm_test_data.cc), and every other test enters the
// kernel with a non-zero hidden state.  With that hidden state the t=0 forget
// gate collapses to ~1.7e-7, so the incoming cell state can shift any asserted
// element by at most 1.73e-3 -- 5.8x INSIDE the 1e-2 tolerance.  A kernel that
// reads garbage for the incoming cell state, or ignores it entirely, still
// passes.  (Verified by mutation: zeroing the incoming cell state while leaving
// the write-back intact passes every other test in this file.)
//
// Seeding hidden = 0 instead pushes the t=0 forget gate to ~0.99 for batch 2
// (exactly [0.99005, 0.99990]), so the incoming cell state propagates rather
// than being annihilated.  Measured deviation of these goldens from the
// zero-cell result, i.e. how far a kernel that ignores the incoming cell state
// would land off:
//
//   output  b2: t0 0.2724 / 0.2736   t1 0.1266 / 0.1265   t2 0.0603 / 0.0603
//   hidden  b2: 0.0603 / 0.0603
//   cell    b2: 0.0733 / 0.0733
//
// i.e. 6x to 27x outside the 1e-2 tolerance.  Batch 1 still forgets hard (its
// deltas are <= 0.0082), so batch 2 is what carries the signal here.
//
// These are float goldens, derived by evaluating the float LSTM recurrence with
// this seeding -- the same way the goldens above were derived, NOT by recording
// what the quantized kernel under test happens to produce.  The same derivation
// reproduces kExpectedSecondOutput/Hidden/Cell and the first-invocation goldens
// in lstm_test_data.cc to within 4.5e-9.
constexpr float kExpectedSeededCellOutput[] = {
    0.27273426f, 0.26885695f, 0.48181164f, 0.48182232f,
    0.58104594f, 0.58104600f, 0.27100885f, 0.27361716f,
    0.59545371f, 0.59545373f, 0.56088665f, 0.56088660f};
constexpr float kExpectedSeededCellHidden[] = {0.58104594f, 0.58104600f,
                                               0.56088665f, 0.56088660f};
constexpr float kExpectedSeededCellCell[] = {0.89835594f, 0.89835609f,
                                             0.87660349f, 0.87660337f};
constexpr float kZeroHiddenState[] = {0.0f, 0.0f, 0.0f, 0.0f};
#endif  // TFLM_LSTM_QUANTIZED_STATEFUL

// The second invocation feeds the (quantized) state of the first invocation
// back into the model, so its quantization error is amplified compared to a
// single invocation. The batch-two output of the very first time step is the
// most sensitive element: it sits in the steep region of the gate
// nonlinearities. Measured maximum deviations from the goldens above
// (host build, 2x3x2x2 model):
//
//   backend            int8      int16
//   HELIA              0.0027    0.0004
//   reference          0.0504    0.0538
//
// The loose bound is therefore driven by the reference kernel, not by HELIA,
// so the two backends get separate tolerances instead of sharing the widest
// one. For reference, a stateless kernel repeats the first-invocation output;
// its smallest violating element deviates by 0.0686 and its largest by 0.94,
// so 6e-2 still detects lost state, but only by ~14%. HELIA's 1e-2 keeps a
// ~3.7x margin over the measured error while catching much subtler state
// corruption.
//
// Which elements actually carry the discriminating signal matters here, because
// 6e-2 is close enough to the reset deltas that most of the vector is silent.
// Under a *total* state reset (i.e. a stateless kernel that just repeats the
// first-invocation result) the per-element deviation from the goldens above is:
//
//   output  b1:  t0 0.3494 / 0.3453   t1 0.1445 / 0.1445   t2 0.0465 / 0.0465
//           b2:  t0 0.3374 / 0.3355   t1 0.1468 / 0.1468   t2 0.0685 / 0.0685
//   hidden  b1:  0.0465 / 0.0465      b2:  0.0685 / 0.0685
//   cell    b1:  0.0437 / 0.0437      b2:  0.0824 / 0.0824
//
// So at 6e-2 the check is carried by the t0/t1 output elements of both batches
// (>= 0.1445), plus batch 2's t2 output and hidden state (0.0685) and batch 2's
// cell state (0.0824).  Batch 1's entire hidden state, batch 1's entire cell
// state and batch 1's final-timestep output all sit INSIDE 6e-2 and stay silent
// even under a complete state reset.  If the goldens or the model inputs are
// ever retuned, re-measure this table: shrinking the t0/t1 deltas would hollow
// the reference-backend check out without any test turning red.
#if defined(HELIA)
constexpr float kQuantizedSecondInvokeTolerance = 1e-2;
#else
constexpr float kQuantizedSecondInvokeTolerance = 6e-2;
#endif

// Reorder a [batch, time, depth] buffer into [time, batch, depth].
template <int batch_size, int time_steps, int depth>
void ToTimeMajor(const float* batch_major, float* time_major) {
  for (int b = 0; b < batch_size; ++b) {
    for (int t = 0; t < time_steps; ++t) {
      for (int d = 0; d < depth; ++d) {
        time_major[(t * batch_size + b) * depth + d] =
            batch_major[(b * time_steps + t) * depth + d];
      }
    }
  }
}

template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
void ValidateInvokeResult(
    LstmNodeContent<ActivationType, WeightType, BiasType, CellType, batch_size,
                    time_steps, input_dimension, state_dimension>&
        node_contents,
    const float* expected_output, const float* expected_hidden_state,
    const float* expected_cell_state, const float hidden_state_tolerance,
    const float cell_state_tolerance) {
  const auto& quantization_settings = node_contents.QuantizationSettings();

#ifdef TFLM_LSTM_QUANTIZED_STATEFUL
  float dequantized_hidden_state[batch_size * state_dimension] = {};
  Dequantize(node_contents.GetHiddenStateData(), batch_size * state_dimension,
             quantization_settings.hidden_state.scale,
             quantization_settings.hidden_state.zero_point,
             dequantized_hidden_state);
  ValidateResultGoldens(expected_hidden_state, dequantized_hidden_state,
                        batch_size * state_dimension, hidden_state_tolerance);

  float dequantized_cell_state[batch_size * state_dimension] = {};
  Dequantize(node_contents.GetCellStateData(), batch_size * state_dimension,
             quantization_settings.cell_state.scale,
             quantization_settings.cell_state.zero_point,
             dequantized_cell_state);
  ValidateResultGoldens(expected_cell_state, dequantized_cell_state,
                        batch_size * state_dimension, cell_state_tolerance);
#else
  (void)expected_hidden_state;
  (void)expected_cell_state;
  (void)cell_state_tolerance;
#endif

  float dequantized_output[batch_size * state_dimension * time_steps] = {};
  Dequantize(node_contents.GetOutputData(),
             batch_size * state_dimension * time_steps,
             quantization_settings.output.scale,
             quantization_settings.output.zero_point, dequantized_output);
  ValidateResultGoldens(expected_output, dequantized_output,
                        batch_size * state_dimension * time_steps,
                        hidden_state_tolerance);
}

template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
void TestUnidirectionalLSTMInteger(
    const LstmEvalCheckData<
        batch_size * time_steps * input_dimension, batch_size * state_dimension,
        batch_size * state_dimension * time_steps>& eval_check_data,
    const float hidden_state_tolerance, const float cell_state_tolerance,
    LstmNodeContent<ActivationType, WeightType, BiasType, CellType, batch_size,
                    time_steps, input_dimension, state_dimension>&
        node_contents) {
  SetConstTensors(node_contents.GetTensors());
  const TFLMRegistration registration = Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  auto buildin_data = node_contents.BuiltinData();
  micro::KernelRunner runner(
      registration, node_contents.GetTensors(), kLstmMaxNumInputOutputTensors,
      node_contents.KernelInputs(), node_contents.KernelOutputs(),
      reinterpret_cast<void*>(&buildin_data));
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  ValidateInvokeResult(node_contents, eval_check_data.expected_output,
                       eval_check_data.expected_hidden_state,
                       eval_check_data.expected_cell_state,
                       hidden_state_tolerance, cell_state_tolerance);

#ifdef TFLM_LSTM_QUANTIZED_STATEFUL
  // The hidden and cell state tensors are variable tensors, so a second
  // invocation must continue from the state left behind by the first one.
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  ValidateInvokeResult(node_contents, kExpectedSecondOutput,
                       kExpectedSecondHidden, kExpectedSecondCell,
                       kQuantizedSecondInvokeTolerance,
                       kQuantizedSecondInvokeTolerance);
#endif
}

#ifdef TFLM_LSTM_QUANTIZED_STATEFUL
// Same model, but with the input/output tensors laid out as
// [time, batch, depth] and time_major enabled in the builtin data.
template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
void TestUnidirectionalLSTMIntegerTimeMajor(
    const float* expected_output, const float* expected_hidden_state,
    const float* expected_cell_state, const float* expected_second_output,
    const float hidden_state_tolerance, const float cell_state_tolerance,
    LstmNodeContent<ActivationType, WeightType, BiasType, CellType, batch_size,
                    time_steps, input_dimension, state_dimension>&
        node_contents) {
  SetConstTensors(node_contents.GetTensors());
  TfLiteTensor* tensors = node_contents.GetTensors();
  // [time, batch, depth] shapes for the input and the output tensor.
  // IntArrayFromInts aliases these buffers rather than copying them, and the
  // tensors keep pointing at them after this function returns, so they need
  // static storage duration -- function locals would dangle.  The values are
  // template constants, so one instance per instantiation is correct.
  static int input_dims[4] = {3, time_steps, batch_size, input_dimension};
  static int output_dims[4] = {3, time_steps, batch_size, state_dimension};
  // Ask the node for its output tensor index instead of hard-coding 24.
  const int output_tensor_index = node_contents.KernelOutputs()->data[0];
  tensors[kLstmInputTensor].dims = IntArrayFromInts(input_dims);
  tensors[output_tensor_index].dims = IntArrayFromInts(output_dims);

  const TFLMRegistration registration = Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  auto buildin_data = node_contents.BuiltinData();
  buildin_data.time_major = true;
  micro::KernelRunner runner(
      registration, tensors, kLstmMaxNumInputOutputTensors,
      node_contents.KernelInputs(), node_contents.KernelOutputs(),
      reinterpret_cast<void*>(&buildin_data));
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  ValidateInvokeResult(node_contents, expected_output, expected_hidden_state,
                       expected_cell_state, hidden_state_tolerance,
                       cell_state_tolerance);

  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  ValidateInvokeResult(node_contents, expected_second_output,
                       kExpectedSecondHidden, kExpectedSecondCell,
                       kQuantizedSecondInvokeTolerance,
                       kQuantizedSecondInvokeTolerance);
}

// Run a single invocation on an already-seeded node and check output/hidden/cell
// against the supplied goldens.  Used by the seeded-initial-state tests, which
// assert on what the kernel READS rather than on what it writes back.
template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
void TestUnidirectionalLSTMSeededState(
    const float* expected_output, const float* expected_hidden_state,
    const float* expected_cell_state, const float hidden_state_tolerance,
    const float cell_state_tolerance,
    LstmNodeContent<ActivationType, WeightType, BiasType, CellType, batch_size,
                    time_steps, input_dimension, state_dimension>&
        node_contents) {
  SetConstTensors(node_contents.GetTensors());
  const TFLMRegistration registration = Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  auto buildin_data = node_contents.BuiltinData();
  micro::KernelRunner runner(
      registration, node_contents.GetTensors(), kLstmMaxNumInputOutputTensors,
      node_contents.KernelInputs(), node_contents.KernelOutputs(),
      reinterpret_cast<void*>(&buildin_data));
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  ValidateInvokeResult(node_contents, expected_output, expected_hidden_state,
                       expected_cell_state, hidden_state_tolerance,
                       cell_state_tolerance);
}
#endif  // TFLM_LSTM_QUANTIZED_STATEFUL

template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
void TestUnidirectionalLSTMFloat(
    const LstmEvalCheckData<
        batch_size * time_steps * input_dimension, batch_size * state_dimension,
        batch_size * state_dimension * time_steps>& eval_check_data,
    const float* expected_second_output,
    const float* expected_second_hidden_state,
    const float* expected_second_cell_state,
    const float hidden_state_tolerance, const float cell_state_tolerance,
    LstmNodeContent<float, float, float, float, batch_size, time_steps,
                    input_dimension, state_dimension>& node_contents) {
  SetConstTensors(node_contents.GetTensors());
  const TFLMRegistration registration = Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  auto buildin_data = node_contents.BuiltinData();
  micro::KernelRunner runner(
      registration, node_contents.GetTensors(), kLstmMaxNumInputOutputTensors,
      node_contents.KernelInputs(), node_contents.KernelOutputs(),
      reinterpret_cast<void*>(&buildin_data));
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  ValidateResultGoldens(eval_check_data.expected_hidden_state,
                        node_contents.GetHiddenStateData(),
                        batch_size * state_dimension, hidden_state_tolerance);
  ValidateResultGoldens(eval_check_data.expected_cell_state,
                        node_contents.GetCellStateData(),
                        batch_size * state_dimension, cell_state_tolerance);
  ValidateResultGoldens(eval_check_data.expected_output,
                        node_contents.GetOutputData(),
                        batch_size * state_dimension, hidden_state_tolerance);

  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  ValidateResultGoldens(expected_second_hidden_state,
                        node_contents.GetHiddenStateData(),
                        batch_size * state_dimension, hidden_state_tolerance);
  ValidateResultGoldens(expected_second_cell_state,
                        node_contents.GetCellStateData(),
                        batch_size * state_dimension, cell_state_tolerance);
  ValidateResultGoldens(expected_second_output, node_contents.GetOutputData(),
                        batch_size * state_dimension * time_steps,
                        hidden_state_tolerance);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

// TODO(b/230666079) enable below tests for xtensa when the xtensa
// kernel is reconciled with reference kernel
TEST(UnidirectionalSequenceLstmTest, TestUnidirectionalLSTMFloat) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<float, float, float, float, 2, 3, 2, 2>
      float_node_contents = tflite::testing::Create2x3x2X2FloatNodeContents(
          kernel_eval_data.input_data, kernel_eval_data.hidden_state);

  // Tolerance accommodates float32 accumulation-order differences between
  // reference and optimized (e.g. CMSIS-NN) LSTM implementations. The
  // reference kernel still matches golden well within this bound.
  const float tolerance = 1e-4;
  tflite::testing::TestUnidirectionalLSTMFloat(
      kernel_eval_data, tflite::testing::kExpectedSecondOutput,
      tflite::testing::kExpectedSecondHidden,
      tflite::testing::kExpectedSecondCell, tolerance, tolerance,
      float_node_contents);
}

TEST(UnidirectionalSequenceLstmTest, TestUnidirectionalLSTMInt8) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents(
          kernel_eval_data.input_data, kernel_eval_data.hidden_state);

  const float hidden_state_tolerance = 1e-2;
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-2;
  tflite::testing::TestUnidirectionalLSTMInteger(
      kernel_eval_data, hidden_state_tolerance, cell_state_tolerance,
      int8_node_contents);
}

TEST(UnidirectionalSequenceLstmTest, TestUnidirectionalLSTMInt16) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<int16_t, int8_t, int64_t, int16_t, 2, 3, 2,
                                   2>
      int16_node_contents = tflite::testing::Create2x3x2X2Int16NodeContents(
          kernel_eval_data.input_data, kernel_eval_data.hidden_state);

  const float hidden_state_tolerance = 1e-3;  // actually very close to 1e-4
  // cell state degrade due to integer overflow
  const float cell_state_tolerance = 1e-2;
  tflite::testing::TestUnidirectionalLSTMInteger(
      kernel_eval_data, hidden_state_tolerance, cell_state_tolerance,
      int16_node_contents);
}

#ifdef TFLM_LSTM_QUANTIZED_STATEFUL
// A model seeded with the state left behind by a first invocation must produce
// the second-invocation goldens on its very first invocation, i.e. the kernel
// consumes the incoming (non-zero) hidden and cell state.
TEST(UnidirectionalSequenceLstmTest, TestUnidirectionalLSTMInt8InitialState) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents(
          kernel_eval_data.input_data, kernel_eval_data.expected_hidden_state,
          kernel_eval_data.expected_cell_state);

  tflite::testing::SetConstTensors(int8_node_contents.GetTensors());
  const TFLMRegistration registration =
      tflite::Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  auto buildin_data = int8_node_contents.BuiltinData();
  tflite::micro::KernelRunner runner(
      registration, int8_node_contents.GetTensors(),
      tflite::testing::kLstmMaxNumInputOutputTensors,
      int8_node_contents.KernelInputs(), int8_node_contents.KernelOutputs(),
      reinterpret_cast<void*>(&buildin_data));
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  tflite::testing::ValidateInvokeResult(
      int8_node_contents, tflite::testing::kExpectedSecondOutput,
      tflite::testing::kExpectedSecondHidden,
      tflite::testing::kExpectedSecondCell,
      tflite::testing::kQuantizedSecondInvokeTolerance,
      tflite::testing::kQuantizedSecondInvokeTolerance);
}

TEST(UnidirectionalSequenceLstmTest, TestUnidirectionalLSTMInt16InitialState) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<int16_t, int8_t, int64_t, int16_t, 2, 3, 2,
                                   2>
      int16_node_contents = tflite::testing::Create2x3x2X2Int16NodeContents(
          kernel_eval_data.input_data, kernel_eval_data.expected_hidden_state,
          kernel_eval_data.expected_cell_state);

  tflite::testing::SetConstTensors(int16_node_contents.GetTensors());
  const TFLMRegistration registration =
      tflite::Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  auto buildin_data = int16_node_contents.BuiltinData();
  tflite::micro::KernelRunner runner(
      registration, int16_node_contents.GetTensors(),
      tflite::testing::kLstmMaxNumInputOutputTensors,
      int16_node_contents.KernelInputs(), int16_node_contents.KernelOutputs(),
      reinterpret_cast<void*>(&buildin_data));
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  tflite::testing::ValidateInvokeResult(
      int16_node_contents, tflite::testing::kExpectedSecondOutput,
      tflite::testing::kExpectedSecondHidden,
      tflite::testing::kExpectedSecondCell,
      tflite::testing::kQuantizedSecondInvokeTolerance,
      tflite::testing::kQuantizedSecondInvokeTolerance);
}

// Seed a NON-ZERO cell state with a ZERO hidden state, so that the incoming
// cell state actually reaches the assertions (see kExpectedSeededCellOutput for
// why the other tests cannot see it).  These tests fail if the kernel drops,
// zeroes or misreads the incoming cell state, even when the write-back path is
// perfectly correct.
TEST(UnidirectionalSequenceLstmTest, TestUnidirectionalLSTMInt8SeededCellState) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents(
          kernel_eval_data.input_data, tflite::testing::kZeroHiddenState,
          kernel_eval_data.expected_cell_state);

  tflite::testing::TestUnidirectionalLSTMSeededState(
      tflite::testing::kExpectedSeededCellOutput,
      tflite::testing::kExpectedSeededCellHidden,
      tflite::testing::kExpectedSeededCellCell, 1e-2, 1e-2, int8_node_contents);
}

TEST(UnidirectionalSequenceLstmTest,
     TestUnidirectionalLSTMInt16SeededCellState) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  tflite::testing::LstmNodeContent<int16_t, int8_t, int64_t, int16_t, 2, 3, 2,
                                   2>
      int16_node_contents = tflite::testing::Create2x3x2X2Int16NodeContents(
          kernel_eval_data.input_data, tflite::testing::kZeroHiddenState,
          kernel_eval_data.expected_cell_state);

  tflite::testing::TestUnidirectionalLSTMSeededState(
      tflite::testing::kExpectedSeededCellOutput,
      tflite::testing::kExpectedSeededCellHidden,
      tflite::testing::kExpectedSeededCellCell, 1e-2, 1e-2,
      int16_node_contents);
}

TEST(UnidirectionalSequenceLstmTest, TestUnidirectionalLSTMInt8TimeMajor) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  float time_major_input[12] = {};
  float time_major_output[12] = {};
  float time_major_second_output[12] = {};
  tflite::testing::ToTimeMajor<2, 3, 2>(kernel_eval_data.input_data,
                                        time_major_input);
  tflite::testing::ToTimeMajor<2, 3, 2>(kernel_eval_data.expected_output,
                                        time_major_output);
  tflite::testing::ToTimeMajor<2, 3, 2>(tflite::testing::kExpectedSecondOutput,
                                        time_major_second_output);

  tflite::testing::LstmNodeContent<int8_t, int8_t, int32_t, int16_t, 2, 3, 2, 2>
      int8_node_contents = tflite::testing::Create2x3x2X2Int8NodeContents(
          time_major_input, kernel_eval_data.hidden_state);

  tflite::testing::TestUnidirectionalLSTMIntegerTimeMajor(
      time_major_output, kernel_eval_data.expected_hidden_state,
      kernel_eval_data.expected_cell_state, time_major_second_output,
      // Measured max deviation for int8 time-major is 0.005175, so 1e-2 leaves
      // only a ~1.93x margin.  That is deliberate: the int8 path genuinely
      // needs the looser bound, unlike int16 below.
      1e-2, 1e-2, int8_node_contents);
}

TEST(UnidirectionalSequenceLstmTest, TestUnidirectionalLSTMInt16TimeMajor) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  float time_major_input[12] = {};
  float time_major_output[12] = {};
  float time_major_second_output[12] = {};
  tflite::testing::ToTimeMajor<2, 3, 2>(kernel_eval_data.input_data,
                                        time_major_input);
  tflite::testing::ToTimeMajor<2, 3, 2>(kernel_eval_data.expected_output,
                                        time_major_output);
  tflite::testing::ToTimeMajor<2, 3, 2>(tflite::testing::kExpectedSecondOutput,
                                        time_major_second_output);

  tflite::testing::LstmNodeContent<int16_t, int8_t, int64_t, int16_t, 2, 3, 2,
                                   2>
      int16_node_contents = tflite::testing::Create2x3x2X2Int16NodeContents(
          time_major_input, kernel_eval_data.hidden_state);

  tflite::testing::TestUnidirectionalLSTMIntegerTimeMajor(
      time_major_output, kernel_eval_data.expected_hidden_state,
      kernel_eval_data.expected_cell_state, time_major_second_output,
      // Measured max deviation for int16 time-major is 0.000149, so 1e-3 still
      // leaves a ~6.7x margin while being 10x tighter than the int8 bound.
      1e-3, 1e-3, int16_node_contents);
}
#endif  // TFLM_LSTM_QUANTIZED_STATEFUL

#if ARM_NN_ENABLE_F16
namespace tflite {
namespace testing {
TEST(UnidirectionalSequenceLstmTest, TestUnidirectionalLSTMFloat16) {
  const tflite::testing::LstmEvalCheckData<12, 4, 12> kernel_eval_data =
      tflite::testing::Get2X2LstmEvalCheckData();
  auto source = tflite::testing::Create2x3x2X2FloatNodeContents(
      kernel_eval_data.input_data, kernel_eval_data.hidden_state);

  TfLiteTensor tensors[25];
  float16_t storage[25][64] = {};
  for (int i = 0; i < 25; ++i) {
    tensors[i] = source.GetTensors()[i];
    if (tensors[i].data.raw == nullptr || tensors[i].dims == nullptr) {
      continue;
    }
    const int elements = ElementCount(*tensors[i].dims);
    for (int j = 0; j < elements; ++j) {
      storage[i][j] = static_cast<float16_t>(tensors[i].data.f[j]);
    }
    tensors[i].data.f16 = reinterpret_cast<TfLiteFloat16*>(storage[i]);
    tensors[i].type = kTfLiteFloat16;
    tensors[i].bytes = elements * sizeof(float16_t);
  }
  for (int i = 1; i < 9; ++i) {
    tensors[i].allocation_type = kTfLiteMmapRo;
  }
  for (int i = 12; i < 16; ++i) {
    tensors[i].allocation_type = kTfLiteMmapRo;
  }

  auto builtin_data = source.BuiltinData();
  const TFLMRegistration registration = Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  micro::KernelRunner runner(registration, tensors, 25, source.KernelInputs(),
                             source.KernelOutputs(),
                             reinterpret_cast<void*>(&builtin_data));
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  const int output_tensor_index = source.KernelOutputs()->data[0];
  const auto* out_f16 =
      reinterpret_cast<float16_t*>(tensors[output_tensor_index].data.raw);
  // The fp16 pipeline deviates from the float32 goldens by up to ~2.7e-2 on
  // this data; 4e-2 leaves headroom for toolchain-dependent fp16 fma /
  // reduction ordering without loosening the check materially.
  for (int i = 0; i < 12; ++i) {
    EXPECT_NEAR(kernel_eval_data.expected_output[i], static_cast<float>(out_f16[i]),
                4e-2f);
  }

#if NS_CMSIS_NN_VERSION >= 7029000
  constexpr float kExpectedSecondOutput[] = {
      0.64306641f, 0.64306641f, 0.65332031f, 0.65332031f,
      0.65625000f, 0.65625000f, 0.36450195f, 0.36450195f,
      0.64550781f, 0.64550781f, 0.60107422f, 0.60107422f};
  constexpr float kExpectedSecondHidden[] = {
      0.65625000f, 0.65625000f, 0.60107422f, 0.60107422f};
  constexpr float kExpectedSecondCell[] = {
      0.97021484f, 0.97021484f, 0.92089844f, 0.92089844f};
  // TODO(#242): interim 2.5e-2 bound, captured not derived; do not loosen
  // further. see ns-cmsis-nn#324.
  constexpr float kSecondInvokeTolerance = 2.5e-2f;

  const auto* hidden_f16 = reinterpret_cast<const float16_t*>(
      tensors[kLstmOutputStateTensor].data.raw);
  const auto* cell_f16 = reinterpret_cast<const float16_t*>(
      tensors[kLstmCellStateTensor].data.raw);
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  for (int i = 0; i < 12; ++i) {
    EXPECT_NEAR(kExpectedSecondOutput[i], static_cast<float>(out_f16[i]),
                kSecondInvokeTolerance);
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(kExpectedSecondHidden[i], static_cast<float>(hidden_f16[i]),
                kSecondInvokeTolerance);
    EXPECT_NEAR(kExpectedSecondCell[i], static_cast<float>(cell_f16[i]),
                kSecondInvokeTolerance);
  }
#endif
}
}  // namespace testing
}  // namespace tflite
#endif
TF_LITE_MICRO_TESTS_MAIN
