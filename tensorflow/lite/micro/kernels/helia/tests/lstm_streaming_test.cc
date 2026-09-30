/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Streaming shapes for the stateful quantized UNIDIRECTIONAL_SEQUENCE_LSTM:
// time_steps == 1 invoked once per step, and batch == 1. Both run the 2x2
// model of testdata/lstm_test_data.cc.
//
// Integer arithmetic does not depend on how the work is split, so
//   - three single-step invocations must equal one three-step invocation, and
//   - a one-batch run must equal that batch of the two-batch run,
// bit for bit. Both are also checked against the float goldens of
// Get2X2LstmEvalCheckData() with the tolerances the shared LSTM test uses.
// see AmbiqAI/helia-rt#211

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/testdata/lstm_test_data.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {

// Defined in testdata/lstm_test_data.cc.
NodeQuantizationParameters Get2X2Int16LstmQuantizationSettings();

namespace {

constexpr int kNumTensors = 24 + 1;
constexpr int kBatchSize = 2;
constexpr int kTimeSteps = 3;
constexpr int kInputDimension = 2;
constexpr int kStateDimension = 2;

// The builtin data and gates of Create2x3x2X2FloatNodeContents().
constexpr TfLiteUnidirectionalSequenceLSTMParams kBuiltinData = {
    /*.activation=*/kTfLiteActTanh,
    /*.cell_clip=*/6,
    /*.proj_clip=*/3,
    /*.time_major=*/false,
    /*.asymmetric_quantize_inputs=*/true,
    /*diagonal_recurrent_tensors=*/false};

template <int batch_size, int time_steps>
LstmNodeContent<float, float, float, float, batch_size, time_steps,
                kInputDimension, kStateDimension>
CreateFloatNodeContents(const float* input_data) {
  const GateData<float, float, kInputDimension, kStateDimension> forget_gate = {
      {-10, -10, -20, -20}, {-10, -10, -20, -20}, {1, 2}, {0, 0}, {0, 0}};
  const GateData<float, float, kInputDimension, kStateDimension> input_gate = {
      {10, 10, 20, 20}, {10, 10, 20, 20}, {-1, -2}, {0, 0}, {0, 0}};
  const GateData<float, float, kInputDimension, kStateDimension> cell_gate = {
      {1, 1, 1, 1}, {1, 1, 1, 1}, {0, 0}, {0, 0}, {0, 0}};
  const GateData<float, float, kInputDimension, kStateDimension> output_gate = {
      {1, 1, 1, 1}, {1, 1, 1, 1}, {0, 0}, {0, 0}, {0, 0}};
  LstmNodeContent<float, float, float, float, batch_size, time_steps,
                  kInputDimension, kStateDimension>
      contents(kBuiltinData, forget_gate, input_gate, cell_gate, output_gate);
  contents.SetInputData(input_data);
  return contents;
}

template <typename ActivationType, typename BiasType, int batch_size,
          int time_steps>
LstmNodeContent<ActivationType, int8_t, BiasType, int16_t, batch_size,
                time_steps, kInputDimension, kStateDimension>
CreateIntegerContents(const float* input_data) {
  constexpr bool kInt8 = sizeof(ActivationType) == 1;
  auto float_contents =
      CreateFloatNodeContents<batch_size, time_steps>(input_data);
  return CreateIntegerNodeContents<ActivationType, int8_t, BiasType, int16_t,
                                   batch_size, time_steps, kInputDimension,
                                   kStateDimension>(
      kInt8 ? Get2X2Int8LstmQuantizationSettings()
            : Get2X2Int16LstmQuantizationSettings(),
      /*fold_zero_point=*/kInt8, float_contents);
}

template <typename Contents>
void SetConstTensors(Contents& contents) {
  TfLiteTensor* tensors = contents.GetTensors();
  for (int i = 1; i < 9; ++i) {
    tensors[i].allocation_type = kTfLiteMmapRo;  // weights
  }
  for (int i = 12; i < 16; ++i) {
    tensors[i].allocation_type = kTfLiteMmapRo;  // biases
  }
}

template <typename T>
void ExpectNearGolden(const T* data, int size, float scale, int zero_point,
                      const float* golden, float tolerance) {
  for (int i = 0; i < size; ++i) {
    const float value = scale * (static_cast<float>(data[i]) - zero_point);
    EXPECT_NEAR(golden[i], value, tolerance);
  }
}

// Runs the reference three-step, two-batch invocation once, then the
// streaming shapes against it.
template <typename ActivationType, typename BiasType>
void TestStreamingShapes(float hidden_tolerance, float cell_tolerance) {
  const LstmEvalCheckData<12, 4, 12> eval_data = Get2X2LstmEvalCheckData();
  const TFLMRegistration registration = Register_UNIDIRECTIONAL_SEQUENCE_LSTM();

  auto sequence =
      CreateIntegerContents<ActivationType, BiasType, kBatchSize, kTimeSteps>(
          eval_data.input_data);
  SetConstTensors(sequence);
  auto sequence_params = sequence.BuiltinData();
  // Runners share KernelRunner's static arena, so each is finished with
  // before the next one is built.
  {
    micro::KernelRunner runner(registration, sequence.GetTensors(), kNumTensors,
                               sequence.KernelInputs(),
                               sequence.KernelOutputs(), &sequence_params);
    ASSERT_EQ(kTfLiteOk, runner.InitAndPrepare());
    ASSERT_EQ(kTfLiteOk, runner.Invoke());
  }
  const ActivationType* sequence_output = sequence.GetOutputData();
  const auto& quantization = sequence.QuantizationSettings();

  // time_steps == 1: one invocation per step, state carried by the kernel.
  float step_input[kTimeSteps][kBatchSize * kInputDimension];
  for (int t = 0; t < kTimeSteps; ++t) {
    for (int b = 0; b < kBatchSize; ++b) {
      for (int d = 0; d < kInputDimension; ++d) {
        step_input[t][b * kInputDimension + d] =
            eval_data.input_data[(b * kTimeSteps + t) * kInputDimension + d];
      }
    }
  }
  auto streaming =
      CreateIntegerContents<ActivationType, BiasType, kBatchSize, 1>(
          step_input[0]);
  SetConstTensors(streaming);
  auto streaming_params = streaming.BuiltinData();
  {
    micro::KernelRunner runner(registration, streaming.GetTensors(),
                               kNumTensors, streaming.KernelInputs(),
                               streaming.KernelOutputs(), &streaming_params);
    ASSERT_EQ(kTfLiteOk, runner.InitAndPrepare());
    for (int t = 0; t < kTimeSteps; ++t) {
      // Same quantization settings, so the quantized step input can be copied.
      auto step =
          CreateIntegerContents<ActivationType, BiasType, kBatchSize, 1>(
              step_input[t]);
      streaming.SetInputData(step.GetInputData());
      ASSERT_EQ(kTfLiteOk, runner.Invoke());

      float step_golden[kBatchSize * kStateDimension];
      for (int b = 0; b < kBatchSize; ++b) {
        for (int s = 0; s < kStateDimension; ++s) {
          const int sequence_index = (b * kTimeSteps + t) * kStateDimension + s;
          EXPECT_EQ(sequence_output[sequence_index],
                    streaming.GetOutputData()[b * kStateDimension + s]);
          step_golden[b * kStateDimension + s] =
              eval_data.expected_output[sequence_index];
        }
      }
      ExpectNearGolden(streaming.GetOutputData(), kBatchSize * kStateDimension,
                       quantization.output.scale,
                       quantization.output.zero_point, step_golden,
                       hidden_tolerance);
    }
  }
  for (int i = 0; i < kBatchSize * kStateDimension; ++i) {
    EXPECT_EQ(sequence.GetHiddenStateData()[i],
              streaming.GetHiddenStateData()[i]);
    EXPECT_EQ(sequence.GetCellStateData()[i], streaming.GetCellStateData()[i]);
  }
  ExpectNearGolden(streaming.GetHiddenStateData(), kBatchSize * kStateDimension,
                   quantization.hidden_state.scale,
                   quantization.hidden_state.zero_point,
                   eval_data.expected_hidden_state, hidden_tolerance);
  ExpectNearGolden(streaming.GetCellStateData(), kBatchSize * kStateDimension,
                   quantization.cell_state.scale,
                   quantization.cell_state.zero_point,
                   eval_data.expected_cell_state, cell_tolerance);

  // batch == 1: batch 0 of the two-batch sequence.
  auto single = CreateIntegerContents<ActivationType, BiasType, 1, kTimeSteps>(
      eval_data.input_data);
  SetConstTensors(single);
  auto single_params = single.BuiltinData();
  {
    micro::KernelRunner runner(registration, single.GetTensors(), kNumTensors,
                               single.KernelInputs(), single.KernelOutputs(),
                               &single_params);
    ASSERT_EQ(kTfLiteOk, runner.InitAndPrepare());
    ASSERT_EQ(kTfLiteOk, runner.Invoke());
  }
  for (int i = 0; i < kTimeSteps * kStateDimension; ++i) {
    EXPECT_EQ(sequence_output[i], single.GetOutputData()[i]);
  }
  for (int i = 0; i < kStateDimension; ++i) {
    EXPECT_EQ(sequence.GetHiddenStateData()[i], single.GetHiddenStateData()[i]);
    EXPECT_EQ(sequence.GetCellStateData()[i], single.GetCellStateData()[i]);
  }
  ExpectNearGolden(single.GetOutputData(), kTimeSteps * kStateDimension,
                   quantization.output.scale, quantization.output.zero_point,
                   eval_data.expected_output, hidden_tolerance);
  ExpectNearGolden(single.GetHiddenStateData(), kStateDimension,
                   quantization.hidden_state.scale,
                   quantization.hidden_state.zero_point,
                   eval_data.expected_hidden_state, hidden_tolerance);
  ExpectNearGolden(single.GetCellStateData(), kStateDimension,
                   quantization.cell_state.scale,
                   quantization.cell_state.zero_point,
                   eval_data.expected_cell_state, cell_tolerance);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(HeliaLstmStreamingTest, Int8StreamingShapes) {
  tflite::testing::TestStreamingShapes<int8_t, int32_t>(
      /*hidden_tolerance=*/1e-2f, /*cell_tolerance=*/1e-2f);
}

TEST(HeliaLstmStreamingTest, Int16StreamingShapes) {
  tflite::testing::TestStreamingShapes<int16_t, int64_t>(
      /*hidden_tolerance=*/1e-3f, /*cell_tolerance=*/1e-2f);
}

TF_LITE_MICRO_TESTS_MAIN
