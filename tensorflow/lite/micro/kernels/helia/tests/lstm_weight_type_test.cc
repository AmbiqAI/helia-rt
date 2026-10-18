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

// UNIDIRECTIONAL_SEQUENCE_LSTM weight types. Quantized LSTMs carry int8
// weights, and the int8 and int16 registrations must run them, in debug
// builds too, bit for bit as the generic registration does. A hybrid model
// (float32 activations, int8 weights) and quantized activations with int16
// weights must fail Prepare. see AmbiqAI/helia-rt#470

#include <cstdint>

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/kernels/testdata/lstm_test_data.h"
#include "tensorflow/lite/micro/kernels/unidirectional_sequence_lstm.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kNumTensors = 24 + 1;
constexpr int kOutputCount = 2 * 3 * 2;
constexpr int kStateCount = 2 * 2;

void SetConstTensors(TfLiteTensor* tensors) {
  for (int i = 1; i < 9; ++i) {
    tensors[i].allocation_type = kTfLiteMmapRo;  // weights
  }
  for (int i = 12; i < 16; ++i) {
    tensors[i].allocation_type = kTfLiteMmapRo;  // biases
  }
}

template <typename Contents>
TfLiteStatus Run(const TFLMRegistration& registration, Contents& contents) {
  SetConstTensors(contents.GetTensors());
  auto params = contents.BuiltinData();
  micro::KernelRunner runner(registration, contents.GetTensors(), kNumTensors,
                             contents.KernelInputs(), contents.KernelOutputs(),
                             &params);
  const TfLiteStatus prepare = runner.InitAndPrepare();
  if (prepare != kTfLiteOk) {
    // InitAndPrepare() checks this only when Prepare succeeds.
    EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
    return prepare;
  }
  return runner.Invoke();
}

// The typed registration against the generic one on the same model.
template <typename Contents>
void ExpectMatchesGeneric(Contents (*create)(const float*, const float*,
                                             const float*),
                          const TFLMRegistration& typed) {
  const LstmEvalCheckData<12, 4, 12> eval_data = Get2X2LstmEvalCheckData();
  Contents generic = create(eval_data.input_data, nullptr, nullptr);
  ASSERT_EQ(kTfLiteOk, Run(Register_UNIDIRECTIONAL_SEQUENCE_LSTM(), generic));
  Contents contents = create(eval_data.input_data, nullptr, nullptr);
  ASSERT_EQ(kTfLiteOk, Run(typed, contents));
  for (int i = 0; i < kOutputCount; ++i) {
    EXPECT_EQ(generic.GetOutputData()[i], contents.GetOutputData()[i]);
  }
  for (int i = 0; i < kStateCount; ++i) {
    EXPECT_EQ(generic.GetHiddenStateData()[i],
              contents.GetHiddenStateData()[i]);
    EXPECT_EQ(generic.GetCellStateData()[i], contents.GetCellStateData()[i]);
  }
}

// Retypes the eight gate weight tensors over buffers of the new type.
template <typename T>
void RetypeWeights(TfLiteTensor* tensors, TfLiteType type, T* storage,
                   int per_tensor) {
  for (int i = 1; i < 9; ++i) {
    T* data = storage + (i - 1) * per_tensor;
    for (int j = 0; j < per_tensor; ++j) {
      data[j] = static_cast<T>(j - 2);
    }
    tensors[i].type = type;
    tensors[i].data.data = data;
    tensors[i].bytes = per_tensor * sizeof(T);
    tensors[i].params = {0.01f, 0};
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ExpectMatchesGeneric;

TEST(HeliaLstmWeightTypeTest, Int8RegistrationRunsInt8Weights) {
  ExpectMatchesGeneric(&tflite::testing::Create2x3x2X2Int8NodeContents,
                       tflite::Register_UNIDIRECTIONAL_SEQUENCE_LSTM_INT8());
}

TEST(HeliaLstmWeightTypeTest, Int16RegistrationRunsInt8Weights) {
  ExpectMatchesGeneric(&tflite::testing::Create2x3x2X2Int16NodeContents,
                       tflite::Register_UNIDIRECTIONAL_SEQUENCE_LSTM_INT16());
}

TEST(HeliaLstmWeightTypeTest, HybridFloatActivationsInt8WeightsFailPrepare) {
  const auto eval_data = tflite::testing::Get2X2LstmEvalCheckData();
  auto contents =
      tflite::testing::Create2x3x2X2FloatNodeContents(eval_data.input_data);
  // Each gate weight tensor holds state_dimension x input_dimension = 4.
  static int8_t weights[8 * 4];
  tflite::testing::RetypeWeights(contents.GetTensors(), kTfLiteInt8, weights,
                                 4);
  EXPECT_EQ(kTfLiteError,
            tflite::testing::Run(
                tflite::Register_UNIDIRECTIONAL_SEQUENCE_LSTM(), contents));
}

TEST(HeliaLstmWeightTypeTest, Int8ActivationsInt16WeightsFailPrepare) {
  const auto eval_data = tflite::testing::Get2X2LstmEvalCheckData();
  auto contents =
      tflite::testing::Create2x3x2X2Int8NodeContents(eval_data.input_data);
  static int16_t weights[8 * 4];
  tflite::testing::RetypeWeights(contents.GetTensors(), kTfLiteInt16, weights,
                                 4);
  EXPECT_EQ(
      kTfLiteError,
      tflite::testing::Run(tflite::Register_UNIDIRECTIONAL_SEQUENCE_LSTM_INT8(),
                           contents));
}

TF_LITE_MICRO_TESTS_MAIN
