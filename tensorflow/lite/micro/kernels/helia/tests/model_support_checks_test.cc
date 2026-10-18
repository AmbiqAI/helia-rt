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

// Valid models the helia kernels must run, or reject with kTfLiteError, without
// reading an input that is absent or not yet planned. A float32 SVDF without a
// bias runs heliaCORE's arm_svdf_f32 when ARM_NN_ENABLE_F32 is on and the
// shared reference SVDF when it is off. TRANSPOSE rejects a perm computed at
// run time, as upstream does: Prepare reads perm, whose data is not planned
// until after Prepare.

#include <cstdint>

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

// A float32 SVDF with rank 1, two batches, two inputs, two filters and a
// memory of two, run once from a zero state. Every value is a small integer
// or a power-of-two fraction, so the outputs are exact.
struct SvdfFloat {
  int input_shape[3] = {2, 2, 2};
  int weights_feature_shape[3] = {2, 2, 2};
  int weights_time_shape[3] = {2, 2, 2};
  int bias_shape[2] = {1, 2};
  int activation_state_shape[3] = {2, 2, 4};
  int output_shape[3] = {2, 2, 2};
  float input[4] = {1, 2, 2, 0};
  float weights_feature[4] = {1, 1, 1, -1};
  float weights_time[4] = {1, 2, 1, 1};
  float bias[2] = {};
  float activation_state[8] = {};
  float output[4] = {};
  TfLiteTensor tensors[6];
  TfLiteSVDFParams params = {1, kTfLiteActNone, false};

  explicit SvdfFloat(const float* bias_values) {
    bias[0] = bias_values[0];
    bias[1] = bias_values[1];
    tensors[0] = CreateTensor(input, IntArrayFromInts(input_shape));
    tensors[1] =
        CreateTensor(weights_feature, IntArrayFromInts(weights_feature_shape));
    tensors[2] =
        CreateTensor(weights_time, IntArrayFromInts(weights_time_shape));
    tensors[3] = CreateTensor(bias, IntArrayFromInts(bias_shape));
    tensors[4] = CreateTensor(activation_state,
                              IntArrayFromInts(activation_state_shape),
                              /*is_variable=*/true);
    tensors[5] = CreateTensor(output, IntArrayFromInts(output_shape));
  }

  TfLiteStatus Run(bool with_bias) {
    int inputs_array_data[] = {5, 0, 1, 2,
                               with_bias ? 3 : kTfLiteOptionalTensor, 4};
    int outputs_array_data[] = {1, 5};
    const TFLMRegistration registration = Register_SVDF();
    micro::KernelRunner runner(registration, tensors, 6,
                               IntArrayFromInts(inputs_array_data),
                               IntArrayFromInts(outputs_array_data), &params);
    TF_LITE_ENSURE_STATUS(runner.InitAndPrepare());
    return runner.Invoke();
  }
};

// Batch 0 filters give 3 and -1, batch 1 filters give 2 and 2; the time
// weights then give units {6, -1} and {4, 2}.
constexpr float kSvdfNoBiasGolden[4] = {6, -1, 4, 2};

// Transposes a 2x3 int8 input by {1, 0}, with the perm tensor read-only
// (constant) or arena-backed, and its data present or still unplanned. Prepare
// must release its temporaries whether it succeeds or not.
TfLiteStatus PrepareTranspose(TfLiteAllocationType perm_allocation,
                              bool perm_planned) {
  int input_shape[] = {2, 2, 3};
  int perm_shape[] = {1, 2};
  int output_shape[] = {2, 3, 2};
  int8_t input[6] = {1, 2, 3, 4, 5, 6};
  int32_t perm[2] = {1, 0};
  int8_t output[6] = {};
  TfLiteTensor tensors[] = {
      CreateTensor(input, IntArrayFromInts(input_shape)),
      CreateTensor(perm, IntArrayFromInts(perm_shape)),
      CreateTensor(output, IntArrayFromInts(output_shape)),
  };
  tensors[1].allocation_type = perm_allocation;
  if (!perm_planned) {
    tensors[1].data.data = nullptr;
  }
  int inputs_array_data[] = {2, 0, 1};
  int outputs_array_data[] = {1, 2};
  const TFLMRegistration registration = Register_TRANSPOSE();
  micro::KernelRunner runner(registration, tensors, 3,
                             IntArrayFromInts(inputs_array_data),
                             IntArrayFromInts(outputs_array_data), nullptr);
  const TfLiteStatus status = runner.InitAndPrepare();
  // InitAndPrepare checks the temporaries only when Prepare succeeds.
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
  return status;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(HeliaModelSupportChecksTest, SvdfFloat32WithoutBiasMatchesGolden) {
  const float unused_bias[2] = {100, 100};
  tflite::testing::SvdfFloat svdf(unused_bias);
  ASSERT_EQ(svdf.Run(/*with_bias=*/false), kTfLiteOk);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(svdf.output[i], tflite::testing::kSvdfNoBiasGolden[i]);
  }
}

// The same model with a bias adds it to every batch, so the absent-bias case
// above is not passing by ignoring the bias input altogether.
TEST(HeliaModelSupportChecksTest, SvdfFloat32WithBiasAddsBias) {
  const float bias[2] = {0.5f, 0.25f};
  tflite::testing::SvdfFloat svdf(bias);
  ASSERT_EQ(svdf.Run(/*with_bias=*/true), kTfLiteOk);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(svdf.output[i],
              tflite::testing::kSvdfNoBiasGolden[i] + bias[i % 2]);
  }
}

TEST(HeliaModelSupportChecksTest, TransposeConstantPermPrepares) {
  EXPECT_EQ(tflite::testing::PrepareTranspose(kTfLiteMmapRo,
                                              /*perm_planned=*/true),
            kTfLiteOk);
}

TEST(HeliaModelSupportChecksTest, TransposeUnplannedRuntimePermFails) {
  EXPECT_EQ(tflite::testing::PrepareTranspose(kTfLiteArenaRw,
                                              /*perm_planned=*/false),
            kTfLiteError);
}

// Eval does not range-check perm, so a run-time perm is rejected even when
// its values happen to be valid at Prepare.
TEST(HeliaModelSupportChecksTest, TransposeRuntimePermFails) {
  EXPECT_EQ(tflite::testing::PrepareTranspose(kTfLiteArenaRw,
                                              /*perm_planned=*/true),
            kTfLiteError);
}

TF_LITE_MICRO_TESTS_MAIN
