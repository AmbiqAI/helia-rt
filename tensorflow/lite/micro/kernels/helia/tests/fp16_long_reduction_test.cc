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

// Long float16 reductions whose exact sum a float16 accumulator cannot reach.
// Inputs are 1.0 and the bias is 0. The weights put a few large values first,
// then zeros, then 0.5 on the remaining taps: once a float16 running sum holds
// the large values, every later +0.5 rounds away. A kernel that folds float16
// partials into float32, with each partial covering no more of the reduction
// than the zero gap, keeps them and returns the exact sum, which float16
// represents:
//   FULLY_CONNECTED, 1x1 CONV_2D, BATCH_MATMUL: 2560 taps, 2048 on taps 0..7,
//   0 to tap 1023, then 0.5: 16384 + 768 = 17152 (float16 gives 16384).
//   DEPTHWISE_CONV_2D, one channel: a 16x18 kernel, 256 on taps 0..7, 0 to tap
//   255, then 0.5: 2048 + 16 = 2064 (float16 gives 2048). The kernel is sized
//   so the float16 depthwise scratch request fits the KernelRunner arena.
// see AmbiqAI/helia-rt#337

#include <cstdint>

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

// float16 bit patterns.
constexpr uint16_t kOne = 0x3c00;
constexpr uint16_t kZero = 0x0000;
constexpr uint16_t kHalf = 0x3800;
constexpr uint16_t k256 = 0x5c00;
constexpr uint16_t k2048 = 0x6800;
constexpr uint16_t k2064 = 0x6808;
constexpr uint16_t k17152 = 0x7430;

constexpr int kDepth = 2560;
constexpr int kKernelHeight = 16;
constexpr int kKernelWidth = 18;
constexpr int kKernelTaps = kKernelHeight * kKernelWidth;

uint16_t g_input[kDepth];
uint16_t g_weights[kDepth];

void FillReduction(int taps, uint16_t large, int zero_end) {
  for (int i = 0; i < taps; ++i) {
    g_input[i] = kOne;
    g_weights[i] = i < 8 ? large : (i < zero_end ? kZero : kHalf);
  }
}

// Runs a node with float16 tensors and checks its single output value. Legs
// without float16 support must reject float16 at Prepare.
void RunAndExpect(const TFLMRegistration& registration, TfLiteTensor* tensors,
                  int tensors_size, int* inputs, int* outputs,
                  void* builtin_data, const uint16_t* output,
                  uint16_t expected) {
  micro::KernelRunner runner(registration, tensors, tensors_size,
                             IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), builtin_data);
#if ARM_NN_ENABLE_F16
  ASSERT_EQ(kTfLiteOk, runner.InitAndPrepare());
  ASSERT_EQ(kTfLiteOk, runner.Invoke());
  EXPECT_EQ(expected, output[0]);
#else
  (void)output;
  (void)expected;
  EXPECT_EQ(kTfLiteError, runner.InitAndPrepare());
#endif
}

void RunBatchMatMul(bool adj_y) {
  FillReduction(kDepth, k2048, 1024);
  int lhs_dims[] = {3, 1, 1, kDepth};
  int rhs_dims_standard[] = {3, 1, kDepth, 1};
  int rhs_dims_adjoint[] = {3, 1, 1, kDepth};
  int output_dims[] = {3, 1, 1, 1};
  uint16_t output[1] = {};
  TfLiteTensor tensors[] = {
      CreateTensor(g_input, IntArrayFromInts(lhs_dims), false, kTfLiteFloat16),
      CreateTensor(
          g_weights,
          IntArrayFromInts(adj_y ? rhs_dims_adjoint : rhs_dims_standard), false,
          kTfLiteFloat16),
      CreateTensor(output, IntArrayFromInts(output_dims), false,
                   kTfLiteFloat16),
  };
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  TfLiteBatchMatMulParams params = {false, adj_y, false};
  RunAndExpect(tflite::Register_BATCH_MATMUL(), tensors, 3, inputs, outputs,
               &params, output, k17152);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(HeliaFp16LongReductionTest, FullyConnected) {
  using namespace tflite::testing;
  FillReduction(kDepth, k2048, 1024);
  int input_dims[] = {2, 1, kDepth};
  int weights_dims[] = {2, 1, kDepth};
  int bias_dims[] = {1, 1};
  int output_dims[] = {2, 1, 1};
  uint16_t bias[] = {kZero};
  uint16_t output[1] = {};
  TfLiteTensor tensors[] = {
      CreateTensor(g_input, IntArrayFromInts(input_dims), false,
                   kTfLiteFloat16),
      CreateTensor(g_weights, IntArrayFromInts(weights_dims), false,
                   kTfLiteFloat16),
      CreateTensor(bias, IntArrayFromInts(bias_dims), false, kTfLiteFloat16),
      CreateTensor(output, IntArrayFromInts(output_dims), false,
                   kTfLiteFloat16),
  };
  int inputs[] = {3, 0, 1, 2};
  int outputs[] = {1, 3};
  TfLiteFullyConnectedParams params = {
      kTfLiteActNone, kTfLiteFullyConnectedWeightsFormatDefault, false, false,
      kTfLiteNoType};
  RunAndExpect(tflite::Register_FULLY_CONNECTED(), tensors, 4, inputs, outputs,
               &params, output, k17152);
}

TEST(HeliaFp16LongReductionTest, Conv1x1) {
  using namespace tflite::testing;
  FillReduction(kDepth, k2048, 1024);
  int input_dims[] = {4, 1, 1, 1, kDepth};
  int filter_dims[] = {4, 1, 1, 1, kDepth};
  int bias_dims[] = {1, 1};
  int output_dims[] = {4, 1, 1, 1, 1};
  uint16_t bias[] = {kZero};
  uint16_t output[1] = {};
  TfLiteTensor tensors[] = {
      CreateTensor(g_input, IntArrayFromInts(input_dims), false,
                   kTfLiteFloat16),
      CreateTensor(g_weights, IntArrayFromInts(filter_dims), false,
                   kTfLiteFloat16),
      CreateTensor(bias, IntArrayFromInts(bias_dims), false, kTfLiteFloat16),
      CreateTensor(output, IntArrayFromInts(output_dims), false,
                   kTfLiteFloat16),
  };
  int inputs[] = {3, 0, 1, 2};
  int outputs[] = {1, 3};
  TfLiteConvParams params = {kTfLitePaddingValid, 1, 1, kTfLiteActNone, 1, 1,
                             kTfLiteNoType};
  RunAndExpect(tflite::Register_CONV_2D(), tensors, 4, inputs, outputs, &params,
               output, k17152);
}

TEST(HeliaFp16LongReductionTest, DepthwiseConv) {
  using namespace tflite::testing;
  FillReduction(kKernelTaps, k256, 256);
  int input_dims[] = {4, 1, kKernelHeight, kKernelWidth, 1};
  int filter_dims[] = {4, 1, kKernelHeight, kKernelWidth, 1};
  int bias_dims[] = {1, 1};
  int output_dims[] = {4, 1, 1, 1, 1};
  uint16_t bias[] = {kZero};
  uint16_t output[1] = {};
  TfLiteTensor tensors[] = {
      CreateTensor(g_input, IntArrayFromInts(input_dims), false,
                   kTfLiteFloat16),
      CreateTensor(g_weights, IntArrayFromInts(filter_dims), false,
                   kTfLiteFloat16),
      CreateTensor(bias, IntArrayFromInts(bias_dims), false, kTfLiteFloat16),
      CreateTensor(output, IntArrayFromInts(output_dims), false,
                   kTfLiteFloat16),
  };
  int inputs[] = {3, 0, 1, 2};
  int outputs[] = {1, 3};
  TfLiteDepthwiseConvParams params = {};
  params.padding = kTfLitePaddingValid;
  params.stride_width = 1;
  params.stride_height = 1;
  params.depth_multiplier = 1;
  params.activation = kTfLiteActNone;
  params.dilation_width_factor = 1;
  params.dilation_height_factor = 1;
  RunAndExpect(tflite::Register_DEPTHWISE_CONV_2D(), tensors, 4, inputs,
               outputs, &params, output, k2064);
}

TEST(HeliaFp16LongReductionTest, BatchMatMul) {
  tflite::testing::RunBatchMatMul(/*adj_y=*/false);
}

TEST(HeliaFp16LongReductionTest, BatchMatMulAdjointRhs) {
  tflite::testing::RunBatchMatMul(/*adj_y=*/true);
}

#if !ARM_NN_ENABLE_F16 && defined(__ARM_FEATURE_MVE) && \
    ((__ARM_FEATURE_MVE) & 2)
// helia.inc defines ARM_NN_ENABLE_F16 for TARGET_ARCH=cortex-m55 only, and a
// silent compile-out would pass every case above through its Prepare-rejects
// branch. see AmbiqAI/helia-rt#231
TEST(HeliaFp16LongReductionTest, Float16CoverageMustNotSilentlyDisappear) {
  FAIL(
      "ARM_NN_ENABLE_F16 is not defined on a build with MVE floating point. "
      "The float16 long-reduction coverage silently compiled out.");
}
#endif

TF_LITE_MICRO_TESTS_MAIN
