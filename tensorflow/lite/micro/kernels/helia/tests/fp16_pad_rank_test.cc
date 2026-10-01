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

// Float16 PAD and PADV2 below rank 4, and empty tensors at any rank. The
// helia float16 path runs only on heliaCORE, which pads NHWC tensors, so a
// rank-r input must land on the last r of (N, H, W, C). The expected outputs
// are exact in float16. see AmbiqAI/helia-rt#348, AmbiqAI/helia-rt#351

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

// float16 bit patterns.
constexpr uint16_t k0 = 0x0000;
constexpr uint16_t k1 = 0x3c00;
constexpr uint16_t k2 = 0x4000;
constexpr uint16_t k3 = 0x4200;
constexpr uint16_t k4 = 0x4400;
constexpr uint16_t k5 = 0x4500;
constexpr uint16_t k6 = 0x4600;
constexpr uint16_t k7 = 0x4700;
// Fills the output buffer so an unwritten or overrun cell is visible.
constexpr uint16_t kSentinel = 0x7e01;
constexpr int kOutputCapacity = 16;

// Runs PAD (or PADV2 when `constant` is set) and compares the output bits.
// Legs without float16 support must reject float16 at Prepare.
void RunPad(int* input_dims, uint16_t* input, int* pad_dims,
            const int32_t* pads, uint16_t* constant, int* output_dims,
            const uint16_t* expected, int output_size) {
  uint16_t output[kOutputCapacity];
  for (int i = 0; i < kOutputCapacity; ++i) {
    output[i] = kSentinel;
  }
  int scalar_dims[] = {0};
  TfLiteTensor tensors[] = {
      CreateTensor(input, IntArrayFromInts(input_dims), false, kTfLiteFloat16),
      CreateTensor(pads, IntArrayFromInts(pad_dims), false, kTfLiteInt32),
      CreateTensor(output, IntArrayFromInts(output_dims), false,
                   kTfLiteFloat16),
      CreateTensor(constant, IntArrayFromInts(scalar_dims), false,
                   kTfLiteFloat16),
  };
  tensors[1].allocation_type = kTfLiteMmapRo;

  int pad_inputs[] = {2, 0, 1};
  int padv2_inputs[] = {3, 0, 1, 3};
  int outputs[] = {1, 2};
  const TFLMRegistration registration =
      constant == nullptr ? Register_PAD() : Register_PADV2();
  micro::KernelRunner runner(
      registration, tensors, constant == nullptr ? 3 : 4,
      IntArrayFromInts(constant == nullptr ? pad_inputs : padv2_inputs),
      IntArrayFromInts(outputs), nullptr);
#if ARM_NN_ENABLE_F16
  ASSERT_EQ(kTfLiteOk, runner.InitAndPrepare());
  ASSERT_EQ(kTfLiteOk, runner.Invoke());
  for (int i = 0; i < output_size; ++i) {
    EXPECT_EQ(expected[i], output[i]);
  }
  for (int i = output_size; i < kOutputCapacity; ++i) {
    EXPECT_EQ(kSentinel, output[i]);
  }
#else
  (void)expected;
  (void)output_size;
  EXPECT_EQ(kTfLiteError, runner.InitAndPrepare());
#endif
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(HeliaFp16PadRankTest, Rank1) {
  using namespace tflite::testing;
  int input_dims[] = {1, 3};
  uint16_t input[] = {k1, k2, k3};
  int pad_dims[] = {2, 1, 2};
  const int32_t pads[] = {1, 2};
  int output_dims[] = {1, 6};
  const uint16_t expected[] = {k0, k1, k2, k3, k0, k0};
  RunPad(input_dims, input, pad_dims, pads, nullptr, output_dims, expected, 6);
}

TEST(HeliaFp16PadRankTest, Rank2) {
  using namespace tflite::testing;
  int input_dims[] = {2, 2, 2};
  uint16_t input[] = {k1, k2, k3, k4};
  int pad_dims[] = {2, 2, 2};
  const int32_t pads[] = {1, 0, 0, 1};
  int output_dims[] = {2, 3, 3};
  const uint16_t expected[] = {k0, k0, k0, k1, k2, k0, k3, k4, k0};
  RunPad(input_dims, input, pad_dims, pads, nullptr, output_dims, expected, 9);
}

TEST(HeliaFp16PadRankTest, Rank3WithConstant) {
  using namespace tflite::testing;
  int input_dims[] = {3, 1, 2, 3};
  uint16_t input[] = {k1, k2, k3, k4, k5, k6};
  int pad_dims[] = {2, 3, 2};
  const int32_t pads[] = {0, 0, 1, 0, 0, 1};
  uint16_t constant = k7;
  int output_dims[] = {3, 1, 3, 4};
  const uint16_t expected[] = {k7, k7, k7, k7, k1, k2, k3, k7,
                               k4, k5, k6, k7};
  RunPad(input_dims, input, pad_dims, pads, &constant, output_dims, expected,
         12);
}

// An empty output has nothing to write, as in the reference kernel.
// see AmbiqAI/helia-rt#351
TEST(HeliaFp16PadRankTest, Rank4EmptyOutput) {
  using namespace tflite::testing;
  int input_dims[] = {4, 1, 0, 2, 1};
  uint16_t input[] = {k1};
  int pad_dims[] = {2, 4, 2};
  const int32_t pads[] = {0, 0, 0, 0, 1, 1, 0, 0};
  int output_dims[] = {4, 1, 0, 4, 1};
  RunPad(input_dims, input, pad_dims, pads, nullptr, output_dims, nullptr, 0);
}

TEST(HeliaFp16PadRankTest, Rank2EmptyOutputWithConstant) {
  using namespace tflite::testing;
  int input_dims[] = {2, 0, 3};
  uint16_t input[] = {k1};
  int pad_dims[] = {2, 2, 2};
  const int32_t pads[] = {0, 0, 1, 0};
  uint16_t constant = k7;
  int output_dims[] = {2, 0, 4};
  RunPad(input_dims, input, pad_dims, pads, &constant, output_dims, nullptr, 0);
}

TEST(HeliaFp16PadRankTest, Rank2EmptyInput) {
  using namespace tflite::testing;
  int input_dims[] = {2, 0, 3};
  uint16_t input[] = {k1};
  int pad_dims[] = {2, 2, 2};
  const int32_t pads[] = {1, 1, 0, 0};
  int output_dims[] = {2, 2, 3};
  const uint16_t expected[] = {k0, k0, k0, k0, k0, k0};
  RunPad(input_dims, input, pad_dims, pads, nullptr, output_dims, expected, 6);
}

// An empty input has no arena buffer; the output is all padding.
TEST(HeliaFp16PadRankTest, Rank4EmptyNullInputWithConstant) {
  using namespace tflite::testing;
  int input_dims[] = {4, 1, 0, 2, 1};
  int pad_dims[] = {2, 4, 2};
  const int32_t pads[] = {0, 0, 1, 1, 0, 0, 0, 0};
  uint16_t constant = k7;
  int output_dims[] = {4, 1, 2, 2, 1};
  const uint16_t expected[] = {k7, k7, k7, k7};
  RunPad(input_dims, nullptr, pad_dims, pads, &constant, output_dims,
         expected, 4);
}

#if !ARM_NN_ENABLE_F16 && defined(__ARM_FEATURE_MVE) && \
    ((__ARM_FEATURE_MVE) & 2)
// helia.inc defines ARM_NN_ENABLE_F16 for TARGET_ARCH=cortex-m55 only, and a
// silent compile-out would pass every case above through its Prepare-rejects
// branch. see AmbiqAI/helia-rt#231
TEST(HeliaFp16PadRankTest, Float16CoverageMustNotSilentlyDisappear) {
  FAIL(
      "ARM_NN_ENABLE_F16 is not defined on a build with MVE floating point. "
      "The float16 PAD rank coverage silently compiled out.");
}
#endif

TF_LITE_MICRO_TESTS_MAIN
