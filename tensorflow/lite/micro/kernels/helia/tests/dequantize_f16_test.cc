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

// float16 -> float32 DEQUANTIZE through the helia kernel for all 65536 binary16
// patterns, against Float16BitsToFloat32, the reference widening: every
// pattern matches bit for bit on every build, a NaN keeping its sign and
// payload. The kernel calls arm_dequantize_f16_bits_f32 on every build; GCC
// and ATfE links wrap the entry to prove it.

#include <cstdint>
#include <cstring>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/dequantize.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

int g_bits_calls = 0;

}  // namespace

#if HELIA_DEQUANTIZE_F16_LINK_WRAP
extern "C" {

arm_cmsis_nn_status __real_arm_dequantize_f16_bits_f32(const uint16_t*, float*,
                                                       int32_t);
arm_cmsis_nn_status __wrap_arm_dequantize_f16_bits_f32(const uint16_t* input,
                                                       float* output,
                                                       int32_t block_size) {
  ++g_bits_calls;
  return __real_arm_dequantize_f16_bits_f32(input, output, block_size);
}

}  // extern "C"
#endif  // HELIA_DEQUANTIZE_F16_LINK_WRAP

namespace tflite {
namespace testing {
namespace {


// An odd chunk exercises the conversion's tail as well as its full blocks.
constexpr int kChunk = 2047;

bool IsHalfNaN(uint16_t bits) {
  return (bits & 0x7C00u) == 0x7C00u && (bits & 0x03FFu) != 0;
}

uint32_t Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// Dequantizes patterns [first, first + count) and counts mismatches against
// the reference widening, NaNs and non-NaNs separately.
void RunChunk(uint32_t first, int count, int* number_mismatches,
              int* nan_mismatches) {
  static uint16_t input[kChunk];
  static float output[kChunk];
  for (int i = 0; i < count; ++i) {
    input[i] = static_cast<uint16_t>(first + i);
  }
  int dims[] = {1, count};
  TfLiteTensor tensors[2] = {
      CreateTensor(input, IntArrayFromInts(dims), false, kTfLiteFloat16),
      CreateTensor(output, IntArrayFromInts(dims)),
  };
  int inputs[] = {1, 0};
  int outputs[] = {1, 1};
  const TFLMRegistration registration = Register_DEQUANTIZE();
  micro::KernelRunner runner(registration, tensors, 2, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), nullptr);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  g_bits_calls = 0;
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
#if HELIA_DEQUANTIZE_F16_LINK_WRAP
  EXPECT_EQ(1, g_bits_calls);
#endif
  for (int i = 0; i < count; ++i) {
    const uint32_t expected = Bits(micro::Float16BitsToFloat32(input[i]));
    if (Bits(output[i]) != expected) {
      ++*(IsHalfNaN(input[i]) ? nan_mismatches : number_mismatches);
    }
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(HeliaDequantizeF16Test, EveryBinary16PatternMatchesTheReference) {
  int number_mismatches = 0;
  int nan_mismatches = 0;
  for (uint32_t first = 0; first < 0x10000u;
       first += tflite::testing::kChunk) {
    const uint32_t left = 0x10000u - first;
    const int count = left < static_cast<uint32_t>(tflite::testing::kChunk)
                          ? static_cast<int>(left)
                          : tflite::testing::kChunk;
    tflite::testing::RunChunk(first, count, &number_mismatches,
                              &nan_mismatches);
  }
  EXPECT_EQ(0, number_mismatches);
  EXPECT_EQ(0, nan_mismatches);
}

TEST(HeliaDequantizeF16Test, EmptyTensorIsANoOp) {
  uint16_t input[1] = {0x3C00};
  float output[1] = {-1.0f};
  int dims[] = {1, 0};
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;
  TfLiteTensor tensors[2] = {
      CreateTensor(input, IntArrayFromInts(dims), false, kTfLiteFloat16),
      CreateTensor(output, IntArrayFromInts(dims)),
  };
  int inputs[] = {1, 0};
  int outputs[] = {1, 1};
  const TFLMRegistration registration = tflite::Register_DEQUANTIZE();
  tflite::micro::KernelRunner runner(registration, tensors, 2,
                                     IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), nullptr);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  EXPECT_EQ(-1.0f, output[0]);
}

TF_LITE_MICRO_TESTS_MAIN
