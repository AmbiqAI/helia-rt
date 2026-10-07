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

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

#if defined(HELIA_QUANTIZE_S16_LINK_WRAP)
struct RequantizeCall {
  int calls = 0;
  int32_t size = 0;
  int32_t multiplier = 0;
  int32_t shift = 0;
  int32_t input_zero_point = 0;
  int32_t output_zero_point = 0;
};

RequantizeCall g_requantize_call;
#endif

}  // namespace

#if defined(HELIA_QUANTIZE_S16_LINK_WRAP)
extern "C" {
arm_cmsis_nn_status __real_arm_requantize_s16_s16(const int16_t*, int16_t*,
                                                  int32_t, int32_t, int32_t,
                                                  int32_t, int32_t);

arm_cmsis_nn_status __wrap_arm_requantize_s16_s16(
    const int16_t* input, int16_t* output, int32_t size, int32_t multiplier,
    int32_t shift, int32_t input_zero_point, int32_t output_zero_point) {
  ++g_requantize_call.calls;
  g_requantize_call.size = size;
  g_requantize_call.multiplier = multiplier;
  g_requantize_call.shift = shift;
  g_requantize_call.input_zero_point = input_zero_point;
  g_requantize_call.output_zero_point = output_zero_point;
  return __real_arm_requantize_s16_s16(input, output, size, multiplier, shift,
                                       input_zero_point, output_zero_point);
}
}
#endif

namespace tflite {
namespace testing {
namespace {

template <int N>
void ExpectS16Requantize(const int16_t (&input)[N], float input_scale,
                         int32_t expected_shift, const int16_t (&expected)[N]) {
#if !defined(HELIA_QUANTIZE_S16_LINK_WRAP)
  (void)expected_shift;
#endif
  int dims_data[] = {1, N};
  TfLiteIntArray* dims = IntArrayFromInts(dims_data);
  constexpr int16_t kSentinel = 0x5a5a;
  int16_t output[N + 1] = {};
  output[N] = kSentinel;
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input, dims, input_scale, 0),
      CreateQuantizedTensor(output, dims, 1.0f, 0),
  };
  float input_scales[] = {1, input_scale};
  float output_scales[] = {1, 1.0f};
  int zero_points[] = {1, 0};
  TfLiteAffineQuantization quantization[2] = {};
  quantization[0].scale = FloatArrayFromFloats(input_scales);
  quantization[1].scale = FloatArrayFromFloats(output_scales);
  for (int i = 0; i < 2; ++i) {
    quantization[i].zero_point = IntArrayFromInts(zero_points);
    tensors[i].quantization = {kTfLiteAffineQuantization, &quantization[i]};
  }
  int inputs_data[] = {1, 0};
  int outputs_data[] = {1, 1};
  const TFLMRegistration registration = Register_QUANTIZE();
  micro::KernelRunner runner(registration, tensors, 2,
                             IntArrayFromInts(inputs_data),
                             IntArrayFromInts(outputs_data), nullptr);
#if defined(HELIA_QUANTIZE_S16_LINK_WRAP)
  g_requantize_call = RequantizeCall{};
#endif
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
#if defined(HELIA_QUANTIZE_S16_LINK_WRAP)
  EXPECT_EQ(0, g_requantize_call.calls);
#endif
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
#if defined(HELIA_QUANTIZE_S16_LINK_WRAP)
  EXPECT_EQ(1, g_requantize_call.calls);
  EXPECT_EQ(N, g_requantize_call.size);
  EXPECT_EQ(1 << 30, g_requantize_call.multiplier);
  EXPECT_EQ(expected_shift, g_requantize_call.shift);
  EXPECT_EQ(0, g_requantize_call.input_zero_point);
  EXPECT_EQ(0, g_requantize_call.output_zero_point);
#endif
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(expected[i], output[i]);
  }
  EXPECT_EQ(kSentinel, output[N]);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(HeliaQuantizeS16Test, LargeScalePositiveSaturation) {
  const int16_t input[] = {1000};
  const int16_t expected[] = {32767};
  tflite::testing::ExpectS16Requantize(input, 2097152.0f, 22, expected);
}

TEST(HeliaQuantizeS16Test, LargeScaleBodyAndTail) {
  const int16_t input[] = {1000, -1000, 0, 1, -1, 32767, -32768, 2, -2};
  // Exact scale ratio 2^21: every nonzero input saturates with its sign.
  const int16_t expected[] = {32767, -32768, 0,     32767, -32768,
                              32767, -32768, 32767, -32768};
  tflite::testing::ExpectS16Requantize(input, 2097152.0f, 22, expected);
}

TEST(HeliaQuantizeS16Test, OrdinaryScalePreservesValues) {
  const int16_t input[] = {1000, -1000, 0, 32767, -32768};
  tflite::testing::ExpectS16Requantize(input, 1.0f, 1, input);
}

TF_LITE_MICRO_TESTS_MAIN
