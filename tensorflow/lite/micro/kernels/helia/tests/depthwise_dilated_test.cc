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

// DEPTHWISE_CONV_2D int8 and 16x8 through the helia kernel, compared bit for
// bit with the reference integer implementation. Dilated 1D layers take the
// optimized heliaCORE kernels (s8_opt, fast_s16); the non-dilated 16x8 cases
// pin that routing 16x8 through the wrapper leaves those results unchanged.
// see AmbiqAI/helia-rt#314

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

struct Shape {
  int in_h, in_w, channels, k_h, k_w, dil_w, ch_mult;
  TfLitePadding padding;
};

constexpr int kMaxInput = 240 * 35;
constexpr int kMaxFilter = 9 * 64;
constexpr int kMaxChannels = 64;
constexpr int kMaxOutput = 240 * 64;

uint32_t g_seed = 12345;
int NextInt(int lo, int hi) {
  g_seed = g_seed * 1103515245u + 12345u;
  return lo + static_cast<int>((g_seed >> 16) % (hi - lo + 1));
}

int OutSize(TfLitePadding padding, int in, int k, int dil) {
  const int effective = (k - 1) * dil + 1;
  return padding == kTfLitePaddingSame ? in : in - effective + 1;
}

template <typename TIn, typename TBias>
void ExpectMatchesReference(const Shape& s) {
  static TIn input[kMaxInput];
  static int8_t filter[kMaxFilter];
  static TBias bias[kMaxChannels];
  static TIn output[kMaxOutput];
  static TIn expected[kMaxOutput];
  constexpr bool kIs16 = sizeof(TIn) == 2;

  const int out_c = s.channels * s.ch_mult;
  const int out_h = OutSize(s.padding, s.in_h, s.k_h, 1);
  const int out_w = OutSize(s.padding, s.in_w, s.k_w, s.dil_w);
  const int input_count = s.in_h * s.in_w * s.channels;
  const int filter_count = s.k_h * s.k_w * out_c;
  const int output_count = out_h * out_w * out_c;
  EXPECT_LE(input_count, kMaxInput);
  EXPECT_LE(filter_count, kMaxFilter);
  EXPECT_LE(output_count, kMaxOutput);
  EXPECT_LE(out_c, kMaxChannels);
  if (input_count > kMaxInput || filter_count > kMaxFilter ||
      output_count > kMaxOutput || out_c > kMaxChannels) {
    return;
  }

  for (int i = 0; i < input_count; ++i) {
    input[i] =
        static_cast<TIn>(kIs16 ? NextInt(-1000, 1000) : NextInt(-127, 127));
  }
  for (int i = 0; i < filter_count; ++i) {
    filter[i] = static_cast<int8_t>(NextInt(-127, 127));
  }
  for (int i = 0; i < out_c; ++i) {
    bias[i] = static_cast<TBias>(NextInt(-1000, 1000));
  }

  const float input_scale = kIs16 ? 1.0f / 1024 : 0.05f;
  const int input_zero_point = kIs16 ? 0 : 3;
  const float output_scale = kIs16 ? 1.0f / 64 : 0.4f;
  const int output_zero_point = kIs16 ? 0 : -5;
  float filter_scales[kMaxChannels + 1] = {static_cast<float>(out_c)};
  int filter_zero_points[kMaxChannels + 1] = {out_c};
  float bias_scales[kMaxChannels + 1] = {static_cast<float>(out_c)};
  int bias_zero_points[kMaxChannels + 1] = {out_c};
  int32_t multipliers[kMaxChannels];
  int32_t shifts[kMaxChannels];
  for (int c = 0; c < out_c; ++c) {
    filter_scales[c + 1] = 0.002f * (1 + (c % 5));
    filter_zero_points[c + 1] = 0;
    bias_scales[c + 1] = input_scale * filter_scales[c + 1];
    bias_zero_points[c + 1] = 0;
    int32_t multiplier = 0;
    int shift = 0;
    QuantizeMultiplier(
        static_cast<double>(input_scale) * filter_scales[c + 1] / output_scale,
        &multiplier, &shift);
    multipliers[c] = multiplier;
    shifts[c] = shift;
  }

  int input_dims[] = {4, 1, s.in_h, s.in_w, s.channels};
  int filter_dims[] = {4, 1, s.k_h, s.k_w, out_c};
  int bias_dims[] = {1, out_c};
  int output_dims[] = {4, 1, out_h, out_w, out_c};
  TfLiteAffineQuantization filter_quant = {FloatArrayFromFloats(filter_scales),
                                           IntArrayFromInts(filter_zero_points),
                                           3};
  TfLiteAffineQuantization bias_quant = {FloatArrayFromFloats(bias_scales),
                                         IntArrayFromInts(bias_zero_points), 0};
  TfLiteTensor tensors[4] = {
      CreateQuantizedTensor(input, IntArrayFromInts(input_dims), input_scale,
                            input_zero_point),
      CreateTensor(filter, IntArrayFromInts(filter_dims)),
      CreateTensor(bias, IntArrayFromInts(bias_dims)),
      CreateQuantizedTensor(output, IntArrayFromInts(output_dims), output_scale,
                            output_zero_point),
  };
  tensors[1].quantization = {kTfLiteAffineQuantization, &filter_quant};
  tensors[2].quantization = {kTfLiteAffineQuantization, &bias_quant};

  TfLiteDepthwiseConvParams params = {};
  params.padding = s.padding;
  params.stride_width = 1;
  params.stride_height = 1;
  params.dilation_width_factor = s.dil_w;
  params.dilation_height_factor = 1;
  params.depth_multiplier = s.ch_mult;
  params.activation = kTfLiteActNone;

  int inputs[] = {3, 0, 1, 2};
  int outputs[] = {1, 3};
  const TFLMRegistration registration =
      kIs16 ? Register_DEPTHWISE_CONV_2D_INT16() : Register_DEPTHWISE_CONV_2D();
  micro::KernelRunner runner(registration, tensors, 4, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  DepthwiseParams op_params = {};
  op_params.padding_type = s.padding == kTfLitePaddingSame
                               ? PaddingType::kSame
                               : PaddingType::kValid;
  op_params.padding_values.width =
      s.padding == kTfLitePaddingSame ? ((s.k_w - 1) * s.dil_w) / 2 : 0;
  op_params.padding_values.height =
      s.padding == kTfLitePaddingSame ? (s.k_h - 1) / 2 : 0;
  op_params.stride_width = 1;
  op_params.stride_height = 1;
  op_params.dilation_width_factor = s.dil_w;
  op_params.dilation_height_factor = 1;
  op_params.depth_multiplier = s.ch_mult;
  op_params.input_offset = -input_zero_point;
  op_params.weights_offset = 0;
  op_params.output_offset = output_zero_point;
  op_params.quantized_activation_min = kIs16 ? -32768 : -128;
  op_params.quantized_activation_max = kIs16 ? 32767 : 127;
  const int32_t in_shape[] = {1, s.in_h, s.in_w, s.channels};
  const int32_t f_shape[] = {1, s.k_h, s.k_w, out_c};
  const int32_t b_shape[] = {out_c};
  const int32_t o_shape[] = {1, out_h, out_w, out_c};
  reference_integer_ops::DepthwiseConvPerChannel(
      op_params, multipliers, shifts, RuntimeShape(4, in_shape), input,
      RuntimeShape(4, f_shape), filter, RuntimeShape(1, b_shape), bias,
      RuntimeShape(4, o_shape), expected);

  int mismatches = 0;
  for (int i = 0; i < output_count; ++i) {
    mismatches += output[i] != expected[i];
  }
  EXPECT_EQ(0, mismatches);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ExpectMatchesReference;
using tflite::testing::Shape;

TEST(HeliaDepthwiseDilatedTest, Int8Dilated1DValid) {
  ExpectMatchesReference<int8_t, int32_t>(
      Shape{1, 240, 32, 1, 7, 8, 1, kTfLitePaddingValid});
}

// SAME padding, and a channel count that is not a multiple of four, so the
// optimized kernel's channel tail runs.
TEST(HeliaDepthwiseDilatedTest, Int8Dilated1DSameOddChannels) {
  ExpectMatchesReference<int8_t, int32_t>(
      Shape{1, 96, 35, 1, 5, 4, 1, kTfLitePaddingSame});
}

TEST(HeliaDepthwiseDilatedTest, Int16x8Dilated1DValid) {
  ExpectMatchesReference<int16_t, int64_t>(
      Shape{1, 240, 32, 1, 7, 8, 1, kTfLitePaddingValid});
}

TEST(HeliaDepthwiseDilatedTest, Int16x8Dilated1DSameOddChannels) {
  ExpectMatchesReference<int16_t, int64_t>(
      Shape{1, 96, 35, 1, 5, 4, 1, kTfLitePaddingSame});
}

// Non-dilated 16x8 layers that the wrapper now routes to fast_s16.
TEST(HeliaDepthwiseDilatedTest, Int16x8Undilated2D) {
  ExpectMatchesReference<int16_t, int64_t>(
      Shape{9, 11, 19, 3, 3, 1, 1, kTfLitePaddingSame});
}

// ch_mult > 1 stays on arm_depthwise_conv_s16.
TEST(HeliaDepthwiseDilatedTest, Int16x8ChannelMultiplier) {
  ExpectMatchesReference<int16_t, int64_t>(
      Shape{5, 7, 8, 3, 3, 1, 2, kTfLitePaddingValid});
}

TF_LITE_MICRO_TESTS_MAIN
