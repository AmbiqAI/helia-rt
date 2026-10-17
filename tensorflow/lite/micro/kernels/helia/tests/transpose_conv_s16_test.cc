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

// 16x8 TRANSPOSE_CONV through the helia kernel, compared bit for bit with the
// reference integer implementation over strides, paddings, kernel sizes and
// int64, int16 and absent biases. GCC and ATfE links wrap
// arm_transpose_conv_s16 and assert each invoke ran it once.

#include <cstdint>
#include <cstring>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/transpose_conv.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/transpose_conv.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

int g_s16_calls = 0;

}  // namespace

#if HELIA_TRANSPOSE_CONV_S16_LINK_WRAP
extern "C" {

arm_cmsis_nn_status __real_arm_transpose_conv_s16(
    const cmsis_nn_context*, const cmsis_nn_context*,
    const cmsis_nn_transpose_conv_params*,
    const cmsis_nn_per_channel_quant_params*, const cmsis_nn_dims*,
    const int16_t*, const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*,
    const int64_t*, const cmsis_nn_dims*, int16_t*);
arm_cmsis_nn_status __wrap_arm_transpose_conv_s16(
    const cmsis_nn_context* ctx, const cmsis_nn_context* output_ctx,
    const cmsis_nn_transpose_conv_params* transpose_conv_params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int16_t* input_data,
    const cmsis_nn_dims* filter_dims, const int8_t* filter_data,
    const cmsis_nn_dims* bias_dims, const int64_t* bias_data,
    const cmsis_nn_dims* output_dims, int16_t* output_data) {
  ++g_s16_calls;
  return __real_arm_transpose_conv_s16(
      ctx, output_ctx, transpose_conv_params, quant_params, input_dims,
      input_data, filter_dims, filter_data, bias_dims, bias_data, output_dims,
      output_data);
}

}  // extern "C"
#endif  // HELIA_TRANSPOSE_CONV_S16_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

enum class Bias { kInt64, kInt16, kNone };

struct Shape {
  int in_h, in_w, in_c, out_c, kernel, stride;
  TfLitePadding padding;
  Bias bias;
};

constexpr int kMaxInput = 6 * 7 * 16;
constexpr int kMaxFilter = 8 * 5 * 5 * 16;
constexpr int kMaxChannels = 8;
constexpr int kMaxOutput = 16 * 18 * 8;

uint32_t g_seed = 1066;
int NextInt(int lo, int hi) {
  g_seed = g_seed * 1103515245u + 12345u;
  return lo + static_cast<int>((g_seed >> 16) % (hi - lo + 1));
}

int OutSize(TfLitePadding padding, int in, int kernel, int stride) {
  return padding == kTfLitePaddingSame ? in * stride
                                       : (in - 1) * stride + kernel;
}

// TFLite pads a transpose convolution as the forward convolution from its
// output back to its input.
int PadSize(TfLitePadding padding, int in, int out, int kernel, int stride) {
  if (padding != kTfLitePaddingSame) {
    return 0;
  }
  const int total = (in - 1) * stride + kernel - out;
  return total > 0 ? total / 2 : 0;
}

void ExpectMatchesReference(const Shape& s) {
  static int16_t input[kMaxInput];
  static int8_t filter[kMaxFilter];
  static int64_t bias64[kMaxChannels];
  static int16_t bias16[kMaxChannels];
  static int16_t output[kMaxOutput];
  static int16_t expected[kMaxOutput];
  static int64_t scratch[kMaxOutput];

  const int out_h = OutSize(s.padding, s.in_h, s.kernel, s.stride);
  const int out_w = OutSize(s.padding, s.in_w, s.kernel, s.stride);
  const int input_count = s.in_h * s.in_w * s.in_c;
  const int filter_count = s.out_c * s.kernel * s.kernel * s.in_c;
  const int output_count = out_h * out_w * s.out_c;
  EXPECT_LE(input_count, kMaxInput);
  EXPECT_LE(filter_count, kMaxFilter);
  EXPECT_LE(output_count, kMaxOutput);
  EXPECT_LE(s.out_c, kMaxChannels);
  if (input_count > kMaxInput || filter_count > kMaxFilter ||
      output_count > kMaxOutput || s.out_c > kMaxChannels) {
    return;
  }

  for (int i = 0; i < input_count; ++i) {
    input[i] = static_cast<int16_t>(NextInt(-4000, 4000));
  }
  for (int i = 0; i < filter_count; ++i) {
    filter[i] = static_cast<int8_t>(NextInt(-127, 127));
  }
  for (int i = 0; i < s.out_c; ++i) {
    bias16[i] = static_cast<int16_t>(NextInt(-2000, 2000));
    bias64[i] = s.bias == Bias::kInt16 ? bias16[i]
                                       : static_cast<int64_t>(NextInt(
                                             -(1 << 20), 1 << 20));
  }

  const float input_scale = 1.0f / 1024;
  const float output_scale = 1.0f / 64;
  float filter_scales[kMaxChannels + 1] = {static_cast<float>(s.out_c)};
  int filter_zero_points[kMaxChannels + 1] = {s.out_c};
  float bias_scales[kMaxChannels + 1] = {static_cast<float>(s.out_c)};
  int bias_zero_points[kMaxChannels + 1] = {s.out_c};
  int32_t multipliers[kMaxChannels];
  int32_t shifts[kMaxChannels];
  for (int c = 0; c < s.out_c; ++c) {
    filter_scales[c + 1] = 0.002f * (1 + (c % 5));
    filter_zero_points[c + 1] = 0;
    bias_scales[c + 1] = input_scale * filter_scales[c + 1];
    bias_zero_points[c + 1] = 0;
    int32_t multiplier = 0;
    int shift = 0;
    QuantizeMultiplier(static_cast<double>(input_scale) *
                           static_cast<double>(filter_scales[c + 1]) /
                           static_cast<double>(output_scale),
                       &multiplier, &shift);
    multipliers[c] = multiplier;
    shifts[c] = shift;
  }

  int output_shape_dims[] = {1, 0};
  int32_t* output_shape = nullptr;
  int input_dims[] = {4, 1, s.in_h, s.in_w, s.in_c};
  int filter_dims[] = {4, s.out_c, s.kernel, s.kernel, s.in_c};
  int bias_dims[] = {1, s.out_c};
  int output_dims[] = {4, 1, out_h, out_w, s.out_c};
  TfLiteAffineQuantization filter_quant = {FloatArrayFromFloats(filter_scales),
                                           IntArrayFromInts(filter_zero_points),
                                           0};
  TfLiteAffineQuantization bias_quant = {FloatArrayFromFloats(bias_scales),
                                         IntArrayFromInts(bias_zero_points), 0};
  TfLiteTensor tensors[5] = {
      CreateTensor(output_shape, IntArrayFromInts(output_shape_dims)),
      CreateTensor(filter, IntArrayFromInts(filter_dims)),
      CreateQuantizedTensor(input, IntArrayFromInts(input_dims), input_scale,
                            0),
      s.bias == Bias::kInt16
          ? CreateTensor(bias16, IntArrayFromInts(bias_dims))
          : CreateTensor(bias64, IntArrayFromInts(bias_dims)),
      CreateQuantizedTensor(output, IntArrayFromInts(output_dims), output_scale,
                            0),
  };
  tensors[1].quantization = {kTfLiteAffineQuantization, &filter_quant};
  tensors[3].quantization = {kTfLiteAffineQuantization, &bias_quant};

  TfLiteTransposeConvParams params = {};
  params.padding = s.padding;
  params.stride_width = s.stride;
  params.stride_height = s.stride;
  params.activation = kTfLiteActNone;

  const bool has_bias = s.bias != Bias::kNone;
  int inputs_with_bias[] = {4, 0, 1, 2, 3};
  int inputs_without_bias[] = {3, 0, 1, 2};
  int outputs[] = {1, 4};
  const TFLMRegistration registration = Register_TRANSPOSE_CONV();
  micro::KernelRunner runner(
      registration, tensors, 5,
      IntArrayFromInts(has_bias ? inputs_with_bias : inputs_without_bias),
      IntArrayFromInts(outputs), &params);
  EXPECT_EQ(kTfLiteOk,
            runner.InitAndPrepare(reinterpret_cast<const char*>(&params)));
  memset(output, 0x55, sizeof(output));
  g_s16_calls = 0;
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
#if HELIA_TRANSPOSE_CONV_S16_LINK_WRAP
  EXPECT_EQ(1, g_s16_calls);
#endif

  ConvParams op_params = {};
  op_params.padding_type = s.padding == kTfLitePaddingSame
                               ? PaddingType::kSame
                               : PaddingType::kValid;
  op_params.padding_values.width =
      PadSize(s.padding, s.in_w, out_w, s.kernel, s.stride);
  op_params.padding_values.height =
      PadSize(s.padding, s.in_h, out_h, s.kernel, s.stride);
  op_params.stride_width = s.stride;
  op_params.stride_height = s.stride;
  op_params.quantized_activation_min = -32768;
  op_params.quantized_activation_max = 32767;
  const int32_t in_shape[] = {1, s.in_h, s.in_w, s.in_c};
  const int32_t f_shape[] = {s.out_c, s.kernel, s.kernel, s.in_c};
  const int32_t b_shape[] = {s.out_c};
  const int32_t o_shape[] = {1, out_h, out_w, s.out_c};
  tflite::micro::reference_integer_ops::TransposeConv(
      op_params, multipliers, shifts, RuntimeShape(4, in_shape), input,
      RuntimeShape(4, f_shape), filter, RuntimeShape(1, b_shape),
      has_bias ? bias64 : nullptr, RuntimeShape(4, o_shape), expected,
      RuntimeShape(), nullptr, scratch);

  int mismatches = 0;
  for (int i = 0; i < output_count; ++i) {
    mismatches += output[i] != expected[i];
  }
  EXPECT_EQ(0, mismatches);
  int overwrites = 0;
  for (int i = output_count; i < kMaxOutput; ++i) {
    overwrites += output[i] != 0x5555;
  }
  EXPECT_EQ(0, overwrites);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::Bias;
using tflite::testing::ExpectMatchesReference;
using tflite::testing::Shape;

TEST(HeliaTransposeConvS16Test, SameStrideTwoInt64Bias) {
  ExpectMatchesReference(
      Shape{4, 5, 3, 5, 3, 2, kTfLitePaddingSame, Bias::kInt64});
}

TEST(HeliaTransposeConvS16Test, ValidStrideOneInt16Bias) {
  ExpectMatchesReference(
      Shape{5, 6, 4, 6, 3, 1, kTfLitePaddingValid, Bias::kInt16});
}

TEST(HeliaTransposeConvS16Test, ValidStrideTwoEvenKernelNoBias) {
  ExpectMatchesReference(
      Shape{6, 7, 5, 3, 2, 2, kTfLitePaddingValid, Bias::kNone});
}

// Kernel 4 at stride 1 pads an odd total, so the extra row and column fall
// at the end.
TEST(HeliaTransposeConvS16Test, SameOddTotalPaddingInt64Bias) {
  ExpectMatchesReference(
      Shape{5, 5, 2, 4, 4, 1, kTfLitePaddingSame, Bias::kInt64});
}

TEST(HeliaTransposeConvS16Test, DeepInputStrideTwoInt16Bias) {
  ExpectMatchesReference(
      Shape{6, 7, 16, 8, 3, 2, kTfLitePaddingSame, Bias::kInt16});
}

TEST(HeliaTransposeConvS16Test, FiveByFiveKernelStrideThree) {
  ExpectMatchesReference(
      Shape{3, 4, 6, 7, 5, 3, kTfLitePaddingValid, Bias::kInt64});
}

TF_LITE_MICRO_TESTS_MAIN
