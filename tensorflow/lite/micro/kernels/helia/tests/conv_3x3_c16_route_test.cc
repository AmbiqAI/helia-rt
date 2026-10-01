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

// int8 CONV_2D through the helia kernel, compared bit for bit with the
// reference integer implementation, on layers inside and just outside the gate
// of arm_convolve_s8_3x3_c16_s1 (input and filter depth 16, 3x3, stride and
// dilation 1). On MVE builds arm_convolve_wrapper_s8 runs that entry for
// layers in the gate and arm_convolve_s8 otherwise; GCC and ATfE links wrap
// both leaves and each case asserts which one its invoke reached.
// see AmbiqAI/helia-rt#371

#include <cstdint>
#include <cstring>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

enum class Leaf { kC16S1, kS8, kCount };

int g_leaf_calls[static_cast<int>(Leaf::kCount)] = {};

#if HELIA_CONV_ROUTE_LINK_WRAP
void CountCall(Leaf leaf) { ++g_leaf_calls[static_cast<int>(leaf)]; }
#endif

}  // namespace

#if HELIA_CONV_ROUTE_LINK_WRAP
extern "C" {

arm_cmsis_nn_status __real_arm_convolve_s8_3x3_c16_s1(
    const cmsis_nn_context*, const cmsis_nn_context*,
    const cmsis_nn_conv_params*, const cmsis_nn_per_channel_quant_params*,
    const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*, const int8_t*,
    const cmsis_nn_dims*, const int32_t*, const cmsis_nn_dims*,
    const cmsis_nn_dims*, int8_t*);
arm_cmsis_nn_status __wrap_arm_convolve_s8_3x3_c16_s1(
    const cmsis_nn_context* ctx, const cmsis_nn_context* weight_sum_ctx,
    const cmsis_nn_conv_params* conv_params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int8_t* input_data,
    const cmsis_nn_dims* filter_dims, const int8_t* filter_data,
    const cmsis_nn_dims* bias_dims, const int32_t* bias_data,
    const cmsis_nn_dims* upscale_dims, const cmsis_nn_dims* output_dims,
    int8_t* output_data) {
  CountCall(Leaf::kC16S1);
  return __real_arm_convolve_s8_3x3_c16_s1(
      ctx, weight_sum_ctx, conv_params, quant_params, input_dims, input_data,
      filter_dims, filter_data, bias_dims, bias_data, upscale_dims,
      output_dims, output_data);
}

arm_cmsis_nn_status __real_arm_convolve_s8(
    const cmsis_nn_context*, const cmsis_nn_context*,
    const cmsis_nn_conv_params*, const cmsis_nn_per_channel_quant_params*,
    const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*, const int8_t*,
    const cmsis_nn_dims*, const int32_t*, const cmsis_nn_dims*,
    const cmsis_nn_dims*, int8_t*);
arm_cmsis_nn_status __wrap_arm_convolve_s8(
    const cmsis_nn_context* ctx, const cmsis_nn_context* weight_sum_ctx,
    const cmsis_nn_conv_params* conv_params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int8_t* input_data,
    const cmsis_nn_dims* filter_dims, const int8_t* filter_data,
    const cmsis_nn_dims* bias_dims, const int32_t* bias_data,
    const cmsis_nn_dims* upscale_dims, const cmsis_nn_dims* output_dims,
    int8_t* output_data) {
  CountCall(Leaf::kS8);
  return __real_arm_convolve_s8(ctx, weight_sum_ctx, conv_params, quant_params,
                                input_dims, input_data, filter_dims,
                                filter_data, bias_dims, bias_data,
                                upscale_dims, output_dims, output_data);
}

}  // extern "C"
#endif  // HELIA_CONV_ROUTE_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

struct Shape {
  int batch, in_h, in_w, in_c, out_c, stride, dilation;
  TfLitePadding padding;
};

enum class Route { kC16S1, kGeneric };

// The wrapper runs the direct entry only on MVE builds; elsewhere every layer
// takes arm_convolve_s8.
void ExpectRoute(Route route) {
#if HELIA_CONV_ROUTE_LINK_WRAP
#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
  const bool direct = route == Route::kC16S1;
#else
  (void)route;
  const bool direct = false;
#endif
  EXPECT_EQ(direct ? 1 : 0, g_leaf_calls[static_cast<int>(Leaf::kC16S1)]);
  EXPECT_EQ(direct ? 0 : 1, g_leaf_calls[static_cast<int>(Leaf::kS8)]);
#else
  (void)route;
#endif
}

constexpr int kKernel = 3;
constexpr int kMaxInput = 2 * 12 * 12 * 16;
constexpr int kMaxFilter = 24 * kKernel * kKernel * 16;
constexpr int kMaxChannels = 24;
constexpr int kMaxOutput = 2 * 12 * 12 * 24;

uint32_t g_seed = 371;
int NextInt(int lo, int hi) {
  g_seed = g_seed * 1103515245u + 12345u;
  return lo + static_cast<int>((g_seed >> 16) % (hi - lo + 1));
}

int OutSize(TfLitePadding padding, int in, int stride, int dilation) {
  const int effective = (kKernel - 1) * dilation + 1;
  return padding == kTfLitePaddingSame ? (in + stride - 1) / stride
                                       : (in - effective + stride) / stride;
}

int PadSize(TfLitePadding padding, int in, int out, int stride, int dilation) {
  if (padding != kTfLitePaddingSame) {
    return 0;
  }
  const int effective = (kKernel - 1) * dilation + 1;
  const int total = (out - 1) * stride + effective - in;
  return total > 0 ? total / 2 : 0;
}

void ExpectMatchesReference(const Shape& s, Route route) {
  static int8_t input[kMaxInput];
  static int8_t filter[kMaxFilter];
  static int32_t bias[kMaxChannels];
  static int8_t output[kMaxOutput];
  static int8_t expected[kMaxOutput];

  const int out_h = OutSize(s.padding, s.in_h, s.stride, s.dilation);
  const int out_w = OutSize(s.padding, s.in_w, s.stride, s.dilation);
  const int input_count = s.batch * s.in_h * s.in_w * s.in_c;
  const int filter_count = s.out_c * kKernel * kKernel * s.in_c;
  const int output_count = s.batch * out_h * out_w * s.out_c;
  EXPECT_LE(input_count, kMaxInput);
  EXPECT_LE(filter_count, kMaxFilter);
  EXPECT_LE(output_count, kMaxOutput);
  EXPECT_LE(s.out_c, kMaxChannels);
  if (input_count > kMaxInput || filter_count > kMaxFilter ||
      output_count > kMaxOutput || s.out_c > kMaxChannels) {
    return;
  }

  for (int i = 0; i < input_count; ++i) {
    input[i] = static_cast<int8_t>(NextInt(-127, 127));
  }
  for (int i = 0; i < filter_count; ++i) {
    filter[i] = static_cast<int8_t>(NextInt(-127, 127));
  }
  for (int i = 0; i < s.out_c; ++i) {
    bias[i] = NextInt(-2000, 2000);
  }

  const float input_scale = 0.05f;
  const int input_zero_point = 3;
  const float output_scale = 0.8f;
  const int output_zero_point = -5;
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

  int input_dims[] = {4, s.batch, s.in_h, s.in_w, s.in_c};
  int filter_dims[] = {4, s.out_c, kKernel, kKernel, s.in_c};
  int bias_dims[] = {1, s.out_c};
  int output_dims[] = {4, s.batch, out_h, out_w, s.out_c};
  TfLiteAffineQuantization filter_quant = {FloatArrayFromFloats(filter_scales),
                                           IntArrayFromInts(filter_zero_points),
                                           0};
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

  TfLiteConvParams params = {};
  params.padding = s.padding;
  params.stride_width = s.stride;
  params.stride_height = s.stride;
  params.dilation_width_factor = s.dilation;
  params.dilation_height_factor = s.dilation;
  params.activation = kTfLiteActNone;

  int inputs[] = {3, 0, 1, 2};
  int outputs[] = {1, 3};
  const TFLMRegistration registration = Register_CONV_2D();
  micro::KernelRunner runner(registration, tensors, 4, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  memset(output, 0x55, sizeof(output));
  memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  ExpectRoute(route);

  ConvParams op_params = {};
  op_params.padding_type = s.padding == kTfLitePaddingSame
                               ? PaddingType::kSame
                               : PaddingType::kValid;
  op_params.padding_values.width =
      PadSize(s.padding, s.in_w, out_w, s.stride, s.dilation);
  op_params.padding_values.height =
      PadSize(s.padding, s.in_h, out_h, s.stride, s.dilation);
  op_params.stride_width = s.stride;
  op_params.stride_height = s.stride;
  op_params.dilation_width_factor = s.dilation;
  op_params.dilation_height_factor = s.dilation;
  op_params.input_offset = -input_zero_point;
  op_params.weights_offset = 0;
  op_params.output_offset = output_zero_point;
  op_params.quantized_activation_min = -128;
  op_params.quantized_activation_max = 127;
  const int32_t in_shape[] = {s.batch, s.in_h, s.in_w, s.in_c};
  const int32_t f_shape[] = {s.out_c, kKernel, kKernel, s.in_c};
  const int32_t b_shape[] = {s.out_c};
  const int32_t o_shape[] = {s.batch, out_h, out_w, s.out_c};
  reference_integer_ops::ConvPerChannel(
      op_params, multipliers, shifts, RuntimeShape(4, in_shape), input,
      RuntimeShape(4, f_shape), filter, RuntimeShape(1, b_shape), bias,
      RuntimeShape(4, o_shape), expected);

  int mismatches = 0;
  for (int i = 0; i < output_count; ++i) {
    mismatches += output[i] != expected[i];
  }
  EXPECT_EQ(0, mismatches);
  // Nothing is written past the output tensor.
  int overwrites = 0;
  for (int i = output_count; i < kMaxOutput; ++i) {
    overwrites += output[i] != 0x55;
  }
  EXPECT_EQ(0, overwrites);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ExpectMatchesReference;
using tflite::testing::Route;
using tflite::testing::Shape;

// 9x11 = 99 output pixels, so the entry's pixel tail runs next to the
// padded border.
TEST(HeliaConv3x3C16RouteTest, SamePaddingInGate) {
  ExpectMatchesReference(Shape{1, 9, 11, 16, 24, 1, 1, kTfLitePaddingSame},
                         Route::kC16S1);
}

// No border, 6x9 = 54 output pixels (a pixel tail) and an odd output depth.
TEST(HeliaConv3x3C16RouteTest, ValidPaddingPixelTailInGate) {
  ExpectMatchesReference(Shape{1, 8, 11, 16, 13, 1, 1, kTfLitePaddingValid},
                         Route::kC16S1);
}

TEST(HeliaConv3x3C16RouteTest, BatchTwoInGate) {
  ExpectMatchesReference(Shape{2, 7, 9, 16, 8, 1, 1, kTfLitePaddingSame},
                         Route::kC16S1);
}

TEST(HeliaConv3x3C16RouteTest, StrideTwoIsGeneric) {
  ExpectMatchesReference(Shape{1, 9, 11, 16, 24, 2, 1, kTfLitePaddingSame},
                         Route::kGeneric);
}

TEST(HeliaConv3x3C16RouteTest, DilationTwoIsGeneric) {
  ExpectMatchesReference(Shape{1, 12, 12, 16, 8, 1, 2, kTfLitePaddingValid},
                         Route::kGeneric);
}

TEST(HeliaConv3x3C16RouteTest, InputDepthEightIsGeneric) {
  ExpectMatchesReference(Shape{1, 9, 11, 8, 24, 1, 1, kTfLitePaddingSame},
                         Route::kGeneric);
}

TF_LITE_MICRO_TESTS_MAIN
