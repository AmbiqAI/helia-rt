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

// int8 1x1 CONV_2D through the helia kernel, compared bit for bit with the
// reference integer implementation, on layers inside and just outside the gate
// of arm_convolve_1x1_s8_short_k (no padding, unit stride and dilation, input
// depth 1 to 16). On MVE builds the kernel calls that entry for layers in the
// gate and arm_convolve_wrapper_s8 otherwise or when the entry declines. GCC
// and ATfE links wrap both, assert which ran, and inject entry statuses.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

enum class Leaf { kShortK, kWrapper, kCount };

int g_leaf_calls[static_cast<int>(Leaf::kCount)] = {};

// When set, the wrapped entry returns this status without running.
bool g_inject = false;
arm_cmsis_nn_status g_injected_status = ARM_CMSIS_NN_SUCCESS;

#if HELIA_CONV_SHORT_K_LINK_WRAP
void CountCall(Leaf leaf) { ++g_leaf_calls[static_cast<int>(leaf)]; }
#endif

}  // namespace

#if HELIA_CONV_SHORT_K_LINK_WRAP
extern "C" {

arm_cmsis_nn_status __real_arm_convolve_1x1_s8_short_k(
    const cmsis_nn_context*, const cmsis_nn_context*,
    const cmsis_nn_conv_params*, const cmsis_nn_per_channel_quant_params*,
    const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*, const int8_t*,
    const cmsis_nn_dims*, const int32_t*, const cmsis_nn_dims*, int8_t*);
arm_cmsis_nn_status __wrap_arm_convolve_1x1_s8_short_k(
    const cmsis_nn_context* ctx, const cmsis_nn_context* weight_sum_ctx,
    const cmsis_nn_conv_params* conv_params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int8_t* input_data,
    const cmsis_nn_dims* filter_dims, const int8_t* filter_data,
    const cmsis_nn_dims* bias_dims, const int32_t* bias_data,
    const cmsis_nn_dims* output_dims, int8_t* output_data) {
  CountCall(Leaf::kShortK);
  if (g_inject) {
    return g_injected_status;
  }
  return __real_arm_convolve_1x1_s8_short_k(
      ctx, weight_sum_ctx, conv_params, quant_params, input_dims, input_data,
      filter_dims, filter_data, bias_dims, bias_data, output_dims,
      output_data);
}

arm_cmsis_nn_status __real_arm_convolve_wrapper_s8(
    const cmsis_nn_context*, const cmsis_nn_context*,
    const cmsis_nn_conv_params*, const cmsis_nn_per_channel_quant_params*,
    const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*, const int8_t*,
    const cmsis_nn_dims*, const int32_t*, const cmsis_nn_dims*, int8_t*);
arm_cmsis_nn_status __wrap_arm_convolve_wrapper_s8(
    const cmsis_nn_context* ctx, const cmsis_nn_context* weight_sum_ctx,
    const cmsis_nn_conv_params* conv_params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int8_t* input_data,
    const cmsis_nn_dims* filter_dims, const int8_t* filter_data,
    const cmsis_nn_dims* bias_dims, const int32_t* bias_data,
    const cmsis_nn_dims* output_dims, int8_t* output_data) {
  CountCall(Leaf::kWrapper);
  return __real_arm_convolve_wrapper_s8(
      ctx, weight_sum_ctx, conv_params, quant_params, input_dims, input_data,
      filter_dims, filter_data, bias_dims, bias_data, output_dims,
      output_data);
}

}  // extern "C"
#endif  // HELIA_CONV_SHORT_K_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
constexpr bool kMve = true;
#else
constexpr bool kMve = false;
#endif

struct Shape {
  int batch, in_h, in_w, in_c, out_c, stride;
  float output_scale = 0.3f;
  TfLiteFusedActivation activation = kTfLiteActNone;
};

// The int8 output range TFLite derives from a fused activation.
void ActivationRange(TfLiteFusedActivation activation, float scale,
                     int zero_point, int32_t* lo, int32_t* hi) {
  const auto quantize = [&](float f) {
    return zero_point + static_cast<int32_t>(std::round(f / scale));
  };
  *lo = -128;
  *hi = 127;
  if (activation == kTfLiteActRelu || activation == kTfLiteActRelu6) {
    *lo = std::max(*lo, quantize(0.0f));
  }
  if (activation == kTfLiteActRelu6) {
    *hi = std::min(*hi, quantize(6.0f));
  }
}

enum class Route { kShortK, kWrapper };

// Entry calls and wrapper calls for a layer the kernel runs to completion.
// Elsewhere than MVE every layer takes the wrapper.
void ExpectRoute(Route route, bool declined) {
#if HELIA_CONV_SHORT_K_LINK_WRAP
  const bool entry = kMve && route == Route::kShortK;
  EXPECT_EQ(entry ? 1 : 0, g_leaf_calls[static_cast<int>(Leaf::kShortK)]);
  EXPECT_EQ(!entry || declined ? 1 : 0,
            g_leaf_calls[static_cast<int>(Leaf::kWrapper)]);
#else
  (void)kMve;
  (void)route;
  (void)declined;
#endif
}

constexpr int kMaxInput = 2 * 9 * 11 * 17;
constexpr int kMaxFilter = 24 * 17;
constexpr int kMaxChannels = 24;
constexpr int kMaxOutput = 2 * 9 * 11 * 24;

uint32_t g_seed = 1141;
int NextInt(int lo, int hi) {
  g_seed = g_seed * 1103515245u + 12345u;
  return lo + static_cast<int>((g_seed >> 16) % (hi - lo + 1));
}

// Runs the layer and returns the invoke status. On kTfLiteOk the output is
// compared with the reference and the bytes past it are checked untouched.
TfLiteStatus RunAgainstReference(const Shape& s) {
  static int8_t input[kMaxInput];
  static int8_t filter[kMaxFilter];
  static int32_t bias[kMaxChannels];
  static int8_t output[kMaxOutput];
  static int8_t expected[kMaxOutput];

  const int out_h = (s.in_h - 1) / s.stride + 1;
  const int out_w = (s.in_w - 1) / s.stride + 1;
  const int input_count = s.batch * s.in_h * s.in_w * s.in_c;
  const int filter_count = s.out_c * s.in_c;
  const int output_count = s.batch * out_h * out_w * s.out_c;
  EXPECT_LE(input_count, kMaxInput);
  EXPECT_LE(filter_count, kMaxFilter);
  EXPECT_LE(output_count, kMaxOutput);
  EXPECT_LE(s.out_c, kMaxChannels);
  if (input_count > kMaxInput || filter_count > kMaxFilter ||
      output_count > kMaxOutput || s.out_c > kMaxChannels) {
    return kTfLiteError;
  }

  for (int i = 0; i < input_count; ++i) {
    input[i] = static_cast<int8_t>(NextInt(-128, 127));
  }
  for (int i = 0; i < filter_count; ++i) {
    filter[i] = static_cast<int8_t>(NextInt(-127, 127));
  }
  for (int i = 0; i < s.out_c; ++i) {
    bias[i] = NextInt(-2000, 2000);
  }

  const float input_scale = 0.05f;
  const int input_zero_point = 3;
  const float output_scale = s.output_scale;
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
  int filter_dims[] = {4, s.out_c, 1, 1, s.in_c};
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
  params.padding = kTfLitePaddingValid;
  params.stride_width = s.stride;
  params.stride_height = s.stride;
  params.dilation_width_factor = 1;
  params.dilation_height_factor = 1;
  params.activation = s.activation;

  int inputs[] = {3, 0, 1, 2};
  int outputs[] = {1, 3};
  const TFLMRegistration registration = Register_CONV_2D();
  micro::KernelRunner runner(registration, tensors, 4, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  memset(output, 0x55, sizeof(output));
  memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
  const TfLiteStatus invoke = runner.Invoke();
  if (invoke != kTfLiteOk) {
    return invoke;
  }

  ConvParams op_params = {};
  op_params.padding_type = PaddingType::kValid;
  op_params.stride_width = s.stride;
  op_params.stride_height = s.stride;
  op_params.dilation_width_factor = 1;
  op_params.dilation_height_factor = 1;
  op_params.input_offset = -input_zero_point;
  op_params.weights_offset = 0;
  op_params.output_offset = output_zero_point;
  ActivationRange(s.activation, output_scale, output_zero_point,
                  &op_params.quantized_activation_min,
                  &op_params.quantized_activation_max);
  const int32_t in_shape[] = {s.batch, s.in_h, s.in_w, s.in_c};
  const int32_t f_shape[] = {s.out_c, 1, 1, s.in_c};
  const int32_t b_shape[] = {s.out_c};
  const int32_t o_shape[] = {s.batch, out_h, out_w, s.out_c};
  tflite::micro::reference_integer_ops::ConvPerChannel(
      op_params, multipliers, shifts, RuntimeShape(4, in_shape), input,
      RuntimeShape(4, f_shape), filter, RuntimeShape(1, b_shape), bias,
      RuntimeShape(4, o_shape), expected);

  int mismatches = 0;
  for (int i = 0; i < output_count; ++i) {
    mismatches += output[i] != expected[i];
  }
  EXPECT_EQ(0, mismatches);
  int overwrites = 0;
  for (int i = output_count; i < kMaxOutput; ++i) {
    overwrites += output[i] != 0x55;
  }
  EXPECT_EQ(0, overwrites);
  return kTfLiteOk;
}

void ExpectMatchesReference(const Shape& s, Route route) {
  EXPECT_EQ(kTfLiteOk, RunAgainstReference(s));
  ExpectRoute(route, /*declined=*/false);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ExpectMatchesReference;
using tflite::testing::Route;
using tflite::testing::Shape;

// 5x7 = 35 pixels, so the entry's four-pixel steps leave a tail.
TEST(HeliaConv1x1ShortKRouteTest, DepthSixteenPixelTailInGate) {
  ExpectMatchesReference(Shape{1, 5, 7, 16, 24, 1}, Route::kShortK);
}

// Depth 8 loads two pixels per vector.
TEST(HeliaConv1x1ShortKRouteTest, DepthEightInGate) {
  ExpectMatchesReference(Shape{1, 9, 9, 8, 13, 1}, Route::kShortK);
}

TEST(HeliaConv1x1ShortKRouteTest, DepthOneInGate) {
  ExpectMatchesReference(Shape{1, 6, 6, 1, 7, 1}, Route::kShortK);
}

TEST(HeliaConv1x1ShortKRouteTest, BatchTwoDepthThreeInGate) {
  ExpectMatchesReference(Shape{2, 4, 5, 3, 16, 1}, Route::kShortK);
}

// An output scale below the input x filter scales gives effective multipliers
// from 0.5 to 2.5, so channels requantize with shift 0 and with shift > 0.
TEST(HeliaConv1x1ShortKRouteTest, NonNegativeShiftInGate) {
  ExpectMatchesReference(Shape{1, 5, 7, 16, 24, 1, 2e-4f}, Route::kShortK);
}

TEST(HeliaConv1x1ShortKRouteTest, FusedRelu6InGate) {
  ExpectMatchesReference(Shape{1, 5, 7, 16, 24, 1, 0.05f, kTfLiteActRelu6},
                         Route::kShortK);
}

TEST(HeliaConv1x1ShortKRouteTest, DepthSeventeenIsWrapper) {
  ExpectMatchesReference(Shape{1, 5, 7, 17, 24, 1}, Route::kWrapper);
}

TEST(HeliaConv1x1ShortKRouteTest, StrideTwoIsWrapper) {
  ExpectMatchesReference(Shape{1, 9, 11, 16, 24, 2}, Route::kWrapper);
}

// A declining entry leaves the layer to the wrapper.
TEST(HeliaConv1x1ShortKRouteTest, DeclinedEntryFallsBackToWrapper) {
  g_inject = true;
  g_injected_status = ARM_CMSIS_NN_NO_IMPL_ERROR;
  EXPECT_EQ(kTfLiteOk,
            tflite::testing::RunAgainstReference(Shape{1, 5, 7, 16, 24, 1}));
  g_inject = false;
  tflite::testing::ExpectRoute(Route::kShortK, /*declined=*/true);
}

// Any other entry status fails the invoke without running the wrapper.
TEST(HeliaConv1x1ShortKRouteTest, EntryErrorFailsInvoke) {
  g_inject = true;
  g_injected_status = ARM_CMSIS_NN_ARG_ERROR;
  const TfLiteStatus invoke =
      tflite::testing::RunAgainstReference(Shape{1, 5, 7, 16, 24, 1});
  g_inject = false;
#if HELIA_CONV_SHORT_K_LINK_WRAP
  if (tflite::testing::kMve) {
    EXPECT_EQ(kTfLiteError, invoke);
    EXPECT_EQ(1, g_leaf_calls[static_cast<int>(Leaf::kShortK)]);
    EXPECT_EQ(0, g_leaf_calls[static_cast<int>(Leaf::kWrapper)]);
    return;
  }
#endif
  EXPECT_EQ(kTfLiteOk, invoke);
}

TF_LITE_MICRO_TESTS_MAIN
