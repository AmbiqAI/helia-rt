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

// int8 3x3 DEPTHWISE_CONV_2D through the helia kernel, compared bit for bit
// with the reference integer implementation, on layers inside and just outside
// the gate of arm_depthwise_conv_s8_opt_3x3 and its 64-channel, vertical
// stride 1 variant. On MVE builds the kernel calls the matching entry for
// layers in the gate and arm_depthwise_conv_wrapper_s8 otherwise or when the
// entry declines. GCC and ATfE links wrap all three, assert which ran, and
// inject entry statuses.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

enum class Leaf { kOpt3x3, kC64S1, kWrapper, kCount };

int g_leaf_calls[static_cast<int>(Leaf::kCount)] = {};
// Prepare queries the entries' scratch size only for layers in their gate.
int g_opt_3x3_size_queries = 0;

// When set, the wrapped entries return this status without running.
bool g_inject = false;
arm_cmsis_nn_status g_injected_status = ARM_CMSIS_NN_SUCCESS;

#if HELIA_DW_3X3_LINK_WRAP
void CountCall(Leaf leaf) { ++g_leaf_calls[static_cast<int>(leaf)]; }
#endif

}  // namespace

#if HELIA_DW_3X3_LINK_WRAP
extern "C" {

#define HELIA_DW_S8_ARGS                                                     \
  const cmsis_nn_context *ctx, const cmsis_nn_context *weight_sum_ctx,       \
      const cmsis_nn_dw_conv_params *dw_conv_params,                         \
      const cmsis_nn_per_channel_quant_params *quant_params,                 \
      const cmsis_nn_dims *input_dims, const int8_t *input_data,             \
      const cmsis_nn_dims *filter_dims, const int8_t *filter_data,           \
      const cmsis_nn_dims *bias_dims, const int32_t *bias_data,              \
      const cmsis_nn_dims *output_dims, int8_t *output_data
#define HELIA_DW_S8_FORWARD                                                   \
  ctx, weight_sum_ctx, dw_conv_params, quant_params, input_dims, input_data, \
      filter_dims, filter_data, bias_dims, bias_data, output_dims, output_data

arm_cmsis_nn_status __real_arm_depthwise_conv_s8_opt_3x3(HELIA_DW_S8_ARGS);
arm_cmsis_nn_status __wrap_arm_depthwise_conv_s8_opt_3x3(HELIA_DW_S8_ARGS) {
  CountCall(Leaf::kOpt3x3);
  if (g_inject) {
    return g_injected_status;
  }
  return __real_arm_depthwise_conv_s8_opt_3x3(HELIA_DW_S8_FORWARD);
}

arm_cmsis_nn_status __real_arm_depthwise_conv_s8_opt_3x3_c64_s1(
    HELIA_DW_S8_ARGS);
arm_cmsis_nn_status __wrap_arm_depthwise_conv_s8_opt_3x3_c64_s1(
    HELIA_DW_S8_ARGS) {
  CountCall(Leaf::kC64S1);
  if (g_inject) {
    return g_injected_status;
  }
  return __real_arm_depthwise_conv_s8_opt_3x3_c64_s1(HELIA_DW_S8_FORWARD);
}

arm_cmsis_nn_status __real_arm_depthwise_conv_wrapper_s8(HELIA_DW_S8_ARGS);
arm_cmsis_nn_status __wrap_arm_depthwise_conv_wrapper_s8(HELIA_DW_S8_ARGS) {
  CountCall(Leaf::kWrapper);
  return __real_arm_depthwise_conv_wrapper_s8(HELIA_DW_S8_FORWARD);
}

int32_t __real_arm_depthwise_conv_s8_opt_3x3_get_buffer_size(
    const cmsis_nn_dims* input_dims);
int32_t __wrap_arm_depthwise_conv_s8_opt_3x3_get_buffer_size(
    const cmsis_nn_dims* input_dims) {
  ++g_opt_3x3_size_queries;
  return __real_arm_depthwise_conv_s8_opt_3x3_get_buffer_size(input_dims);
}

#undef HELIA_DW_S8_FORWARD
#undef HELIA_DW_S8_ARGS

}  // extern "C"
#endif  // HELIA_DW_3X3_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

#if defined(ARM_MATH_MVEI) && !defined(ARM_MATH_AUTOVECTORIZE)
constexpr bool kMve = true;
#else
constexpr bool kMve = false;
#endif

// stride is the vertical stride, and the horizontal one unless stride_w is set.
struct Shape {
  int in_h, in_w, channels, stride, dilation;
  TfLitePadding padding;
  int stride_w = 0;
  TfLiteFusedActivation activation = kTfLiteActNone;
};

int StrideW(const Shape& s) { return s.stride_w > 0 ? s.stride_w : s.stride; }

// kDeclined: the entry is called, declines, and the wrapper runs the layer.
enum class Route { kOpt3x3, kC64S1, kWrapper, kDeclined };

#if HELIA_DW_3X3_LINK_WRAP
int LeafCalls(Leaf leaf) { return g_leaf_calls[static_cast<int>(leaf)]; }
#endif

// Elsewhere than MVE every layer takes the wrapper.
void ExpectRoute(Route route, Leaf entry) {
#if HELIA_DW_3X3_LINK_WRAP
  const bool tried = kMve && route != Route::kWrapper;
  const bool finished = tried && route != Route::kDeclined;
  EXPECT_EQ(tried && entry == Leaf::kOpt3x3 ? 1 : 0, LeafCalls(Leaf::kOpt3x3));
  EXPECT_EQ(tried && entry == Leaf::kC64S1 ? 1 : 0, LeafCalls(Leaf::kC64S1));
  EXPECT_EQ(finished ? 0 : 1, LeafCalls(Leaf::kWrapper));
  EXPECT_EQ(tried ? 1 : 0, g_opt_3x3_size_queries);
#else
  (void)kMve;
  (void)route;
  (void)entry;
#endif
}

constexpr int kKernel = 3;
constexpr int kMaxInput = 9 * 11 * 130;
constexpr int kMaxFilter = kKernel * kKernel * 130;
constexpr int kMaxChannels = 130;
constexpr int kMaxOutput = 9 * 11 * 130;

uint32_t g_seed = 2743;
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

// Runs the layer and returns the invoke status. On kTfLiteOk the output is
// compared with the reference and the bytes past it are checked untouched.
TfLiteStatus RunAgainstReference(const Shape& s) {
  static int8_t input[kMaxInput];
  static int8_t filter[kMaxFilter];
  static int32_t bias[kMaxChannels];
  static int8_t output[kMaxOutput];
  static int8_t expected[kMaxOutput];

  const int out_h = OutSize(s.padding, s.in_h, s.stride, s.dilation);
  const int out_w = OutSize(s.padding, s.in_w, StrideW(s), s.dilation);
  const int input_count = s.in_h * s.in_w * s.channels;
  const int filter_count = kKernel * kKernel * s.channels;
  const int output_count = out_h * out_w * s.channels;
  EXPECT_LE(input_count, kMaxInput);
  EXPECT_LE(filter_count, kMaxFilter);
  EXPECT_LE(output_count, kMaxOutput);
  EXPECT_LE(s.channels, kMaxChannels);
  if (input_count > kMaxInput || filter_count > kMaxFilter ||
      output_count > kMaxOutput || s.channels > kMaxChannels) {
    return kTfLiteError;
  }

  for (int i = 0; i < input_count; ++i) {
    input[i] = static_cast<int8_t>(NextInt(-128, 127));
  }
  for (int i = 0; i < filter_count; ++i) {
    filter[i] = static_cast<int8_t>(NextInt(-127, 127));
  }
  for (int i = 0; i < s.channels; ++i) {
    bias[i] = NextInt(-2000, 2000);
  }

  const float input_scale = 0.05f;
  const int input_zero_point = 3;
  const float output_scale = 0.4f;
  const int output_zero_point = -5;
  float filter_scales[kMaxChannels + 1] = {static_cast<float>(s.channels)};
  int filter_zero_points[kMaxChannels + 1] = {s.channels};
  float bias_scales[kMaxChannels + 1] = {static_cast<float>(s.channels)};
  int bias_zero_points[kMaxChannels + 1] = {s.channels};
  int32_t multipliers[kMaxChannels];
  int32_t shifts[kMaxChannels];
  for (int c = 0; c < s.channels; ++c) {
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

  int input_dims[] = {4, 1, s.in_h, s.in_w, s.channels};
  int filter_dims[] = {4, 1, kKernel, kKernel, s.channels};
  int bias_dims[] = {1, s.channels};
  int output_dims[] = {4, 1, out_h, out_w, s.channels};
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
  params.stride_width = StrideW(s);
  params.stride_height = s.stride;
  params.dilation_width_factor = s.dilation;
  params.dilation_height_factor = s.dilation;
  params.depth_multiplier = 1;
  params.activation = s.activation;

  int inputs[] = {3, 0, 1, 2};
  int outputs[] = {1, 3};
  const TFLMRegistration registration = Register_DEPTHWISE_CONV_2D();
  micro::KernelRunner runner(registration, tensors, 4, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  g_opt_3x3_size_queries = 0;
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  memset(output, 0x55, sizeof(output));
  memset(g_leaf_calls, 0, sizeof(g_leaf_calls));
  const TfLiteStatus invoke = runner.Invoke();
  if (invoke != kTfLiteOk) {
    return invoke;
  }

  DepthwiseParams op_params = {};
  op_params.padding_type = s.padding == kTfLitePaddingSame
                               ? PaddingType::kSame
                               : PaddingType::kValid;
  op_params.padding_values.width =
      PadSize(s.padding, s.in_w, out_w, StrideW(s), s.dilation);
  op_params.padding_values.height =
      PadSize(s.padding, s.in_h, out_h, s.stride, s.dilation);
  op_params.stride_width = StrideW(s);
  op_params.stride_height = s.stride;
  op_params.dilation_width_factor = s.dilation;
  op_params.dilation_height_factor = s.dilation;
  op_params.depth_multiplier = 1;
  op_params.input_offset = -input_zero_point;
  op_params.weights_offset = 0;
  op_params.output_offset = output_zero_point;
  ActivationRange(s.activation, output_scale, output_zero_point,
                  &op_params.quantized_activation_min,
                  &op_params.quantized_activation_max);
  const int32_t in_shape[] = {1, s.in_h, s.in_w, s.channels};
  const int32_t f_shape[] = {1, kKernel, kKernel, s.channels};
  const int32_t b_shape[] = {s.channels};
  const int32_t o_shape[] = {1, out_h, out_w, s.channels};
  tflite::micro::reference_integer_ops::DepthwiseConvPerChannel(
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

void ExpectMatchesReference(const Shape& s, Route route, Leaf entry) {
  EXPECT_EQ(kTfLiteOk, RunAgainstReference(s));
  ExpectRoute(route, entry);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ExpectMatchesReference;
using tflite::testing::Route;
using tflite::testing::Shape;

TEST(HeliaDepthwiseS8Opt3x3RouteTest, SamePaddingStrideOneInGate) {
  ExpectMatchesReference(Shape{8, 9, 16, 1, 1, kTfLitePaddingSame},
                         Route::kOpt3x3, Leaf::kOpt3x3);
}

TEST(HeliaDepthwiseS8Opt3x3RouteTest, ValidPaddingOddDepthInGate) {
  ExpectMatchesReference(Shape{7, 8, 33, 1, 1, kTfLitePaddingValid},
                         Route::kOpt3x3, Leaf::kOpt3x3);
}

TEST(HeliaDepthwiseS8Opt3x3RouteTest, StrideTwoInGate) {
  ExpectMatchesReference(Shape{9, 11, 24, 2, 1, kTfLitePaddingSame},
                         Route::kOpt3x3, Leaf::kOpt3x3);
}

// Even input width at stride 2: SAME pads only the trailing column.
TEST(HeliaDepthwiseS8Opt3x3RouteTest, StrideTwoEvenWidthInGate) {
  ExpectMatchesReference(Shape{9, 10, 16, 2, 1, kTfLitePaddingSame},
                         Route::kOpt3x3, Leaf::kOpt3x3);
}

TEST(HeliaDepthwiseS8Opt3x3RouteTest, Depth64StrideOneTakesC64S1) {
  ExpectMatchesReference(Shape{6, 7, 64, 1, 1, kTfLitePaddingSame},
                         Route::kC64S1, Leaf::kC64S1);
}

TEST(HeliaDepthwiseS8Opt3x3RouteTest, Depth64StrideTwoTakesOpt3x3) {
  ExpectMatchesReference(Shape{9, 11, 64, 2, 1, kTfLitePaddingSame},
                         Route::kOpt3x3, Leaf::kOpt3x3);
}

TEST(HeliaDepthwiseS8Opt3x3RouteTest, DepthEightIsWrapper) {
  ExpectMatchesReference(Shape{8, 9, 8, 1, 1, kTfLitePaddingSame},
                         Route::kWrapper, Leaf::kCount);
}

TEST(HeliaDepthwiseS8Opt3x3RouteTest, DilationTwoIsWrapper) {
  ExpectMatchesReference(Shape{9, 11, 16, 1, 2, kTfLitePaddingValid},
                         Route::kWrapper, Leaf::kCount);
}

// Two output rows, below the entry's three.
TEST(HeliaDepthwiseS8Opt3x3RouteTest, TwoOutputRowsIsWrapper) {
  ExpectMatchesReference(Shape{2, 11, 16, 1, 1, kTfLitePaddingSame},
                         Route::kWrapper, Leaf::kCount);
}

// 3 x 5 = 15 output pixels, below the entry's 16.
TEST(HeliaDepthwiseS8Opt3x3RouteTest, SmallOutputAreaIsWrapper) {
  ExpectMatchesReference(Shape{3, 5, 16, 1, 1, kTfLitePaddingSame},
                         Route::kWrapper, Leaf::kCount);
}

// Input W 2 is below the gate's 3, so Prepare must not request the entries'
// scratch: an 8x2 output keeps every other condition in the gate.
TEST(HeliaDepthwiseS8Opt3x3RouteTest, InputWidthTwoIsWrapper) {
  ExpectMatchesReference(Shape{8, 2, 16, 1, 1, kTfLitePaddingSame},
                         Route::kWrapper, Leaf::kCount);
}

// More than one 64-channel pass: 96 channels, and 130 with a channel tail.
TEST(HeliaDepthwiseS8Opt3x3RouteTest, Depth96InGate) {
  ExpectMatchesReference(Shape{6, 7, 96, 1, 1, kTfLitePaddingSame},
                         Route::kOpt3x3, Leaf::kOpt3x3);
}

TEST(HeliaDepthwiseS8Opt3x3RouteTest, Depth130InGate) {
  ExpectMatchesReference(Shape{7, 9, 130, 1, 1, kTfLitePaddingSame},
                         Route::kOpt3x3, Leaf::kOpt3x3);
}

// 64 channels at vertical stride 1 and horizontal stride 2 still take c64_s1.
TEST(HeliaDepthwiseS8Opt3x3RouteTest, Depth64HorizontalStrideTwoTakesC64S1) {
  ExpectMatchesReference(Shape{6, 9, 64, 1, 1, kTfLitePaddingSame, 2},
                         Route::kC64S1, Leaf::kC64S1);
}

TEST(HeliaDepthwiseS8Opt3x3RouteTest, FusedRelu6InGate) {
  ExpectMatchesReference(
      Shape{8, 9, 16, 1, 1, kTfLitePaddingSame, 0, kTfLiteActRelu6},
      Route::kOpt3x3, Leaf::kOpt3x3);
}

TEST(HeliaDepthwiseS8Opt3x3RouteTest, DeclinedEntryFallsBackToWrapper) {
  g_inject = true;
  g_injected_status = ARM_CMSIS_NN_NO_IMPL_ERROR;
  EXPECT_EQ(kTfLiteOk, tflite::testing::RunAgainstReference(
                           Shape{8, 9, 16, 1, 1, kTfLitePaddingSame}));
  g_inject = false;
  tflite::testing::ExpectRoute(Route::kDeclined, Leaf::kOpt3x3);
}

// Any other entry status fails the invoke without running the wrapper.
TEST(HeliaDepthwiseS8Opt3x3RouteTest, EntryErrorFailsInvoke) {
  g_inject = true;
  g_injected_status = ARM_CMSIS_NN_ARG_ERROR;
  const TfLiteStatus invoke = tflite::testing::RunAgainstReference(
      Shape{6, 7, 64, 1, 1, kTfLitePaddingSame});
  g_inject = false;
#if HELIA_DW_3X3_LINK_WRAP
  if (tflite::testing::kMve) {
    EXPECT_EQ(kTfLiteError, invoke);
    EXPECT_EQ(1, tflite::testing::LeafCalls(Leaf::kC64S1));
    EXPECT_EQ(0, tflite::testing::LeafCalls(Leaf::kWrapper));
    return;
  }
#endif
  EXPECT_EQ(kTfLiteOk, invoke);
}

TF_LITE_MICRO_TESTS_MAIN
