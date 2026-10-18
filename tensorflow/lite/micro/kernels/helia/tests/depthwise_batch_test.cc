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

// int8 DEPTHWISE_CONV_2D with more than one batch, int8 and int4 filters,
// through the helia kernel, compared bit for bit with the reference integer
// implementation. Debug builds must run these layers as release builds do.
// see AmbiqAI/helia-rt#470

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

struct Shape {
  int batch, in_h, in_w, in_c, depth_multiplier, kernel, stride, dilation;
  TfLitePadding padding;
  bool int4 = false;
};

constexpr int kMaxInput = 3 * 8 * 9 * 16;
constexpr int kMaxFilter = 5 * 5 * 32;
constexpr int kMaxChannels = 32;
constexpr int kMaxOutput = 3 * 8 * 9 * 32;

uint32_t g_seed = 9151;
int NextInt(int lo, int hi) {
  g_seed = g_seed * 1103515245u + 12345u;
  return lo + static_cast<int>((g_seed >> 16) % (hi - lo + 1));
}

int OutSize(const Shape& s, int in) {
  const int effective = (s.kernel - 1) * s.dilation + 1;
  return s.padding == kTfLitePaddingSame
             ? (in + s.stride - 1) / s.stride
             : (in - effective + s.stride) / s.stride;
}

int PadSize(const Shape& s, int in, int out) {
  if (s.padding != kTfLitePaddingSame) {
    return 0;
  }
  const int effective = (s.kernel - 1) * s.dilation + 1;
  const int total = (out - 1) * s.stride + effective - in;
  return total > 0 ? total / 2 : 0;
}

void ExpectMatchesReference(const Shape& s) {
  static int8_t input[kMaxInput];
  static int8_t filter[kMaxFilter];
  static int8_t unpacked_filter[kMaxFilter];
  static int32_t bias[kMaxChannels];
  static int8_t output[kMaxOutput];
  static int8_t expected[kMaxOutput];

  const int out_c = s.in_c * s.depth_multiplier;
  const int out_h = OutSize(s, s.in_h);
  const int out_w = OutSize(s, s.in_w);
  const int input_count = s.batch * s.in_h * s.in_w * s.in_c;
  const int filter_count = s.kernel * s.kernel * out_c;
  const int output_count = s.batch * out_h * out_w * out_c;
  EXPECT_LE(input_count, kMaxInput);
  EXPECT_LE(filter_count, kMaxFilter);
  EXPECT_LE(output_count, kMaxOutput);
  EXPECT_LE(out_c, kMaxChannels);
  EXPECT_TRUE(!s.int4 || filter_count % 2 == 0);
  if (input_count > kMaxInput || filter_count > kMaxFilter ||
      output_count > kMaxOutput || out_c > kMaxChannels) {
    return;
  }

  for (int i = 0; i < input_count; ++i) {
    input[i] = static_cast<int8_t>(NextInt(-128, 127));
  }
  if (s.int4) {
    for (int i = 0; i < filter_count / 2; ++i) {
      filter[i] = static_cast<int8_t>(NextInt(-128, 127));
    }
    tensor_utils::UnpackDenseInt4IntoInt8(filter, filter_count,
                                          unpacked_filter);
  } else {
    for (int i = 0; i < filter_count; ++i) {
      filter[i] = static_cast<int8_t>(NextInt(-127, 127));
    }
    memcpy(unpacked_filter, filter, filter_count);
  }
  for (int i = 0; i < out_c; ++i) {
    bias[i] = NextInt(-2000, 2000);
  }

  const float input_scale = 0.05f;
  const int input_zero_point = 3;
  const float output_scale = s.int4 ? 0.05f : 0.4f;
  const int output_zero_point = -5;
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
    QuantizeMultiplier(static_cast<double>(input_scale) *
                           static_cast<double>(filter_scales[c + 1]) /
                           static_cast<double>(output_scale),
                       &multiplier, &shift);
    multipliers[c] = multiplier;
    shifts[c] = shift;
  }

  int input_dims[] = {4, s.batch, s.in_h, s.in_w, s.in_c};
  int filter_dims[] = {4, 1, s.kernel, s.kernel, out_c};
  int bias_dims[] = {1, out_c};
  int output_dims[] = {4, s.batch, out_h, out_w, out_c};
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
  if (s.int4) {
    tensors[1].type = kTfLiteInt4;
    tensors[1].bytes = filter_count / 2;
  }

  TfLiteDepthwiseConvParams params = {};
  params.padding = s.padding;
  params.stride_width = s.stride;
  params.stride_height = s.stride;
  params.dilation_width_factor = s.dilation;
  params.dilation_height_factor = s.dilation;
  params.depth_multiplier = s.depth_multiplier;
  params.activation = kTfLiteActNone;

  int inputs[] = {3, 0, 1, 2};
  int outputs[] = {1, 3};
  const TFLMRegistration registration = Register_DEPTHWISE_CONV_2D();
  micro::KernelRunner runner(registration, tensors, 4, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  memset(output, 0x55, sizeof(output));
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  DepthwiseParams op_params = {};
  op_params.padding_type = s.padding == kTfLitePaddingSame
                               ? PaddingType::kSame
                               : PaddingType::kValid;
  op_params.padding_values.width = PadSize(s, s.in_w, out_w);
  op_params.padding_values.height = PadSize(s, s.in_h, out_h);
  op_params.stride_width = s.stride;
  op_params.stride_height = s.stride;
  op_params.dilation_width_factor = s.dilation;
  op_params.dilation_height_factor = s.dilation;
  op_params.depth_multiplier = s.depth_multiplier;
  op_params.input_offset = -input_zero_point;
  op_params.weights_offset = 0;
  op_params.output_offset = output_zero_point;
  op_params.quantized_activation_min = -128;
  op_params.quantized_activation_max = 127;
  const int32_t in_shape[] = {s.batch, s.in_h, s.in_w, s.in_c};
  const int32_t f_shape[] = {1, s.kernel, s.kernel, out_c};
  const int32_t b_shape[] = {out_c};
  const int32_t o_shape[] = {s.batch, out_h, out_w, out_c};
  tflite::micro::reference_integer_ops::DepthwiseConvPerChannel(
      op_params, multipliers, shifts, RuntimeShape(4, in_shape), input,
      RuntimeShape(4, f_shape), unpacked_filter, RuntimeShape(1, b_shape), bias,
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
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ExpectMatchesReference;
using tflite::testing::Shape;

// Inside the arm_depthwise_conv_s8_opt_3x3 gate but for the batch.
TEST(HeliaDepthwiseBatchTest, ThreeByThreeDepth16BatchTwo) {
  ExpectMatchesReference(Shape{2, 6, 7, 16, 1, 3, 1, 1, kTfLitePaddingSame});
}

TEST(HeliaDepthwiseBatchTest, StrideTwoBatchThree) {
  ExpectMatchesReference(Shape{3, 8, 9, 8, 1, 3, 2, 1, kTfLitePaddingValid});
}

TEST(HeliaDepthwiseBatchTest, DepthMultiplierTwoBatchTwo) {
  ExpectMatchesReference(Shape{2, 5, 6, 4, 2, 3, 1, 1, kTfLitePaddingSame});
}

TEST(HeliaDepthwiseBatchTest, DilatedBatchTwo) {
  ExpectMatchesReference(Shape{2, 8, 9, 6, 1, 3, 1, 2, kTfLitePaddingSame});
}

TEST(HeliaDepthwiseBatchTest, FiveByFiveBatchTwo) {
  ExpectMatchesReference(Shape{2, 7, 8, 6, 1, 5, 1, 1, kTfLitePaddingValid});
}

TEST(HeliaDepthwiseBatchTest, Int4FilterBatchTwo) {
  ExpectMatchesReference(
      Shape{2, 6, 7, 8, 1, 3, 1, 1, kTfLitePaddingSame, true});
}

TEST(HeliaDepthwiseBatchTest, BatchOneStillMatches) {
  ExpectMatchesReference(Shape{1, 6, 7, 16, 1, 3, 1, 1, kTfLitePaddingSame});
}

TF_LITE_MICRO_TESTS_MAIN
