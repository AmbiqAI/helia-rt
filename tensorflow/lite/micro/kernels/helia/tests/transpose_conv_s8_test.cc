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

// int8 TRANSPOSE_CONV through the helia kernel, compared bit for bit with the
// reference integer implementation under each fused activation, and the
// shape and bias defects Prepare must reject.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/transpose_conv.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/transpose_conv.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

// A model defect Prepare must reject. see AmbiqAI/helia-rt#443
enum class Malform { kNone, kInt64Bias, kBatch, kChannels, kBiasCount };

struct Shape {
  int in_h, in_w, in_c, out_c, kernel, stride;
  TfLitePadding padding;
  TfLiteFusedActivation activation;
  Malform malform = Malform::kNone;
};

constexpr int kMaxInput = 6 * 7 * 16;
constexpr int kMaxFilter = 8 * 3 * 3 * 16;
constexpr int kMaxChannels = 8;
constexpr int kMaxOutput = 12 * 14 * 8;
constexpr int kInputZeroPoint = 4;
constexpr int kOutputZeroPoint = -6;

uint32_t g_seed = 442;
int NextInt(int lo, int hi) {
  g_seed = g_seed * 1103515245u + 12345u;
  return lo + static_cast<int>((g_seed >> 16) % (hi - lo + 1));
}

int OutSize(TfLitePadding padding, int in, int kernel, int stride) {
  return padding == kTfLitePaddingSame ? in * stride
                                       : (in - 1) * stride + kernel;
}

int PadSize(TfLitePadding padding, int in, int out, int kernel, int stride) {
  if (padding != kTfLitePaddingSame) {
    return 0;
  }
  const int total = (in - 1) * stride + kernel - out;
  return total > 0 ? total / 2 : 0;
}

// TFLite's quantized bounds for a fused activation on an output with this
// scale and zero point.
void ActivationRange(TfLiteFusedActivation activation, float scale,
                     int32_t zero_point, int32_t* act_min,
                     int32_t* act_max) {
  auto quantize = [&](float x) -> int32_t {
    return zero_point + static_cast<int32_t>(std::round(x / scale));
  };
  *act_min = -128;
  *act_max = 127;
  if (activation == kTfLiteActRelu) {
    *act_min = std::max<int32_t>(-128, quantize(0.0f));
  } else if (activation == kTfLiteActRelu6) {
    *act_min = std::max<int32_t>(-128, quantize(0.0f));
    *act_max = std::min<int32_t>(127, quantize(6.0f));
  } else if (activation == kTfLiteActReluN1To1) {
    *act_min = std::max<int32_t>(-128, quantize(-1.0f));
    *act_max = std::min<int32_t>(127, quantize(1.0f));
  }
}

void ExpectMatchesReference(const Shape& s) {
  static int8_t input[kMaxInput];
  static int8_t filter[kMaxFilter];
  static int32_t bias[kMaxChannels];
  static int64_t bias64[kMaxChannels];
  static int8_t output[kMaxOutput];
  static int8_t expected[kMaxOutput];
  static int32_t scratch[kMaxOutput];

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
    input[i] = static_cast<int8_t>(NextInt(-128, 127));
  }
  for (int i = 0; i < filter_count; ++i) {
    filter[i] = static_cast<int8_t>(NextInt(-127, 127));
  }
  for (int i = 0; i < s.out_c; ++i) {
    bias[i] = NextInt(-2000, 2000);
  }

  const float input_scale = 0.05f;
  const float output_scale = 0.05f;
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
  if (s.malform == Malform::kBatch) {
    output_dims[1] = 2;
  } else if (s.malform == Malform::kChannels) {
    filter_dims[1] = s.out_c - 1;
  } else if (s.malform == Malform::kBiasCount) {
    bias_dims[1] = s.out_c - 1;
  }
  TfLiteAffineQuantization filter_quant = {FloatArrayFromFloats(filter_scales),
                                           IntArrayFromInts(filter_zero_points),
                                           0};
  TfLiteAffineQuantization bias_quant = {FloatArrayFromFloats(bias_scales),
                                         IntArrayFromInts(bias_zero_points), 0};
  TfLiteTensor tensors[5] = {
      CreateTensor(output_shape, IntArrayFromInts(output_shape_dims)),
      CreateTensor(filter, IntArrayFromInts(filter_dims)),
      CreateQuantizedTensor(input, IntArrayFromInts(input_dims), input_scale,
                            kInputZeroPoint),
      s.malform == Malform::kInt64Bias
          ? CreateTensor(bias64, IntArrayFromInts(bias_dims))
          : CreateTensor(bias, IntArrayFromInts(bias_dims)),
      CreateQuantizedTensor(output, IntArrayFromInts(output_dims), output_scale,
                            kOutputZeroPoint),
  };
  tensors[1].quantization = {kTfLiteAffineQuantization, &filter_quant};
  tensors[3].quantization = {kTfLiteAffineQuantization, &bias_quant};

  TfLiteTransposeConvParams params = {};
  params.padding = s.padding;
  params.stride_width = s.stride;
  params.stride_height = s.stride;
  params.activation = s.activation;

  int inputs[] = {4, 0, 1, 2, 3};
  int outputs[] = {1, 4};
  const TFLMRegistration registration = Register_TRANSPOSE_CONV();
  micro::KernelRunner runner(registration, tensors, 5, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  const TfLiteStatus prepare =
      runner.InitAndPrepare(reinterpret_cast<const char*>(&params));
  if (s.malform != Malform::kNone) {
    EXPECT_EQ(kTfLiteError, prepare);
    return;
  }
  EXPECT_EQ(kTfLiteOk, prepare);
  memset(output, 0x55, sizeof(output));
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

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
  op_params.input_offset = -kInputZeroPoint;
  op_params.weights_offset = 0;
  op_params.output_offset = kOutputZeroPoint;
  ActivationRange(s.activation, output_scale, kOutputZeroPoint,
                  &op_params.quantized_activation_min,
                  &op_params.quantized_activation_max);
  const int32_t in_shape[] = {1, s.in_h, s.in_w, s.in_c};
  const int32_t f_shape[] = {s.out_c, s.kernel, s.kernel, s.in_c};
  const int32_t b_shape[] = {s.out_c};
  const int32_t o_shape[] = {1, out_h, out_w, s.out_c};
  tflite::micro::reference_integer_ops::TransposeConv(
      op_params, multipliers, shifts, RuntimeShape(4, in_shape), input,
      RuntimeShape(4, f_shape), filter, RuntimeShape(1, b_shape), bias,
      RuntimeShape(4, o_shape), expected, RuntimeShape(), nullptr, scratch);

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
using tflite::testing::Malform;
using tflite::testing::Shape;

TEST(HeliaTransposeConvS8Test, NoActivation) {
  ExpectMatchesReference(
      Shape{6, 7, 16, 8, 3, 2, kTfLitePaddingSame, kTfLiteActNone});
}

// The fused activation clamps the output. see AmbiqAI/helia-rt#442
TEST(HeliaTransposeConvS8Test, FusedRelu) {
  ExpectMatchesReference(
      Shape{6, 7, 16, 8, 3, 2, kTfLitePaddingSame, kTfLiteActRelu});
}

TEST(HeliaTransposeConvS8Test, FusedRelu6) {
  ExpectMatchesReference(
      Shape{5, 6, 4, 6, 3, 1, kTfLitePaddingValid, kTfLiteActRelu6});
}

TEST(HeliaTransposeConvS8Test, FusedReluN1To1) {
  ExpectMatchesReference(
      Shape{5, 6, 4, 6, 3, 1, kTfLitePaddingValid, kTfLiteActReluN1To1});
}

TEST(HeliaTransposeConvS8Test, Int64BiasFailsPrepare) {
  ExpectMatchesReference(Shape{5, 6, 4, 6, 3, 1, kTfLitePaddingValid,
                               kTfLiteActNone, Malform::kInt64Bias});
}

TEST(HeliaTransposeConvS8Test, BatchMismatchFailsPrepare) {
  ExpectMatchesReference(Shape{5, 6, 4, 6, 3, 1, kTfLitePaddingValid,
                               kTfLiteActNone, Malform::kBatch});
}

TEST(HeliaTransposeConvS8Test, FilterChannelMismatchFailsPrepare) {
  ExpectMatchesReference(Shape{5, 6, 4, 6, 3, 1, kTfLitePaddingValid,
                               kTfLiteActNone,
                               Malform::kChannels});
}

TEST(HeliaTransposeConvS8Test, BiasCountMismatchFailsPrepare) {
  ExpectMatchesReference(Shape{5, 6, 4, 6, 3, 1, kTfLitePaddingValid,
                               kTfLiteActNone,
                               Malform::kBiasCount});
}

TF_LITE_MICRO_TESTS_MAIN
