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

// Coverage for a 3-input (bias omitted) int16 TRANSPOSE_CONV.
//
// bias is optional for TRANSPOSE_CONV, and the kernel's Eval path already
// guards on `NumInputs(node) == 4`. Prepare did not: CalculateOpData() read
// `bias->type` unconditionally on the int16 branch, so a legal 3-input int16
// model dereferenced null inside AllocateTensors(). The upstream test helper
// hardcodes 4 inputs (see the "TODO(b/358151309): support optional bias
// tensor" in kernels/transpose_conv_test.cc), which is why the case was never
// exercised.
//
// It also holds the int8 scratch-context contract: on GCC and ATfE links the
// heliaCORE wrapper and both size queries are wrapped, and the contexts passed
// at Invoke must carry the byte counts Prepare requested.
// see AmbiqAI/helia-rt#238
//
// This lives under kernels/helia/tests/ rather than in the upstream test file
// because the fix is helia-side; adding it upstream would fail the reference
// builds that still carry the defect. Wired in via ext_libs/helia_tests.inc,
// which is only included when OPTIMIZED_KERNEL_DIR=helia.

#include <cstdint>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

struct TransposeConvLinkState {
  int wrapper_calls;
  int32_t buffer_size_query;
  int32_t reverse_size_query;
  int32_t ctx_size;
  int32_t reverse_ctx_size;
};

TransposeConvLinkState g_link_state = {};

}  // namespace

#if HELIA_TRANSPOSE_CONV_LINK_WRAP
extern "C" {

int32_t __real_arm_transpose_conv_s8_get_buffer_size(
    const cmsis_nn_transpose_conv_params*, const cmsis_nn_dims*,
    const cmsis_nn_dims*, const cmsis_nn_dims*);
int32_t __wrap_arm_transpose_conv_s8_get_buffer_size(
    const cmsis_nn_transpose_conv_params* params,
    const cmsis_nn_dims* input_dims, const cmsis_nn_dims* filter_dims,
    const cmsis_nn_dims* output_dims) {
  g_link_state.buffer_size_query = __real_arm_transpose_conv_s8_get_buffer_size(
      params, input_dims, filter_dims, output_dims);
  return g_link_state.buffer_size_query;
}

int32_t __real_arm_transpose_conv_s8_get_reverse_conv_buffer_size(
    const cmsis_nn_transpose_conv_params*, const cmsis_nn_dims*,
    const cmsis_nn_dims*);
int32_t __wrap_arm_transpose_conv_s8_get_reverse_conv_buffer_size(
    const cmsis_nn_transpose_conv_params* params,
    const cmsis_nn_dims* input_dims, const cmsis_nn_dims* filter_dims) {
  g_link_state.reverse_size_query =
      __real_arm_transpose_conv_s8_get_reverse_conv_buffer_size(
          params, input_dims, filter_dims);
  return g_link_state.reverse_size_query;
}

arm_cmsis_nn_status __real_arm_transpose_conv_wrapper_s8(
    const cmsis_nn_context*, const cmsis_nn_context*, const cmsis_nn_context*,
    const cmsis_nn_transpose_conv_params*,
    const cmsis_nn_per_channel_quant_params*, const cmsis_nn_dims*,
    const int8_t*, const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*,
    const int32_t*, const cmsis_nn_dims*, int8_t*);
arm_cmsis_nn_status __wrap_arm_transpose_conv_wrapper_s8(
    const cmsis_nn_context* ctx, const cmsis_nn_context* weight_sum_ctx,
    const cmsis_nn_context* reverse_conv_ctx,
    const cmsis_nn_transpose_conv_params* params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int8_t* input,
    const cmsis_nn_dims* filter_dims, const int8_t* filter,
    const cmsis_nn_dims* bias_dims, const int32_t* bias,
    const cmsis_nn_dims* output_dims, int8_t* output) {
  ++g_link_state.wrapper_calls;
  g_link_state.ctx_size = ctx->size;
  g_link_state.reverse_ctx_size = reverse_conv_ctx->size;
  return __real_arm_transpose_conv_wrapper_s8(
      ctx, weight_sum_ctx, reverse_conv_ctx, params, quant_params, input_dims,
      input, filter_dims, filter, bias_dims, bias, output_dims, output);
}

}  // extern "C"
#endif  // HELIA_TRANSPOSE_CONV_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

// Transpose conv uses TfLiteConvParams.
const TfLiteConvParams kConvParams = {
    kTfLitePaddingSame,  // padding
    1,                   // stride_width
    1,                   // stride_height
    kTfLiteActNone,
    1,  // dilation_width_factor
    1,  // dilation_height_factor
    kTfLiteNoType};

constexpr int kElements = 4;

int kInputShape[] = {4, 1, 2, 2, 1};
int kFilterShape[] = {4, 1, 2, 2, 1};
int kOutputShape[] = {4, 1, 2, 2, 1};

const float kInputData[kElements] = {1.0f, 2.0f, 3.0f, 4.0f};
const float kFilterData[kElements] = {1.0f, 0.0f, 0.0f, 1.0f};

}  // namespace
}  // namespace testing
}  // namespace tflite

// The assertion is that Prepare and Invoke complete at all: before the fix
// this dereferenced a null bias tensor inside Prepare.
TEST(HeliaTransposeConvTest, Int16WithoutBiasPreparesAndInvokes) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateSymmetricPerChannelQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;

  TfLiteIntArray* input_dims =
      IntArrayFromInts(tflite::testing::kInputShape);
  TfLiteIntArray* filter_dims =
      IntArrayFromInts(tflite::testing::kFilterShape);
  TfLiteIntArray* output_dims =
      IntArrayFromInts(tflite::testing::kOutputShape);

  int16_t input_quantized[tflite::testing::kElements];
  int8_t filter_quantized[tflite::testing::kElements];
  int16_t output_data[tflite::testing::kElements];

  int filter_zero_points[2];
  float filter_scales[2];
  TfLiteAffineQuantization filter_quant;
  TfLiteTensor filter_tensor = CreateSymmetricPerChannelQuantizedTensor(
      tflite::testing::kFilterData, filter_quantized, filter_dims,
      filter_scales, filter_zero_points, &filter_quant,
      /*quantized_dimension=*/0);

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape = nullptr;
  TfLiteIntArray* output_shape_dims =
      IntArrayFromInts(output_shape_dims_data);

  // Three inputs only: output_shape, filter, input. No bias tensor.
  constexpr int tensors_size = 4;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(output_shape, output_shape_dims),
      filter_tensor,
      CreateQuantizedTensor(tflite::testing::kInputData, input_quantized,
                            input_dims, /*scale=*/1.0f / 32.0f,
                            /*zero_point=*/0),
      CreateQuantizedTensor(output_data, output_dims, /*scale=*/1.0f / 32.0f,
                            /*zero_point=*/0),
  };

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TFLMRegistration registration = tflite::Register_TRANSPOSE_CONV();
  tflite::micro::KernelRunner runner(registration, tensors, tensors_size,
                                     inputs_array, outputs_array,
                                     &tflite::testing::kConvParams);

  const char* init_data =
      reinterpret_cast<const char*>(&tflite::testing::kConvParams);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare(init_data));
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

// Input depth above the reverse-convolution threshold (16) and stride 1, so
// both int8 scratch buffers have a nonzero size on every target.
TEST(HeliaTransposeConvTest, Int8ScratchContextsCarryRequestedSizes) {
  using tflite::testing::CreatePerChannelQuantizedBiasTensor;
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateSymmetricPerChannelQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;

  constexpr int kInCh = 20;
  constexpr int kOutCh = 8;
  constexpr int kInCount = 3 * 3 * kInCh;
  constexpr int kFilterCount = kOutCh * 3 * 3 * kInCh;
  constexpr int kOutCount = 3 * 3 * kOutCh;
  int input_shape[] = {4, 1, 3, 3, kInCh};
  int filter_shape[] = {4, kOutCh, 3, 3, kInCh};
  int bias_shape[] = {1, kOutCh};
  int output_shape[] = {4, 1, 3, 3, kOutCh};

  float input_data[kInCount];
  float filter_data[kFilterCount];
  float bias_data[kOutCh];
  for (int i = 0; i < kInCount; ++i) input_data[i] = (i % 7) - 3;
  for (int i = 0; i < kFilterCount; ++i) filter_data[i] = (i % 5) - 2;
  for (int i = 0; i < kOutCh; ++i) bias_data[i] = i;

  constexpr float kInputScale = 0.5f;
  int8_t input_quantized[kInCount];
  int8_t filter_quantized[kFilterCount];
  int32_t bias_quantized[kOutCh];
  int8_t output_data[kOutCount];
  float filter_scales[kOutCh + 1];
  int filter_zero_points[kOutCh + 1];
  float bias_scales[kOutCh + 1];
  int bias_zero_points[kOutCh + 1];
  TfLiteAffineQuantization filter_quant;
  TfLiteAffineQuantization bias_quant;

  int output_shape_dims_data[] = {1, 0};
  int32_t* output_shape_data = nullptr;
  constexpr int kTensorsSize = 5;
  TfLiteTensor tensors[kTensorsSize] = {
      CreateTensor(output_shape_data, IntArrayFromInts(output_shape_dims_data)),
      CreateSymmetricPerChannelQuantizedTensor(
          filter_data, filter_quantized, IntArrayFromInts(filter_shape),
          filter_scales, filter_zero_points, &filter_quant,
          /*quantized_dimension=*/0),
      CreateQuantizedTensor(input_data, input_quantized,
                            IntArrayFromInts(input_shape), kInputScale,
                            /*zero_point=*/0),
      CreatePerChannelQuantizedBiasTensor(
          bias_data, bias_quantized, IntArrayFromInts(bias_shape), kInputScale,
          filter_scales, bias_scales, bias_zero_points, &bias_quant,
          /*quantized_dimension=*/0),
      CreateQuantizedTensor(output_data, IntArrayFromInts(output_shape),
                            /*scale=*/4.0f, /*zero_point=*/0),
  };

  int inputs_array_data[] = {4, 0, 1, 2, 3};
  int outputs_array_data[] = {1, 4};
  const TFLMRegistration registration = tflite::Register_TRANSPOSE_CONV();
  tflite::micro::KernelRunner runner(
      registration, tensors, kTensorsSize, IntArrayFromInts(inputs_array_data),
      IntArrayFromInts(outputs_array_data), &tflite::testing::kConvParams);

  g_link_state = {};
  const char* init_data =
      reinterpret_cast<const char*>(&tflite::testing::kConvParams);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare(init_data));
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
#if HELIA_TRANSPOSE_CONV_LINK_WRAP
  EXPECT_EQ(1, g_link_state.wrapper_calls);
  EXPECT_GT(g_link_state.buffer_size_query, 0);
  EXPECT_EQ(kInCh * 3 * 3 * kOutCh, g_link_state.reverse_size_query);
  EXPECT_EQ(g_link_state.buffer_size_query, g_link_state.ctx_size);
  EXPECT_EQ(g_link_state.reverse_size_query, g_link_state.reverse_ctx_size);
#endif
}

TF_LITE_MICRO_TESTS_MAIN
