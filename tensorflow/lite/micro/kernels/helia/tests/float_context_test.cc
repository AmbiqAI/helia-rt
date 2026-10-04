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

// Float CONV_2D, DEPTHWISE_CONV_2D and BATCH_MATMUL must hand heliaCORE the
// scratch byte count Prepare requested, not a size re-queried at Invoke. GNU
// link wraps make each sizer report a larger size after its Prepare call and
// record the context each entry point receives; every leg checks the golden
// output. see AmbiqAI/helia-rt#376

#include <cstdint>

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

#if ARM_NN_ENABLE_F32 || ARM_NN_ENABLE_F16
#include "arm_nnfunctions_flt.h"
#endif

namespace {

enum class Route {
  kConvF32,
  kDepthwiseF32,
  kBatchMatmulF32,
  kConvF16,
  kDepthwiseF16,
  kBatchMatmulF16,
  kCount
};

struct RouteState {
  int sizer_calls = 0;
  int32_t prepare_size = 0;
  int32_t sizer_batch = -1;
  int entry_calls = 0;
  int32_t entry_size = -1;
  int32_t entry_batch = -1;
  bool entry_buffer = false;
};

RouteState g_state[static_cast<int>(Route::kCount)];

#if ARM_NN_ENABLE_F32 || ARM_NN_ENABLE_F16
void ResetState() {
  for (RouteState& state : g_state) {
    state = RouteState{};
  }
}
#endif

#if defined(HELIA_FLOAT_CONTEXT_LINK_WRAP)
// Smallest size a wrapped sizer reports at Prepare, so every route requests
// scratch, and the amount it adds on any later call.
constexpr int32_t kMinPrepareSize = 64;
constexpr int32_t kLaterQueryDrift = 32;

RouteState& State(Route route) { return g_state[static_cast<int>(route)]; }

// `dims` is the input (or BATCH_MATMUL lhs) shape the call carries.
int32_t RecordSizer(Route route, const cmsis_nn_dims* dims, int32_t actual) {
  RouteState& state = State(route);
  if (state.sizer_calls++ == 0) {
    state.sizer_batch = dims->n;
    state.prepare_size = actual > kMinPrepareSize ? actual : kMinPrepareSize;
    return state.prepare_size;
  }
  return state.prepare_size + kLaterQueryDrift;
}

void RecordEntry(Route route, const cmsis_nn_dims* dims,
                 const cmsis_nn_context* ctx) {
  RouteState& state = State(route);
  ++state.entry_calls;
  state.entry_batch = dims->n;
  state.entry_size = ctx == nullptr ? -1 : ctx->size;
  state.entry_buffer = ctx != nullptr && ctx->buf != nullptr;
}
#endif

}  // namespace

#if defined(HELIA_FLOAT_CONTEXT_LINK_WRAP)
extern "C" {

#define HELIA_WRAP_SIZER(route, name, params_type)                         \
  int32_t __real_##name(const params_type*, const cmsis_nn_dims*,          \
                        const cmsis_nn_dims*, const cmsis_nn_dims*);       \
  int32_t __wrap_##name(const params_type* params, const cmsis_nn_dims* a, \
                        const cmsis_nn_dims* b, const cmsis_nn_dims* c) {  \
    return RecordSizer(route, a, __real_##name(params, a, b, c));          \
  }

#define HELIA_WRAP_CONV(route, name, params_type, T)                         \
  arm_cmsis_nn_status __real_##name(                                         \
      const cmsis_nn_context*, const params_type*, const cmsis_nn_dims*,     \
      const T*, const cmsis_nn_dims*, const T*, const cmsis_nn_dims*,        \
      const T*, const cmsis_nn_dims*, T*);                                   \
  arm_cmsis_nn_status __wrap_##name(                                         \
      const cmsis_nn_context* ctx, const params_type* params,                \
      const cmsis_nn_dims* input_dims, const T* input,                       \
      const cmsis_nn_dims* filter_dims, const T* filter,                     \
      const cmsis_nn_dims* bias_dims, const T* bias,                         \
      const cmsis_nn_dims* output_dims, T* output) {                         \
    RecordEntry(route, input_dims, ctx);                                     \
    return __real_##name(ctx, params, input_dims, input, filter_dims,        \
                         filter, bias_dims, bias, output_dims, output);      \
  }

#define HELIA_WRAP_BMM(route, name, params_type, T)                          \
  arm_cmsis_nn_status __real_##name(                                         \
      const cmsis_nn_context*, const params_type*, const cmsis_nn_dims*,     \
      const T*, const cmsis_nn_dims*, const T*, const cmsis_nn_dims*, T*);   \
  arm_cmsis_nn_status __wrap_##name(                                         \
      const cmsis_nn_context* ctx, const params_type* params,                \
      const cmsis_nn_dims* lhs_dims, const T* lhs,                           \
      const cmsis_nn_dims* rhs_dims, const T* rhs,                           \
      const cmsis_nn_dims* output_dims, T* output) {                         \
    RecordEntry(route, lhs_dims, ctx);                                       \
    return __real_##name(ctx, params, lhs_dims, lhs, rhs_dims, rhs,          \
                         output_dims, output);                               \
  }

HELIA_WRAP_SIZER(Route::kConvF32, arm_convolve_wrapper_f32_get_buffer_size,
                 cmsis_nn_conv_params_f32)
HELIA_WRAP_SIZER(Route::kDepthwiseF32,
                 arm_depthwise_conv_wrapper_f32_get_buffer_size,
                 cmsis_nn_dw_conv_params_f32)
HELIA_WRAP_SIZER(Route::kBatchMatmulF32, arm_batch_matmul_f32_get_buffer_size,
                 cmsis_nn_bmm_params_f32)
HELIA_WRAP_CONV(Route::kConvF32, arm_convolve_wrapper_f32,
                cmsis_nn_conv_params_f32, float32_t)
HELIA_WRAP_CONV(Route::kDepthwiseF32, arm_depthwise_conv_wrapper_f32,
                cmsis_nn_dw_conv_params_f32, float32_t)
HELIA_WRAP_BMM(Route::kBatchMatmulF32, arm_batch_matmul_f32,
               cmsis_nn_bmm_params_f32, float32_t)

#if ARM_NN_ENABLE_F16
HELIA_WRAP_SIZER(Route::kConvF16, arm_convolve_wrapper_f16_get_buffer_size,
                 cmsis_nn_conv_params_f16)
HELIA_WRAP_SIZER(Route::kDepthwiseF16,
                 arm_depthwise_conv_wrapper_f16_get_buffer_size,
                 cmsis_nn_dw_conv_params_f16)
HELIA_WRAP_SIZER(Route::kBatchMatmulF16, arm_batch_matmul_f16_get_buffer_size,
                 cmsis_nn_bmm_params_f16)
HELIA_WRAP_CONV(Route::kConvF16, arm_convolve_wrapper_f16,
                cmsis_nn_conv_params_f16, float16_t)
HELIA_WRAP_CONV(Route::kDepthwiseF16, arm_depthwise_conv_wrapper_f16,
                cmsis_nn_dw_conv_params_f16, float16_t)
HELIA_WRAP_BMM(Route::kBatchMatmulF16, arm_batch_matmul_f16,
               cmsis_nn_bmm_params_f16, float16_t)
#endif

}  // extern "C"
#endif

#if ARM_NN_ENABLE_F32 || ARM_NN_ENABLE_F16
namespace {

// float32 and float16 (bit pattern) element access.
template <typename T>
TfLiteType TypeOf();
template <typename T>
T One();

#if ARM_NN_ENABLE_F32
float AsFloat(float value) { return value; }
template <>
TfLiteType TypeOf<float>() {
  return kTfLiteFloat32;
}
template <>
float One<float>() {
  return 1.0f;
}
#endif

#if ARM_NN_ENABLE_F16
float AsFloat(uint16_t bits) {
  // Exact for the small integers these goldens use.
  const int exponent = ((bits >> 10) & 0x1f) - 15;
  const float mantissa = 1.0f + static_cast<float>(bits & 0x3ff) / 1024.0f;
  float value = mantissa;
  for (int i = 0; i < exponent; ++i) value *= 2.0f;
  for (int i = 0; i > exponent; --i) value *= 0.5f;
  return (bits & 0x8000) ? -value : value;
}
template <>
TfLiteType TypeOf<uint16_t>() {
  return kTfLiteFloat16;
}
template <>
uint16_t One<uint16_t>() {
  return 0x3c00;
}
#endif

void ExpectContext(Route route) {
#if defined(HELIA_FLOAT_CONTEXT_LINK_WRAP)
  const RouteState& state = State(route);
  EXPECT_EQ(1, state.sizer_calls);
  EXPECT_EQ(1, state.entry_calls);
  EXPECT_EQ(state.prepare_size, state.entry_size);
  EXPECT_EQ(state.sizer_batch, state.entry_batch);
  EXPECT_TRUE(state.entry_buffer);
#else
  (void)route;
#endif
}

// 1x4x4x2 input, 2x3x3x2 filter, VALID: every output is 18.
template <typename T>
void RunConv(Route route) {
  ResetState();
  int input_dims[] = {4, 1, 4, 4, 2};
  int filter_dims[] = {4, 2, 3, 3, 2};
  int bias_dims[] = {1, 2};
  int output_dims[] = {4, 1, 2, 2, 2};
  T input[32];
  T filter[36];
  T bias[2] = {};
  T output[8] = {};
  for (T& v : input) v = One<T>();
  for (T& v : filter) v = One<T>();
  const TfLiteType type = TypeOf<T>();
  TfLiteTensor tensors[] = {
      tflite::testing::CreateTensor(
          input, tflite::testing::IntArrayFromInts(input_dims), false, type),
      tflite::testing::CreateTensor(
          filter, tflite::testing::IntArrayFromInts(filter_dims), false, type),
      tflite::testing::CreateTensor(
          bias, tflite::testing::IntArrayFromInts(bias_dims), false, type),
      tflite::testing::CreateTensor(
          output, tflite::testing::IntArrayFromInts(output_dims), false, type),
  };
  int inputs[] = {3, 0, 1, 2};
  int outputs[] = {1, 3};
  TfLiteConvParams params = {};
  params.padding = kTfLitePaddingValid;
  params.stride_width = 1;
  params.stride_height = 1;
  params.dilation_width_factor = 1;
  params.dilation_height_factor = 1;
  params.activation = kTfLiteActNone;
  const TFLMRegistration registration = tflite::Register_CONV_2D();
  tflite::micro::KernelRunner runner(
      registration, tensors, 4, tflite::testing::IntArrayFromInts(inputs),
      tflite::testing::IntArrayFromInts(outputs), &params);
  ASSERT_EQ(kTfLiteOk, runner.InitAndPrepare());
  ASSERT_EQ(kTfLiteOk, runner.Invoke());
  for (const T& v : output) EXPECT_EQ(18.0f, AsFloat(v));
  ExpectContext(route);
}

// Bx4x4x1 input, depth multiplier 8, 1x3x3x8 filter, VALID: every output is 9.
template <typename T>
void RunDepthwise(Route route, int batch) {
  ResetState();
  int input_dims[] = {4, batch, 4, 4, 1};
  int filter_dims[] = {4, 1, 3, 3, 8};
  int bias_dims[] = {1, 8};
  int output_dims[] = {4, batch, 2, 2, 8};
  T input[32];
  T filter[72];
  T bias[8] = {};
  T output[64] = {};
  for (T& v : input) v = One<T>();
  for (T& v : filter) v = One<T>();
  const TfLiteType type = TypeOf<T>();
  TfLiteTensor tensors[] = {
      tflite::testing::CreateTensor(
          input, tflite::testing::IntArrayFromInts(input_dims), false, type),
      tflite::testing::CreateTensor(
          filter, tflite::testing::IntArrayFromInts(filter_dims), false, type),
      tflite::testing::CreateTensor(
          bias, tflite::testing::IntArrayFromInts(bias_dims), false, type),
      tflite::testing::CreateTensor(
          output, tflite::testing::IntArrayFromInts(output_dims), false, type),
  };
  int inputs[] = {3, 0, 1, 2};
  int outputs[] = {1, 3};
  TfLiteDepthwiseConvParams params = {};
  params.padding = kTfLitePaddingValid;
  params.stride_width = 1;
  params.stride_height = 1;
  params.depth_multiplier = 8;
  params.dilation_width_factor = 1;
  params.dilation_height_factor = 1;
  params.activation = kTfLiteActNone;
  const TFLMRegistration registration = tflite::Register_DEPTHWISE_CONV_2D();
  tflite::micro::KernelRunner runner(
      registration, tensors, 4, tflite::testing::IntArrayFromInts(inputs),
      tflite::testing::IntArrayFromInts(outputs), &params);
  ASSERT_EQ(kTfLiteOk, runner.InitAndPrepare());
  ASSERT_EQ(kTfLiteOk, runner.Invoke());
  for (int i = 0; i < batch * 32; ++i) EXPECT_EQ(9.0f, AsFloat(output[i]));
  ExpectContext(route);
}

// [1,2,3] x [1,3,4]: every output is 3.
template <typename T>
void RunBatchMatmul(Route route) {
  ResetState();
  int lhs_dims[] = {3, 1, 2, 3};
  int rhs_dims[] = {3, 1, 3, 4};
  int output_dims[] = {3, 1, 2, 4};
  T lhs[6];
  T rhs[12];
  T output[8] = {};
  for (T& v : lhs) v = One<T>();
  for (T& v : rhs) v = One<T>();
  const TfLiteType type = TypeOf<T>();
  TfLiteTensor tensors[] = {
      tflite::testing::CreateTensor(
          lhs, tflite::testing::IntArrayFromInts(lhs_dims), false, type),
      tflite::testing::CreateTensor(
          rhs, tflite::testing::IntArrayFromInts(rhs_dims), false, type),
      tflite::testing::CreateTensor(
          output, tflite::testing::IntArrayFromInts(output_dims), false, type),
  };
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  TfLiteBatchMatMulParams params = {};
  params.adj_x = false;
  params.adj_y = false;
  const TFLMRegistration registration = tflite::Register_BATCH_MATMUL();
  tflite::micro::KernelRunner runner(
      registration, tensors, 3, tflite::testing::IntArrayFromInts(inputs),
      tflite::testing::IntArrayFromInts(outputs), &params);
  ASSERT_EQ(kTfLiteOk, runner.InitAndPrepare());
  ASSERT_EQ(kTfLiteOk, runner.Invoke());
  for (const T& v : output) EXPECT_EQ(3.0f, AsFloat(v));
  ExpectContext(route);
}

}  // namespace
#endif

#if ARM_NN_ENABLE_F32
TEST(HeliaFloatContextTest, ConvFloat32UsesPrepareSize) {
  RunConv<float>(Route::kConvF32);
}

TEST(HeliaFloatContextTest, DepthwiseFloat32UsesPrepareSize) {
  RunDepthwise<float>(Route::kDepthwiseF32, 1);
}

// The sizer must see the batch Eval passes to heliaCORE.
TEST(HeliaFloatContextTest, DepthwiseFloat32Batch2SizesWithBatch) {
  RunDepthwise<float>(Route::kDepthwiseF32, 2);
}

TEST(HeliaFloatContextTest, BatchMatmulFloat32UsesPrepareSize) {
  RunBatchMatmul<float>(Route::kBatchMatmulF32);
}
#endif

#if ARM_NN_ENABLE_F16
TEST(HeliaFloatContextTest, ConvFloat16UsesPrepareSize) {
  RunConv<uint16_t>(Route::kConvF16);
}

TEST(HeliaFloatContextTest, DepthwiseFloat16UsesPrepareSize) {
  RunDepthwise<uint16_t>(Route::kDepthwiseF16, 1);
}

TEST(HeliaFloatContextTest, DepthwiseFloat16Batch2SizesWithBatch) {
  RunDepthwise<uint16_t>(Route::kDepthwiseF16, 2);
}

TEST(HeliaFloatContextTest, BatchMatmulFloat16UsesPrepareSize) {
  RunBatchMatmul<uint16_t>(Route::kBatchMatmulF16);
}
#endif

TF_LITE_MICRO_TESTS_MAIN
