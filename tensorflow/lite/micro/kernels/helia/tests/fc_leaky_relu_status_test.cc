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

#include <cstdint>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

// FULLY_CONNECTED kernel-sum and LEAKY_RELU status contracts: a CORE failure
// must surface as kTfLiteError and leave the output untouched. The pinned
// heliaCORE never fails these entry points on its own, so a GNU link wrap
// injects the failure; without the wrap the failure cases assert the valid
// path only. see AmbiqAI/helia-rt#238

#ifndef HELIA_FC_LEAKY_LINK_WRAP
#define HELIA_FC_LEAKY_LINK_WRAP 0
#endif

namespace {

constexpr int8_t kSentinel8 = -77;
constexpr int16_t kSentinel16 = -7777;

#if HELIA_FC_LEAKY_LINK_WRAP
struct LinkState {
  bool fail_vector_sum;
  bool fail_leaky_s8;
  bool fail_leaky_s16;
  int size_query_calls;
  int vector_sum_calls;
  int leaky_s8_calls;
  int leaky_s16_calls;
};

LinkState g_link_state;

void ResetLinkState() { g_link_state = LinkState{}; }

extern "C" {

int32_t __real_arm_fully_connected_s8_get_buffer_size(const cmsis_nn_dims*);
int32_t __wrap_arm_fully_connected_s8_get_buffer_size(
    const cmsis_nn_dims* filter_dims) {
  ++g_link_state.size_query_calls;
  const int32_t actual =
      __real_arm_fully_connected_s8_get_buffer_size(filter_dims);
  // Report the MVE kernel-sum size on every target so the arm_vector_sum_s8
  // route executes on non-MVE legs too; CORE's scalar kernels ignore it.
  return actual > 0 ? actual
                    : filter_dims->c * static_cast<int32_t>(sizeof(int32_t));
}

arm_cmsis_nn_status __real_arm_vector_sum_s8(int32_t*, int32_t, int32_t,
                                             const int8_t*, int32_t, int32_t,
                                             const int32_t*);
arm_cmsis_nn_status __wrap_arm_vector_sum_s8(
    int32_t* vector_sum_buf, int32_t vector_cols, int32_t vector_rows,
    const int8_t* vector_data, int32_t lhs_offset, int32_t rhs_offset,
    const int32_t* bias_data) {
  ++g_link_state.vector_sum_calls;
  if (g_link_state.fail_vector_sum) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  return __real_arm_vector_sum_s8(vector_sum_buf, vector_cols, vector_rows,
                                  vector_data, lhs_offset, rhs_offset,
                                  bias_data);
}

arm_cmsis_nn_status __real_arm_leaky_relu_s8(const int8_t*, int32_t, int32_t,
                                             int32_t, int32_t, int32_t,
                                             int32_t, int8_t*, int32_t);
arm_cmsis_nn_status __wrap_arm_leaky_relu_s8(
    const int8_t* input, int32_t input_offset, int32_t output_offset,
    int32_t output_multiplier_alpha, int32_t output_shift_alpha,
    int32_t output_multiplier_identity, int32_t output_shift_identity,
    int8_t* output, int32_t output_size) {
  ++g_link_state.leaky_s8_calls;
  if (g_link_state.fail_leaky_s8) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  return __real_arm_leaky_relu_s8(
      input, input_offset, output_offset, output_multiplier_alpha,
      output_shift_alpha, output_multiplier_identity, output_shift_identity,
      output, output_size);
}

arm_cmsis_nn_status __real_arm_leaky_relu_s16(const int16_t*, int32_t,
                                              int32_t, int32_t, int32_t,
                                              int32_t, int32_t, int16_t*,
                                              int32_t);
arm_cmsis_nn_status __wrap_arm_leaky_relu_s16(
    const int16_t* input, int32_t input_offset, int32_t output_offset,
    int32_t output_multiplier_alpha, int32_t output_shift_alpha,
    int32_t output_multiplier_identity, int32_t output_shift_identity,
    int16_t* output, int32_t output_size) {
  ++g_link_state.leaky_s16_calls;
  if (g_link_state.fail_leaky_s16) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  return __real_arm_leaky_relu_s16(
      input, input_offset, output_offset, output_multiplier_alpha,
      output_shift_alpha, output_multiplier_identity, output_shift_identity,
      output, output_size);
}

}  // extern "C"
#endif  // HELIA_FC_LEAKY_LINK_WRAP

struct FcResult {
  TfLiteStatus prepare;
  TfLiteStatus invoke;
  int8_t output[3];
};

// Two-dimensional shapes keep the adapter on the arm_fully_connected_wrapper_s8
// route (not the fast 1x1 convolution one), which is where both kernel-sum
// call sites live.
FcResult RunFullyConnected() {
  int input_dims_data[] = {2, 1, 4};
  int filter_dims_data[] = {2, 3, 4};
  int bias_dims_data[] = {1, 3};
  int output_dims_data[] = {2, 1, 3};
  const int8_t input[] = {1, -2, 3, 4};
  const int8_t filter[] = {
      1, 2, -1, 1, -2, 0, 1, 3, 1, -1, 1, -1,
  };
  const int32_t bias[] = {1, -3, 2};
  int8_t output[] = {kSentinel8, kSentinel8, kSentinel8};

  TfLiteTensor filter_tensor = tflite::testing::CreateQuantizedTensor(
      filter, tflite::testing::IntArrayFromInts(filter_dims_data), 1.0f);
  TfLiteTensor bias_tensor = tflite::testing::CreateQuantizedTensor(
      bias, tflite::testing::IntArrayFromInts(bias_dims_data), 1.0f);
  filter_tensor.allocation_type = kTfLiteMmapRo;
  bias_tensor.allocation_type = kTfLiteMmapRo;

  TfLiteTensor tensors[] = {
      tflite::testing::CreateQuantizedTensor(
          input, tflite::testing::IntArrayFromInts(input_dims_data), 1.0f),
      filter_tensor,
      bias_tensor,
      tflite::testing::CreateQuantizedTensor(
          output, tflite::testing::IntArrayFromInts(output_dims_data), 1.0f),
  };
  int inputs_data[] = {3, 0, 1, 2};
  int outputs_data[] = {1, 3};
  TfLiteFullyConnectedParams params = {};
  params.activation = kTfLiteActNone;
  params.weights_format = kTfLiteFullyConnectedWeightsFormatDefault;

  tflite::micro::KernelRunner runner(
      tflite::Register_FULLY_CONNECTED(), tensors, 4,
      tflite::testing::IntArrayFromInts(inputs_data),
      tflite::testing::IntArrayFromInts(outputs_data), &params);
  const TfLiteStatus prepare = runner.InitAndPrepare();
  const TfLiteStatus invoke =
      prepare == kTfLiteOk ? runner.Invoke() : kTfLiteError;
  return {prepare, invoke, {output[0], output[1], output[2]}};
}

void ExpectFullyConnectedGolden(const FcResult& result) {
  EXPECT_EQ(kTfLiteOk, result.prepare);
  EXPECT_EQ(kTfLiteOk, result.invoke);
  const int8_t expected[] = {-1, 10, 4};
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(expected[i], result.output[i]);
  }
}

template <typename T>
struct LeakyResult {
  TfLiteStatus prepare;
  TfLiteStatus invoke;
  T output[5];
};

template <typename T>
LeakyResult<T> RunLeakyRelu(const T (&input)[5], T sentinel) {
  int dims_data[] = {2, 1, 5};
  T output[] = {sentinel, sentinel, sentinel, sentinel, sentinel};
  TfLiteTensor tensors[] = {
      tflite::testing::CreateQuantizedTensor(
          input, tflite::testing::IntArrayFromInts(dims_data), 1.0f,
          /*zero_point=*/0),
      tflite::testing::CreateQuantizedTensor(
          output, tflite::testing::IntArrayFromInts(dims_data), 1.0f,
          /*zero_point=*/0),
  };
  int inputs_data[] = {1, 0};
  int outputs_data[] = {1, 1};
  TfLiteLeakyReluParams params = {};
  params.alpha = 0.5f;

  tflite::micro::KernelRunner runner(
      tflite::Register_LEAKY_RELU(), tensors, 2,
      tflite::testing::IntArrayFromInts(inputs_data),
      tflite::testing::IntArrayFromInts(outputs_data), &params);
  const TfLiteStatus prepare = runner.InitAndPrepare();
  const TfLiteStatus invoke =
      prepare == kTfLiteOk ? runner.Invoke() : kTfLiteError;
  LeakyResult<T> result = {prepare, invoke, {}};
  for (int i = 0; i < 5; ++i) {
    result.output[i] = output[i];
  }
  return result;
}

constexpr int8_t kLeakyInput8[] = {-100, -10, 0, 10, 100};
constexpr int8_t kLeakyExpected8[] = {-50, -5, 0, 10, 100};
constexpr int16_t kLeakyInput16[] = {-1000, -10, 0, 10, 1000};
constexpr int16_t kLeakyExpected16[] = {-500, -5, 0, 10, 1000};

template <typename T>
void ExpectLeakyGolden(const LeakyResult<T>& result, const T (&expected)[5]) {
  EXPECT_EQ(kTfLiteOk, result.prepare);
  EXPECT_EQ(kTfLiteOk, result.invoke);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(expected[i], result.output[i]);
  }
}

template <typename T>
void ExpectLeakyRejected(const LeakyResult<T>& result, T sentinel) {
  EXPECT_EQ(kTfLiteOk, result.prepare);
  EXPECT_EQ(kTfLiteError, result.invoke);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(sentinel, result.output[i]);
  }
}

}  // namespace

TEST(HeliaFcLeakyReluStatusTest, FullyConnectedInt8Golden) {
#if HELIA_FC_LEAKY_LINK_WRAP
  ResetLinkState();
#endif
  ExpectFullyConnectedGolden(RunFullyConnected());
#if HELIA_FC_LEAKY_LINK_WRAP
  EXPECT_TRUE(g_link_state.size_query_calls >= 1);
  EXPECT_EQ(1, g_link_state.vector_sum_calls);
#endif
}

TEST(HeliaFcLeakyReluStatusTest, FullyConnectedInt8VectorSumFailure) {
#if HELIA_FC_LEAKY_LINK_WRAP
  ResetLinkState();
  g_link_state.fail_vector_sum = true;
  const FcResult result = RunFullyConnected();
  EXPECT_EQ(1, g_link_state.vector_sum_calls);
#if defined(FC_KERNEL_OPTIMIZED_FOR_SPEED)
  // Kernel sums are precomputed into persistent memory at Prepare.
  EXPECT_EQ(kTfLiteError, result.prepare);
#else
  // Kernel sums are computed into arena scratch at Invoke.
  EXPECT_EQ(kTfLiteOk, result.prepare);
  EXPECT_EQ(kTfLiteError, result.invoke);
#endif
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(kSentinel8, result.output[i]);
  }
#else
  ExpectFullyConnectedGolden(RunFullyConnected());
#endif
}

TEST(HeliaFcLeakyReluStatusTest, LeakyReluInt8Golden) {
#if HELIA_FC_LEAKY_LINK_WRAP
  ResetLinkState();
#endif
  ExpectLeakyGolden(RunLeakyRelu(kLeakyInput8, kSentinel8), kLeakyExpected8);
#if HELIA_FC_LEAKY_LINK_WRAP
  EXPECT_EQ(1, g_link_state.leaky_s8_calls);
#endif
}

TEST(HeliaFcLeakyReluStatusTest, LeakyReluInt16Golden) {
#if HELIA_FC_LEAKY_LINK_WRAP
  ResetLinkState();
#endif
  ExpectLeakyGolden(RunLeakyRelu(kLeakyInput16, kSentinel16),
                    kLeakyExpected16);
#if HELIA_FC_LEAKY_LINK_WRAP
  EXPECT_EQ(1, g_link_state.leaky_s16_calls);
#endif
}

TEST(HeliaFcLeakyReluStatusTest, LeakyReluInt8Failure) {
#if HELIA_FC_LEAKY_LINK_WRAP
  ResetLinkState();
  g_link_state.fail_leaky_s8 = true;
  ExpectLeakyRejected(RunLeakyRelu(kLeakyInput8, kSentinel8), kSentinel8);
  EXPECT_EQ(1, g_link_state.leaky_s8_calls);
#else
  ExpectLeakyGolden(RunLeakyRelu(kLeakyInput8, kSentinel8), kLeakyExpected8);
#endif
}

TEST(HeliaFcLeakyReluStatusTest, LeakyReluInt16Failure) {
#if HELIA_FC_LEAKY_LINK_WRAP
  ResetLinkState();
  g_link_state.fail_leaky_s16 = true;
  ExpectLeakyRejected(RunLeakyRelu(kLeakyInput16, kSentinel16), kSentinel16);
  EXPECT_EQ(1, g_link_state.leaky_s16_calls);
#else
  ExpectLeakyGolden(RunLeakyRelu(kLeakyInput16, kSentinel16),
                    kLeakyExpected16);
#endif
}

TF_LITE_MICRO_TESTS_MAIN
