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
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/testdata/lstm_test_data.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

// UNIDIRECTIONAL_SEQUENCE_LSTM kernel-sum status and the scratch-context size
// contract of FULLY_CONNECTED, BATCH_MATMUL and AVERAGE_POOL_2D: a CORE
// failure must surface as kTfLiteError at Prepare, and every context backed
// by a scratch or persistent buffer must carry the byte count requested for
// it, since CORE treats size 0 as undeclared and skips its bounds check. GNU
// link wraps inject the failures, report a positive size on legs whose CORE
// needs no buffer, and record each context; without them the cases assert the
// valid outputs only. see AmbiqAI/helia-rt#238

#ifndef HELIA_STATUS_CONTEXT_LINK_WRAP
#define HELIA_STATUS_CONTEXT_LINK_WRAP 0
#endif

namespace {

#if HELIA_STATUS_CONTEXT_LINK_WRAP
constexpr int kLstmVectorSumCalls = 8;

struct ContextRecord {
  int calls;
  int32_t size;
  bool has_buffer;
};

struct LinkState {
  int vector_sum_fail_at;  // 1-based call number to fail; 0 fails none.
  int vector_sum_s8_calls;
  int vector_sum_s8_s64_calls;
  int32_t fc_s8_requested;
  int32_t fc_s16_requested;
  int32_t bmm_requested;
  int bmm_size_calls;
  int32_t pool_requested;
  ContextRecord fc_s8;
  ContextRecord fc_s16;
  ContextRecord bmm;
  ContextRecord pool;
};

LinkState g_link;

void ResetLinkState() { g_link = LinkState{}; }

void Record(ContextRecord* record, const cmsis_nn_context* ctx) {
  ++record->calls;
  record->size = ctx != nullptr ? ctx->size : -1;
  record->has_buffer = ctx != nullptr && ctx->buf != nullptr;
}

extern "C" {

arm_cmsis_nn_status __real_arm_vector_sum_s8(int32_t*, int32_t, int32_t,
                                             const int8_t*, int32_t, int32_t,
                                             const int32_t*);
arm_cmsis_nn_status __wrap_arm_vector_sum_s8(
    int32_t* vector_sum_buf, int32_t vector_cols, int32_t vector_rows,
    const int8_t* vector_data, int32_t lhs_offset, int32_t rhs_offset,
    const int32_t* bias_data) {
  if (++g_link.vector_sum_s8_calls == g_link.vector_sum_fail_at) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  return __real_arm_vector_sum_s8(vector_sum_buf, vector_cols, vector_rows,
                                  vector_data, lhs_offset, rhs_offset,
                                  bias_data);
}

arm_cmsis_nn_status __real_arm_vector_sum_s8_s64(int64_t*, int32_t, int32_t,
                                                 const int8_t*, int32_t,
                                                 const int64_t*);
arm_cmsis_nn_status __wrap_arm_vector_sum_s8_s64(
    int64_t* vector_sum_buf, int32_t vector_cols, int32_t vector_rows,
    const int8_t* vector_data, int32_t lhs_offset, const int64_t* bias_data) {
  if (++g_link.vector_sum_s8_s64_calls == g_link.vector_sum_fail_at) {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
  return __real_arm_vector_sum_s8_s64(vector_sum_buf, vector_cols, vector_rows,
                                      vector_data, lhs_offset, bias_data);
}

int32_t __real_arm_fully_connected_s8_get_buffer_size(const cmsis_nn_dims*);
int32_t __wrap_arm_fully_connected_s8_get_buffer_size(
    const cmsis_nn_dims* filter_dims) {
  const int32_t actual =
      __real_arm_fully_connected_s8_get_buffer_size(filter_dims);
  // Report the MVE kernel-sum size on every target so the buffer-backed
  // route executes on non-MVE legs too.
  g_link.fc_s8_requested =
      actual != 0 ? actual
                  : filter_dims->c * static_cast<int32_t>(sizeof(int32_t));
  return g_link.fc_s8_requested;
}

int32_t __real_arm_fully_connected_per_channel_s16_get_buffer_size(
    const cmsis_nn_dims*);
int32_t __wrap_arm_fully_connected_per_channel_s16_get_buffer_size(
    const cmsis_nn_dims* filter_dims) {
  g_link.fc_s16_requested =
      __real_arm_fully_connected_per_channel_s16_get_buffer_size(filter_dims);
  return g_link.fc_s16_requested;
}

arm_cmsis_nn_status __real_arm_fully_connected_wrapper_s8(
    const cmsis_nn_context*, const cmsis_nn_fc_params*,
    const cmsis_nn_quant_params*, const cmsis_nn_dims*, const int8_t*,
    const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*, const int32_t*,
    const cmsis_nn_dims*, int8_t*);
arm_cmsis_nn_status __wrap_arm_fully_connected_wrapper_s8(
    const cmsis_nn_context* ctx, const cmsis_nn_fc_params* fc_params,
    const cmsis_nn_quant_params* quant_params, const cmsis_nn_dims* input_dims,
    const int8_t* input, const cmsis_nn_dims* filter_dims, const int8_t* filter,
    const cmsis_nn_dims* bias_dims, const int32_t* bias,
    const cmsis_nn_dims* output_dims, int8_t* output) {
  Record(&g_link.fc_s8, ctx);
  return __real_arm_fully_connected_wrapper_s8(
      ctx, fc_params, quant_params, input_dims, input, filter_dims, filter,
      bias_dims, bias, output_dims, output);
}

arm_cmsis_nn_status __real_arm_fully_connected_wrapper_s16(
    const cmsis_nn_context*, const cmsis_nn_fc_params*,
    const cmsis_nn_quant_params*, const cmsis_nn_dims*, const int16_t*,
    const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*, const int64_t*,
    const cmsis_nn_dims*, int16_t*);
arm_cmsis_nn_status __wrap_arm_fully_connected_wrapper_s16(
    const cmsis_nn_context* ctx, const cmsis_nn_fc_params* fc_params,
    const cmsis_nn_quant_params* quant_params, const cmsis_nn_dims* input_dims,
    const int16_t* input, const cmsis_nn_dims* filter_dims,
    const int8_t* filter, const cmsis_nn_dims* bias_dims, const int64_t* bias,
    const cmsis_nn_dims* output_dims, int16_t* output) {
  Record(&g_link.fc_s16, ctx);
  return __real_arm_fully_connected_wrapper_s16(
      ctx, fc_params, quant_params, input_dims, input, filter_dims, filter,
      bias_dims, bias, output_dims, output);
}

int32_t __real_arm_batch_matmul_s8_get_buffer_size(const cmsis_nn_dims*);
int32_t __wrap_arm_batch_matmul_s8_get_buffer_size(
    const cmsis_nn_dims* input_rhs_dims) {
  ++g_link.bmm_size_calls;
  const int32_t actual =
      __real_arm_batch_matmul_s8_get_buffer_size(input_rhs_dims);
  g_link.bmm_requested =
      actual != 0 ? actual
                  : input_rhs_dims->w * static_cast<int32_t>(sizeof(int32_t));
  return g_link.bmm_requested;
}

arm_cmsis_nn_status __real_arm_batch_matmul_s8(
    const cmsis_nn_context*, const cmsis_nn_bmm_params*,
    const cmsis_nn_per_tensor_quant_params*, const cmsis_nn_dims*,
    const int8_t*, const cmsis_nn_dims*, const int8_t*, const cmsis_nn_dims*,
    int8_t*);
arm_cmsis_nn_status __wrap_arm_batch_matmul_s8(
    const cmsis_nn_context* ctx, const cmsis_nn_bmm_params* bmm_params,
    const cmsis_nn_per_tensor_quant_params* quant_params,
    const cmsis_nn_dims* input_lhs_dims, const int8_t* input_lhs,
    const cmsis_nn_dims* input_rhs_dims, const int8_t* input_rhs,
    const cmsis_nn_dims* output_dims, int8_t* output) {
  Record(&g_link.bmm, ctx);
  return __real_arm_batch_matmul_s8(ctx, bmm_params, quant_params,
                                    input_lhs_dims, input_lhs, input_rhs_dims,
                                    input_rhs, output_dims, output);
}

int32_t __real_arm_avgpool_s8_get_buffer_size(int, int);
int32_t __wrap_arm_avgpool_s8_get_buffer_size(int output_x, int ch_src) {
  const int32_t actual =
      __real_arm_avgpool_s8_get_buffer_size(output_x, ch_src);
  g_link.pool_requested =
      actual != 0 ? actual : ch_src * static_cast<int32_t>(sizeof(int32_t));
  return g_link.pool_requested;
}

int32_t __real_arm_avgpool_s16_get_buffer_size(int, int);
int32_t __wrap_arm_avgpool_s16_get_buffer_size(int output_x, int ch_src) {
  const int32_t actual =
      __real_arm_avgpool_s16_get_buffer_size(output_x, ch_src);
  g_link.pool_requested =
      actual != 0 ? actual : ch_src * static_cast<int32_t>(sizeof(int32_t));
  return g_link.pool_requested;
}

arm_cmsis_nn_status __real_arm_avgpool_s8(const cmsis_nn_context*,
                                          const cmsis_nn_pool_params*,
                                          const cmsis_nn_dims*, const int8_t*,
                                          const cmsis_nn_dims*,
                                          const cmsis_nn_dims*, int8_t*);
arm_cmsis_nn_status __wrap_arm_avgpool_s8(
    const cmsis_nn_context* ctx, const cmsis_nn_pool_params* pool_params,
    const cmsis_nn_dims* input_dims, const int8_t* input,
    const cmsis_nn_dims* filter_dims, const cmsis_nn_dims* output_dims,
    int8_t* output) {
  Record(&g_link.pool, ctx);
  return __real_arm_avgpool_s8(ctx, pool_params, input_dims, input, filter_dims,
                               output_dims, output);
}

arm_cmsis_nn_status __real_arm_avgpool_s16(const cmsis_nn_context*,
                                           const cmsis_nn_pool_params*,
                                           const cmsis_nn_dims*, const int16_t*,
                                           const cmsis_nn_dims*,
                                           const cmsis_nn_dims*, int16_t*);
arm_cmsis_nn_status __wrap_arm_avgpool_s16(
    const cmsis_nn_context* ctx, const cmsis_nn_pool_params* pool_params,
    const cmsis_nn_dims* input_dims, const int16_t* input,
    const cmsis_nn_dims* filter_dims, const cmsis_nn_dims* output_dims,
    int16_t* output) {
  Record(&g_link.pool, ctx);
  return __real_arm_avgpool_s16(ctx, pool_params, input_dims, input,
                                filter_dims, output_dims, output);
}

}  // extern "C"

void ExpectRecordedSize(const ContextRecord& record, int32_t requested) {
  EXPECT_EQ(1, record.calls);
  EXPECT_TRUE(requested > 0);
  EXPECT_TRUE(record.has_buffer);
  EXPECT_EQ(requested, record.size);
}
#endif  // HELIA_STATUS_CONTEXT_LINK_WRAP

struct Status {
  TfLiteStatus prepare;
  TfLiteStatus invoke;
};

Status Run(const TFLMRegistration& registration, TfLiteTensor* tensors,
           int tensors_size, int* inputs, int* outputs, void* params) {
  tflite::micro::KernelRunner runner(registration, tensors, tensors_size,
                                     tflite::testing::IntArrayFromInts(inputs),
                                     tflite::testing::IntArrayFromInts(outputs),
                                     params);
  const TfLiteStatus prepare = runner.InitAndPrepare();
  return {prepare, prepare == kTfLiteOk ? runner.Invoke() : kTfLiteError};
}

// UNIDIRECTIONAL_SEQUENCE_LSTM, using the upstream 2x3x2x2 model and goldens.

template <typename NodeContents>
Status RunLstm(NodeContents& contents) {
  // Weights and biases are constant, as a converter emits them.
  TfLiteTensor* tensors = contents.GetTensors();
  for (int i = 1; i < 9; ++i) tensors[i].allocation_type = kTfLiteMmapRo;
  for (int i = 12; i < 16; ++i) tensors[i].allocation_type = kTfLiteMmapRo;
  auto builtin_data = contents.BuiltinData();
  const TFLMRegistration registration =
      tflite::Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  tflite::micro::KernelRunner runner(registration, tensors, 24 + 1,
                                     contents.KernelInputs(),
                                     contents.KernelOutputs(), &builtin_data);
  const TfLiteStatus prepare = runner.InitAndPrepare();
  return {prepare, prepare == kTfLiteOk ? runner.Invoke() : kTfLiteError};
}

auto MakeLstmInt8() {
  const auto data = tflite::testing::Get2X2LstmEvalCheckData();
  return tflite::testing::Create2x3x2X2Int8NodeContents(data.input_data,
                                                        data.hidden_state);
}

auto MakeLstmInt16() {
  const auto data = tflite::testing::Get2X2LstmEvalCheckData();
  return tflite::testing::Create2x3x2X2Int16NodeContents(data.input_data,
                                                         data.hidden_state);
}

template <typename NodeContents>
void ExpectLstmGolden(NodeContents& contents, float tolerance) {
  const Status status = RunLstm(contents);
  EXPECT_EQ(kTfLiteOk, status.prepare);
  EXPECT_EQ(kTfLiteOk, status.invoke);
  const auto data = tflite::testing::Get2X2LstmEvalCheckData();
  const auto& output = contents.QuantizationSettings().output;
  float dequantized[12];
  tflite::Dequantize(contents.GetOutputData(), 12,
                     static_cast<float>(output.scale), output.zero_point,
                     dequantized);
  for (int i = 0; i < 12; ++i) {
    EXPECT_NEAR(data.expected_output[i], dequantized[i], tolerance);
  }
}

#if HELIA_STATUS_CONTEXT_LINK_WRAP
// Fails each Prepare-time kernel-sum call in turn; Prepare must stop at it.
template <typename MakeContents>
void ExpectEachLstmVectorSumFailureRejected(MakeContents make,
                                            const int& calls) {
  for (int fail_at = 1; fail_at <= kLstmVectorSumCalls; ++fail_at) {
    ResetLinkState();
    g_link.vector_sum_fail_at = fail_at;
    auto contents = make();
    EXPECT_EQ(kTfLiteError, RunLstm(contents).prepare);
    EXPECT_EQ(fail_at, calls);
  }
}
#endif

// FULLY_CONNECTED. The int8 case is two-dimensional so it takes the
// arm_fully_connected_wrapper_s8 route; its non-zero input zero point makes
// the MVE result depend on the kernel sums.

Status RunFullyConnectedInt8(int8_t (&output)[3]) {
  int input_dims_data[] = {2, 1, 4};
  int filter_dims_data[] = {2, 3, 4};
  int bias_dims_data[] = {1, 3};
  int output_dims_data[] = {2, 1, 3};
  const int8_t input[] = {1, -2, 3, 4};
  const int8_t filter[] = {1, 2, -1, 1, -2, 0, 1, 3, 1, -1, 1, -1};
  const int32_t bias[] = {1, -3, 2};

  TfLiteTensor tensors[] = {
      tflite::testing::CreateQuantizedTensor(
          input, tflite::testing::IntArrayFromInts(input_dims_data), 1.0f,
          /*zero_point=*/1),
      tflite::testing::CreateQuantizedTensor(
          filter, tflite::testing::IntArrayFromInts(filter_dims_data), 1.0f),
      tflite::testing::CreateQuantizedTensor(
          bias, tflite::testing::IntArrayFromInts(bias_dims_data), 1.0f),
      tflite::testing::CreateQuantizedTensor(
          output, tflite::testing::IntArrayFromInts(output_dims_data), 1.0f),
  };
  tensors[1].allocation_type = kTfLiteMmapRo;
  tensors[2].allocation_type = kTfLiteMmapRo;
  int inputs[] = {3, 0, 1, 2};
  int outputs[] = {1, 3};
  TfLiteFullyConnectedParams params = {};
  params.activation = kTfLiteActNone;
  params.weights_format = kTfLiteFullyConnectedWeightsFormatDefault;
  return Run(tflite::Register_FULLY_CONNECTED(), tensors, 4, inputs, outputs,
             &params);
}

Status RunFullyConnectedInt16PerChannel(int16_t (&output)[3]) {
  int input_dims_data[] = {2, 1, 4};
  int filter_dims_data[] = {2, 3, 4};
  int bias_dims_data[] = {1, 3};
  int output_dims_data[] = {2, 1, 3};
  const int16_t input[] = {1, -2, 3, 4};
  int8_t filter[] = {1, 2, -1, 1, -2, 0, 1, 3, 1, -1, 1, -1};
  int64_t bias[] = {1, -3, 2};
  float filter_scales_data[] = {3, 1.0f, 1.0f, 1.0f};
  int filter_zero_points_data[] = {3, 0, 0, 0};
  float bias_scales_data[] = {3, 1.0f, 1.0f, 1.0f};
  int bias_zero_points_data[] = {3, 0, 0, 0};
  TfLiteAffineQuantization filter_quantization = {};
  TfLiteAffineQuantization bias_quantization = {};
  TfLiteFloatArray* filter_scales =
      tflite::testing::FloatArrayFromFloats(filter_scales_data);

  TfLiteTensor tensors[] = {
      tflite::testing::CreateQuantizedTensor(
          input, tflite::testing::IntArrayFromInts(input_dims_data), 1.0f),
      tflite::testing::CreatePerChannelQuantizedTensor(
          filter, tflite::testing::IntArrayFromInts(filter_dims_data),
          filter_scales,
          tflite::testing::IntArrayFromInts(filter_zero_points_data),
          &filter_quantization, 0),
      tflite::testing::CreatePerChannelQuantizedBiasTensor(
          bias, tflite::testing::IntArrayFromInts(bias_dims_data), 1.0f,
          filter_scales,
          tflite::testing::FloatArrayFromFloats(bias_scales_data),
          tflite::testing::IntArrayFromInts(bias_zero_points_data),
          &bias_quantization, 0),
      tflite::testing::CreateQuantizedTensor(
          output, tflite::testing::IntArrayFromInts(output_dims_data), 1.0f),
  };
  tensors[1].allocation_type = kTfLiteMmapRo;
  tensors[2].allocation_type = kTfLiteMmapRo;
  int inputs[] = {3, 0, 1, 2};
  int outputs[] = {1, 3};
  TfLiteFullyConnectedParams params = {};
  params.activation = kTfLiteActNone;
  params.weights_format = kTfLiteFullyConnectedWeightsFormatDefault;
  return Run(tflite::Register_FULLY_CONNECTED(), tensors, 4, inputs, outputs,
             &params);
}

template <typename T>
void ExpectOutput(const Status& status, const T* expected, const T* output,
                  int size) {
  EXPECT_EQ(kTfLiteOk, status.prepare);
  EXPECT_EQ(kTfLiteOk, status.invoke);
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(expected[i], output[i]);
  }
}

// BATCH_MATMUL int8, M = 2, K = 3, N = 4, so the kernel-sum row count N
// differs from the inner dimension and from M. The rhs is given as [1, K, N],
// or as [1, N, K] with adj_y. The lhs zero point of 1 makes the MVE result
// depend on the kernel sums.
constexpr int8_t kBmmLhs[] = {2, -1, 3, 0, 4, -2};
constexpr int8_t kBmmRhsKN[] = {1, 0, -1, 2, 2, 1, 0, -1, -1, 3, 1, 0};
constexpr int8_t kBmmRhsNK[] = {1, 2, -1, 0, 1, 3, -1, 0, 1, 2, -1, 0};
// (lhs - 1) x rhs.
constexpr int8_t kBmmExpected[] = {-5, 4, 1, 4, 8, -6, -2, -5};

Status RunBatchMatmulInt8(bool adj_y, int8_t (&output)[8]) {
  int lhs_dims_data[] = {3, 1, 2, 3};
  int rhs_kn_dims_data[] = {3, 1, 3, 4};
  int rhs_nk_dims_data[] = {3, 1, 4, 3};
  int output_dims_data[] = {3, 1, 2, 4};
  TfLiteTensor tensors[] = {
      tflite::testing::CreateQuantizedTensor(
          kBmmLhs, tflite::testing::IntArrayFromInts(lhs_dims_data), 1.0f,
          /*zero_point=*/1),
      tflite::testing::CreateQuantizedTensor(
          adj_y ? kBmmRhsNK : kBmmRhsKN,
          tflite::testing::IntArrayFromInts(adj_y ? rhs_nk_dims_data
                                                  : rhs_kn_dims_data),
          1.0f),
      tflite::testing::CreateQuantizedTensor(
          output, tflite::testing::IntArrayFromInts(output_dims_data), 1.0f),
  };
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  TfLiteBatchMatMulParams params = {false, adj_y, false};
  return Run(tflite::Register_BATCH_MATMUL(), tensors, 3, inputs, outputs,
             &params);
}

void ExpectBatchMatmulInt8(bool adj_y) {
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  ResetLinkState();
#endif
  int8_t output[8] = {};
  ExpectOutput(RunBatchMatmulInt8(adj_y, output), kBmmExpected, output, 8);
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  EXPECT_EQ(1, g_link.bmm_size_calls);
  // One int32 kernel sum per rhs row: N = 4.
  EXPECT_EQ(4 * static_cast<int32_t>(sizeof(int32_t)), g_link.bmm_requested);
  ExpectRecordedSize(g_link.bmm, g_link.bmm_requested);
#endif
}

// AVERAGE_POOL_2D, one 2x2 window over three channels.
template <typename T>
Status RunAveragePool(const TFLMRegistration& registration, const T* input,
                      T (&output)[3]) {
  int input_dims_data[] = {4, 1, 2, 2, 3};
  int output_dims_data[] = {4, 1, 1, 1, 3};
  TfLiteTensor tensors[] = {
      tflite::testing::CreateQuantizedTensor(
          input, tflite::testing::IntArrayFromInts(input_dims_data), 1.0f),
      tflite::testing::CreateQuantizedTensor(
          output, tflite::testing::IntArrayFromInts(output_dims_data), 1.0f),
  };
  int inputs[] = {1, 0};
  int outputs[] = {1, 1};
  TfLitePoolParams params = {kTfLitePaddingValid, 2, 2, 2, 2,
                             kTfLiteActNone,      {}};
  return Run(registration, tensors, 2, inputs, outputs, &params);
}

template <typename T>
void ExpectAveragePool(const TFLMRegistration& registration) {
  // Per channel: {1, 3, 5, 7} -> 4, all -2 -> -2, {2, 4, 6, 8} -> 5.
  const T input[] = {1, -2, 2, 3, -2, 4, 5, -2, 6, 7, -2, 8};
  const T expected[] = {4, -2, 5};
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  ResetLinkState();
#endif
  T output[3] = {};
  ExpectOutput(RunAveragePool(registration, input, output), expected, output,
               3);
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  ExpectRecordedSize(g_link.pool, g_link.pool_requested);
#endif
}

}  // namespace

TEST(HeliaStatusContextTest, LstmInt8Golden) {
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  ResetLinkState();
#endif
  auto contents = MakeLstmInt8();
  ExpectLstmGolden(contents, 1e-2f);
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  EXPECT_EQ(kLstmVectorSumCalls, g_link.vector_sum_s8_calls);
#endif
}

TEST(HeliaStatusContextTest, LstmInt16Golden) {
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  ResetLinkState();
#endif
  auto contents = MakeLstmInt16();
  ExpectLstmGolden(contents, 1e-3f);
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  EXPECT_EQ(kLstmVectorSumCalls, g_link.vector_sum_s8_s64_calls);
#endif
}

TEST(HeliaStatusContextTest, LstmInt8VectorSumFailure) {
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  ExpectEachLstmVectorSumFailureRejected(MakeLstmInt8,
                                         g_link.vector_sum_s8_calls);
#else
  auto contents = MakeLstmInt8();
  ExpectLstmGolden(contents, 1e-2f);
#endif
}

TEST(HeliaStatusContextTest, LstmInt16VectorSumFailure) {
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  ExpectEachLstmVectorSumFailureRejected(MakeLstmInt16,
                                         g_link.vector_sum_s8_s64_calls);
#else
  auto contents = MakeLstmInt16();
  ExpectLstmGolden(contents, 1e-3f);
#endif
}

TEST(HeliaStatusContextTest, FullyConnectedInt8WeightContext) {
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  ResetLinkState();
#endif
  int8_t output[3] = {};
  // input - zero point = {0, -3, 2, 3}.
  const int8_t expected[] = {-4, 8, 4};
  ExpectOutput(RunFullyConnectedInt8(output), expected, output, 3);
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  ExpectRecordedSize(g_link.fc_s8, g_link.fc_s8_requested);
#endif
}

TEST(HeliaStatusContextTest, FullyConnectedInt16PerChannelContext) {
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  ResetLinkState();
#endif
  int16_t output[3] = {};
  const int16_t expected[] = {-1, 10, 4};
  ExpectOutput(RunFullyConnectedInt16PerChannel(output), expected, output, 3);
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  ExpectRecordedSize(g_link.fc_s16, g_link.fc_s16_requested);
#endif
}

TEST(HeliaStatusContextTest, BatchMatmulInt8Context) {
  ExpectBatchMatmulInt8(/*adj_y=*/false);
}

TEST(HeliaStatusContextTest, BatchMatmulInt8AdjYContext) {
  ExpectBatchMatmulInt8(/*adj_y=*/true);
}

TEST(HeliaStatusContextTest, BatchMatmulInt8SizersAgree) {
  // The previous sizer, arm_fully_connected_s8_get_buffer_size() over the
  // output dims, reads .c; the kernel indexes its sums by rhs .w. Both are N
  // for the shapes above, so the switch changes no request at the pin.
  const cmsis_nn_dims rhs_dims = {1, 1, 4, 3};
  const cmsis_nn_dims output_dims = {1, 1, 2, 4};
#if HELIA_STATUS_CONTEXT_LINK_WRAP
  EXPECT_EQ(__real_arm_fully_connected_s8_get_buffer_size(&output_dims),
            __real_arm_batch_matmul_s8_get_buffer_size(&rhs_dims));
#else
  EXPECT_EQ(arm_fully_connected_s8_get_buffer_size(&output_dims),
            arm_batch_matmul_s8_get_buffer_size(&rhs_dims));
#endif
}

TEST(HeliaStatusContextTest, AveragePoolInt8Context) {
  ExpectAveragePool<int8_t>(tflite::Register_AVERAGE_POOL_2D());
}

TEST(HeliaStatusContextTest, AveragePoolInt16Context) {
  ExpectAveragePool<int16_t>(tflite::Register_AVERAGE_POOL_2D_INT16());
}

TF_LITE_MICRO_TESTS_MAIN
