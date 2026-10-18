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

// Each kernel whose Init or Prepare writes into a persistent buffer must fail
// when the arena cannot hold it, and must not write through the nullptr.
// The convolutions use a per-tensor filter scale, so they write one entry per
// output channel: with 4096 channels neither 16 KiB per-channel array fits
// KernelRunner's 10000-byte arena, with 1500 the first fits and the second
// does not, and with 2^30 the byte count wraps a 32-bit size_t. The
// single-arena allocator must not hand out memory that overlaps a live temp
// allocation or lies outside the arena. see AmbiqAI/helia-rt#449

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/helia/helia_allocation_sweep.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/transpose_conv.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kChannels = 4096;
// The first per-channel array fits KernelRunner's arena, the second does not.
constexpr int kChannelsSecondArrayFails = 1500;
// num_channels * sizeof(int32_t) is 0 in a 32-bit size_t.
constexpr int kChannelsWrapSizeT = 1 << 30;

int8_t g_filter[kChannels];
int8_t g_input_s8[kChannels];
int8_t g_output_s8[kChannels];
int16_t g_input_s16[kChannels];
int16_t g_output_s16[kChannels];

// A per-tensor scale: PopulateConvolutionQuantizationParams then writes
// num_channels entries without comparing them with the scale count.
struct PerTensorQuantization {
  explicit PerTensorQuantization(int quantized_dimension) {
    params.scale = FloatArrayFromFloats(scales);
    params.zero_point = IntArrayFromInts(zero_points);
    params.quantized_dimension = quantized_dimension;
  }
  // FloatArrayFromFloats overwrites the leading count in place.
  float scales[2] = {1, 0.01f};
  int zero_points[2] = {1, 0};
  TfLiteAffineQuantization params;
};

TfLiteTensor PerTensorFilter(TfLiteIntArray* dims,
                             TfLiteAffineQuantization* quant) {
  TfLiteTensor filter = CreateQuantizedTensor(g_filter, dims, 0.01f, 0);
  filter.quantization = {kTfLiteAffineQuantization, quant};
  return filter;
}

TfLiteStatus PrepareConv(const TFLMRegistration& registration,
                         TfLiteTensor* tensors, int tensors_size,
                         const void* params) {
  int inputs_array_data[] = {3, 0, 1, kTfLiteOptionalTensor};
  int outputs_array_data[] = {1, tensors_size - 1};
  KernelRunner runner(registration, tensors, tensors_size,
                      IntArrayFromInts(inputs_array_data),
                      IntArrayFromInts(outputs_array_data), params);
  return runner.InitAndPrepare();
}

TfLiteStatus PrepareConv2DInt8(int channels) {
  int input_shape[] = {4, 1, 1, 1, 1};
  int filter_shape[] = {4, channels, 1, 1, 1};
  int output_shape[] = {4, 1, 1, 1, channels};
  PerTensorQuantization quant(0);
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(g_input_s8, IntArrayFromInts(input_shape), 0.5f, 0),
      PerTensorFilter(IntArrayFromInts(filter_shape), &quant.params),
      CreateQuantizedTensor(g_output_s8, IntArrayFromInts(output_shape), 0.5f,
                            0),
  };
  TfLiteConvParams params = {kTfLitePaddingValid, 1, 1, kTfLiteActNone, 1, 1,
                             kTfLiteNoType};
  const TFLMRegistration registration = Register_CONV_2D();
  return PrepareConv(registration, tensors, 3, &params);
}

TfLiteStatus PrepareDepthwiseConv2DInt8(int channels) {
  int input_shape[] = {4, 1, 1, 1, channels};
  int filter_shape[] = {4, 1, 1, 1, channels};
  int output_shape[] = {4, 1, 1, 1, channels};
  PerTensorQuantization quant(3);
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(g_input_s8, IntArrayFromInts(input_shape), 0.5f, 0),
      PerTensorFilter(IntArrayFromInts(filter_shape), &quant.params),
      CreateQuantizedTensor(g_output_s8, IntArrayFromInts(output_shape), 0.5f,
                            0),
  };
  TfLiteDepthwiseConvParams params = {};
  params.padding = kTfLitePaddingValid;
  params.stride_width = 1;
  params.stride_height = 1;
  params.depth_multiplier = 1;
  params.activation = kTfLiteActNone;
  params.dilation_width_factor = 1;
  params.dilation_height_factor = 1;
  const TFLMRegistration registration = Register_DEPTHWISE_CONV_2D();
  return PrepareConv(registration, tensors, 3, &params);
}

template <typename T>
TfLiteStatus PrepareTransposeConv(T* input_data, T* output_data, int channels) {
  int output_shape_shape[] = {1, 4};
  int32_t output_shape_data[] = {1, 1, 1, channels};
  int input_shape[] = {4, 1, 1, 1, 1};
  int filter_shape[] = {4, channels, 1, 1, 1};
  int output_shape[] = {4, 1, 1, 1, channels};
  // Built on one element and given its shape after, so the byte count of a
  // 2^30-element int16 tensor is never computed in int.
  int one_element_shape[] = {1, 1};
  PerTensorQuantization quant(0);
  TfLiteTensor tensors[] = {
      CreateTensor(output_shape_data, IntArrayFromInts(output_shape_shape)),
      PerTensorFilter(IntArrayFromInts(filter_shape), &quant.params),
      CreateQuantizedTensor(input_data, IntArrayFromInts(input_shape), 0.5f, 0),
      CreateQuantizedTensor(output_data, IntArrayFromInts(one_element_shape),
                            0.5f, 0),
  };
  tensors[3].dims = IntArrayFromInts(output_shape);
  int inputs_array_data[] = {4, 0, 1, 2, kTfLiteOptionalTensor};
  int outputs_array_data[] = {1, 3};
  TfLiteTransposeConvParams params = {kTfLitePaddingValid, 1, 1, kTfLiteActNone,
                                      kTfLiteNoType};
  const TFLMRegistration registration = Register_TRANSPOSE_CONV();
  KernelRunner runner(registration, tensors, 4,
                      IntArrayFromInts(inputs_array_data),
                      IntArrayFromInts(outputs_array_data), &params);
  return runner.InitAndPrepare();
}

// Takes every persistent byte KernelRunner's arena has left.
void ExhaustPersistentArena(TfLiteContext* context) {
  for (size_t chunk = 4096; chunk > 0; chunk /= 2) {
    while (context->AllocatePersistentBuffer(context, chunk) != nullptr) {
    }
  }
}

TFLMRegistration g_real;
void* g_init_result = nullptr;

void* InitThenExhaust(TfLiteContext* context, const char* buffer,
                      size_t length) {
  g_init_result = g_real.init(context, buffer, length);
  ExhaustPersistentArena(context);
  return g_init_result;
}

void* ExhaustThenInit(TfLiteContext* context, const char* buffer,
                      size_t length) {
  ExhaustPersistentArena(context);
  g_init_result = g_real.init(context, buffer, length);
  return g_init_result;
}

TFLMRegistration WithInit(const TFLMRegistration& real,
                          void* (*init)(TfLiteContext*, const char*, size_t)) {
  g_real = real;
  g_init_result = nullptr;
  TFLMRegistration registration = real;
  registration.init = init;
  return registration;
}

// A reduction over the last axis of a 1x4 input, with the arena full before
// Init.
template <typename T>
TfLiteStatus PrepareReductionWithFullArena(const TFLMRegistration& real,
                                           TfLiteType type) {
  int input_shape[] = {2, 1, 4};
  int axis_shape[] = {1, 1};
  int output_shape[] = {2, 1, 1};
  T input[4] = {};
  int32_t axis[1] = {1};
  T output[1] = {};
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input, IntArrayFromInts(input_shape), 0.5f, 0),
      CreateTensor(axis, IntArrayFromInts(axis_shape)),
      CreateQuantizedTensor(output, IntArrayFromInts(output_shape), 0.5f, 0),
  };
  tensors[0].type = type;
  tensors[2].type = type;
  int inputs_array_data[] = {2, 0, 1};
  int outputs_array_data[] = {1, 2};
  TfLiteReducerParams params = {true};
  const TFLMRegistration registration = WithInit(real, ExhaustThenInit);
  KernelRunner runner(registration, tensors, 3,
                      IntArrayFromInts(inputs_array_data),
                      IntArrayFromInts(outputs_array_data), &params);
  return runner.InitAndPrepare();
}

// An int8 SVDF with an int16 state, rank 1, two filters and two inputs.
struct SvdfInt8 {
  int input_shape[3] = {2, 1, 2};
  int weights_feature_shape[3] = {2, 2, 2};
  int weights_time_shape[3] = {2, 2, 2};
  int bias_shape[2] = {1, 2};
  int activation_state_shape[3] = {2, 1, 4};
  int output_shape[3] = {2, 1, 2};
  int8_t input[2] = {10, -20};
  int8_t weights_feature[4] = {1, 2, 3, 4};
  int16_t weights_time[4] = {100, 200, -100, 50};
  int32_t bias[2] = {0, 0};
  int16_t activation_state[4] = {};
  int8_t output[2] = {};
  TfLiteTensor tensors[6];
  TfLiteSVDFParams params = {1, kTfLiteActNone, false};

  SvdfInt8() {
    tensors[0] =
        CreateQuantizedTensor(input, IntArrayFromInts(input_shape), 0.5f, 0);
    tensors[1] = CreateQuantizedTensor(
        weights_feature, IntArrayFromInts(weights_feature_shape), 0.5f, 0);
    tensors[2] = CreateQuantizedTensor(
        weights_time, IntArrayFromInts(weights_time_shape), 0.5f, 0);
    // The bias scale is the state scale times the weights-time scale.
    tensors[3] =
        CreateQuantizedTensor(bias, IntArrayFromInts(bias_shape), 0.25f, 0);
    tensors[4] = CreateQuantizedTensor(activation_state,
                                       IntArrayFromInts(activation_state_shape),
                                       0.5f, 0, /*is_variable=*/true);
    tensors[5] =
        CreateQuantizedTensor(output, IntArrayFromInts(output_shape), 0.5f, 0);
  }
};

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::kChannels;
using tflite::testing::kChannelsSecondArrayFails;
using tflite::testing::kChannelsWrapSizeT;

TEST(HeliaPersistentBufferFailureTest, Conv2DInt8TooManyChannelsForArena) {
  EXPECT_EQ(tflite::testing::PrepareConv2DInt8(kChannels), kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest, Conv2DInt8SecondArrayDoesNotFit) {
  EXPECT_EQ(tflite::testing::PrepareConv2DInt8(kChannelsSecondArrayFails),
            kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest, Conv2DInt8ChannelBytesWrapSizeT) {
  EXPECT_EQ(tflite::testing::PrepareConv2DInt8(kChannelsWrapSizeT),
            kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest, Conv2DInt8ZeroChannels) {
  EXPECT_EQ(tflite::testing::PrepareConv2DInt8(0), kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest,
     DepthwiseConv2DInt8TooManyChannelsForArena) {
  EXPECT_EQ(tflite::testing::PrepareDepthwiseConv2DInt8(kChannels),
            kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest,
     DepthwiseConv2DInt8SecondArrayDoesNotFit) {
  EXPECT_EQ(
      tflite::testing::PrepareDepthwiseConv2DInt8(kChannelsSecondArrayFails),
      kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest,
     DepthwiseConv2DInt8ChannelBytesWrapSizeT) {
  EXPECT_EQ(tflite::testing::PrepareDepthwiseConv2DInt8(kChannelsWrapSizeT),
            kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest,
     TransposeConvInt16TooManyChannelsForArena) {
  EXPECT_EQ(tflite::testing::PrepareTransposeConv(tflite::testing::g_input_s16,
                                                  tflite::testing::g_output_s16,
                                                  kChannels),
            kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest,
     TransposeConvInt16SecondArrayDoesNotFit) {
  EXPECT_EQ(tflite::testing::PrepareTransposeConv(tflite::testing::g_input_s16,
                                                  tflite::testing::g_output_s16,
                                                  kChannelsSecondArrayFails),
            kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest,
     TransposeConvInt16ChannelBytesWrapSizeT) {
  EXPECT_EQ(tflite::testing::PrepareTransposeConv(tflite::testing::g_input_s16,
                                                  tflite::testing::g_output_s16,
                                                  kChannelsWrapSizeT),
            kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest,
     TransposeConvInt8TooManyChannelsForArena) {
  EXPECT_EQ(
      tflite::testing::PrepareTransposeConv(
          tflite::testing::g_input_s8, tflite::testing::g_output_s8, kChannels),
      kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest, TransposeConvInt8SecondArrayDoesNotFit) {
  EXPECT_EQ(tflite::testing::PrepareTransposeConv(tflite::testing::g_input_s8,
                                                  tflite::testing::g_output_s8,
                                                  kChannelsSecondArrayFails),
            kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest, ConcatenationInt8ArenaFullAfterInit) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::IntArrayFromInts;
  int input_shape[] = {2, 1, 2};
  int output_shape[] = {2, 1, 4};
  int8_t input1[2] = {};
  int8_t input2[2] = {};
  int8_t output[4] = {};
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input1, IntArrayFromInts(input_shape), 0.5f, 0),
      CreateQuantizedTensor(input2, IntArrayFromInts(input_shape), 0.5f, 0),
      CreateQuantizedTensor(output, IntArrayFromInts(output_shape), 0.5f, 0),
  };
  int inputs_array_data[] = {2, 0, 1};
  int outputs_array_data[] = {1, 2};
  TfLiteConcatenationParams params = {1, kTfLiteActNone};
  const TFLMRegistration registration = tflite::testing::WithInit(
      tflite::Register_CONCATENATION(), tflite::testing::InitThenExhaust);
  tflite::micro::KernelRunner runner(
      registration, tensors, 3, IntArrayFromInts(inputs_array_data),
      IntArrayFromInts(outputs_array_data), &params);
  EXPECT_EQ(runner.InitAndPrepare(), kTfLiteError);
  EXPECT_NE(tflite::testing::g_init_result, nullptr);
}

// The scale array fits and the zero-point array, the third persistent request
// after Init's OpData and the scales, does not.
TEST(HeliaPersistentBufferFailureTest, ConcatenationInt8ZeroPointArrayFails) {
  using tflite::micro::helia::AllocationKind;
  using tflite::micro::helia::RunWithAllocationFailure;
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::IntArrayFromInts;
  int input_shape[] = {2, 1, 2};
  int output_shape[] = {2, 1, 4};
  int8_t input1[2] = {};
  int8_t input2[2] = {};
  int8_t output[4] = {};
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input1, IntArrayFromInts(input_shape), 0.5f, 0),
      CreateQuantizedTensor(input2, IntArrayFromInts(input_shape), 0.5f, 0),
      CreateQuantizedTensor(output, IntArrayFromInts(output_shape), 0.5f, 0),
  };
  int inputs_array_data[] = {2, 0, 1};
  int outputs_array_data[] = {1, 2};
  TfLiteConcatenationParams params = {1, kTfLiteActNone};
  TfLiteNode node = {};
  node.inputs = IntArrayFromInts(inputs_array_data);
  node.outputs = IntArrayFromInts(outputs_array_data);
  node.builtin_data = &params;
  const TFLMRegistration registration = tflite::Register_CONCATENATION();
  const auto full =
      RunWithAllocationFailure(registration, node, tensors, nullptr, 0, -1,
                               false, AllocationKind::kPersistentBuffer);
  ASSERT_EQ(full.prepare_status, kTfLiteOk);
  ASSERT_EQ(full.allocations, 3);
  const auto failed =
      RunWithAllocationFailure(registration, node, tensors, nullptr, 0, 2,
                               false, AllocationKind::kPersistentBuffer);
  EXPECT_FALSE(failed.init_failed);
  EXPECT_EQ(failed.prepare_status, kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest, MeanInt8ArenaFullBeforeInit) {
  EXPECT_EQ(tflite::testing::PrepareReductionWithFullArena<int8_t>(
                tflite::Register_MEAN(), kTfLiteInt8),
            kTfLiteError);
  EXPECT_EQ(tflite::testing::g_init_result, nullptr);
}

TEST(HeliaPersistentBufferFailureTest, ReduceMaxInt8ArenaFullBeforeInit) {
  EXPECT_EQ(tflite::testing::PrepareReductionWithFullArena<int8_t>(
                tflite::Register_REDUCE_MAX(), kTfLiteInt8),
            kTfLiteError);
  EXPECT_EQ(tflite::testing::g_init_result, nullptr);
}

TEST(HeliaPersistentBufferFailureTest, ReduceAllArenaFullBeforeInit) {
  EXPECT_EQ(tflite::testing::PrepareReductionWithFullArena<bool>(
                tflite::Register_REDUCE_ALL(), kTfLiteBool),
            kTfLiteError);
  EXPECT_EQ(tflite::testing::g_init_result, nullptr);
}

// An absent optional bias is a zero bias.
TEST(HeliaPersistentBufferFailureTest, SvdfInt8WithoutBiasMatchesZeroBias) {
  using tflite::testing::IntArrayFromInts;
  int8_t outputs[2][2] = {};
  for (int with_bias = 0; with_bias < 2; ++with_bias) {
    tflite::testing::SvdfInt8 svdf;
    int inputs_array_data[] = {
        5, 0, 1, 2, with_bias ? 3 : kTfLiteOptionalTensor, 4};
    int outputs_array_data[] = {1, 5};
    const TFLMRegistration registration = tflite::Register_SVDF();
    tflite::micro::KernelRunner runner(
        registration, svdf.tensors, 6, IntArrayFromInts(inputs_array_data),
        IntArrayFromInts(outputs_array_data), &svdf.params);
    ASSERT_EQ(runner.InitAndPrepare(), kTfLiteOk);
    ASSERT_EQ(runner.Invoke(), kTfLiteOk);
    outputs[with_bias][0] = svdf.output[0];
    outputs[with_bias][1] = svdf.output[1];
  }
  EXPECT_NE(outputs[1][0], 0);
  EXPECT_EQ(outputs[0][0], outputs[1][0]);
  EXPECT_EQ(outputs[0][1], outputs[1][1]);
}

// The model names a bias, so a nullptr is a failed temp allocation, not an
// absent bias. Bias is the fourth temp tensor Prepare allocates.
TEST(HeliaPersistentBufferFailureTest, SvdfInt8BiasAllocationFails) {
  using tflite::micro::helia::AllocationKind;
  using tflite::micro::helia::RunWithAllocationFailure;
  using tflite::testing::IntArrayFromInts;
  tflite::testing::SvdfInt8 svdf;
  int inputs_array_data[] = {5, 0, 1, 2, 3, 4};
  int outputs_array_data[] = {1, 5};
  TfLiteNode node = {};
  node.inputs = IntArrayFromInts(inputs_array_data);
  node.outputs = IntArrayFromInts(outputs_array_data);
  node.builtin_data = &svdf.params;
  const TFLMRegistration registration = tflite::Register_SVDF();
  const auto full =
      RunWithAllocationFailure(registration, node, svdf.tensors, nullptr, 0, -1,
                               false, AllocationKind::kTempTensor);
  ASSERT_EQ(full.prepare_status, kTfLiteOk);
  const auto failed =
      RunWithAllocationFailure(registration, node, svdf.tensors, nullptr, 0, 3,
                               false, AllocationKind::kTempTensor);
  EXPECT_EQ(failed.prepare_status, kTfLiteError);
}

// A persistent request must not grow the tail over a live temp allocation,
// and one larger than the tail address must not wrap.
TEST(HeliaPersistentBufferFailureTest, PersistentBufferSparesLiveTemp) {
  alignas(16) uint8_t arena[256];
  tflite::SingleArenaBufferAllocator allocator(arena, sizeof(arena));
  uint8_t* temp = allocator.AllocateTemp(200, 1);
  ASSERT_NE(temp, nullptr);
  EXPECT_EQ(allocator.AllocatePersistentBuffer(100, 1), nullptr);
  EXPECT_EQ(allocator.AllocatePersistentBuffer(SIZE_MAX, 1), nullptr);
  uint8_t* persistent = allocator.AllocatePersistentBuffer(56, 1);
  ASSERT_NE(persistent, nullptr);
  EXPECT_GE(persistent, temp + 200);
  allocator.DeallocateTemp(temp);
}

// Temp memory that ends within one alignment unit of an unaligned tail.
TEST(HeliaPersistentBufferFailureTest, TempAtUnalignedTailDoesNotWrap) {
  alignas(16) uint8_t arena[100];
  tflite::SingleArenaBufferAllocator allocator(arena, sizeof(arena));
  uint8_t* temp = allocator.AllocateTemp(sizeof(arena), 1);
  ASSERT_NE(temp, nullptr);
  EXPECT_EQ(allocator.GetAvailableMemory(16), 0u);
  EXPECT_EQ(allocator.AllocateTemp(0, 16), nullptr);
  allocator.DeallocateTemp(temp);
}

// Aligning the head up passes the tail of an 8-byte arena.
TEST(HeliaPersistentBufferFailureTest, ResizableBufferPastTailDoesNotWrap) {
  alignas(16) uint8_t storage[32];
  tflite::SingleArenaBufferAllocator allocator(storage + 1, 8);
  EXPECT_EQ(allocator.AllocateResizableBuffer(4, 16), nullptr);
  EXPECT_EQ(allocator.ReserveNonPersistentOverlayMemory(4, 16), kTfLiteError);
  EXPECT_EQ(allocator.GetNonPersistentUsedBytes(), 0u);
  uint8_t* temp = allocator.AllocateTemp(8, 1);
  EXPECT_EQ(temp, storage + 1);
  allocator.DeallocateTemp(temp);
}

TEST(HeliaPersistentBufferFailureTest, OversizedResizeLeavesAllocatorUsable) {
  alignas(16) uint8_t arena[256];
  tflite::SingleArenaBufferAllocator allocator(arena, sizeof(arena));
  EXPECT_EQ(allocator.AllocateResizableBuffer(SIZE_MAX, 16), nullptr);
  EXPECT_EQ(allocator.GetNonPersistentUsedBytes(), 0u);
  EXPECT_EQ(allocator.AllocateResizableBuffer(64, 16), arena);
  EXPECT_NE(allocator.AllocatePersistentBuffer(64, 16), nullptr);
}

TF_LITE_MICRO_TESTS_MAIN
