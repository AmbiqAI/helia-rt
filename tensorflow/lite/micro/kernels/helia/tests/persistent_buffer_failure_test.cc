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
// The convolutions use a per-tensor filter scale with 4096 output channels,
// so their two per-channel arrays (16 KiB each) cannot fit KernelRunner's
// 10000-byte arena. The single-arena allocator must not hand out memory that
// overlaps a live temp allocation or lies outside the arena.
// see AmbiqAI/helia-rt#449

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/transpose_conv.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kChannels = 4096;

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
  return PrepareConv(Register_CONV_2D(), tensors, 3, &params);
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
  return PrepareConv(Register_DEPTHWISE_CONV_2D(), tensors, 3, &params);
}

template <typename T>
TfLiteStatus PrepareTransposeConv(T* input_data, T* output_data) {
  int output_shape_shape[] = {1, 4};
  int32_t output_shape_data[] = {1, 1, 1, kChannels};
  int input_shape[] = {4, 1, 1, 1, 1};
  int filter_shape[] = {4, kChannels, 1, 1, 1};
  int output_shape[] = {4, 1, 1, 1, kChannels};
  PerTensorQuantization quant(0);
  TfLiteTensor tensors[] = {
      CreateTensor(output_shape_data, IntArrayFromInts(output_shape_shape)),
      PerTensorFilter(IntArrayFromInts(filter_shape), &quant.params),
      CreateQuantizedTensor(input_data, IntArrayFromInts(input_shape), 0.5f, 0),
      CreateQuantizedTensor(output_data, IntArrayFromInts(output_shape), 0.5f,
                            0),
  };
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

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(HeliaPersistentBufferFailureTest, Conv2DInt8TooManyChannelsForArena) {
  EXPECT_EQ(tflite::testing::PrepareConv2DInt8(tflite::testing::kChannels),
            kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest, Conv2DInt8ZeroChannels) {
  EXPECT_EQ(tflite::testing::PrepareConv2DInt8(0), kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest,
     DepthwiseConv2DInt8TooManyChannelsForArena) {
  EXPECT_EQ(
      tflite::testing::PrepareDepthwiseConv2DInt8(tflite::testing::kChannels),
      kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest,
     TransposeConvInt16TooManyChannelsForArena) {
  EXPECT_EQ(tflite::testing::PrepareTransposeConv(
                tflite::testing::g_input_s16, tflite::testing::g_output_s16),
            kTfLiteError);
}

TEST(HeliaPersistentBufferFailureTest,
     TransposeConvInt8TooManyChannelsForArena) {
  EXPECT_EQ(tflite::testing::PrepareTransposeConv(tflite::testing::g_input_s8,
                                                  tflite::testing::g_output_s8),
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

TEST(HeliaPersistentBufferFailureTest, MeanInt8ArenaFullBeforeInit) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::CreateTensor;
  using tflite::testing::IntArrayFromInts;
  int input_shape[] = {2, 1, 4};
  int axis_shape[] = {1, 1};
  int output_shape[] = {2, 1, 1};
  int8_t input[4] = {};
  int32_t axis[1] = {1};
  int8_t output[1] = {};
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input, IntArrayFromInts(input_shape), 0.5f, 0),
      CreateTensor(axis, IntArrayFromInts(axis_shape)),
      CreateQuantizedTensor(output, IntArrayFromInts(output_shape), 0.5f, 0),
  };
  int inputs_array_data[] = {2, 0, 1};
  int outputs_array_data[] = {1, 2};
  TfLiteReducerParams params = {true};
  const TFLMRegistration registration = tflite::testing::WithInit(
      tflite::Register_MEAN(), tflite::testing::ExhaustThenInit);
  tflite::micro::KernelRunner runner(
      registration, tensors, 3, IntArrayFromInts(inputs_array_data),
      IntArrayFromInts(outputs_array_data), &params);
  EXPECT_EQ(runner.InitAndPrepare(), kTfLiteError);
  EXPECT_EQ(tflite::testing::g_init_result, nullptr);
}

// The integer SVDF reads the bias scale, and a temp allocation in a full arena
// returns nullptr for the bias just as an absent bias does.
TEST(HeliaPersistentBufferFailureTest, SvdfInt8WithoutBias) {
  using tflite::testing::CreateQuantizedTensor;
  using tflite::testing::IntArrayFromInts;
  int input_shape[] = {2, 1, 2};
  int weights_feature_shape[] = {2, 2, 2};
  int weights_time_shape[] = {2, 2, 2};
  int activation_state_shape[] = {2, 1, 4};
  int output_shape[] = {2, 1, 2};
  int8_t input[2] = {};
  int8_t weights_feature[4] = {};
  int16_t weights_time[4] = {};
  int16_t activation_state[4] = {};
  int8_t output[2] = {};
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input, IntArrayFromInts(input_shape), 0.5f, 0),
      CreateQuantizedTensor(weights_feature,
                            IntArrayFromInts(weights_feature_shape), 0.5f, 0),
      CreateQuantizedTensor(weights_time, IntArrayFromInts(weights_time_shape),
                            0.5f, 0),
      CreateQuantizedTensor(activation_state,
                            IntArrayFromInts(activation_state_shape), 0.5f, 0,
                            /*is_variable=*/true),
      CreateQuantizedTensor(output, IntArrayFromInts(output_shape), 0.5f, 0),
  };
  int inputs_array_data[] = {5, 0, 1, 2, kTfLiteOptionalTensor, 3};
  int outputs_array_data[] = {1, 4};
  TfLiteSVDFParams params = {1, kTfLiteActNone, false};
  const TFLMRegistration registration = tflite::Register_SVDF();
  tflite::micro::KernelRunner runner(
      registration, tensors, 5, IntArrayFromInts(inputs_array_data),
      IntArrayFromInts(outputs_array_data), &params);
  EXPECT_EQ(runner.InitAndPrepare(), kTfLiteError);
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

TF_LITE_MICRO_TESTS_MAIN
