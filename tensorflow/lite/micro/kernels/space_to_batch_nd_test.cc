/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kBasicInputOutputSize = 16;
int basic_input_dims[] = {4, 1, 4, 4, 1};
const float basic_input[kBasicInputOutputSize] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
int basic_block_shape_dims[] = {1, 2};
const int32_t basic_block_shape[] = {2, 2};
int basic_crops_dims[] = {1, 4};
const int32_t basic_crops[] = {0, 0, 0, 0};
int basic_output_dims[] = {4, 4, 2, 2, 1};
const float basic_golden[kBasicInputOutputSize] = {1, 3, 9,  11, 2, 4, 10, 12,
                                                   5, 7, 13, 15, 6, 8, 14, 16};

void* InitWithNonzeroPadding(TfLiteContext* context, const char* buffer,
                             size_t length) {
  void* data = Register_SPACE_TO_BATCH_ND().init(context, buffer, length);
  // Persistent storage is not guaranteed to be zero-filled before Prepare.
  static_cast<SpaceToBatchParams*>(data)->output_offset = 0x55555555;
  return data;
}

template <typename T>
TfLiteStatus ValidateSpaceToBatchNdGoldens(TfLiteTensor* tensors,
                                           int tensors_size, const T* golden,
                                           T* output, int output_size,
                                           bool invoke = true) {
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TFLMRegistration registration = Register_SPACE_TO_BATCH_ND();
  registration.init = InitWithNonzeroPadding;
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_ENSURE_STATUS(runner.InitAndPrepare());
  if (!invoke) return kTfLiteOk;
  TF_LITE_ENSURE_STATUS(runner.Invoke());

  for (int i = 0; i < output_size; ++i) {
    // TODO(b/158102673): workaround for not having fatal test assertions.
    EXPECT_EQ(golden[i], output[i]);
    if (golden[i] != output[i]) {
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus TestSpaceToBatchNdFloat(
    int* input_dims_data, const float* input_data, int* block_shape_dims_data,
    const int32_t* block_shape_data, int* crops_dims_data,
    const int32_t* crops_data, int* output_dims_data, const float* golden,
    float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* block_shape_dims = IntArrayFromInts(block_shape_dims_data);
  TfLiteIntArray* crops_dims = IntArrayFromInts(crops_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(block_shape_data, block_shape_dims),
      CreateTensor(crops_data, crops_dims),
      CreateTensor(output_data, output_dims),
  };

  return ValidateSpaceToBatchNdGoldens(tensors, tensors_size, golden,
                                       output_data, ElementCount(*output_dims));
}

template <typename T>
TfLiteStatus TestSpaceToBatchNdQuantized(
    int* input_dims_data, const float* input_data, T* input_quantized,
    float input_scale, int input_zero_point, int* block_shape_dims_data,
    const int32_t* block_shape_data, int* crops_dims_data,
    const int32_t* crops_data, int* output_dims_data, const float* golden,
    T* golden_quantized, float output_scale, int output_zero_point,
    T* output_data, bool invoke = true) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* block_shape_dims = IntArrayFromInts(block_shape_dims_data);
  TfLiteIntArray* crops_dims = IntArrayFromInts(crops_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      tflite::testing::CreateQuantizedTensor(input_data, input_quantized,
                                             input_dims, input_scale,
                                             input_zero_point),
      tflite::testing::CreateTensor(block_shape_data, block_shape_dims),
      tflite::testing::CreateTensor(crops_data, crops_dims),
      tflite::testing::CreateQuantizedTensor(output_data, output_dims,
                                             output_scale, output_zero_point),
  };
  tflite::Quantize(golden, golden_quantized, ElementCount(*output_dims),
                   output_scale, output_zero_point);

  return ValidateSpaceToBatchNdGoldens(tensors, tensors_size, golden_quantized,
                                       output_data, ElementCount(*output_dims),
                                       invoke);
}

// helia: constant block shape and paddings, as converter output has them; the
// output's stored shape must be the one they give. Prepare-only cases pass no
// data. see AmbiqAI/helia-rt#407
template <typename BlockT = int32_t>
TfLiteStatus RunSpaceToBatchConstant(int* input_dims, const float* input,
                                     const BlockT* block,
                                     const int32_t* paddings, int* output_dims,
                                     float* output, bool constant_block = true,
                                     bool constant_paddings = true,
                                     int block_size = -1) {
  int block_dims[] = {1, block_size >= 0 ? block_size : input_dims[0] - 2};
  int paddings_dims[] = {2, input_dims[0] - 2, 2};
  TfLiteTensor tensors[] = {
      CreateTensor(input, IntArrayFromInts(input_dims)),
      CreateTensor(block, IntArrayFromInts(block_dims)),
      CreateTensor(paddings, IntArrayFromInts(paddings_dims)),
      CreateTensor(output, IntArrayFromInts(output_dims)),
  };
  if (constant_block) {
    tensors[1].allocation_type = kTfLiteMmapRo;
  }
  if (constant_paddings) {
    tensors[2].allocation_type = kTfLiteMmapRo;
  }
  int inputs_array_data[] = {3, 0, 1, 2};
  int outputs_array_data[] = {1, 3};
  const TFLMRegistration registration = Register_SPACE_TO_BATCH_ND();
  micro::KernelRunner runner(registration, tensors, 4,
                             IntArrayFromInts(inputs_array_data),
                             IntArrayFromInts(outputs_array_data), nullptr);
  const TfLiteStatus status = runner.InitAndPrepare();
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
  if (status != kTfLiteOk || input == nullptr) {
    return status;
  }
  return runner.Invoke();
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(SpaceToBatchNdTest, SpaceToBatchBasicFloat) {
  float output[tflite::testing::kBasicInputOutputSize];
  EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestSpaceToBatchNdFloat(
          tflite::testing::basic_input_dims, tflite::testing::basic_input,
          tflite::testing::basic_block_shape_dims,
          tflite::testing::basic_block_shape, tflite::testing::basic_crops_dims,
          tflite::testing::basic_crops, tflite::testing::basic_output_dims,
          tflite::testing::basic_golden, output));
}

TEST(SpaceToBatchNdTest, SpaceToBatchBasicInt8) {
  int8_t output[tflite::testing::kBasicInputOutputSize];
  int8_t input_quantized[tflite::testing::kBasicInputOutputSize];
  int8_t golden_quantized[tflite::testing::kBasicInputOutputSize];
  EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestSpaceToBatchNdQuantized(
          tflite::testing::basic_input_dims, tflite::testing::basic_input,
          input_quantized, 1.0f, 0, tflite::testing::basic_block_shape_dims,
          tflite::testing::basic_block_shape, tflite::testing::basic_crops_dims,
          tflite::testing::basic_crops, tflite::testing::basic_output_dims,
          tflite::testing::basic_golden, golden_quantized, 1.0f, 0, output));
}

TEST(SpaceToBatchNdTest, PaddedInt8) {
  int input_dims[] = {4, 1, 1, 1, 2};
  const float input[] = {1, 2};
  int block_dims[] = {1, 2};
  const int32_t block[] = {2, 2};
  int padding_dims[] = {2, 2, 2};
  const int32_t padding[] = {1, 0, 0, 1};
  int output_dims[] = {4, 4, 1, 1, 2};
  const float golden[] = {0, 0, 0, 0, 1, 2, 0, 0};
  for (int zero_point : {-128, -17, 0, 42, 127}) {
    int8_t input_quantized[2];
    int8_t golden_quantized[8];
    int8_t output[8];
    EXPECT_EQ(kTfLiteOk,
              tflite::testing::TestSpaceToBatchNdQuantized(
                  input_dims, input, input_quantized, 0.5f, zero_point,
                  block_dims, block, padding_dims, padding, output_dims, golden,
                  golden_quantized, 0.5f, zero_point, output));
  }
}

TEST(SpaceToBatchNdTest, PaddedFloat) {
  int input_dims[] = {4, 1, 1, 1, 2};
  const float input[] = {-1.5f, 2.5f};
  int block_dims[] = {1, 2};
  const int32_t block[] = {2, 2};
  int padding_dims[] = {2, 2, 2};
  const int32_t padding[] = {1, 0, 0, 1};
  int output_dims[] = {4, 4, 1, 1, 2};
  const float golden[] = {0, 0, 0, 0, -1.5f, 2.5f, 0, 0};
  float output[8];
  EXPECT_EQ(kTfLiteOk, tflite::testing::TestSpaceToBatchNdFloat(
                           input_dims, input, block_dims, block, padding_dims,
                           padding, output_dims, golden, output));
}

TEST(SpaceToBatchNdTest, PaddedInt8ThreeDimensions) {
  int input_dims[] = {3, 1, 1, 2};
  const float input[] = {1, 2};
  int block_dims[] = {1, 1};
  const int32_t block[] = {2};
  int padding_dims[] = {2, 1, 2};
  const int32_t padding[] = {1, 0};
  int output_dims[] = {3, 2, 1, 2};
  const float golden[] = {0, 0, 1, 2};
  int8_t input_quantized[2];
  int8_t golden_quantized[4];
  int8_t output[4];
  EXPECT_EQ(kTfLiteOk,
            tflite::testing::TestSpaceToBatchNdQuantized(
                input_dims, input, input_quantized, 0.5f, -128, block_dims,
                block, padding_dims, padding, output_dims, golden,
                golden_quantized, 0.5f, -128, output));
}

TEST(SpaceToBatchNdTest, RejectMismatchedInt8Quantization) {
  int8_t output[tflite::testing::kBasicInputOutputSize];
  int8_t input_quantized[tflite::testing::kBasicInputOutputSize];
  int8_t golden_quantized[tflite::testing::kBasicInputOutputSize];
  const float output_scales[] = {2.0f, 1.0f};
  const int output_zero_points[] = {0, 1};
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(
        kTfLiteError,
        tflite::testing::TestSpaceToBatchNdQuantized(
            tflite::testing::basic_input_dims, tflite::testing::basic_input,
            input_quantized, 1.0f, 0, tflite::testing::basic_block_shape_dims,
            tflite::testing::basic_block_shape,
            tflite::testing::basic_crops_dims, tflite::testing::basic_crops,
            tflite::testing::basic_output_dims, tflite::testing::basic_golden,
            golden_quantized, output_scales[i], output_zero_points[i], output,
            false));
  }
}

TEST(SpaceToBatchNdTest, ConstantBlockShapeAndPaddingsMatchGolden) {
  // [1,1,4,1] padded to width 6, block [1,2]: batch 2, width 3.
  int input_dims[] = {4, 1, 1, 4, 1};
  const float input[] = {1, 2, 3, 4};
  const int32_t block[] = {1, 2};
  const int32_t paddings[] = {0, 0, 1, 1};
  int output_dims[] = {4, 2, 1, 3, 1};
  float output[6];
  ASSERT_EQ(kTfLiteOk,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, input, block, paddings, output_dims, output));
  const float golden[] = {0, 2, 4, 1, 3, 0};
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(golden[i], output[i]);
  }
}

TEST(SpaceToBatchNdTest, ConstantBlockShapeMatchesGoldenThreeDimensions) {
  int input_dims[] = {3, 1, 4, 1};
  const float input[] = {1, 2, 3, 4};
  const int32_t block[] = {2};
  const int32_t paddings[] = {0, 0};
  int output_dims[] = {3, 2, 2, 1};
  float output[4];
  ASSERT_EQ(kTfLiteOk,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, input, block, paddings, output_dims, output));
  const float golden[] = {1, 3, 2, 4};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(golden[i], output[i]);
  }
}

// A dynamic-batch export stores batch 1 where the block shape makes 2.
TEST(SpaceToBatchNdTest, RejectsPlaceholderOutputBatch) {
  int input_dims[] = {4, 1, 1, 4, 1};
  const float input[] = {1, 2, 3, 4};
  const int32_t block[] = {1, 2};
  const int32_t paddings[] = {0, 0, 1, 1};
  int output_dims[] = {4, 1, 1, 3, 1};
  float output[6];
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, input, block, paddings, output_dims, output));
}

TEST(SpaceToBatchNdTest, RejectsPlaceholderOutputBatchThreeDimensions) {
  int input_dims[] = {3, 1, 4, 1};
  const float input[] = {1, 2, 3, 4};
  const int32_t block[] = {2};
  const int32_t paddings[] = {0, 0};
  int output_dims[] = {3, 1, 2, 1};
  float output[4];
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, input, block, paddings, output_dims, output));
}

TEST(SpaceToBatchNdTest, RejectsMismatchedSpatialOutputDim) {
  int input_dims[] = {4, 1, 1, 4, 1};
  const float input[] = {1, 2, 3, 4};
  const int32_t block[] = {1, 2};
  const int32_t paddings[] = {0, 0, 1, 1};
  int output_dims[] = {4, 2, 1, 4, 1};
  float output[8];
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, input, block, paddings, output_dims, output));
}

// The batch product exceeds INT32_MAX; checked without overflow.
TEST(SpaceToBatchNdTest, RejectsBatchAboveInt32) {
  int input_dims[] = {4, 1, 1, 1, 1};
  const float input[] = {1};
  const int32_t block[] = {65536, 65536};
  const int32_t paddings[] = {65535, 0, 65535, 0};
  int output_dims[] = {4, 1, 1, 1, 1};
  float output[1];
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, input, block, paddings, output_dims, output));
}

// Unbounded, 2^20 * 2^30 * 2^30 would wrap int64 to the stored batch 0.
TEST(SpaceToBatchNdTest, RejectsBatchProductThatWouldOverflow) {
  int input_dims[] = {4, 1048576, 1, 1, 1};
  const int32_t block[] = {1073741824, 1073741824};
  const int32_t paddings[] = {1073741823, 0, 1073741823, 0};
  int output_dims[] = {4, 0, 1, 1, 1};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, nullptr, block, paddings, output_dims, nullptr));
}

// A negative batch times the blocks would overflow int64.
TEST(SpaceToBatchNdTest, RejectsNegativeInputBatch) {
  int input_dims[] = {4, -306184046, 1, 1, 1};
  const int32_t block[] = {92737, 649657};
  const int32_t paddings[] = {92736, 0, 649656, 0};
  int output_dims[] = {4, 2, 1, 1, 1};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, nullptr, block, paddings, output_dims, nullptr));
}

TEST(SpaceToBatchNdTest, RejectsZeroBlock) {
  int input_dims[] = {4, 1, 1, 4, 1};
  const int32_t block[] = {1, 0};
  const int32_t paddings[] = {0, 0, 0, 0};
  int output_dims[] = {4, 1, 1, 4, 1};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, nullptr, block, paddings, output_dims, nullptr));
}

TEST(SpaceToBatchNdTest, RejectsNegativeBlock) {
  int input_dims[] = {4, 1, 1, 4, 1};
  const int32_t block[] = {1, -2};
  const int32_t paddings[] = {0, 0, 0, 0};
  int output_dims[] = {4, -2, 1, -2, 1};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, nullptr, block, paddings, output_dims, nullptr));
}

TEST(SpaceToBatchNdTest, RejectsNegativePadding) {
  int input_dims[] = {4, 1, 1, 4, 1};
  const int32_t block[] = {1, 2};
  const int32_t paddings[] = {0, 0, -1, -1};
  int output_dims[] = {4, 2, 1, 1, 1};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, nullptr, block, paddings, output_dims, nullptr));
}

TEST(SpaceToBatchNdTest, RejectsPaddedSizeNotMultipleOfBlock) {
  int input_dims[] = {4, 1, 1, 5, 1};
  const int32_t block[] = {1, 2};
  const int32_t paddings[] = {0, 0, 0, 0};
  int output_dims[] = {4, 2, 1, 2, 1};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, nullptr, block, paddings, output_dims, nullptr));
}

TEST(SpaceToBatchNdTest, RejectsNegativeInputSpatialDim) {
  int input_dims[] = {4, 1, 1, -2, 1};
  const int32_t block[] = {1, 2};
  const int32_t paddings[] = {0, 0, 2, 2};
  int output_dims[] = {4, 2, 1, 1, 1};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, nullptr, block, paddings, output_dims, nullptr));
}

TEST(SpaceToBatchNdTest, RejectsNegativeChannelDim) {
  int input_dims[] = {4, 1, 1, 4, -1};
  const int32_t block[] = {1, 2};
  const int32_t paddings[] = {0, 0, 0, 0};
  int output_dims[] = {4, 2, 1, 2, -1};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, nullptr, block, paddings, output_dims, nullptr));
}

TEST(SpaceToBatchNdTest, RejectsInt64BlockShape) {
  int input_dims[] = {3, 1, 4, 1};
  const int64_t block[] = {2};
  const int32_t paddings[] = {0, 0};
  int output_dims[] = {3, 2, 2, 1};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, nullptr, block, paddings, output_dims, nullptr));
}

TEST(SpaceToBatchNdTest, RejectsBlockShapeOfWrongLength) {
  int input_dims[] = {4, 1, 1, 4, 1};
  const int32_t block[] = {1, 2};
  const int32_t paddings[] = {0, 0, 0, 0};
  int output_dims[] = {4, 2, 1, 2, 1};
  EXPECT_EQ(kTfLiteError, tflite::testing::RunSpaceToBatchConstant(
                              input_dims, nullptr, block, paddings, output_dims,
                              nullptr, true, true, 1));
}

TEST(SpaceToBatchNdTest, RejectsOutputRankMismatch) {
  int input_dims[] = {3, 1, 4, 1};
  const int32_t block[] = {2};
  const int32_t paddings[] = {0, 0};
  int output_dims[] = {4, 2, 2, 1, 1};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, nullptr, block, paddings, output_dims, nullptr));
}

TEST(SpaceToBatchNdTest, RejectsChannelMismatch) {
  int input_dims[] = {4, 1, 1, 4, 1};
  const int32_t block[] = {1, 2};
  const int32_t paddings[] = {0, 0, 0, 0};
  int output_dims[] = {4, 2, 1, 2, 2};
  EXPECT_EQ(kTfLiteError,
            tflite::testing::RunSpaceToBatchConstant(
                input_dims, nullptr, block, paddings, output_dims, nullptr));
}

// The batch depends only on the block shape.
TEST(SpaceToBatchNdTest, RejectsPlaceholderBatchWithRuntimePaddings) {
  int input_dims[] = {4, 1, 1, 4, 1};
  const int32_t block[] = {1, 2};
  const int32_t paddings[] = {0, 0, 0, 0};
  int output_dims[] = {4, 1, 1, 2, 1};
  EXPECT_EQ(kTfLiteError, tflite::testing::RunSpaceToBatchConstant(
                              input_dims, nullptr, block, paddings, output_dims,
                              nullptr, true, false));
}

// Not known before Eval: upstream behaviour.
TEST(SpaceToBatchNdTest, AcceptsAnyStoredShapeWithRuntimeBlockShape) {
  int input_dims[] = {4, 1, 1, 4, 1};
  const int32_t block[] = {1, 2};
  const int32_t paddings[] = {0, 0, 0, 0};
  int output_dims[] = {4, 1, 1, 3, 1};
  EXPECT_EQ(kTfLiteOk, tflite::testing::RunSpaceToBatchConstant(
                           input_dims, nullptr, block, paddings, output_dims,
                           nullptr, false, false));
}

TF_LITE_MICRO_TESTS_MAIN
