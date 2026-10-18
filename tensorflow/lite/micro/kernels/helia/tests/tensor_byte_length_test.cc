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

// Tensor byte sizes come from model shapes, so BytesRequiredForTensor and
// TfLiteEvalTensorByteLength must fail on a negative dimension, an element
// count above INT_MAX or a byte count above SIZE_MAX instead of wrapping, and
// that failure must reach AllocateTensors and CopyOpInputsToOpOutputs.
// see AmbiqAI/helia-rt#453

#include <climits>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

// [16, 2^28 + 1] holds 2^32 + 16 elements, which a 32-bit product wraps to 16.
constexpr int32_t kWrapsToSixteen = (1 << 28) + 1;

const tflite::Tensor* FlatbufferInt8Tensor(
    const std::initializer_list<int32_t>& shape) {
  // The second tensor of this model carries `shape`; the model is rebuilt in
  // place on each call.
  return tflite::testing::GetNoOpModelWithTensorShape(shape)
      ->subgraphs()
      ->Get(0)
      ->tensors()
      ->Get(1);
}

TfLiteStatus FlatbufferBytes(const tflite::Tensor* tensor, size_t* bytes) {
  size_t type_size = 0;
  return tflite::BytesRequiredForTensor(*tensor, bytes, &type_size);
}

TfLiteStatus EvalBytes(TfLiteType type, int* dims_data, size_t* bytes) {
  TfLiteEvalTensor tensor = {};
  tensor.type = type;
  tensor.dims = tflite::testing::IntArrayFromInts(dims_data);
  return tflite::TfLiteEvalTensorByteLength(&tensor, bytes);
}

TfLiteEvalTensor copy_tensors[2];

TfLiteEvalTensor* GetCopyTensor(const TfLiteContext* context, int index) {
  return &copy_tensors[index];
}

void IgnoreError(TfLiteContext* context, const char* format, ...) {}

}  // namespace

TEST(TensorByteLengthTest, FlatbufferElementCountOverflowFails) {
  size_t bytes = 0;
  EXPECT_EQ(kTfLiteError,
            FlatbufferBytes(FlatbufferInt8Tensor({65536, 65536}), &bytes));
  EXPECT_EQ(
      kTfLiteError,
      FlatbufferBytes(FlatbufferInt8Tensor({16, kWrapsToSixteen}), &bytes));
  EXPECT_EQ(
      kTfLiteError,
      FlatbufferBytes(FlatbufferInt8Tensor({1, 65536, 65536, 1}), &bytes));
}

TEST(TensorByteLengthTest, FlatbufferNegativeDimensionFails) {
  size_t bytes = 0;
  EXPECT_EQ(kTfLiteError,
            FlatbufferBytes(FlatbufferInt8Tensor({3, 2, -1}), &bytes));
  EXPECT_EQ(kTfLiteError,
            FlatbufferBytes(FlatbufferInt8Tensor({-1, -1}), &bytes));
  // A zero before the negative dimension must not hide it.
  EXPECT_EQ(kTfLiteError,
            FlatbufferBytes(FlatbufferInt8Tensor({0, -1}), &bytes));
}

TEST(TensorByteLengthTest, FlatbufferByteCountOverflowFailsOn32BitSizeT) {
  // 2^30 int32 elements are 2^32 bytes.
  size_t bytes = 0;
  const TfLiteStatus status = FlatbufferBytes(
      tflite::testing::Create1dFlatbufferTensor(1 << 30), &bytes);
  if (sizeof(size_t) == 4) {
    EXPECT_EQ(kTfLiteError, status);
  } else {
    EXPECT_EQ(kTfLiteOk, status);
    EXPECT_EQ(uint64_t{1} << 32, static_cast<uint64_t>(bytes));
  }
}

TEST(TensorByteLengthTest, FlatbufferLargestElementCountSucceeds) {
  size_t bytes = 0;
  EXPECT_EQ(kTfLiteOk,
            FlatbufferBytes(FlatbufferInt8Tensor({INT_MAX}), &bytes));
  EXPECT_EQ(static_cast<size_t>(INT_MAX), bytes);
  EXPECT_EQ(kTfLiteOk,
            FlatbufferBytes(FlatbufferInt8Tensor({0, 65536, 65536}), &bytes));
  EXPECT_EQ(static_cast<size_t>(0), bytes);
  EXPECT_EQ(kTfLiteOk, FlatbufferBytes(FlatbufferInt8Tensor({3, 2}), &bytes));
  EXPECT_EQ(static_cast<size_t>(6), bytes);
}

TEST(TensorByteLengthTest, EvalTensorElementCountOverflowFails) {
  size_t bytes = 0;
  int square[] = {2, 65536, 65536};
  EXPECT_EQ(kTfLiteError, EvalBytes(kTfLiteInt8, square, &bytes));
  int wraps[] = {2, 16, kWrapsToSixteen};
  EXPECT_EQ(kTfLiteError, EvalBytes(kTfLiteInt8, wraps, &bytes));
}

TEST(TensorByteLengthTest, EvalTensorNegativeDimensionFails) {
  size_t bytes = 0;
  int negative[] = {3, 3, 2, -1};
  EXPECT_EQ(kTfLiteError, EvalBytes(kTfLiteInt8, negative, &bytes));
  int zero_then_negative[] = {2, 0, -1};
  EXPECT_EQ(kTfLiteError, EvalBytes(kTfLiteInt8, zero_then_negative, &bytes));
}

TEST(TensorByteLengthTest, EvalTensorByteCountOverflowFailsOn32BitSizeT) {
  size_t bytes = 0;
  int dims[] = {2, 1, 1 << 30};
  const TfLiteStatus status = EvalBytes(kTfLiteInt32, dims, &bytes);
  if (sizeof(size_t) == 4) {
    EXPECT_EQ(kTfLiteError, status);
  } else {
    EXPECT_EQ(kTfLiteOk, status);
    EXPECT_EQ(uint64_t{1} << 32, static_cast<uint64_t>(bytes));
  }
}

TEST(TensorByteLengthTest, EvalTensorLargestElementCountSucceeds) {
  size_t bytes = 0;
  int largest[] = {1, INT_MAX};
  EXPECT_EQ(kTfLiteOk, EvalBytes(kTfLiteInt8, largest, &bytes));
  EXPECT_EQ(static_cast<size_t>(INT_MAX), bytes);
  int scalar[] = {0};
  EXPECT_EQ(kTfLiteOk, EvalBytes(kTfLiteFloat32, scalar, &bytes));
  EXPECT_EQ(sizeof(float), bytes);
}

// The size_t bound, exercised with a 32-bit limit so it runs on 64-bit hosts.
TEST(TensorByteLengthTest, MultiplyTensorSizeHonoursLimit) {
  constexpr size_t kLimit = UINT32_MAX;
  size_t size = size_t{1} << 30;
  EXPECT_EQ(kTfLiteError, tflite::MultiplyTensorSize(&size, 4, kLimit));
  EXPECT_EQ(size_t{1} << 30, size);
  EXPECT_EQ(kTfLiteOk, tflite::MultiplyTensorSize(&size, 3, kLimit));
  EXPECT_EQ(size_t{3} << 30, size);

  size = kLimit / 7 + 1;
  EXPECT_EQ(kTfLiteError, tflite::MultiplyTensorSize(&size, 7, kLimit));
  size = kLimit / 7;
  EXPECT_EQ(kTfLiteOk, tflite::MultiplyTensorSize(&size, 7, kLimit));
  EXPECT_EQ(kLimit / 7 * 7, size);

  size = kLimit;
  EXPECT_EQ(kTfLiteOk, tflite::MultiplyTensorSize(&size, 1, kLimit));
  EXPECT_EQ(kLimit, size);
  EXPECT_EQ(kTfLiteOk, tflite::MultiplyTensorSize(&size, 0, kLimit));
  EXPECT_EQ(static_cast<size_t>(0), size);

  // A factor above the limit, which does not fit a 32-bit size_t.
  constexpr int64_t kAboveLimit = int64_t{UINT32_MAX} + 1;
  size = 1;
  EXPECT_EQ(kTfLiteError,
            tflite::MultiplyTensorSize(&size, kAboveLimit, kLimit));
  EXPECT_EQ(static_cast<size_t>(1), size);
  size = 0;
  EXPECT_EQ(kTfLiteOk, tflite::MultiplyTensorSize(&size, kAboveLimit, kLimit));
  EXPECT_EQ(static_cast<size_t>(0), size);

  size = 1;
  EXPECT_EQ(kTfLiteError, tflite::MultiplyTensorSize(&size, -1, kLimit));
  EXPECT_EQ(static_cast<size_t>(1), size);
}

TEST(TensorByteLengthTest, AllocateTensorsFailsOnSizeOverflow) {
  tflite::testing::TestingOpResolver op_resolver;
  ASSERT_EQ(kTfLiteOk, tflite::testing::GetTestingOpResolver(op_resolver));
  constexpr size_t kArenaSize = 2000;
  alignas(16) static uint8_t arena[kArenaSize];

  {
    tflite::MicroInterpreter interpreter(
        tflite::testing::GetNoOpModelWithTensorShape({16, kWrapsToSixteen}),
        op_resolver, arena, kArenaSize);
    EXPECT_EQ(kTfLiteError, interpreter.AllocateTensors());
  }
  {
    tflite::MicroInterpreter interpreter(
        tflite::testing::GetNoOpModelWithTensorShape({65536, 65536}),
        op_resolver, arena, kArenaSize);
    EXPECT_EQ(kTfLiteError, interpreter.AllocateTensors());
  }
  {
    // The same model with a representable shape still allocates.
    tflite::MicroInterpreter interpreter(
        tflite::testing::GetNoOpModelWithTensorShape({16, 2}), op_resolver,
        arena, kArenaSize);
    EXPECT_EQ(kTfLiteOk, interpreter.AllocateTensors());
  }
}

TEST(TensorByteLengthTest, CopyOpInputsToOpOutputsFailsOnSizeOverflow) {
  int8_t input_data[16] = {1};
  int8_t output_data[16] = {};
  int dims_data[] = {2, 16, kWrapsToSixteen};
  TfLiteIntArray* dims = tflite::testing::IntArrayFromInts(dims_data);
  copy_tensors[0] = {};
  copy_tensors[0].data.int8 = input_data;
  copy_tensors[0].dims = dims;
  copy_tensors[0].type = kTfLiteInt8;
  copy_tensors[1] = copy_tensors[0];
  copy_tensors[1].data.int8 = output_data;

  TfLiteContext context = {};
  context.GetEvalTensor = GetCopyTensor;
  context.ReportError = IgnoreError;
  int inputs_data[] = {1, 0};
  int outputs_data[] = {1, 1};
  TfLiteNode node = {};
  node.inputs = tflite::testing::IntArrayFromInts(inputs_data);
  node.outputs = tflite::testing::IntArrayFromInts(outputs_data);

  EXPECT_EQ(kTfLiteError,
            tflite::micro::CopyOpInputsToOpOutputs(&context, &node));
  EXPECT_EQ(0, output_data[0]);
}

TF_LITE_MICRO_TESTS_MAIN
