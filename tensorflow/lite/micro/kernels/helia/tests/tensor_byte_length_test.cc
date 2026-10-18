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
// count above INT_MAX or a byte count above kMaxTensorBytes, the largest the
// memory planner can hold once rounded up to the arena alignment. The failure
// must stop AllocateTensors before any kernel Prepare and must reach
// CopyOpInputsToOpOutputs. see AmbiqAI/helia-rt#453

#include <climits>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

using flatbuffers::Offset;
using tflite::TensorType;
using tflite::TensorType_FLOAT32;
using tflite::TensorType_INT16;
using tflite::TensorType_INT32;
using tflite::TensorType_INT8;

// [16, 2^28 + 1] holds 2^32 + 16 elements, which a 32-bit product wraps to 16.
constexpr int32_t kWrapsToSixteen = (1 << 28) + 1;
constexpr int32_t kMaxInt8Elements =
    static_cast<int32_t>(tflite::kMaxTensorBytes);
constexpr int32_t kMaxInt16Elements =
    static_cast<int32_t>(tflite::kMaxTensorBytes / 2);

// Hands out static storage, so flatbuffers build without a heap. deallocate()
// keeps the memory, so a finished buffer outlives its builder and stays valid
// until the next Reset().
class StaticAllocator : public flatbuffers::Allocator {
 public:
  void Reset() { used_ = 0; }
  uint8_t* allocate(size_t size) override {
    const size_t aligned = (size + 15) & ~size_t{15};
    if (aligned > sizeof(storage_) - used_) return nullptr;
    uint8_t* result = storage_ + used_;
    used_ += aligned;
    return result;
  }
  void deallocate(uint8_t* p, size_t size) override {}

 private:
  alignas(16) uint8_t storage_[16384];
  size_t used_ = 0;
};

StaticAllocator flatbuffer_storage;

const tflite::Tensor* FlatbufferTensor(
    TensorType type, const std::initializer_list<int32_t>& shape) {
  flatbuffer_storage.Reset();
  flatbuffers::FlatBufferBuilder builder(1024, &flatbuffer_storage);
  builder.Finish(tflite::CreateTensor(
      builder, builder.CreateVector(shape.begin(), shape.size()), type, 0,
      builder.CreateString("t")));
  return flatbuffers::GetRoot<tflite::Tensor>(builder.GetBufferPointer());
}

const tflite::Model* FinishModel(
    flatbuffers::FlatBufferBuilder& builder,
    const Offset<tflite::OperatorCode>& code,
    const std::initializer_list<Offset<tflite::Tensor>>& tensors,
    const std::initializer_list<int32_t>& op_inputs,
    const std::initializer_list<int32_t>& op_outputs,
    const std::initializer_list<int32_t>& graph_inputs,
    const std::initializer_list<int32_t>& graph_outputs,
    const std::initializer_list<Offset<tflite::Buffer>>& buffers) {
  const Offset<tflite::Operator> op = tflite::CreateOperator(
      builder, 0, builder.CreateVector(op_inputs.begin(), op_inputs.size()),
      builder.CreateVector(op_outputs.begin(), op_outputs.size()));
  const Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors.begin(), tensors.size()),
      builder.CreateVector(graph_inputs.begin(), graph_inputs.size()),
      builder.CreateVector(graph_outputs.begin(), graph_outputs.size()),
      builder.CreateVector(&op, 1), builder.CreateString("main"));
  tflite::FinishModelBuffer(
      builder,
      tflite::CreateModel(
          builder, 3, builder.CreateVector(&code, 1),
          builder.CreateVector(&subgraph, 1), builder.CreateString("m"),
          builder.CreateVector(buffers.begin(), buffers.size())));
  return tflite::GetModel(builder.GetBufferPointer());
}

// The custom no_op with outputs [3, 2] int8 and `shape` of `type`.
const tflite::Model* NoOpModel(TensorType type,
                               const std::initializer_list<int32_t>& shape,
                               bool is_variable = false) {
  flatbuffer_storage.Reset();
  flatbuffers::FlatBufferBuilder builder(1024, &flatbuffer_storage);
  const int32_t small_shape[] = {3, 2};
  return FinishModel(
      builder,
      tflite::CreateOperatorCodeDirect(builder, 0, "no_op", 0,
                                       tflite::BuiltinOperator_CUSTOM),
      {tflite::CreateTensor(builder, builder.CreateVector(small_shape, 2),
                            TensorType_INT8, 0, builder.CreateString("a")),
       tflite::CreateTensor(
           builder, builder.CreateVector(shape.begin(), shape.size()), type, 0,
           builder.CreateString("b"), 0, is_variable)},
      {}, {0, 1}, {}, {0, 1}, {tflite::CreateBuffer(builder)});
}

// MEAN over axis 1 of a [rows, columns] int8 input.
const tflite::Model* MeanModel(int32_t rows, int32_t columns) {
  flatbuffer_storage.Reset();
  flatbuffers::FlatBufferBuilder builder(1024, &flatbuffer_storage);
  const int32_t axis[] = {1};
  const int32_t input_shape[] = {rows, columns};
  const int32_t axis_shape[] = {1};
  const int32_t output_shape[] = {rows};
  return FinishModel(
      builder,
      tflite::CreateOperatorCode(builder, 0, 0, 1,
                                 tflite::BuiltinOperator_MEAN),
      {tflite::CreateTensor(builder, builder.CreateVector(input_shape, 2),
                            TensorType_INT8, 0, builder.CreateString("in")),
       tflite::CreateTensor(builder, builder.CreateVector(axis_shape, 1),
                            TensorType_INT32, 1, builder.CreateString("axis")),
       tflite::CreateTensor(builder, builder.CreateVector(output_shape, 1),
                            TensorType_INT8, 0, builder.CreateString("out"))},
      {0, 1}, {2}, {0}, {2},
      {tflite::CreateBuffer(builder),
       tflite::CreateBuffer(
           builder, builder.CreateVector(reinterpret_cast<const uint8_t*>(axis),
                                         sizeof(axis)))});
}

TfLiteStatus FlatbufferBytes(TensorType type,
                             const std::initializer_list<int32_t>& shape,
                             size_t* bytes) {
  size_t type_size = 0;
  return tflite::BytesRequiredForTensor(*FlatbufferTensor(type, shape), bytes,
                                        &type_size);
}

TfLiteStatus EvalBytes(TfLiteType type, int* dims_data, size_t* bytes) {
  TfLiteEvalTensor tensor = {};
  tensor.type = type;
  tensor.dims = tflite::testing::IntArrayFromInts(dims_data);
  return tflite::TfLiteEvalTensorByteLength(&tensor, bytes);
}

constexpr size_t kArenaSize = 8192;
alignas(16) uint8_t arena[kArenaSize];

TfLiteStatus AllocateNoOp(TensorType type,
                          const std::initializer_list<int32_t>& shape,
                          bool is_variable = false) {
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(tflite::testing::GetTestingOpResolver(op_resolver));
  tflite::MicroInterpreter interpreter(NoOpModel(type, shape, is_variable),
                                       op_resolver, arena, kArenaSize);
  return interpreter.AllocateTensors();
}

TfLiteStatus AllocateMean(int32_t rows, int32_t columns) {
  tflite::MicroMutableOpResolver<1> op_resolver;
  TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
  tflite::MicroInterpreter interpreter(MeanModel(rows, columns), op_resolver,
                                       arena, kArenaSize);
  return interpreter.AllocateTensors();
}

TfLiteEvalTensor copy_tensors[2];

TfLiteEvalTensor* GetCopyTensor(const TfLiteContext* context, int index) {
  return &copy_tensors[index];
}

void IgnoreError(TfLiteContext* context, const char* format, ...) {}

// Copies input 0 to output 0 of a node whose two tensors share `dims`.
TfLiteStatus CopyWithDims(TfLiteType type, int* dims_data, int8_t* input,
                          int8_t* output) {
  copy_tensors[0] = {};
  copy_tensors[0].data.int8 = input;
  copy_tensors[0].dims = tflite::testing::IntArrayFromInts(dims_data);
  copy_tensors[0].type = type;
  copy_tensors[1] = copy_tensors[0];
  copy_tensors[1].data.int8 = output;

  TfLiteContext context = {};
  context.GetEvalTensor = GetCopyTensor;
  context.ReportError = IgnoreError;
  int inputs_data[] = {1, 0};
  int outputs_data[] = {1, 1};
  TfLiteNode node = {};
  node.inputs = tflite::testing::IntArrayFromInts(inputs_data);
  node.outputs = tflite::testing::IntArrayFromInts(outputs_data);
  return tflite::micro::CopyOpInputsToOpOutputs(&context, &node);
}

}  // namespace

TEST(TensorByteLengthTest, FlatbufferElementCountOverflowFails) {
  size_t bytes = 0;
  EXPECT_EQ(kTfLiteError,
            FlatbufferBytes(TensorType_INT8, {65536, 65536}, &bytes));
  EXPECT_EQ(kTfLiteError,
            FlatbufferBytes(TensorType_INT8, {16, kWrapsToSixteen}, &bytes));
  EXPECT_EQ(kTfLiteError,
            FlatbufferBytes(TensorType_INT8, {1, 65536, 65536, 1}, &bytes));
}

TEST(TensorByteLengthTest, FlatbufferNegativeDimensionFails) {
  size_t bytes = 0;
  EXPECT_EQ(kTfLiteError, FlatbufferBytes(TensorType_INT8, {3, 2, -1}, &bytes));
  EXPECT_EQ(kTfLiteError, FlatbufferBytes(TensorType_INT8, {-1, -1}, &bytes));
  // A zero before the negative dimension must not hide it.
  EXPECT_EQ(kTfLiteError, FlatbufferBytes(TensorType_INT8, {0, -1}, &bytes));
}

TEST(TensorByteLengthTest, FlatbufferByteCountAboveLimitFails) {
  size_t bytes = 0;
  EXPECT_EQ(kTfLiteError, FlatbufferBytes(TensorType_INT32, {1 << 30}, &bytes));
  EXPECT_EQ(kTfLiteError, FlatbufferBytes(TensorType_INT16, {INT_MAX}, &bytes));
  EXPECT_EQ(kTfLiteError,
            FlatbufferBytes(TensorType_FLOAT32, {INT_MAX}, &bytes));
  // At most INT_MAX bytes, but above INT_MAX once rounded up for the planner.
  EXPECT_EQ(kTfLiteError,
            FlatbufferBytes(TensorType_INT8, {kMaxInt8Elements + 1}, &bytes));
  // The element count alone is within the byte limit; the element size is not.
  EXPECT_EQ(kTfLiteError,
            FlatbufferBytes(TensorType_INT16, {kMaxInt16Elements + 1}, &bytes));
}

TEST(TensorByteLengthTest, FlatbufferLargestByteCountSucceeds) {
  size_t bytes = 0;
  EXPECT_EQ(kTfLiteOk,
            FlatbufferBytes(TensorType_INT8, {kMaxInt8Elements}, &bytes));
  EXPECT_EQ(tflite::kMaxTensorBytes, bytes);
  EXPECT_EQ(kTfLiteOk,
            FlatbufferBytes(TensorType_INT16, {kMaxInt16Elements}, &bytes));
  EXPECT_EQ(tflite::kMaxTensorBytes, bytes);
  EXPECT_EQ(kTfLiteOk,
            FlatbufferBytes(TensorType_INT8, {0, 65536, 65536}, &bytes));
  EXPECT_EQ(static_cast<size_t>(0), bytes);
  EXPECT_EQ(kTfLiteOk, FlatbufferBytes(TensorType_FLOAT32, {3, 2}, &bytes));
  EXPECT_EQ(static_cast<size_t>(24), bytes);
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

TEST(TensorByteLengthTest, EvalTensorByteCountAboveLimitFails) {
  size_t bytes = 0;
  int int32_dims[] = {2, 1, 1 << 30};
  EXPECT_EQ(kTfLiteError, EvalBytes(kTfLiteInt32, int32_dims, &bytes));
  int largest_dims[] = {1, INT_MAX};
  EXPECT_EQ(kTfLiteError, EvalBytes(kTfLiteInt16, largest_dims, &bytes));
  EXPECT_EQ(kTfLiteError, EvalBytes(kTfLiteFloat32, largest_dims, &bytes));
  int rounded_dims[] = {1, kMaxInt8Elements + 1};
  EXPECT_EQ(kTfLiteError, EvalBytes(kTfLiteInt8, rounded_dims, &bytes));
  int int16_dims[] = {1, kMaxInt16Elements + 1};
  EXPECT_EQ(kTfLiteError, EvalBytes(kTfLiteInt16, int16_dims, &bytes));
}

TEST(TensorByteLengthTest, EvalTensorLargestByteCountSucceeds) {
  size_t bytes = 0;
  int int8_dims[] = {1, kMaxInt8Elements};
  EXPECT_EQ(kTfLiteOk, EvalBytes(kTfLiteInt8, int8_dims, &bytes));
  EXPECT_EQ(tflite::kMaxTensorBytes, bytes);
  int int16_dims[] = {1, kMaxInt16Elements};
  EXPECT_EQ(kTfLiteOk, EvalBytes(kTfLiteInt16, int16_dims, &bytes));
  EXPECT_EQ(tflite::kMaxTensorBytes, bytes);
  int scalar[] = {0};
  EXPECT_EQ(kTfLiteOk, EvalBytes(kTfLiteFloat32, scalar, &bytes));
  EXPECT_EQ(sizeof(float), bytes);
}

// The limit, exercised at 32 bits so it runs on 64-bit hosts.
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
  EXPECT_EQ(kTfLiteError, AllocateNoOp(TensorType_INT8, {16, kWrapsToSixteen}));
  EXPECT_EQ(kTfLiteError, AllocateNoOp(TensorType_INT8, {65536, 65536}));
  EXPECT_EQ(kTfLiteError, AllocateNoOp(TensorType_INT16, {INT_MAX}));
  EXPECT_EQ(kTfLiteError, AllocateNoOp(TensorType_FLOAT32, {INT_MAX}));
  // Element counts within the limit whose byte counts do not fit the planner's
  // int: 2^32 bytes, and 2^31 - 14 bytes, which round up to 2^31.
  EXPECT_EQ(kTfLiteError, AllocateNoOp(TensorType_FLOAT32, {1 << 30}));
  EXPECT_EQ(kTfLiteError,
            AllocateNoOp(TensorType_INT16, {kMaxInt16Elements + 1}));
  EXPECT_EQ(kTfLiteError,
            AllocateNoOp(TensorType_INT8, {kMaxInt8Elements + 1}));
  EXPECT_EQ(kTfLiteError,
            AllocateNoOp(TensorType_INT16, {INT_MAX}, /*is_variable=*/true));
  // The same model with a representable shape still allocates.
  EXPECT_EQ(kTfLiteOk, AllocateNoOp(TensorType_INT8, {16, 2}));
  EXPECT_EQ(kTfLiteOk,
            AllocateNoOp(TensorType_INT16, {16, 2}, /*is_variable=*/true));
}

// MEAN's Prepare uses its input tensor without a null check, so the size check
// has to fail AllocateTensors before Prepare runs.
TEST(TensorByteLengthTest, AllocateTensorsFailsBeforeKernelPrepare) {
  EXPECT_EQ(kTfLiteError, AllocateMean(65536, 65536));
  EXPECT_EQ(kTfLiteOk, AllocateMean(16, 8));
}

TEST(TensorByteLengthTest, CopyOpInputsToOpOutputsFailsOnSizeOverflow) {
  int8_t input_data[16] = {1};
  int8_t output_data[16] = {};
  int wraps[] = {2, 16, kWrapsToSixteen};
  EXPECT_EQ(kTfLiteError,
            CopyWithDims(kTfLiteInt8, wraps, input_data, output_data));
  // 2^32 bytes, which the callers' int byte count would truncate to 0.
  int large[] = {1, 1 << 30};
  EXPECT_EQ(kTfLiteError,
            CopyWithDims(kTfLiteFloat32, large, input_data, output_data));
  EXPECT_EQ(0, output_data[0]);
  int fits[] = {1, 16};
  EXPECT_EQ(kTfLiteOk,
            CopyWithDims(kTfLiteInt8, fits, input_data, output_data));
  EXPECT_EQ(1, output_data[0]);
}

TF_LITE_MICRO_TESTS_MAIN
