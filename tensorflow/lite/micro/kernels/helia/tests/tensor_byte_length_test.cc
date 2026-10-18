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
// TfLiteEvalTensorByteLength must fail on an element or byte count above
// kMaxTensorBytes, the largest the memory planner can hold once rounded up to
// the arena alignment, and on a negative dimension other than a -1 a kernel
// resolves in Prepare. The failure must stop AllocateTensors before any kernel
// Prepare and must reach CopyOpInputsToOpOutputs. A resolved -1 must leave the
// tensor sized by its resolved shape. The persistent and non-persistent arena
// allocators must refuse a request that does not fit, and the memory planners a
// plan whose offset + size passes INT_MAX, rather than wrap. see
// AmbiqAI/helia-rt#453

#include <climits>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/memory_planner/linear_memory_planner.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
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

// The custom no_op with outputs `first_shape` int8 and `shape` of `type`.
const tflite::Model* NoOpModel(
    TensorType type, const std::initializer_list<int32_t>& shape,
    bool is_variable = false,
    const std::initializer_list<int32_t>& first_shape = {3, 2}) {
  flatbuffer_storage.Reset();
  flatbuffers::FlatBufferBuilder builder(1024, &flatbuffer_storage);
  return FinishModel(
      builder,
      tflite::CreateOperatorCodeDirect(builder, 0, "no_op", 0,
                                       tflite::BuiltinOperator_CUSTOM),
      {tflite::CreateTensor(
           builder,
           builder.CreateVector(first_shape.begin(), first_shape.size()),
           TensorType_INT8, 0, builder.CreateString("a")),
       tflite::CreateTensor(
           builder, builder.CreateVector(shape.begin(), shape.size()), type, 0,
           builder.CreateString("b"), 0, is_variable)},
      {}, {0, 1}, {}, {0, 1}, {tflite::CreateBuffer(builder)});
}

// The custom big_scratch op with one [3, 2] int8 output.
const tflite::Model* ScratchModel() {
  flatbuffer_storage.Reset();
  flatbuffers::FlatBufferBuilder builder(1024, &flatbuffer_storage);
  const int32_t shape[] = {3, 2};
  return FinishModel(
      builder,
      tflite::CreateOperatorCodeDirect(builder, 0, "big_scratch", 0,
                                       tflite::BuiltinOperator_CUSTOM),
      {tflite::CreateTensor(builder, builder.CreateVector(shape, 2),
                            TensorType_INT8, 0, builder.CreateString("a"))},
      {}, {0}, {}, {0}, {tflite::CreateBuffer(builder)});
}

// RESHAPE of an [8] float32 input to `output_shape`.
const tflite::Model* ReshapeStretchModel(
    const std::initializer_list<int32_t>& output_shape) {
  flatbuffer_storage.Reset();
  flatbuffers::FlatBufferBuilder builder(1024, &flatbuffer_storage);
  const int32_t input_shape[] = {8};
  return FinishModel(
      builder,
      tflite::CreateOperatorCode(builder, 0, 0, 1,
                                 tflite::BuiltinOperator_RESHAPE),
      {tflite::CreateTensor(builder, builder.CreateVector(input_shape, 1),
                            TensorType_FLOAT32, 0, builder.CreateString("in")),
       tflite::CreateTensor(
           builder,
           builder.CreateVector(output_shape.begin(), output_shape.size()),
           TensorType_FLOAT32, 0, builder.CreateString("out"))},
      {0}, {1}, {0}, {1}, {tflite::CreateBuffer(builder)});
}

// EMBEDDING_LOOKUP of [3] int32 ids into [5, 2] float32 values, to a [-1, 2]
// output.
const tflite::Model* EmbeddingLookupStretchModel() {
  flatbuffer_storage.Reset();
  flatbuffers::FlatBufferBuilder builder(1024, &flatbuffer_storage);
  const int32_t ids_shape[] = {3};
  const int32_t value_shape[] = {5, 2};
  const int32_t output_shape[] = {-1, 2};
  return FinishModel(
      builder,
      tflite::CreateOperatorCode(builder, 0, 0, 1,
                                 tflite::BuiltinOperator_EMBEDDING_LOOKUP),
      {tflite::CreateTensor(builder, builder.CreateVector(ids_shape, 1),
                            TensorType_INT32, 0, builder.CreateString("ids")),
       tflite::CreateTensor(builder, builder.CreateVector(value_shape, 2),
                            TensorType_FLOAT32, 0, builder.CreateString("v")),
       tflite::CreateTensor(builder, builder.CreateVector(output_shape, 2),
                            TensorType_FLOAT32, 0,
                            builder.CreateString("out"))},
      {0, 1}, {2}, {0, 1}, {2}, {tflite::CreateBuffer(builder)});
}

// READ_VARIABLE into the [2, 1, 4] output, then RESHAPE of the [8] input to
// [2, 1, -1] and ASSIGN_VARIABLE of it, so each Invoke reads what the
// previous one assigned.
const tflite::Model* ReshapeIntoVariableModel() {
  using tflite::BuiltinOperator;
  flatbuffer_storage.Reset();
  flatbuffers::FlatBufferBuilder builder(2048, &flatbuffer_storage);
  const BuiltinOperator op_codes[] = {
      tflite::BuiltinOperator_VAR_HANDLE, tflite::BuiltinOperator_READ_VARIABLE,
      tflite::BuiltinOperator_RESHAPE, tflite::BuiltinOperator_ASSIGN_VARIABLE};
  Offset<tflite::OperatorCode> codes[4];
  for (int i = 0; i < 4; ++i) {
    codes[i] = tflite::CreateOperatorCode(builder, 0, 0, 1, op_codes[i]);
  }
  const int32_t input_shape[] = {8};
  const int32_t stretch_shape[] = {2, 1, -1};
  const int32_t output_shape[] = {2, 1, 4};
  const Offset<tflite::Tensor> tensors[] = {
      tflite::CreateTensor(builder, builder.CreateVector(input_shape, 1),
                           TensorType_FLOAT32, 0, builder.CreateString("in")),
      tflite::CreateTensor(builder, builder.CreateVector<int32_t>({}),
                           tflite::TensorType_RESOURCE, 0,
                           builder.CreateString("handle")),
      tflite::CreateTensor(builder, builder.CreateVector(stretch_shape, 3),
                           TensorType_FLOAT32, 0,
                           builder.CreateString("reshaped")),
      tflite::CreateTensor(builder, builder.CreateVector(output_shape, 3),
                           TensorType_FLOAT32, 0, builder.CreateString("out")),
  };
  const Offset<void> var_handle_options =
      tflite::CreateVarHandleOptions(builder, builder.CreateString("c"),
                                     builder.CreateString("v"))
          .Union();
  const Offset<tflite::Operator> ops[] = {
      tflite::CreateOperator(builder, 0, builder.CreateVector<int32_t>({}),
                             builder.CreateVector<int32_t>({1}),
                             tflite::BuiltinOptions_VarHandleOptions,
                             var_handle_options),
      tflite::CreateOperator(builder, 1, builder.CreateVector<int32_t>({1}),
                             builder.CreateVector<int32_t>({3})),
      tflite::CreateOperator(builder, 2, builder.CreateVector<int32_t>({0}),
                             builder.CreateVector<int32_t>({2})),
      tflite::CreateOperator(builder, 3, builder.CreateVector<int32_t>({1, 2}),
                             builder.CreateVector<int32_t>({})),
  };
  const Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors, 4),
      builder.CreateVector<int32_t>({0}), builder.CreateVector<int32_t>({3}),
      builder.CreateVector(ops, 4), builder.CreateString("main"));
  const Offset<tflite::Buffer> buffers[] = {tflite::CreateBuffer(builder)};
  tflite::FinishModelBuffer(
      builder, tflite::CreateModel(builder, 3, builder.CreateVector(codes, 4),
                                   builder.CreateVector(&subgraph, 1),
                                   builder.CreateString("m"),
                                   builder.CreateVector(buffers, 1)));
  return tflite::GetModel(builder.GetBufferPointer());
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
                          bool is_variable = false,
                          const std::initializer_list<int32_t>& first_shape = {
                              3, 2}) {
  tflite::testing::TestingOpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(tflite::testing::GetTestingOpResolver(op_resolver));
  tflite::MicroInterpreter interpreter(
      NoOpModel(type, shape, is_variable, first_shape), op_resolver, arena,
      kArenaSize);
  return interpreter.AllocateTensors();
}

size_t scratch_request_bytes = 0;

TfLiteStatus BigScratchPrepare(TfLiteContext* context, TfLiteNode* node) {
  int index = -1;
  return context->RequestScratchBufferInArena(context, scratch_request_bytes,
                                              &index);
}

TfLiteStatus NoInvoke(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus AllocateWithScratch(size_t bytes) {
  scratch_request_bytes = bytes;
  static TFLMRegistration registration =
      tflite::micro::RegisterOp(nullptr, BigScratchPrepare, NoInvoke);
  tflite::MicroMutableOpResolver<1> op_resolver;
  TF_LITE_ENSURE_STATUS(op_resolver.AddCustom("big_scratch", &registration));
  tflite::MicroInterpreter interpreter(ScratchModel(), op_resolver, arena,
                                       kArenaSize);
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
  EXPECT_EQ(kTfLiteError, FlatbufferBytes(TensorType_INT8, {3, 2, -2}, &bytes));
  EXPECT_EQ(kTfLiteError,
            FlatbufferBytes(TensorType_INT8, {INT_MIN, 1}, &bytes));
  // A zero before the negative dimension must not hide it.
  EXPECT_EQ(kTfLiteError, FlatbufferBytes(TensorType_INT8, {0, -2}, &bytes));
}

// A -1 dimension is left for a kernel to resolve in Prepare, but the rest of
// the shape is still checked. The byte count stays the wrapped product
// upstream reports, which EMBEDDING_LOOKUP's Prepare compares against.
TEST(TensorByteLengthTest, FlatbufferStretchDimensionIsUnresolved) {
  size_t bytes = 1;
  EXPECT_EQ(kTfLiteOk, FlatbufferBytes(TensorType_FLOAT32, {2, 1, -1}, &bytes));
  EXPECT_EQ(size_t{0} - 8, bytes);
  EXPECT_EQ(kTfLiteError,
            FlatbufferBytes(TensorType_INT8, {65536, 65536, -1}, &bytes));
  EXPECT_EQ(kTfLiteError,
            FlatbufferBytes(TensorType_INT8, {-1, 3, -2}, &bytes));
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

// The overflowing input must fail AllocateTensors before MEAN's Prepare runs.
TEST(TensorByteLengthTest, AllocateTensorsFailsBeforeKernelPrepare) {
  EXPECT_EQ(kTfLiteError, AllocateMean(65536, 65536));
  EXPECT_EQ(kTfLiteOk, AllocateMean(16, 8));
}

// RESHAPE resolves its output's -1 dimension in Prepare, and the planner then
// sizes the resolved shape.
TEST(TensorByteLengthTest, AllocateTensorsResolvesReshapeStretchDimension) {
  tflite::MicroMutableOpResolver<1> op_resolver;
  ASSERT_EQ(kTfLiteOk, op_resolver.AddReshape());
  tflite::MicroInterpreter interpreter(ReshapeStretchModel({2, 1, -1}),
                                       op_resolver, arena, kArenaSize);
  ASSERT_EQ(kTfLiteOk, interpreter.AllocateTensors());
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  ASSERT_NE(nullptr, input);
  ASSERT_NE(nullptr, output);
  EXPECT_EQ(static_cast<size_t>(32), output->bytes);
  for (int i = 0; i < 8; ++i) {
    input->data.f[i] = static_cast<float>(i + 1);
  }
  ASSERT_EQ(kTfLiteOk, interpreter.Invoke());
  ASSERT_EQ(3, output->dims->size);
  EXPECT_EQ(2, output->dims->data[0]);
  EXPECT_EQ(1, output->dims->data[1]);
  EXPECT_EQ(4, output->dims->data[2]);
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(i + 1), output->data.f[i]);
  }
}

// The other output dimensions multiply to 0, so no stretch size exists.
TEST(TensorByteLengthTest, AllocateTensorsRejectsReshapeZeroStretch) {
  tflite::MicroMutableOpResolver<1> op_resolver;
  ASSERT_EQ(kTfLiteOk, op_resolver.AddReshape());
  tflite::MicroInterpreter interpreter(ReshapeStretchModel({0, -1}),
                                       op_resolver, arena, kArenaSize);
  EXPECT_EQ(kTfLiteError, interpreter.AllocateTensors());
}

// The variable takes its size from the resolved RESHAPE output, so the value
// assigned in one Invoke is read back in the next.
TEST(TensorByteLengthTest, ReshapeStretchOutputSizesResourceVariable) {
  tflite::MicroMutableOpResolver<4> op_resolver;
  ASSERT_EQ(kTfLiteOk, op_resolver.AddVarHandle());
  ASSERT_EQ(kTfLiteOk, op_resolver.AddReadVariable());
  ASSERT_EQ(kTfLiteOk, op_resolver.AddReshape());
  ASSERT_EQ(kTfLiteOk, op_resolver.AddAssignVariable());
  alignas(16) static uint8_t variable_arena[4096];
  tflite::MicroResourceVariables* variables =
      tflite::MicroResourceVariables::Create(
          tflite::MicroAllocator::Create(variable_arena,
                                         sizeof(variable_arena)),
          1);
  ASSERT_NE(nullptr, variables);
  tflite::MicroInterpreter interpreter(ReshapeIntoVariableModel(), op_resolver,
                                       arena, kArenaSize, variables);
  ASSERT_EQ(kTfLiteOk, interpreter.AllocateTensors());
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  for (int i = 0; i < 8; ++i) {
    input->data.f[i] = static_cast<float>(i + 1);
  }
  ASSERT_EQ(kTfLiteOk, interpreter.Invoke());
  for (int i = 0; i < 8; ++i) {
    input->data.f[i] = static_cast<float>(100 + i);
  }
  ASSERT_EQ(kTfLiteOk, interpreter.Invoke());
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(i + 1), output->data.f[i]);
  }
}

// EMBEDDING_LOOKUP sets its output rows in Prepare, after checking the
// flatbuffer byte count, and the output is then sized by the resolved shape.
TEST(TensorByteLengthTest, AllocateTensorsResolvesEmbeddingLookupOutput) {
  tflite::MicroMutableOpResolver<1> op_resolver;
  ASSERT_EQ(kTfLiteOk, op_resolver.AddEmbeddingLookup());
  tflite::MicroInterpreter interpreter(EmbeddingLookupStretchModel(),
                                       op_resolver, arena, kArenaSize);
  ASSERT_EQ(kTfLiteOk, interpreter.AllocateTensors());
  TfLiteTensor* output = interpreter.output(0);
  EXPECT_EQ(static_cast<size_t>(24), output->bytes);
  const int32_t ids[] = {4, 0, 2};
  for (int i = 0; i < 3; ++i) {
    interpreter.input(0)->data.i32[i] = ids[i];
  }
  for (int i = 0; i < 10; ++i) {
    interpreter.input(1)->data.f[i] = static_cast<float>(i);
  }
  ASSERT_EQ(kTfLiteOk, interpreter.Invoke());
  ASSERT_EQ(2, output->dims->size);
  EXPECT_EQ(3, output->dims->data[0]);
  EXPECT_EQ(2, output->dims->data[1]);
  const float expected[] = {8, 9, 0, 1, 4, 5};
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(expected[i], output->data.f[i]);
  }
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

TEST(TensorByteLengthTest, AllocateTensorsFailsOnOversizedScratchRequest) {
  EXPECT_EQ(kTfLiteError, AllocateWithScratch(tflite::kMaxTensorBytes + 1));
  // Rounding this up for the planner would wrap to a small size.
  EXPECT_EQ(kTfLiteError, AllocateWithScratch(SIZE_MAX - 7));
  EXPECT_EQ(kTfLiteOk, AllocateWithScratch(64));
}

TEST(TensorByteLengthTest, PersistentArenaAllocatorRefusesOversizedRequest) {
  alignas(16) static uint8_t buffer[1024];
  tflite::PersistentArenaBufferAllocator allocator(buffer, sizeof(buffer));
  EXPECT_EQ(nullptr, allocator.AllocatePersistentBuffer(SIZE_MAX - 15, 16));
  EXPECT_EQ(nullptr, allocator.AllocatePersistentBuffer(sizeof(buffer) + 1, 1));
  EXPECT_EQ(buffer, allocator.AllocatePersistentBuffer(sizeof(buffer), 16));
}

TEST(TensorByteLengthTest, NonPersistentArenaAllocatorStaysInsideArena) {
  alignas(16) static uint8_t buffer[64];
  {
    // The 16-byte-aligned temp start lies past the 20-byte arena's tail.
    tflite::NonPersistentArenaBufferAllocator allocator(buffer, 20);
    EXPECT_NE(nullptr, allocator.AllocateTemp(17, 1));
    EXPECT_EQ(nullptr, allocator.AllocateTemp(1, 16));
    EXPECT_EQ(static_cast<size_t>(0), allocator.GetAvailableMemory(16));
  }
  {
    // The 16-byte-aligned resizable buffer starts past the arena's tail.
    tflite::NonPersistentArenaBufferAllocator allocator(buffer + 1, 8);
    EXPECT_EQ(nullptr, allocator.AllocateResizableBuffer(4, 16));
    EXPECT_EQ(static_cast<size_t>(0), allocator.GetAvailableMemory(16));
    EXPECT_EQ(buffer + 1, allocator.AllocateResizableBuffer(4, 1));
  }
}

// Room for the 3000 buffers of the largest plan below.
alignas(16) uint8_t planner_scratch[3000 * 40];

TEST(TensorByteLengthTest, GreedyPlannerRefusesPlanAboveIntMax) {
  tflite::GreedyMemoryPlanner planner;
  int offset = 0;
  // Two live 1.2e9-byte buffers: each fits, the second one's end does not.
  ASSERT_EQ(kTfLiteOk, planner.Init(planner_scratch, sizeof(planner_scratch)));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(1200000000, 0, 1));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(1200000000, 0, 1));
  EXPECT_EQ(kTfLiteError, planner.GetOffsetForBuffer(1, &offset));
  EXPECT_EQ(SIZE_MAX, planner.GetMaximumMemorySize());
  EXPECT_EQ(kTfLiteError, planner.AddBuffer(-1, 0, 1));

  // A plan ending exactly at INT_MAX is fine; one more live byte is not.
  ASSERT_EQ(kTfLiteOk, planner.Init(planner_scratch, sizeof(planner_scratch)));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(INT_MAX - 32, 0, 1));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(32, 0, 1));
  EXPECT_EQ(static_cast<size_t>(INT_MAX), planner.GetMaximumMemorySize());
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(1, 0, 1));
  EXPECT_EQ(kTfLiteError, planner.GetOffsetForBuffer(0, &offset));

  // Offline buffers are checked by their own offset + size.
  ASSERT_EQ(kTfLiteOk, planner.Init(planner_scratch, sizeof(planner_scratch)));
  EXPECT_EQ(kTfLiteError, planner.AddBuffer(16, 0, 1, INT_MAX - 8));
  EXPECT_EQ(kTfLiteError, planner.AddBuffer(16, 0, 1, -2));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(32, 0, 1, INT_MAX - 32));
  EXPECT_EQ(static_cast<size_t>(INT_MAX), planner.GetMaximumMemorySize());
}

TEST(TensorByteLengthTest, GreedyPlannerPlansLargeValidModels) {
  ASSERT_GE(sizeof(planner_scratch),
            3000 * tflite::GreedyMemoryPlanner::per_buffer_size());
  tflite::GreedyMemoryPlanner planner;
  int offset = 0;
  // 300 offline 64 KiB buffers spread across a 16 MiB arena.
  ASSERT_EQ(kTfLiteOk, planner.Init(planner_scratch, sizeof(planner_scratch)));
  for (int i = 0; i < 300; ++i) {
    ASSERT_EQ(kTfLiteOk, planner.AddBuffer(65536, i, i, (i % 256) * 65536));
  }
  EXPECT_EQ(static_cast<size_t>(16 << 20), planner.GetMaximumMemorySize());
  EXPECT_EQ(kTfLiteOk, planner.GetOffsetForBuffer(299, &offset));
  EXPECT_EQ(43 * 65536, offset);

  // A chain of 600 4 MiB buffers, each live for two steps, peaking at 8 MiB.
  ASSERT_EQ(kTfLiteOk, planner.Init(planner_scratch, sizeof(planner_scratch)));
  for (int i = 0; i < 600; ++i) {
    ASSERT_EQ(kTfLiteOk, planner.AddBuffer(4 << 20, i, i + 1));
  }
  EXPECT_EQ(static_cast<size_t>(8 << 20), planner.GetMaximumMemorySize());
  EXPECT_EQ(kTfLiteOk, planner.GetOffsetForBuffer(599, &offset));

  // 3000 offline 1 KiB buffers packed into the first 3000 KiB.
  ASSERT_EQ(kTfLiteOk, planner.Init(planner_scratch, sizeof(planner_scratch)));
  for (int i = 0; i < 3000; ++i) {
    ASSERT_EQ(kTfLiteOk, planner.AddBuffer(1024, i, i, i * 1024));
  }
  EXPECT_EQ(static_cast<size_t>(3000 * 1024), planner.GetMaximumMemorySize());
  EXPECT_EQ(kTfLiteOk, planner.GetOffsetForBuffer(2999, &offset));
  EXPECT_EQ(2999 * 1024, offset);
}

TEST(TensorByteLengthTest, LinearPlannerRefusesPlanAboveIntMax) {
  tflite::LinearMemoryPlanner planner;
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(1200000000, 0, 1));
  EXPECT_EQ(kTfLiteError, planner.AddBuffer(1200000000, 0, 1));
  EXPECT_EQ(kTfLiteError, planner.AddBuffer(-1, 0, 1));
  EXPECT_EQ(kTfLiteOk, planner.AddBuffer(INT_MAX - 1200000000, 0, 1));
  EXPECT_EQ(static_cast<size_t>(INT_MAX), planner.GetMaximumMemorySize());
}

TF_LITE_MICRO_TESTS_MAIN
