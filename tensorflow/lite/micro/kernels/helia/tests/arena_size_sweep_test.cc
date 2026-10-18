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

// AllocateTensors at every arena size, in 12-byte steps, from the smallest
// one MicroInterpreter can be constructed in up to the smallest one the model
// runs in: each must return kTfLiteError without writing past the arena end,
// and that size and the sizes just above it must allocate and invoke. The
// keyword model's arena is mostly persistent data and the int8 conv model's
// mostly activations. Sweeping every model this way is too slow for the FVP;
// the arena_exhaustion_injection tests fail each of every model's allocations
// instead. see AmbiqAI/helia-rt#449

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/micro/kernels/helia/tests/arena_sweep_models.h"

namespace {

// A flatbuffer allocator over a static buffer, so the test needs no heap.
class StaticAllocator : public flatbuffers::Allocator {
 public:
  uint8_t* allocate(size_t size) override {
    if (size > sizeof(buffer_) - used_) {
      return nullptr;
    }
    uint8_t* result = buffer_ + used_;
    used_ += size;
    return result;
  }
  void deallocate(uint8_t*, size_t) override {}

 private:
  alignas(16) uint8_t buffer_[2048];
  size_t used_ = 0;
};

// One subgraph with no operators and one variable int8 tensor of `elements`.
const tflite::Model* BuildVariableTensorModel(
    flatbuffers::FlatBufferBuilder& builder, int32_t elements) {
  using flatbuffers::Offset;
  const Offset<tflite::Buffer> buffers[] = {tflite::CreateBuffer(builder)};
  const int32_t shape[] = {elements};
  const Offset<tflite::Tensor> tensors[] = {tflite::CreateTensor(
      builder, builder.CreateVector(shape, 1), tflite::TensorType_INT8, 0,
      builder.CreateString("state"), 0, /*is_variable=*/true)};
  const int32_t io[] = {0};
  const Offset<tflite::Operator>* no_operators = nullptr;
  const Offset<tflite::OperatorCode>* no_operator_codes = nullptr;
  const Offset<tflite::SubGraph> subgraphs[] = {tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors, 1), builder.CreateVector(io, 1),
      builder.CreateVector(io, 1), builder.CreateVector(no_operators, 0))};
  builder.Finish(tflite::CreateModel(builder, TFLITE_SCHEMA_VERSION,
                                     builder.CreateVector(no_operator_codes, 0),
                                     builder.CreateVector(subgraphs, 1),
                                     builder.CreateString(""),
                                     builder.CreateVector(buffers, 1)));
  return tflite::GetModel(builder.GetBufferPointer());
}

}  // namespace

TEST(HeliaArenaSizeSweepTest, KeywordModelPersistentDominated) {
  SweepArenaSizes("keyword_scrambled", g_keyword_scrambled_model_data);
}

TEST(HeliaArenaSizeSweepTest, HelloWorldModels) {
  SweepArenaSizes("hello_world_float", g_hello_world_float_model_data);
  SweepArenaSizes("hello_world_int8", g_hello_world_int8_model_data);
}

TEST(HeliaArenaSizeSweepTest, AddModels) {
  SweepArenaSizes("add_four_numbers", g_add_four_numbers_model_data);
}

TEST(HeliaArenaSizeSweepTest, Int8ConvModelActivationDominated) {
  SweepArenaSizes("test_conv_model", kTestConvModelData);
}

TEST(HeliaArenaSizeSweepTest, ResourceVariablesModel) {
  SweepArenaSizes("resource_variables", g_resource_variables_model_data,
                  /*resource_variables=*/true);
}

// On a 32-bit target 2^30 bytes is larger than the arena's tail address, so
// computing tail - size wrapped and returned a pointer outside the arena.
TEST(HeliaArenaSizeSweepTest, VariableTensorLargerThanTailAddress) {
  StaticAllocator flatbuffer_allocator;
  flatbuffers::FlatBufferBuilder builder(1024, &flatbuffer_allocator);
  const tflite::Model* model = BuildVariableTensorModel(builder, 1 << 30);
  Resolver resolver;
  constexpr size_t kSize = 16 * 1024;
  std::memset(g_arena, kFill, kSize + kGuardSize);
  tflite::MicroInterpreter interpreter(model, resolver, g_arena, kSize);
  EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteError);
  EXPECT_TRUE(GuardIntact(kSize));
}

TF_LITE_MICRO_TESTS_MAIN
