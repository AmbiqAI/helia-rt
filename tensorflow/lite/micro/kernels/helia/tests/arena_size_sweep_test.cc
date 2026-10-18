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

// AllocateTensors at every arena size from the smallest one MicroInterpreter
// can be constructed in up to the smallest one the model runs in: each must
// return kTfLiteError without writing outside the arena, and that size and
// the sizes just above it must succeed. The keyword model's arena is mostly
// persistent data, the int8 conv model's mostly activations, and the conv
// model's CONV_2D requests per-channel quantization arrays.
// see AmbiqAI/helia-rt#449

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/models/keyword_scrambled_model_data.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"
#include "tensorflow/lite/micro/testing/test_conv_model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr size_t kArenaSize = 32 * 1024;
// Bytes past the arena end that must keep their fill. A request whose size
// wraps comes back at the arena end, and its writes start there.
constexpr size_t kGuardSize = 256;
// Also fills the arena, so a pointer read from memory the interpreter never
// set is not null.
constexpr uint8_t kFill = 0xA5;
alignas(16) uint8_t g_arena[kArenaSize + kGuardSize];

// Below the 16-byte persistent request alignment, so every request is the
// failing one at some size, and not a multiple of 8, so the arena end takes
// every 4-byte alignment.
constexpr size_t kStep = 12;

// Sizes above the smallest working arena that must also work.
constexpr size_t kMonotonicSpan = 256;

// MicroInterpreter's constructor places these three in the arena tail without
// checking that they fit, so no smaller arena is swept.
constexpr size_t kConstructorFloor =
    sizeof(tflite::SingleArenaBufferAllocator) +
    alignof(tflite::SingleArenaBufferAllocator) +
    sizeof(tflite::GreedyMemoryPlanner) + alignof(tflite::GreedyMemoryPlanner) +
    sizeof(tflite::MicroAllocator) + alignof(tflite::MicroAllocator);

bool GuardIntact(size_t arena_size) {
  for (size_t i = 0; i < kGuardSize; ++i) {
    if (g_arena[arena_size + i] != kFill) {
      return false;
    }
  }
  return true;
}

template <typename Resolver>
TfLiteStatus AllocateInArena(const tflite::Model* model,
                             const Resolver& resolver, size_t arena_size,
                             bool* guard_intact) {
  std::memset(g_arena, kFill, arena_size + kGuardSize);
  tflite::MicroInterpreter interpreter(model, resolver, g_arena, arena_size);
  const TfLiteStatus status = interpreter.AllocateTensors();
  *guard_intact = GuardIntact(arena_size);
  return status;
}

template <typename Resolver>
void SweepArenaSizes(const char* label, const unsigned char* model_data,
                     const Resolver& resolver) {
  const tflite::Model* model = tflite::GetModel(model_data);
  ASSERT_NE(model, nullptr);

  size_t used_bytes = 0;
  {
    tflite::MicroInterpreter interpreter(model, resolver, g_arena, kArenaSize);
    ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
    used_bytes = interpreter.arena_used_bytes();
  }
  ASSERT_GT(used_bytes, kConstructorFloor);

  size_t required = 0;
  int undersized = 0;
  for (size_t size = kConstructorFloor; size <= kArenaSize; size += kStep) {
    bool guard_intact = false;
    const TfLiteStatus status =
        AllocateInArena(model, resolver, size, &guard_intact);
    if (!guard_intact) {
      MicroPrintf("%s: %d-byte arena wrote past its end", label,
                  static_cast<int>(size));
    }
    ASSERT_TRUE(guard_intact);
    if (status == kTfLiteOk) {
      required = size;
      break;
    }
    ASSERT_EQ(status, kTfLiteError);
    ++undersized;
  }
  MicroPrintf(
      "%s: arena_used_bytes %d, smallest working arena %d, %d "
      "undersized arenas from %d bytes failed cleanly",
      label, static_cast<int>(used_bytes), static_cast<int>(required),
      undersized, static_cast<int>(kConstructorFloor));
  ASSERT_GT(required, 0u);
  ASSERT_LE(required + kMonotonicSpan, kArenaSize);

  for (size_t size = required + kStep; size <= required + kMonotonicSpan;
       size += kStep) {
    bool guard_intact = false;
    EXPECT_EQ(AllocateInArena(model, resolver, size, &guard_intact), kTfLiteOk);
    EXPECT_TRUE(guard_intact);
  }

  std::memset(g_arena, kFill, required + kGuardSize);
  tflite::MicroInterpreter interpreter(model, resolver, g_arena, required);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  for (size_t i = 0; i < interpreter.inputs_size(); ++i) {
    TfLiteTensor* input = interpreter.input(i);
    std::memset(input->data.raw, 0, input->bytes);
  }
  EXPECT_EQ(interpreter.Invoke(), kTfLiteOk);
  EXPECT_TRUE(GuardIntact(required));
}

}  // namespace

TEST(HeliaArenaSizeSweepTest, KeywordModelPersistentDominated) {
  tflite::MicroMutableOpResolver<4> resolver;
  ASSERT_EQ(resolver.AddFullyConnected(), kTfLiteOk);
  ASSERT_EQ(resolver.AddQuantize(), kTfLiteOk);
  ASSERT_EQ(resolver.AddSoftmax(), kTfLiteOk);
  ASSERT_EQ(resolver.AddSvdf(), kTfLiteOk);
  SweepArenaSizes("keyword_scrambled", g_keyword_scrambled_model_data,
                  resolver);
}

TEST(HeliaArenaSizeSweepTest, Int8ConvModelActivationDominated) {
  tflite::MicroMutableOpResolver<6> resolver;
  ASSERT_EQ(resolver.AddConv2D(), kTfLiteOk);
  ASSERT_EQ(resolver.AddQuantize(), kTfLiteOk);
  ASSERT_EQ(resolver.AddMaxPool2D(), kTfLiteOk);
  ASSERT_EQ(resolver.AddReshape(), kTfLiteOk);
  ASSERT_EQ(resolver.AddFullyConnected(), kTfLiteOk);
  ASSERT_EQ(resolver.AddDequantize(), kTfLiteOk);
  SweepArenaSizes("test_conv_model", kTestConvModelData, resolver);
}

TF_LITE_MICRO_TESTS_MAIN
