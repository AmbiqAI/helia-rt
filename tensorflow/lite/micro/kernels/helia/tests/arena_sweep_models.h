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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_HELIA_TESTS_ARENA_SWEEP_MODELS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_HELIA_TESTS_ARENA_SWEEP_MODELS_H_

// The models and the arena-size and allocation-failure sweeps shared by the
// arena_size_sweep and arena_exhaustion_injection tests. Include from one
// translation unit per binary. see AmbiqAI/helia-rt#449

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/examples/dtln/dtln_noise_suppression_model_data.h"
#include "tensorflow/lite/micro/examples/hello_world/models/hello_world_float_model_data.h"
#include "tensorflow/lite/micro/examples/hello_world/models/hello_world_int8_model_data.h"
#include "tensorflow/lite/micro/examples/memory_footprint/models/simple_add_model_model_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/models/audio_preprocessor_float_model_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/models/audio_preprocessor_int8_model_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/models/micro_speech_quantized_model_data.h"
#include "tensorflow/lite/micro/examples/mnist_lstm/trained_lstm_int8_model_data.h"
#include "tensorflow/lite/micro/examples/mnist_lstm/trained_lstm_model_data.h"
#include "tensorflow/lite/micro/examples/recipes/add_four_numbers_model_data.h"
#include "tensorflow/lite/micro/examples/recipes/resource_variables_model_data.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/micro/models/keyword_scrambled_8bit_model_data.h"
#include "tensorflow/lite/micro/models/keyword_scrambled_model_data.h"
#include "tensorflow/lite/micro/models/person_detect_model_data.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"
#include "tensorflow/lite/micro/testing/test_conv_model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr size_t kArenaSize = 160 * 1024;
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

constexpr int kMaxResourceVariables = 4;

// Arena bytes beyond arena_used_bytes() the injection runs get; the smallest
// working arena can exceed arena_used_bytes() by the planner's temp memory.
constexpr size_t kInjectionHeadroom = 8 * 1024;

// The smallest arena at g_arena that holds the three objects
// MicroInterpreter's constructor places in the arena tail without checking
// that they fit, found the way the constructor allocates them; no smaller
// arena is swept.
inline size_t ConstructorFloor() {
  for (size_t size = 0;; ++size) {
    tflite::SingleArenaBufferAllocator tail(g_arena, size);
    if (tail.AllocatePersistentBuffer(
            sizeof(tflite::SingleArenaBufferAllocator),
            alignof(tflite::SingleArenaBufferAllocator)) != nullptr &&
        tail.AllocatePersistentBuffer(sizeof(tflite::GreedyMemoryPlanner),
                                      alignof(tflite::GreedyMemoryPlanner)) !=
            nullptr &&
        tail.AllocatePersistentBuffer(sizeof(tflite::MicroAllocator),
                                      alignof(tflite::MicroAllocator)) !=
            nullptr) {
      return size;
    }
  }
}

using Resolver = tflite::MicroMutableOpResolver<48>;

// Every operator the swept models use.
inline TfLiteStatus AddModelOps(Resolver& r) {
  TF_LITE_ENSURE_STATUS(r.AddAdd());
  TF_LITE_ENSURE_STATUS(r.AddAssignVariable());
  TF_LITE_ENSURE_STATUS(r.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(r.AddCallOnce());
  TF_LITE_ENSURE_STATUS(r.AddCast());
  TF_LITE_ENSURE_STATUS(r.AddConcatenation());
  TF_LITE_ENSURE_STATUS(r.AddConv2D());
  TF_LITE_ENSURE_STATUS(r.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(r.AddDequantize());
  TF_LITE_ENSURE_STATUS(r.AddDiv());
  TF_LITE_ENSURE_STATUS(r.AddEnergy());
  TF_LITE_ENSURE_STATUS(r.AddFftAutoScale());
  TF_LITE_ENSURE_STATUS(r.AddFilterBank());
  TF_LITE_ENSURE_STATUS(r.AddFilterBankLog());
  TF_LITE_ENSURE_STATUS(r.AddFilterBankSpectralSubtraction());
  TF_LITE_ENSURE_STATUS(r.AddFilterBankSquareRoot());
  TF_LITE_ENSURE_STATUS(r.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(r.AddIf());
  TF_LITE_ENSURE_STATUS(r.AddLess());
  TF_LITE_ENSURE_STATUS(r.AddLogistic());
  TF_LITE_ENSURE_STATUS(r.AddMaximum());
  TF_LITE_ENSURE_STATUS(r.AddMaxPool2D());
  TF_LITE_ENSURE_STATUS(r.AddMinimum());
  TF_LITE_ENSURE_STATUS(r.AddMul());
  TF_LITE_ENSURE_STATUS(r.AddPCAN());
  TF_LITE_ENSURE_STATUS(r.AddQuantize());
  TF_LITE_ENSURE_STATUS(r.AddReadVariable());
  TF_LITE_ENSURE_STATUS(r.AddReshape());
  TF_LITE_ENSURE_STATUS(r.AddRfft());
  TF_LITE_ENSURE_STATUS(r.AddSoftmax());
  TF_LITE_ENSURE_STATUS(r.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(r.AddSub());
  TF_LITE_ENSURE_STATUS(r.AddSvdf());
  TF_LITE_ENSURE_STATUS(r.AddUnidirectionalSequenceLSTM());
  TF_LITE_ENSURE_STATUS(r.AddVarHandle());
  TF_LITE_ENSURE_STATUS(r.AddWhile());
  TF_LITE_ENSURE_STATUS(r.AddWindow());
  return kTfLiteOk;
}

inline bool GuardIntact(size_t arena_size) {
  for (size_t i = 0; i < kGuardSize; ++i) {
    if (g_arena[arena_size + i] != kFill) {
      return false;
    }
  }
  return true;
}

// Runs Invoke, on zeroed inputs, after AllocateTensors succeeds.
inline TfLiteStatus InvokeAllocated(tflite::MicroInterpreter& interpreter) {
  for (size_t i = 0; i < interpreter.inputs_size(); ++i) {
    TfLiteTensor* input = interpreter.input(i);
    std::memset(input->data.raw, 0, input->bytes);
  }
  return interpreter.Invoke();
}

// Returns the AllocateTensors status; with `invoke_status`, a successful
// allocation is also invoked and that status stored there.
inline TfLiteStatus AllocateInArena(const tflite::Model* model,
                                    const Resolver& resolver, size_t arena_size,
                                    bool resource_variables, bool* guard_intact,
                                    TfLiteStatus* invoke_status = nullptr) {
  std::memset(g_arena, kFill, arena_size + kGuardSize);
  TfLiteStatus status = kTfLiteError;
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(g_arena, arena_size);
  tflite::MicroResourceVariables* variables =
      resource_variables ? tflite::MicroResourceVariables::Create(
                               allocator, kMaxResourceVariables)
                         : nullptr;
  {
    tflite::MicroInterpreter interpreter(model, resolver, allocator, variables);
    status = interpreter.AllocateTensors();
    if (invoke_status != nullptr) {
      *invoke_status =
          status == kTfLiteOk ? InvokeAllocated(interpreter) : kTfLiteError;
    }
  }
  *guard_intact = GuardIntact(arena_size);
  return status;
}

inline void SweepArenaSizes(const char* label, const unsigned char* model_data,
                            bool resource_variables = false) {
  Resolver resolver;
  ASSERT_EQ(AddModelOps(resolver), kTfLiteOk);
  const tflite::Model* model = tflite::GetModel(model_data);
  ASSERT_NE(model, nullptr);

  bool guard_intact = false;
  ASSERT_EQ(AllocateInArena(model, resolver, kArenaSize, resource_variables,
                            &guard_intact),
            kTfLiteOk);

  const size_t floor = ConstructorFloor();
  size_t required = 0;
  int undersized = 0;
  for (size_t size = floor; size <= kArenaSize; size += kStep) {
    const TfLiteStatus status = AllocateInArena(
        model, resolver, size, resource_variables, &guard_intact);
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
      "%s: smallest working arena %d, %d undersized arenas from %d bytes "
      "failed cleanly",
      label, static_cast<int>(required), undersized, static_cast<int>(floor));
  ASSERT_GT(required, 0u);
  ASSERT_LE(required + kMonotonicSpan, kArenaSize);

  // The smallest working arena and the sizes just above it allocate, invoke
  // and stay inside the arena.
  for (size_t size = required; size <= required + kMonotonicSpan;
       size += kStep) {
    TfLiteStatus invoke_status = kTfLiteError;
    EXPECT_EQ(AllocateInArena(model, resolver, size, resource_variables,
                              &guard_intact, &invoke_status),
              kTfLiteOk);
    EXPECT_EQ(invoke_status, kTfLiteOk);
    EXPECT_TRUE(guard_intact);
  }
}

// Fails the allocation numbered fail_index, and every later one when
// fail_later is set, counting from the first request after Arm().
class FailingArenaAllocator : public tflite::SingleArenaBufferAllocator {
 public:
  FailingArenaAllocator(uint8_t* buffer, size_t size, int fail_index,
                        bool fail_later)
      : SingleArenaBufferAllocator(buffer, size),
        fail_index_(fail_index),
        fail_later_(fail_later) {}

  void Arm() { armed_ = true; }
  int allocations() const { return allocations_; }

  uint8_t* AllocatePersistentBuffer(size_t size, size_t alignment) override {
    return Fail() ? nullptr
                  : SingleArenaBufferAllocator::AllocatePersistentBuffer(
                        size, alignment);
  }
  uint8_t* AllocateTemp(size_t size, size_t alignment) override {
    return Fail() ? nullptr
                  : SingleArenaBufferAllocator::AllocateTemp(size, alignment);
  }
  TfLiteStatus ResizeBuffer(uint8_t* buffer, size_t size,
                            size_t alignment) override {
    return Fail() ? kTfLiteError
                  : SingleArenaBufferAllocator::ResizeBuffer(buffer, size,
                                                             alignment);
  }
  uint8_t* AllocateResizableBuffer(size_t size, size_t alignment) override {
    return Fail() ? nullptr
                  : SingleArenaBufferAllocator::AllocateResizableBuffer(
                        size, alignment);
  }

 private:
  bool Fail() {
    if (!armed_) {
      return false;
    }
    const int index = allocations_++;
    return fail_later_ ? index >= fail_index_ : index == fail_index_;
  }

  const int fail_index_;
  const bool fail_later_;
  bool armed_ = false;
  int allocations_ = 0;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

// Returns the number of allocations AllocateTensors made.
inline int AllocateWithFailure(const tflite::Model* model,
                               const Resolver& resolver,
                               bool resource_variables, size_t arena_size,
                               int fail_index, bool fail_later,
                               TfLiteStatus* status) {
  std::memset(g_arena, kFill, arena_size + kGuardSize);
  FailingArenaAllocator arena(g_arena, arena_size, fail_index, fail_later);
  tflite::GreedyMemoryPlanner planner;
  tflite::MicroAllocator* allocator =
      tflite::MicroAllocator::Create(&arena, &planner);
  arena.Arm();
  tflite::MicroResourceVariables* variables =
      resource_variables ? tflite::MicroResourceVariables::Create(
                               allocator, kMaxResourceVariables)
                         : nullptr;
  {
    tflite::MicroInterpreter interpreter(model, resolver, allocator, variables);
    *status = interpreter.AllocateTensors();
  }
  return arena.allocations();
}

// Fails each allocation AllocateTensors makes together with every later one,
// as an exhausted arena does, and, unless `alone_too` is false, alone, as when
// a later smaller request still fits: AllocateTensors must return
// kTfLiteError each time.
inline void InjectAllocationFailures(const char* label,
                                     const tflite::Model* model,
                                     bool resource_variables = false,
                                     bool alone_too = true) {
  Resolver resolver;
  ASSERT_EQ(AddModelOps(resolver), kTfLiteOk);
  // Large enough for the model, so only the injected failures fail.
  size_t arena_size = kArenaSize;
  {
    tflite::MicroInterpreter interpreter(model, resolver, g_arena, kArenaSize);
    if (!resource_variables) {
      ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
      arena_size = interpreter.arena_used_bytes() + kInjectionHeadroom;
    }
  }
  ASSERT_LE(arena_size, kArenaSize);
  TfLiteStatus status = kTfLiteError;
  const int allocations = AllocateWithFailure(
      model, resolver, resource_variables, arena_size, -1, false, &status);
  ASSERT_EQ(status, kTfLiteOk);
  int unexpected = 0;
  for (int index = 0; index < allocations; ++index) {
    for (int fail_later = alone_too ? 0 : 1; fail_later < 2; ++fail_later) {
      AllocateWithFailure(model, resolver, resource_variables, arena_size,
                          index, fail_later != 0, &status);
      if (status != kTfLiteError) {
        MicroPrintf(
            "%s: AllocateTensors returned %d with allocation %d%s failing",
            label, static_cast<int>(status), index,
            fail_later != 0 ? " and later ones" : "");
        ++unexpected;
      }
      EXPECT_TRUE(GuardIntact(arena_size));
    }
  }
  MicroPrintf("%s: failed each of %d arena allocations%s and with the rest",
              label, allocations, alone_too ? " alone" : "");
  EXPECT_EQ(unexpected, 0);
}

inline void InjectAllocationFailures(const char* label,
                                     const unsigned char* model_data,
                                     bool resource_variables = false,
                                     bool alone_too = true) {
  InjectAllocationFailures(label, tflite::GetModel(model_data),
                           resource_variables, alone_too);
}

}  // namespace

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_HELIA_TESTS_ARENA_SWEEP_MODELS_H_
