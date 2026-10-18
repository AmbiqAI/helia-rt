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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_ALLOCATION_SWEEP_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_ALLOCATION_SWEEP_H_

// Arena-exhaustion sweep for every KernelRunner test of a helia make build.
//
// Before the real Init and Prepare, KernelRunner::InitAndPrepare runs the
// kernel's Init and Prepare again on a copy of the node once per arena
// allocation the kernel makes (persistent buffer, temp tensor or temp
// buffer): first failing that allocation alone, then failing it and every
// later one, as an undersized arena does. Each kernel and input/output type
// pair is swept for its first 16 nodes in a binary. As MicroInterpreterGraph
// does,
// Prepare is skipped when Init saw a failure. A kernel that uses the nullptr
// crashes the test binary; any returned status is accepted, because the
// interpreter fails AllocateTensors after a failed allocation regardless.
// see AmbiqAI/helia-rt#449

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/fake_micro_context.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/mock_micro_graph.h"

namespace tflite {
namespace micro {
namespace helia {

// Tests whose link wraps count heliaCORE calls made in Prepare turn the sweep
// off, since it repeats those calls.
inline bool& AllocationSweepEnabled() {
  static bool enabled = true;
  return enabled;
}

enum class AllocationKind { kAny, kPersistentBuffer, kTempTensor, kTempBuffer };

// Fails the allocation of `kind` numbered fail_index, and every later one of
// that kind when fail_later is set.
class AllocationFailureContext : public FakeMicroContext {
 public:
  AllocationFailureContext(TfLiteTensor* tensors,
                           SingleArenaBufferAllocator* allocator,
                           MicroGraph* graph, int fail_index, bool fail_later,
                           AllocationKind kind)
      : FakeMicroContext(tensors, allocator, graph),
        fail_index_(fail_index),
        fail_later_(fail_later),
        kind_(kind) {}

  void* AllocatePersistentBuffer(size_t bytes) override {
    return Fail(AllocationKind::kPersistentBuffer)
               ? nullptr
               : FakeMicroContext::AllocatePersistentBuffer(bytes);
  }
  TfLiteTensor* AllocateTempTfLiteTensor(int tensor_index) override {
    return Fail(AllocationKind::kTempTensor)
               ? nullptr
               : FakeMicroContext::AllocateTempTfLiteTensor(tensor_index);
  }
  uint8_t* AllocateTempBuffer(size_t size, size_t alignment) override {
    return Fail(AllocationKind::kTempBuffer)
               ? nullptr
               : FakeMicroContext::AllocateTempBuffer(size, alignment);
  }
  void DeallocateTempTfLiteTensor(TfLiteTensor* tensor) override {
    if (tensor != nullptr) {
      FakeMicroContext::DeallocateTempTfLiteTensor(tensor);
    }
  }

  int allocations() const { return allocations_; }
  bool failed() const { return failed_; }

 private:
  bool Fail(AllocationKind kind) {
    if (kind_ != AllocationKind::kAny && kind != kind_) {
      return false;
    }
    const int index = allocations_++;
    const bool fail = fail_later_ ? index >= fail_index_ : index == fail_index_;
    failed_ = failed_ || fail;
    return fail;
  }

  const int fail_index_;
  const bool fail_later_;
  const AllocationKind kind_;
  int allocations_ = 0;
  bool failed_ = false;

  TF_LITE_REMOVE_VIRTUAL_DELETE
};

inline void SilentReportError(TfLiteContext*, const char*, ...) {}

constexpr int kMaxSweptTensors = 64;
constexpr int kMaxSweptAllocations = 128;
constexpr size_t kSweepArenaSize = 10000;

struct AllocationRun {
  // Allocations of the selected kind the run made.
  int allocations;
  // Init saw a failed allocation, so Prepare did not run.
  bool init_failed;
  TfLiteStatus prepare_status;
};

// Runs Init and, unless Init saw a failed allocation, Prepare on a copy of
// `node` in a fresh 10000-byte arena.
inline AllocationRun RunWithAllocationFailure(
    const TFLMRegistration& registration, const TfLiteNode& node,
    TfLiteTensor* tensors, const char* init_data, size_t length, int fail_index,
    bool fail_later, AllocationKind kind = AllocationKind::kAny) {
  alignas(16) static uint8_t arena[kSweepArenaSize];
  SingleArenaBufferAllocator* allocator =
      SingleArenaBufferAllocator::Create(arena, sizeof(arena));
  MockMicroGraph graph(allocator);
  AllocationFailureContext micro_context(tensors, allocator, &graph, fail_index,
                                         fail_later, kind);
  TfLiteContext context = {};
  context.impl_ = &micro_context;
  context.ReportError = SilentReportError;
  context.GetTensor = MicroContextGetTensor;
  context.GetEvalTensor = MicroContextGetEvalTensor;
  context.AllocatePersistentBuffer = MicroContextAllocatePersistentBuffer;
  TfLiteNode swept = node;
  swept.user_data = nullptr;
  AllocationRun run = {0, false, kTfLiteOk};
  if (registration.init != nullptr) {
    swept.user_data = registration.init(&context, init_data, length);
  }
  run.init_failed = micro_context.failed();
  if (!run.init_failed && registration.prepare != nullptr) {
    context.RequestScratchBufferInArena =
        MicroContextRequestScratchBufferInArena;
    context.GetExternalContext = MicroContextGetExternalContext;
    run.prepare_status = registration.prepare(&context, &swept);
  }
  if (registration.free != nullptr) {
    registration.free(&context, swept.user_data);
  }
  run.allocations = micro_context.allocations();
  return run;
}

// Each kernel and input/output type pair is swept at most this many times per
// binary, so a test that prepares one configuration many times stays fast on
// the FVP.
constexpr int kMaxSweepsPerConfiguration = 16;
constexpr int kMaxSweptConfigurations = 256;

inline TfLiteType NodeTensorType(const TfLiteIntArray* indices,
                                 const TfLiteTensor* tensors) {
  if (indices == nullptr || indices->size == 0 || indices->data[0] < 0) {
    return kTfLiteNoType;
  }
  return tensors[indices->data[0]].type;
}

// Returns true while the configuration is under its sweep budget.
inline bool CountSweep(const TFLMRegistration& registration,
                       const TfLiteNode& node, const TfLiteTensor* tensors) {
  struct Configuration {
    const void* kernel;
    TfLiteType input_type;
    TfLiteType output_type;
    int sweeps;
  };
  static Configuration configurations[kMaxSweptConfigurations];
  static int count = 0;
  const void* kernel = reinterpret_cast<const void*>(registration.prepare);
  const TfLiteType input_type = NodeTensorType(node.inputs, tensors);
  const TfLiteType output_type = NodeTensorType(node.outputs, tensors);
  for (int i = 0; i < count; ++i) {
    Configuration& c = configurations[i];
    if (c.kernel == kernel && c.input_type == input_type &&
        c.output_type == output_type) {
      return ++c.sweeps <= kMaxSweepsPerConfiguration;
    }
  }
  if (count == kMaxSweptConfigurations) {
    return false;
  }
  configurations[count++] = {kernel, input_type, output_type, 1};
  return true;
}

inline void SweepAllocationFailures(const TFLMRegistration& registration,
                                    const TfLiteNode& node,
                                    TfLiteTensor* tensors, int tensors_size,
                                    const char* init_data, size_t length) {
  if (!AllocationSweepEnabled() || tensors_size > kMaxSweptTensors ||
      !CountSweep(registration, node, tensors)) {
    return;
  }
  // Prepare may repoint a tensor's dims into the sweep arena.
  TfLiteTensor saved[kMaxSweptTensors];
  std::memcpy(saved, tensors, sizeof(TfLiteTensor) * tensors_size);
  int allocations = RunWithAllocationFailure(registration, node, tensors,
                                             init_data, length, -1, false)
                        .allocations;
  std::memcpy(tensors, saved, sizeof(TfLiteTensor) * tensors_size);
  if (allocations > kMaxSweptAllocations) {
    allocations = kMaxSweptAllocations;
  }
  for (int index = 0; index < allocations; ++index) {
    for (int fail_later = 0; fail_later < 2; ++fail_later) {
      RunWithAllocationFailure(registration, node, tensors, init_data, length,
                               index, fail_later != 0);
      std::memcpy(tensors, saved, sizeof(TfLiteTensor) * tensors_size);
    }
  }
}

}  // namespace helia
}  // namespace micro
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_ALLOCATION_SWEEP_H_
