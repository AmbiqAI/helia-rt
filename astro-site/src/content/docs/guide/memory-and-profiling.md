---
title: Memory and profiling
description: Size and place the tensor arena, audit allocations and measure runtime execution.
---

The application owns the model buffer, resolver and tensor arena. They must remain valid for the interpreter's lifetime. heliaRT receives their pointers; the firmware's linker and board configuration determine where those objects live.

## Allocate and align the arena

Use a persistent, aligned writable buffer. The capacity below is an application-defined constant, not a recommended model size:

```cpp
alignas(16) static uint8_t tensor_arena[kArenaSize];
```

Check `AllocateTensors()` before reading inputs or using allocation statistics. A preparation failure can mean an unsupported model configuration, not just insufficient memory. After successful allocation, `arena_used_bytes()` reports allocator usage for that model and build. Leave room for alignment and verify the final chosen capacity through execution; the [interpreter header](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_interpreter.h) describes this measurement.

Recheck memory after changing the model, backend, kernel profile or runtime version. The arena does not include the model buffer, firmware stacks, profiler storage or every other application allocation.

## Record allocation categories

Enable recording support in a source build:

| Integration | Setting |
| --- | --- |
| CMake / neuralSPOT-X | `HELIA_RT_ENABLE_RECORDING=ON` |
| Zephyr | `CONFIG_HELIA_RT_ENABLE_RECORDING=y` |

Use `RecordingMicroInterpreter` for an allocation audit:

```cpp
#include "tensorflow/lite/micro/recording_micro_interpreter.h"

tflite::RecordingMicroInterpreter interpreter(
    model, resolver, tensor_arena, sizeof(tensor_arena));
if (interpreter.AllocateTensors() != kTfLiteOk) {
  return kTfLiteError;
}
interpreter.GetMicroAllocator().PrintAllocations();
```

This is an excerpt inside a function returning `TfLiteStatus`; model validation and resolver setup are shown in [First inference](/helia-rt/getting-started/first-inference/). Recording adds its own overhead. Use additional arena capacity for the diagnostic build and confirm the production interpreter separately. See [RecordingMicroInterpreter](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/recording_micro_interpreter.h).

## Place buffers deliberately

A `const` model array can live in a read-only linker region; the arena must be writable. A writable initialized model copy may require both stored initialization data and RAM. Ensure alignment, lifetime, access permissions and any cache/DMA coherency requirements match the board's memory system.

For Zephyr, use the board's devicetree memory regions, section attributes and application linker fragments. A section such as DTCM is usable only when the board actually defines it. Do not copy addresses, capacities or section names from a different board.

Inspect the generated linker map for the exact model and arena symbols. Check both the output section and address against your board's linker regions. Source annotations alone do not prove placement. Record placement with performance results because moving buffers can change execution behavior.

## Profile operators

`MicroInterpreter` accepts a `MicroProfilerInterface`. The provided `MicroProfiler` records events and can print per-event or per-tag results:

```cpp
#include "tensorflow/lite/micro/micro_profiler.h"

static tflite::MicroProfiler profiler;
tflite::MicroInterpreter interpreter(
    model, resolver, tensor_arena, sizeof(tensor_arena), nullptr, &profiler);
```

After checked allocation and input preparation, clear old events, invoke, check status, then log the events:

```cpp
profiler.ClearEvents();
if (interpreter.Invoke() != kTfLiteOk) {
  return kTfLiteError;
}
profiler.LogTicksPerTagCsv();
```

Use a build that retains diagnostic strings. `TF_LITE_STRIP_ERROR_STRINGS`, used by the `release` flavor, makes `ScopedMicroProfiler` a no-op. `release_with_logs` is useful for operator profiling. See [MicroProfiler](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_profiler.h).

The platform must implement a working timer through [micro_time.h](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_time.h). Some platform configurations return zero. Verify the timer and its tick frequency before interpreting the output as cycles or time. Clear events between runs to avoid exhausting the profiler's fixed event storage.

## Compare complete applications

Keep input preparation, inference and output processing as separate timing intervals when they answer different questions. Report model and data revisions, runtime/kernel versions, target, compiler, kernel profile, memory placement, warmup policy and sample count. Measure energy directly when making an energy claim; fewer cycles alone do not establish power or energy savings.
