# Memory Placement

## Overview

heliaRT receives pointers to the model flatbuffer and tensor arena. It does not decide where those objects live in memory. Placement is an application, board, and linker concern, which means the same heliaRT runtime can work with models in MRAM, arenas in SRAM, or performance-sensitive buffers in tightly coupled memory when the platform supports it.

On Ambiq Apollo SoCs, memory placement is usually about balancing three goals: keeping frequently accessed tensors close to the core, preserving scarce low-latency memory for the workloads that need it most, and keeping the build repeatable across debug and release configurations.

## What heliaRT Expects

- A valid `.tflite` flatbuffer, usually exposed through `tflite::GetModel()`.
- A statically allocated tensor arena with the alignment expected by LiteRT.
- A build system that keeps those objects in memory regions available at runtime.

The runtime does not require a heliaRT-specific allocator. If the model pointer and arena pointer are valid, heliaRT can use them.

## Zephyr Placement Options

For Zephyr applications, placement is normally handled with standard Zephyr and linker mechanisms:

| Mechanism | Use when |
|---|---|
| Section attributes | You want to place a specific model or arena object in a named section. |
| Linker fragments | You need repeatable placement rules shared by multiple source files. |
| Devicetree memory regions | You want board-level memory partitions that the application can reference. |
| Linker map inspection | You need to verify the final placement after optimization and dead-code removal. |

## Practical Guidance

- Keep the model flatbuffer in nonvolatile memory when startup latency and access patterns allow it.
- Place the tensor arena in a writable region with enough contiguous space for the largest planned model.
- Reserve faster memory for tensors or kernels that materially affect latency on the target workload.
- Check the linker map as part of bring-up; placement assumptions are easy to invalidate when build flags change.
- Keep debug and release placement rules aligned so benchmark numbers match the artifact you intend to ship.

## Example Shape

Application code typically owns the model data and arena:

```cpp
alignas(16) const unsigned char g_model[] = {
    // generated .tflite bytes
};

alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
```

Board-specific section attributes or linker fragments can then move those objects without changing heliaRT itself.

## Next Steps

- [Zephyr setup](../getting-started/zephyr.md) — choose the module path for your product
- [Zephyr example](../examples/zephyr.md) — see a complete application flow
- [Troubleshooting](troubleshooting.md) — debug arena sizing, backend selection, and build issues
