# API Surface

heliaRT's public API is intentionally minimal and fully compatible with upstream LiteRT. This page documents the heliaRT-specific additions.

## heliaRT Version Header

```c
#include "tensorflow/lite/micro/helia_rt_version.h"

// Returns the heliaRT release version string
const char* version = HELIA_RT_VERSION;  // e.g. "v1.16.0"
```

This header provides the `HELIA_RT_VERSION` macro, which is managed by [release-please](https://github.com/googleapis/release-please) and updated automatically on every release.

## Upstream LiteRT API

heliaRT does **not** modify or extend the upstream LiteRT for Micro API. The following core types work identically:

| Type | Header | Purpose |
|---|---|---|
| `tflite::MicroInterpreter` | `micro_interpreter.h` | Run inference |
| `tflite::Model` | `model.h` | Load a `.tflite` flatbuffer |
| `tflite::MicroMutableOpResolver<N>` | `micro_mutable_op_resolver.h` | Register operators |
| `tflite::MicroAllocator` | `micro_allocator.h` | Arena memory management |
| `tflite::MicroProfiler` | `micro_profiler.h` | Per-layer profiling |

For detailed API documentation, refer to the [upstream LiteRT for Micro API docs](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro).

## Build-Time Configuration

### Compile Definitions

| Macro | Set by | Meaning |
|---|---|---|
| `CMSIS_NN` | HELIA backend | NN library exposes CMSIS-NN API surface |
| `HELIA` | HELIA backend | Build uses `OPTIMIZED_KERNEL_DIR=helia` |
| `CONV_KERNEL_OPTIMIZED_FOR_SPEED` | Per-kernel knob | Conv2D optimized for latency |
| `FC_KERNEL_OPTIMIZED_FOR_SIZE` | Per-kernel knob | FullyConnected optimized for code size |

### Zephyr Kconfig Options

| Option | Type | Description |
|---|---|---|
| `CONFIG_HELIA_RT` | bool | Enable heliaRT module |
| `CONFIG_HELIA_RT_BACKEND_REFERENCE` | choice | Use Reference kernels |
| `CONFIG_HELIA_RT_BACKEND_CMSIS_NN` | choice | Use open CMSIS-NN kernels |
| `CONFIG_HELIA_RT_BACKEND_HELIA` | choice | Use Ambiq HELIA kernels (default when available) |
| `CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SPEED` | choice | Use HELIA speed-optimized kernel paths |
| `CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SIZE` | choice | Use HELIA size-optimized kernel paths |

## Next Steps

- [Operator Coverage](operator-coverage.md) — which operators have optimized kernels
- [Upgrading from LiteRT](../guides/upgrading-from-litert.md) — migration guide
