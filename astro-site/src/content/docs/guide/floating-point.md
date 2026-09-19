---
title: Floating point
description: Configure FP32 and FP16 kernels and distinguish floating-point storage from computation.
---

HELIA floating-point operators depend on both the RT adapter and the heliaCORE library linked into the application. The library is distributed as `ns-cmsis-nn`. Its `ARM_NN_ENABLE_F32` and `ARM_NN_ENABLE_F16` features select the optimized float APIs available to RT.

## Weights are not compute

A `.tflite` model can store weights as FP16, widen them through `DEQUANTIZE`, and perform computation in FP32. That is different from a graph whose operators consume and produce FP16 tensors.

The [DEQUANTIZE adapter](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/kernels/helia/dequantize.cc) widens FP16 storage without requiring `ARM_NN_ENABLE_F16`. `RESHAPE` and `TRANSPOSE` also have bitwise FP16 storage paths. These cases do not establish that FP16 arithmetic is available.

## Source CMake and neuralSPOT-X

Configure the dependency's options before adding `ns-cmsis-nn`, then add heliaRT. A parent CMake project can use:

```cmake
set(HELIA_RT_ENABLE_HELIA ON CACHE BOOL "" FORCE)
set(ARM_NN_ENABLE_F32 ON CACHE BOOL "" FORCE)
set(ARM_NN_ENABLE_F16 ON CACHE BOOL "" FORCE)

add_subdirectory(${NS_CMSIS_NN_DIR} ns-cmsis-nn)
add_subdirectory(${HELIA_RT_DIR} helia-rt)
target_link_libraries(app PRIVATE helia_rt::helia)
```

Supply the parent project's cross-toolchain and board configuration. Enable FP16 only when the target and toolchain support the required MVE floating-point implementation.

For neuralSPOT-X, put the feature options before `nsx_bootstrap_app()`. The bootstrap adds the kernel dependency before the runtime module. Setting compiler definitions afterward cannot change the dependency's selected sources.

Both float features are opt-in on the NSX source path. The [NSX wrapper](https://github.com/AmbiqAI/helia-rt/blob/main/nsx/CMakeLists.txt) reports the resolved kernel features and publishes:

| Output | Meaning |
| --- | --- |
| `HELIA_RT_FLOAT32_ENABLED` | Effective FP32 kernel availability |
| `HELIA_RT_FLOAT16_ENABLED` | Effective FP16 kernel availability |
| `HELIA_RT_TARGET_HAS_MVE_FP` | Cached compile-probe result using the board's compiler flags |

The first two are outputs, not input switches. heliaRT asks the dependency's float-support query where available, otherwise it reads exported definitions. It warns when a requested feature differs from the library's resolved feature set. See the [query implementation](https://github.com/AmbiqAI/helia-rt/blob/main/cmake/helia_rt_sources.cmake).

:::caution[Renamed options]
Use `ARM_NN_ENABLE_F32` and `ARM_NN_ENABLE_F16`. The former `NSX_CMSIS_NN_ENABLE_F32/F16` CMake inputs were removed by ns-cmsis-nn v7.32.0. Remove obsolete cache entries when migrating a build, and check the dependency revision selected by your SDK or module registry.
:::

## Zephyr

Kconfig resolves the feature set before either module is compiled:

```ini
CONFIG_HELIA_RT=y
CONFIG_HELIA_RT_BACKEND_HELIA=y
CONFIG_NS_CMSIS_NN_ENABLE_F32=y
CONFIG_NS_CMSIS_NN_ENABLE_F16=y
```

The HELIA backend implies FP32 and implies FP16 when `ARMV8_1_M_MVEF` is available. `imply` is a weak default: an explicit user setting can disable it. Inspect the final `build/zephyr/.config`, not just `prj.conf`. FP16 remains subject to the dependency's target requirements. See the [RT Kconfig](https://github.com/AmbiqAI/helia-rt/blob/main/zephyr/Kconfig).

## Make and release archives

The [HELIA Make fragment](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/make/ext_libs/helia.inc) enables FP32 and enables FP16 for `TARGET_ARCH=cortex-m55`. Its source selection and matching definitions are applied together. The release builder uses this path for its combined runtime/kernel archive.

| Release target | Optimized float feature set |
| --- | --- |
| `cortex-m4+fp` | FP32 |
| `cortex-m55` | FP32 and FP16 |

The M55 archive assumes the required floating-point MVE configuration. The architecture name alone is not a substitute for checking application compiler flags and ABI compatibility.

When using a separate prebuilt kernel library with a source runtime, retain its feature manifest. The kernel dependency must export the same features the archive contains. A successful static-library build is not a final link check.

## Adapter and kernel version pairing

Keep the RT release paired with its documented kernel dependency. The Make source pin is recorded in `helia.inc`; separately supplied source targets or archives must be updated deliberately.

| Adapter capability | Minimum heliaCORE version |
| --- | --- |
| Float SPLIT, SPLIT_V, PACK, UNPACK and FILL; FP16 SQRT and RSQRT | `v7.33.0` |
| GATHER and GATHER_ND | `v7.34.0` |
| FP16 ARG_MIN and ARG_MAX | `v7.35.0` |

These boundaries are recorded in the adapter changes and [RT changelog](https://github.com/AmbiqAI/helia-rt/blob/main/CHANGELOG.md). They do not replace the release's full dependency requirement.

## Unsupported configurations and numeric behavior

Many FP32 adapters retain a reference path when optimized support is disabled or rejects a shape. Many FP16 arithmetic adapters have no reference fallback and return `kTfLiteError`. A known type or shape restriction can fail `AllocateTensors()`; execution-dependent failures can reach `Invoke()`. Check both.

Compatibility also includes numeric semantics. For example, optimized activation paths can differ from the reference path for NaN inputs; do not assume NaN propagation from a dtype name. Grouped float convolution, non-unit softmax beta, broadcast ranks and stateful LSTM each have their own contracts. Use [model compatibility](/helia-rt/guide/model-compatibility/) and the [operator coverage](/helia-rt/guide/operators/) for the adapter-specific boundaries, and test representative inputs including any non-finite values your application permits.

## LSTM variants and state

The shared LSTM preparation validates all four gates and rejects peephole, projection and internal layer-normalization tensors. CIFG is not accepted by that shared contract. These restrictions apply before float evaluation, so FP32 reference fallback does not make those variants supported. See [LSTM validation](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/kernels/lstm_eval.cc).

For the supported float LSTM form, preserve the hidden and cell state tensors across invocations and test reset behavior. Optimized state carry requires the corresponding heliaCORE stateful implementation; the version boundary documented for the float path is v7.29.0. Keep RT and CORE paired and verify a sequence, not only the first invocation. The quantized release-history boundary is documented separately in the [support policy](/helia-rt/guide/maintenance/support/).
