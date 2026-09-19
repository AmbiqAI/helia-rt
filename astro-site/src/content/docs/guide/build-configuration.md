---
title: Build configuration
description: Choose a kernel backend, source or prebuilt integration, and the right build settings for your model.
---

Choose an [integration path](/helia-rt/getting-started/choose-your-path/), then configure the runtime and kernel library together. Backend, kernel profile, build flavor and floating-point support are separate decisions.

## Source or prebuilt

| Choose | When you need | Check before integration |
| --- | --- | --- |
| Source | Control over kernel profiles, float features, compiler options or debugging | Configure the kernel dependency before adding heliaRT. |
| Prebuilt archive | A release library that matches your firmware | Match the architecture, compiler, ABI and build flavor; keep the accompanying headers and library together. |

The [release builder](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/ci_build/build_helia.sh) combines heliaRT and HELIA kernel objects into one archive. In a CMake source build, `helia_rt::helia` instead carries the kernel dependency through its link interface. Linking one source-built archive by filename does not reproduce that dependency graph.

Changing application flags cannot change a prebuilt library. Use a source build when its compiled configuration does not fit your application.

## Select a backend

| Backend | CMake target | Kernel dependency |
| --- | --- | --- |
| HELIA | `helia_rt::helia` | heliaCORE, distributed as `ns-cmsis-nn` |
| CMSIS-NN | `helia_rt::cmsis_nn` | Upstream Arm CMSIS-NN |
| Reference | `helia_rt::reference` | Runtime reference implementations |

The root [CMake integration](https://github.com/AmbiqAI/helia-rt/blob/main/CMakeLists.txt) enables the HELIA and CMSIS-NN targets with `HELIA_RT_ENABLE_HELIA` and `HELIA_RT_ENABLE_CMSIS_NN`. Supply the matching kernel dependency target. The HELIA and upstream CMSIS-NN libraries have different API contracts and are not interchangeable.

Use your integration's backend selector:

- **neuralSPOT-X:** `NSX_HELIA_RT_BACKEND=helia`, `cmsis_nn` or `reference`; applications link `nsx::helia_rt`.
- **Zephyr:** `CONFIG_HELIA_RT_BACKEND_HELIA`, `CONFIG_HELIA_RT_BACKEND_CMSIS_NN` or `CONFIG_HELIA_RT_BACKEND_REFERENCE`.
- **Make:** `OPTIMIZED_KERNEL_DIR=helia` selects the HELIA adapters.

Backend selection happens at build time. Operator types, shapes and options still determine whether a particular model is supported. A reference fallback is an operator-specific behavior, not a guarantee for every unsupported configuration.

## Choose SPEED or SIZE

The HELIA kernel profile selects implementations for kernels that offer a latency or code-footprint tradeoff. `SPEED` is the default. Measure both profiles with your model if code size is a constraint; the profile name alone does not establish a power or memory improvement.

| Integration | Profile setting |
| --- | --- |
| CMake source | `HELIA_RT_GLOBAL_KERNEL_OPTIMIZE=SPEED` or `SIZE` |
| Make source | `GLOBAL_KERNEL_OPTIMIZE=SPEED` or `SIZE` |
| Zephyr source | `CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SPEED=y` or `CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SIZE=y` |

CMake source builds also expose `HELIA_RT_CONV_OPT` and `HELIA_RT_FC_OPT`; Make uses `CONV_OPT` and `FC_OPT`. Each accepts `SPEED` or `SIZE` to override the global profile for its kernel family. See the [source configuration contract](https://github.com/AmbiqAI/helia-rt/blob/main/cmake/helia_rt_sources.cmake).

The release matrix has no separate SPEED/SIZE axis. Use source integration to select a different kernel profile.

## Keep build flavor separate

Build flavor controls runtime assertions and diagnostic strings. It does not select the HELIA kernel profile.

| `HELIA_RT_BUILD_TYPE` in CMake / `BUILD_TYPE` in Make | Runtime definitions |
| --- | --- |
| `debug` | Does not add `NDEBUG` or strip error strings. |
| `release_with_logs` | Adds `NDEBUG`; keeps error strings. |
| `release` | Adds `NDEBUG` and `TF_LITE_STRIP_ERROR_STRINGS`. |

The neuralSPOT-X wrapper uses `HELIA_RT_VARIANT` with `debug`, `release-with-logs` or `release`, mapping the hyphenated value to the runtime's underscore spelling. Start with diagnostics available while bringing up a model, and check `AllocateTensors()` and `Invoke()` return values in every flavor.

## Match the compiler and float features

The [release workflow](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/helia_release.yml) builds GCC, Arm Compiler 6 (`armclang`) and Arm Toolchain for Embedded (`atfe`) archives for `cortex-m4+fp` and `cortex-m55`. Select the archive that matches your application's architecture and toolchain. A release build matrix is not evidence that every compiler configuration has executed the same runtime tests.

For CMake source builds, configure `ARM_NN_ENABLE_F32` and `ARM_NN_ENABLE_F16` **before** the `ns-cmsis-nn` dependency is added. heliaRT reads the float capabilities exported by that dependency. Enabling a macro only on the application cannot add missing kernel implementations to the linked library.

FP16 weight storage and FP16 computation are different requirements. A model that widens FP16 weights to FP32 through `DEQUANTIZE` does not require optimized FP16 arithmetic. Models with FP16 compute operators need the corresponding adapter, kernel feature and target support. See the [floating-point integration contract](https://github.com/AmbiqAI/helia-rt/blob/main/docs/guides/floating-point.md) for each build system's defaults and unsupported cases.

## Verify the application

Build and link the final firmware, then run the model with the intended inputs on your target. Check registration, allocation and invocation results before comparing output correctness, arena use and execution time. A successful archive build cannot establish that all of the final application's kernel symbols resolve or that the model executes correctly.
