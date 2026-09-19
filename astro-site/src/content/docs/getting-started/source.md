---
title: Build from source
description: Build a runtime archive with Make or integrate heliaRT source targets with CMake.
---

Use source builds to control the backend, floating-point features, kernel profiles and platform glue. Make produces a combined HELIA archive; CMake links the runtime and kernel dependency as separate targets.

## Prepare a checkout

Install Git, Python 3 and the build tools for your chosen path. CMake source integration requires CMake 3.21 or newer. Check out the runtime release you intend to use:

```bash
git clone https://github.com/AmbiqAI/helia-rt.git
cd helia-rt
git checkout helia-rt-v1.21.0
```

The examples below use this release's interfaces. If you choose another release, use its matching configuration and dependency pins.

## Build a HELIA archive with Make

Run from the repository root. This example selects Cortex-M55, GCC and a build retaining diagnostics:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 \
  TOOLCHAIN=gcc OPTIMIZED_KERNEL_DIR=helia \
  BUILD_TYPE=release_with_logs third_party_downloads

make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 \
  TOOLCHAIN=gcc OPTIMIZED_KERNEL_DIR=helia \
  BUILD_TYPE=release_with_logs microlite -j8
```

The build downloads its configured dependencies and places `libtensorflow-microlite.a` under `gen/.../lib/`. Use the path reported by the build; architecture, backend and build options affect the output directory. The HELIA path compiles selected heliaCORE sources into this archive, so it does not require a second heliaCORE archive at application link time.

`TOOLCHAIN` also accepts `armclang` and `atfe` on this target. Their installation and root settings come from the [target build rules](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/make/targets/cortex_m_generic_makefile.inc). `BUILD_TYPE` is separate from `GLOBAL_KERNEL_OPTIMIZE=SPEED|SIZE`; the latter selects supported kernel implementations.

An omitted embedded target produces a host build. Host success does not exercise MVE kernels. The [HELIA make fragment](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/make/ext_libs/helia.inc) is authoritative for dependency pins, float features and kernel options.

## Consume source with CMake

The repository includes the `third_party_static` headers and sources used by its CMake manifest. Put a pinned checkout under your application's `third_party/helia-rt`, then add the target after your project's toolchain and languages are configured:

```cmake
cmake_minimum_required(VERSION 3.21)
project(rt_app LANGUAGES C CXX ASM)

set(HELIA_RT_BUILD_TYPE "release_with_logs" CACHE STRING "Runtime build flavor")
add_subdirectory(third_party/helia-rt)

add_executable(rt_app src/main.cc src/model_data.cc)
target_link_libraries(rt_app PRIVATE helia_rt::reference)
```

This builds the Reference backend. The target propagates its required includes and definitions. Your application still supplies startup code, a linker script, board support and a cross-compilation toolchain file for embedded targets:

```bash
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/board-toolchain.cmake
cmake --build build
```

### Add HELIA kernels

Create the heliaCORE target **before** adding heliaRT. With pinned source checkouts under `third_party`, replace the runtime setup above with:

```cmake
set(ARM_NN_ENABLE_F32 OFF CACHE BOOL "Enable FP32 kernels")
set(ARM_NN_ENABLE_F16 OFF CACHE BOOL "Enable FP16 kernels")
add_subdirectory(third_party/ns-cmsis-nn)

set(HELIA_RT_ENABLE_HELIA ON CACHE BOOL "Build HELIA backend")
set(HELIA_RT_NSCMSISNN_TARGET "ns-cmsis-nn" CACHE STRING "Kernel dependency")
add_subdirectory(third_party/helia-rt)
```

Link `rt_app` to `helia_rt::helia` instead of `helia_rt::reference`. Supply the CMSIS Core headers and target compiler flags required by your heliaCORE checkout. Enable float features before adding that dependency when the model needs them. For the adapters in this runtime release, use heliaCORE v7.35.0 or newer.

CMake links heliaCORE transitively into the final application; it does not physically merge two static libraries. For the upstream Arm backend, use `HELIA_RT_ENABLE_CMSIS_NN=ON`, provide its target through `HELIA_RT_CMSISNN_TARGET`, and link `helia_rt::cmsis_nn` instead. Do not substitute heliaCORE for upstream CMSIS-NN.

### Platform logging and memory diagnostics

The default CMake debug logger uses the C library's standard error stream. If your firmware supplies another `DebugLog` implementation, set `HELIA_RT_PROVIDE_DEBUG_LOG=OFF` before adding heliaRT and compile your implementation into the application. Enable `HELIA_RT_ENABLE_RECORDING=ON` when using recording allocators; test helpers have a separate option and are unnecessary for ordinary inference.

See the [CMake interface](https://github.com/AmbiqAI/helia-rt/blob/main/CMakeLists.txt) for the complete option contract. Continue with [First inference](/helia-rt/getting-started/first-inference/) and [Model compatibility](/helia-rt/guide/model-compatibility/).
