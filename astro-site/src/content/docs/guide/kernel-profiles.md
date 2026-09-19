---
title: Kernel profiles
description: Select SPEED or SIZE globally and override convolution or fully connected kernels.
---

The HELIA backend offers `SPEED` and `SIZE` source-build profiles. These choose different implementations for selected kernels. They are independent of `debug` / `release_with_logs` / `release`, compiler optimization levels, and whether the target enables MVE.

## What each setting controls

| Scope | Make setting | CMake setting | Default |
| --- | --- | --- | --- |
| Global profile | `GLOBAL_KERNEL_OPTIMIZE` | `HELIA_RT_GLOBAL_KERNEL_OPTIMIZE` | `SPEED` |
| Convolution family | `CONV_OPT` | `HELIA_RT_CONV_OPT` | Inherit global |
| Fully connected | `FC_OPT` | `HELIA_RT_FC_OPT` | Inherit global |

All explicit values are `SPEED` or `SIZE`. The convolution override covers `CONV_2D`, `DEPTHWISE_CONV_2D` and `TRANSPOSE_CONV`. The fully connected override covers `FULLY_CONNECTED`. `SVDF` uses the global profile; it has no independent `SVDF_OPT` setting. These are family-wide build choices, not controls for individual nodes within a model.

The build emits `CONV_KERNEL_OPTIMIZED_FOR_<profile>`, `FC_KERNEL_OPTIMIZED_FOR_<profile>` and `KERNELS_OPTIMIZED_FOR_<profile>`. Use the build options instead of manually defining both SPEED and SIZE macros.

These scopes follow the [CMake definitions](https://github.com/AmbiqAI/helia-rt/blob/main/cmake/helia_rt_sources.cmake), [Make integration](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/make/ext_libs/helia.inc) and their use in the [HELIA adapters](https://github.com/AmbiqAI/helia-rt/tree/main/tensorflow/lite/micro/kernels/helia).

## CMake and neuralSPOT-X

Set options before heliaRT is added to the project. For neuralSPOT-X, place them before the application's module bootstrap.

```cmake
set(HELIA_RT_GLOBAL_KERNEL_OPTIMIZE SIZE CACHE STRING "" FORCE)
set(HELIA_RT_CONV_OPT SPEED CACHE STRING "" FORCE)
set(HELIA_RT_FC_OPT SIZE CACHE STRING "" FORCE)
```

This selects the SIZE global path, including SVDF, while choosing SPEED for the convolution family. To restore inheritance, set the family override to an empty string:

```cmake
set(HELIA_RT_CONV_OPT "" CACHE STRING "" FORCE)
```

CMake validates the three resolved values. Check `CMakeCache.txt` and the generated compilation commands when comparing configurations.

## Make

This source-build example selects the same mixed profile:

```sh
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 \
  TOOLCHAIN=gcc OPTIMIZED_KERNEL_DIR=helia \
  GLOBAL_KERNEL_OPTIMIZE=SIZE CONV_OPT=SPEED FC_OPT=SIZE \
  BUILD_TYPE=release_with_logs microlite
```

Remove `CONV_OPT` or `FC_OPT` to inherit the global value. Use a clean or separately isolated build when changing flags so objects compiled with a previous profile are not reused.

## Zephyr

Choose the global profile in `prj.conf`:

```ini
CONFIG_HELIA_RT=y
CONFIG_HELIA_RT_BACKEND_HELIA=y
CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SIZE=y
```

Use `CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SPEED=y` for SPEED. The [Zephyr integration](https://github.com/AmbiqAI/helia-rt/blob/main/zephyr/CMakeLists.txt) maps that Kconfig choice to `HELIA_RT_GLOBAL_KERNEL_OPTIMIZE`. It does not expose separate per-family Kconfig symbols. For family overrides in a source integration, set the CMake cache variables before Zephyr configures the module, for example through your build's CMake arguments, and inspect the resulting definitions.

## Prebuilt libraries

The [release workflow](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/helia_release.yml) selects architecture, toolchain and build flavor. It does not publish a SPEED/SIZE axis. Changing an application's profile options cannot alter the already compiled archive; build from source to choose a different profile.

## Compare results

Use the same model, inputs, dependency revision and target for both builds. Check output correctness first, then compare final firmware code size, arena use, other memory allocations and inference time. SPEED can require different scratch or persistent buffers; SIZE does not promise lower energy or less RAM. A profile is a configuration choice whose result must be measured for the application.
