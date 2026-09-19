---
title: Zephyr
description: Add heliaRT to a west workspace using source modules or a released prebuilt module.
---

Start with a working Zephyr workspace, a board application that already builds, and the SDK/toolchain for that board. Integrate either source modules or a prebuilt module. Do not add both to the same application.

## Add HELIA source modules

Add these projects to the `manifest.projects` list in your workspace's west manifest. This example pairs runtime v1.21.0 with the heliaCORE version required by its adapters:

```yaml
- name: helia-rt
  url: https://github.com/AmbiqAI/helia-rt
  revision: helia-rt-v1.21.0
  path: modules/helia-rt
- name: ns-cmsis-nn
  url: https://github.com/AmbiqAI/ns-cmsis-nn
  revision: v7.35.0
  path: modules/ns-cmsis-nn
```

Fetch both projects from the workspace root:

```bash
west update helia-rt ns-cmsis-nn
```

West discovers their module metadata automatically. For local checkouts outside the manifest, add their absolute paths to `ZEPHYR_EXTRA_MODULES` before `find_package(Zephyr ...)` instead.

### Configure the application

A minimal application `CMakeLists.txt` is:

```cmake
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(helia_rt_app)
target_sources(app PRIVATE src/main.cpp src/model_data.cpp)
```

Add to `prj.conf`:

```ini
CONFIG_STD_CPP17=y
CONFIG_HELIA_RT=y
CONFIG_NS_CMSIS_NN=y
CONFIG_HELIA_RT_BACKEND_HELIA=y
CONFIG_PRINTK=y
CONFIG_CONSOLE=y
```

The HELIA backend is conditional on an Ambiq target and the heliaCORE module. Kconfig implies the float features available to the target; inspect `build/zephyr/.config` for the resolved `CONFIG_NS_CMSIS_NN_ENABLE_F32/F16` values. Explicitly disable unneeded float features for an integer-only build, or enable the required ones for a float model. FP16 arithmetic needs an MVE floating-point target.

For source builds, choose `CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SPEED=y` (the default) or `CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SIZE=y`. These select kernel profiles, not debug/release flavors.

### Other source backends

For the Reference backend, replace the HELIA settings with:

```ini
CONFIG_HELIA_RT=y
CONFIG_NS_CMSIS_NN=n
CONFIG_HELIA_RT_BACKEND_REFERENCE=y
```

For upstream Arm CMSIS-NN, make its module available in the workspace and use:

```ini
CONFIG_HELIA_RT=y
CONFIG_NS_CMSIS_NN=n
CONFIG_CMSIS_NN=y
CONFIG_HELIA_RT_BACKEND_CMSIS_NN=y
```

Enable the upstream CMSIS-NN kernel groups required by your model using that module's Kconfig. Its symbols and dependency are separate from `NS_CMSIS_NN`; do not substitute one module for the other.

## Use a prebuilt module instead

Extract a matching [release bundle](https://github.com/AmbiqAI/helia-rt/releases) into your workspace. Remove the source modules from this application's module discovery. Add the bundle before Zephyr is loaded:

```cmake
list(APPEND ZEPHYR_EXTRA_MODULES
  "${CMAKE_CURRENT_SOURCE_DIR}/../../modules/helia-rt-bundle")
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
```

Adjust the relative path to your extracted bundle. Configure:

```ini
CONFIG_STD_CPP17=y
CONFIG_HELIA_RT=y
CONFIG_FPU=y
CONFIG_HELIA_RT_PREBUILT_BUILD_RELEASE_WITH_LOGS=y
```

The module selects the archive from the board CPU, toolchain and flavor. It accepts Cortex-M4 with FPU or Cortex-M55; the archive must match the target's floating-point ABI and features. Alternative flavor settings are `CONFIG_HELIA_RT_PREBUILT_BUILD_DEBUG` and `CONFIG_HELIA_RT_PREBUILT_BUILD_RELEASE`.

Do not add a separate heliaCORE module for the prebuilt archive. Its kernel objects are already included. Source-only backend and SPEED/SIZE options do not configure a prebuilt library. The module supplies `TF_LITE_STATIC_MEMORY` and applies `-fshort-enums`. Confirm that the selected archive was produced with matching enum-width and floating-point ABI settings; a successful link alone does not establish matching tensor layouts.

## Add inference, build and run

Implement `src/main.cpp` and `src/model_data.cpp` using [First inference](/helia-rt/getting-started/first-inference/). Select a board identifier supported by your Zephyr checkout:

```bash
west build -p always -b <board> -s app/helia_rt_app -d build/helia_rt_app
west flash -d build/helia_rt_app
```

Replace `<board>` and the application path before running. Use the board's configured console to inspect allocation/invocation diagnostics and compare output against known test inputs. Board runners, console devices and compiler setup are owned by your Zephyr installation; changing the runtime backend does not configure them.

## Verify the integration

Inspect `build/zephyr/.config` for the backend and float features you intended. Check the link map for the selected runtime/kernel symbols, and test inference on the target. A link-only test cannot establish numerical correctness or optimized execution.

The [source module](https://github.com/AmbiqAI/helia-rt/tree/main/zephyr) and [prebuilt module template](https://github.com/AmbiqAI/helia-rt/tree/main/tensorflow/lite/micro/tools/ci_build/templates/zephyr_prebuilt/zephyr) are the source of truth for settings. See [Model compatibility](/helia-rt/guide/model-compatibility/) when a model fails to allocate or invoke.
