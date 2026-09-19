---
title: Link a prebuilt archive
description: Consume a released heliaRT archive from a custom CMake firmware project.
---

Use a prebuilt archive when its architecture, compiler and build flavor match your application. For configurable kernel profiles or float features, use [source integration](/helia-rt/getting-started/source/).

## Inspect the release bundle

Download and extract the bundle from [heliaRT releases](https://github.com/AmbiqAI/helia-rt/releases). Read its `MANIFEST.txt` and retain the version and commit identity with your firmware dependencies. The bundle layout is:

```text
helia-rt-.../
├── lib/libhelia-rt-cm55-gcc-release-with-logs.a
├── tensorflow/
├── third_party/
├── zephyr/
└── MANIFEST.txt
```

Archives use `cm4` or `cm55` architecture tokens, `gcc`, `armclang` or `atfe` compiler tokens, and `debug`, `release-with-logs` or `release` flavor tokens. Choose an actual file in the bundle. The `cm4` artifacts target Cortex-M4 with an FPU; the Cortex-M55 float kernels require the corresponding MVE floating-point target configuration.

The bundle contains runtime and HELIA kernel objects in each archive. Do not add another heliaCORE archive for the same symbols. Header paths start at the bundle root, not an `include/` subdirectory. These contracts come from the [packager](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/ci_build/package_helia_bundle.sh) and [prebuilt module](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/ci_build/templates/zephyr_prebuilt/zephyr/CMakeLists.txt).

## Define an imported target

Add this to an existing firmware CMake project after `project()`. The example selects the GCC Cortex-M55 archive retaining logs:

```cmake
set(HELIA_RT_BUNDLE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/helia-rt")
set(HELIA_RT_ARCHIVE
    "${HELIA_RT_BUNDLE}/lib/libhelia-rt-cm55-gcc-release-with-logs.a")
if(NOT EXISTS "${HELIA_RT_ARCHIVE}")
  message(FATAL_ERROR "Selected heliaRT archive is missing")
endif()

add_library(helia_rt_prebuilt STATIC IMPORTED)
set_target_properties(helia_rt_prebuilt PROPERTIES
  IMPORTED_LOCATION "${HELIA_RT_ARCHIVE}"
  INTERFACE_INCLUDE_DIRECTORIES
    "${HELIA_RT_BUNDLE};${HELIA_RT_BUNDLE}/third_party/flatbuffers/include;${HELIA_RT_BUNDLE}/third_party/gemmlowp;${HELIA_RT_BUNDLE}/third_party/ruy;${HELIA_RT_BUNDLE}/third_party/kissfft;${HELIA_RT_BUNDLE}/third_party/ns_cmsis_nn/Include"
  INTERFACE_COMPILE_DEFINITIONS "TF_LITE_STATIC_MEMORY"
)
target_link_libraries(my_firmware PRIVATE helia_rt_prebuilt)
```

Replace `my_firmware` with your existing application target. Its toolchain file and board support must already supply startup code, linker placement, CPU/FPU flags and compatible C/C++ runtime libraries. Match the archive's floating-point ABI and compiler ABI, including enum sizing. The generated Zephyr prebuilt integration applies `-fshort-enums`; consult the selected compiler's [release build flags](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/make/targets/cortex_m_generic_makefile.inc) when integrating outside Zephyr.

:::caution[Keep the tensor layout consistent]
`TF_LITE_STATIC_MEMORY` affects public tensor structures. Compile application code including LiteRT headers with the same definition as the archive. Use the headers delivered with that archive, rather than headers from another runtime checkout.
:::

## Build and check inference

Configure your firmware with its existing cross-toolchain file, then build:

```bash
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/board-toolchain.cmake
cmake --build build
```

Use [First inference](/helia-rt/getting-started/first-inference/) for checked model loading, registration, allocation and invocation. Check known inputs and expected outputs on your target before relying on the integration.

If linking fails, first check the selected archive and its symbols, header version, ABI settings, and platform logging dependencies. If allocation fails, inspect model compatibility and diagnostics before increasing the arena. Prebuilt float support cannot be changed with application-only compiler definitions.
