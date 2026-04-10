# :material-hammer-wrench: Source Builds

Use this path when you want to build heliaRT directly from source and link it into an existing embedded application.

## Best For

- custom build systems
- explicit control over target, toolchain, and build type
- source-level debugging
- custom archive generation and packaging

## High-Level Flow

1. Clone the repository.
2. Select target architecture, toolchain, and build type.
3. Download required third-party dependencies.
4. Build the static library.
5. Link the resulting archive into your application.

## Basic Setup

```bash
git clone https://github.com/AmbiqAI/helia-rt
cd helia-rt

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

TARGET_ARCH=cortex-m55
TOOLCHAIN=gcc
BUILD_TYPE=release
TARGET=cortex_m_generic
OPTIMIZED_KERNEL=ambiq
```

## Download Dependencies

```bash
readable_run make -f tensorflow/lite/micro/tools/make/Makefile \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    TOOLCHAIN=${TOOLCHAIN} \
    third_party_downloads
```

## Build the Static Library

```bash
readable_run make -f tensorflow/lite/micro/tools/make/Makefile \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    TOOLCHAIN=${TOOLCHAIN} \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL} \
    BUILD_TYPE=${BUILD_TYPE} \
    microlite -j8
```

The generated static library will be placed under a `gen/.../lib/` output directory and can be linked into your application with the appropriate include paths and linker flags.

## Optional: Generate a Source Tree for IDE Use

```bash
python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  --makefile_options "TARGET=$TARGET TARGET_ARCH=$TARGET_ARCH OPTIMIZED_KERNEL_DIR=$OPTIMIZED_KERNEL" \
  "gen/${TARGET}_${TARGET_ARCH}_${BUILD_TYPE}_${OPTIMIZED_KERNEL}_${TOOLCHAIN}"
```

This generates a local source tree for intellisense, browsing, and debugging.

## Related Pages

- [Features](../features/index.md)
- [Zephyr setup](zephyr.md)
- [Examples](../examples/source.md)
