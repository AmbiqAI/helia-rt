# CMake Examples

Link a prebuilt heliaRT archive into a plain CMake project (no Zephyr, no neuralSPOT).

## Prerequisites

- A cross-compilation toolchain (GCC, armclang, or ATfE) on your `PATH`
- A heliaRT release bundle extracted (see [Static vs Source](../guides/static-vs-source.md))

## Minimal CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 17)

# Point at the extracted heliaRT release bundle
set(HELIA_RT_DIR ${CMAKE_SOURCE_DIR}/third_party/helia-rt)
set(HELIA_RT_ARCH "cortex-m55")      # or cortex-m4+fp
set(HELIA_RT_TOOLCHAIN "atfe")       # or gcc, armclang
set(HELIA_RT_BUILD "release")        # or debug, release_with_logs

# Include paths
include_directories(
    ${HELIA_RT_DIR}/include
    ${HELIA_RT_DIR}/include/third_party/flatbuffers/include
    ${HELIA_RT_DIR}/include/third_party/gemmlowp
    ${HELIA_RT_DIR}/include/third_party/ruy
)

# Prebuilt library
add_library(helia_rt STATIC IMPORTED)
set_target_properties(helia_rt PROPERTIES
    IMPORTED_LOCATION
    ${HELIA_RT_DIR}/${HELIA_RT_ARCH}/${HELIA_RT_TOOLCHAIN}/${HELIA_RT_BUILD}/libtensorflow-microlite.a
)

# Your application
add_executable(my_app src/main.cc)
target_link_libraries(my_app PRIVATE helia_rt)
```

## Building

```bash
cmake -B build \
    -DCMAKE_TOOLCHAIN_FILE=toolchain-cortex-m55.cmake \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Minimal Application

```cpp
// src/main.cc
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/helia_rt_version.h"

#include "model_data.h"  // your .tflite as a C array

constexpr int kArenaSize = 32 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];

int main() {
    tflite::InitializeTarget();

    const tflite::Model* model = tflite::GetModel(g_model_data);

    tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddMaxPool2D();

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kArenaSize);
    interpreter.AllocateTensors();

    // Fill input, invoke, read output ...
    interpreter.Invoke();

    return 0;
}
```

## Next Steps

- [Source Builds](../getting-started/source.md) — building from source instead of prebuilt
- [Static vs Source](../guides/static-vs-source.md) — choosing the right distribution form
- [SPEED vs SIZE](../guides/speed-vs-size.md) — HELIA source-build kernel profiles
