# :material-chip: Zephyr + heliaRT

!!! note "Prerequisite"
    This guide assumes you already have a working Zephyr development environment
    (Zephyr repo, west, SDK). If not, follow the
    [Zephyr Getting Started Guide](https://docs.zephyrproject.org/latest/develop/getting_started/index.html)
    first.

This example shows the minimum process to create, build, and run a Zephyr application that uses heliaRT.

Use one integration path or the other:

- source modules: `helia-rt` + open `cmsis-nn`
- source modules: `helia-rt` + `ns-cmsis-nn`
- prebuilt release module: a single `helia-rt` archive with heliaRT + `ns-cmsis-nn` already linked in

This guide assumes a workspace like:

```plaintext
<ws>/
├── zephyr/
├── modules/
└── app/
    └── helia_rt_app/
        ├── CMakeLists.txt
        ├── prj.conf
        └── src/
            └── main.cpp
```

Known-good versions:

- **Zephyr:** 4.3
- **Zephyr SDK:** zephyr-sdk-1.0.1

!!! note

    Do not add both the source-module and prebuilt-release variants to the same app.

=== "Source Modules + CMSIS-NN"

    ## 1. Fetch the modules

    Open `cmsis-nn` is already provided by the standard Zephyr west manifest
    at `modules/lib/cmsis-nn`, so you only need to add `helia-rt` to your
    workspace `west.yml` projects list. (If your workspace does not inherit
    the standard manifest, add `cmsis-nn` from `github.com/zephyrproject-rtos/cmsis-nn`
    as a second project with `path: modules/lib/cmsis-nn`.)

    Add `helia-rt`:

    ```yaml
    - name: helia-rt
      url: https://github.com/AmbiqAI/helia-rt
      revision: <helia-rt-version>   # e.g. helia-rt-v1.16.0
      path: modules/helia-rt
    ```

    Then fetch modules:

    !!! note
        If this is your first time setting up the workspace, run `west update`
        to fetch all modules. If you already have a workspace and are just
        adding these entries, `west update helia-rt ns-cmsis-nn` fetches only
        the new modules without re-downloading everything else.

    ```bash
    west update helia-rt cmsis-nn
    ```

    Result:

    ```plaintext
    <ws>/
    ├── zephyr/
    ├── modules/
    │   ├── helia-rt/
    │   └── lib/
    │       └── cmsis-nn/
    └── app/
        └── helia_rt_app/
    ```

    ## 2. Create the application

    `CMakeLists.txt`

    ```cmake
    cmake_minimum_required(VERSION 3.20.0)

    find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
    project(helia_rt_app)

    set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
    zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

    target_sources(app PRIVATE src/main.cpp)
    ```

    `prj.conf`

    ```conf
    CONFIG_STD_CPP17=y

    CONFIG_PRINTK=y
    CONFIG_CONSOLE=y
    CONFIG_UART_CONSOLE=y

    CONFIG_HELIA_RT=y
    CONFIG_NS_CMSIS_NN=n
    CONFIG_CMSIS_NN=y
    CONFIG_CMSIS_NN_CONVOLUTION=y
    CONFIG_CMSIS_NN_FULLYCONNECTED=y
    ```

    Required heliaRT-specific settings:

    - `CONFIG_HELIA_RT=y`
    - `CONFIG_NS_CMSIS_NN=n` (disable the HELIA backend that heliaRT enables by default)
    - `CONFIG_CMSIS_NN=y` and per-op kernel configs for your model

    Notes:

    - With west-managed modules, no `ZEPHYR_EXTRA_MODULES` is needed.
    - Do not add open `cmsis-nn` to `ZEPHYR_EXTRA_MODULES` when it already comes from the standard west workspace.
    - In the default Zephyr workspace layout, open `cmsis-nn` is discovered at `modules/lib/cmsis-nn`.
    - Open `cmsis-nn` does not provide an `ALL` Kconfig switch in Zephyr. You must enable the CMSIS-NN kernel groups your model needs.

    Do not enable HELIA-only settings on this path:

    - `CONFIG_NS_CMSIS_NN`

=== "Source Modules + HELIA"

    ## 1. Fetch the modules

    Add both `helia-rt` and `ns-cmsis-nn` as west projects in your workspace's `west.yml`:

    ```yaml
    - name: helia-rt
      url: https://github.com/AmbiqAI/helia-rt
      revision: <helia-rt-version>     # e.g. helia-rt-v1.16.0
      path: modules/helia-rt

    - name: ns-cmsis-nn
      url: https://github.com/AmbiqAI/ns-cmsis-nn
      revision: <ns-cmsis-nn-version>  # e.g. v7.25.0
      path: modules/ns-cmsis-nn
    ```

    Then fetch both modules:

    ```bash
    west update helia-rt ns-cmsis-nn
    ```

    Result:

    ```plaintext
    <ws>/
    ├── zephyr/
    ├── modules/
    │   ├── helia-rt/
    │   └── ns-cmsis-nn/
    └── app/
        └── helia_rt_app/
    ```

    ## 2. Create the application

    `CMakeLists.txt`

    ```cmake
    cmake_minimum_required(VERSION 3.20.0)

    find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
    project(helia_rt_app)

    set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
    zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

    target_sources(app PRIVATE src/main.cpp)
    ```

    `prj.conf`

    ```conf
    CONFIG_STD_CPP17=y

    CONFIG_PRINTK=y
    CONFIG_CONSOLE=y
    CONFIG_UART_CONSOLE=y

    CONFIG_HELIA_RT=y
    ```

    Required heliaRT-specific settings:

    - `CONFIG_HELIA_RT=y`

    Optional HELIA kernel profile:

    - `CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SPEED=y`
      This is the default.
    - `CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SIZE=y`

=== "Prebuilt Release Module"

    ## 1. Download the prebuilt release

    Download the prebuilt heliaRT release archive from the Ambiq content portal.

    This bundle already contains the HELIA runtime and the `ns-cmsis-nn` kernel implementation inside the static archive.

    After extracting it, copy the bundle into `modules/`:

    ```bash
    cp -r <download-dir>/helia-rt-m55-release <ws>/modules/
    ```

    Result:

    ```plaintext
    <ws>/
    ├── zephyr/
    ├── modules/
    │   └── helia-rt-m55-release/
    └── app/
        └── helia_rt_app/
    ```

    ## 2. Create the application

    `CMakeLists.txt`

    Only add the prebuilt bundle as a Zephyr module:

    ```cmake
    cmake_minimum_required(VERSION 3.20.0)

    list(APPEND ZEPHYR_EXTRA_MODULES
      ${CMAKE_CURRENT_SOURCE_DIR}/../../modules/helia-rt-m55-release
    )

    find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
    project(helia_rt_app)

    set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
    zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

    target_sources(app PRIVATE src/main.cpp)
    ```

    `prj.conf`

    ```conf
    CONFIG_STD_CPP17=y

    CONFIG_PRINTK=y
    CONFIG_CONSOLE=y
    CONFIG_UART_CONSOLE=y

    CONFIG_HELIA_RT=y
    ```

    Optional prebuilt flavor selection:

    - `CONFIG_HELIA_RT_PREBUILT_BUILD_RELEASE=y`
      This is the default.
    - `CONFIG_HELIA_RT_PREBUILT_BUILD_DEBUG=y`
    - `CONFIG_HELIA_RT_PREBUILT_BUILD_RELEASE_WITH_LOGS=y`

    These options select the prebuilt archive's build flavor, not the HELIA
    kernel SPEED/SIZE profile. Current prebuilt release bundles do not publish
    separate SPEED/SIZE kernel-profile archives; use the source-module path if
    you need to choose `CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SPEED` or
    `CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SIZE`.

    Do not enable these source-module-only settings with the prebuilt bundle:

    - `CONFIG_NS_CMSIS_NN`

    Notes:

    - Do not add `modules/ns-cmsis-nn` to `ZEPHYR_EXTRA_MODULES` for the prebuilt bundle.
    - Enable `CONFIG_FPU=y` when using the prebuilt `cm55` archive. The published Cortex-M55 prebuilts use hard-float calling conventions.
    - The prebuilt Zephyr module also forces `TF_LITE_STATIC_MEMORY` for the application build. That is required so your app sees the same `TfLiteTensor` layout as the prebuilt archive.
    - The prebuilt Zephyr module supports Cortex-M55, and Cortex-M4 with FPU.
    - The prebuilt archive selection is automatic from board CPU, toolchain, and selected flavor.

!!! tip "Using Reference kernels"

    To use generic Reference kernels instead of an accelerated backend, suppress the
    `ns-cmsis-nn` auto-imply in your `prj.conf`:

    ```conf
    CONFIG_STD_CPP17=y
    CONFIG_HELIA_RT=y
    CONFIG_NS_CMSIS_NN=n
    ```

    No `ns-cmsis-nn` or `cmsis-nn` module is needed. The Reference backend is selected
    automatically when neither is active.

## 3. Minimal bring-up

After wiring the module and `prj.conf`, the smallest useful app flow is:

- embed a `.tflite` flatbuffer as a C array
- map it with `tflite::GetModel()`
- register the ops your model uses
- allocate tensors
- write input data
- call `Invoke()`
- read the output tensor

The example below is intentionally small, but it is a real inference flow.
It assumes:

- one embedded model in `g_model`
- one `int8` input tensor
- one `int8` output tensor
- a model that uses `FULLY_CONNECTED`

If your model uses different operators or tensor types, change the resolver and
tensor access accordingly.

`src/model_data.h`

```c
extern const unsigned char g_model[];
extern const int g_model_len;
```

`src/model_data.cpp`

```c
// Convert your .tflite file into a C array, for example:
// xxd -i model.tflite > model_data.cpp
```

Then add `src/model_data.cpp` to `CMakeLists.txt`:

```cmake
target_sources(app PRIVATE src/main.cpp src/model_data.cpp)
```

`src/main.cpp`

```cpp
#include <cstdint>

#include <zephyr/sys/printk.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_data.h"

namespace {

constexpr int kTensorArenaSize = 96 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

int TensorElementCount(const TfLiteTensor* tensor) {
  int count = 1;
  for (int i = 0; i < tensor->dims->size; ++i) {
    count *= tensor->dims->data[i];
  }
  return count;
}

}  // namespace

int main() {
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printk("Model schema mismatch: %d != %d\n",
           model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddFullyConnected();

  tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printk("AllocateTensors failed\n");
    return 1;
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  if (input == nullptr || output == nullptr) {
    printk("Missing input or output tensor\n");
    return 1;
  }

  const int input_count = TensorElementCount(input);
  for (int i = 0; i < input_count; ++i) {
    input->data.int8[i] = static_cast<int8_t>(input->params.zero_point);
  }

  if (interpreter.Invoke() != kTfLiteOk) {
    printk("Invoke failed\n");
    return 1;
  }

  const int output_count = TensorElementCount(output);
  const int preview = output_count < 8 ? output_count : 8;
  for (int i = 0; i < preview; ++i) {
    printk("out[%d] = %d\n", i, output->data.int8[i]);
  }

  while (true) {

  }

  return 0;
}
```

## 4. Build

The examples below use Apollo510 EVB; substitute your board and app source path as needed.

=== "GCC (default)"

    GCC is the Zephyr default. No extra flags are required when `ZEPHYR_TOOLCHAIN_VARIANT` is unset or set to `zephyr`.

    ```bash
    west build -p always -b apollo510_evb \
      -s app/helia_rt_app -d build/helia_rt_app_gcc
    ```

    If you installed the Arm GNU Toolchain separately (outside the Zephyr SDK), set the variant explicitly:

    ```bash
    west build -p always -b apollo510_evb \
      -s app/helia_rt_app -d build/helia_rt_app_gcc \
      -- -DZEPHYR_TOOLCHAIN_VARIANT=gnuarmemb \
         -DGNUARMEMB_TOOLCHAIN_PATH=/path/to/gcc-arm-none-eabi
    ```

=== "ATfE (recommended)"

    [ATfE](https://github.com/arm/arm-toolchain) (Arm Toolchain for Embedded) is LLVM-based and open-source.
    On Cortex-M55 + Helium workloads it produces code that is **up to 25 % more efficient**[^atfe-bench] than GCC — fewer cycles *and* more inferences per Joule.

    Point `LLVM_TOOLCHAIN_PATH` at the ATfE install root:

    ```bash
    west build -p always -b apollo510_evb \
      -s app/helia_rt_app -d build/helia_rt_app_atfe \
      -- -DZEPHYR_TOOLCHAIN_VARIANT=host \
         -DTOOLCHAIN_VARIANT_COMPILER=llvm \
         -DLLVM_TOOLCHAIN_PATH=/path/to/ATfE-<version> \
         -DCONFIG_LLVM_USE_LLD=y \
         -DCONFIG_COMPILER_RT_RTLIB=y
    ```

    | Flag | Purpose |
    |---|---|
    | `-DZEPHYR_TOOLCHAIN_VARIANT=host` | Select the host toolchain variant |
    | `-DTOOLCHAIN_VARIANT_COMPILER=llvm` | Use LLVM/Clang as the compiler within the host variant |
    | `-DLLVM_TOOLCHAIN_PATH=...` | Root of the ATfE installation (contains `bin/`, `lib/`, …) |
    | `-DCONFIG_LLVM_USE_LLD=y` | Use LLD instead of GNU ld |
    | `-DCONFIG_COMPILER_RT_RTLIB=y` | Link compiler-rt instead of libgcc |

[^atfe-bench]:
    Measured across the [MLPerf Tiny v1.1](https://mlcommons.org/benchmarks/inference-tiny/) reference suite on the Apollo510 EVB (Cortex-M55 + Helium @ 192 MHz, 10 iterations) using heliaRT v1.13.1. Latency derived from PMU cycles; energy captured with a Joulescope. Compilers: ATfE 22.1 vs `arm-none-eabi-gcc` 14.2. Headline **"up to 25 %"** refers to the inferences-per-Joule improvement on Image Classification (ResNet, +24.4 %, rounded). Every model also ran with **lower latency** under ATfE (4 %–13 % fewer cycles) and **lower energy per inference** (6 %–20 %). See [Toolchains → Why ATfE](../guides/toolchains.md#why-atfe) for the full per-model table.

## 5. Flash

```bash
# Substitute the build directory you used in step 4
# (e.g. build/helia_rt_app_gcc or build/helia_rt_app_atfe)
west flash -d build/helia_rt_app_gcc
```

## 6. View logs

If UART console is enabled, open the board serial port at `115200 8N1`.

Example:

```bash
screen /dev/cu.usbmodemXXXX 115200
```

## 7. Checklist

Source modules + CMSIS-NN:

- `modules/helia-rt` exists (via `west update helia-rt`)
- `modules/lib/cmsis-nn` exists (via `west update cmsis-nn`)
- `CONFIG_HELIA_RT=y` is enabled
- `CONFIG_NS_CMSIS_NN=n` suppresses the auto-imply
- `CONFIG_CMSIS_NN=y` and per-op kernel configs are enabled
- no HELIA-only Kconfig options are enabled

Source modules + HELIA:

- `modules/helia-rt` exists (via `west update helia-rt`)
- `modules/ns-cmsis-nn` exists (via `west update ns-cmsis-nn`)
- `CONFIG_HELIA_RT=y` is enabled
- backend, CPP, FPU, and ns-cmsis-nn auto-configure

Prebuilt release module:

- `modules/helia-rt-v1.16.0` exists
- only that path is listed in `ZEPHYR_EXTRA_MODULES`
- `CONFIG_HELIA_RT=y` is enabled
- `CONFIG_FPU=y` is enabled for Cortex-M55 builds
- no source-backend Kconfig options are enabled

For the broader setup guide, including `HELIA`, open `CMSIS-NN`, and prebuilt flows, see [Zephyr setup](../getting-started/zephyr.md).
