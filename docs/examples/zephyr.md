# :material-chip: Zephyr + heliaRT

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
- **Zephyr SDK:** zephyr-sdk-0.17.4

!!! note

    Do not add both the source-module and prebuilt-release variants to the same app.

=== "Source Modules + CMSIS-NN"

    ## 1. Fetch the default `cmsis-nn` module

    In a standard Zephyr west workspace, open `cmsis-nn` is already provided by the upstream manifest at `modules/lib/cmsis-nn`.

    Fetch it with west:

    ```bash
    west update cmsis-nn
    ```

    ## 2. Add the local `helia-rt` source copy

    Download the raw `helia-rt` source archive from the Ambiq content portal and copy it into `modules/`:

    ```bash
    cp -r <download-dir>/helia-rt <ws>/modules/
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

    ## 3. Create the application

    `CMakeLists.txt`

    ```cmake
    cmake_minimum_required(VERSION 3.20.0)

    list(APPEND ZEPHYR_EXTRA_MODULES
      ${CMAKE_CURRENT_SOURCE_DIR}/../../modules/helia-rt
    )

    find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
    project(helia_rt_app)

    target_sources(app PRIVATE src/main.cpp)
    ```

    `prj.conf`

    ```conf
    CONFIG_CPP=y
    CONFIG_STD_CPP17=y
    CONFIG_FPU=y

    CONFIG_PRINTK=y
    CONFIG_CONSOLE=y
    CONFIG_UART_CONSOLE=y

    CONFIG_CMSIS_NN=y
    CONFIG_CMSIS_NN_CONVOLUTION=y
    CONFIG_CMSIS_NN_FULLYCONNECTED=y
    CONFIG_HELIA_RT=y
    CONFIG_HELIA_RT_BACKEND_CMSIS_NN=y
    ```

    Required heliaRT-specific settings:

    - `CONFIG_CMSIS_NN=y`
    - `CONFIG_HELIA_RT=y`
    - `CONFIG_HELIA_RT_BACKEND_CMSIS_NN=y`

    Notes:

    - Add only `helia-rt` to `ZEPHYR_EXTRA_MODULES` on this path.
    - Do not add open `cmsis-nn` to `ZEPHYR_EXTRA_MODULES` when it already comes from the standard west workspace.
    - In the default Zephyr workspace layout, open `cmsis-nn` is discovered at `modules/lib/cmsis-nn`.
    - Open `cmsis-nn` does not provide an `ALL` Kconfig switch in Zephyr. You must enable the CMSIS-NN kernel groups your model needs.

    Do not enable HELIA-only settings on this path:

    - `CONFIG_NS_CMSIS_NN`
    - `CONFIG_HELIA_RT_BACKEND_HELIA`

=== "Source Modules + HELIA"

    ## 1. Download the sources

    Download the raw source archives for both `helia-rt` and `ns-cmsis-nn` from the Ambiq content portal.

    After extracting them, copy the repositories into `modules/`:

    ```bash
    cp -r <download-dir>/helia-rt <ws>/modules/
    cp -r <download-dir>/ns-cmsis-nn <ws>/modules/
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

    list(APPEND ZEPHYR_EXTRA_MODULES
      ${CMAKE_CURRENT_SOURCE_DIR}/../../modules/helia-rt
      ${CMAKE_CURRENT_SOURCE_DIR}/../../modules/ns-cmsis-nn
    )

    find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
    project(helia_rt_app)

    target_sources(app PRIVATE src/main.cpp)
    ```

    `prj.conf`

    ```conf
    CONFIG_CPP=y
    CONFIG_STD_CPP17=y
    CONFIG_FPU=y

    CONFIG_PRINTK=y
    CONFIG_CONSOLE=y
    CONFIG_UART_CONSOLE=y

    CONFIG_HELIA_RT=y
    CONFIG_NS_CMSIS_NN=y
    CONFIG_HELIA_RT_BACKEND_HELIA=y
    ```

    Required heliaRT-specific settings:

    - `CONFIG_HELIA_RT=y`
    - `CONFIG_NS_CMSIS_NN=y`
    - `CONFIG_HELIA_RT_BACKEND_HELIA=y`

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

    target_sources(app PRIVATE src/main.cpp)
    ```

    `prj.conf`

    ```conf
    CONFIG_CPP=y
    CONFIG_STD_CPP17=y
    CONFIG_FPU=y

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

    Do not enable these source-module-only settings with the prebuilt bundle:

    - `CONFIG_NS_CMSIS_NN`
    - `CONFIG_HELIA_RT_BACKEND_HELIA`

    Notes:

    - Do not add `modules/ns-cmsis-nn` to `ZEPHYR_EXTRA_MODULES` for the prebuilt bundle.
    - Enable `CONFIG_FPU=y` when using the prebuilt `cm55` archive. The published Cortex-M55 prebuilts use hard-float calling conventions.
    - The prebuilt Zephyr module also forces `TF_LITE_STATIC_MEMORY` for the application build. That is required so your app sees the same `TfLiteTensor` layout as the prebuilt archive.
    - The prebuilt Zephyr module supports Cortex-M55, and Cortex-M4 with FPU.
    - The prebuilt archive selection is automatic from board CPU, toolchain, and selected flavor.

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

`src/main.cpp`

```cpp
#include <cstdint>

#include <zephyr/kernel.h>
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
    k_msleep(1000);
  }

  return 0;
}
```

## 4. Build

Example build for Apollo510 EVB:

```bash
west build -p always -b apollo510_evb -s app/helia_rt_app -d build/helia_rt_app
```

## 5. Flash

```bash
west flash -d build/helia_rt_app
```

## 6. View logs

If UART console is enabled, open the board serial port at `115200 8N1`.

Example:

```bash
screen /dev/cu.usbmodemXXXX 115200
```

## 7. Checklist

Source modules + CMSIS-NN:

- `modules/helia-rt` exists
- `modules/lib/cmsis-nn` exists
- `modules/helia-rt` is listed in `ZEPHYR_EXTRA_MODULES`
- `CONFIG_CMSIS_NN=y` is enabled
- `CONFIG_HELIA_RT=y` is enabled
- `CONFIG_HELIA_RT_BACKEND_CMSIS_NN=y` is enabled
- no HELIA-only Kconfig options are enabled

Source modules + HELIA:

- `modules/helia-rt` exists
- `modules/ns-cmsis-nn` exists
- both module paths are listed in `ZEPHYR_EXTRA_MODULES`
- `CONFIG_HELIA_RT=y` is enabled
- `CONFIG_NS_CMSIS_NN=y` is enabled
- `CONFIG_HELIA_RT_BACKEND_HELIA=y` is enabled

Prebuilt release module:

- `modules/helia-rt-heliaRT-v1.10.1` exists
- only that path is listed in `ZEPHYR_EXTRA_MODULES`
- `CONFIG_HELIA_RT=y` is enabled
- `CONFIG_FPU=y` is enabled for Cortex-M55 builds
- no source-backend Kconfig options are enabled

For the broader setup guide, including `Reference`, open `CMSIS-NN`, and prebuilt flows, see [Zephyr setup](../usage/zephyr.md).
