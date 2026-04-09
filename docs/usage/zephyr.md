# :material-chip: Zephyr Setup

This guide covers the two supported Zephyr integration paths for heliaRT:

- raw/source module
- prebuilt release bundle

Use the raw module when you want source visibility and maximum flexibility. Use the prebuilt bundle when you want the fastest setup on a supported configuration.

## Workspace Expectations

This guide assumes a west workspace with:

- `zephyr/` for the Zephyr repository
- `modules/` for external Zephyr modules
- `app/` for your application

Expected layout for the public source path:

```plaintext
<ws>/
├── zephyr/
│   ├── west.yml
│   └── ...
├── modules/
│   ├── helia-rt/
│   └── cmsis-nn/               # raw module path when using open CMSIS-NN
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

## 1. Install Prerequisites

Install Zephyr and Python dependencies using the official Zephyr getting-started flow, then install the Zephyr SDK.

References:

- [Zephyr getting started](https://docs.zephyrproject.org/latest/develop/getting_started/index.html)
- [Zephyr application development](https://docs.zephyrproject.org/latest/develop/application/index.html#application)

## 2. Create or Enter a West Workspace

- create a workspace with `west init` and `west update`, or enter an existing one
- confirm `zephyr/` exists at the workspace root
- run `west update` after editing `zephyr/west.yml`

## 3. Choose a Module Style

### Raw Module

Use the raw repository when you want:

- source-based integration
- the ability to debug or modify heliaRT internals
- configurations outside the published prebuilt matrix
- a public-safe path that does not require HELIA acceleration sources

Clone the repository into `modules/`:

```bash
git clone https://github.com/AmbiqAI/helia-rt <ws>/modules/helia-rt
```

The raw/source module supports three backend choices:

- `Reference`: generic TFLM kernels
- `CMSIS-NN`: open-source Arm CMSIS-NN backend
- `HELIA`: Ambiq-optimized backend provided through a separate Ambiq-distributed module

On Cortex-M builds with an enabled open CMSIS-NN module, `CMSIS-NN` is the default raw-module backend. Otherwise `Reference` is the default.

Cloning `helia-rt` alone gives you a working source integration path, but it does not include HELIA acceleration. The public source path uses `Reference` or open `CMSIS-NN`. To use `HELIA`, you need access to Ambiq's private backend module, currently distributed as `ns-cmsis-nn`.

### Prebuilt Release Bundle

Use the prebuilt bundle when you want:

- the fastest path to a working Zephyr module
- a release-pinned package from GitHub releases
- Ambiq-optimized kernels already built into the archive

Download and extract the GitHub release asset:

```bash
unzip zephyr-helia-rt-<tag>.zip -d <ws>/modules
mv <ws>/modules/zephyr-helia-rt-<tag> <ws>/modules/helia-rt
```

With the prebuilt bundle, HELIA acceleration is already embedded in the archive, so you do not add a separate private backend module.

Initial supported prebuilt matrix:

- `cortex-m4+fp` and `cortex-m55`
- `gcc` and `armclang`
- `debug`, `release`, and `release_with_logs`

## 4. Choose a Raw-Module Backend

### `Reference`

Use `Reference` when you want the simplest source path and do not need an optimized Cortex-M backend.

Example `prj.conf`:

```conf
CONFIG_HELIA_RT=y
CONFIG_HELIA_RT_BACKEND_REFERENCE=y
```

### `CMSIS-NN`

Use `CMSIS-NN` when you want the public source path with open-source Arm optimized kernels.

For this backend, add an open CMSIS-NN Zephyr module to the workspace.

Example `zephyr/west.yml` snippet:

```yaml
remotes:
  - name: arm-software
    url-base: https://github.com/ARM-software

projects:
  - name: cmsis-nn
    remote: arm-software
    repo-path: CMSIS-NN
    path: modules/cmsis-nn
    revision: main
```

Then run:

```bash
west update
```

Example app-local copy:

```bash
cp -r <path-to-cmsis-nn> <ws>/modules/cmsis-nn
```

Example `prj.conf`:

```conf
CONFIG_HELIA_RT=y
CONFIG_HELIA_RT_BACKEND_CMSIS_NN=y
```

### `HELIA`

Use `HELIA` when you want Ambiq's private accelerated backend from the raw/source path.

HELIA acceleration is not included in the public `helia-rt` repository. To use it:

1. Reach out to Ambiq to get access to the HELIA backend package.
2. Obtain the package currently distributed as `ns-cmsis-nn`.
3. Place that module in your Zephyr workspace.
4. Toggle the HELIA backend in `prj.conf`.

Assume the private module exposes the necessary Zephyr `module.yml` and CMake integration and is available at `<ws>/modules/ns-cmsis-nn`.

Example `zephyr/west.yml` snippet:

```yaml
remotes:
  - name: ambiqai
    url-base: git@github.com:AmbiqAI

projects:
  - name: ns-cmsis-nn
    remote: ambiqai
    repo-path: ns-cmsis-nn
    path: modules/ns-cmsis-nn
    revision: main
```

Then run:

```bash
west update
```

For reproducibility, pin `revision` to a tag or commit SHA instead of `main`.

Example app-local copy:

```bash
cp -r <path-to-ns-cmsis-nn> <ws>/modules/ns-cmsis-nn
```

Example `prj.conf`:

```conf
CONFIG_HELIA_RT=y
CONFIG_HELIA_RT_BACKEND_HELIA=y
```

## 5. Choose Module Discovery

Zephyr can discover modules either through the west manifest or through `ZEPHYR_EXTRA_MODULES`.

### West-Managed Modules

This is usually best for CI and stable, pinned workspaces.

### `ZEPHYR_EXTRA_MODULES`

This is often the easiest path when:

- you are iterating locally
- you want app-local control over module paths
- you are using a downloaded prebuilt bundle
- you are switching between `Reference`, `CMSIS-NN`, and `HELIA` source backends

## 6. Create the Application

Application layout:

```plaintext
<ws>/app/helia_rt_app/
├── CMakeLists.txt
├── prj.conf
└── src/
    └── main.cpp
```

### `CMakeLists.txt`

Raw module:

```cmake
cmake_minimum_required(VERSION 3.20.0)

list(APPEND ZEPHYR_EXTRA_MODULES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../modules/helia-rt
)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(helia_rt_app)

target_sources(app PRIVATE src/main.cpp)
```

Raw module with open `CMSIS-NN`:

```cmake
cmake_minimum_required(VERSION 3.20.0)

list(APPEND ZEPHYR_EXTRA_MODULES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../modules/helia-rt
)
list(APPEND ZEPHYR_EXTRA_MODULES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../modules/cmsis-nn
)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(helia_rt_app)

target_sources(app PRIVATE src/main.cpp)
```

Raw module with `HELIA`:

```cmake
cmake_minimum_required(VERSION 3.20.0)

list(APPEND ZEPHYR_EXTRA_MODULES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../modules/helia-rt
)
list(APPEND ZEPHYR_EXTRA_MODULES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../modules/ns-cmsis-nn
)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(helia_rt_app)

target_sources(app PRIVATE src/main.cpp)
```

Prebuilt bundle:

```cmake
cmake_minimum_required(VERSION 3.20.0)

list(APPEND ZEPHYR_EXTRA_MODULES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../modules/helia-rt
)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(helia_rt_app)

target_sources(app PRIVATE src/main.cpp)
```

### `prj.conf`

Raw module:

```conf
CONFIG_FPU=y
CONFIG_PRINTK=y
CONFIG_CONSOLE=y
CONFIG_UART_CONSOLE=y

CONFIG_HELIA_RT=y
CONFIG_HELIA_RT_BACKEND_CMSIS_NN=y
```

Raw module with `Reference`:

```conf
CONFIG_FPU=y
CONFIG_PRINTK=y
CONFIG_CONSOLE=y
CONFIG_UART_CONSOLE=y

CONFIG_HELIA_RT=y
CONFIG_HELIA_RT_BACKEND_REFERENCE=y
```

Raw module with `HELIA`:

```conf
CONFIG_FPU=y
CONFIG_PRINTK=y
CONFIG_CONSOLE=y
CONFIG_UART_CONSOLE=y

CONFIG_HELIA_RT=y
CONFIG_HELIA_RT_BACKEND_HELIA=y
```

Prebuilt bundle:

```conf
CONFIG_FPU=y
CONFIG_PRINTK=y
CONFIG_CONSOLE=y
CONFIG_UART_CONSOLE=y

CONFIG_HELIA_RT=y
CONFIG_HELIA_RT_PREBUILT_BUILD_RELEASE=y
```

Use `CONFIG_HELIA_RT_PREBUILT_BUILD_DEBUG=y` or `CONFIG_HELIA_RT_PREBUILT_BUILD_RELEASE_WITH_LOGS=y` when you need another published prebuilt flavor.

For the raw/source module:

- `Reference` works everywhere
- `CMSIS-NN` is the recommended public Cortex-M backend when an open CMSIS-NN module is present
- `HELIA` requires the Ambiq-provided private backend module

### `src/main.cpp`

The interpreter flow remains familiar if you already know TFLM:

```cpp
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <zephyr/sys/printk.h>

int main(void) {
  printk("heliaRT application startup\n");
  return 0;
}
```

## 7. Build

```bash
west build -p always -b <board> -s <ws>/app/helia_rt_app -d <ws>/out/build
```

Example:

```bash
west build -p always -b apollo510_evb -s <ws>/app/helia_rt_app -d <ws>/out/build
```

## 8. Flash

```bash
west flash --build-dir <ws>/out/build
```

To view logs, connect to the board console using your preferred serial terminal.

## 9. Integration Checks

- confirm `<ws>/modules/helia-rt/zephyr/module.yml` exists
- if using the raw module with `CMSIS-NN`, confirm `<ws>/modules/cmsis-nn` exists
- if using the raw module with `HELIA`, confirm `<ws>/modules/ns-cmsis-nn` exists
- verify `ZEPHYR_EXTRA_MODULES` paths if the module is not discovered
- if using the prebuilt bundle, ensure the selected prebuilt flavor matches arch, toolchain, and build type

## Raw vs Prebuilt

| Option | Advantages | Tradeoffs |
| --- | --- | --- |
| Raw module | source-visible, flexible, public-safe with `Reference` or open `CMSIS-NN` | HELIA acceleration requires a separate Ambiq-provided module |
| Prebuilt bundle | fast setup, release-pinned, HELIA acceleration already embedded | limited to the published prebuilt matrix |

For examples built around these flows, see [Zephyr examples](../examples/zephyr.md).
