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

Expected layout:

```plaintext
<ws>/
├── zephyr/
│   ├── west.yml
│   └── ...
├── modules/
│   ├── helia-rt/
│   └── ns-cmsis-nn/            # raw module path only
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

Clone the repository into `modules/`:

```bash
git clone https://github.com/AmbiqAI/helia-rt <ws>/modules/helia-rt
```

With the raw module, the Ambiq-optimized backend depends on the separate `ns-cmsis-nn` Zephyr module.

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

With the prebuilt bundle, `ns-cmsis-nn` is already embedded in the archive, so you do not add a separate `ns-cmsis-nn` module.

Initial supported prebuilt matrix:

- `cortex-m4+fp` and `cortex-m55`
- `gcc` and `armclang`
- `debug`, `release`, and `release_with_logs`

## 4. Add `ns-cmsis-nn` for the Raw Module

If you are using the raw module with the Ambiq-optimized backend, add `ns-cmsis-nn` to the workspace.

### West-Managed

Add `ns-cmsis-nn` to `zephyr/west.yml`:

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

### Local Copy

```bash
cp -r <path-to-ns-cmsis-nn> <ws>/modules/ns-cmsis-nn
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
CONFIG_HELIA_RT_BACKEND_NS_CMSIS_NN=y
CONFIG_NS_CMSIS_NN=y
CONFIG_NS_CMSIS_NN_ALL=y
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
- if using the raw module with the Ambiq backend, confirm `<ws>/modules/ns-cmsis-nn` exists
- verify `ZEPHYR_EXTRA_MODULES` paths if the module is not discovered
- if using the prebuilt bundle, ensure the selected prebuilt flavor matches arch, toolchain, and build type

## Raw vs Prebuilt

| Option | Advantages | Tradeoffs |
| --- | --- | --- |
| Raw module | source-visible, flexible, easier to debug | requires separate `ns-cmsis-nn` for the Ambiq backend |
| Prebuilt bundle | fast setup, release-pinned, fewer module dependencies | limited to the published prebuilt matrix |

For examples built around these flows, see [Zephyr examples](../examples/zephyr.md).
