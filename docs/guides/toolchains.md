# Toolchains

heliaRT supports three toolchains for Cortex-M targets. All three are tested in CI across every release.

## Comparison

| Toolchain | ID | License | Typical Perf vs GCC | Best For |
|---|---|---|---|---|
| **GCC** (arm-none-eabi-gcc) | `gcc` | Open source | Baseline | Default, broadest availability |
| **Arm Compiler 6** (armclang) | `armclang` | Commercial | ~5–15 % faster | Keil MDK shops |
| **ATfE** (Arm Toolchain for Embedded) | `atfe` | **Open source** | **~10–20 % faster** | **Recommended** |

!!! success "Recommended: ATfE"
    ATfE is LLVM-based, fully open-source, and actively maintained by Arm. It produces measurably faster code than GCC on Cortex-M55 MVE workloads — without any licensing cost.

    [:octicons-link-external-16: ATfE on GitHub](https://github.com/nicowilliams/llvm-project-armfe){ target="_blank" }

## Installation

=== "GCC"

    ```bash
    # Ubuntu / Debian
    sudo apt install gcc-arm-none-eabi

    # macOS
    brew install --cask gcc-arm-embedded
    ```

    Or download from [Arm Developer](https://developer.arm.com/downloads/-/gnu-rm).

=== "armclang"

    Install [Keil MDK](https://www.keil.arm.com/) or [Arm Development Studio](https://developer.arm.com/Tools%20and%20Software/Arm%20Development%20Studio). `armclang` is included.

    Requires a commercial license.

=== "ATfE"

    Download from the [Arm Toolchain for Embedded releases](https://github.com/nicowilliams/llvm-project-armfe/releases):

    ```bash
    # Example (adjust version and host OS)
    wget https://github.com/.../atfe-<version>-linux-x86_64.tar.gz
    tar xzf atfe-*.tar.gz
    export PATH=$PWD/atfe/bin:$PATH
    ```

## Using with heliaRT

=== "Makefile"

    ```bash
    # GCC (default)
    make -f tensorflow/lite/micro/tools/make/Makefile \
        TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 \
        OPTIMIZED_KERNEL_DIR=helia TOOLCHAIN=gcc microlite

    # armclang
    make ... TOOLCHAIN=armclang microlite

    # ATfE
    make ... TOOLCHAIN=atfe microlite
    ```

=== "Prebuilt archive"

    Each release ships archives for all three toolchains:

    ```
    helia-rt-<tag>/cortex-m55/gcc/release/
    helia-rt-<tag>/cortex-m55/armclang/release/
    helia-rt-<tag>/cortex-m55/atfe/release/
    ```

    Link the `.a` that matches your project's toolchain.

=== "Zephyr"

    Zephyr toolchain selection is handled by `ZEPHYR_TOOLCHAIN_VARIANT` and is independent of heliaRT.

    ```bash
    west build -b apollo510_evb -- -DZEPHYR_TOOLCHAIN_VARIANT=gnuarmemb
    ```

## CI Matrix

The release workflow builds **18 combinations** (2 architectures × 3 toolchains × 3 build types):

| Arch | Toolchain | Build types |
|---|---|---|
| `cortex-m4+fp` | gcc, armclang, atfe | debug, release, release_with_logs |
| `cortex-m55` | gcc, armclang, atfe | debug, release, release_with_logs |

## Next Steps

- [SPEED vs SIZE](speed-vs-size.md) — choose the build variant
- [Kernel Selection](kernel-selection.md) — choose the backend
