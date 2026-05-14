# Toolchains

heliaRT supports three toolchains for Cortex-M targets. All three are tested in CI across every release.

## Comparison

| Toolchain | ID | License | Typical Perf vs GCC | Best For |
|---|---|---|---|---|
| **GCC** (arm-none-eabi-gcc) | `gcc` | Open source | Baseline | Default, broadest availability |
| **Arm Compiler 6** (armclang) | `armclang` | Commercial | ~5–15 % faster | Keil MDK shops |
| **ATfE** (Arm Toolchain for Embedded) | `atfe` | **Open source** | **up to 24 % faster**[^atfe-bench] | **Recommended** |

!!! success "Recommended: ATfE"
    ATfE is LLVM-based, fully open-source, and actively maintained by Arm. It produces measurably faster code than GCC on Cortex-M55 MVE workloads — without any licensing cost.

    [:octicons-link-external-16: ATfE on GitHub](https://github.com/arm/arm-toolchain){ target="_blank" }

[^atfe-bench]:
    Measured across the [MLPerf Tiny v1.1](https://mlcommons.org/benchmarks/inference-tiny/) model suite on Apollo510 (Cortex-M55 + Helium) using heliaRT v1.13.1 with `heliaPROFILER`. Compilers: ATfE 22.1.0 vs arm-none-eabi-gcc 14.3.0. Per-model speedup ranges 8 %–24 %; "up to 24 %" reflects the best-case model in this matrix.

## Why ATfE

[ATfE](https://github.com/arm/arm-toolchain) (Arm Toolchain for Embedded) is Arm's LLVM-based, open-source toolchain for bare-metal embedded targets. On Cortex-M55 + Helium workloads it consistently outperforms `arm-none-eabi-gcc` for three reasons:

- **MVE auto-vectorization.** LLVM's loop vectorizer targets the M-Profile Vector Extension (MVE / Helium) more aggressively than GCC at parity optimization levels, lighting up predicated vector paths on inner loops that GCC still emits as scalar.
- **Picolibc over newlib-nano.** ATfE ships [Picolibc](https://github.com/picolibc/picolibc), a modernized C library that is smaller, faster, and tuned for embedded LLVM workflows.
- **compiler-rt builtins.** Arm-tuned soft-float and integer helpers replace `libgcc`, typically with better register usage on M-profile cores.

### Measured performance

We profiled the MLPerf Tiny v1.1 reference suite on the **Apollo510 EVB** (Cortex-M55 + Helium) using heliaRT v1.13.1 with `heliaPROFILER`:

| Configuration | Value |
|---|---|
| heliaRT version | `v1.13.1` |
| Hardware | Apollo510 EVB — Cortex-M55, Helium MVE |
| Compilers | ATfE `22.1.0` vs `arm-none-eabi-gcc 14.3.0` |
| Models | [MLPerf Tiny v1.1](https://mlcommons.org/benchmarks/inference-tiny/) — Keyword Spotting (KWS), Visual Wake Words (VWW), Image Classification (IC), Anomaly Detection (AD) |
| Build | `release`, `-O3`, MVE enabled |
| Measurement | `heliaPROFILER` — mean inference latency |

Across the four reference models, ATfE produced **8 % – 24 % faster** inference than the same code built with GCC. The headline **"up to 24 %"** reflects the best-case model in this matrix; the lowest-impact model still showed an 8 % improvement. We have not observed a model where GCC outperformed ATfE on this target.

!!! tip "When to expect the biggest gains"
    Speedup tracks how vectorizable a model is on MVE. Heavily quantized int8 convolutional and fully-connected layers benefit most. Models dominated by elementwise activations or operators that fall to HELIA kernels (rather than compiler-emitted code) see smaller wins, because the hot path is already hand-tuned assembly underneath.

### Trade-offs

- **Newer toolchain.** ATfE is younger than GCC; expect occasional rough edges around uncommon link-script directives or proprietary SDK glue code.
- **Picolibc instead of newlib.** Most projects work without changes, but if you rely on newlib-specific behavior (e.g. certain `_sbrk` patterns) you may need a small shim.
- **No Arm Compiler 5 compatibility shims.** ATfE follows the modern LLVM toolchain conventions; legacy `armcc`-era assembly may need updates.

For a full build walkthrough on Apollo510 + Zephyr, see [Zephyr + heliaRT → Build](../examples/zephyr.md#4-build).

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

    Download from the [Arm Toolchain for Embedded releases](https://github.com/arm/arm-toolchain/releases):

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
    The prebuilt module auto-selects the matching archive for `gcc` or `atfe`.

    For full build commands and per-toolchain flags, see
    [Zephyr + heliaRT → Build](../examples/zephyr.md#4-build).

    Quick reference:

    ```bash
    # GCC (default — no extra flags)
    west build -b apollo510_evb -s app/helia_rt_app

    # ATfE
    west build -b apollo510_evb -s app/helia_rt_app -- \
      -DZEPHYR_TOOLCHAIN_VARIANT=host/llvm \
      -DLLVM_TOOLCHAIN_PATH=/path/to/ATfE \
      -DCONFIG_LLVM_USE_LLD=y -DCONFIG_COMPILER_RT_RTLIB=y
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
