# Toolchains

heliaRT supports three toolchains for Cortex-M targets. All three are tested in CI across every release.

## Comparison

| Toolchain | ID | License | Typical Perf vs GCC | Best For |
|---|---|---|---|---|
| **GCC** (arm-none-eabi-gcc) | `gcc` | Open source | Baseline | Default, broadest availability |
| **Arm Compiler 6** (armclang) | `armclang` | Commercial | ~5–15 % faster | Keil MDK shops |
| **ATfE** (Arm Toolchain for Embedded) | `atfe` | **Open source** | **up to 25 % more efficient**[^atfe-bench] | **Recommended** |

!!! success "Recommended: ATfE"
    ATfE is LLVM-based, fully open-source, and actively maintained by Arm. On Cortex-M55 + Helium workloads it delivers **fewer cycles *and* more inferences per Joule** than GCC — a compounding win for battery-powered devices.

    [:octicons-link-external-16: ATfE on GitHub](https://github.com/arm/arm-toolchain){ target="_blank" }

[^atfe-bench]:
    Measured across the [MLPerf Tiny v1.1](https://mlcommons.org/benchmarks/inference-tiny/) reference suite on the Apollo510 EVB (Cortex-M55 + Helium @ 192 MHz, 10 iterations) using heliaRT v1.13.1. Latency derived from PMU cycles; energy captured with a Joulescope. Compilers: ATfE 22.1 vs `arm-none-eabi-gcc` 14.2. Headline **"up to 25 %"** refers to the inferences-per-Joule improvement on Image Classification (ResNet, +24.4 %, rounded). Every model also ran with **lower latency** under ATfE (4 %–13 % fewer cycles) and **lower energy per inference** (6 %–20 %).

## Why ATfE

[ATfE](https://github.com/arm/arm-toolchain) (Arm Toolchain for Embedded) is Arm's LLVM-based, open-source toolchain for bare-metal embedded targets. On Cortex-M55 + Helium workloads it consistently outperforms `arm-none-eabi-gcc` for three reasons:

- **MVE auto-vectorization.** LLVM's loop vectorizer targets the M-Profile Vector Extension (MVE / Helium) more aggressively than GCC at parity optimization levels, lighting up predicated vector paths on inner loops that GCC still emits as scalar.
- **Picolibc over newlib-nano.** ATfE ships [Picolibc](https://github.com/picolibc/picolibc), a modernized C library that is smaller, faster, and tuned for embedded LLVM workflows.
- **compiler-rt builtins.** Arm-tuned soft-float and integer helpers replace `libgcc`, typically with better register usage on M-profile cores.

### Measured performance

We profiled the MLPerf Tiny v1.1 reference suite on the **Apollo510 EVB** (Cortex-M55 + Helium @ 192 MHz) using heliaRT v1.13.1 with `heliaPROFILER` for latency and a Joulescope for energy.

<canvas id="atfe-bench-chart" data-chart-config='{
  "type": "bar",
  "data": {
    "labels": [
      ["Keyword Spotting", "(DS-CNN)"],
      ["Visual Wake Words", "(MobileNetV1)"],
      ["Anomaly Detection", "(Deep Autoencoder)"],
      ["Image Classification", "(ResNet)"]
    ],
    "datasets": [
      {"label": "Latency reduction",       "data": [9.67, 12.53, 4.43, 10.49], "backgroundColor": "#00c1b3", "borderRadius": 4},
      {"label": "Energy reduction",        "data": [5.92, 15.94, 12.05, 19.60], "backgroundColor": "#1d99ff", "borderRadius": 4},
      {"label": "Efficiency improvement",  "data": [6.30, 18.96, 13.71, 24.38], "backgroundColor": "#7c4dff", "borderRadius": 4}
    ]
  },
  "options": {
    "indexAxis": "y",
    "responsive": true,
    "maintainAspectRatio": false,
    "plugins": {
      "legend": {"position": "top", "align": "start", "labels": {"boxWidth": 12, "boxHeight": 12, "padding": 14}},
      "title": {"display": false},
      "tooltip": {
        "callbacks": {}
      }
    },
    "scales": {
      "x": {
        "beginAtZero": true,
        "title": {"display": true, "text": "Improvement over GCC 14.2 (%)"},
        "grid": {"color": "rgba(0,0,0,0.06)"}
      },
      "y": {"grid": {"display": false}, "ticks": {"font": {"size": 12}}}
    }
  }
}' style="width:100%;max-height:340px;height:340px;"></canvas>

#### Configuration

| Field | Value |
|---|---|
| heliaRT version | `v1.13.1` |
| Hardware | Apollo510 EVB — Cortex-M55, Helium MVE @ 192 MHz |
| Compilers | ATfE `22.1` vs `arm-none-eabi-gcc 14.2` |
| Models | [MLPerf Tiny v1.1](https://mlcommons.org/benchmarks/inference-tiny/) — Keyword Spotting (DS-CNN), Visual Wake Words (MobileNetV1), Anomaly Detection (Deep Autoencoder), Image Classification (ResNet) |
| Build | `release`, `-O3`, MVE enabled |
| Iterations | 10 per configuration, mean reported |
| Latency | Derived from PMU cycle counts ÷ 192 MHz |
| Energy | Joulescope capture, normalized per inference (latency × average power) |

#### Per-model results

| Model | Latency reduction | Energy reduction | **Efficiency (inf / Joule)** |
|---|---:|---:|---:|
| Keyword Spotting (DS-CNN) | **−9.7 %** | **−5.9 %** | **+6.3 %** |
| Visual Wake Words (MobileNetV1) | **−12.5 %** | **−15.9 %** | **+19.0 %** |
| Anomaly Detection (Deep Autoencoder) | **−4.4 %** | **−12.1 %** | **+13.7 %** |
| Image Classification (ResNet) | **−10.5 %** | **−19.6 %** | **+24.4 %** |

All values are ATfE relative to GCC; negative is better for latency and energy, positive is better for efficiency.

Across the four reference models, ATfE delivered **4 %–13 % fewer cycles**, **6 %–20 % less energy per inference**, and **6 %–25 % more inferences per Joule** than the same code built with GCC. The headline **"up to 25 %"** refers to the inferences-per-Joule improvement on Image Classification (the most demanding model in the suite). Critically, **no model regressed on any metric** — ATfE is strictly better than GCC across this benchmark.

!!! tip "When to expect the biggest gains"
    Speedup tracks how vectorizable a model is on MVE. Heavily quantized int8 convolutional and fully-connected layers benefit most (ResNet, MobileNetV1). Models dominated by very small kernels or operators that fall to HELIA hand-tuned paths (Anomaly Detection) see smaller compute-side wins, but still benefit from ATfE's tighter code generation — reflected as the larger energy gain than latency gain on AD.

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
