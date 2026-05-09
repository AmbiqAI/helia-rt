# Toolchains

<!-- TODO: Step 5 — Full content -->

!!! note "Work in progress"
    This page is part of the documentation refresh tracked in [#135](https://github.com/AmbiqAI/helia-rt/issues/135).

## Supported Toolchains

heliaRT supports three toolchains for Cortex-M targets:

| Toolchain | ID | License | Notes |
|---|---|---|---|
| **GCC** (arm-none-eabi-gcc) | `gcc` | Open source | Baseline; widely available |
| **Arm Compiler 6** (armclang) | `armclang` | Commercial | Keil MDK / Arm DS |
| **ATfE** (Arm Toolchain for Embedded) | `atfe` | Open source | LLVM-based; ~10–20% faster than GCC |

!!! tip "Recommended"
    **ATfE** is our recommended toolchain. It is fully open source, produces faster code than GCC on Cortex-M55 MVE workloads, and is actively maintained by Arm.

## Installation

<!-- TODO: Install instructions per toolchain, tabbed -->

## Build Flags

<!-- TODO: Show the Makefile / CMake / Zephyr flags for selecting each toolchain -->

## Next Steps

- [SPEED vs SIZE variants](speed-vs-size.md)
- [Getting Started](../getting-started/index.md)
