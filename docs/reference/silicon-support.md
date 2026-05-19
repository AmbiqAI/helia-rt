# Silicon Support

heliaRT runs on all Ambiq SoC families with Cortex-M cores.

## SoC Feature Matrix

| SoC Family | Core | FPU | DSP | MVE / Helium | Release arch | Zephyr board |
|---|---|:---:|:---:|:---:|---|---|
| **Apollo3 / Apollo3p** | Cortex-M4F | ✓ | ✓ | — | `cortex-m4+fp` | `apollo3p_evb` |
| **Apollo4 / Apollo4p** | Cortex-M4F | ✓ | ✓ | — | `cortex-m4+fp` | `apollo4p_evb` |
| **Apollo510** | Cortex-M55 | ✓ | ✓ | ✓ | `cortex-m55` | `apollo510_evb` |
| **Atomiq** | _(planned)_ | | | | | |

!!! tip "Apollo510 for best performance"
    Apollo510's Cortex-M55 includes Arm Helium (MVE), which enables vectorized math in both CMSIS-NN and HELIA kernels. This is where heliaRT delivers the largest speedup over Reference.

## Backend Availability per SoC

| SoC Family | Reference | CMSIS-NN | HELIA |
|---|:---:|:---:|:---:|
| Apollo3 / Apollo3p | ✓ | ✓ | ✓ |
| Apollo4 / Apollo4p | ✓ | ✓ | ✓ |
| Apollo510 | ✓ | ✓ | ✓ |

All backends are supported on all SoCs. The performance advantage of CMSIS-NN and HELIA over Reference is largest on Apollo510 due to MVE.

## Toolchain Compatibility

| SoC Family | GCC | armclang | ATfE |
|---|:---:|:---:|:---:|
| Apollo3 / Apollo3p | ✓ | ✓ | ✓ |
| Apollo4 / Apollo4p | ✓ | ✓ | ✓ |
| Apollo510 | ✓ | ✓ | ✓ :material-star: |

:material-star: ATfE recommended on Apollo510 for best MVE code generation.

## Memory Regions

Each Ambiq SoC provides multiple memory regions. Where you place the tensor arena and model affects performance significantly.

| Region | Apollo3/3p | Apollo4/4p | Apollo510 |
|---|---|---|---|
| TCM | 64 KB unified | See datasheet | Split ITCM + DTCM |
| SRAM | Up to 384 KB | Up to 2 MB | Up to 3 MB |
| MRAM (XIP) | Up to 1 MB | Up to 2 MB | Up to 4 MB |

!!! info
    TCM sizes vary by specific part number. Check your SoC's datasheet for exact values.

[:octicons-arrow-right-24: Memory placement guide](../guides/memory-placement.md)

## Next Steps

- [Operator Coverage](operator-coverage.md) — which kernels are optimized per backend
- [Toolchains](../guides/toolchains.md) — install and select your toolchain
- [Benchmarks](benchmarks/index.md) — measured performance data
