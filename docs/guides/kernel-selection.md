# Kernel Selection

<!-- TODO: Step 5 — Full content -->

!!! note "Work in progress"
    This page is part of the documentation refresh tracked in [#135](https://github.com/AmbiqAI/helia-rt/issues/135).

## Overview

heliaRT supports three kernel backends, selected at build time:

| Backend | Kconfig | Description |
|---|---|---|
| **Reference** | `HELIA_RT_BACKEND_REFERENCE` | Generic TFLM C kernels. Works on any architecture. |
| **CMSIS-NN** | `HELIA_RT_BACKEND_CMSIS_NN` | Open-source Arm CMSIS-NN kernels. Cortex-M only. |
| **HELIA** | `HELIA_RT_BACKEND_HELIA` | Ambiq-optimized heliaCORE kernels. Cortex-M only. |

## How Selection Works

<!-- TODO: Decision flowchart (Mermaid) showing how OpResolver picks the implementation -->

## Fallback Behaviour

Operators without an optimized variant in the selected backend automatically fall through to the Reference implementation.

## Next Steps

- [Operator Coverage](../reference/operator-coverage.md)
- [Toolchains](toolchains.md)
