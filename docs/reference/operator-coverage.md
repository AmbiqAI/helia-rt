# Operator Coverage

<!-- TODO: Step 6 — Move content from README operator matrix here, add source-of-truth YAML -->

!!! note "Work in progress"
    This page is part of the documentation refresh tracked in [#135](https://github.com/AmbiqAI/helia-rt/issues/135).

## Overview

heliaRT supports three kernel backends. Each operator is available at the Reference level. The **CMSIS-NN** and **HELIA** columns indicate where optimized implementations replace the generic reference kernels.

- **Reference** — Generic TFLM C kernels. Works on any architecture.
- **CMSIS-NN** — Open-source Arm CMSIS-NN optimized kernels. Cortex-M only.
- **HELIA** — Ambiq-optimized kernels (heliaCORE / ns-cmsis-nn). Cortex-M only.

## Operator Matrix

<!-- TODO: Render from docs/_data/operator-coverage.yml -->
<!-- For now, see the [README](https://github.com/AmbiqAI/helia-rt#operator-support-matrix) for the current table -->

## Next Steps

- [Kernel Selection](../guides/kernel-selection.md)
- [Silicon Support](silicon-support.md)
