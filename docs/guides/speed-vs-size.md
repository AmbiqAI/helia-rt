# SPEED vs SIZE Build Variants

<!-- TODO: Step 5 — Full content -->

!!! note "Work in progress"
    This page is part of the documentation refresh tracked in [#135](https://github.com/AmbiqAI/helia-rt/issues/135).

## Overview

heliaRT provides two build variants:

- **SPEED** — compiler flags optimise for minimum latency (`-O2` / `-Ofast`).
- **SIZE** — compiler flags optimise for smallest code footprint (`-Os` / `-Oz`).

## How to Select

<!-- TODO: Show Zephyr Kconfig, CMake flag, and Makefile flag for each -->

## Trade-Offs

| Metric | SPEED | SIZE |
|---|---|---|
| Inference latency | Lower | Higher |
| Flash footprint | Larger | Smaller |
| Typical use case | Real-time audio / always-on | Battery / flash-constrained |

## Next Steps

- [Static vs Source](static-vs-source.md)
- [Toolchains](toolchains.md)
