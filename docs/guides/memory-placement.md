# Memory Placement

<!-- TODO: Step 5 — Content will migrate from PR #111 -->

!!! note "Work in progress"
    This page is part of the documentation refresh tracked in [#135](https://github.com/AmbiqAI/helia-rt/issues/135). Content is being developed in [PR #111](https://github.com/AmbiqAI/helia-rt/pull/111).

## Overview

heliaRT receives pointers (`GetModel()`, tensor arena) and never controls where those objects live in memory. Placement is purely an application-side and board-level concern.

This guide covers how to use Zephyr-native primitives to place the model flatbuffer and tensor arena in specific memory regions on Ambiq Apollo SoCs.

## Topics

- Apollo memory region summary (TCM / SRAM / MRAM per SoC)
- Zephyr linker section tags (`__itcm_section`, `__dtcm_bss_section`, etc.)
- Custom linker fragments
- Devicetree SRAM partitioning
- Verifying placement via the linker map

## Next Steps

- [Troubleshooting](troubleshooting.md)
- [Getting Started with Zephyr](../getting-started/zephyr.md)
