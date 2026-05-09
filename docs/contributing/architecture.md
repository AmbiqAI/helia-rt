# Architecture

<!-- TODO: Step 9 — Full content -->

!!! note "Work in progress"
    This page is part of the documentation refresh tracked in [#135](https://github.com/AmbiqAI/helia-rt/issues/135).

## Source Layout

<!-- TODO: Annotated directory tree showing key areas -->

## Where heliaCORE Lives

<!-- TODO: Explain tensorflow/lite/micro/kernels/helia/ and how it wires into the build -->

## Kernel Registration

<!-- TODO: How OpResolver picks implementations, Makefile.inc mechanics -->

## Design Principles

- **Minimal diff from upstream** — isolate Ambiq additions to reduce merge conflicts on upstream syncs.
- **Preserve TFLM API** — no public API changes that break upstream compatibility.
- **Backend as extension** — heliaCORE is an additive layer, not a fork of CMSIS-NN.

## Next Steps

- [Upstream Sync](upstream-sync.md)
- [Kernel Selection](../guides/kernel-selection.md)
