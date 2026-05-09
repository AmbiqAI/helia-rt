# Upgrading from Upstream LiteRT

<!-- TODO: Step 3 — Full content -->

!!! note "Work in progress"
    This page is part of the documentation refresh tracked in [#135](https://github.com/AmbiqAI/helia-rt/issues/135).

## Overview

heliaRT is a **drop-in replacement** for upstream LiteRT for Microcontrollers (TFLM). The API surface is identical — `MicroInterpreter`, `Model`, `MicroMutableOpResolver` — so switching requires only changing the dependency source.

## Step-by-Step

1. Replace upstream TFLM source / archive with heliaRT
2. Update include paths (if any differ)
3. Optionally switch `OpResolver` registration to use HELIA-optimized operators
4. Rebuild

<!-- TODO: Show concrete git diff / CMake diff for each integration path -->

## What Changes

- More optimized kernels available for the same operator set
- heliaCORE backend available as an additional option
- Build variants (SPEED / SIZE) available in release artifacts

## What Stays the Same

- `.tflite` model format — no retraining or re-quantisation needed
- `MicroInterpreter` lifecycle (allocate → invoke → read output)
- Tensor arena sizing and static memory planning
- `MicroMutableOpResolver` registration pattern

## Next Steps

- [Why heliaRT](../why-helia-rt.md)
- [Operator Coverage](../reference/operator-coverage.md)
