---
title: User guide
description: Configure the runtime, understand model compatibility and validate an application.
---

Use this guide after selecting an [integration path](/helia-rt/getting-started/choose-your-path/). Work from the model's requirements toward the firmware configuration, then verify the resulting application.

## Understand the runtime

- [Runtime concepts](/helia-rt/guide/runtime/): model, resolver, interpreter, arena and state lifetimes.
- [Model compatibility](/helia-rt/guide/model-compatibility/): tensor types, shapes, feature gates and failure stages.
- [Operator coverage](/helia-rt/guide/operators/): inspect the implementation paths relevant to your model.

## Configure the build

- [Build configuration](/helia-rt/guide/build-configuration/): source versus prebuilt, backend and build flavor.
- [Kernel profiles](/helia-rt/guide/kernel-profiles/): SPEED/SIZE defaults and convolution/fully connected overrides.
- [Floating point](/helia-rt/guide/floating-point/): FP16 storage, FP16/FP32 computation and dependency features.
- [Build option reference](/helia-rt/guide/build-options/): RT settings for CMake, NSX, Zephyr and Make.
- [Targets and integration scope](/helia-rt/guide/targets/): release architecture matrix, framework responsibilities and target validation.
- [Toolchains and artifacts](/helia-rt/guide/toolchains/): compiler and ABI matching, release builder and verification boundaries.

## Validate and diagnose

- [Memory and profiling](/helia-rt/guide/memory-and-profiling/): arena auditing, placement and operator timing.
- [Benchmark an application](/helia-rt/guide/benchmarks/): harnesses, reproducibility and historical-result boundaries.
- [Troubleshooting](/helia-rt/guide/troubleshooting/): diagnose configuration, linking, preparation and invocation in order.

## Maintain and support

[Source architecture](/helia-rt/guide/maintenance/architecture/), [Testing and CI](/helia-rt/guide/maintenance/testing/), [Upstream maintenance](/helia-rt/guide/maintenance/upstream-sync/) and [Releases](/helia-rt/guide/maintenance/releases/) cover contributor workflows. Consult the [support policy](/helia-rt/guide/maintenance/support/) and [attribution and licensing](/helia-rt/guide/maintenance/attribution/) for product boundaries and notices.
