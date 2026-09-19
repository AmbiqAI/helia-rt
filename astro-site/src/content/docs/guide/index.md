---
title: User guide
description: Understand and configure heliaRT's runtime integration.
---

Start with [Runtime concepts](/helia-rt/guide/runtime/) to understand what the interpreter owns, what your application supplies and how operators reach the selected backend.

## Choose your next task

- [Model compatibility](/helia-rt/guide/model-compatibility/): check operator types, shapes, floating-point features and error stages.
- [Build configuration](/helia-rt/guide/build-configuration/): select a backend, source or archive integration, kernel profile and diagnostic flavor.
- [Migrate from LiteRT](/helia-rt/getting-started/migrate-from-litert/): move an existing application with a known model baseline.

## Configuration decisions

Choose source or prebuilt integration first. Then match the target architecture, toolchain, backend, kernel profile and floating-point options to your model and firmware.

Source configuration does not change an already built static library. For an archive, use the configuration recorded in its release bundle.

## Diagnose in execution order

1. Confirm the model's operator and tensor-type requirements.
2. Check every operator registration result.
3. Check tensor allocation before writing inputs.
4. Check invocation and inspect the model's output contract.
5. Measure with the intended model, target and build configuration.

For initial integration, follow [First inference](/helia-rt/getting-started/first-inference/). For declarations and header locations, use the [API reference](/helia-rt/reference/).
