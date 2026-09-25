---
title: Getting started
description: Choose an integration and run your first inference with heliaRT.
---

heliaRT is Ambiq's optimized fork of LiteRT for Microcontrollers. It keeps the model, interpreter and operator-resolver programming model while integrating HELIA kernel adapters.

## Run a known model first

[Run your first model](/helia-rt/getting-started/first-model/) embeds a supplied sine model, builds a complete host application with CMake, and checks its outputs. Start here for a working reference-backend baseline before integrating your own model or board.

## Start with your integration

[Choose your path](/helia-rt/getting-started/choose-your-path/) to connect heliaRT to Zephyr, neuralSPOT-X, a CMSIS-Pack project or your own firmware build.

| Your project | Start here |
|---|---|
| Zephyr / west | [Zephyr](/helia-rt/getting-started/zephyr/) |
| neuralSPOT-X application | [neuralSPOT-X](/helia-rt/getting-started/neuralspot-x/) |
| Custom source build | [Make and CMake source](/helia-rt/getting-started/source/) |
| Custom firmware using release libraries | [Prebuilt archive](/helia-rt/getting-started/cmake/) |
| CMSIS-based project | [CMSIS-Pack](/helia-rt/getting-started/cmsis-pack/) |

Have a runtime library integrated already? Follow [First inference](/helia-rt/getting-started/first-inference/) for the sequence from model loading to output handling.

## What you need

- A `.tflite` model whose operators and tensor types are supported by your selected runtime backend.
- A toolchain and runtime library built for the same target and compatible compiler options as your application.
- Application-owned memory for the model, resolver and tensor arena.
- A way to feed model inputs and inspect its outputs on your target.

Source builds and prebuilt libraries have different configuration boundaries. Choose the distribution before setting kernel and floating-point options.

:::note[Model compatibility]
A kernel in heliaCORE is not sufficient on its own. heliaRT also needs an adapter for the operator and tensor type used by your model. Registering an operator does not establish that every configuration is supported.
:::

## Learn the runtime contract

[Runtime concepts](/helia-rt/guide/runtime/) explains model lifetime, arena ownership, operator registration and backend selection. The [runtime API](/helia-rt/reference/runtime/) links each core type to its header.
