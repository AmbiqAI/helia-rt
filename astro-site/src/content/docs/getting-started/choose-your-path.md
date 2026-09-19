---
title: Choose your path
description: Select a heliaRT integration for your application and build system.
---

Choose the path that matches the project you already have. Keep the runtime source or release bundle version consistent with the integration instructions you follow.

## Zephyr

Use the repository's west module and Kconfig integration when your application builds with Zephyr. The module supports selecting a backend and using source or prebuilt libraries.

Read the [Zephyr setup guide in the repository](https://github.com/AmbiqAI/helia-rt/blob/main/docs/getting-started/zephyr.md) and inspect [the module configuration](https://github.com/AmbiqAI/helia-rt/tree/main/zephyr).

## neuralSPOT-X

Use the `nsx` module when your firmware is built with neuralSPOT-X. It integrates the runtime with the SDK's CMake build.

Start with [the neuralSPOT-X module contract](https://github.com/AmbiqAI/helia-rt/tree/main/nsx). Confirm the SDK/runtime version pairing before copying options from a different release.

## CMSIS-Pack

Use the source pack generator for a project managed through CMSIS tooling. Pack generation and validation do not by themselves establish that a particular IDE or board application has been tested.

Follow the [CMSIS-Pack build and validation instructions](https://github.com/AmbiqAI/helia-rt/blob/main/docs/getting-started/cmsis-pack.md).

## Source and CMake

Build from source when you need to control compiler options, kernel selection or debugging. Use a prebuilt release archive when its architecture, toolchain and configuration match your firmware.

The [source build guide](https://github.com/AmbiqAI/helia-rt/blob/main/docs/getting-started/source.md) covers the make build. The [CMake source integration](https://github.com/AmbiqAI/helia-rt/blob/main/CMakeLists.txt) and [prebuilt CMake example](https://github.com/AmbiqAI/helia-rt/blob/main/docs/examples/cmake.md) cover different integration paths.

## Runtime or ahead-of-time compilation

Use heliaRT for a LiteRT interpreter integration. For supported models, [heliaAOT](https://ambiqai.github.io/helia-aot/) is the recommended option when targeting latency, power and memory efficiency. Check its compatibility and deployment requirements before choosing the compiler path.
