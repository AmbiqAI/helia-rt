---
title: Choose your path
description: Select a heliaRT integration for your application and build system.
---

Choose the path that matches the project you already have. Keep the runtime source or release bundle version consistent with the integration instructions you follow.

## One source of truth

CMake, Zephyr, neuralSPOT-X source mode and the CMSIS-Pack generator consume the same [source manifest and backend-selection rules](https://github.com/AmbiqAI/helia-rt/blob/main/cmake/README.md). This single source of truth (SSoT) keeps runtime file lists and kernel selection consistent across those integrations.

Make-built release archives use a separate build path. Match an archive's toolchain, architecture and feature settings to your application; shared source selection does not make build configurations interchangeable.

## Zephyr

Use the repository's west module and Kconfig integration when your application builds with Zephyr. The module supports selecting a backend and using source or prebuilt libraries.

Follow [Zephyr setup](/helia-rt/getting-started/zephyr/) for source and prebuilt modules, backend settings, and the application build flow.

## neuralSPOT-X

Use the `nsx` module when your firmware is built with neuralSPOT-X. It integrates the runtime with the SDK's CMake build.

Follow [neuralSPOT-X setup](/helia-rt/getting-started/neuralspot-x/) to add the source module, configure its features and link the runtime target.

## CMSIS-Pack

Use the source pack generator for a project managed through CMSIS tooling. Pack generation and validation do not by themselves establish that a particular IDE or board application has been tested.

Follow [CMSIS-Pack setup](/helia-rt/getting-started/cmsis-pack/) to generate, validate and install a local source pack.

## Source and CMake

Build from source when you need to control compiler options, kernel selection or debugging. Use a prebuilt release archive when its architecture, toolchain and configuration match your firmware.

[Build from source](/helia-rt/getting-started/source/) covers Make archives and CMake source targets. [Link a prebuilt archive](/helia-rt/getting-started/cmake/) covers consuming a released library with matching headers and ABI settings.

## Runtime or ahead-of-time compilation

Use heliaRT for a LiteRT interpreter integration. For supported models, [heliaAOT](https://ambiqai.github.io/helia-aot/) is the recommended option when targeting latency, power and memory efficiency. Check its compatibility and deployment requirements before choosing the compiler path.
