---
title: Targets and integration scope
description: Distinguish release architecture coverage, source integration and board validation.
---

A runtime architecture target, an SDK integration and a validated board application are different levels of support. Choose all three deliberately before selecting an archive or build configuration.

## Release architecture matrix

The [release workflow](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/helia_release.yml) builds the following combinations:

| Architecture setting | Compilers | Flavors |
| --- | --- | --- |
| `cortex-m4+fp` | GCC, Arm Compiler 6, ATfE | `debug`, `release_with_logs`, `release` |
| `cortex-m55` | GCC, Arm Compiler 6, ATfE | `debug`, `release_with_logs`, `release` |

The HELIA Make path enables FP32 for these builds and FP16 for `cortex-m55`. Match the actual CPU, floating-point/MVE configuration and ABI to the selected library. There is no separate SPEED/SIZE axis in the release matrix.

These are build configurations, not a list of every physical board or silicon revision validated with every model. A source build for another architecture is not automatically a supported release artifact.

## Framework integration

| Path | What the repository supplies | What the application supplies |
| --- | --- | --- |
| Zephyr | West module, Kconfig and source manifest integration | Board, SDK/toolchain, model, memory placement and dependencies |
| neuralSPOT-X | `nsx::helia_rt` source module | SDK module versions, board flags, model and application lifecycle |
| CMSIS-Pack | Source pack generation from the shared manifest | Tooling project, target configuration, kernel dependencies and application |
| Standalone source/CMake | Runtime backend targets and source selection | Toolchain, platform glue, dependency targets and final firmware link |
| Release archive | Compiled runtime/kernel configuration and bundle assets | Compatible application build, platform setup and runtime inputs |

Follow [Choose your path](/helia-rt/getting-started/choose-your-path/) for setup. Configuration options and model compatibility remain relevant across every path.

## Host, simulator and hardware

Host builds are useful for reference checks and development. Corstone-300 FVP runs exercise selected target paths in [CI](/helia-rt/guide/maintenance/testing/). Neither proves physical-board behavior, measured energy or peripheral integration. CPU timing from an FVP is not a substitute for a board benchmark.

For a physical target, verify the board's own SDK documentation, linker regions, clock/timer configuration and toolchain setup. Measure your model on that configuration. This guide does not infer memory capacities, power figures or product availability from a CPU architecture name.

## Optional Ethos-U dispatch

The CMake runtime can include Ethos-U custom-op dispatch through `HELIA_RT_ENABLE_ETHOSU`; neuralSPOT-X uses `NSX_HELIA_RT_ENABLE_ETHOSU`. The application must provide and initialize the matching driver and use an appropriately compiled model. This switch alone does not establish an NPU-equipped board integration or add support to an incompatible model. See the [NSX contract](https://github.com/AmbiqAI/helia-rt/blob/main/nsx/CMakeLists.txt) and [Ethos-U adapter](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/kernels/ethos_u/ethosu.cc).

The [support policy](/helia-rt/guide/maintenance/support/) identifies supported runtime releases. The [license](/helia-rt/guide/maintenance/attribution/) separately governs production deployment.
