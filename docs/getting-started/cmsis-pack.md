# CMSIS-Pack Integration

CMSIS-Pack support is planned for teams that use Keil MDK, CMSIS-Toolbox, or CMSIS-Pack Manager workflows. The goal is to make heliaRT available through the Arm ecosystem package flow while preserving the same backend choices and release artifacts documented elsewhere.

!!! note "Planned package path"
    CMSIS-Pack distribution is not the recommended bring-up path yet. Use Zephyr, neuralSPOT, or source builds for current integration work, and track [#124](https://github.com/AmbiqAI/helia-rt/issues/124) for package availability.

## Expected Workflow

When available, the CMSIS-Pack path is expected to cover:

- installing heliaRT through CMSIS-Pack Manager or `cpackget`
- selecting a SPEED or SIZE release variant
- choosing the Reference, CMSIS-NN, or HELIA backend
- building with Keil MDK, Arm Compiler 6, or CMSIS-Toolbox
- consuming headers and static libraries without cloning the full source tree

## Current Recommendation

Choose one of the supported integration paths below while the pack is being prepared:

| Path | Best for |
|---|---|
| [Zephyr module](zephyr.md) | Product integration, Kconfig backend selection, source or prebuilt bundles. |
| [neuralSPOT](neuralspot.md) | Fast model profiling and deployment on Ambiq evaluation boards. |
| [Source build](source.md) | Custom build systems, direct static library linkage, and source-level debugging. |

## Compatibility Goals

The CMSIS-Pack flow should not change the application-facing runtime model. Teams should still use the familiar `.tflite` model format, `MicroInterpreter` lifecycle, resolver registration pattern, and static tensor arena strategy.

## Next Steps

- [Zephyr setup](zephyr.md) — current recommended product integration path
- [Toolchains](../guides/toolchains.md) — compare GCC, Arm Compiler 6, and ATfE
- [SPEED vs SIZE](../guides/speed-vs-size.md) — choose the release artifact shape
