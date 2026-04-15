# :material-star-four-points: Features

heliaRT keeps the familiar TensorFlow Lite for Microcontrollers programming model while adding Ambiq-focused runtime and kernel optimizations.

## Familiar TFLM Workflow

If you already know TFLM, the core concepts stay the same:

- `.tflite` flatbuffer model inputs
- `MicroInterpreter`-based execution
- `MicroMutableOpResolver` for selecting operators
- statically allocated tensor arenas
- embedded-focused logging, profiling, and memory tradeoffs

That means existing TFLM knowledge transfers directly to heliaRT, while Ambiq-specific integrations give you a faster path to efficient deployment on Apollo devices.

## What heliaRT Adds

- public source integration with `Reference` and open `CMSIS-NN` backends
- optional HELIA acceleration through Ambiq's private backend module
- build and packaging flows aligned with Ambiq silicon targets
- prebuilt release bundles for faster bring-up
- Zephyr module support alongside source-based integration
- profiling and deployment flows that fit Ambiq developer workflows

## High-Level Capabilities

### Runtime Compatibility

heliaRT is intended for the same class of microcontroller inference workloads as TFLM:

- quantized inference on memory-constrained devices
- operator-resolver-based builds
- static memory planning
- embedded application integration without dynamic runtime dependencies

### Ambiq-Focused Optimization

heliaRT is tuned for Ambiq Apollo platforms and related build flows. Depending on target and configuration, this includes:

- optimized int8 and int16 operator paths
- DSP and MVE-aware kernel implementations where available
- release bundles prepared for supported Ambiq-oriented build matrices

### Flexible Integration Paths

Users can start in the way that best matches their project stage:

- `neuralSPOT` for quick profiling and deployment with `ns_autodeploy`
- Zephyr raw modules for source-visible integration
- Zephyr prebuilt bundles for fast-start integration
- direct source builds for custom environments

## Choosing a Path

Use [Zephyr setup](../usage/zephyr.md) if you are integrating heliaRT into a west workspace.

Use [neuralSPOT setup](../usage/neuralspot.md) if you want the fastest path to evaluate and profile a model on Ambiq hardware.

Use [Examples](../examples/index.md) when you want working integration patterns rather than general setup guidance.
