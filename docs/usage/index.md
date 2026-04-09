# :material-rocket-launch: Getting Started

Welcome to the heliaRT getting-started guide. heliaRT keeps the familiar TensorFlow Lite for Microcontrollers programming model and adds Ambiq-focused runtime and kernel optimizations for Apollo platforms.

## Start Here

Choose the setup path that matches how you want to evaluate or integrate heliaRT:

- [Zephyr setup](zephyr.md): integrate heliaRT into a Zephyr west workspace using either the raw module or the prebuilt release bundle.
- [neuralSPOT setup](neuralspot.md): profile and deploy a `.tflite` model with `ns_autodeploy` using a fast Ambiq-oriented workflow.
- [Source builds](source.md): build heliaRT directly when you need a custom environment or tighter control over the build.

## Core Concepts

If you have already used TFLM, the high-level model is the same:

- `.tflite` flatbuffer models
- operator resolvers
- tensor arenas
- `MicroInterpreter`-based inference
- embedded-friendly logging and profiling

The main differences are in packaging, supported integration paths, and Ambiq-optimized kernels.

## Setup Paths at a Glance

| Path | Best for | Notes |
| --- | --- | --- |
| Zephyr raw module | Source-visible integration and custom builds | Separate `ns-cmsis-nn` module required for the Ambiq backend |
| Zephyr prebuilt bundle | Fast-start Zephyr integration | Ambiq-optimized kernels embedded in the archive |
| neuralSPOT with `ns_autodeploy` | Quick profiling and deployment | Good first step when evaluating a model on hardware |
| Source build | Custom build systems and low-level integration | Most flexible, but most manual |

## Recommended Order

1. Start with [neuralSPOT setup](neuralspot.md) if your first goal is profiling and basic model validation.
2. Move to [Zephyr setup](zephyr.md) when you are integrating heliaRT into a product or application workspace.
3. Use [source builds](source.md) when you need direct control over toolchains, archives, or custom packaging.

## Related Pages

- [Features](../features/index.md)
- [Zephyr setup](zephyr.md)
- [neuralSPOT setup](neuralspot.md)
- [Source builds](source.md)
- [Examples](../examples/index.md)
