# Getting Started with heliaRT

Welcome to the heliaRT getting-started guide. heliaRT keeps the familiar TensorFlow Lite for Microcontrollers programming model and adds Ambiq-focused runtime and kernel optimizations for Apollo platforms.

## Recommended Setup Paths

- [Zephyr setup](../../../../docs/usage/zephyr.md): integrate heliaRT into a west workspace using either the raw module or a prebuilt release bundle
- [neuralSPOT setup](../../../../docs/usage/neuralspot.md): use `ns_autodeploy` for quick profiling and deployment
- [Source builds](../../../../docs/usage/source.md): build heliaRT directly from source
- [Features](../../../../docs/features/index.md): see how heliaRT maps onto familiar TFLM concepts

## Core Concepts

If you already know TFLM, the core mental model is unchanged:

- `.tflite` flatbuffer models
- `MicroInterpreter`
- operator resolvers
- tensor arenas
- embedded-friendly inference and profiling

The main heliaRT additions are Ambiq-focused optimization, supported packaging flows, and integration paths for Zephyr and profiling workflows.

## Source Build Overview

For direct archive generation and lower-level integration, use the dedicated [Source builds](../../../../docs/usage/source.md) guide.

## Next Steps

- use [neuralSPOT setup](../../../../docs/usage/neuralspot.md) for quick model profiling
- use [Zephyr setup](../../../../docs/usage/zephyr.md) for west-workspace integration
- use [Source builds](../../../../docs/usage/source.md) for direct archive generation
- use [Examples](../../../../docs/examples/index.md) for applied integration patterns
