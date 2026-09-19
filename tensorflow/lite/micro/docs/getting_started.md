# Getting Started with heliaRT

Welcome to the heliaRT getting-started guide. heliaRT keeps the familiar TensorFlow Lite for Microcontrollers programming model and adds Ambiq-focused runtime and kernel optimizations for Apollo platforms.

## Recommended Setup Paths

- [Zephyr setup](https://ambiqai.github.io/helia-rt/getting-started/zephyr/): integrate heliaRT into a west workspace using either the raw module or a prebuilt release bundle
- [neuralSPOT-X setup](https://ambiqai.github.io/helia-rt/getting-started/neuralspot-x/): add the runtime source module to an NSX application
- [Source builds](https://ambiqai.github.io/helia-rt/getting-started/source/): build heliaRT directly from source
- [Features](https://ambiqai.github.io/helia-rt/guide/model-compatibility/): see how heliaRT maps onto familiar TFLM concepts

## Core Concepts

If you already know TFLM, the core mental model is unchanged:

- `.tflite` flatbuffer models
- `MicroInterpreter`
- operator resolvers
- tensor arenas
- embedded-friendly inference and profiling

The main heliaRT additions are Ambiq-focused optimization, supported packaging flows, and integration paths for Zephyr and profiling workflows.

## Source Build Overview

For direct archive generation and lower-level integration, use the dedicated [Source builds](https://ambiqai.github.io/helia-rt/getting-started/source/) guide.

## Next Steps

- use [neuralSPOT-X setup](https://ambiqai.github.io/helia-rt/getting-started/neuralspot-x/) for NSX application integration
- use [Zephyr setup](https://ambiqai.github.io/helia-rt/getting-started/zephyr/) for west-workspace integration
- use [Source builds](https://ambiqai.github.io/helia-rt/getting-started/source/) for direct archive generation
- use [Examples](https://ambiqai.github.io/helia-rt/getting-started/) for applied integration patterns
