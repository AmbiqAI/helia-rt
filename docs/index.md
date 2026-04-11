#

[![](./assets/helia-rt-banner-light.png#only-light)](https://ambiqai.github.io/helia-rt/)
[![](./assets/helia-rt-banner-dark.png#only-dark)](https://ambiqai.github.io/helia-rt/)

## Built for Ambiq Edge AI

heliaRT is Ambiq's optimized TensorFlow Lite for Microcontrollers runtime for Apollo platforms. It is designed to help developers bring efficient inference to ultra-low-power Ambiq silicon, with tuned kernels that take advantage of Apollo CPU, DSP, and MVE capabilities where available.

## Start Here

- [Getting Started](usage/index.md): choose a setup path for Zephyr, neuralSPOT, or source builds
- [Features](features/index.md): understand how heliaRT maps onto familiar TFLM concepts
- [Examples](examples/index.md): see recommended starting points for Ambiq application bring-up
- [Benchmarks](benchmarks/index.md): review performance-focused documentation for supported Ambiq targets

## Why heliaRT

- Optimized specifically for Ambiq Apollo devices
- Focused on low-power embedded inference
- Available as both source and prebuilt integration paths
- Aligned with Ambiq developer workflows such as neuralSPOT AutoDeploy and Zephyr

## Supported Ambiq Targets

heliaRT is maintained for Ambiq Apollo devices, including:

- Apollo3
- Apollo4
- Apollo4 Plus
- Apollo510

## Recommended Flows

### Start with neuralSPOT

Use [neuralSPOT setup](usage/neuralspot.md) with `ns_autodeploy` when you want the fastest path to profiling a `.tflite` model on Ambiq hardware. See the [neuralSPOT AutoDeploy guide](https://ambiqai.github.io/neuralSPOT/docs/From%20TF%20to%20EVB%20-%20testing%2C%20profiling%2C%20and%20deploying%20AI%20models.html) for an end-to-end walkthrough.

### Integrate into Zephyr

Use [Zephyr setup](usage/zephyr.md) for a west-workspace-based guide covering both the raw heliaRT Zephyr module and the prebuilt release bundle.

### Build from source

Use [Getting Started](usage/index.md) when you need direct control over architecture, toolchain, and build configuration.

## Documentation Highlights

- [Getting Started](usage/index.md)
- [Features](features/index.md)
- [Examples](examples/index.md)
- [Benchmarks](benchmarks/index.md)
- [Continuous Integration](continuous_integration.md)
- [Python Development Guide](python.md)

---

> **Ready to get started?**
> Head over to the [Getting Started](./usage/index.md) guide and bring up heliaRT on Ambiq hardware.
