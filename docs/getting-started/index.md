# :material-rocket-launch: Getting Started

heliaRT keeps the familiar LiteRT programming model and adds Ambiq-tuned kernels for Apollo platforms. Pick the integration path that matches your project.

## Pick Your Path

<div class="grid cards" markdown>

- :material-home-automation:{ .lg .middle } **Zephyr Module**

    ---

    Source module or prebuilt bundle via `west`. Switch backend with a single Kconfig.

    **Best for:** product integration, Zephyr-based applications.

    [:octicons-arrow-right-24: Zephyr setup](zephyr.md)

- :material-package-variant:{ .lg .middle } **CMSIS-Pack**

    ---

    Install via CMSIS-Pack Manager. _(Coming soon)_

    **Best for:** Keil MDK, CMSIS-Toolbox users.

    [:octicons-arrow-right-24: Details](cmsis-pack.md)

- :material-rocket-launch:{ .lg .middle } **neuralSPOT**

    ---

    Profile and deploy a `.tflite` model with `ns_autodeploy` in minutes.

    **Best for:** fast model evaluation on Ambiq EVBs.

    [:octicons-arrow-right-24: neuralSPOT setup](neuralspot.md)

- :material-hammer-wrench:{ .lg .middle } **Source / CMake**

    ---

    Full control over target, toolchain, and build type. Link the `.a` into any project.

    **Best for:** custom build systems, source-level debugging.

    [:octicons-arrow-right-24: Source builds](source.md)

</div>

## Core Concepts

If you've used LiteRT for Micro before, you already know the model:

| Concept | Same in heliaRT? |
|---|---|
| `.tflite` flatbuffer models | ✓ |
| `MicroMutableOpResolver` | ✓ |
| `MicroInterpreter` | ✓ |
| Statically-allocated tensor arenas | ✓ |
| Embedded logging and profiling | ✓ |

The key additions are [three kernel backends](../guides/kernel-selection.md) (Reference, CMSIS-NN, HELIA), [two build variants](../guides/speed-vs-size.md) (SPEED, SIZE), and [three toolchain options](../guides/toolchains.md) (GCC, armclang, ATfE).

## Recommended Order

1. **Evaluating a model?** Start with [neuralSPOT](neuralspot.md) and `ns_autodeploy`.
2. **Building a product?** Move to [Zephyr](zephyr.md) for module-based integration.
3. **Need full control?** Use [source builds](source.md) with custom toolchain and target flags.

## Related Pages

- [Why heliaRT](../why-helia-rt.md) — the pitch
- [Features](../features/index.md) — capabilities overview
- [Examples](../examples/index.md) — working integration patterns
- [Upgrading from upstream LiteRT](../guides/upgrading-from-litert.md) — step-by-step swap guide
