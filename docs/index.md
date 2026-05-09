---
hide:
  - navigation
  - toc
---

#

[![](./assets/helia-rt-banner-light.png#only-light)](https://ambiqai.github.io/helia-rt/)
[![](./assets/helia-rt-banner-dark.png#only-dark)](https://ambiqai.github.io/helia-rt/)

<p style="text-align: center; font-size: 1.25em; max-width: 720px; margin: 0 auto 1.5em;">
<strong>heliaRT</strong> is a drop-in replacement for LiteRT for Microcontrollers with
Ambiq-tuned kernels. Same API. Faster inference. Built for Apollo.
</p>

<div class="grid cards" markdown>

- :material-language-c:{ .lg .middle } **Drop-in Upgrade**

    ---

    Same `MicroInterpreter` / `Model` / `OpResolver` API as upstream LiteRT-Micro.
    Swap the dependency, rebuild, ship.

    [:octicons-arrow-right-24: Why heliaRT](why-helia-rt.md)

- :material-chip:{ .lg .middle } **Ambiq-Tuned Kernels**

    ---

    heliaCORE adds optimised operator paths where upstream only offers Reference —
    including activations, reduce, concat, and more.

    [:octicons-arrow-right-24: Operator Coverage](reference/operator-coverage.md)

- :material-speedometer:{ .lg .middle } **SPEED & SIZE Variants**

    ---

    Choose between latency-optimised and footprint-optimised builds.
    Both ship as prebuilt static libraries _and_ source.

    [:octicons-arrow-right-24: SPEED vs SIZE](guides/speed-vs-size.md)

- :material-wrench:{ .lg .middle } **Open-Source Toolchains**

    ---

    GCC, Arm Compiler 6, and **ATfE** (LLVM-Embedded for Arm) —
    ATfE is open-source and typically 10–20 % faster than GCC.

    [:octicons-arrow-right-24: Toolchains](guides/toolchains.md)

</div>

---

## Pick Your Integration Path

<div class="grid cards" markdown>

- :material-home-automation:{ .lg .middle } **Zephyr Module**

    ---

    Source module or prebuilt bundle via `west`. Switch backend with a single Kconfig.

    [:octicons-arrow-right-24: Get started](getting-started/zephyr.md)

- :material-package-variant:{ .lg .middle } **CMSIS-Pack**

    ---

    Install via CMSIS-Pack Manager. _(Coming soon — [#124](https://github.com/AmbiqAI/helia-rt/issues/124))_

    [:octicons-arrow-right-24: Details](getting-started/cmsis-pack.md)

- :material-rocket-launch:{ .lg .middle } **neuralSPOT**

    ---

    Profile and deploy a `.tflite` model with `ns_autodeploy` in minutes.

    [:octicons-arrow-right-24: Get started](getting-started/neuralspot.md)

- :material-hammer-wrench:{ .lg .middle } **Source / CMake**

    ---

    Full control over target, toolchain, and build type. Link the resulting `.a` into any project.

    [:octicons-arrow-right-24: Get started](getting-started/source.md)

</div>

---

## At a Glance

| | |
|---|---|
| **Silicon** | Apollo3 · Apollo3p · Apollo4 · Apollo4p · Apollo510 · _(Atomiq — planned)_ |
| **Toolchains** | GCC · Arm Compiler 6 · ATfE :material-star:{ title="Recommended" } |
| **Build Variants** | SPEED (`-O2`) · SIZE (`-Os`) |
| **Distributions** | Prebuilt static libraries · Full source |
| **Backends** | Reference · CMSIS-NN · **HELIA** (heliaCORE) |

---

## Kernel Coverage Highlights

heliaRT's **HELIA** backend adds optimised kernels where upstream LiteRT-Micro only offers Reference implementations:

| Category | HELIA-exclusive optimisations |
|---|---|
| **Activations** | `relu` · `relu6` · `logistic` · `tanh` · `leaky_relu` · `hard_swish` (+ i16) |
| **Reduce / Pool** | `reduce_mean` · `reduce_max` |
| **Data Movement** | `concatenation` · `reshape` · `split` / `split_v` · `pack` · `squeeze` · `strided_slice` · `fill` · `zeros_like` · `dequantize` |
| **Arithmetic** | `sub` · comparisons |
| **Compute** | `fully_connected` A16W16 path (unique to HELIA) |

All operators also support the standard Reference and CMSIS-NN backends. [:octicons-arrow-right-24: Full operator matrix](reference/operator-coverage.md)

---

## Learn More

<div class="grid cards" markdown>

- [:octicons-book-24: **Guides**](guides/index.md) — static vs source, toolchains, memory placement, troubleshooting
- [:octicons-beaker-24: **Examples**](examples/index.md) — per-target app walkthroughs with build + flash + UART output
- [:octicons-graph-24: **Benchmarks**](reference/benchmarks/index.md) — performance data for supported Ambiq targets
- [:octicons-people-24: **Contributing**](contributing/index.md) — architecture, upstream sync process, releases

</div>
