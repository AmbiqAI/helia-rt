---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>
<div class="hero-content" markdown>

![heliaRT](./assets/helia-rt-banner-light.png#only-light){ width="480" }
![heliaRT](./assets/helia-rt-banner-dark.png#only-dark){ width="480" }

# Ambiq-tuned AI inference for Cortex-M

A **drop-in replacement** for LiteRT for Microcontrollers — same API, faster inference, purpose-built for Apollo silicon.

[:octicons-arrow-right-24: Get Started](getting-started/index.md){ .md-button .md-button--primary }
[:octicons-book-24: Why heliaRT](why-helia-rt.md){ .md-button }

</div>
</div>

<div class="grid cards" markdown>

- :material-swap-horizontal:{ .lg .middle } **Drop-in Upgrade**

    ---

    Same `MicroInterpreter`, `OpResolver`, and `.tflite` format. Swap the dependency, rebuild, ship — no retraining needed.

- :material-chip:{ .lg .middle } **36 Optimised Kernels**

    ---

    heliaCORE fills gaps where upstream only has Reference — activations, reduce, concat, reshape, and [more](reference/operator-coverage.md).

- :material-speedometer:{ .lg .middle } **SPEED & SIZE Variants**

    ---

    Latency-optimised or footprint-optimised builds, shipped as prebuilt `.a` libraries and full source.

- :material-wrench:{ .lg .middle } **Three Toolchains**

    ---

    GCC, Arm Compiler 6, and **ATfE** — open-source LLVM, ~10–20 % faster than GCC on Cortex-M55.

- :material-developer-board:{ .lg .middle } **All Apollo Silicon**

    ---

    Apollo3 · Apollo4 · Apollo510, with Atomiq planned. [Full matrix](reference/silicon-support.md).

- :material-shield-check:{ .lg .middle } **Production Ready**

    ---

    18-combo CI matrix (2 archs × 3 toolchains × 3 builds). Every release tested, every artifact published.

</div>

---

## Get Started

<div class="grid cards" markdown>

- :material-home-automation:{ .lg .middle } **Zephyr Module**

    ---

    Source module or prebuilt bundle via `west`.

    [:octicons-arrow-right-24: Setup](getting-started/zephyr.md)

- :material-rocket-launch:{ .lg .middle } **neuralSPOT**

    ---

    Profile and deploy with `ns_autodeploy`.

    [:octicons-arrow-right-24: Setup](getting-started/neuralspot.md)

- :material-hammer-wrench:{ .lg .middle } **Source / CMake**

    ---

    Full control. Link the `.a` into any project.

    [:octicons-arrow-right-24: Setup](getting-started/source.md)

- :material-package-variant:{ .lg .middle } **CMSIS-Pack**

    ---

    _(Coming soon)_

    [:octicons-arrow-right-24: Details](getting-started/cmsis-pack.md)

</div>

---

## Quick Start — Zephyr

Add heliaRT to your `west.yml` and enable the HELIA backend:

```yaml
# west.yml — projects:
- name: helia-rt
  url: https://github.com/AmbiqAI/helia-rt
  revision: main
  path: modules/lib/helia-rt
```

```cfg
# prj.conf
CONFIG_HELIA_RT=y
CONFIG_HELIA_RT_BACKEND_HELIA=y
```

```bash
west build -b apollo510_evb app
west flash
```

[:octicons-arrow-right-24: Full Zephyr guide](getting-started/zephyr.md) · [:octicons-arrow-right-24: Other integration paths](getting-started/index.md)

---

[:octicons-book-24: Guides](guides/index.md) · [:octicons-beaker-24: Examples](examples/index.md) · [:octicons-graph-24: Benchmarks](reference/benchmarks/index.md) · [:octicons-people-24: Contributing](contributing/index.md)
