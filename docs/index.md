---
hide:
  - navigation
  - toc
---

<div style="text-align: center; margin: 2em auto 1em;" markdown>

![heliaRT](./assets/helia-rt-banner-light.png#only-light){ width="360" }
![heliaRT](./assets/helia-rt-banner-dark.png#only-dark){ width="360" }

**Drop-in replacement for LiteRT for Microcontrollers with Ambiq-tuned kernels.**
{ style="font-size: 1.2em; max-width: 640px; margin: 0.5em auto 1.5em;" }

[:octicons-arrow-right-24: Get Started](getting-started/index.md){ .md-button .md-button--primary }
[:octicons-book-24: Why heliaRT](why-helia-rt.md){ .md-button }

</div>

---

- :material-swap-horizontal: **Drop-in upgrade** — same `MicroInterpreter`, `OpResolver`, and `.tflite` model format as upstream LiteRT-Micro. Swap the dependency, rebuild, ship.
- :material-chip: **36 optimised kernels** — heliaCORE fills the gaps where upstream only offers Reference — activations, reduce, concat, reshape, and [more](reference/operator-coverage.md).
- :material-speedometer: **SPEED & SIZE variants** — choose latency-optimised or footprint-optimised builds, shipped as prebuilt `.a` libraries and full source.
- :material-wrench: **Three toolchains** — GCC, Arm Compiler 6, and **ATfE** (open-source, ~10–20 % faster than GCC on Cortex-M55).
- :material-developer-board: **All Apollo silicon** — Apollo3, Apollo4, Apollo510, with Atomiq planned.

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
