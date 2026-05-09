---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

![heliaRT](./assets/helia-rt-banner-light.png#only-light){ width="380" }
![heliaRT](./assets/helia-rt-banner-dark.png#only-dark){ width="380" }

# Ambiq-optimised AI inference for Cortex-M

<p class="hero-tagline">Accelerated <strong>TensorFlow Lite for Microcontrollers</strong> runtime with Ambiq-tuned kernels — purpose-built for Apollo silicon.</p>

[:octicons-arrow-right-24: Get Started](getting-started/index.md){ .md-button .md-button--primary }
[:octicons-book-24: Why heliaRT](why-helia-rt.md){ .md-button }

</div>

<div class="stats-strip" markdown>

<div class="stat" markdown>
<span class="stat-icon">:material-chip:</span>
<span class="stat-num">36</span>
<span class="stat-label">Optimised kernels</span>
</div>

<div class="stat" markdown>
<span class="stat-icon">:material-shield-check:</span>
<span class="stat-num">18</span>
<span class="stat-label">CI build combos</span>
</div>

<div class="stat" markdown>
<span class="stat-icon">:material-wrench:</span>
<span class="stat-num">3</span>
<span class="stat-label">Toolchains supported</span>
</div>

<div class="stat" markdown>
<span class="stat-icon">:material-developer-board:</span>
<span class="stat-num">3+</span>
<span class="stat-label">Apollo SoC families</span>
</div>

</div>

---

## Why heliaRT { .section-heading }

heliaRT pairs the familiar **TensorFlow Lite for Microcontrollers** programming model with a kernel backend tuned by Ambiq for Apollo silicon. Models built with the standard LiteRT tooling run unchanged — and run faster, with a larger pool of operators getting the optimised path instead of falling back to generic Reference C.

The runtime is the same one you already know: `MicroInterpreter`, `OpResolver`, statically-allocated tensor arenas, `.tflite` flatbuffers. What changes is what happens *underneath* — more operators take the fast path, on more hardware, with more toolchains.

<div class="two-col" markdown>

<div markdown>

**What stays the same**

- `.tflite` model format — no retraining, no re-quantisation
- `MicroInterpreter` lifecycle (allocate → invoke → read)
- `MicroMutableOpResolver` registration pattern
- Tensor arena sizing and static allocation
- All upstream Reference and CMSIS-NN kernels remain available

**What gets better**

- HELIA backend covers **36 operators** vs CMSIS-NN's 14
- Activations, reduce, and data-movement ops take a vectorised path instead of falling back to Reference C
- Builds matrix-tested across **18** (arch × toolchain × build-type) combos every release
- Distribution as **both** source modules and prebuilt static libraries

</div>

<div markdown>

```cpp
// Same TFLM API you already know
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/heliart_version.h"

constexpr int kArenaSize = 32 * 1024;
alignas(16) uint8_t arena[kArenaSize];

const tflite::Model* model = tflite::GetModel(g_model);
tflite::MicroMutableOpResolver<5> resolver;
resolver.AddConv2D();
resolver.AddFullyConnected();
resolver.AddSoftmax();
resolver.AddReshape();
resolver.AddMaxPool2D();

tflite::MicroInterpreter interpreter(
    model, resolver, arena, kArenaSize);
interpreter.AllocateTensors();
interpreter.Invoke();
```

</div>

</div>

---

## Built for the Apollo family { .section-heading }

heliaRT runs across every Cortex-M-based Ambiq SoC family. The HELIA backend is most impactful on Apollo510, where Cortex-M55 + Helium (MVE) lets vectorised kernels deliver the largest speedups — but every SoC benefits from the broader operator coverage.

<div class="chip-row" markdown>

<div class="chip chip--ok" markdown>
:material-check-circle:{ .chip-icon } **Apollo3 / 3p**<br/>
<span class="chip-meta">Cortex-M4F · DSP</span>
</div>

<div class="chip chip--ok" markdown>
:material-check-circle:{ .chip-icon } **Apollo4 / 4p**<br/>
<span class="chip-meta">Cortex-M4F · DSP</span>
</div>

<div class="chip chip--star" markdown>
:material-star:{ .chip-icon } **Apollo510**<br/>
<span class="chip-meta">Cortex-M55 · MVE / Helium</span>
</div>

<div class="chip chip--planned" markdown>
:material-clock-outline:{ .chip-icon } **Atomiq**<br/>
<span class="chip-meta">Planned</span>
</div>

</div>

[:octicons-arrow-right-24: Full silicon matrix](reference/silicon-support.md)

---

## Three toolchains, one matrix { .section-heading }

Every release ships pre-built artifacts for all three supported toolchains — pick whichever fits your build environment. **ATfE** is our recommendation: open-source, LLVM-based, and consistently 10–20 % faster than GCC on Cortex-M55 MVE workloads.

<div class="chip-row" markdown>

<div class="chip" markdown>
:material-language-c:{ .chip-icon } **GCC**<br/>
<span class="chip-meta">arm-none-eabi · open source · baseline</span>
</div>

<div class="chip" markdown>
:material-package-variant-closed:{ .chip-icon } **Arm Compiler 6**<br/>
<span class="chip-meta">armclang · commercial · ~5–15 % faster</span>
</div>

<div class="chip chip--star" markdown>
:material-star:{ .chip-icon } **ATfE**<br/>
<span class="chip-meta">LLVM-Embedded · open source · ~10–20 % faster</span>
</div>

</div>

[:octicons-arrow-right-24: Toolchain guide](guides/toolchains.md)

---

## Pick your integration path { .section-heading }

heliaRT meets you where you build. Drop it into a Zephyr workspace, deploy through Ambiq's neuralSPOT toolkit, link a prebuilt static library into a custom CMake project, or — coming soon — install via CMSIS-Pack.

<div class="grid cards" markdown>

- :material-home-automation:{ .lg .middle } **Zephyr Module**

    ---

    Add heliaRT as a `west` module. Switch backend with a single Kconfig option. Source-build or use the prebuilt bundle.

    [:octicons-arrow-right-24: Zephyr setup](getting-started/zephyr.md)

- :material-rocket-launch:{ .lg .middle } **neuralSPOT**

    ---

    Profile and deploy a `.tflite` model on Ambiq EVBs in minutes with `ns_autodeploy`. heliaRT is bundled.

    [:octicons-arrow-right-24: neuralSPOT setup](getting-started/neuralspot.md)

- :material-hammer-wrench:{ .lg .middle } **Source / CMake**

    ---

    Full control over target, toolchain, and build type. Link the `libtensorflow-microlite.a` into any project.

    [:octicons-arrow-right-24: Source builds](getting-started/source.md)

- :material-package-variant:{ .lg .middle } **CMSIS-Pack** _(soon)_

    ---

    Install via CMSIS-Pack Manager. Planned for an upcoming release. Track [#124](https://github.com/AmbiqAI/helia-rt/issues/124).

    [:octicons-arrow-right-24: Details](getting-started/cmsis-pack.md)

</div>

---

## Quick start — Zephyr { .section-heading }

Three files. Three commands. A model running on Apollo510.

=== "1. west.yml"

    ```yaml
    manifest:
      projects:
        - name: helia-rt
          url: https://github.com/AmbiqAI/helia-rt
          revision: main
          path: modules/lib/helia-rt
    ```

=== "2. prj.conf"

    ```cfg
    CONFIG_HELIA_RT=y
    CONFIG_HELIA_RT_BACKEND_HELIA=y
    ```

=== "3. Build & flash"

    ```bash
    west update
    west build -b apollo510_evb app
    west flash
    ```

[:octicons-arrow-right-24: Full Zephyr guide](getting-started/zephyr.md) ·
[:octicons-arrow-right-24: Other integration paths](getting-started/index.md) ·
[:octicons-arrow-right-24: Upgrading from upstream LiteRT](guides/upgrading-from-litert.md)

---

## Operator coverage at a glance { .section-heading }

heliaRT's HELIA backend covers categories that upstream CMSIS-NN doesn't touch — most notably activations, reduce, and data-movement ops that would otherwise fall back to slow Reference C.

| Category | REF | CMSIS-NN | **HELIA** |
|---|:---:|:---:|:---:|
| Conv / DW-Conv / Transpose-Conv | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Fully Connected (incl. A16W16) | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Pooling, Softmax, Pad | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Activations (relu, tanh, logistic, …) | :white_check_mark: | — | :white_check_mark: |
| Reduce (mean, max) | :white_check_mark: | — | :white_check_mark: |
| Data movement (concat, reshape, split, …) | :white_check_mark: | — | :white_check_mark: |
| Comparisons & arithmetic (sub, equal, …) | :white_check_mark: | — | :white_check_mark: |

[:octicons-arrow-right-24: Full operator matrix](reference/operator-coverage.md)

---

## Resources { .section-heading }

<div class="grid cards" markdown>

- [:octicons-book-24: **Guides**](guides/index.md)

    ---

    Static vs source, SPEED vs SIZE, kernel selection, memory placement, troubleshooting.

- [:octicons-beaker-24: **Examples**](examples/index.md)

    ---

    Working integration patterns for Zephyr, neuralSPOT, source builds, and CMake.

- [:octicons-graph-24: **Benchmarks**](reference/benchmarks/index.md)

    ---

    Per-target measurements and comparisons across toolchains and build variants.

- [:octicons-people-24: **Contributing**](contributing/index.md)

    ---

    Source layout, upstream sync workflow, release process, and design principles.

</div>
