---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

<div class="hero-bg" aria-hidden="true"></div>

![heliaRT](./assets/helia-rt-banner-light.png#only-light){ .hero-logo width="260" }
![heliaRT](./assets/helia-rt-banner-dark.png#only-dark){ .hero-logo width="260" }

<p class="hero-eyebrow"><span class="dot"></span> v1.13 · Apollo510 ready · ATfE supported</p>

<h1 class="hero-title">Ambiq-optimized <span class="grad">AI inference</span><br/>for Cortex-M</h1>

<p class="hero-tagline">Accelerated <strong>TensorFlow Lite for Microcontrollers</strong> runtime with Ambiq-tuned kernels — purpose-built for Apollo silicon.</p>

<p class="hero-cta">
<a href="getting-started/" class="md-button md-button--primary">Get Started&nbsp;→</a>
<a href="why-helia-rt/" class="md-button">Why heliaRT</a>
</p>

</div>

<div class="stats-strip" markdown>

<div class="stat" markdown>
<span class="stat-icon">:material-chip:</span>
<span class="stat-num">230+</span>
<span class="stat-label">Kernel variants</span>
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

## <span class="eyebrow">00 — The HELIA AI stack</span><br/>Silicon-adjacent AI for ultra-low-power devices { .section-heading }

heliaRT is the runtime layer of the broader **HELIA AI** stack — Ambiq's silicon-adjacent software portfolio for always-on, battery-powered intelligence. HELIA AI is built on top of Ambiq's patented **SPOT®** (Subthreshold Power Optimized Technology) platform, the same sub-threshold design philosophy that lets Apollo SoCs run AI workloads at a fraction of the power of conventional MCUs.

The stack is co-designed top-to-bottom: model architectures, kernels, runtime, and silicon all evolve together so that each layer extracts the most from the one beneath it.

<div class="stack-diagram" markdown>

<div class="stack-layer stack-layer--app" markdown>
<span class="stack-tag">Apps</span>
<span class="stack-title">Audio · Vision · Health · Sensor fusion</span>
<span class="stack-meta">Customer applications and reference designs</span>
</div>

<div class="stack-layer stack-layer--models" markdown>
<span class="stack-tag">HELIA AI · Models</span>
<span class="stack-title">Model zoo · Quantization recipes · Deployment tools</span>
<span class="stack-meta">neuralSPOT-X · ns_autodeploy · model templates</span>
</div>

<div class="stack-layer stack-layer--runtime" markdown>
<span class="stack-tag">HELIA AI · Runtime <span class="stack-pill">You are here</span></span>
<span class="stack-title">heliaRT — TFLM runtime + HELIA kernels</span>
<span class="stack-meta">Optimized operators, vectorized paths, multi-toolchain builds</span>
</div>

<div class="stack-layer stack-layer--silicon" markdown>
<span class="stack-tag">Silicon · SPOT®</span>
<span class="stack-title">Apollo3 · Apollo4 · Apollo510 · Atomiq</span>
<span class="stack-meta">Subthreshold Power Optimized Technology — sub-mW always-on AI</span>
</div>

</div>

[:octicons-arrow-right-24: Learn about HELIA AI](https://ambiq.com/helia-ai/){ .external-link } ·
[:octicons-arrow-right-24: Ambiq SPOT® platform](https://ambiq.com/technology/){ .external-link }

## <span class="eyebrow">01 — Foundation</span><br/>Why heliaRT { .section-heading }

heliaRT pairs the familiar **TensorFlow Lite for Microcontrollers** programming model with a kernel backend tuned by Ambiq for Apollo silicon. Models built with the standard LiteRT tooling run unchanged — and run faster, with a larger pool of operators getting the optimized path instead of falling back to generic Reference C.

The runtime is the same one you already know: `MicroInterpreter`, `OpResolver`, statically-allocated tensor arenas, `.tflite` flatbuffers. What changes is what happens *underneath* — more operators take the fast path, with int8 and int16 quantization variants getting their own hand-tuned kernels.

<div class="two-col" markdown>

<div markdown>

**What stays the same**

- `.tflite` model format — no retraining, no re-quantization
- `MicroInterpreter` lifecycle (allocate → invoke → read)
- `MicroMutableOpResolver` registration pattern
- Tensor arena sizing and static allocation
- All upstream Reference and CMSIS-NN kernels remain available

**What gets better**

- HELIA backend covers **36 operators** (**230+ kernel variants** counting int8 / int16 / float paths) vs CMSIS-NN's 14 operators
- Activations, reduce, and data-movement ops take a vectorized path instead of falling back to Reference C
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

## <span class="eyebrow">02 — Silicon</span><br/>Built for the Apollo family { .section-heading }

heliaRT runs across every Cortex-M-based Ambiq SoC family. The HELIA backend is most impactful on Apollo510, where Cortex-M55 + Helium (MVE) lets vectorized kernels deliver the largest speedups — but every SoC benefits from the broader operator coverage.

<div class="chip-row" markdown>

<div class="chip chip--ok" markdown>
<span class="chip-dot"></span>
<span class="chip-title">Apollo3 / 3p</span>
<span class="chip-meta">Cortex-M4F · DSP</span>
</div>

<div class="chip chip--ok" markdown>
<span class="chip-dot"></span>
<span class="chip-title">Apollo4 / 4p</span>
<span class="chip-meta">Cortex-M4F · DSP</span>
</div>

<div class="chip chip--star" markdown>
<span class="chip-dot"></span>
<span class="chip-title">Apollo510 <span class="chip-badge">Recommended</span></span>
<span class="chip-meta">Cortex-M55 · MVE / Helium</span>
</div>

<div class="chip chip--planned" markdown>
<span class="chip-dot"></span>
<span class="chip-title">Atomiq</span>
<span class="chip-meta">Planned</span>
</div>

</div>

[:octicons-arrow-right-24: Full silicon matrix](reference/silicon-support.md)

## <span class="eyebrow">03 — Toolchains</span><br/>Three toolchains, one matrix { .section-heading }

Every release ships pre-built artifacts for all three supported toolchains — pick whichever fits your build environment. **ATfE** is our recommendation: open-source, LLVM-based, and consistently 10–20 % faster than GCC on Cortex-M55 MVE workloads.

<div class="chip-row" markdown>

<div class="chip chip--ok" markdown>
<span class="chip-dot"></span>
<span class="chip-title">GCC</span>
<span class="chip-meta">arm-none-eabi · open source · baseline</span>
</div>

<div class="chip chip--ok" markdown>
<span class="chip-dot"></span>
<span class="chip-title">Arm Compiler 6</span>
<span class="chip-meta">armclang · commercial · ~5–15 % faster</span>
</div>

<div class="chip chip--star" markdown>
<span class="chip-dot"></span>
<span class="chip-title">ATfE <span class="chip-badge">Recommended</span></span>
<span class="chip-meta">LLVM-Embedded · open source · ~10–20 % faster</span>
</div>

</div>

[:octicons-arrow-right-24: Toolchain guide](guides/toolchains.md)

## <span class="eyebrow">04 — Integration</span><br/>Pick your integration path { .section-heading }

heliaRT meets you where you build. Three flagship platforms get first-class support — **Zephyr**, **neuralSPOT-X**, and **CMSIS-Pack** — with raw source / CMake always available as an escape hatch for custom build systems.

<div class="platform-row" markdown>

<div class="platform platform--featured" markdown>
<span class="platform-tag">Zephyr RTOS</span>
<span class="platform-icon">:material-home-automation:</span>
<span class="platform-title">Zephyr Module</span>
<span class="platform-desc">First-class `west` module. Toggle the HELIA backend with a single Kconfig option, source-build or use the prebuilt bundle.</span>
<span class="platform-status"><span class="chip-dot"></span> Available now · v1.13</span>
<a href="getting-started/zephyr/" class="platform-link">Zephyr setup →</a>
</div>

<div class="platform platform--featured" markdown>
<span class="platform-tag">Ambiq SDK</span>
<span class="platform-icon">:material-rocket-launch:</span>
<span class="platform-title">neuralSPOT-X</span>
<span class="platform-desc">Profile, deploy, and benchmark `.tflite` models on Apollo EVBs in minutes with `ns_autodeploy`. heliaRT ships bundled.</span>
<span class="platform-status"><span class="chip-dot"></span> Available now · v1.13</span>
<a href="getting-started/neuralspot/" class="platform-link">neuralSPOT-X setup →</a>
</div>

<div class="platform platform--featured platform--soon" markdown>
<span class="platform-tag">Arm ecosystem</span>
<span class="platform-icon">:material-package-variant:</span>
<span class="platform-title">CMSIS-Pack <span class="chip-badge">Soon</span></span>
<span class="platform-desc">Install with CMSIS-Pack Manager, drop into Keil / IAR / Open-CMSIS-Pack toolchains. Tracking [issue #124](https://github.com/AmbiqAI/helia-rt/issues/124).</span>
<span class="platform-status"><span class="chip-dot chip-dot--planned"></span> In progress</span>
<a href="getting-started/cmsis-pack/" class="platform-link">Roadmap →</a>
</div>

</div>

<div class="platform-row platform-row--secondary" markdown>

<div class="platform platform--secondary" markdown>
<span class="platform-icon">:material-hammer-wrench:</span>
<span class="platform-title">Source / CMake</span>
<span class="platform-desc">Full control over target, toolchain, and build type. Link `libtensorflow-microlite.a` into any project.</span>
<a href="getting-started/source/" class="platform-link">Source builds →</a>
</div>

</div>

## <span class="eyebrow">05 — Quick start</span><br/>Three files. Three commands. { .section-heading }

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

## <span class="eyebrow">06 — Coverage</span><br/>Operator coverage at a glance { .section-heading }

heliaRT's HELIA backend covers categories that upstream CMSIS-NN doesn't touch — most notably activations, reduce, and data-movement ops that would otherwise fall back to slow Reference C.

<div class="op-matrix" markdown>

<div class="op-matrix-head">
  <div class="op-cat">Operator category</div>
  <div class="op-col" title="TFLM Reference">REF</div>
  <div class="op-col" title="Arm CMSIS-NN">CMSIS-NN</div>
  <div class="op-col op-col--helia">HELIA</div>
</div>

<div class="op-row">
  <div class="op-cat">Conv · DW-Conv · Transpose-Conv</div>
  <div class="op-cell op-cell--ok">✓</div>
  <div class="op-cell op-cell--ok">✓</div>
  <div class="op-cell op-cell--helia">✓</div>
</div>
<div class="op-row">
  <div class="op-cat">Fully Connected (incl. A16W16)</div>
  <div class="op-cell op-cell--ok">✓</div>
  <div class="op-cell op-cell--ok">✓</div>
  <div class="op-cell op-cell--helia">✓</div>
</div>
<div class="op-row">
  <div class="op-cat">Pooling · Softmax · Pad</div>
  <div class="op-cell op-cell--ok">✓</div>
  <div class="op-cell op-cell--ok">✓</div>
  <div class="op-cell op-cell--helia">✓</div>
</div>
<div class="op-row">
  <div class="op-cat">Activations (relu · tanh · logistic …)</div>
  <div class="op-cell op-cell--ok">✓</div>
  <div class="op-cell op-cell--no">—</div>
  <div class="op-cell op-cell--helia">✓</div>
</div>
<div class="op-row">
  <div class="op-cat">Reduce (mean · max)</div>
  <div class="op-cell op-cell--ok">✓</div>
  <div class="op-cell op-cell--no">—</div>
  <div class="op-cell op-cell--helia">✓</div>
</div>
<div class="op-row">
  <div class="op-cat">Data movement (concat · reshape · split …)</div>
  <div class="op-cell op-cell--ok">✓</div>
  <div class="op-cell op-cell--no">—</div>
  <div class="op-cell op-cell--helia">✓</div>
</div>
<div class="op-row">
  <div class="op-cat">Comparisons & arithmetic (sub · equal …)</div>
  <div class="op-cell op-cell--ok">✓</div>
  <div class="op-cell op-cell--no">—</div>
  <div class="op-cell op-cell--helia">✓</div>
</div>

<div class="op-matrix-foot">
  <span><strong>36</strong> operators in HELIA</span>
  <span><strong>230+</strong> kernel variants (int8 / int16 / float)</span>
  <span><strong>+22</strong> operators vs CMSIS-NN</span>
</div>

</div>

[:octicons-arrow-right-24: Full operator matrix](reference/operator-coverage.md)

## <span class="eyebrow">07 — Learn more</span><br/>Resources { .section-heading }

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
