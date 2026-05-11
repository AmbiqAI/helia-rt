# heliaRT

**Drop-in LiteRT for Microcontrollers with Ambiq-tuned kernels.**

[![Tests](https://github.com/AmbiqAI/helia-rt/actions/workflows/tests_entry.yml/badge.svg)](https://github.com/AmbiqAI/helia-rt/actions/workflows/tests_entry.yml)
[![Release](https://img.shields.io/github/v/release/AmbiqAI/helia-rt?label=latest)](https://github.com/AmbiqAI/helia-rt/releases/latest)
[![License](https://img.shields.io/badge/license-Ambiq%20Apollo%20SDK-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-ambiqai.github.io%2Fhelia--rt-cyan)](https://ambiqai.github.io/helia-rt/)

heliaRT is Ambiq's optimised TensorFlow Lite for Microcontrollers (LiteRT-Micro) runtime for Apollo platforms. It adds heliaCORE — a set of Ambiq-tuned kernel implementations — on top of the standard TFLM API so you get faster inference without changing your application code.

## Why heliaRT?

| | |
|---|---|
| **Drop-in** | Same `MicroInterpreter` / `Model` / `OpResolver` API. Swap the dependency, rebuild, ship. |
| **More kernels** | heliaCORE adds optimised paths for activations, reduce, concat, reshape, and more — where upstream only offers Reference. |
| **Open-source toolchains** | GCC, Arm Compiler 6, and **ATfE** (LLVM-Embedded for Arm, ~10–20 % faster than GCC). |
| **Two variants** | **SPEED** (`-O2`) for latency, **SIZE** (`-Os`) for footprint. Both ship as prebuilt `.a` and source. |

## Supported Silicon

| SoC | Core | DSP | MVE / Helium |
|---|---|---|---|
| Apollo3 / Apollo3p | Cortex-M4F | ✓ | — |
| Apollo4 / Apollo4p | Cortex-M4F | ✓ | — |
| Apollo510 | Cortex-M55 | ✓ | ✓ |
| Atomiq | _(planned)_ | | |

## Quick Start — Zephyr

Add heliaRT to your west workspace and build:

```yaml
# west.yml — add under projects:
- name: helia-rt
  url: https://github.com/AmbiqAI/helia-rt
  revision: main
  path: modules/lib/helia-rt
```

```cfg
# prj.conf
CONFIG_HELIA_RT=y
CONFIG_HELIA_RT_BACKEND_HELIA=y   # or CMSIS_NN / REFERENCE
```

```bash
west build -b apollo510_evb app
west flash
```

See the full [Zephyr getting-started guide](https://ambiqai.github.io/helia-rt/getting-started/zephyr/) for module variants, backend selection, and prebuilt bundles.

## Integration Paths

| Path | Best for | Guide |
|---|---|---|
| **Zephyr module** | Product integration via `west` | [Getting Started — Zephyr](https://ambiqai.github.io/helia-rt/getting-started/zephyr/) |
| **CMSIS-Pack** | Keil / CMSIS-Toolbox _(coming soon)_ | [#124](https://github.com/AmbiqAI/helia-rt/issues/124) |
| **neuralSPOT** | Fast model profiling with `ns_autodeploy` | [Getting Started — neuralSPOT](https://ambiqai.github.io/helia-rt/getting-started/neuralspot/) |
| **Source / CMake** | Full control over build and link | [Getting Started — Source](https://ambiqai.github.io/helia-rt/getting-started/source/) |

## Kernel Backends

| Backend | Description | Requires |
|---|---|---|
| **Reference** | Generic TFLM C kernels | Nothing extra |
| **CMSIS-NN** | Open-source Arm CMSIS-NN | Cortex-M |
| **HELIA** | Ambiq heliaCORE (ns-cmsis-nn) | Cortex-M + Ambiq module |

The full operator coverage matrix is in the [docs](https://ambiqai.github.io/helia-rt/reference/operator-coverage/).

## Documentation

| | |
|---|---|
| [Why heliaRT](https://ambiqai.github.io/helia-rt/why-helia-rt/) | The pitch — drop-in upgrade, kernel coverage, perf gains |
| [Guides](https://ambiqai.github.io/helia-rt/guides/) | Static vs source, SPEED vs SIZE, toolchains, memory placement, troubleshooting |
| [Examples](https://ambiqai.github.io/helia-rt/examples/) | Per-target app walkthroughs |
| [Reference](https://ambiqai.github.io/helia-rt/reference/) | Operator matrix, silicon support, benchmarks, CI |
| [Contributing](https://ambiqai.github.io/helia-rt/contributing/) | Architecture, upstream sync process, releases |

## License

heliaRT is released under the [Ambiq Apollo SDK License](LICENSE). Free use, modification, and redistribution **solely for execution on Ambiq-manufactured CPUs**. See [LICENSE](LICENSE) for details.

## Getting Help

- [Submit an issue](https://github.com/AmbiqAI/helia-rt/issues/new/choose)
- [Contact Ambiq AITG](mailto:support.aitg@ambiq.com)
