# Why heliaRT

<!-- TODO: Step 3 — Full content -->

!!! note "Work in progress"
    This page is part of the documentation refresh tracked in [#135](https://github.com/AmbiqAI/helia-rt/issues/135).

## The Pitch

heliaRT is a **drop-in replacement** for upstream LiteRT for Microcontrollers (TFLM) with Ambiq-tuned kernels.

## Key Differentiators

- **More optimized operators** — heliaCORE adds tuned implementations where upstream only offers Reference kernels.
- **Drop-in API** — same `MicroInterpreter`, `Model`, `OpResolver` surface. Swap the dependency, rebuild, ship.
- **Ambiq-native** — tuned for Apollo3, Apollo4, and Apollo510 silicon families.
- **Open-source toolchain advantage** — ATfE (LLVM-Embedded for Arm) delivers 10–20% faster inference than GCC at no licensing cost.

## How It Works

<!-- TODO: Stack diagram showing Application → heliaRT → heliaCORE → CMSIS-NN → Cortex-M -->

## Next Steps

- [Getting Started](getting-started/index.md)
- [Operator Coverage](reference/operator-coverage.md)
- [Upgrading from upstream LiteRT](guides/upgrading-from-litert.md)
