# :material-chart-bar: Benchmarks

This section is reserved for heliaRT performance data on supported Ambiq targets.

## Benchmark Focus

Benchmark results published here should reflect:

- Ambiq Apollo target configurations
- clearly identified build settings
- reproducible measurement methodology
- heliaRT runtime behavior on representative models

For now, use the main [Getting Started](../../getting-started/index.md) guide to profile models on hardware with `ns_autodeploy`.

## Toolchain comparison (ATfE vs GCC)

For the first published benchmark — ATfE 22.1 vs `arm-none-eabi-gcc` 14.2 across the MLPerf Tiny v1.1 suite on Apollo510 — see [Toolchains → Why ATfE](../../guides/toolchains.md#why-atfe). That section includes the full methodology, a per-model results table, and a Chart.js plot of latency, energy, and efficiency improvements.
