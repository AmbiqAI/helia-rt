# :material-chart-bar: Benchmarks

Performance data for heliaRT kernels on Apollo510 EVB (Cortex-M55 + Helium MVE).

## Kernel-Level Benchmarks

Per-operator comparison of heliaRT-optimized implementations vs upstream TFLM.

### Test Environment

| Parameter | Value |
|---|---|
| Board | Apollo510 EVB (Cortex-M55 + Helium MVE) |
| Toolchain | `arm-none-eabi-gcc` v15.2.1 |
| heliaRT | v1.16.0 |
| tflm | upstream tflm, latest main |
| Iterations | 100 (10 warmup) |
| Quantization | int8 (all models) |

### Results

| # | Operator | helia-rt Cycles | tflm Cycles | helia-rt vs tflm |
|---|---|---:|---:|---:|
| 1 | `CONV_2D` | 1,621,810 | 1,642,809 | **1.01×** |
| 2 | `DEPTHWISE_CONV_2D` | 613,204 | 636,011 | **1.04×** |
| 3 | `FULLY_CONNECTED` | 20,844 | 27,950 | **1.34×** |
| 4 | `TRANSPOSE_CONV` | 358,543 | 359,397 | **1.00×** |
| 5 | `AVERAGE_POOL_2D` | 98,581 | 98,590 | **1.00×** |
| 6 | `SOFTMAX` | 9,379 | 9,385 | **1.00×** |
| 7 | `ADD` | 218,395 | 218,369 | **1.00×** |
| 8 | `MUL` | 95,203 | 132,152 | **1.39×** |
| 9 | `LOGISTIC` | 1,015 | 53,208 | **52.4×** |
| 10 | `PAD` | 6,357 | 6,354 | **1.00×** |
| 11 | `RELU` | 98,760 | 985,349 | **10.0×** |
| 12 | `HARD_SWISH` | 65,938 | 870,531 | **13.2×** |
| 13 | `SUB` | 218,337 | 1,921,818 | **8.8×** |
| 14 | `CONCATENATION` | 38,745 | 93,984 | **2.4×** |
| 15 | `SPLIT` | 251,288 | 801,614 | **3.2×** |
| 16 | `STRIDED_SLICE` | 2,250 | 13,900 | **6.2×** |
| 17 | `MEAN` | 22,408 | 2,230,860 | **99.6×** |
| 18 | `REDUCE_MAX` | 3,808 | 2,688,873 | **706×** |

### Methodology

Each operator is exercised by a single-operator int8 TFLite model (input
shape `[1,32,32,16]` for spatial ops).

All cycle counts are median values over 100 iterations after 10 warmup
iterations. The same Apollo510 EVB, and GCC toolchain are
used for all runs.

## Toolchain Comparison (ATfE vs GCC)

For the first published benchmark — ATfE 22.1 vs `arm-none-eabi-gcc` 14.2 across the MLPerf Tiny v1.1 suite on Apollo510 — see [Toolchains → Why ATfE](../../guides/toolchains.md#why-atfe). That section includes the full methodology, a per-model results table, and a Chart.js plot of latency, energy, and efficiency improvements.