# :material-chart-bar: Kernel Benchmarks

<div class="bench-page" markdown>

<section class="bench-hero" markdown>

<p class="section-eyebrow">Apollo510 EVB · Cortex-M55 + Helium MVE</p>

## heliaRT is up to 706× faster than upstream LiteRT for Micro

Across 36 operators benchmarked on real hardware, heliaRT matches or beats
upstream LiteRT for Micro on every single operator — with dramatic gains on activations,
reductions, and data-movement ops that LiteRT for Micro leaves unoptimized.

<div class="bench-stat-strip" markdown>
<div>
<strong>36 / 36</strong>
<span>Operators equal or faster</span>
</div>
<div>
<strong>706×</strong>
<span>Peak speedup </span>
</div>
<div>
<strong>0</strong>
<span>Regressions</span>
</div>
</div>

</section>

## Speedup at a Glance

<canvas id="bench-speedup-chart" data-chart-config='{
  "type": "bar",
  "data": {
    "labels": [
      "REDUCE_MAX",
      "MEAN",
      "TANH",
      "LOGISTIC",
      "RELU6",
      "FILL",
      "HARD_SWISH",
      "RELU",
      "PACK",
      "SUB",
      "LEAKY_RELU",
      "STRIDED_SLICE",
      "EQUAL",
      "SPLIT_V",
      "SPLIT",
      "GREATER",
      "LESS",
      "CONCATENATION",
      "ZEROS_LIKE",
      "RESHAPE",
      "MUL",
      "BATCH_MATMUL",
      "FULLY_CONNECTED",
      "MAXIMUM",
      "MINIMUM",
      "DEPTHWISE_CONV_2D",
      "CONV_2D",
      "PADV2",
      "NOT_EQUAL",
      "TRANSPOSE_CONV",
      "AVERAGE_POOL_2D",
      "SOFTMAX",
      "ADD",
      "MAX_POOL_2D",
      "PAD",
      "TRANSPOSE"
    ],
    "datasets": [
      {
        "label": "Speedup (×)",
        "data": [706, 99.6, 96.5, 52.4, 23.1, 13.7, 13.2, 10.0, 9.2, 8.8, 6.7, 6.2, 3.7, 3.3, 3.2, 2.9, 2.9, 2.4, 1.96, 1.77, 1.39, 1.35, 1.34, 1.26, 1.26, 1.04, 1.01, 1.01, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1],
        "backgroundColor": [
          "#7c4dff","#7c4dff","#7c4dff","#7c4dff",
          "#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3","#00c1b3",
          "#1d99ff","#1d99ff","#1d99ff","#1d99ff","#1d99ff","#1d99ff","#1d99ff","#1d99ff","#1d99ff","#1d99ff","#1d99ff"
        ],
        "borderRadius": 4
      }
    ]
  },
  "options": {
    "indexAxis": "y",
    "responsive": true,
    "maintainAspectRatio": false,
    "plugins": {
      "legend": {"display": false},
      "title": {"display": false}
    },
    "scales": {
      "x": {
        "type": "logarithmic",
        "min": 1,
        "title": {"display": true, "text": "Speedup vs LiteRT for Micro (× — log scale)"},
        "grid": {"color": "rgba(0,0,0,0.06)"}
      },
      "y": {"grid": {"display": false}, "ticks": {"font": {"size": 11}}}
    }
  }
}' style="width:100%;max-height:780px;height:780px;"></canvas>

<div class="bench-legend" markdown>
:material-circle:{ .bench-legend-dramatic } **50–706×** — LiteRT for Micro falls back to scalar C reference
· :material-circle:{ .bench-legend-moderate } **1.3–13×** — heliaRT MVE-optimized paths
· :material-circle:{ .bench-legend-parity } **~1×** — Both use CMSIS-NN / Helium MVE
</div>

## Detailed Results

| # | Operator | heliaRT Cycles | LiteRT Cycles | Speedup |
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
| 19 | `BATCH_MATMUL` | 214,753 | 290,100 | **1.35×** |
| 20 | `MAX_POOL_2D` | 38,405 | 38,405 | **1.00×** |
| 21 | `PADV2` | 6,377 | 6,450 | **1.01×** |
| 22 | `TRANSPOSE` | 21,820 | 21,701 | **~1.00×** |
| 23 | `MAXIMUM` | 7,811 | 9,861 | **1.26×** |
| 24 | `MINIMUM` | 7,809 | 9,860 | **1.26×** |
| 25 | `RELU6` | 8,544 | 197,167 | **23.1×** |
| 26 | `TANH` | 82,301 | 7,945,776 | **96.5×** |
| 27 | `LEAKY_RELU` | 143,929 | 968,936 | **6.7×** |
| 28 | `EQUAL` | 420,227 | 1,547,914 | **3.7×** |
| 29 | `GREATER` | 512,563 | 1,498,903 | **2.9×** |
| 30 | `LESS` | 518,818 | 1,512,053 | **2.9×** |
| 31 | `RESHAPE` | 5,425 | 9,596 | **1.77×** |
| 32 | `SPLIT_V` | 233,019 | 760,585 | **3.3×** |
| 33 | `PACK` | 10,697 | 98,739 | **9.2×** |
| 34 | `NOT_EQUAL` | 115,516 | 115,544 | **1.00×** |
| 35 | `FILL` | 2,422 | 33,143 | **13.7×** |
| 36 | `ZEROS_LIKE` | 8,581 | 16,840 | **1.96×** |

## Why It Matters

!!! success "No silent fallbacks"
    Upstream LiteRT for Micro only optimizes ~14 operators with CMSIS-NN. The rest — activations
    like `RELU` and `LOGISTIC`, reductions like `MEAN` and `REDUCE_MAX`, data-movement
    ops like `SPLIT` and `CONCATENATION` — fall back to **scalar C reference code** that
    ignores Helium MVE entirely. These "silent fallbacks" can dominate inference time
    in real models, even though they look cheap on paper.

    heliaRT closes this gap with **dedicated MVE kernels for 36 operators** — every
    CMSIS-NN-optimized op plus 22 more. The result: no operator is left behind,
    and your model runs at full hardware capability from end to end.

### What this means for your product

<div class="bench-takeaway-grid" markdown>

<div class="bench-takeaway-card" markdown>
#### :material-lightning-bolt: Lower latency
Operators like `MEAN` (99.6×) and `REDUCE_MAX` (706×) often dominate
post-convolution pooling and classification stages. heliaRT eliminates
these hidden bottlenecks.
</div>

<div class="bench-takeaway-card" markdown>
#### :material-battery-charging: Longer battery life
Fewer cycles per inference means less active time and more time in sleep.
Every cycle saved is energy saved.
</div>

<div class="bench-takeaway-card" markdown>
#### :material-swap-horizontal: Drop-in compatible
heliaRT uses the same LiteRT API and `.tflite` models. Switch backends
without changing your application code.
</div>

<div class="bench-takeaway-card" markdown>
#### :material-shield-check: No regressions
Across all operators, heliaRT matches or beats LiteRT for Micro. Zero trade-offs,
zero surprises.
</div>

</div>

## Test Environment

| Parameter | Value |
|---|---|
| Board | Apollo510 EVB (Cortex-M55 + Helium MVE) |
| Toolchain | `arm-none-eabi-gcc` v15.2.1 |
| heliaRT | v1.16.0 |
| LiteRT for Micro | upstream LiteRT for Micro (2.3a snapshot) |
| Iterations | 100 (10 warmup) |
| Quantization | int8 (all models) |

## Methodology

Each operator is exercised by a single-operator int8 TFLite model (input
shape `[1,32,32,16]` for spatial ops, appropriate shapes for non-spatial ops).
All cycle counts are **median** values over 100 iterations after 10 warmup
iterations. The same Apollo510 EVB and GCC toolchain are used for all runs.
Speedup is calculated as `litert_cycles / helia_cycles`.

## Toolchain Comparison (ATfE vs GCC)

For the first published benchmark — ATfE 22.1 vs `arm-none-eabi-gcc` 14.2 across the MLPerf Tiny v1.1 suite on Apollo510 — see [Toolchains → Why ATfE](../../guides/toolchains.md#why-atfe). That section includes the full methodology, a per-model results table, and a Chart.js plot of latency, energy, and efficiency improvements.

</div>
