# Operator Coverage

heliaRT provides three kernel backends. Every operator has a **Reference** implementation. The **CMSIS-NN** and **HELIA** columns show where optimized implementations replace the generic code.

!!! info "How to read this table"
    - **REF** = Reference (generic C, all architectures)
    - **CMSIS** = open-source Arm CMSIS-NN (Cortex-M only)
    - **HELIA** = Ambiq-optimized heliaCORE (Cortex-M only)
    - :white_check_mark: = optimized kernel exists
    - :material-minus: = falls back to Reference

## Compute Operators

| Operator | REF | CMSIS | HELIA | Notes |
|---|:---:|:---:|:---:|---|
| `CONV_2D` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `DEPTHWISE_CONV_2D` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `FULLY_CONNECTED` | :white_check_mark: | :white_check_mark: | :white_check_mark: | HELIA adds A16W16 path |
| `TRANSPOSE_CONV` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `BATCH_MATMUL` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `SVDF` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `UNIDIRECTIONAL_SEQUENCE_LSTM` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |

## Pooling & Padding

| Operator | REF | CMSIS | HELIA | Notes |
|---|:---:|:---:|:---:|---|
| `AVERAGE_POOL_2D` / `MAX_POOL_2D` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `PAD` / `PADV2` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `SOFTMAX` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `TRANSPOSE` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `MAXIMUM` / `MINIMUM` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |

## Activations

| Operator | REF | CMSIS | HELIA | Notes |
|---|:---:|:---:|:---:|---|
| `RELU` / `RELU6` / `RELU_N1_TO_1` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `LOGISTIC` (sigmoid) | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `TANH` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `LEAKY_RELU` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `HARD_SWISH` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA adds int16 path |

## Arithmetic

| Operator | REF | CMSIS | HELIA | Notes |
|---|:---:|:---:|:---:|---|
| `ADD` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `MUL` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `SUB` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `EQUAL` / `NOT_EQUAL` / `GREATER` / `LESS` / etc. | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |

## Data Movement

| Operator | REF | CMSIS | HELIA | Notes |
|---|:---:|:---:|:---:|---|
| `CONCATENATION` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `RESHAPE` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `SPLIT` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `SPLIT_V` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `PACK` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `SQUEEZE` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `STRIDED_SLICE` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `FILL` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `ZEROS_LIKE` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `DEQUANTIZE` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |

## Quantization

| Operator | REF | CMSIS | HELIA | Notes |
|---|:---:|:---:|:---:|---|
| `QUANTIZE` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive (common path) |

## Reduce

| Operator | REF | CMSIS | HELIA | Notes |
|---|:---:|:---:|:---:|---|
| `MEAN` / `REDUCE_MAX` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |

## Summary

| Backend | Optimized kernels | Coverage |
|---|:---:|---|
| Reference | 109 | All operators (generic C) |
| CMSIS-NN | 14 | Core compute-heavy ops |
| **HELIA** | **36** | **Superset of CMSIS-NN + 22 additional** |

!!! success "HELIA advantage"
    HELIA covers **every** operator that CMSIS-NN does, plus 22 additional operators that would otherwise fall back to slow Reference kernels. This means fewer "silent fallbacks" and more consistent performance across your entire model.

## Next Steps

- [Kernel Selection](../guides/kernel-selection.md) — how to choose the backend
- [Silicon Support](silicon-support.md) — which SoCs support which backends
- [Benchmarks](benchmarks/index.md) — measured performance data
