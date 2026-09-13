# Operator Coverage

heliaRT provides three kernel backends. Every operator has a **Reference** implementation. The **CMSIS-NN** and **HELIA** columns show where optimized implementations replace the generic code.

!!! info "How to read this table"
    - **REF** = Reference (generic C, all architectures)
    - **CMSIS** = open-source Arm CMSIS-NN (Cortex-M only)
    - **HELIA** = Ambiq-optimized heliaCORE (Cortex-M only)
    - :white_check_mark: = optimized kernel exists
    - :material-minus: = falls back to Reference

    These columns are **data-type agnostic**: a :white_check_mark: means an
    optimized kernel exists for at least one data type. Most entries cover
    int8 and int16. For the floating-point picture, see
    [Floating-Point Coverage](#floating-point-coverage) below.

## Compute Operators

| Operator | REF | CMSIS | HELIA | Notes |
|---|:---:|:---:|:---:|---|
| `CONV_2D` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `DEPTHWISE_CONV_2D` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `FULLY_CONNECTED` | :white_check_mark: | :white_check_mark: | :white_check_mark: | HELIA adds A16W16 path |
| `TRANSPOSE_CONV` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `BATCH_MATMUL` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `SVDF` | :white_check_mark: | :white_check_mark: | :white_check_mark: | |
| `UNIDIRECTIONAL_SEQUENCE_LSTM` | :white_check_mark: | :white_check_mark: | :white_check_mark: | HELIA int8/int16 kernels are stateful: they consume the variable hidden/cell state tensors and write the final state back, so consecutive invocations continue the sequence. The CMSIS backend restarts from zero state on every invocation: its `kernels/cmsis_nn` wrapper never sets `hidden_state`, so this holds regardless of the pinned CMSIS-NN version |

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
| `LOGISTIC` (sigmoid) | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive. NaN is not a supported input on the optimized float path; see [Non-finite inputs](../guides/floating-point.md#non-finite-inputs-nan-and-infinities) |
| `TANH` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive. NaN is not a supported input on the optimized float path; see [Non-finite inputs](../guides/floating-point.md#non-finite-inputs-nan-and-infinities) |
| `LEAKY_RELU` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `HARD_SWISH` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA adds int16 path, and FP32/FP16 |

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
| `SPLIT_V` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA adds FP16; see [float feature gates](../guides/floating-point.md#feature-contract) |
| `PACK` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `SQUEEZE` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `STRIDED_SLICE` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `FILL` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `ZEROS_LIKE` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive |
| `DEQUANTIZE` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive; int8 / int16 / uint8 / float16 input, float32 output |
| `GATHER` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive; int8/int16 and FP32/FP16 with CORE ≥ 7.34.0 |
| `GATHER_ND` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive; int8/int16 and FP32/FP16 with CORE ≥ 7.34.0 |

## Quantization

| Operator | REF | CMSIS | HELIA | Notes |
|---|:---:|:---:|:---:|---|
| `QUANTIZE` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive (common path) |

## Reduce

| Operator | REF | CMSIS | HELIA | Notes |
|---|:---:|:---:|:---:|---|
| `MEAN` / `REDUCE_MAX` / `SUM` | :white_check_mark: | :material-minus: | :white_check_mark: | HELIA-exclusive. `MEAN` and `SUM` (REDUCE_SUM) add FP32/FP16 |

## Floating-Point Coverage

The HELIA backend also dispatches FP32 and FP16 operators to heliaCORE. These
paths are gated at build time by `ARM_NN_ENABLE_F32` / `ARM_NN_ENABLE_F16`.
The Make build and the published static libraries always enable FP32 (plus
FP16 on Cortex-M55) and Zephyr implies both from the target; NSX and
standalone CMake are fully opt-in, and an int8-only build carries neither.
See the
[FP16 and FP32 guide](../guides/floating-point.md) for how each build system
resolves them.

!!! warning "FP32 and FP16 degrade differently"
    **FP32** falls back to the Reference kernel whenever the optimized kernel
    is disabled or rejects a configuration — results stay correct, only slower.
    **Most FP16 operators have no TFLM Reference implementation** (pure data
    movement such as `TRANSPOSE` and `RESHAPE`, and the f16-to-f32 widening
    in `DEQUANTIZE`, are the exceptions). Where a
    limitation is known at graph preparation the operator fails
    `AllocateTensors()`; otherwise it returns `kTfLiteError` from `Invoke()`,
    in most cases with a logged diagnostic.

    FP16 *arithmetic* additionally requires Armv8.1-M with MVE floating point
    (Cortex-M55) and is not available on Cortex-M4+FP. Operators that only
    move or widen f16 storage, `TRANSPOSE`, `RESHAPE` and `DEQUANTIZE`, run
    on any supported core.

| Operator | FP32 | FP16 | Constraints |
|---|:---:|:---:|---|
| `CONV_2D` | :white_check_mark: | :white_check_mark: | Grouped convolution is not supported by the optimized kernels: FP32 uses Reference, FP16 is rejected at prepare |
| `DEPTHWISE_CONV_2D` | :white_check_mark: | :white_check_mark: | |
| `FULLY_CONNECTED` | :white_check_mark: | :white_check_mark: | |
| `TRANSPOSE_CONV` | :white_check_mark: | :white_check_mark: | |
| `BATCH_MATMUL` | :white_check_mark: | :white_check_mark: | |
| `SVDF` | :white_check_mark: | :white_check_mark: | |
| `UNIDIRECTIONAL_SEQUENCE_LSTM` | :white_check_mark: | :white_check_mark: | Standard four-gate LSTM only; peephole, projection, layer-norm and CIFG variants use Reference (FP32) or are rejected (FP16). Hidden/cell state carry across invocations requires ns-cmsis-nn v7.29.0+ |
| `AVERAGE_POOL_2D` / `MAX_POOL_2D` | :white_check_mark: | :white_check_mark: | 4-D tensors only |
| `SOFTMAX` | :white_check_mark: | :white_check_mark: | `beta == 1.0` only; other values use Reference (FP32) or are rejected at prepare (FP16) |
| `PAD` / `PADV2` | :white_check_mark: | :white_check_mark: | FP16 requires 4-D tensors, enforced at prepare |
| `TRANSPOSE` | :white_check_mark: | :white_check_mark: | Optimized for rank ≤ 4; higher ranks use Reference (float path for FP32, bitwise 16-bit path for FP16). FP16 works even without `ARM_NN_ENABLE_F16` |
| `MAXIMUM` / `MINIMUM` | :white_check_mark: | :white_check_mark: | Optimized for rank ≤ 4; higher ranks use Reference (FP32) |
| `ADD` / `SUB` / `MUL` | :white_check_mark: | :white_check_mark: | Identical input shapes are optimized at any rank; broadcasting is optimized for rank ≤ 4 with every dimension pair equal or 1, and a higher-rank broadcast uses Reference (FP32) or is rejected at prepare (FP16) |
| `HARD_SWISH` | :white_check_mark: | :white_check_mark: | FP16 requires `ARM_NN_ENABLE_F16`; without it there is no reference to fall back to, so it is rejected at prepare |
| `MEAN` / `SUM` | :white_check_mark: | :white_check_mark: | Optimized for rank ≤ 4 with any axis set; higher ranks use Reference (FP32). FP16 requires `ARM_NN_ENABLE_F16` and rank ≤ 4, both enforced at prepare |
| `CONCATENATION` | :white_check_mark: | :white_check_mark: | Optimized for rank ≤ 4; higher ranks use the Reference path (FP32 and FP16) |
| `SPLIT` | :white_check_mark: | :white_check_mark: | CORE ≥ 7.33.0; constant axis, equal output extents; rank ≥ 1 |
| `SPLIT_V` | :white_check_mark: | :white_check_mark: | CORE ≥ 7.33.0; constant axis and split sizes; one inferred `-1`; zero-length pieces |
| `PACK` | :white_check_mark: | :white_check_mark: | CORE ≥ 7.33.0; equal input shapes, including scalars |
| `UNPACK` | :white_check_mark: | :white_check_mark: | CORE ≥ 7.33.0; one output per selected axis element |
| `FILL` | :white_check_mark: | :white_check_mark: | CORE ≥ 7.33.0; constant dimensions, scalar fill value |
| `GATHER` | :white_check_mark: | :white_check_mark: | CORE ≥ 7.34.0; native ranks are ≤ 4. Scalar indices are native for floats, use Reference for int8, and are rejected for int16. Higher ranks use Reference for FP32/int8 and are rejected for FP16/int16 |
| `GATHER_ND` | :white_check_mark: | :white_check_mark: | CORE ≥ 7.34.0; native ranks are ≤ 4 with positive tuple width. Higher ranks and zero-width tuples use Reference for FP32/int8 and are rejected for FP16/int16 |
| `RESHAPE` | :white_check_mark: | :white_check_mark: | Pure data movement; FP16 works even without `ARM_NN_ENABLE_F16` via a bitwise copy |
| `DEQUANTIZE` | :white_check_mark: | :white_check_mark: | FP16 is an input storage type widened to an FP32 output, not FP16 arithmetic; works even without `ARM_NN_ENABLE_F16` |
| `RELU` / `RELU6` | :white_check_mark: | :white_check_mark: | |
| `LOGISTIC` (sigmoid) | :white_check_mark: | :white_check_mark: | NaN is not a supported input on the optimized float path; see [Non-finite inputs](../guides/floating-point.md#non-finite-inputs-nan-and-infinities) |
| `TANH` | :white_check_mark: | :white_check_mark: | NaN is not a supported input on the optimized float path; see [Non-finite inputs](../guides/floating-point.md#non-finite-inputs-nan-and-infinities) |

Other operators use the Reference implementation for FP32, with one
exception: on their quantized-integer side, `QUANTIZE` and `DEQUANTIZE`
always convert through optimized heliaCORE kernels, independent of
`ARM_NN_ENABLE_F32`. `DEQUANTIZE` with a FLOAT16 input is the one path that
does not: it widens f16 storage to float32 in the kernel itself, so it needs
neither heliaCORE nor `ARM_NN_ENABLE_F16`. The published static libraries
ship FP32 kernels for Cortex-M4+FP and both FP32 and FP16 for Cortex-M55.

## Summary

| Backend | Optimized kernels | Coverage |
|---|:---:|---|
| Reference | 109 | All operators (generic C) |
| CMSIS-NN | 14 | Core compute-heavy ops |
| **HELIA** | **38** | **Superset of CMSIS-NN + 24 additional** |

!!! success "HELIA advantage"
    HELIA covers **every** operator that CMSIS-NN does, plus 24 additional operators that would otherwise fall back to slow Reference kernels. This means fewer "silent fallbacks" and more consistent performance across your entire model.

## Next Steps

- [Kernel Selection](../guides/kernel-selection.md) — how to choose the backend
- [FP16 and FP32](../guides/floating-point.md) — enabling and verifying the floating-point kernels
- [Silicon Support](silicon-support.md) — which SoCs support which backends
- [Benchmarks](benchmarks/index.md) — measured performance data
