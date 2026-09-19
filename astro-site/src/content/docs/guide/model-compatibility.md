---
title: Model compatibility
description: Check operators, tensor types and build features before running a model with heliaRT.
---

heliaRT reads the LiteRT `.tflite` model format. Compatibility also depends on the operators you register, their tensor types and parameters, and the kernel backend compiled into your application. A model loading successfully does not establish that every node can execute.

## Check the whole model

1. Register every operator the model uses with `MicroMutableOpResolver`. Check each registration result; the resolver's template argument is its registration capacity.
2. Check the input, weight and output types for each node. An operator supporting int8 does not imply support for every int16 or floating-point configuration.
3. Check shape, axis, quantization and operator options against the selected adapter. These can change whether an optimized path is available or the operation is supported at all.
4. Build with matching runtime and heliaCORE dependencies, then check both `AllocateTensors()` and `Invoke()` using representative inputs.

The [resolver declarations](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h) and [HELIA adapters](https://github.com/AmbiqAI/helia-rt/tree/main/tensorflow/lite/micro/kernels/helia) define these contracts. Use the source revision matching your application when checking a released library.

## Representative operator families

These examples explain what to inspect; they are not an exhaustive coverage matrix.

| Family | Examples | Compatibility checks |
|---|---|---|
| Compute | [Convolution](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/kernels/helia/conv.cc), [pooling](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/kernels/helia/pooling.cc) | Tensor types, dimensions, quantization and float feature availability. Check returned status when a kernel rejects a configuration. |
| Data movement | [SPLIT_V](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/kernels/helia/split_v.cc) | Constant axis and split sizes, matching output shapes and types. One inferred `-1` split size and zero-length pieces are supported. |
| Indexing | [GATHER](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/kernels/helia/gather.cc), [GATHER_ND](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/kernels/helia/gather_nd.cc) | Axis/index metadata and rank limits. Optimized support and reference fallback differ by tensor type. |
| Elementwise math | [SQRT and RSQRT](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/kernels/helia/elementwise.cc) | FP16 feature availability, matching tensor shapes and operator-specific numeric behavior. The FP32 implementations use the reference path. |
| Index reductions | [ARG_MIN and ARG_MAX](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/kernels/helia/arg_min_max.cc) | The FP16 path accepts rank-one through rank-four inputs, an INT32 axis and an INT32 output with the reduced dimension removed. The reduced extent must be positive. |

## Match floating-point features to the build

`ARM_NN_ENABLE_F32` and `ARM_NN_ENABLE_F16` describe the float kernels available in the linked heliaCORE build. The runtime adapters and kernel library must agree. Adding a compiler definition to an application cannot add missing kernels to a prebuilt archive.

| Integration | Where float features are resolved |
|---|---|
| CMake and neuralSPOT-X | Configure the heliaCORE options before its target is created. heliaRT reads the resolved target's definitions. See the [CMake integration](https://github.com/AmbiqAI/helia-rt/blob/main/CMakeLists.txt) and [NSX integration](https://github.com/AmbiqAI/helia-rt/blob/main/nsx/CMakeLists.txt). |
| Zephyr | Inspect the final `CONFIG_NS_CMSIS_NN_ENABLE_F32/F16` settings. HELIA's Kconfig uses `imply`, so an explicit user setting can override the default. See [Kconfig](https://github.com/AmbiqAI/helia-rt/blob/main/zephyr/Kconfig). |
| Make | The [HELIA build fragment](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/make/ext_libs/helia.inc) enables FP32 and enables FP16 for `TARGET_ARCH=cortex-m55`. |
| Prebuilt archive | Features are fixed when the archive is built. Match its architecture, floating-point ABI and toolchain to your application. |

:::caution[Fallback is operator-specific]
Several FP32 adapters can use a reference implementation when an optimized path is unavailable. Many FP16 arithmetic operators have no equivalent reference fallback and return an error instead. Operations that only move FP16 storage can have different requirements from FP16 arithmetic. Check the adapter rather than assuming one fallback rule applies to the model.
:::

## Runtime revision matters

Check the source or release notes for the version you link. For example, [quantized convolution error propagation](https://github.com/AmbiqAI/helia-rt/commit/2e122086) was fixed after the 1.21.0 release commit. Do not assume an older archive has the behavior shown by `main`.

## Identify where execution failed

**Registration:** a failed `Add...()` call means the resolver could not accept that registration. Fix this before constructing the interpreter.

**Allocation and preparation:** `AllocateTensors()` prepares the graph as well as allocating memory. An error can indicate an unsupported type, invalid shape or option, missing operator, or insufficient arena space. Increasing the arena is not a general fix for preparation failures.

**Invocation:** `Invoke()` executes the prepared graph. An adapter can still reject inputs or propagate a kernel error at this stage. Check its return value on every invocation and do not consume outputs from a failed invocation as valid results.

Keep runtime logging enabled while diagnosing the failing node. The [interpreter implementation](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_interpreter.cc) shows how preparation and invocation statuses reach the application. Follow [First inference](/helia-rt/getting-started/first-inference/) for a checked application sequence and [Runtime concepts](/helia-rt/guide/runtime/) for model and arena ownership.
