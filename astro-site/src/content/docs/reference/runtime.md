---
title: Runtime API
description: Headers for model loading, interpreter execution, resolvers, allocation and profiling.
---

## Core types

Header paths below are relative to `tensorflow/lite/` in the repository.

| Type or function | Header | Role |
| --- | --- | --- |
| `tflite::MicroInterpreter` | [micro/micro_interpreter.h](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_interpreter.h) | Tensor allocation and graph execution |
| `tflite::Model`, `tflite::GetModel` | [schema/schema_generated.h](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/schema/schema_generated.h) | FlatBuffer model view |
| `tflite::MicroOpResolver` | [micro/micro_op_resolver.h](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_op_resolver.h) | Operator lookup interface |
| `tflite::MicroMutableOpResolver<N>` | [micro/micro_mutable_op_resolver.h](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h) | Operator registration with fixed capacity |
| `tflite::MicroAllocator` | [micro/micro_allocator.h](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_allocator.h) | Arena allocation |
| `tflite::MicroProfilerInterface` | [micro/micro_profiler_interface.h](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_profiler_interface.h) | Profiling callback interface |
| `tflite::MicroProfiler` | [micro/micro_profiler.h](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_profiler.h) | Profiling implementation |
| `tflite::MicroResourceVariables` | [micro/micro_resource_variable.h](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_resource_variable.h) | Resource-variable storage |

## Version

Include `tensorflow/lite/micro/helia_rt_version.h` and read `HELIA_RT_VERSION` for the runtime source release string. This macro does not report the linked heliaCORE version or the compiler configuration.

## Invocation contract

The interpreter has constructors accepting either an arena or an existing allocator. Both borrow resources from the caller. `AllocateTensors()` and `Invoke()` return `TfLiteStatus`; check the returned status before proceeding.

Tensor accessors expose memory associated with the interpreter. Typed accessors require the requested C++ type to match the tensor type and can return null. See the header for each accessor's index requirements.
