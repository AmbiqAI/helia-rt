---
title: Runtime concepts
description: Model ownership, operator resolution and kernel backends in heliaRT.
---

## Model

A `.tflite` FlatBuffer describes the graph, tensors and constants. `tflite::GetModel` gives the application a view of that data. The model buffer must remain valid while the interpreter uses it.

## Operator resolver

`MicroMutableOpResolver<N>` registers the operators the application makes available. Its template parameter is the registration capacity, not the number of nodes in the model. Multiple nodes can use the same registered operator.

Check each `Add...` result. Registering an operator and finding a compatible implementation for all of its tensor types and options are separate requirements.

## Interpreter and arena

`MicroInterpreter` schedules graph execution. The application supplies the model, resolver and arena; the interpreter does not take ownership of those objects.

Call `AllocateTensors` before filling inputs. After successful invocation, consume outputs before reusing or releasing the memory they depend on. Use `arena_used_bytes()` after allocation to inspect the allocator's reported usage for that configuration.

## Kernel backend

Backend selection is a build decision. A HELIA adapter connects a supported LiteRT operation to heliaCORE kernels. Reference and CMSIS-NN builds provide other backend choices. Runtime adapter support and heliaCORE kernel availability must both match the model.

Do not assume every unsupported combination falls back automatically. Follow the selected adapter's support and error behavior.

## Stateful inference

A persistent interpreter can preserve operator state across invocations when the operator supports that behavior. Recreating the interpreter or calling `Reset` changes the lifecycle. Verify recurrent-model behavior against the runtime release and kernel configuration in use.

The authoritative declarations are in [micro_interpreter.h](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_interpreter.h) and [micro_mutable_op_resolver.h](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h).
