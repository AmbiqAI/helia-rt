---
title: Migrate from LiteRT
description: Move an existing LiteRT for Microcontrollers integration to heliaRT and verify the model contract.
---

heliaRT retains the LiteRT for Microcontrollers programming model: a `.tflite` FlatBuffer, an operator resolver and a `MicroInterpreter` using application-supplied memory. Migration still requires checking the runtime revision, build configuration and model behavior together.

## Record your baseline

Keep the model, representative input tensors and expected outputs from the existing application. Record the runtime revision, operator registrations, arena size and toolchain. For a stateful model, include a sequence of inputs and the reset behavior, not just one invocation.

## Replace the runtime integration

Choose the matching [integration path](/helia-rt/getting-started/choose-your-path/). Build against heliaRT's headers and libraries together; do not mix headers from an upstream checkout with an unrelated archive.

Select the backend explicitly where your integration supports it. HELIA connects RT adapters to heliaCORE; Reference and CMSIS-NN are other build choices. Selecting HELIA does not make every model or tensor configuration supported. Check [model compatibility](/helia-rt/guide/model-compatibility/) before changing data types or model operators.

For a source build, align heliaCORE's version and float features with RT. For a prebuilt archive, match its architecture, compiler and build flavor to the application. See [Build configuration](/helia-rt/guide/build-configuration/).

## Verify the execution contract

1. Check the model schema and every operator-registration result.
2. Check `AllocateTensors()` before accessing inputs. Recheck arena usage for this build rather than assuming the upstream size is sufficient.
3. Populate inputs using the model's shape, type and quantization parameters.
4. Check `Invoke()` and compare outputs against your baseline using tolerances appropriate to the model.
5. Repeat the sequence and reset checks for stateful models. Measure latency and memory on the intended target and configuration.

[First inference](/helia-rt/getting-started/first-inference/) shows the checked initialization and invocation sequence. [Runtime concepts](/helia-rt/guide/runtime/) explains the caller-owned lifetimes and state.

## Identify the runtime

The runtime header exposes a version string for application diagnostics:

```cpp
#include "tensorflow/lite/micro/helia_rt_version.h"

const char* runtime_version = HELIA_RT_VERSION;
```

This identifies the header version. Also retain the library's provenance in your build records; a version string alone does not establish that the intended archive was linked.

The source contracts are [MicroInterpreter](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_interpreter.h), [MicroMutableOpResolver](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h) and the [RT version header](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/helia_rt_version.h).
