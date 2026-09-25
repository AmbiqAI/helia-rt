---
title: First inference
description: Model loading, operator registration, tensor allocation and inference with MicroInterpreter.
---

Once heliaRT is linked into your firmware, inference follows the LiteRT for Microcontrollers sequence. The example below illustrates a model containing one fully connected operator. Replace the resolver registrations and tensor handling with those required by your model.

For a complete runnable application with a supplied model and expected outputs, start with [Run your first model](/helia-rt/getting-started/first-model/). This page explains how to adapt that sequence to your firmware.

## Embed a model

The repository includes an array generator. From the heliaRT checkout, with Python, NumPy and Pillow installed in your environment, convert a trusted `model.tflite` file:

```bash
python3 tensorflow/lite/micro/tools/generate_cc_arrays.py genfiles model.tflite
```

This produces `genfiles/model_model_data.cc` and `.h`. The source defines `g_model_model_data` with `alignas(16)`; the header declares the array and `g_model_model_data_size`. Add the generated `.cc` to your application target and `genfiles` to its include directories:

```cmake
target_sources(my_firmware PRIVATE genfiles/model_model_data.cc)
target_include_directories(my_firmware PRIVATE genfiles)
```

Replace `my_firmware` with your application's executable target. Include `model_model_data.h` and pass `g_model_model_data` to the inference function below. Renaming the input file changes the generated symbols. Keep the array in memory accessible to the runtime; [memory placement](/helia-rt/guide/memory-and-profiling/) remains a firmware/linker decision.

The [generator](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/generate_cc_arrays.py) embeds bytes; it does not validate model operators or prove compatibility.

## Keep the inputs alive

The interpreter borrows its model, resolver and tensor arena. Keep them alive for the entire interpreter lifetime. The same applies to optional resource variables and profiler objects. Supply an appropriately aligned arena and measure the required size for your model and build; a fixed example size is not a requirement for every model.

## Create and allocate the interpreter

```cpp
#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

using TensorHandler = TfLiteStatus (*)(TfLiteTensor*);

TfLiteStatus RunModel(const unsigned char* model_data,
                      uint8_t* arena, size_t arena_size,
                      TensorHandler populate_input,
                      TensorHandler consume_output) {
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return kTfLiteError;
  }

  tflite::MicroMutableOpResolver<1> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    return kTfLiteError;
  }

  tflite::MicroInterpreter interpreter(
      model, resolver, arena, arena_size);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return kTfLiteError;
  }
  if (interpreter.inputs_size() != 1 || interpreter.outputs_size() != 1) {
    return kTfLiteError;
  }

  TfLiteTensor* input = interpreter.input(0);
  if (input == nullptr || populate_input(input) != kTfLiteOk) {
    return kTfLiteError;
  }
  const TfLiteStatus status = interpreter.Invoke();
  if (status != kTfLiteOk) {
    return status;
  }
  TfLiteTensor* output = interpreter.output(0);
  return output == nullptr ? kTfLiteError : consume_output(output);
}
```

`model_data` must point to a valid, suitably aligned FlatBuffer. `GetModel` and the schema-version comparison are not validation of an arbitrary byte stream. This example assumes trusted model data and non-null callbacks supplied by the application.

## Populate inputs and consume outputs

Implement the callbacks for your model's tensor shapes, types and quantization parameters. Copy the input into the allocated tensor before calling `Invoke`. Consume or copy the output while the interpreter and its arena remain alive. Do not retain the callback's tensor pointer after this function returns.

The example creates an interpreter for one invocation. For streaming or stateful models, keep an interpreter alive between invocations and follow the operator's state contract; repeatedly constructing it is a different execution model.

## Check each stage

| Failure | Check |
| --- | --- |
| Operator registration | Resolver capacity, duplicate registrations and model operators |
| Tensor allocation | Arena capacity, backend support, tensor types and preparation errors |
| Invocation | Input contents and the operator's runtime error |
| Unexpected output | Shape, tensor type, quantization and state across invocations |

The [MicroInterpreter header](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/micro_interpreter.h) is the contract for object lifetime and invocation. See [Runtime concepts](/helia-rt/guide/runtime/) for the division between model, resolver and backend.
