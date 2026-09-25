---
title: Run your first model
description: Embed a known sine model, build a host application with CMake and check its inference outputs.
---

Run a small, known model on your development machine before connecting your own model or board. This walkthrough embeds the repository's float32 sine model, links the **reference backend**, and checks four predictions against `sin(x)`.

The commands target Linux with a native GCC C/C++ toolchain, Git, CMake 3.21 or later, and Python 3.11 or later with venv support. No board, cross-compiler, TensorFlow installation or model training is needed. This is host inference validation; it does not exercise the optimized HELIA backend, MVE, a board's memory layout or its performance.

## Prepare the fixture

Start in a directory where you want to keep the example. Pin the runtime and model together:

```bash
mkdir hello-rt
cd hello-rt
git clone https://github.com/AmbiqAI/helia-rt.git
git -C helia-rt checkout a89c70e7ea207308d7ff86fbbd755f374c7a73f0
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy==2.4.4 Pillow==12.2.0
mkdir app
cp helia-rt/tensorflow/lite/micro/examples/hello_world/models/hello_world_float.tflite app/model.tflite
cd app
sha256sum model.tflite
```

The 3,164-byte fixture's SHA-256 is:

```text
ee939863195ca37ce063b18e14fb82aa0d98db6596ba41095757f6b560da1070
```

This is the existing [upstream-derived hello-world fixture](https://github.com/AmbiqAI/helia-rt/tree/a89c70e7ea207308d7ff86fbbd755f374c7a73f0/tensorflow/lite/micro/examples/hello_world), not a newly trained model. Its example sources retain the TensorFlow Authors' Apache-2.0 notices. Preserve the repository's [license and third-party notices](/helia-rt/guide/maintenance/attribution/) when redistributing. The runtime license permits development and validation on a host; production deployment has separate CPU restrictions.

## Embed the model bytes

From `hello-rt/app`, run the existing array generator:

```bash
python ../helia-rt/tensorflow/lite/micro/tools/generate_cc_arrays.py genfiles model.tflite
```

It creates `genfiles/model_model_data.cc` and `.h`, declaring `g_model_model_data` and its byte count. The array has 16-byte alignment. NumPy and Pillow are imports of this generator even for a `.tflite` input.

This step converts **file bytes into a C++ array**. It does not convert a trained model into LiteRT operators or quantize it. The supplied fixture is already exported. When starting from your own trained model, export a `.tflite` file using its framework's converter, then check [model compatibility](/helia-rt/guide/model-compatibility/) before embedding it. The repository's [hello-world training and quantization guide](https://github.com/AmbiqAI/helia-rt/blob/a89c70e7ea207308d7ff86fbbd755f374c7a73f0/tensorflow/lite/micro/examples/hello_world/README.md#train-your-own-model) describes that separate workflow; a retrained model is not this pinned fixture.

## Create the application

Save this as `hello-rt/app/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.21)
project(first_model LANGUAGES C CXX)

set(HELIA_RT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../helia-rt" CACHE PATH
    "Path to the pinned heliaRT checkout")
add_subdirectory("${HELIA_RT_DIR}" helia-rt-build)
add_executable(first_model main.cc genfiles/model_model_data.cc)
target_include_directories(first_model PRIVATE genfiles)
target_link_libraries(first_model PRIVATE helia_rt::reference)
target_compile_features(first_model PRIVATE cxx_std_17)
```

Save this as `hello-rt/app/main.cc`:

```cpp
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "model_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

int main() {
  const tflite::Model* model = tflite::GetModel(g_model_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    std::fprintf(stderr, "Model schema mismatch\n");
    return 1;
  }

  tflite::MicroMutableOpResolver<1> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    std::fprintf(stderr, "Operator registration failed\n");
    return 1;
  }

  alignas(16) static uint8_t arena[8192];
  tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof(arena));
  if (interpreter.AllocateTensors() != kTfLiteOk ||
      interpreter.inputs_size() != 1 || interpreter.outputs_size() != 1) {
    std::fprintf(stderr, "Allocation or tensor count check failed\n");
    return 1;
  }
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  if (!input || !output || input->type != kTfLiteFloat32 ||
      output->type != kTfLiteFloat32 || input->bytes != sizeof(float) ||
      output->bytes != sizeof(float)) {
    std::fprintf(stderr, "Expected one float32 input and output value\n");
    return 1;
  }

  constexpr float kInputs[] = {0.0f, 1.0f, 3.0f, 5.0f};
  constexpr float kTolerance = 0.05f;
  for (float x : kInputs) {
    input->data.f[0] = x;
    if (interpreter.Invoke() != kTfLiteOk) {
      std::fprintf(stderr, "Inference failed\n");
      return 1;
    }
    const float predicted = output->data.f[0];
    const float expected = std::sin(x);
    const float error = std::fabs(predicted - expected);
    std::printf("x=%.1f predicted=%.6f expected=%.6f error=%.6f\n",
                x, predicted, expected, error);
    if (!std::isfinite(predicted) || error > kTolerance) {
      std::fprintf(stderr, "Output check failed\n");
      return 1;
    }
  }
  std::puts("PASS: 4 sine predictions within 0.05");
  return 0;
}
```

The model, resolver and arena stay alive throughout inference. The 8 KiB arena is an example budget, not a requirement for other models. `GetModel()` assumes the trusted fixture bytes; the schema check is not validation of arbitrary input files. One resolver registration serves all fully connected nodes in this model.

## Build and check the outputs

Still in `hello-rt/app`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target first_model -j4
./build/first_model
```

The program prints each input, prediction, mathematical reference and absolute error. The reference values are approximately `0.000000`, `0.841471`, `0.141120` and `-0.958924`. It must exit with status zero and end with:

```text
PASS: 4 sine predictions within 0.05
```

The model approximates sine, so predictions need not equal the mathematical reference. The absolute tolerance `0.05` comes from the fixture's [upstream example check](https://github.com/AmbiqAI/helia-rt/blob/a89c70e7ea207308d7ff86fbbd755f374c7a73f0/tensorflow/lite/micro/examples/hello_world/hello_world_test.cc). It is a four-input smoke check, not accuracy evidence for the whole input range or a tolerance for your own model. Non-finite predictions are rejected explicitly; these checks remain active in a release build.

If it fails, check the fixture hash, generated array, backend, tensor types and reported error. Do not widen the tolerance to get a pass. A successful build alone does not establish successful inference.

## Move to your model and board

Keep the working fixture as a baseline. For your model, replace the array, register its operators, and adapt input/output handling to its shapes, types and quantization. Compare against known outputs from that exact exported model. See [First inference](/helia-rt/getting-started/first-inference/) for the reusable application sequence and lifetime rules.

Choose [Zephyr, neuralSPOT-X, CMSIS-Pack or a source/prebuilt integration](/helia-rt/getting-started/choose-your-path/) for firmware. Select a compatible HELIA kernel dependency and board toolchain, provide startup/linker/logging support, and size the arena for the target. Repeat the input/output check on that target before measuring performance. The host executable above does not provide those board integration steps or prove an optimized path was taken.
