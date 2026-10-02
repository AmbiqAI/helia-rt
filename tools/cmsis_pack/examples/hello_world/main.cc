/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Builds and links a heliaRT interpreter from the CMSIS-Pack component. The
// application supplies its model; with none set, main returns before use.

#include <cstdint>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
constexpr int kArenaSize = 16 * 1024;
alignas(16) uint8_t g_arena[kArenaSize];
// Replace with your model's .tflite flatbuffer. The volatile read keeps the
// interpreter linked while no model is supplied.
const unsigned char* volatile g_model_data = nullptr;
}  // namespace

int main() {
  if (g_model_data == nullptr) {
    return 0;
  }
  const tflite::Model* model = tflite::GetModel(g_model_data);
  tflite::MicroMutableOpResolver<2> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  tflite::MicroInterpreter interpreter(model, resolver, g_arena, kArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return 1;
  }
  return interpreter.Invoke() == kTfLiteOk ? 0 : 1;
}
