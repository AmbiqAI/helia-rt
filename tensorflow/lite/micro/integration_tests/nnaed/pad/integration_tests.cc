/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <string.h>

#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "python/tflite_micro/python_ops_resolver.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad0_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad0_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad0_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad1_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad1_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad1_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad2_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad2_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad2_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad3_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad3_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad3_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad4_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad4_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad4_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad5_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad5_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad5_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad6_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad6_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad6_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad7_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad7_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad7_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad8_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad8_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad8_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad9_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad9_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad9_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad10_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad10_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad10_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad11_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad11_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad11_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad12_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad12_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad12_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad13_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad13_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad13_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad14_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad14_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad14_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad15_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad15_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad15_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad16_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad16_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad16_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad17_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad17_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad17_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad18_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad18_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad18_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad19_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad19_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad19_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad20_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad20_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad20_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad21_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad21_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad21_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad22_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad22_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad22_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad23_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad23_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad23_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad24_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad24_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad24_golden_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad25_model_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad25_input0_int8_test_data.h"
#include "tensorflow/lite/micro/integration_tests/nnaed/pad/pad25_golden_int8_test_data.h"

// constexpr size_t kTensorArenaSize = 1024 * 100;
constexpr size_t kTensorArenaSize = 1024 * 120;

uint8_t tensor_arena[kTensorArenaSize];
bool print_log = false;

namespace tflite {
namespace micro {
namespace {

void RunModel(const uint8_t* model,
              const int8_t* input0,
              const uint32_t input0_size,
              const int8_t* golden,
              const uint32_t golden_size,
              const char* name) {
  InitializeTarget();
  MicroProfiler profiler;
  PythonOpsResolver op_resolver;

  MicroInterpreter interpreter(GetModel(model), op_resolver, tensor_arena,
                               kTensorArenaSize,
                               nullptr, &profiler);
  interpreter.AllocateTensors();
  TfLiteTensor* input_tensor0 = interpreter.input(0);
  TF_LITE_MICRO_EXPECT_EQ(input_tensor0->bytes,
                          input0_size * sizeof(
                              int8_t));
  memcpy(interpreter.input(0)->data.raw,
         input0,
         input_tensor0->bytes);
  if (kTfLiteOk != interpreter.Invoke()) {
    TF_LITE_MICRO_EXPECT(false);
    return;
  }
  if (print_log) {
    profiler.Log();
  }
  MicroPrintf("");

  TfLiteTensor* output_tensor = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(output_tensor->bytes,
                          golden_size * sizeof(int8_t));
  int8_t* output = ::tflite::GetTensorData<int8_t>(output_tensor);
  for (uint32_t i = 0; i < golden_size; i++) {
    // TODO(b/205046520): Better understand why TfLite and TFLM can sometimes be
    // off by 1.
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output[i], 1);
  }
}

}  // namespace
}  // namespace micro
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN


TF_LITE_MICRO_TEST(pad0_test) {tflite::micro::RunModel(
g_pad0_model_data,
g_pad0_input0_int8_test_data,
g_pad0_input0_int8_test_data_size,
g_pad0_golden_int8_test_data,
g_pad0_golden_int8_test_data_size,
"pad0 test");
}


TF_LITE_MICRO_TEST(pad1_test) {tflite::micro::RunModel(
g_pad1_model_data,
g_pad1_input0_int8_test_data,
g_pad1_input0_int8_test_data_size,
g_pad1_golden_int8_test_data,
g_pad1_golden_int8_test_data_size,
"pad1 test");
}


TF_LITE_MICRO_TEST(pad2_test) {tflite::micro::RunModel(
g_pad2_model_data,
g_pad2_input0_int8_test_data,
g_pad2_input0_int8_test_data_size,
g_pad2_golden_int8_test_data,
g_pad2_golden_int8_test_data_size,
"pad2 test");
}


TF_LITE_MICRO_TEST(pad3_test) {tflite::micro::RunModel(
g_pad3_model_data,
g_pad3_input0_int8_test_data,
g_pad3_input0_int8_test_data_size,
g_pad3_golden_int8_test_data,
g_pad3_golden_int8_test_data_size,
"pad3 test");
}


TF_LITE_MICRO_TEST(pad4_test) {tflite::micro::RunModel(
g_pad4_model_data,
g_pad4_input0_int8_test_data,
g_pad4_input0_int8_test_data_size,
g_pad4_golden_int8_test_data,
g_pad4_golden_int8_test_data_size,
"pad4 test");
}


TF_LITE_MICRO_TEST(pad5_test) {tflite::micro::RunModel(
g_pad5_model_data,
g_pad5_input0_int8_test_data,
g_pad5_input0_int8_test_data_size,
g_pad5_golden_int8_test_data,
g_pad5_golden_int8_test_data_size,
"pad5 test");
}


TF_LITE_MICRO_TEST(pad6_test) {tflite::micro::RunModel(
g_pad6_model_data,
g_pad6_input0_int8_test_data,
g_pad6_input0_int8_test_data_size,
g_pad6_golden_int8_test_data,
g_pad6_golden_int8_test_data_size,
"pad6 test");
}


TF_LITE_MICRO_TEST(pad7_test) {tflite::micro::RunModel(
g_pad7_model_data,
g_pad7_input0_int8_test_data,
g_pad7_input0_int8_test_data_size,
g_pad7_golden_int8_test_data,
g_pad7_golden_int8_test_data_size,
"pad7 test");
}


TF_LITE_MICRO_TEST(pad8_test) {tflite::micro::RunModel(
g_pad8_model_data,
g_pad8_input0_int8_test_data,
g_pad8_input0_int8_test_data_size,
g_pad8_golden_int8_test_data,
g_pad8_golden_int8_test_data_size,
"pad8 test");
}


TF_LITE_MICRO_TEST(pad9_test) {tflite::micro::RunModel(
g_pad9_model_data,
g_pad9_input0_int8_test_data,
g_pad9_input0_int8_test_data_size,
g_pad9_golden_int8_test_data,
g_pad9_golden_int8_test_data_size,
"pad9 test");
}


TF_LITE_MICRO_TEST(pad10_test) {tflite::micro::RunModel(
g_pad10_model_data,
g_pad10_input0_int8_test_data,
g_pad10_input0_int8_test_data_size,
g_pad10_golden_int8_test_data,
g_pad10_golden_int8_test_data_size,
"pad10 test");
}


TF_LITE_MICRO_TEST(pad11_test) {tflite::micro::RunModel(
g_pad11_model_data,
g_pad11_input0_int8_test_data,
g_pad11_input0_int8_test_data_size,
g_pad11_golden_int8_test_data,
g_pad11_golden_int8_test_data_size,
"pad11 test");
}


TF_LITE_MICRO_TEST(pad12_test) {tflite::micro::RunModel(
g_pad12_model_data,
g_pad12_input0_int8_test_data,
g_pad12_input0_int8_test_data_size,
g_pad12_golden_int8_test_data,
g_pad12_golden_int8_test_data_size,
"pad12 test");
}


TF_LITE_MICRO_TEST(pad13_test) {tflite::micro::RunModel(
g_pad13_model_data,
g_pad13_input0_int8_test_data,
g_pad13_input0_int8_test_data_size,
g_pad13_golden_int8_test_data,
g_pad13_golden_int8_test_data_size,
"pad13 test");
}


TF_LITE_MICRO_TEST(pad14_test) {tflite::micro::RunModel(
g_pad14_model_data,
g_pad14_input0_int8_test_data,
g_pad14_input0_int8_test_data_size,
g_pad14_golden_int8_test_data,
g_pad14_golden_int8_test_data_size,
"pad14 test");
}


TF_LITE_MICRO_TEST(pad15_test) {tflite::micro::RunModel(
g_pad15_model_data,
g_pad15_input0_int8_test_data,
g_pad15_input0_int8_test_data_size,
g_pad15_golden_int8_test_data,
g_pad15_golden_int8_test_data_size,
"pad15 test");
}


TF_LITE_MICRO_TEST(pad16_test) {tflite::micro::RunModel(
g_pad16_model_data,
g_pad16_input0_int8_test_data,
g_pad16_input0_int8_test_data_size,
g_pad16_golden_int8_test_data,
g_pad16_golden_int8_test_data_size,
"pad16 test");
}


TF_LITE_MICRO_TEST(pad17_test) {tflite::micro::RunModel(
g_pad17_model_data,
g_pad17_input0_int8_test_data,
g_pad17_input0_int8_test_data_size,
g_pad17_golden_int8_test_data,
g_pad17_golden_int8_test_data_size,
"pad17 test");
}


TF_LITE_MICRO_TEST(pad18_test) {tflite::micro::RunModel(
g_pad18_model_data,
g_pad18_input0_int8_test_data,
g_pad18_input0_int8_test_data_size,
g_pad18_golden_int8_test_data,
g_pad18_golden_int8_test_data_size,
"pad18 test");
}


TF_LITE_MICRO_TEST(pad19_test) {tflite::micro::RunModel(
g_pad19_model_data,
g_pad19_input0_int8_test_data,
g_pad19_input0_int8_test_data_size,
g_pad19_golden_int8_test_data,
g_pad19_golden_int8_test_data_size,
"pad19 test");
}


TF_LITE_MICRO_TEST(pad20_test) {tflite::micro::RunModel(
g_pad20_model_data,
g_pad20_input0_int8_test_data,
g_pad20_input0_int8_test_data_size,
g_pad20_golden_int8_test_data,
g_pad20_golden_int8_test_data_size,
"pad20 test");
}


TF_LITE_MICRO_TEST(pad21_test) {tflite::micro::RunModel(
g_pad21_model_data,
g_pad21_input0_int8_test_data,
g_pad21_input0_int8_test_data_size,
g_pad21_golden_int8_test_data,
g_pad21_golden_int8_test_data_size,
"pad21 test");
}


TF_LITE_MICRO_TEST(pad22_test) {tflite::micro::RunModel(
g_pad22_model_data,
g_pad22_input0_int8_test_data,
g_pad22_input0_int8_test_data_size,
g_pad22_golden_int8_test_data,
g_pad22_golden_int8_test_data_size,
"pad22 test");
}


TF_LITE_MICRO_TEST(pad23_test) {tflite::micro::RunModel(
g_pad23_model_data,
g_pad23_input0_int8_test_data,
g_pad23_input0_int8_test_data_size,
g_pad23_golden_int8_test_data,
g_pad23_golden_int8_test_data_size,
"pad23 test");
}


TF_LITE_MICRO_TEST(pad24_test) {tflite::micro::RunModel(
g_pad24_model_data,
g_pad24_input0_int8_test_data,
g_pad24_input0_int8_test_data_size,
g_pad24_golden_int8_test_data,
g_pad24_golden_int8_test_data_size,
"pad24 test");
}


TF_LITE_MICRO_TEST(pad25_test) {tflite::micro::RunModel(
g_pad25_model_data,
g_pad25_input0_int8_test_data,
g_pad25_input0_int8_test_data_size,
g_pad25_golden_int8_test_data,
g_pad25_golden_int8_test_data_size,
"pad25 test");
}


TF_LITE_MICRO_TESTS_END
