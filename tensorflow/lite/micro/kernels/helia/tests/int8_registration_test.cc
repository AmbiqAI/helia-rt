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

// Register_PAD_INT8() and Register_TRANSPOSE_INT8(), which pad.h and
// transpose.h declare without a definition when CMSIS_NN is defined, as it is
// for every helia build. Linking this binary is the first check; each
// registration then runs an int8 layer. see AmbiqAI/helia-rt#470

#include <cstdint>
#include <cstring>

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/pad.h"
#include "tensorflow/lite/micro/kernels/transpose.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

TEST(HeliaInt8RegistrationTest, PadInt8) {
  using namespace tflite::testing;
  int8_t input[] = {1, 2, 3, 4};
  int32_t paddings[] = {0, 0, 1, 0, 0, 1, 0, 0};
  int8_t output[3 * 3 + 1];
  const int8_t golden[] = {-3, -3, -3, 1, 2, -3, 3, 4, -3};
  int input_dims[] = {4, 1, 2, 2, 1};
  int pad_dims[] = {2, 4, 2};
  int output_dims[] = {4, 1, 3, 3, 1};
  TfLiteTensor tensors[3] = {
      CreateQuantizedTensor(input, IntArrayFromInts(input_dims), 0.5f, -3),
      CreateTensor(paddings, IntArrayFromInts(pad_dims)),
      CreateQuantizedTensor(output, IntArrayFromInts(output_dims), 0.5f, -3),
  };
  tensors[1].allocation_type = kTfLiteMmapRo;
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = tflite::Register_PAD_INT8();
  tflite::micro::KernelRunner runner(registration, tensors, 3,
                                     IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), nullptr);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  memset(output, 0x55, sizeof(output));
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(golden[i], output[i]);
  }
  EXPECT_EQ(0x55, output[9]);
}

TEST(HeliaInt8RegistrationTest, TransposeInt8) {
  using namespace tflite::testing;
  int8_t input[2 * 3 * 4];
  for (int i = 0; i < 24; ++i) {
    input[i] = static_cast<int8_t>(i - 12);
  }
  int32_t perm[] = {2, 0, 1};
  int8_t output[24 + 1];
  int input_dims[] = {3, 2, 3, 4};
  int perm_dims[] = {1, 3};
  int output_dims[] = {3, 4, 2, 3};
  TfLiteTensor tensors[3] = {
      CreateTensor(input, IntArrayFromInts(input_dims)),
      CreateTensor(perm, IntArrayFromInts(perm_dims)),
      CreateTensor(output, IntArrayFromInts(output_dims)),
  };
  tensors[1].allocation_type = kTfLiteMmapRo;
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = tflite::Register_TRANSPOSE_INT8();
  tflite::micro::KernelRunner runner(registration, tensors, 3,
                                     IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), nullptr);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  memset(output, 0x55, sizeof(output));
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  // output[k][i][j] = input[i][j][k]
  for (int k = 0; k < 4; ++k) {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        EXPECT_EQ(input[(i * 3 + j) * 4 + k], output[(k * 2 + i) * 3 + j]);
      }
    }
  }
  EXPECT_EQ(0x55, output[24]);
}

TF_LITE_MICRO_TESTS_MAIN
