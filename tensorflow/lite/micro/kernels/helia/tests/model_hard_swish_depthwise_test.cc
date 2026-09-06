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

// Model-level coverage for HARD_SWISH int8 and float32 DEPTHWISE_CONV_2D.
// The kernel tests drive the kernels through KernelRunner with hand-built
// tensors; nothing else in the tree takes a real .tflite through
// MicroInterpreter for these two operators, so the quantization parameters,
// the builtin option decode and the kernel all get checked together here.
//
// Goldens come from the TFLite reference interpreter at generation time, so a
// failure means the kernels the build selected disagree with reference TFLite
// on the same flatbuffer, not that the expectation drifted.
// see AmbiqAI/ns-cmsis-nn#461, AmbiqAI/ns-cmsis-nn#448
//
// Regenerate the data with
// kernels/helia/tests/gen_model_hard_swish_depthwise.py.

#include <cmath>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/helia/tests/model_hard_swish_depthwise_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

// Sized from the arena_used_bytes() each case logs, taking the largest of the
// three models (3312 bytes on the host) and rounding up; the models are
// fixed, so the only movement is per-target alignment padding.
constexpr size_t kTensorArenaSize = 4096;
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize];

// Bound on the float depthwise output, scaled by the largest golden magnitude
// so it stays an accuracy statement rather than an absolute-magnitude one.
// see AmbiqAI/ns-cmsis-nn#448
constexpr float kDepthwiseRelativeTolerance = 1.0e-5f;

float MaxAbs(const float* values, size_t size) {
  float result = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    const float magnitude = std::fabs(values[i]);
    if (magnitude > result) {
      result = magnitude;
    }
  }
  return result;
}

void CheckDepthwiseModel(const char* label, const unsigned char* model_data,
                         const float* input, size_t input_size,
                         const float* golden, size_t golden_size) {
  tflite::MicroMutableOpResolver<1> resolver;
  ASSERT_EQ(resolver.AddDepthwiseConv2D(), kTfLiteOk);

  tflite::MicroInterpreter interpreter(tflite::GetModel(model_data), resolver,
                                       g_tensor_arena, kTensorArenaSize);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  TfLiteTensor* input_tensor = interpreter.input(0);
  ASSERT_EQ(input_tensor->type, kTfLiteFloat32);
  ASSERT_EQ(input_tensor->bytes, input_size * sizeof(float));
  std::memcpy(input_tensor->data.f, input, input_size * sizeof(float));

  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);

  TfLiteTensor* output_tensor = interpreter.output(0);
  ASSERT_EQ(output_tensor->type, kTfLiteFloat32);
  ASSERT_EQ(output_tensor->bytes, golden_size * sizeof(float));

  const float tolerance =
      kDepthwiseRelativeTolerance * MaxAbs(golden, golden_size);
  float max_error = 0.0f;
  for (size_t i = 0; i < golden_size; ++i) {
    const float error = std::fabs(output_tensor->data.f[i] - golden[i]);
    if (error > max_error) {
      max_error = error;
    }
  }
  MicroPrintf("%s: arena_used_bytes %d, max abs error %f, tolerance %f", label,
              static_cast<int>(interpreter.arena_used_bytes()),
              static_cast<double>(max_error), static_cast<double>(tolerance));
  EXPECT_LE(max_error, tolerance);
}

}  // namespace

// Exact equality: the int8 hard swish path is required to reproduce the TFLM
// reference bit for bit. see AmbiqAI/ns-cmsis-nn#461
TEST(HeliaModelHardSwishDepthwiseTest, HardSwishInt8MatchesReference) {
  tflite::MicroMutableOpResolver<1> resolver;
  ASSERT_EQ(resolver.AddHardSwish(), kTfLiteOk);

  tflite::MicroInterpreter interpreter(tflite::GetModel(kHardSwishS8ModelData),
                                       resolver, g_tensor_arena,
                                       kTensorArenaSize);
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  TfLiteTensor* input_tensor = interpreter.input(0);
  ASSERT_EQ(input_tensor->type, kTfLiteInt8);
  ASSERT_EQ(input_tensor->bytes, kHardSwishS8InputSize);
  // The generator picks an asymmetric input range so this is non-zero; a zero
  // zero point would let a misplaced offset pass.
  ASSERT_NE(input_tensor->params.zero_point, 0);
  std::memcpy(input_tensor->data.int8, kHardSwishS8Input,
              kHardSwishS8InputSize);

  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);

  TfLiteTensor* output_tensor = interpreter.output(0);
  ASSERT_EQ(output_tensor->type, kTfLiteInt8);
  ASSERT_EQ(output_tensor->bytes, kHardSwishS8GoldenSize);
  MicroPrintf("hard_swish_s8: arena_used_bytes %d, input zero point %d",
              static_cast<int>(interpreter.arena_used_bytes()),
              static_cast<int>(input_tensor->params.zero_point));
  for (unsigned int i = 0; i < kHardSwishS8GoldenSize; ++i) {
    EXPECT_EQ(output_tensor->data.int8[i], kHardSwishS8Golden[i]);
  }
}

TEST(HeliaModelHardSwishDepthwiseTest, DepthwiseFloatMatchesReference) {
  CheckDepthwiseModel("depthwise_f32", kDepthwiseF32ModelData,
                      kDepthwiseF32Input, kDepthwiseF32InputSize,
                      kDepthwiseF32Golden, kDepthwiseF32GoldenSize);
}

// depth_multiplier 2 is a different branch of the depthwise dispatch than the
// depth_multiplier 1 case above. see AmbiqAI/ns-cmsis-nn#448
TEST(HeliaModelHardSwishDepthwiseTest, DepthwiseFloatDepthMultiplier2) {
  CheckDepthwiseModel("depthwise_dm2_f32", kDepthwiseDm2F32ModelData,
                      kDepthwiseDm2F32Input, kDepthwiseDm2F32InputSize,
                      kDepthwiseDm2F32Golden, kDepthwiseDm2F32GoldenSize);
}

TF_LITE_MICRO_TESTS_MAIN
