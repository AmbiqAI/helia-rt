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

#include <climits>
#include <cstdint>
#include <cstring>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/kernels/helia/tests/sqrt_rsqrt_fp16_models.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

using tflite::testing::CreateTensor;
using tflite::testing::IntArrayFromInts;

#if HELIA_SQRT_RSQRT_LINK_WRAP && ARM_NN_ENABLE_F16
arm_cmsis_nn_status g_sqrt_status = ARM_CMSIS_NN_SUCCESS;
arm_cmsis_nn_status g_rsqrt_status = ARM_CMSIS_NN_SUCCESS;
int g_sqrt_calls = 0;
int g_rsqrt_calls = 0;

extern "C" arm_cmsis_nn_status __real_arm_nn_sqrt_f16(const float16_t* input,
                                                      float16_t* output,
                                                      int32_t block_size);
extern "C" arm_cmsis_nn_status __real_arm_rsqrt_f16(const float16_t* input,
                                                    float16_t* output,
                                                    int32_t block_size);

extern "C" arm_cmsis_nn_status __wrap_arm_nn_sqrt_f16(const float16_t* input,
                                                      float16_t* output,
                                                      int32_t block_size) {
  ++g_sqrt_calls;
  return g_sqrt_status == ARM_CMSIS_NN_SUCCESS
             ? __real_arm_nn_sqrt_f16(input, output, block_size)
             : g_sqrt_status;
}

extern "C" arm_cmsis_nn_status __wrap_arm_rsqrt_f16(const float16_t* input,
                                                    float16_t* output,
                                                    int32_t block_size) {
  ++g_rsqrt_calls;
  return g_rsqrt_status == ARM_CMSIS_NN_SUCCESS
             ? __real_arm_rsqrt_f16(input, output, block_size)
             : g_rsqrt_status;
}
#endif

constexpr uint16_t kInputBits[] = {
    0x0000, 0x8000, 0x7c00, 0xfc00, 0xbc00, 0x3c00, 0x0001, 0x03ff, 0x0400,
    0x7bff, 0x7c01, 0xfc03, 0x7e07, 0xfe0b, 0x8001, 0x3bff, 0x3c01,
};
constexpr uint16_t kSqrtBits[] = {
    0x0000, 0x8000, 0x7c00, 0x7e00, 0x7e00, 0x3c00, 0x0c00, 0x1fff, 0x2000,
    0x5bff, 0x7e01, 0xfe03, 0x7e07, 0xfe0b, 0x7e00, 0x3bff, 0x3c00,
};
constexpr uint16_t kRsqrtBits[] = {
    0x7c00, 0xfc00, 0x0000, 0x7e00, 0x7e00, 0x3c00, 0x6c00, 0x5801, 0x5800,
    0x1c00, 0x7e01, 0xfe03, 0x7e07, 0xfe0b, 0x7e00, 0x3c00, 0x3bff,
};

#if ARM_NN_ENABLE_F16
void CheckBits(const uint16_t* actual, const uint16_t* expected, int count) {
  for (int i = 0; i < count; ++i) EXPECT_EQ(expected[i], actual[i]);
}
#endif

void RunModel(const unsigned char* model_data, tflite::BuiltinOperator op,
              const uint16_t* input_bits, const uint16_t* expected, int count,
              bool scalar) {
  const tflite::Model* model = tflite::GetModel(model_data);
  ASSERT_EQ(model->subgraphs()->size(), 1u);
  const auto* subgraph = model->subgraphs()->Get(0);
  ASSERT_EQ(subgraph->operators()->size(), 1u);
  ASSERT_EQ(subgraph->tensors()->Get(0)->type(), tflite::TensorType_FLOAT16);
  ASSERT_EQ(subgraph->tensors()->Get(1)->type(), tflite::TensorType_FLOAT16);
  const auto* input_shape = subgraph->tensors()->Get(0)->shape();
  const auto* output_shape = subgraph->tensors()->Get(1)->shape();
  ASSERT_EQ(input_shape->size(), scalar ? 0u : 1u);
  ASSERT_EQ(output_shape->size(), scalar ? 0u : 1u);
  if (!scalar) {
    ASSERT_EQ(input_shape->Get(0), count);
    ASSERT_EQ(output_shape->Get(0), count);
  }
  const int opcode = subgraph->operators()->Get(0)->opcode_index();
  ASSERT_EQ(model->operator_codes()->Get(opcode)->builtin_code(), op);

  tflite::MicroMutableOpResolver<1> resolver;
  ASSERT_EQ(op == tflite::BuiltinOperator_SQRT ? resolver.AddSqrt()
                                               : resolver.AddRsqrt(),
            kTfLiteOk);
  alignas(16) uint8_t arena[2048];
  tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof(arena));
#if ARM_NN_ENABLE_F16
  ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);
  const TfLiteIntArray* runtime_input_shape = interpreter.input(0)->dims;
  ASSERT_EQ(runtime_input_shape->size, scalar ? 0 : 1);
  if (!scalar) {
    ASSERT_EQ(runtime_input_shape->data[0], count);
  }
  ASSERT_EQ(interpreter.input(0)->bytes,
            static_cast<size_t>(count) * sizeof(uint16_t));
  std::memcpy(interpreter.input(0)->data.raw, input_bits,
              count * sizeof(uint16_t));
  ASSERT_EQ(interpreter.Invoke(), kTfLiteOk);
  const TfLiteIntArray* runtime_output_shape = interpreter.output(0)->dims;
  ASSERT_EQ(runtime_output_shape->size, scalar ? 0 : 1);
  if (!scalar) ASSERT_EQ(runtime_output_shape->data[0], count);
  CheckBits(reinterpret_cast<const uint16_t*>(interpreter.output(0)->data.raw),
            expected, count);
#else
  (void)input_bits;
  (void)expected;
  (void)count;
  EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteError);
#endif
}

void RunBits(const TFLMRegistration& registration, const uint16_t* expected,
             uint16_t in_place_expected, bool is_sqrt) {
  uint16_t input[17];
  std::memcpy(input, kInputBits, sizeof(input));
  uint16_t in_place[17];
  for (auto& value : in_place) value = 0x4400;
  uint16_t guarded[19];
  std::memset(guarded, 0x5a, sizeof(guarded));
  int dims_data[] = {1, 17};
  TfLiteTensor tensors[] = {
      CreateTensor(input, IntArrayFromInts(dims_data), false, kTfLiteFloat16),
      CreateTensor(guarded + 1, IntArrayFromInts(dims_data), false,
                   kTfLiteFloat16),
  };
  int inputs[] = {1, 0}, outputs[] = {1, 1};
  tflite::micro::KernelRunner runner(registration, tensors, 2,
                                     IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), nullptr);
#if ARM_NN_ENABLE_F16
  ASSERT_EQ(runner.InitAndPrepare(), kTfLiteOk);
#if HELIA_SQRT_RSQRT_LINK_WRAP
  const int calls_before = is_sqrt ? g_sqrt_calls : g_rsqrt_calls;
#endif
  ASSERT_EQ(runner.Invoke(), kTfLiteOk);
#if HELIA_SQRT_RSQRT_LINK_WRAP
  EXPECT_EQ(is_sqrt ? g_sqrt_calls : g_rsqrt_calls, calls_before + 1);
#endif
  CheckBits(guarded + 1, expected, 17);
  EXPECT_EQ(guarded[0], 0x5a5a);
  EXPECT_EQ(guarded[18], 0x5a5a);
  uint16_t first_output[17];
  std::memcpy(first_output, guarded + 1, sizeof(first_output));

  tensors[0].data.raw = reinterpret_cast<char*>(in_place);
  tensors[1].data.raw = tensors[0].data.raw;
  ASSERT_EQ(runner.Invoke(), kTfLiteOk);
#if HELIA_SQRT_RSQRT_LINK_WRAP
  EXPECT_EQ(is_sqrt ? g_sqrt_calls : g_rsqrt_calls, calls_before + 2);
#endif
  for (const auto value : in_place) EXPECT_EQ(value, in_place_expected);
  CheckBits(guarded + 1, first_output, 17);

  tensors[0].data.raw = nullptr;
  tensors[1].data.raw = reinterpret_cast<char*>(guarded + 1);
  EXPECT_EQ(runner.Invoke(), kTfLiteError);
  tensors[0].data.raw = reinterpret_cast<char*>(input);
  tensors[1].data.raw = nullptr;
  EXPECT_EQ(runner.Invoke(), kTfLiteError);
#else
  (void)is_sqrt;
  EXPECT_EQ(runner.InitAndPrepare(), kTfLiteError);
#endif
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
}

TfLiteStatus PrepareWithShapes(const TFLMRegistration& registration,
                               int* input_dims, int* output_dims,
                               void* input_data, void* output_data,
                               TfLiteType output_type = kTfLiteFloat16,
                               bool invoke = false) {
  int storage_dims[] = {1, 1};
  TfLiteTensor tensors[] = {
      CreateTensor(static_cast<uint16_t*>(input_data),
                   IntArrayFromInts(storage_dims), false, kTfLiteFloat16),
      CreateTensor(static_cast<uint16_t*>(output_data),
                   IntArrayFromInts(storage_dims), false, output_type),
  };
  tensors[0].dims = IntArrayFromInts(input_dims);
  tensors[1].dims = IntArrayFromInts(output_dims);
  int inputs[] = {1, 0}, outputs[] = {1, 1};
  tflite::micro::KernelRunner runner(registration, tensors, 2,
                                     IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), nullptr);
  const TfLiteStatus status = runner.InitAndPrepare();
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
  if (status != kTfLiteOk || !invoke) return status;
  return runner.Invoke();
}

void ExpectPrepareThenInvokeError(const TFLMRegistration& registration,
                                  uint16_t* input, uint16_t* output,
                                  int* dims) {
  TfLiteTensor tensors[] = {
      CreateTensor(input, IntArrayFromInts(dims), false, kTfLiteFloat16),
      CreateTensor(output, IntArrayFromInts(dims), false, kTfLiteFloat16),
  };
  int inputs[] = {1, 0}, outputs[] = {1, 1};
  tflite::micro::KernelRunner runner(registration, tensors, 2,
                                     IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), nullptr);
  ASSERT_EQ(runner.InitAndPrepare(), kTfLiteOk);
  ASSERT_TRUE(runner.ValidateTempBufferDeallocated());
  EXPECT_EQ(runner.Invoke(), kTfLiteError);
}

}  // namespace

TEST(HeliaSqrtRsqrtFp16Test, ActualModelsUseFp16Registrations) {
  RunModel(kSqrtF16ModelData, tflite::BuiltinOperator_SQRT,
           kSqrtRsqrtModelInputBits, kSqrtF16ModelExpectedBits, 9, false);
  RunModel(kRsqrtF16ModelData, tflite::BuiltinOperator_RSQRT,
           kSqrtRsqrtModelInputBits, kRsqrtF16ModelExpectedBits, 9, false);
  RunModel(kSqrtF16ScalarModelData, tflite::BuiltinOperator_SQRT,
           kSqrtRsqrtScalarModelInputBits, kSqrtF16ScalarModelExpectedBits, 1,
           true);
  RunModel(kRsqrtF16ScalarModelData, tflite::BuiltinOperator_RSQRT,
           kSqrtRsqrtScalarModelInputBits, kRsqrtF16ScalarModelExpectedBits, 1,
           true);
}

TEST(HeliaSqrtRsqrtFp16Test, FiniteSpecialTailAndInPlace) {
  RunBits(tflite::Register_SQRT(), kSqrtBits, 0x4000, true);
  RunBits(tflite::Register_RSQRT(), kRsqrtBits, 0x3800, false);
}

TEST(HeliaSqrtRsqrtFp16Test, ScalarAndMultiRank) {
  uint16_t input[] = {0x4000, 0x4400, 0x4800, 0x5000};
  uint16_t output[4] = {};
  int scalar[] = {0};
  int rank3[] = {3, 1, 2, 2};
#if ARM_NN_ENABLE_F16
  EXPECT_EQ(PrepareWithShapes(tflite::Register_SQRT(), scalar, scalar, input,
                              output, kTfLiteFloat16, true),
            kTfLiteOk);
  EXPECT_EQ(output[0], 0x3da8);
  EXPECT_EQ(PrepareWithShapes(tflite::Register_RSQRT(), rank3, rank3, input,
                              output, kTfLiteFloat16, true),
            kTfLiteOk);
  constexpr uint16_t expected[] = {0x39a8, 0x3800, 0x35a8, 0x31a8};
  CheckBits(output, expected, 4);
#else
  EXPECT_EQ(
      PrepareWithShapes(tflite::Register_SQRT(), scalar, scalar, input, output),
      kTfLiteError);
  EXPECT_EQ(
      PrepareWithShapes(tflite::Register_RSQRT(), rank3, rank3, input, output),
      kTfLiteError);
#endif
}

TEST(HeliaSqrtRsqrtFp16Test, MetadataEmptyAndNull) {
  uint16_t value = 0x3c00;
  int one[] = {1, 1};
  int two[] = {1, 2};
  int negative[] = {1, -1};
  int overflow[] = {2, 46341, 46341};
  int two_by_two[] = {2, 2, 2};
  int one_by_four[] = {2, 1, 4};
  int scalar[] = {0};
  int empty_after_large_prefix[] = {3, INT_MAX, INT_MAX, 0};
  int empty_with_negative[] = {2, 0, -1};
  for (const auto registration :
       {tflite::Register_SQRT(), tflite::Register_RSQRT()}) {
    EXPECT_EQ(PrepareWithShapes(registration, one, two, &value, &value),
              kTfLiteError);
    EXPECT_EQ(
        PrepareWithShapes(registration, negative, negative, &value, &value),
        kTfLiteError);
    EXPECT_EQ(
        PrepareWithShapes(registration, overflow, overflow, &value, &value),
        kTfLiteError);
    EXPECT_EQ(PrepareWithShapes(registration, two_by_two, one_by_four, &value,
                                &value),
              kTfLiteError);
    EXPECT_EQ(PrepareWithShapes(registration, scalar, one, &value, &value),
              kTfLiteError);
    EXPECT_EQ(PrepareWithShapes(registration, empty_with_negative,
                                empty_with_negative, nullptr, nullptr),
              kTfLiteError);
    EXPECT_EQ(PrepareWithShapes(registration, one, one, &value, &value,
                                kTfLiteFloat32),
              kTfLiteError);
#if ARM_NN_ENABLE_F16
#if HELIA_SQRT_RSQRT_LINK_WRAP
    const int calls_before = g_sqrt_calls + g_rsqrt_calls;
#endif
    EXPECT_EQ(PrepareWithShapes(registration, empty_after_large_prefix,
                                empty_after_large_prefix, nullptr, nullptr,
                                kTfLiteFloat16, true),
              kTfLiteOk);
#if HELIA_SQRT_RSQRT_LINK_WRAP
    EXPECT_EQ(g_sqrt_calls + g_rsqrt_calls, calls_before);
#endif
    EXPECT_EQ(PrepareWithShapes(registration, one, one, nullptr, &value,
                                kTfLiteFloat16, true),
              kTfLiteError);
    EXPECT_EQ(PrepareWithShapes(registration, one, one, &value, nullptr,
                                kTfLiteFloat16, true),
              kTfLiteError);
#else
    EXPECT_EQ(PrepareWithShapes(registration, empty_after_large_prefix,
                                empty_after_large_prefix, nullptr, nullptr),
              kTfLiteError);
#endif
  }
}

TEST(HeliaSqrtRsqrtFp16Test, StatusAndPointerRefresh) {
#if ARM_NN_ENABLE_F16 && HELIA_SQRT_RSQRT_LINK_WRAP
  uint16_t input = 0x4000;
  uint16_t output = 0x5a5a;
  int dims[] = {1, 1};
  g_sqrt_status = ARM_CMSIS_NN_ARG_ERROR;
  EXPECT_EQ(PrepareWithShapes(tflite::Register_SQRT(), dims, dims, &input,
                              &output, kTfLiteFloat16, true),
            kTfLiteError);
  EXPECT_EQ(output, 0x5a5a);
  g_sqrt_status = ARM_CMSIS_NN_SUCCESS;
  g_rsqrt_status = ARM_CMSIS_NN_ARG_ERROR;
  EXPECT_EQ(PrepareWithShapes(tflite::Register_RSQRT(), dims, dims, &input,
                              &output, kTfLiteFloat16, true),
            kTfLiteError);
  EXPECT_EQ(output, 0x5a5a);
  g_rsqrt_status = ARM_CMSIS_NN_SUCCESS;
#else
  EXPECT_TRUE(true);
#endif
}

TEST(HeliaSqrtRsqrtFp16Test, SiblingAdmissionAndLegacyRoutes) {
  uint16_t input = 0x3c00;
  uint16_t output = 0;
  int dims[] = {1, 1};
  ExpectPrepareThenInvokeError(tflite::Register_SQUARE(), &input, &output,
                               dims);
  ExpectPrepareThenInvokeError(tflite::Register_ABS(), &input, &output, dims);

  float f32_input = 4.0f;
  float f32_output = 0.0f;
  TfLiteTensor tensors[] = {CreateTensor(&f32_input, IntArrayFromInts(dims)),
                            CreateTensor(&f32_output, IntArrayFromInts(dims))};
  int inputs[] = {1, 0}, outputs[] = {1, 1};
  for (const auto registration :
       {tflite::Register_SQRT(), tflite::Register_RSQRT()}) {
    tflite::micro::KernelRunner runner(registration, tensors, 2,
                                       IntArrayFromInts(inputs),
                                       IntArrayFromInts(outputs), nullptr);
    ASSERT_EQ(runner.InitAndPrepare(), kTfLiteOk);
    ASSERT_EQ(runner.Invoke(), kTfLiteOk);
    EXPECT_EQ(f32_output, registration.invoke == tflite::Register_SQRT().invoke
                              ? 2.0f
                              : 0.5f);
  }
}

TF_LITE_MICRO_TESTS_MAIN
