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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/kernels/helia/tests/arg_extrema_fp16_models.h"
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

struct CallRecord {
  const void* input;
  int32_t* output;
  cmsis_nn_dims dims;
  int32_t axis;
  int count;
};

#if HELIA_ARG_EXTREMA_LINK_WRAP && ARM_NN_ENABLE_F16
CallRecord g_min_call = {};
CallRecord g_max_call = {};
arm_cmsis_nn_status g_min_status = ARM_CMSIS_NN_SUCCESS;
arm_cmsis_nn_status g_max_status = ARM_CMSIS_NN_SUCCESS;

extern "C" arm_cmsis_nn_status __real_arm_argmin_f16(const float16_t*,
                                                     const cmsis_nn_dims*,
                                                     int32_t, int32_t*);
extern "C" arm_cmsis_nn_status __real_arm_argmax_f16(const float16_t*,
                                                     const cmsis_nn_dims*,
                                                     int32_t, int32_t*);

extern "C" arm_cmsis_nn_status __wrap_arm_argmin_f16(const float16_t* input,
                                                     const cmsis_nn_dims* dims,
                                                     int32_t axis,
                                                     int32_t* output) {
  g_min_call = {input, output, *dims, axis, g_min_call.count + 1};
  return g_min_status == ARM_CMSIS_NN_SUCCESS
             ? __real_arm_argmin_f16(input, dims, axis, output)
             : g_min_status;
}

extern "C" arm_cmsis_nn_status __wrap_arm_argmax_f16(const float16_t* input,
                                                     const cmsis_nn_dims* dims,
                                                     int32_t axis,
                                                     int32_t* output) {
  g_max_call = {input, output, *dims, axis, g_max_call.count + 1};
  return g_max_status == ARM_CMSIS_NN_SUCCESS
             ? __real_arm_argmax_f16(input, dims, axis, output)
             : g_max_status;
}
#endif

void ResetCalls() {
#if HELIA_ARG_EXTREMA_LINK_WRAP && ARM_NN_ENABLE_F16
  g_min_call = {};
  g_max_call = {};
  g_min_status = ARM_CMSIS_NN_SUCCESS;
  g_max_status = ARM_CMSIS_NN_SUCCESS;
#endif
}

int CallCount(bool maximum) {
#if HELIA_ARG_EXTREMA_LINK_WRAP && ARM_NN_ENABLE_F16
  return maximum ? g_max_call.count : g_min_call.count;
#else
  (void)maximum;
  return 0;
#endif
}

[[maybe_unused]] const CallRecord& LastCall(bool maximum) {
#if HELIA_ARG_EXTREMA_LINK_WRAP && ARM_NN_ENABLE_F16
  return maximum ? g_max_call : g_min_call;
#else
  static const CallRecord empty = {};
  (void)maximum;
  return empty;
#endif
}

uint32_t OrderedKey(uint16_t bits) {
  if ((bits & 0x7fff) == 0) return 0x8000;
  return (bits & 0x8000) ? (~bits & 0xffff) : (bits | 0x8000);
}

bool IsNan(uint16_t bits) {
  return (bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0;
}

int32_t OracleLine(const uint16_t* input, size_t base, size_t reduction,
                   size_t inner, bool maximum) {
  for (size_t k = 0; k < reduction; ++k) {
    if (IsNan(input[base + k * inner])) return static_cast<int32_t>(k);
  }
  int32_t best_index = 0;
  uint32_t best_key = OrderedKey(input[base]);
  for (size_t k = 1; k < reduction; ++k) {
    const uint32_t key = OrderedKey(input[base + k * inner]);
    if (maximum ? key > best_key : key < best_key) {
      best_key = key;
      best_index = static_cast<int32_t>(k);
    }
  }
  return best_index;
}

void OracleTensor(const uint16_t* input, const int* dims, int rank, int axis,
                  bool maximum, int32_t* output) {
  if (axis < 0) axis += rank;
  size_t outer = 1;
  size_t inner = 1;
  for (int i = 0; i < axis; ++i) outer *= static_cast<size_t>(dims[i]);
  for (int i = axis + 1; i < rank; ++i) {
    inner *= static_cast<size_t>(dims[i]);
  }
  const size_t reduction = static_cast<size_t>(dims[axis]);
  for (size_t o = 0; o < outer; ++o) {
    for (size_t i = 0; i < inner; ++i) {
      const size_t base = o * reduction * inner + i;
      output[o * inner + i] =
          OracleLine(input, base, reduction, inner, maximum);
    }
  }
}

void ExpectArray(const int32_t* expected, const int32_t* actual, int count) {
  for (int i = 0; i < count; ++i) EXPECT_EQ(expected[i], actual[i]);
}

void CheckModel(const unsigned char* model_data, tflite::BuiltinOperator code,
                const int* input_shape, int input_rank, int32_t axis,
                const int* output_shape, int output_rank) {
  const tflite::Model* model = tflite::GetModel(model_data);
  ASSERT_EQ(model->version(), static_cast<unsigned int>(TFLITE_SCHEMA_VERSION));
  ASSERT_EQ(model->subgraphs()->size(), 1u);
  const auto* graph = model->subgraphs()->Get(0);
  ASSERT_EQ(graph->operators()->size(), 1u);
  ASSERT_EQ(graph->tensors()->size(), 3u);
  ASSERT_EQ(graph->inputs()->size(), 1u);
  ASSERT_EQ(graph->inputs()->Get(0), 0);
  ASSERT_EQ(graph->outputs()->size(), 1u);
  ASSERT_EQ(graph->outputs()->Get(0), 2);
  ASSERT_EQ(graph->tensors()->Get(0)->type(), tflite::TensorType_FLOAT16);
  ASSERT_EQ(graph->tensors()->Get(1)->type(), tflite::TensorType_INT32);
  ASSERT_EQ(graph->tensors()->Get(2)->type(), tflite::TensorType_INT32);
  ASSERT_EQ(graph->tensors()->Get(1)->shape()->size(), 1u);
  ASSERT_EQ(graph->tensors()->Get(1)->shape()->Get(0), 1);
  ASSERT_EQ(graph->tensors()->Get(0)->shape()->size(),
            static_cast<unsigned int>(input_rank));
  ASSERT_EQ(graph->tensors()->Get(2)->shape()->size(),
            static_cast<unsigned int>(output_rank));
  for (int i = 0; i < input_rank; ++i) {
    EXPECT_EQ(graph->tensors()->Get(0)->shape()->Get(i), input_shape[i]);
  }
  for (int i = 0; i < output_rank; ++i) {
    EXPECT_EQ(graph->tensors()->Get(2)->shape()->Get(i), output_shape[i]);
  }
  const auto* axis_buffer =
      model->buffers()->Get(graph->tensors()->Get(1)->buffer())->data();
  ASSERT_EQ(axis_buffer->size(), sizeof(axis));
  int32_t stored_axis;
  std::memcpy(&stored_axis, axis_buffer->data(), sizeof(stored_axis));
  EXPECT_EQ(stored_axis, axis);
  const auto* op = graph->operators()->Get(0);
  const int opcode_index = op->opcode_index();
  EXPECT_EQ(model->operator_codes()->Get(opcode_index)->builtin_code(), code);
  if (code == tflite::BuiltinOperator_ARG_MIN) {
    ASSERT_NE(op->builtin_options_as_ArgMinOptions(), nullptr);
    EXPECT_EQ(op->builtin_options_as_ArgMinOptions()->output_type(),
              tflite::TensorType_INT32);
  } else {
    ASSERT_NE(op->builtin_options_as_ArgMaxOptions(), nullptr);
    EXPECT_EQ(op->builtin_options_as_ArgMaxOptions()->output_type(),
              tflite::TensorType_INT32);
  }
}

TfLiteStatus AllocateModel(const unsigned char* model_data,
                           tflite::BuiltinOperator code, bool invoke,
                           const uint16_t* input = nullptr,
                           const int32_t* expected = nullptr,
                           int input_count = 0, int output_count = 0) {
  const tflite::Model* model = tflite::GetModel(model_data);
  tflite::MicroMutableOpResolver<1> resolver;
  if (code == tflite::BuiltinOperator_ARG_MIN) {
    EXPECT_EQ(resolver.AddArgMin(), kTfLiteOk);
  } else {
    EXPECT_EQ(resolver.AddArgMax(), kTfLiteOk);
  }
  alignas(16) uint8_t arena[8192];
  tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof(arena));
  const TfLiteStatus allocation_status = interpreter.AllocateTensors();
  if (allocation_status != kTfLiteOk || !invoke) return allocation_status;
  EXPECT_EQ(interpreter.input(0)->type, kTfLiteFloat16);
  EXPECT_EQ(interpreter.output(0)->type, kTfLiteInt32);
  EXPECT_EQ(interpreter.input(0)->bytes,
            static_cast<size_t>(input_count) * sizeof(uint16_t));
  EXPECT_EQ(interpreter.output(0)->bytes,
            static_cast<size_t>(output_count) * sizeof(int32_t));
  if (input_count != 0) {
    std::memcpy(interpreter.input(0)->data.raw, input,
                input_count * sizeof(uint16_t));
  }
  const TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status == kTfLiteOk && output_count != 0) {
    ExpectArray(expected, interpreter.output(0)->data.i32, output_count);
  }
  return invoke_status;
}

void CheckDirect(const TFLMRegistration& registration, bool maximum,
                 uint16_t* input, int* input_dims, int32_t* axis,
                 int* output_dims, int32_t* output, int output_count,
                 const cmsis_nn_dims& expected_dims, int expected_axis) {
  int axis_dims[] = {1, 1};
  TfLiteTensor tensors[] = {
      CreateTensor(input, IntArrayFromInts(input_dims), false, kTfLiteFloat16),
      CreateTensor(axis, IntArrayFromInts(axis_dims)),
      CreateTensor(output, IntArrayFromInts(output_dims)),
  };
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  tflite::micro::KernelRunner runner(registration, tensors, 3,
                                     IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), nullptr);
#if ARM_NN_ENABLE_F16
  int32_t expected[16] = {};
  OracleTensor(input, input_dims + 1, input_dims[0], *axis, maximum, expected);
  const int calls_before = CallCount(maximum);
  ASSERT_EQ(runner.InitAndPrepare(), kTfLiteOk);
  ASSERT_EQ(runner.Invoke(), kTfLiteOk);
  ExpectArray(expected, output, output_count);
#if HELIA_ARG_EXTREMA_LINK_WRAP
  EXPECT_EQ(CallCount(maximum), calls_before + 1);
  const CallRecord& call = LastCall(maximum);
  EXPECT_EQ(call.input, input);
  EXPECT_EQ(call.output, output);
  EXPECT_EQ(call.dims.n, expected_dims.n);
  EXPECT_EQ(call.dims.h, expected_dims.h);
  EXPECT_EQ(call.dims.w, expected_dims.w);
  EXPECT_EQ(call.dims.c, expected_dims.c);
  EXPECT_EQ(call.axis, expected_axis);
#else
  (void)expected_dims;
  (void)expected_axis;
#endif
#else
  (void)maximum;
  (void)output_count;
  (void)expected_dims;
  (void)expected_axis;
  EXPECT_EQ(runner.InitAndPrepare(), kTfLiteError);
#endif
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
}

TfLiteStatus PrepareShape(const TFLMRegistration& registration, int* input_dims,
                          int32_t* axis, int* axis_dims, int* output_dims,
                          void* input_data, void* output_data,
                          TfLiteType axis_type = kTfLiteInt32,
                          TfLiteType output_type = kTfLiteInt32) {
  int storage_dims[] = {1, 1};
  TfLiteTensor tensors[] = {
      CreateTensor(static_cast<uint16_t*>(input_data),
                   IntArrayFromInts(storage_dims), false, kTfLiteFloat16),
      CreateTensor(axis, IntArrayFromInts(axis_dims), false, axis_type),
      CreateTensor(static_cast<int32_t*>(output_data),
                   IntArrayFromInts(storage_dims), false, output_type),
  };
  tensors[0].dims = IntArrayFromInts(input_dims);
  tensors[2].dims = IntArrayFromInts(output_dims);
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  tflite::micro::KernelRunner runner(registration, tensors, 3,
                                     IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), nullptr);
  const TfLiteStatus status = runner.InitAndPrepare();
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
  return status;
}

}  // namespace

TEST(HeliaArgExtremaFp16Test, SerializedValidModels) {
  int input_shape[] = {2, 1, 3, 9};
  int output_shape[] = {2, 1, 3};
  CheckModel(kArgMinF16ModelData, tflite::BuiltinOperator_ARG_MIN, input_shape,
             4, 3, output_shape, 3);
  CheckModel(kArgMaxF16NegativeAxisModelData, tflite::BuiltinOperator_ARG_MAX,
             input_shape, 4, -1, output_shape, 3);

  constexpr uint16_t finite[] = {0xbc00, 0xb800, 0x0001, 0x0000, 0x8000,
                                 0x3400, 0x3c00, 0x4000, 0x4400, 0x4800,
                                 0x4a00, 0x4c00, 0x5000};
  uint16_t input[54];
  for (int i = 0; i < 54; ++i) input[i] = finite[(i * 7 + i / 9) % 13];
  int32_t expected_min[6];
  int32_t expected_max[6];
  OracleTensor(input, input_shape, 4, 3, false, expected_min);
  OracleTensor(input, input_shape, 4, -1, true, expected_max);

  ResetCalls();
#if ARM_NN_ENABLE_F16
  EXPECT_EQ(AllocateModel(kArgMinF16ModelData, tflite::BuiltinOperator_ARG_MIN,
                          true, input, expected_min, 54, 6),
            kTfLiteOk);
  EXPECT_EQ(AllocateModel(kArgMaxF16NegativeAxisModelData,
                          tflite::BuiltinOperator_ARG_MAX, true, input,
                          expected_max, 54, 6),
            kTfLiteOk);
#if HELIA_ARG_EXTREMA_LINK_WRAP
  EXPECT_EQ(CallCount(false), 1);
  EXPECT_EQ(CallCount(true), 1);
#endif
#else
  EXPECT_EQ(AllocateModel(kArgMinF16ModelData, tflite::BuiltinOperator_ARG_MIN,
                          false),
            kTfLiteError);
  EXPECT_EQ(AllocateModel(kArgMaxF16NegativeAxisModelData,
                          tflite::BuiltinOperator_ARG_MAX, false),
            kTfLiteError);
#endif
}

TEST(HeliaArgExtremaFp16Test, SerializedValidation) {
  ResetCalls();
  int input_shape[] = {2, 1, 3, 9};
  int output_shape[] = {2, 1, 3};
  int rank_five_shape[] = {1, 2, 1, 3, 9};
  int rank_five_output[] = {1, 2, 1, 3};
  int mismatch_output[] = {2, 1, 3, 1};
  int reduced_empty_shape[] = {2, 1, 3, 0};
  int valid_empty_shape[] = {9, 0};
  int valid_empty_output[] = {0};
  CheckModel(kArgMinF16PositiveAxisOutOfRangeModelData,
             tflite::BuiltinOperator_ARG_MIN, input_shape, 4, 4, output_shape,
             3);
  CheckModel(kArgMaxF16NegativeAxisOutOfRangeModelData,
             tflite::BuiltinOperator_ARG_MAX, input_shape, 4, -5, output_shape,
             3);
  CheckModel(kArgMinF16RankFiveModelData, tflite::BuiltinOperator_ARG_MIN,
             rank_five_shape, 5, 4, rank_five_output, 4);
  CheckModel(kArgMaxF16OutputMismatchModelData, tflite::BuiltinOperator_ARG_MAX,
             input_shape, 4, 3, mismatch_output, 4);
  CheckModel(kArgMinF16ReducedEmptyModelData, tflite::BuiltinOperator_ARG_MIN,
             reduced_empty_shape, 4, 3, output_shape, 3);
  CheckModel(kArgMaxF16ValidEmptyModelData, tflite::BuiltinOperator_ARG_MAX,
             valid_empty_shape, 2, 0, valid_empty_output, 1);

  EXPECT_EQ(AllocateModel(kArgMinF16PositiveAxisOutOfRangeModelData,
                          tflite::BuiltinOperator_ARG_MIN, false),
            kTfLiteError);
  EXPECT_EQ(AllocateModel(kArgMaxF16NegativeAxisOutOfRangeModelData,
                          tflite::BuiltinOperator_ARG_MAX, false),
            kTfLiteError);
  EXPECT_EQ(AllocateModel(kArgMinF16RankFiveModelData,
                          tflite::BuiltinOperator_ARG_MIN, false),
            kTfLiteError);
  EXPECT_EQ(AllocateModel(kArgMaxF16OutputMismatchModelData,
                          tflite::BuiltinOperator_ARG_MAX, false),
            kTfLiteError);
  EXPECT_EQ(AllocateModel(kArgMinF16ReducedEmptyModelData,
                          tflite::BuiltinOperator_ARG_MIN, false),
            kTfLiteError);
#if ARM_NN_ENABLE_F16
  EXPECT_EQ(AllocateModel(kArgMaxF16ValidEmptyModelData,
                          tflite::BuiltinOperator_ARG_MAX, true, nullptr,
                          nullptr, 0, 0),
            kTfLiteOk);
#if HELIA_ARG_EXTREMA_LINK_WRAP
  EXPECT_EQ(CallCount(false), 0);
  EXPECT_EQ(CallCount(true), 1);
#endif
#else
  EXPECT_EQ(AllocateModel(kArgMaxF16ValidEmptyModelData,
                          tflite::BuiltinOperator_ARG_MAX, false),
            kTfLiteError);
#endif
}

TEST(HeliaArgExtremaFp16Test, FiniteAxesAndRankPadding) {
  ResetCalls();
  constexpr uint16_t finite[] = {0xc800, 0xc400, 0xc000, 0xbc00, 0x0001,
                                 0x0000, 0x8000, 0x3400, 0x3c00, 0x4000,
                                 0x4200, 0x4400, 0x4800};
  uint16_t rank_three_input[30];
  for (int i = 0; i < 30; ++i) {
    rank_three_input[i] = finite[(i * 5 + i / 5) % 13];
  }
  int rank_three_dims[] = {3, 2, 3, 5};
  int rank_three_output_dims[] = {2, 2, 5};
  int32_t axis = 1;
  int32_t output[10] = {};
  CheckDirect(tflite::Register_ARG_MIN(), false, rank_three_input,
              rank_three_dims, &axis, rank_three_output_dims, output, 10,
              cmsis_nn_dims{1, 2, 3, 5}, 2);
  axis = -2;
  CheckDirect(tflite::Register_ARG_MAX(), true, rank_three_input,
              rank_three_dims, &axis, rank_three_output_dims, output, 10,
              cmsis_nn_dims{1, 2, 3, 5}, 2);

  uint16_t rank_one_input[] = {0x3c00, 0xbc00, 0x4000, 0x0001, 0x8000,
                               0x4400, 0xc400, 0x3400, 0x4800};
  int rank_one_dims[] = {1, 9};
  int scalar_output_dims[] = {0};
  int32_t scalar_output = -1;
  axis = -1;
  CheckDirect(tflite::Register_ARG_MIN(), false, rank_one_input, rank_one_dims,
              &axis, scalar_output_dims, &scalar_output, 1,
              cmsis_nn_dims{1, 1, 1, 9}, 3);
#if ARM_NN_ENABLE_F16
  EXPECT_EQ(scalar_output, 6);
#endif
}

TEST(HeliaArgExtremaFp16Test, SpecialValuesAndTies) {
  ResetCalls();
  struct Case {
    uint16_t values[4];
    int count;
    int32_t minimum;
    int32_t maximum;
  };
  const Case cases[] = {
      {{0x3c00, 0x7e01, 0xc000, 0}, 3, 1, 1},
      {{0x7c01, 0x7e02, 0x4400, 0}, 3, 0, 0},
      {{0x4000, 0xbc00, 0xbc00, 0x4000}, 4, 1, 0},
      {{0x0000, 0x8000, 0, 0}, 2, 0, 0},
      {{0x8000, 0x0000, 0, 0}, 2, 0, 0},
      {{0xfc00, 0x8001, 0x0001, 0x7c00}, 4, 0, 3},
  };
  int32_t axis = 0;
  int scalar_output_dims[] = {0};
  for (const Case& test : cases) {
    int input_dims[] = {1, test.count};
    int32_t output = -1;
    CheckDirect(tflite::Register_ARG_MIN(), false,
                const_cast<uint16_t*>(test.values), input_dims, &axis,
                scalar_output_dims, &output, 1,
                cmsis_nn_dims{1, 1, 1, test.count}, 3);
#if ARM_NN_ENABLE_F16
    EXPECT_EQ(output, test.minimum);
#endif
    CheckDirect(tflite::Register_ARG_MAX(), true,
                const_cast<uint16_t*>(test.values), input_dims, &axis,
                scalar_output_dims, &output, 1,
                cmsis_nn_dims{1, 1, 1, test.count}, 3);
#if ARM_NN_ENABLE_F16
    EXPECT_EQ(output, test.maximum);
#endif
  }
}

TEST(HeliaArgExtremaFp16Test, BoundariesStatusAndPointerRefresh) {
  ResetCalls();
  uint16_t input = 0x3c00;
  int32_t output = 0x5a5a5a5a;
  int32_t axis = 0;
  int axis_dims[] = {1, 1};
  int one_dim[] = {1, 1};
  int scalar_output[] = {0};
  int rank_zero[] = {0};
  int axis_two_dims[] = {1, 2};
  int32_t axes[2] = {0, 0};
  int rank_five[] = {5, 1, 1, 1, 1, 1};
  int rank_five_output[] = {4, 1, 1, 1, 1};
  int bad_output[] = {1, 1};
  int reduced_empty[] = {2, 1, 0};
  int reduced_empty_output[] = {1, 1};
  int32_t axis_one = 1;
  int negative_extent[] = {2, -1, 1};
  int negative_output[] = {1, -1};
  int input_byte_overflow[] = {4, 1, 1, 2, INT_MAX};
  int input_overflow_output[] = {3, 1, 1, 2};
  int32_t last_axis = 3;
  int output_byte_overflow[] = {4, 1, 1, 1, 536870912};
  int output_overflow_shape[] = {3, 1, 1, 536870912};
  int huge_then_zero[] = {4, INT_MAX, INT_MAX, 0, 1};
  int huge_zero_output[] = {3, INT_MAX, INT_MAX, 0};

  for (const TFLMRegistration registration :
       {tflite::Register_ARG_MIN(), tflite::Register_ARG_MAX()}) {
    EXPECT_EQ(PrepareShape(registration, rank_zero, &axis, axis_dims,
                           scalar_output, &input, &output),
              kTfLiteError);
    EXPECT_EQ(PrepareShape(registration, rank_five, &axis, axis_dims,
                           rank_five_output, &input, &output),
              kTfLiteError);
    EXPECT_EQ(PrepareShape(registration, one_dim, axes, axis_two_dims,
                           scalar_output, &input, &output),
              kTfLiteError);
    EXPECT_EQ(PrepareShape(registration, one_dim, nullptr, axis_dims,
                           scalar_output, &input, &output),
              kTfLiteError);
    EXPECT_EQ(PrepareShape(registration, one_dim, &axis, axis_dims, bad_output,
                           &input, &output),
              kTfLiteError);
    EXPECT_EQ(PrepareShape(registration, reduced_empty, &axis_one, axis_dims,
                           reduced_empty_output, nullptr, nullptr),
              kTfLiteError);
    EXPECT_EQ(PrepareShape(registration, negative_extent, &axis_one, axis_dims,
                           negative_output, &input, &output),
              kTfLiteError);
    EXPECT_EQ(PrepareShape(registration, input_byte_overflow, &last_axis,
                           axis_dims, input_overflow_output, &input, &output),
              kTfLiteError);
    axis = 0;
    EXPECT_EQ(PrepareShape(registration, output_byte_overflow, &axis, axis_dims,
                           output_overflow_shape, &input, &output),
              kTfLiteError);
#if ARM_NN_ENABLE_F16
    last_axis = 3;
    EXPECT_EQ(PrepareShape(registration, huge_then_zero, &last_axis, axis_dims,
                           huge_zero_output, nullptr, nullptr),
              kTfLiteOk);
#else
    EXPECT_EQ(PrepareShape(registration, huge_then_zero, &last_axis, axis_dims,
                           huge_zero_output, nullptr, nullptr),
              kTfLiteError);
#endif
    EXPECT_EQ(PrepareShape(registration, one_dim, &axis, axis_dims,
                           scalar_output, &input, &output, kTfLiteFloat16),
              kTfLiteError);
    EXPECT_EQ(
        PrepareShape(registration, one_dim, &axis, axis_dims, scalar_output,
                     &input, &output, kTfLiteInt32, kTfLiteFloat16),
        kTfLiteError);
  }

#if ARM_NN_ENABLE_F16
  uint16_t first_input[] = {0x3c00, 0x4000, 0x4200, 0x4400, 0x4800,
                            0x4a00, 0x4c00, 0x5000, 0x5200};
  uint16_t second_input[] = {0x5200, 0x5000, 0x4c00, 0x4a00, 0x4800,
                             0x4400, 0x4200, 0x4000, 0x3c00};
  int32_t first_axis = 1;
  int32_t second_axis = 0;
  int matrix_dims[] = {2, 3, 3};
  int vector_dims[] = {1, 3};
  int32_t first_guarded[] = {0x13572468, -1, -1, -1, 0x24681357};
  int32_t second_guarded[] = {0x13572468, -1, -1, -1, 0x24681357};
  TfLiteTensor tensors[] = {
      CreateTensor(first_input, IntArrayFromInts(matrix_dims), false,
                   kTfLiteFloat16),
      CreateTensor(&first_axis, IntArrayFromInts(axis_dims)),
      CreateTensor(first_guarded + 1, IntArrayFromInts(vector_dims)),
  };
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = tflite::Register_ARG_MIN();
  tflite::micro::KernelRunner runner(registration, tensors, 3,
                                     IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), nullptr);
  ASSERT_EQ(runner.InitAndPrepare(), kTfLiteOk);
  ASSERT_EQ(runner.Invoke(), kTfLiteOk);
  int32_t expected[3];
  OracleTensor(first_input, matrix_dims + 1, 2, first_axis, false, expected);
  ExpectArray(expected, first_guarded + 1, 3);

  tensors[0].data.raw = reinterpret_cast<char*>(second_input);
  tensors[1].data.i32 = &second_axis;
  tensors[2].data.i32 = second_guarded + 1;
  ASSERT_EQ(runner.Invoke(), kTfLiteOk);
  OracleTensor(second_input, matrix_dims + 1, 2, second_axis, false, expected);
  ExpectArray(expected, second_guarded + 1, 3);
#if HELIA_ARG_EXTREMA_LINK_WRAP
  const CallRecord& refreshed = LastCall(false);
  EXPECT_EQ(refreshed.input, second_input);
  EXPECT_EQ(refreshed.output, second_guarded + 1);
  EXPECT_EQ(refreshed.axis, 2);
#endif
  EXPECT_EQ(second_guarded[0], 0x13572468);
  EXPECT_EQ(second_guarded[4], 0x24681357);

  const int calls_before_invalid = CallCount(false);
  second_axis = 2;
  std::memset(second_guarded + 1, 0x5a, 3 * sizeof(int32_t));
  EXPECT_EQ(runner.Invoke(), kTfLiteError);
  EXPECT_EQ(CallCount(false), calls_before_invalid);
  EXPECT_EQ(second_guarded[0], 0x13572468);
  EXPECT_EQ(second_guarded[4], 0x24681357);
  for (int i = 1; i <= 3; ++i) EXPECT_EQ(second_guarded[i], 0x5a5a5a5a);

  second_axis = 1;
  matrix_dims[1] = 4;
  vector_dims[1] = 4;
  EXPECT_EQ(runner.Invoke(), kTfLiteError);
  EXPECT_EQ(CallCount(false), calls_before_invalid);
  matrix_dims[1] = 3;
  vector_dims[1] = 3;
  tensors[0].data.raw = nullptr;
  EXPECT_EQ(runner.Invoke(), kTfLiteError);
  EXPECT_EQ(CallCount(false), calls_before_invalid);
  tensors[0].data.raw = reinterpret_cast<char*>(second_input);
  tensors[2].data.raw = nullptr;
  EXPECT_EQ(runner.Invoke(), kTfLiteError);
  EXPECT_EQ(CallCount(false), calls_before_invalid);
  tensors[2].data.i32 = second_guarded + 1;
  tensors[1].data.raw = nullptr;
  EXPECT_EQ(runner.Invoke(), kTfLiteError);
  EXPECT_EQ(CallCount(false), calls_before_invalid);
  tensors[1].data.i32 = &second_axis;

#if HELIA_ARG_EXTREMA_LINK_WRAP
  std::memset(second_guarded + 1, 0x5a, 3 * sizeof(int32_t));
  g_min_status = ARM_CMSIS_NN_ARG_ERROR;
  EXPECT_EQ(runner.Invoke(), kTfLiteError);
  EXPECT_EQ(CallCount(false), calls_before_invalid + 1);
  for (int i = 1; i <= 3; ++i) EXPECT_EQ(second_guarded[i], 0x5a5a5a5a);
  g_min_status = ARM_CMSIS_NN_SUCCESS;
#endif
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
#endif
}

TEST(HeliaArgExtremaFp16Test, ReferenceRoutesUnchanged) {
  ResetCalls();
  float nan = std::numeric_limits<float>::quiet_NaN();
  float float_input[] = {1.0f, nan, -2.0f};
  int32_t axis = 0;
  int input_dims[] = {1, 3};
  int axis_dims[] = {1, 1};
  int scalar_dims[] = {0};
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  for (const auto item : {false, true}) {
    int32_t output = -1;
    TfLiteTensor tensors[] = {
        CreateTensor(float_input, IntArrayFromInts(input_dims)),
        CreateTensor(&axis, IntArrayFromInts(axis_dims)),
        CreateTensor(&output, IntArrayFromInts(scalar_dims)),
    };
    tflite::micro::KernelRunner runner(
        item ? tflite::Register_ARG_MAX() : tflite::Register_ARG_MIN(), tensors,
        3, IntArrayFromInts(inputs), IntArrayFromInts(outputs), nullptr);
    ASSERT_EQ(runner.InitAndPrepare(), kTfLiteOk);
    ASSERT_EQ(runner.Invoke(), kTfLiteOk);
    EXPECT_EQ(output, item ? 0 : 2);
    EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
  }

  int8_t int8_input[] = {3, -4, 7};
  int32_t int8_output = -1;
  TfLiteTensor int8_tensors[] = {
      CreateTensor(int8_input, IntArrayFromInts(input_dims), false,
                   kTfLiteInt8),
      CreateTensor(&axis, IntArrayFromInts(axis_dims)),
      CreateTensor(&int8_output, IntArrayFromInts(scalar_dims)),
  };
  const TFLMRegistration int8_registration = tflite::Register_ARG_MAX();
  tflite::micro::KernelRunner int8_runner(int8_registration, int8_tensors, 3,
                                          IntArrayFromInts(inputs),
                                          IntArrayFromInts(outputs), nullptr);
  ASSERT_EQ(int8_runner.InitAndPrepare(), kTfLiteOk);
  ASSERT_EQ(int8_runner.Invoke(), kTfLiteOk);
  EXPECT_EQ(int8_output, 2);

  float rank_five_input[] = {3.0f, -4.0f, 7.0f};
  int rank_five_dims[] = {5, 1, 1, 1, 1, 3};
  int rank_four_output[] = {4, 1, 1, 1, 1};
  int32_t last_axis = 4;
  int32_t rank_five_result = -1;
  TfLiteTensor rank_five_tensors[] = {
      CreateTensor(rank_five_input, IntArrayFromInts(rank_five_dims)),
      CreateTensor(&last_axis, IntArrayFromInts(axis_dims)),
      CreateTensor(&rank_five_result, IntArrayFromInts(rank_four_output)),
  };
  const TFLMRegistration rank_five_registration = tflite::Register_ARG_MIN();
  tflite::micro::KernelRunner rank_five_runner(
      rank_five_registration, rank_five_tensors, 3, IntArrayFromInts(inputs),
      IntArrayFromInts(outputs), nullptr);
  ASSERT_EQ(rank_five_runner.InitAndPrepare(), kTfLiteOk);
  ASSERT_EQ(rank_five_runner.Invoke(), kTfLiteOk);
  EXPECT_EQ(rank_five_result, 1);
  EXPECT_EQ(CallCount(false), 0);
  EXPECT_EQ(CallCount(true), 0);
}

TF_LITE_MICRO_TESTS_MAIN
