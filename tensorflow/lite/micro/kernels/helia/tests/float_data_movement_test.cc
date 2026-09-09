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

#include <cstdint>
#include <cstring>
#include <initializer_list>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/helia/tests/float_data_movement_goldens.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {
using tflite::testing::CreateTensor;
using tflite::testing::IntArrayFromInts;

enum class Op { kSplit, kPack, kUnpack, kFill };
enum class Fault {
  kNone,
  kAxis,
  kCount,
  kShape,
  kType,
  kNonConstant,
  kParameterType,
  kValueShape,
  kNegativeAxis,
  kZeroCount,
  kOutputCount,
  kInputShape,
  kNegativeExtent,
  kOverflow,
};

constexpr int kMaxTensors = 36;
constexpr int kMaxElements = 512;

int Elements(const int* shape) {
  int count = 1;
  for (int i = 1; i <= shape[0]; ++i) count *= shape[i];
  return count;
}

template <typename T>
uint32_t Bits(const T& value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(T));
  return bits;
}

template <typename T>
T Pattern(uint32_t index) {
  uint32_t bits = index;
  if (sizeof(T) == 4) {
    const uint32_t special[] = {0,          0x80000000, 0x7f800000, 0xff800000,
                                0x7fc12345, 0x7f812345, 1,          0x807fffff};
    bits = index < 8 ? special[index] : 0x3f000000 + index * 7919;
  }
  T value;
  memcpy(&value, &bits, sizeof(T));
  return value;
}

// shape describes PACK inputs, FILL output, or SPLIT/UNPACK input.
template <typename T>
void Run(Op op, TfLiteType type, const int* shape, int axis, int count,
         Fault fault = Fault::kNone, uint32_t pattern_base = 0,
         bool unavailable = false, const uint32_t* golden = nullptr) {
  static T input[kMaxTensors][kMaxElements];
  static T output[kMaxTensors][kMaxElements + 2];
  static TfLiteTensor tensors[kMaxTensors];
  int input_shape[10] = {};
  int output_shape[10] = {};
  memcpy(input_shape, shape, (shape[0] + 1) * sizeof(int));
  memcpy(output_shape, shape, (shape[0] + 1) * sizeof(int));
  const int normalized = axis < 0 ? axis + shape[0] + (op == Op::kPack) : axis;
  if (op == Op::kPack) {
    output_shape[0]++;
    for (int d = 0; d < output_shape[0]; ++d) {
      output_shape[d + 1] =
          d == normalized ? count : shape[d + 1 - (d > normalized)];
    }
  } else if (op == Op::kSplit) {
    output_shape[normalized + 1] /= count;
  } else if (op == Op::kUnpack) {
    output_shape[0]--;
    for (int d = 0; d < output_shape[0]; ++d) {
      output_shape[d + 1] = shape[d + 1 + (d >= normalized)];
    }
  }
  const int input_elements = Elements(input_shape);
  const int output_elements = Elements(output_shape);
  const int data_inputs = op == Op::kPack ? count : 1;
  const int data_outputs = op == Op::kSplit || op == Op::kUnpack ? count : 1;
  for (int n = 0; n < data_inputs; ++n) {
    for (int i = 0; i < input_elements; ++i) {
      input[n][i] = Pattern<T>(pattern_base + n * input_elements + i);
    }
  }
  const T sentinel = Pattern<T>(0x5a5a);
  for (int n = 0; n < data_outputs; ++n) {
    for (int i = 0; i < output_elements + 2; ++i) output[n][i] = sentinel;
  }
  int inputs[kMaxTensors] = {};
  int outputs[kMaxTensors] = {};
  int tensor_count = 0;
  int scalar_shape[] = {0};
  int parameter_shape[] = {1, shape[0]};
  int32_t parameter[10] = {};
  int32_t split_axis = fault == Fault::kAxis ? shape[0] : axis;
  if (fault == Fault::kNegativeAxis) split_axis = -shape[0] - 1;
  if (op == Op::kSplit || op == Op::kFill) {
    for (int d = 0; d < shape[0]; ++d) parameter[d] = shape[d + 1];
    tensors[tensor_count] = CreateTensor(
        op == Op::kSplit ? &split_axis : parameter,
        IntArrayFromInts(op == Op::kSplit ? scalar_shape : parameter_shape));
    tensors[tensor_count].allocation_type =
        fault == Fault::kNonConstant ? kTfLiteArenaRw : kTfLiteMmapRo;
    if (fault == Fault::kParameterType)
      tensors[tensor_count].type = kTfLiteFloat32;
    inputs[++inputs[0]] = tensor_count++;
  }
  int bad_value_shape[] = {1, 2};
  int bad_input_shape[10];
  memcpy(bad_input_shape, input_shape, sizeof(input_shape));
  bad_input_shape[input_shape[0]]++;
  for (int n = 0; n < data_inputs; ++n) {
    int* dims = op == Op::kFill ? scalar_shape : input_shape;
    if (fault == Fault::kInputShape && n == 1) dims = bad_input_shape;
    if (fault == Fault::kValueShape && op == Op::kFill) dims = bad_value_shape;
    tensors[tensor_count] =
        CreateTensor(input[n], IntArrayFromInts(dims), false, type);
    inputs[++inputs[0]] = tensor_count++;
  }
  if (fault == Fault::kShape) {
    if (output_shape[0] == 0) {
      output_shape[0] = 1;
      output_shape[1] = 2;
    } else
      output_shape[output_shape[0]]++;
  }
  for (int n = 0; n < data_outputs; ++n) {
    tensors[tensor_count] = CreateTensor(
        output[n] + 1, IntArrayFromInts(output_shape), false, type);
    if (fault == Fault::kType && n == data_outputs - 1)
      tensors[tensor_count].type = kTfLiteInt32;
    outputs[++outputs[0]] = tensor_count++;
  }
  TfLitePackParams pack = {};
  pack.axis = fault == Fault::kAxis ? shape[0] + 1 : axis;
  pack.values_count = count + (fault == Fault::kCount);
  TfLiteUnpackParams unpack = {};
  unpack.axis = fault == Fault::kAxis ? shape[0] : axis;
  unpack.num = count + (fault == Fault::kCount);
  TfLiteSplitParams split = {};
  split.num_splits = count + (fault == Fault::kCount);
  if (fault == Fault::kNegativeAxis) {
    pack.axis = -shape[0] - 2;
    unpack.axis = -shape[0] - 1;
  }
  if (fault == Fault::kZeroCount) {
    pack.values_count = 0;
    unpack.num = 0;
    split.num_splits = 0;
  }
  if (fault == Fault::kOutputCount) outputs[0]--;
  if (fault == Fault::kNegativeExtent || fault == Fault::kOverflow) {
    int* dims = op == Op::kFill ? output_shape : input_shape;
    dims[1] = fault == Fault::kNegativeExtent ? -1 : 46341;
    if (fault == Fault::kOverflow) dims[2] = 46341;
  }
  TFLMRegistration registration = tflite::Register_FILL();
  const void* params = nullptr;
  if (op == Op::kPack) {
    registration = tflite::Register_PACK();
    params = &pack;
  }
  if (op == Op::kUnpack) {
    registration = tflite::Register_UNPACK();
    params = &unpack;
  }
  if (op == Op::kSplit) {
    registration = tflite::Register_SPLIT();
    params = &split;
  }
  tflite::micro::KernelRunner runner(registration, tensors, tensor_count,
                                     IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), params);
  const TfLiteStatus prepared = runner.InitAndPrepare();
  if (fault != Fault::kNone || unavailable) {
    EXPECT_EQ(kTfLiteError, prepared);
    return;
  }
  EXPECT_EQ(kTfLiteOk, prepared);
  if (prepared != kTfLiteOk) return;
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  int stride = 1;
  for (int d = normalized + 1; d < shape[0]; ++d) stride *= shape[d + 1];
  if (op == Op::kPack && normalized < shape[0]) stride *= shape[normalized + 1];
  for (int n = 0; n < data_outputs; ++n) {
    EXPECT_EQ(Bits(sentinel), Bits(output[n][0]));
    EXPECT_EQ(Bits(sentinel), Bits(output[n][output_elements + 1]));
    for (int i = 0; i < output_elements; ++i) {
      int source_tensor = 0;
      int source_index = 0;
      if (op == Op::kPack) {
        source_tensor = (i / stride) % count;
        source_index = (i / (stride * count)) * stride + i % stride;
      } else if (op == Op::kSplit || op == Op::kUnpack) {
        const int block =
            op == Op::kSplit ? stride * shape[normalized + 1] / count : stride;
        source_index = (i / block) * block * count + n * block + i % block;
      }
      EXPECT_EQ(Bits(input[source_tensor][source_index]),
                Bits(output[n][i + 1]));
      if (golden != nullptr)
        EXPECT_EQ(golden[n * output_elements + i], Bits(output[n][i + 1]));
    }
  }
}

template <typename T>
void ValidCases(TfLiteType type) {
  int cube[] = {3, 2, 2, 3};
  for (int axis = 0; axis < 3; ++axis) {
    Run<T>(Op::kSplit, type, cube, axis - 3, cube[axis + 1]);
    Run<T>(Op::kUnpack, type, cube, axis - 3, cube[axis + 1]);
  }
  for (int axis = 0; axis <= 3; ++axis)
    Run<T>(Op::kPack, type, cube, axis - 4, 2);
  int split_blocks[] = {3, 2, 6, 3};
  Run<T>(Op::kSplit, type, split_blocks, -2, 3);
  int scalar[] = {0};
  Run<T>(Op::kPack, type, scalar, -1, 3);
  Run<T>(Op::kFill, type, scalar, 0, 1);
  int vector[] = {1, 3};
  Run<T>(Op::kUnpack, type, vector, -1, 3);
  for (uint32_t pattern = 0; pattern < 9; ++pattern) {
    Run<T>(Op::kFill, type, cube, 0, 1, Fault::kNone, pattern);
  }
  int high[] = {7, 1, 1, 1, 1, 1, 2, 17};
  Run<T>(Op::kSplit, type, high, -1, 17);
  Run<T>(Op::kUnpack, type, high, -1, 17);
  Run<T>(Op::kFill, type, high, 0, 1);
  int high_pack[] = {7, 1, 1, 1, 1, 1, 1, 2};
  Run<T>(Op::kPack, type, high_pack, -1, 17);
  int empty[] = {3, 2, 0, 3};
  Run<T>(Op::kSplit, type, empty, 0, 2);
  Run<T>(Op::kUnpack, type, empty, 0, 2);
  Run<T>(Op::kPack, type, empty, 0, 2);
  Run<T>(Op::kFill, type, empty, 0, 1);
}

template <typename T>
void InvalidCases(TfLiteType type) {
  int shape[] = {2, 2, 3};
  for (Op op : {Op::kSplit, Op::kPack, Op::kUnpack, Op::kFill}) {
    Run<T>(op, type, shape, 0, 2, Fault::kShape);
    Run<T>(op, type, shape, 0, 2, Fault::kType);
    Run<T>(op, type, shape, 0, 2, Fault::kNegativeExtent);
    Run<T>(op, type, shape, 0, 2, Fault::kOverflow);
    if (op != Op::kFill) {
      Run<T>(op, type, shape, 0, 2, Fault::kAxis);
      Run<T>(op, type, shape, 0, 2, Fault::kCount);
      Run<T>(op, type, shape, 0, 2, Fault::kZeroCount);
      Run<T>(op, type, shape, 0, 2, Fault::kNegativeAxis);
    }
    if (op == Op::kFill || op == Op::kSplit) {
      Run<T>(op, type, shape, 0, 2, Fault::kNonConstant);
      Run<T>(op, type, shape, 0, 2, Fault::kParameterType);
    }
  }
  Run<T>(Op::kFill, type, shape, 0, 1, Fault::kValueShape);
  Run<T>(Op::kPack, type, shape, 0, 2, Fault::kInputShape);
  Run<T>(Op::kSplit, type, shape, 0, 2, Fault::kOutputCount);
  Run<T>(Op::kUnpack, type, shape, 0, 2, Fault::kOutputCount);
}
}  // namespace

TEST(HeliaFloatDataMovementTest, Float32ExactBitsAndShapes) {
  ValidCases<float>(kTfLiteFloat32);
}
TEST(HeliaFloatDataMovementTest, Float32RejectsMalformedTensors) {
  InvalidCases<float>(kTfLiteFloat32);
}

TEST(HeliaFloatDataMovementTest, LiteRtFloat32Goldens) {
  int shape[] = {3, 2, 2, 3};
  Run<float>(Op::kSplit, kTfLiteFloat32, shape, -2, 2, Fault::kNone, 8, false,
             kSplitGolden);
  Run<float>(Op::kPack, kTfLiteFloat32, shape, -3, 2, Fault::kNone, 8, false,
             kPackGolden);
  Run<float>(Op::kUnpack, kTfLiteFloat32, shape, -2, 2, Fault::kNone, 8, false,
             kUnpackGolden);
  Run<float>(Op::kFill, kTfLiteFloat32, shape, 0, 1, Fault::kNone, 8, false,
             kFillGolden);
}

#if ARM_NN_ENABLE_F16
TEST(HeliaFloatDataMovementTest, LiteRtFloat16Goldens) {
  int shape[] = {3, 2, 2, 3};
  Run<uint16_t>(Op::kUnpack, kTfLiteFloat16, shape, -2, 2, Fault::kNone, 8,
                false, kUnpackHalfGolden);
  Run<uint16_t>(Op::kFill, kTfLiteFloat16, shape, 0, 1, Fault::kNone, 8, false,
                kFillHalfGolden);
}
TEST(HeliaFloatDataMovementTest, Float16ExactBitsAndShapes) {
  ValidCases<uint16_t>(kTfLiteFloat16);
}
TEST(HeliaFloatDataMovementTest, Float16RejectsMalformedTensors) {
  InvalidCases<uint16_t>(kTfLiteFloat16);
}
TEST(HeliaFloatDataMovementTest, EveryHalfEncodingAcrossCopyOperations) {
  int copy_shape[] = {3, 2, 2, 64};
  int pack_shape[] = {2, 2, 64};
  for (uint32_t base = 0; base < 65536; base += 256) {
    Run<uint16_t>(Op::kSplit, kTfLiteFloat16, copy_shape, 1, 2, Fault::kNone,
                  base);
    Run<uint16_t>(Op::kUnpack, kTfLiteFloat16, copy_shape, 1, 2, Fault::kNone,
                  base);
    Run<uint16_t>(Op::kPack, kTfLiteFloat16, pack_shape, 1, 2, Fault::kNone,
                  base);
  }
}
TEST(HeliaFloatDataMovementTest, EveryHalfEncodingThroughFill) {
  int scalar[] = {0};
  int parameter_shape[] = {1, 1};
  int output_shape[] = {1, 9};
  int32_t extent[] = {9};
  uint16_t value = 0;
  uint16_t output[11] = {};
  TfLiteTensor tensors[] = {
      CreateTensor(extent, IntArrayFromInts(parameter_shape)),
      CreateTensor(&value, IntArrayFromInts(scalar), false, kTfLiteFloat16),
      CreateTensor(output + 1, IntArrayFromInts(output_shape), false,
                   kTfLiteFloat16)};
  tensors[0].allocation_type = kTfLiteMmapRo;
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  // KernelRunner's fake context retains Eval tensors between invocations.
  for (uint32_t base = 0; base < 65536; base += 128) {
    const auto registration = tflite::Register_FILL();
    tflite::micro::KernelRunner runner(registration, tensors, 3,
                                       IntArrayFromInts(inputs),
                                       IntArrayFromInts(outputs), nullptr);
    const auto status = runner.InitAndPrepare();
    EXPECT_EQ(kTfLiteOk, status);
    if (status != kTfLiteOk) return;
    for (uint32_t bits = base; bits < base + 128; ++bits) {
      value = bits;
      memset(output, 0x5a, sizeof(output));
      EXPECT_EQ(kTfLiteOk, runner.Invoke());
      EXPECT_EQ(0x5a5a, output[0]);
      EXPECT_EQ(0x5a5a, output[10]);
      for (int i = 1; i <= 9; ++i) EXPECT_EQ(bits, output[i]);
    }
  }
}
#else
TEST(HeliaFloatDataMovementTest, UnavailableFloat16RejectedDuringPrepare) {
  int shape[] = {2, 2, 3};
  for (Op op : {Op::kSplit, Op::kPack, Op::kUnpack, Op::kFill}) {
    Run<uint16_t>(op, kTfLiteFloat16, shape, 0, 2, Fault::kNone, 0, true);
  }
}
#endif

TF_LITE_MICRO_TESTS_MAIN
