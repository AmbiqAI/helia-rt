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
#include "tensorflow/lite/micro/kernels/helia/tests/split_v_float_goldens.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {
using tflite::testing::CreateTensor;
using tflite::testing::IntArrayFromInts;

template <typename T>
void Count17(TfLiteType type, bool legacy_options = false) {
  T input[17], output[17] = {};
  int32_t sizes[17];
  int in_dims[] = {1, 17}, out_dims[] = {1, 1};
  int sizes_dims[] = {1, 17}, scalar[] = {0};
  int32_t axis = 0;
  int inputs[] = {3, 0, 1, 2}, outputs[18] = {17};
  TfLiteTensor tensors[20];
  for (int i = 0; i < 17; ++i) {
    input[i] = i + 1;
    sizes[i] = 1;
    outputs[i + 1] = i + 3;
    tensors[i + 3] =
        CreateTensor(output + i, IntArrayFromInts(out_dims), false, type);
  }
  tensors[0] = CreateTensor(input, IntArrayFromInts(in_dims), false, type);
  tensors[1] = CreateTensor(sizes, IntArrayFromInts(sizes_dims));
  tensors[2] = CreateTensor(&axis, IntArrayFromInts(scalar));
  tensors[1].allocation_type = tensors[2].allocation_type = kTfLiteMmapRo;
  TfLiteSplitVParams params = {legacy_options ? 0 : 17};
  const auto reg = tflite::Register_SPLIT_V();
  tflite::micro::KernelRunner runner(reg, tensors, 20, IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), &params);
  const auto status = runner.InitAndPrepare();
  EXPECT_EQ(kTfLiteOk, status);
  if (status != kTfLiteOk) return;
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  for (int i = 0; i < 17; ++i) EXPECT_EQ(input[i], output[i]);
}
enum class Fault {
  kNone,
  kAxisHigh,
  kAxisLow,
  kAxisType,
  kAxisShape,
  kSizesType,
  kSizesRank,
  kSizesCount,
  kOutputCount,
  kNoOutputs,
  kBuiltinCount,
  kOutputType,
  kOutputRank,
  kOutputShape,
  kNegativeExtent,
  kOverflow,
  kTwoInferred,
  kNegativeSize,
  kInferredNegative,
  kSizeSum,
  kSizeOverflow,
  kDynamicAxis,
  kDynamicSizes,
  kScalarInput,
  kNullAxis,
  kNullSizes
};

template <typename T>
T Pattern(uint32_t value) {
  if (sizeof(T) == 4) {
    const uint32_t special[] = {0,          0x80000000, 0x7f800000, 0xff800000,
                                0x7fc12345, 0x7f812345, 1,          0x807fffff};
    value = value < 8 ? special[value] : 0x3f000000 + value * 7919;
  }
  T result;
  memcpy(&result, &value, sizeof(T));
  return result;
}

template <typename T>
uint32_t Bits(T value) {
  uint32_t result = 0;
  memcpy(&result, &value, sizeof(T));
  return result;
}

template <typename T>
void Run(TfLiteType type, std::initializer_list<int> dimensions, int32_t axis,
         std::initializer_list<int32_t> pieces, Fault fault = Fault::kNone,
         uint32_t base = 0, const uint32_t* golden = nullptr,
         const int (*golden_shapes)[3] = nullptr,
         const int* golden_counts = nullptr) {
  constexpr int kCapacity = 512;
  static T input[kCapacity];
  static T output[2][20][kCapacity + 2];
  static TfLiteTensor tensors[23];
  int shape[9] = {}, output_shapes[20][9] = {};
  shape[0] = dimensions.size();
  int d = 1;
  int elements = 1;
  for (int extent : dimensions) {
    shape[d++] = extent;
    elements *= extent;
  }
  const int normalized = axis < 0 ? axis + shape[0] : axis;
  const int count = pieces.size();
  int32_t sizes[20] = {}, resolved[20] = {};
  int known = 0, inferred = -1, i = 0;
  for (int32_t length : pieces) {
    sizes[i] = resolved[i] = length;
    if (length == -1)
      inferred = i;
    else
      known += length;
    ++i;
  }
  if (inferred >= 0) resolved[inferred] = shape[normalized + 1] - known;
  for (i = 0; i < elements; ++i) {
    input[i] = golden != nullptr && type == kTfLiteInt32
                   ? static_cast<T>((i - 15) * 7919)
                   : Pattern<T>(base + i);
  }
  int sizes_shape[] = {1, count}, axis_shape[] = {0, 1};
  int inputs[] = {3, 0, 1, 2}, outputs[21] = {count};
  tensors[0] = CreateTensor(input, IntArrayFromInts(shape), false, type);
  tensors[1] = CreateTensor(sizes, IntArrayFromInts(sizes_shape));
  tensors[2] = CreateTensor(&axis, IntArrayFromInts(axis_shape));
  tensors[1].allocation_type = tensors[2].allocation_type = kTfLiteMmapRo;
  for (i = 0; i < count; ++i) {
    memcpy(output_shapes[i], shape, sizeof(shape));
    output_shapes[i][normalized + 1] = resolved[i];
    tensors[i + 3] = CreateTensor(
        output[0][i] + 1, IntArrayFromInts(output_shapes[i]), false, type);
    outputs[i + 1] = i + 3;
  }
  TfLiteSplitVParams params = {count};
  switch (fault) {
    case Fault::kAxisHigh:
      axis = shape[0];
      break;
    case Fault::kAxisLow:
      axis = -shape[0] - 1;
      break;
    case Fault::kAxisType:
      tensors[2].type = kTfLiteFloat32;
      break;
    case Fault::kAxisShape:
      axis_shape[0] = 1;
      axis_shape[1] = 2;
      break;
    case Fault::kSizesType:
      tensors[1].type = kTfLiteFloat32;
      break;
    case Fault::kSizesRank:
      sizes_shape[0] = 0;
      break;
    case Fault::kSizesCount:
      sizes_shape[1]--;
      break;
    case Fault::kOutputCount:
      outputs[0]--;
      break;
    case Fault::kNoOutputs:
      outputs[0] = 0;
      break;
    case Fault::kBuiltinCount:
      params.num_splits++;
      break;
    case Fault::kOutputType:
      tensors[3].type = kTfLiteUInt8;
      break;
    case Fault::kOutputRank:
      output_shapes[0][0]--;
      break;
    case Fault::kOutputShape:
      output_shapes[0][shape[0]]++;
      break;
    case Fault::kNegativeExtent:
      shape[1] = -1;
      break;
    case Fault::kOverflow:
      shape[1] = 46341;
      shape[2] = 46341;
      break;
    case Fault::kTwoInferred:
      sizes[0] = sizes[1] = -1;
      break;
    case Fault::kNegativeSize:
      sizes[0] = -2;
      break;
    case Fault::kInferredNegative:
      sizes[0] = shape[normalized + 1] + 1;
      break;
    case Fault::kSizeSum:
      sizes[0] = sizes[1] = sizes[2] = 0;
      break;
    case Fault::kSizeOverflow:
      sizes[0] = sizes[1] = INT32_MAX;
      break;
    case Fault::kDynamicAxis:
      tensors[2].allocation_type = kTfLiteArenaRw;
      break;
    case Fault::kDynamicSizes:
      tensors[1].allocation_type = kTfLiteArenaRw;
      break;
    case Fault::kScalarInput:
      shape[0] = 0;
      break;
    case Fault::kNullAxis:
      tensors[2].data.raw = nullptr;
      break;
    case Fault::kNullSizes:
      tensors[1].data.raw = nullptr;
      break;
    default:
      break;
  }
  const auto reg = tflite::Register_SPLIT_V();
  tflite::micro::KernelRunner runner(reg, tensors, count + 3,
                                     IntArrayFromInts(inputs),
                                     IntArrayFromInts(outputs), &params);
  bool unavailable = false;
#if !ARM_NN_ENABLE_F16
  unavailable = type == kTfLiteFloat16;
#endif
  const auto prepared = runner.InitAndPrepare();
  EXPECT_EQ(fault != Fault::kNone || unavailable ? kTfLiteError : kTfLiteOk,
            prepared);
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
  if (prepared != kTfLiteOk || fault != Fault::kNone || unavailable) return;
  for (int invocation = 0; invocation < 2; ++invocation) {
    memset(output[invocation], 0x5a, sizeof(output[invocation]));
    tensors[0].data.raw =
        elements == 0 ? nullptr : reinterpret_cast<char*>(input);
    for (i = 0; i < count; ++i) {
      tensors[i + 3].data.raw =
          elements == 0 || resolved[i] == 0
              ? nullptr
              : reinterpret_cast<char*>(output[invocation][i] + 1);
    }
    EXPECT_EQ(kTfLiteOk, runner.Invoke());
    int offset = 0, golden_index = 0;
    for (i = 0; i < count; ++i) {
      int out_count = 1;
      for (d = 1; d <= shape[0]; ++d) out_count *= output_shapes[i][d];
      if (golden_shapes != nullptr) {
        EXPECT_EQ(3, shape[0]);
        for (d = 0; d < 3; ++d) {
          EXPECT_EQ(golden_shapes[i][d], output_shapes[i][d + 1]);
        }
        EXPECT_EQ(golden_counts[i], out_count);
      }
      for (int j = 0; j < out_count; ++j) {
        int remainder = j, coordinate[8] = {};
        for (d = shape[0] - 1; d >= 0; --d) {
          coordinate[d] = remainder % output_shapes[i][d + 1];
          remainder /= output_shapes[i][d + 1];
        }
        coordinate[normalized] += offset;
        int source = 0;
        for (d = 0; d < shape[0]; ++d)
          source = source * shape[d + 1] + coordinate[d];
        EXPECT_EQ(Bits(input[source]), Bits(output[invocation][i][j + 1]));
        if (golden != nullptr)
          EXPECT_EQ(golden[golden_index++], Bits(output[invocation][i][j + 1]));
      }
      T guard;
      memset(&guard, 0x5a, sizeof(guard));
      EXPECT_EQ(Bits(guard), Bits(output[invocation][i][0]));
      EXPECT_EQ(Bits(guard), Bits(output[invocation][i][out_count + 1]));
      offset += resolved[i];
    }
  }
}

template <typename T>
void Valid(TfLiteType type) {
  Run<T>(type, {2, 5, 3}, -2, {1, -1, 2});
  Run<T>(type, {2, 5, 3}, 1, {0, 2, 0, 3, 0});
  Run<T>(type, {2, 5, 3}, 1, {2, 3, -1});
  Run<T>(type, {2, 5, 0}, 1, {1, -1, 2});
  Run<T>(type, {0, 5, 3}, 1, {1, -1, 2});
  Run<T>(type, {2, 0, 3}, 1, {0, -1, 0});
  Run<T>(type, {1, 1, 2, 1, 5, 1, 3}, -3, {1, -1, 2});
}

template <typename T>
void Invalid(TfLiteType type) {
  for (Fault fault :
       {Fault::kAxisHigh,     Fault::kAxisLow,        Fault::kAxisType,
        Fault::kAxisShape,    Fault::kSizesType,      Fault::kSizesRank,
        Fault::kSizesCount,   Fault::kOutputCount,    Fault::kNoOutputs,
        Fault::kBuiltinCount, Fault::kOutputType,     Fault::kOutputRank,
        Fault::kOutputShape,  Fault::kNegativeExtent, Fault::kOverflow,
        Fault::kTwoInferred,  Fault::kNegativeSize,   Fault::kInferredNegative,
        Fault::kSizeSum,      Fault::kSizeOverflow,   Fault::kDynamicAxis,
        Fault::kDynamicSizes, Fault::kScalarInput,    Fault::kNullAxis,
        Fault::kNullSizes}) {
    Run<T>(type, {2, 5, 3}, -2, {1, -1, 2}, fault);
  }
}
}  // namespace

TEST(HeliaSplitVTest, Int8SeventeenOutputs) { Count17<int8_t>(kTfLiteInt8); }
TEST(HeliaSplitVTest, LegacyModelWithoutSplitVOptions) {
  Count17<int8_t>(kTfLiteInt8, true);
}
TEST(HeliaSplitVTest, Int16SeventeenOutputs) { Count17<int16_t>(kTfLiteInt16); }
TEST(HeliaSplitVTest, Float32SeventeenOutputs) {
  Count17<float>(kTfLiteFloat32);
}
TEST(HeliaSplitVTest, Int32SeventeenOutputs) { Count17<int32_t>(kTfLiteInt32); }
TEST(HeliaSplitVTest, Float32BitsShapesAndPointerRefresh) {
  Valid<float>(kTfLiteFloat32);
}
TEST(HeliaSplitVTest, IntegerShapesAndPointerRefresh) {
  Valid<int8_t>(kTfLiteInt8);
  Valid<int16_t>(kTfLiteInt16);
  Valid<int32_t>(kTfLiteInt32);
}
TEST(HeliaSplitVTest, MalformedMetadata) {
  Invalid<float>(kTfLiteFloat32);
  Invalid<int8_t>(kTfLiteInt8);
  Invalid<int16_t>(kTfLiteInt16);
  Invalid<int32_t>(kTfLiteInt32);
}
TEST(HeliaSplitVTest, LiteRtReferenceGoldens) {
  Run<float>(kTfLiteFloat32, {2, 5, 3}, -2, {1, -1, 2}, Fault::kNone, 8,
             kSplitVInferredFloatGolden, kSplitVInferredFloatShapes,
             kSplitVInferredFloatCounts);
  Run<float>(kTfLiteFloat32, {2, 5, 3}, -2, {0, 2, 0, 3, 0}, Fault::kNone, 8,
             kSplitVMixedZeroFloatGolden, kSplitVMixedZeroFloatShapes,
             kSplitVMixedZeroFloatCounts);
  Run<float>(kTfLiteFloat32, {2, 5, 0}, -2, {1, -1, 2}, Fault::kNone, 8,
             kSplitVTotalEmptyFloatGolden, kSplitVTotalEmptyFloatShapes,
             kSplitVTotalEmptyFloatCounts);
  Run<int32_t>(kTfLiteInt32, {2, 5, 3}, -2, {1, -1, 2}, Fault::kNone, 8,
               kSplitVInferredInt32Golden, kSplitVInferredInt32Shapes,
               kSplitVInferredInt32Counts);
  Run<int32_t>(kTfLiteInt32, {2, 5, 3}, -2, {0, 2, 0, 3, 0}, Fault::kNone, 8,
               kSplitVMixedZeroInt32Golden, kSplitVMixedZeroInt32Shapes,
               kSplitVMixedZeroInt32Counts);
  Run<int32_t>(kTfLiteInt32, {2, 5, 0}, -2, {1, -1, 2}, Fault::kNone, 8,
               kSplitVTotalEmptyInt32Golden, kSplitVTotalEmptyInt32Shapes,
               kSplitVTotalEmptyInt32Counts);
}
#if ARM_NN_ENABLE_F16
TEST(HeliaSplitVTest, Float16SeventeenOutputs) {
  Count17<uint16_t>(kTfLiteFloat16);
}
TEST(HeliaSplitVTest, Float16BitsShapesAndPointerRefresh) {
  Valid<uint16_t>(kTfLiteFloat16);
}
TEST(HeliaSplitVTest, Float16MalformedMetadata) {
  Invalid<uint16_t>(kTfLiteFloat16);
}
TEST(HeliaSplitVTest, EveryHalfEncoding) {
  for (uint32_t base = 0; base < 65536; base += 256)
    Run<uint16_t>(kTfLiteFloat16, {2, 4, 32}, 1, {1, -1, 1}, Fault::kNone,
                  base);
}
#else
TEST(HeliaSplitVTest, UnavailableHalfRejectedAtPrepare) {
  Run<uint16_t>(kTfLiteFloat16, {2, 5, 3}, 1, {1, -1, 2});
  Run<uint16_t>(kTfLiteFloat16, {2, 5, 0}, 1, {1, -1, 2});
}
#endif
TF_LITE_MICRO_TESTS_MAIN
