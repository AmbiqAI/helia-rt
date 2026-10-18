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

// SPLIT, SQUEEZE, DIV and ADD return kTfLiteError on a malformed model or a
// zero int32 divisor instead of indexing out of bounds, dividing by zero or
// reaching TFLITE_ABORT. see AmbiqAI/helia-rt#451

#include <cstdint>

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kSplitElements = 16;
constexpr int kSplitPiece = kSplitElements / 2;

enum class SplitFault {
  kNone,
  kOutputShape,
  kOutputRank,
  kOutputType,
  kNumSplits,
};

// Splits a {1, 2, 2, 4} tensor into two outputs along `axis`. Each output
// buffer holds one guard element on either side of its kSplitPiece elements.
// For an out-of-range axis both outputs take the input shape, which every
// check but the axis range check accepts.
template <typename T>
TfLiteStatus RunSplit(TfLiteType type, int32_t axis, SplitFault fault,
                      T (*outputs)[kSplitPiece + 2], bool invoke) {
  T input[kSplitElements];
  for (int i = 0; i < kSplitElements; ++i) input[i] = static_cast<T>(i + 1);
  int32_t axis_value = axis;
  int axis_shape[] = {0};
  // The trailing 2 is data[4], so an axis of 4 that escapes the range check
  // reads an even extent inside this array.
  int input_shape[] = {4, 1, 2, 2, 4, 2};
  constexpr int kRank = 4;
  int normalized = axis < 0 ? axis + kRank : axis;
  int output_shape[] = {4, 1, 2, 2, 4};
  if (normalized >= 0 && normalized < kRank) {
    output_shape[normalized + 1] /= 2;
  } else {
    normalized = kRank - 1;
  }
  int bad_shape[] = {4, 1, 2, 2, 4};
  bad_shape[normalized + 1] = bad_shape[normalized + 1] / 2 + 1;
  int bad_rank[] = {3, 2, 2, 2};
  int* out_dims = output_shape;
  if (fault == SplitFault::kOutputShape) out_dims = bad_shape;
  if (fault == SplitFault::kOutputRank) out_dims = bad_rank;
  TfLiteTensor tensors[] = {
      CreateTensor(&axis_value, IntArrayFromInts(axis_shape)),
      CreateTensor(input, IntArrayFromInts(input_shape), false, type),
      CreateTensor(outputs[0] + 1, IntArrayFromInts(output_shape), false, type),
      CreateTensor(outputs[1] + 1, IntArrayFromInts(out_dims), false, type),
  };
  tensors[0].allocation_type = kTfLiteMmapRo;
  if (fault == SplitFault::kOutputType) {
    tensors[3].type = type == kTfLiteInt32 ? kTfLiteInt8 : kTfLiteInt32;
  }
  TfLiteSplitParams params = {};
  params.num_splits = fault == SplitFault::kNumSplits ? 3 : 2;
  int inputs[] = {2, 0, 1};
  int output_indices[] = {2, 2, 3};
  const TFLMRegistration registration = tflite::Register_SPLIT();
  micro::KernelRunner runner(registration, tensors, 4, IntArrayFromInts(inputs),
                             IntArrayFromInts(output_indices), &params);
  TfLiteStatus status = runner.InitAndPrepare();
  if (status == kTfLiteOk && invoke) status = runner.Invoke();
  return status;
}

template <typename T>
void ExpectSplitRejects(TfLiteType type, int32_t axis, SplitFault fault) {
  T outputs[2][kSplitPiece + 2] = {};
  EXPECT_EQ(kTfLiteError, RunSplit<T>(type, axis, fault, outputs, false));
}

template <typename T>
void ExpectSplitLastAxis(TfLiteType type) {
  constexpr T kGuard = static_cast<T>(-7);
  T outputs[2][kSplitPiece + 2];
  for (auto& output : outputs) {
    for (T& value : output) value = kGuard;
  }
  EXPECT_EQ(kTfLiteOk, RunSplit<T>(type, -1, SplitFault::kNone, outputs, true));
  for (int n = 0; n < 2; ++n) {
    EXPECT_EQ(kGuard, outputs[n][0]);
    EXPECT_EQ(kGuard, outputs[n][kSplitPiece + 1]);
    for (int i = 0; i < kSplitPiece; ++i) {
      // Row i / 2 of the {4, 4} view, half n, column i % 2.
      const int source = (i / 2) * 4 + n * 2 + (i % 2);
      EXPECT_EQ(static_cast<T>(source + 1), outputs[n][i + 1]);
    }
  }
}

// Squeezes a {1, 2, 1, 3} int8 tensor into an output whose dims array is
// `output_dims`.
TfLiteStatus PrepareSqueeze(int* output_dims) {
  int8_t input[6] = {};
  int8_t output[6] = {};
  int input_shape[] = {4, 1, 2, 1, 3};
  TfLiteTensor tensors[] = {
      CreateTensor(input, IntArrayFromInts(input_shape), false, kTfLiteInt8),
      CreateTensor(output, IntArrayFromInts(output_dims), false, kTfLiteInt8),
  };
  TfLiteSqueezeParams params = {};
  int inputs[] = {1, 0};
  int outputs[] = {1, 1};
  const TFLMRegistration registration = tflite::Register_SQUEEZE();
  micro::KernelRunner runner(registration, tensors, 2, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  return runner.InitAndPrepare();
}

// Divides {8, 9, 10, 12} by `divisor` of shape `divisor_dims` and checks the
// output was not written.
TfLiteStatus InvokeDivInt32(const int32_t* divisor, int* divisor_dims) {
  constexpr int32_t kSentinel = 0x5a5a5a5a;
  const int32_t dividend[4] = {8, 9, 10, 12};
  int32_t output[4] = {kSentinel, kSentinel, kSentinel, kSentinel};
  int shape[] = {4, 1, 1, 1, 4};
  TfLiteTensor tensors[] = {
      CreateTensor(dividend, IntArrayFromInts(shape)),
      CreateTensor(divisor, IntArrayFromInts(divisor_dims)),
      CreateTensor(output, IntArrayFromInts(shape)),
  };
  TfLiteDivParams params = {};
  params.activation = kTfLiteActNone;
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = tflite::Register_DIV();
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  TfLiteStatus status = runner.InitAndPrepare();
  EXPECT_EQ(kTfLiteOk, status);
  if (status == kTfLiteOk) status = runner.Invoke();
  for (int32_t value : output) EXPECT_EQ(kSentinel, value);
  return status;
}

// Prepares a quantized ADD in which tensor `unquantized` (0: input1,
// 1: input2, 2: output) carries kTfLiteNoQuantization.
template <typename T>
TfLiteStatus PrepareAddUnquantized(int unquantized) {
  T input1[4] = {};
  T input2[4] = {};
  T output[4] = {};
  int shape[] = {1, 4};
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input1, IntArrayFromInts(shape), 0.5f, 0),
      CreateQuantizedTensor(input2, IntArrayFromInts(shape), 0.25f, 0),
      CreateQuantizedTensor(output, IntArrayFromInts(shape), 1.0f, 0),
  };
  tensors[unquantized].quantization = {kTfLiteNoQuantization, nullptr};
  tensors[unquantized].params = {0.0f, 0};
  TfLiteAddParams params = {};
  params.activation = kTfLiteActNone;
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = tflite::Register_ADD();
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  return runner.InitAndPrepare();
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(HeliaMalformedModelTest, SplitRejectsOutOfRangeAxis) {
  using tflite::testing::ExpectSplitRejects;
  using tflite::testing::SplitFault;
  for (int32_t axis : {4, -5}) {
    ExpectSplitRejects<int8_t>(kTfLiteInt8, axis, SplitFault::kNone);
    ExpectSplitRejects<int16_t>(kTfLiteInt16, axis, SplitFault::kNone);
    ExpectSplitRejects<int32_t>(kTfLiteInt32, axis, SplitFault::kNone);
    ExpectSplitRejects<float>(kTfLiteFloat32, axis, SplitFault::kNone);
  }
}

TEST(HeliaMalformedModelTest, SplitRejectsMismatchedOutputs) {
  using tflite::testing::ExpectSplitRejects;
  using tflite::testing::SplitFault;
  for (SplitFault fault : {SplitFault::kOutputShape, SplitFault::kOutputRank,
                           SplitFault::kOutputType, SplitFault::kNumSplits}) {
    ExpectSplitRejects<int8_t>(kTfLiteInt8, 3, fault);
    ExpectSplitRejects<int16_t>(kTfLiteInt16, 3, fault);
    ExpectSplitRejects<int32_t>(kTfLiteInt32, 3, fault);
  }
}

TEST(HeliaMalformedModelTest, SplitIntegerNegativeAxisSplits) {
  using tflite::testing::ExpectSplitLastAxis;
  ExpectSplitLastAxis<int8_t>(kTfLiteInt8);
  ExpectSplitLastAxis<int16_t>(kTfLiteInt16);
  ExpectSplitLastAxis<int32_t>(kTfLiteInt32);
}

TEST(HeliaMalformedModelTest, SqueezeRejectsShortOrShrunkOutput) {
  using tflite::testing::PrepareSqueeze;
  // Rank 1 where two dims are kept. The trailing 3 is where an unbounded
  // read of data[1] lands, so it stays inside this array.
  int short_rank[] = {1, 6, 3};
  EXPECT_EQ(kTfLiteError, PrepareSqueeze(short_rank));
  int shrunk[] = {2, 1, 6};
  EXPECT_EQ(kTfLiteError, PrepareSqueeze(shrunk));
  int valid[] = {2, 2, 3};
  EXPECT_EQ(kTfLiteOk, PrepareSqueeze(valid));
}

TEST(HeliaMalformedModelTest, DivInt32RejectsZeroDivisor) {
  using tflite::testing::InvokeDivInt32;
  int same_shape[] = {4, 1, 1, 1, 4};
  const int32_t last_zero[4] = {2, 3, 5, 0};
  EXPECT_EQ(kTfLiteError, InvokeDivInt32(last_zero, same_shape));
  int scalar_shape[] = {4, 1, 1, 1, 1};
  const int32_t zero[1] = {0};
  EXPECT_EQ(kTfLiteError, InvokeDivInt32(zero, scalar_shape));
}

TEST(HeliaMalformedModelTest, AddRejectsUnquantizedInt8Int16) {
  using tflite::testing::PrepareAddUnquantized;
  for (int unquantized = 0; unquantized < 3; ++unquantized) {
    EXPECT_EQ(kTfLiteError, PrepareAddUnquantized<int8_t>(unquantized));
    EXPECT_EQ(kTfLiteError, PrepareAddUnquantized<int16_t>(unquantized));
  }
}

TF_LITE_MICRO_TESTS_MAIN
