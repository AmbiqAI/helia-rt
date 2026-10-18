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

// SPLIT, SQUEEZE, DIV and ADD return kTfLiteError on a malformed model, a zero
// divisor or int32 INT32_MIN / -1, instead of indexing out of bounds, hitting
// undefined integer division or reaching TFLITE_ABORT, and release their
// Prepare temporaries when they do.
// see AmbiqAI/helia-rt#451

#include <cmath>
#include <cstdint>
#include <limits>

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
  kNonAxisSmaller,
  kNonAxisLarger,
  kShortRank,
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
  const int non_axis = normalized == 1 ? 2 : 1;
  int non_axis_shape[5];
  for (int i = 0; i < 5; ++i) non_axis_shape[i] = output_shape[i];
  non_axis_shape[non_axis + 1] += fault == SplitFault::kNonAxisLarger ? 1 : -1;
  // Rank 3 with the leading dims of the expected output. The trailing entry
  // is the expected data[3], so a missing rank check reads a matching dim.
  int short_rank[5];
  for (int i = 0; i < 5; ++i) short_rank[i] = output_shape[i];
  short_rank[0] = 3;
  int* out_dims = output_shape;
  if (fault == SplitFault::kOutputShape) out_dims = bad_shape;
  if (fault == SplitFault::kOutputRank) out_dims = bad_rank;
  if (fault == SplitFault::kNonAxisSmaller ||
      fault == SplitFault::kNonAxisLarger) {
    out_dims = non_axis_shape;
  }
  if (fault == SplitFault::kShortRank) out_dims = short_rank;
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
  const TfLiteStatus status = runner.InitAndPrepare();
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
  return status;
}

int Elements(const int* dims) {
  int count = 1;
  for (int i = 1; i <= dims[0]; ++i) count *= dims[i];
  return count;
}

constexpr int kMaxDivElements = 8;

// Divides `dividend` by `divisor` into an output of `output_dims`, at most
// kMaxDivElements elements. On kTfLiteOk the output must equal `expected`
// when one is given; on an error it must be unwritten.
TfLiteStatus InvokeDivInt32(const int32_t* dividend, int* dividend_dims,
                            const int32_t* divisor, int* divisor_dims,
                            int* output_dims, const int32_t* expected,
                            TfLiteStatus expected_prepare = kTfLiteOk) {
  constexpr int32_t kSentinel = 0x5a5a5a5a;
  int32_t output[kMaxDivElements];
  for (int32_t& value : output) value = kSentinel;
  TfLiteTensor tensors[] = {
      CreateTensor(dividend, IntArrayFromInts(dividend_dims)),
      CreateTensor(divisor, IntArrayFromInts(divisor_dims)),
      CreateTensor(output, IntArrayFromInts(output_dims)),
  };
  TfLiteDivParams params = {};
  params.activation = kTfLiteActNone;
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = tflite::Register_DIV();
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  TfLiteStatus status = runner.InitAndPrepare();
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
  EXPECT_EQ(expected_prepare, status);
  if (status == kTfLiteOk) status = runner.Invoke();
  for (int i = 0; i < kMaxDivElements; ++i) {
    if (status != kTfLiteOk || i >= Elements(output_dims)) {
      EXPECT_EQ(kSentinel, output[i]);
    } else if (expected != nullptr) {
      EXPECT_EQ(expected[i], output[i]);
    }
  }
  return status;
}

// Divides four `dividend_value`s (`dividend_scale`, zero point 0) of `dims` by
// an int8 `divisor` of `divisor_dims` (`dims` when null) into an output of
// `dims`. On an error the output must be unwritten.
TfLiteStatus InvokeDivInt8(float divisor_scale, int divisor_zero_point,
                           float output_scale, int output_zero_point,
                           const int8_t* divisor, int* dims,
                           int* divisor_dims = nullptr,
                           float dividend_scale = 0.5f,
                           int8_t dividend_value = 10) {
  constexpr int8_t kSentinel = 0x5a;
  const int8_t dividend[4] = {dividend_value, dividend_value, dividend_value,
                              dividend_value};
  int8_t output[4] = {kSentinel, kSentinel, kSentinel, kSentinel};
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(dividend, IntArrayFromInts(dims), dividend_scale,
                            0),
      CreateQuantizedTensor(
          divisor,
          IntArrayFromInts(divisor_dims != nullptr ? divisor_dims : dims),
          divisor_scale, divisor_zero_point),
      CreateQuantizedTensor(output, IntArrayFromInts(dims), output_scale,
                            output_zero_point),
  };
  TfLiteDivParams params = {};
  params.activation = kTfLiteActNone;
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = tflite::Register_DIV();
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  TfLiteStatus status = runner.InitAndPrepare();
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
  if (status == kTfLiteOk) status = runner.Invoke();
  if (status != kTfLiteOk) {
    for (int8_t value : output) EXPECT_EQ(kSentinel, value);
  }
  return status;
}

// Prepares a two-input op of `type` whose three tensors share `dims`.
TfLiteStatus PrepareBinary(const TFLMRegistration& registration, void* params,
                           TfLiteType type, int* dims) {
  int32_t input1[4] = {1, 1, 1, 1};
  int32_t input2[4] = {1, 1, 1, 1};
  int32_t output[4] = {};
  TfLiteTensor tensors[] = {
      CreateTensor(input1, IntArrayFromInts(dims), false, type),
      CreateTensor(input2, IntArrayFromInts(dims), false, type),
      CreateTensor(output, IntArrayFromInts(dims), false, type),
  };
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), params);
  const TfLiteStatus status = runner.InitAndPrepare();
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
  return status;
}

TfLiteStatus PrepareDiv(TfLiteType type, int* dims) {
  TfLiteDivParams params = {};
  params.activation = kTfLiteActNone;
  const TFLMRegistration registration = tflite::Register_DIV();
  return PrepareBinary(registration, &params, type, dims);
}

TfLiteStatus PrepareAddRank(TfLiteType type, int* dims) {
  TfLiteAddParams params = {};
  params.activation = kTfLiteActNone;
  const TFLMRegistration registration = tflite::Register_ADD();
  return PrepareBinary(registration, &params, type, dims);
}

// Prepares an ADD of type T with the given scales and zero points (input1,
// input2, output). Tensor `unquantized`, if not -1, carries
// kTfLiteNoQuantization. Every return must release the Prepare temporaries.
template <typename T>
TfLiteStatus PrepareAdd(const float (&scales)[3], const int (&zero_points)[3],
                        int unquantized) {
  T input1[4] = {};
  T input2[4] = {};
  T output[4] = {};
  int shape[] = {1, 4};
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input1, IntArrayFromInts(shape), scales[0],
                            zero_points[0]),
      CreateQuantizedTensor(input2, IntArrayFromInts(shape), scales[1],
                            zero_points[1]),
      CreateQuantizedTensor(output, IntArrayFromInts(shape), scales[2],
                            zero_points[2]),
  };
  if (unquantized >= 0) {
    tensors[unquantized].quantization = {kTfLiteNoQuantization, nullptr};
    tensors[unquantized].params = {0.0f, 0};
  }
  TfLiteAddParams params = {};
  params.activation = kTfLiteActNone;
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = tflite::Register_ADD();
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  const TfLiteStatus status = runner.InitAndPrepare();
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
  return status;
}

template <typename T>
TfLiteStatus PrepareAddUnquantized(int unquantized) {
  return PrepareAdd<T>({0.5f, 0.25f, 1.f}, {0, 0, 0}, unquantized);
}

template <typename T>
TfLiteStatus PrepareAddScales(float scale1, float scale2, float output_scale) {
  return PrepareAdd<T>({scale1, scale2, output_scale}, {0, 0, 0}, -1);
}

template <typename T>
void ExpectAddRejectsScales() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  EXPECT_EQ(kTfLiteError, PrepareAddScales<T>(0.f, 0.25f, 1.f));
  EXPECT_EQ(kTfLiteError, PrepareAddScales<T>(0.5f, -0.25f, 1.f));
  EXPECT_EQ(kTfLiteError, PrepareAddScales<T>(0.5f, 0.25f, nan));
  EXPECT_EQ(kTfLiteError, PrepareAddScales<T>(inf, 0.25f, 1.f));
  EXPECT_EQ(kTfLiteError, PrepareAddScales<T>(0.5f, 0.5f, 1e-7f));
  // All negative: every multiplier lands in (0, 1).
  EXPECT_EQ(kTfLiteError, PrepareAddScales<T>(-0.5f, -0.4f, -1.f));
}

// With both input scales 0.5 the output multiplier is
// 2^-left_shift / output_scale, where left_shift is 20 for int8 and 15 for
// int16.
template <typename T>
void ExpectAddOutputMultiplierBoundary(int left_shift) {
  const float at_one = std::ldexp(1.f, -left_shift);
  EXPECT_EQ(kTfLiteError, PrepareAddScales<T>(0.5f, 0.5f, at_one));
  EXPECT_EQ(kTfLiteOk, PrepareAddScales<T>(0.5f, 0.5f, at_one * 1.001f));
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
  for (SplitFault fault :
       {SplitFault::kOutputShape, SplitFault::kOutputRank,
        SplitFault::kOutputType, SplitFault::kNumSplits,
        SplitFault::kNonAxisSmaller, SplitFault::kNonAxisLarger,
        SplitFault::kShortRank}) {
    ExpectSplitRejects<int8_t>(kTfLiteInt8, 3, fault);
    ExpectSplitRejects<int16_t>(kTfLiteInt16, 3, fault);
    ExpectSplitRejects<int32_t>(kTfLiteInt32, 3, fault);
    ExpectSplitRejects<float>(kTfLiteFloat32, 3, fault);
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
  const int32_t dividend[4] = {8, 9, 10, 12};
  const int32_t last_zero[4] = {2, 3, 5, 0};
  EXPECT_EQ(kTfLiteError, InvokeDivInt32(dividend, same_shape, last_zero,
                                         same_shape, same_shape, nullptr));
  int scalar_shape[] = {4, 1, 1, 1, 1};
  const int32_t zero[1] = {0};
  EXPECT_EQ(kTfLiteError, InvokeDivInt32(dividend, same_shape, zero,
                                         scalar_shape, same_shape, nullptr));
}

TEST(HeliaMalformedModelTest, DivInt32RejectsMinByMinusOne) {
  using tflite::testing::InvokeDivInt32;
  constexpr int32_t kMin = std::numeric_limits<int32_t>::min();
  int same_shape[] = {4, 1, 1, 1, 4};
  const int32_t last_min[4] = {1, 2, 3, kMin};
  const int32_t last_minus_one[4] = {1, 1, 1, -1};
  EXPECT_EQ(kTfLiteError, InvokeDivInt32(last_min, same_shape, last_minus_one,
                                         same_shape, same_shape, nullptr));
  // Unpaired INT32_MIN and -1 still divide.
  const int32_t first_min[4] = {kMin, 4, 6, 8};
  const int32_t unpaired[4] = {2, -1, 1, 1};
  const int32_t unpaired_quotient[4] = {kMin / 2, -4, 6, 8};
  EXPECT_EQ(kTfLiteOk,
            InvokeDivInt32(first_min, same_shape, unpaired, same_shape,
                           same_shape, unpaired_quotient));
  int scalar_shape[] = {4, 1, 1, 1, 1};
  const int32_t minus_one[1] = {-1};
  EXPECT_EQ(kTfLiteError, InvokeDivInt32(first_min, same_shape, minus_one,
                                         scalar_shape, same_shape, nullptr));
  // A {2, 2} dividend over a {2, 1} divisor: row 0 divides by divisor[0].
  int square_shape[] = {4, 1, 1, 2, 2};
  int column_shape[] = {4, 1, 1, 2, 1};
  const int32_t row1_minus_one[2] = {2, -1};
  const int32_t row_quotient[4] = {kMin / 2, 2, -6, -8};
  EXPECT_EQ(kTfLiteOk,
            InvokeDivInt32(first_min, square_shape, row1_minus_one,
                           column_shape, square_shape, row_quotient));
  // INT32_MIN at flat index 1 pairs with divisor[0], not divisor[1].
  const int32_t second_min[4] = {4, kMin, 6, 8};
  const int32_t second_quotient[4] = {2, kMin / 2, -6, -8};
  EXPECT_EQ(kTfLiteOk,
            InvokeDivInt32(second_min, square_shape, row1_minus_one,
                           column_shape, square_shape, second_quotient));
  const int32_t row0_minus_one[2] = {-1, 2};
  EXPECT_EQ(kTfLiteError,
            InvokeDivInt32(first_min, square_shape, row0_minus_one,
                           column_shape, square_shape, nullptr));
}

TEST(HeliaMalformedModelTest, DivInt32MinByMinusOneBroadcast) {
  using tflite::testing::InvokeDivInt32;
  constexpr int32_t kMin = std::numeric_limits<int32_t>::min();
  // A one-element dividend over a {4} divisor. The zeros after it are what an
  // unbroadcast read of the dividend would see.
  const int32_t scalar_min[4] = {kMin, 0, 0, 0};
  int one[] = {1, 1};
  int four[] = {1, 4};
  const int32_t third_minus_one[4] = {1, 2, -1, 4};
  EXPECT_EQ(kTfLiteError, InvokeDivInt32(scalar_min, one, third_minus_one, four,
                                         four, nullptr));
  const int32_t positive[4] = {1, 2, 3, 4};
  const int32_t scalar_quotient[4] = {kMin, kMin / 2, kMin / 3, kMin / 4};
  EXPECT_EQ(kTfLiteOk, InvokeDivInt32(scalar_min, one, positive, four, four,
                                      scalar_quotient));
  // A {2, 4} dividend over a {2, 1} divisor. The -1s after the divisor are
  // what a read past it would see.
  const int32_t rows[8] = {1, 2, 3, 4, kMin, 6, 7, 8};
  const int32_t column[6] = {-1, 3, -1, -1, -1, -1};
  int two_by_four[] = {2, 2, 4};
  int two_by_one[] = {2, 2, 1};
  const int32_t rows_quotient[8] = {-1, -2, -3, -4, kMin / 3, 2, 2, 2};
  EXPECT_EQ(kTfLiteOk, InvokeDivInt32(rows, two_by_four, column, two_by_one,
                                      two_by_four, rows_quotient));
  // A {2, 2} dividend over a {1, 1, 2, 2} divisor: the ranks differ, the
  // elements pair by index.
  int two_by_two[] = {2, 2, 2};
  int rank4[] = {4, 1, 1, 2, 2};
  const int32_t first_min[4] = {kMin, 1, 2, 3};
  const int32_t second_minus_one[4] = {2, -1, 1, 1};
  const int32_t rank_quotient[4] = {kMin / 2, -1, 2, 3};
  EXPECT_EQ(kTfLiteOk, InvokeDivInt32(first_min, two_by_two, second_minus_one,
                                      rank4, two_by_two, rank_quotient));
  const int32_t second_min[4] = {1, kMin, 2, 3};
  EXPECT_EQ(kTfLiteError,
            InvokeDivInt32(second_min, two_by_two, second_minus_one, rank4,
                           two_by_two, nullptr));
}

TEST(HeliaMalformedModelTest, DivInt32RejectsUnbroadcastableShapes) {
  using tflite::testing::InvokeDivInt32;
  const int32_t ones[6] = {1, 1, 1, 1, 1, 1};
  int three[] = {2, 1, 3};
  int four[] = {2, 1, 4};
  EXPECT_EQ(kTfLiteError, InvokeDivInt32(ones, three, ones, four, four, nullptr,
                                         kTfLiteError));
  // Element counts match, but neither input broadcasts to a flat output.
  int two_by_three[] = {2, 2, 3};
  int rank3[] = {3, 1, 2, 3};
  int six[] = {1, 6};
  EXPECT_EQ(kTfLiteError, InvokeDivInt32(ones, two_by_three, ones, rank3, six,
                                         nullptr, kTfLiteError));
}

TEST(HeliaMalformedModelTest, DivInt32RejectsOutputLargerThanBroadcast) {
  using tflite::testing::InvokeDivInt32;
  const int32_t dividend[4] = {4, 4, 4, 4};
  const int32_t divisor[4] = {2, 2, 2, 2};
  int row[] = {2, 1, 4};
  int two_rows[] = {2, 2, 4};
  EXPECT_EQ(kTfLiteError, InvokeDivInt32(dividend, row, divisor, row, two_rows,
                                         nullptr, kTfLiteError));
  int one[] = {2, 1, 1};
  EXPECT_EQ(kTfLiteError, InvokeDivInt32(dividend, one, divisor, one, row,
                                         nullptr, kTfLiteError));
}

TEST(HeliaMalformedModelTest, DivRejectsRankAboveSix) {
  using tflite::testing::PrepareDiv;
  int rank7[] = {7, 1, 1, 1, 1, 1, 1, 4};
  EXPECT_EQ(kTfLiteError, PrepareDiv(kTfLiteInt32, rank7));
  EXPECT_EQ(kTfLiteError, PrepareDiv(kTfLiteFloat32, rank7));
  int rank6[] = {6, 1, 1, 1, 1, 1, 4};
  EXPECT_EQ(kTfLiteOk, PrepareDiv(kTfLiteInt32, rank6));
}

TEST(HeliaMalformedModelTest, DivEmptyOutputIgnoresZeroDivisor) {
  using tflite::testing::InvokeDivInt32;
  using tflite::testing::InvokeDivInt8;
  const int32_t dividend[1] = {0};
  const int32_t divisor[4] = {1, 0, 1, 1};
  int empty[] = {2, 0, 4};
  int row[] = {2, 1, 4};
  EXPECT_EQ(kTfLiteOk,
            InvokeDivInt32(dividend, empty, divisor, row, empty, nullptr));
  const int8_t int8_divisor[4] = {3, 3, 3, 3};
  EXPECT_EQ(kTfLiteOk,
            InvokeDivInt8(0.5f, 3, 1.f, 0, int8_divisor, empty, row));
}

TEST(HeliaMalformedModelTest, DivInt8RejectsBadQuantization) {
  using tflite::testing::InvokeDivInt8;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const int8_t divisor[4] = {3, 3, 3, 3};
  int dims[] = {1, 4};
  EXPECT_EQ(kTfLiteError, InvokeDivInt8(0.f, 0, 1.f, 0, divisor, dims));
  EXPECT_EQ(kTfLiteError, InvokeDivInt8(-0.5f, 0, 1.f, 0, divisor, dims));
  EXPECT_EQ(kTfLiteError, InvokeDivInt8(nan, 0, 1.f, 0, divisor, dims));
  EXPECT_EQ(kTfLiteError, InvokeDivInt8(inf, 0, 1.f, 0, divisor, dims));
  EXPECT_EQ(kTfLiteError, InvokeDivInt8(0.5f, 0, 0.f, 0, divisor, dims));
  EXPECT_EQ(kTfLiteError, InvokeDivInt8(0.5f, 128, 1.f, 0, divisor, dims));
  EXPECT_EQ(kTfLiteError, InvokeDivInt8(0.5f, 0, 1.f, -129, divisor, dims));
  // Both scales finite, but their product underflows float.
  EXPECT_EQ(kTfLiteError, InvokeDivInt8(1e-30f, 0, 1e-30f, 0, divisor, dims));
  EXPECT_EQ(kTfLiteOk, InvokeDivInt8(0.5f, 0, 1.f, 0, divisor, dims));
}

TEST(HeliaMalformedModelTest, DivInt8RejectsMultiplierAboveTwoPow23) {
  using tflite::testing::InvokeDivInt8;
  int dims[] = {1, 4};
  const int8_t ones[4] = {1, 1, 1, 1};
  // 1 / (1/128 * 1e-6) is about 2^27: 127 / 1 would right-shift by -3.
  EXPECT_EQ(kTfLiteError, InvokeDivInt8(1.f / 128.f, 0, 1e-6f, 0, ones, dims,
                                        nullptr, 1.f, 127));
  // A multiplier of exactly 2^23 has shift 24; just below it has shift 23.
  const float at_limit = std::ldexp(1.f, -23);
  EXPECT_EQ(kTfLiteError,
            InvokeDivInt8(1.f, 0, at_limit, 0, ones, dims, nullptr, 1.f, 127));
  EXPECT_EQ(kTfLiteOk, InvokeDivInt8(1.f, 0, at_limit * 1.01f, 0, ones, dims,
                                     nullptr, 1.f, 127));
}

TEST(HeliaMalformedModelTest, DivInt8RejectsZeroDivisor) {
  using tflite::testing::InvokeDivInt8;
  const int8_t last_at_zero_point[4] = {1, 2, 4, 3};
  int dims[] = {1, 4};
  EXPECT_EQ(kTfLiteError,
            InvokeDivInt8(0.5f, 3, 1.f, 0, last_at_zero_point, dims));
}

TEST(HeliaMalformedModelTest, AddRejectsRankAboveSix) {
  using tflite::testing::PrepareAddRank;
  int rank7[] = {7, 1, 1, 1, 1, 1, 1, 4};
  EXPECT_EQ(kTfLiteError, PrepareAddRank(kTfLiteInt32, rank7));
  EXPECT_EQ(kTfLiteError, PrepareAddRank(kTfLiteFloat32, rank7));
  int rank6[] = {6, 1, 1, 1, 1, 1, 4};
  EXPECT_EQ(kTfLiteOk, PrepareAddRank(kTfLiteInt32, rank6));
}

TEST(HeliaMalformedModelTest, AddRejectsOutOfRangeZeroPoints) {
  using tflite::testing::PrepareAdd;
  const float scales[3] = {0.5f, 0.25f, 1.f};
  constexpr int kIntMin = std::numeric_limits<int>::min();
  EXPECT_EQ(kTfLiteError, PrepareAdd<int8_t>(scales, {kIntMin, 0, 0}, -1));
  EXPECT_EQ(kTfLiteError, PrepareAdd<int8_t>(scales, {0, 128, 0}, -1));
  EXPECT_EQ(kTfLiteError, PrepareAdd<int8_t>(scales, {0, 0, -129}, -1));
  EXPECT_EQ(kTfLiteOk, PrepareAdd<int8_t>(scales, {-128, 127, 5}, -1));
}

TEST(HeliaMalformedModelTest, AddRejectsUnquantizedInt8Int16) {
  using tflite::testing::PrepareAddUnquantized;
  for (int unquantized = 0; unquantized < 3; ++unquantized) {
    EXPECT_EQ(kTfLiteError, PrepareAddUnquantized<int8_t>(unquantized));
    EXPECT_EQ(kTfLiteError, PrepareAddUnquantized<int16_t>(unquantized));
  }
}

TEST(HeliaMalformedModelTest, AddRejectsInvalidScales) {
  tflite::testing::ExpectAddRejectsScales<int8_t>();
  tflite::testing::ExpectAddRejectsScales<int16_t>();
}

TEST(HeliaMalformedModelTest, AddOutputMultiplierBoundary) {
  tflite::testing::ExpectAddOutputMultiplierBoundary<int8_t>(20);
  tflite::testing::ExpectAddOutputMultiplierBoundary<int16_t>(15);
}

TF_LITE_MICRO_TESTS_MAIN
