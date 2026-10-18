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

// int16 EQUAL, NOT_EQUAL, GREATER, GREATER_EQUAL, LESS and LESS_EQUAL through
// the helia kernel, against the comparison of the real values
// (q - zero_point) * scale, which double holds exactly for int16 operands and
// float scales. With equal scales and zero points, and for unquantized int16
// (scale 0, zero point 0, what the TFLite converter emits), that is the raw
// int16 comparison, and the kernel must match it exactly. With unequal scales
// the kernel resolves real values to max_scale / 2^15 (2^14 with a nonzero
// zero point): exactly equal reals and reals one LSB of either input apart
// must compare right, closer reals may compare equal. Rank 4 runs heliaCORE
// and rank 5 the reference fallback. Out-of-range zero points and non-finite
// or non-positive scales must fail Prepare.
// see AmbiqAI/helia-rt#470

#include <cstdint>
#include <cstring>
#include <limits>

#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace tflite {
namespace testing {
namespace {

enum class Op { kEqual, kNotEqual, kGreater, kGreaterEqual, kLess, kLessEqual };
constexpr Op kAllOps[] = {Op::kEqual,        Op::kNotEqual, Op::kGreater,
                          Op::kGreaterEqual, Op::kLess,     Op::kLessEqual};

TFLMRegistration Registration(Op op) {
  switch (op) {
    case Op::kEqual:
      return Register_EQUAL();
    case Op::kNotEqual:
      return Register_NOT_EQUAL();
    case Op::kGreater:
      return Register_GREATER();
    case Op::kGreaterEqual:
      return Register_GREATER_EQUAL();
    case Op::kLess:
      return Register_LESS();
    case Op::kLessEqual:
      break;
  }
  return Register_LESS_EQUAL();
}

bool Expected(Op op, double a, double b) {
  switch (op) {
    case Op::kEqual:
      return a == b;
    case Op::kNotEqual:
      return a != b;
    case Op::kGreater:
      return a > b;
    case Op::kGreaterEqual:
      return a >= b;
    case Op::kLess:
      return a < b;
    case Op::kLessEqual:
      break;
  }
  return a <= b;
}

struct Quant {
  float scale1;
  int zero_point1;
  float scale2;
  int zero_point2;
};

constexpr int kMaxElements = 3 * 260;

// Fills (x, x + 1), (x + 1, x) and (x, x) triples over the int16 range,
// including both ends.
int FillAdjacentPairs(int16_t* a, int16_t* b) {
  int n = 0;
  for (int x = -32768; x < 32767; x += 257) {
    const int16_t lo = static_cast<int16_t>(x);
    const int16_t hi = static_cast<int16_t>(x + 1);
    a[n] = lo, b[n++] = hi;
    a[n] = hi, b[n++] = lo;
    a[n] = lo, b[n++] = lo;
  }
  a[n] = 32767, b[n++] = 32766;
  a[n] = 32766, b[n++] = 32767;
  a[n] = 32767, b[n++] = 32767;
  return n;
}

// A scale of 0 stands for an unquantized tensor, compared raw.
double Real(int16_t q, float scale, int zero_point) {
  return (static_cast<double>(q) - zero_point) *
         (scale == 0.0f ? 1.0 : static_cast<double>(scale));
}

TfLiteTensor CreateInput(int16_t* data, TfLiteIntArray* dims, float scale,
                         int zero_point) {
  if (scale == 0.0f && zero_point == 0) {
    return CreateTensor(data, dims);
  }
  return CreateQuantizedTensor(data, dims, scale, zero_point);
}

// Runs op on a and b (b broadcast when b_count == 1) with the given rank and
// returns the number of outputs that differ from the real-value comparison.
int CountWrong(Op op, const Quant& q, int rank, int16_t* a, int16_t* b,
               int count, int b_count) {
  static bool output[kMaxElements];
  int a_dims[6] = {rank};
  int b_dims[6] = {rank};
  int o_dims[6] = {rank};
  for (int i = 1; i < rank; ++i) {
    a_dims[i] = b_dims[i] = o_dims[i] = 1;
  }
  a_dims[rank] = o_dims[rank] = count;
  b_dims[rank] = b_count;

  TfLiteTensor tensors[3] = {
      CreateInput(a, IntArrayFromInts(a_dims), q.scale1, q.zero_point1),
      CreateInput(b, IntArrayFromInts(b_dims), q.scale2, q.zero_point2),
      CreateTensor(output, IntArrayFromInts(o_dims)),
  };
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = Registration(op);
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), nullptr);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  memset(output, 0, sizeof(output));
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  int wrong = 0;
  for (int i = 0; i < count; ++i) {
    const int16_t bi = b[b_count == 1 ? 0 : i];
    wrong += output[i] != Expected(op, Real(a[i], q.scale1, q.zero_point1),
                                   Real(bi, q.scale2, q.zero_point2));
  }
  return wrong;
}

constexpr Quant kQuants[] = {
    {1.0f / 32768, 0, 1.0f / 32768, 0},
    {1.0f / 4096, 0, 1.0f / 4096, 0},
    {1.0f / 1024, 0, 1.0f / 1024, 0},
    {0.01f, 0, 0.01f, 0},
    {0.5f, 0, 0.5f, 0},
    {4.0f, 0, 4.0f, 0},
    {1e-6f, 0, 1e-6f, 0},
    {1.0f / 32768, 5, 1.0f / 32768, -7},
    {1.0f / 32768, 0, 2.0f / 32768, 0},
    {3.0f / 32768, 0, 1.0f / 32768, 0},
};

// Returns the Prepare status for a two-element comparison with these
// quantization parameters, and checks that its temporaries were released.
TfLiteStatus PrepareStatus(Op op, const Quant& q) {
  int16_t a[2] = {0, 1};
  int16_t b[2] = {1, 0};
  bool output[2];
  int dims[] = {1, 2};
  TfLiteTensor tensors[3] = {
      CreateQuantizedTensor(a, IntArrayFromInts(dims), q.scale1, q.zero_point1),
      CreateQuantizedTensor(b, IntArrayFromInts(dims), q.scale2, q.zero_point2),
      CreateTensor(output, IntArrayFromInts(dims)),
  };
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration = Registration(op);
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), nullptr);
  const TfLiteStatus status = runner.InitAndPrepare();
  // InitAndPrepare() checks this only when Prepare succeeds.
  EXPECT_TRUE(runner.ValidateTempBufferDeallocated());
  return status;
}

// Pairs at unequal scales whose real values are exactly equal, or one LSB of
// the finer input (input1) above or below that.
int FillBoundaryPairs(const Quant& q, int ratio_num, int ratio_den, int16_t* a,
                      int16_t* b) {
  // scale2 / scale1 == ratio_num / ratio_den, so
  // (a - zp1) * ratio_den == (b - zp2) * ratio_num gives equal reals.
  int n = 0;
  for (int k = -32768 / ratio_num + 1; k < 32767 / ratio_num; k += 97) {
    const int qa = k * ratio_num + q.zero_point1;
    const int qb = k * ratio_den + q.zero_point2;
    if (qa - 1 < -32768 || qa + 1 > 32767 || qb < -32768 || qb > 32767) {
      continue;
    }
    for (int d = -1; d <= 1; ++d) {
      a[n] = static_cast<int16_t>(qa + d), b[n++] = static_cast<int16_t>(qb);
    }
  }
  return n;
}

void ExpectAdjacentPairs(Op op, int rank) {
  static int16_t a[kMaxElements];
  static int16_t b[kMaxElements];
  const int count = FillAdjacentPairs(a, b);
  for (const Quant& q : kQuants) {
    EXPECT_EQ(0, CountWrong(op, q, rank, a, b, count, count));
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::ExpectAdjacentPairs;
using tflite::testing::Op;

TEST(HeliaComparisonsInt16Test, EqualAdjacentPairs) {
  ExpectAdjacentPairs(Op::kEqual, 4);
}

TEST(HeliaComparisonsInt16Test, NotEqualAdjacentPairs) {
  ExpectAdjacentPairs(Op::kNotEqual, 4);
}

TEST(HeliaComparisonsInt16Test, GreaterAdjacentPairs) {
  ExpectAdjacentPairs(Op::kGreater, 4);
}

TEST(HeliaComparisonsInt16Test, GreaterEqualAdjacentPairs) {
  ExpectAdjacentPairs(Op::kGreaterEqual, 4);
}

TEST(HeliaComparisonsInt16Test, LessAdjacentPairs) {
  ExpectAdjacentPairs(Op::kLess, 4);
}

TEST(HeliaComparisonsInt16Test, LessEqualAdjacentPairs) {
  ExpectAdjacentPairs(Op::kLessEqual, 4);
}

// Above rank 4 the kernel takes the reference comparison with the same
// parameters.
TEST(HeliaComparisonsInt16Test, RankFiveReferenceFallback) {
  for (Op op : tflite::testing::kAllOps) {
    ExpectAdjacentPairs(op, 5);
  }
}

// A one-element input2 broadcasts against every element of input1.
TEST(HeliaComparisonsInt16Test, ScalarBroadcast) {
  static int16_t a[tflite::testing::kMaxElements];
  int16_t b[1] = {1000};
  int count = 0;
  for (int x = 990; x <= 1010; ++x) {
    a[count++] = static_cast<int16_t>(x);
  }
  const tflite::testing::Quant q = {1.0f / 32768, 0, 1.0f / 32768, 0};
  for (Op op : tflite::testing::kAllOps) {
    EXPECT_EQ(0, tflite::testing::CountWrong(op, q, 4, a, b, count, 1));
  }
}

// Unquantized int16 tensors compare raw, on both routes.
TEST(HeliaComparisonsInt16Test, UnquantizedComparesRaw) {
  const tflite::testing::Quant q = {0.0f, 0, 0.0f, 0};
  static int16_t a[tflite::testing::kMaxElements];
  static int16_t b[tflite::testing::kMaxElements];
  const int count = tflite::testing::FillAdjacentPairs(a, b);
  for (Op op : tflite::testing::kAllOps) {
    for (int rank = 4; rank <= 5; ++rank) {
      EXPECT_EQ(0,
                tflite::testing::CountWrong(op, q, rank, a, b, count, count));
    }
  }
}

TEST(HeliaComparisonsInt16Test, UnequalScaleEqualityBoundary) {
  using tflite::testing::Quant;
  struct Case {
    Quant q;
    int ratio_num, ratio_den;
  };
  const Case cases[] = {
      {{1.0f / 32768, 0, 3.0f / 32768, 0}, 3, 1},
      {{1.0f / 32768, 0, 1.5f / 32768, 0}, 3, 2},
      {{1.0f / 32768, 5, 3.0f / 32768, -7}, 3, 1},
      {{0.25f, 0, 1.25f, 0}, 5, 1},
  };
  static int16_t a[tflite::testing::kMaxElements];
  static int16_t b[tflite::testing::kMaxElements];
  for (const Case& c : cases) {
    const int count =
        tflite::testing::FillBoundaryPairs(c.q, c.ratio_num, c.ratio_den, a, b);
    EXPECT_GT(count, 100);
    for (Op op : tflite::testing::kAllOps) {
      for (int rank = 4; rank <= 5; ++rank) {
        EXPECT_EQ(
            0, tflite::testing::CountWrong(op, c.q, rank, a, b, count, count));
      }
    }
  }
}

// heliaCORE and the reference evaluate (q - zero_point) << left_shift in int32.
TEST(HeliaComparisonsInt16Test, OutOfRangeZeroPointFailsPrepare) {
  const float s = 1.0f / 32768;
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareStatus(Op::kLess, {s, 200000, s, 0}));
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareStatus(Op::kEqual, {s, 0, s, -32769}));
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareStatus(Op::kGreater, {s, 0, s, 32768}));
  EXPECT_EQ(kTfLiteOk, tflite::testing::PrepareStatus(Op::kGreater,
                                                      {s, -32768, s, 32767}));
}

TEST(HeliaComparisonsInt16Test, InvalidScaleFailsPrepare) {
  const float s = 1.0f / 32768;
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareStatus(Op::kLess, {inf, 0, s, 0}));
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareStatus(Op::kEqual, {inf, 0, inf, 0}));
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareStatus(Op::kEqual, {s, 0, nan, 0}));
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareStatus(Op::kLess, {-s, 0, s, 0}));
  // One unquantized input against a quantized one.
  EXPECT_EQ(kTfLiteError,
            tflite::testing::PrepareStatus(Op::kLess, {0.0f, 0, s, 0}));
}

TF_LITE_MICRO_TESTS_MAIN
