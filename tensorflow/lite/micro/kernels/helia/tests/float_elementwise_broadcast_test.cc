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

// Float ADD, SUB and MUL broadcast coverage for the heliaCORE dispatch in
// kernels/helia/{add,sub,mul}.cc. see AmbiqAI/helia-rt#275
//
// helia_broadcast.h maps a TFLite shape pair onto one of the legs of the
// heliaCORE NHWC broadcast walk, and each leg gets a case here:
//   kSameShape     both operands identical      -> flat elementwise kernel
//   kScalarInput1  operand 1 is one element     -> scalar kernel, operands as given
//   kScalarInput2  operand 2 is one element     -> scalar kernel, operands swapped
//   kGeneral       any other 4-D compatible pair -> strided per-run walk
//   kUnsupported   rank above 4                 -> reference (float32) / rejected (float16)
// The scalar and general cases matter most for SUB, which is not commutative:
// the walk hands the swapped legs to the kernel in reversed order.

#include <algorithm>
#include <cmath>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/add.h"
#include "tensorflow/lite/kernels/internal/reference/mul.h"
#include "tensorflow/lite/kernels/internal/reference/sub.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

#if ARM_NN_ENABLE_F32 || ARM_NN_ENABLE_F16
#include "arm_nnfunctions_flt.h"
#endif

namespace tflite {
namespace testing {
namespace {

enum class BinaryOp { kAdd, kSub, kMul };

constexpr BinaryOp kOps[] = {BinaryOp::kAdd, BinaryOp::kSub, BinaryOp::kMul};
constexpr int kNumOps = 3;

// The widest operand is 1x2x3x2 NHWC.
constexpr int kWide = 12;

constexpr float kWideA[kWide] = {0.5f,  -1.25f, 2.75f, -0.125f,
                                 1.5f,  -2.5f,  3.25f, -0.75f,
                                 0.25f, -3.5f,  1.75f, -1.0f};
constexpr float kWideB[kWide] = {1.25f,  0.5f,  -0.75f, 2.5f,
                                 -1.5f,  3.0f,  0.125f, -2.25f,
                                 2.0f,   -0.5f, 1.0f,   -3.25f};
// 1x2x3x1: broadcast along the channel axis.
constexpr float kNarrow[6] = {0.75f, -1.5f, 2.25f, -0.25f, 3.5f, -2.75f};
// 1x1x1x2: broadcast along the batch, height and width axes.
constexpr float kChannel[2] = {1.5f, -2.25f};
constexpr float kScalar[1] = {-1.75f};

// Both sides apply the same single correctly rounded IEEE operation to the same
// pair of operands and then the same clamp, so they are expected to agree
// bit-exactly; the bound admits one float32 ulp at |output| <= 16, 2^-20 =
// 9.5e-07, and nothing more. Derived from the rounding, not measured: a wider
// gap is a heliaCORE finding for AmbiqAI/ns-cmsis-nn#428, not a bound to widen.
constexpr float kFloat32Tolerance = 1e-6f;

#if ARM_NN_ENABLE_F16
// Same argument in float16: one ulp at |output| <= 16 is 16 * 2^-10.
constexpr float kFloat16Tolerance = 16.0f * 9.765625e-04f;
#endif

TFLMRegistration OpRegistration(BinaryOp op) {
  switch (op) {
    case BinaryOp::kAdd:
      return tflite::Register_ADD();
    case BinaryOp::kSub:
      return tflite::Register_SUB();
    default:
      return tflite::Register_MUL();
  }
}

const char* OpName(BinaryOp op) {
  switch (op) {
    case BinaryOp::kAdd:
      return "ADD";
    case BinaryOp::kSub:
      return "SUB";
    default:
      return "MUL";
  }
}

float ApplyOp(BinaryOp op, float lhs, float rhs) {
  switch (op) {
    case BinaryOp::kAdd:
      return lhs + rhs;
    case BinaryOp::kSub:
      return lhs - rhs;
    default:
      return lhs * rhs;
  }
}

struct BuiltinParams {
  TfLiteAddParams add;
  TfLiteSubParams sub;
  TfLiteMulParams mul;
};

void* BuiltinFor(BinaryOp op, BuiltinParams* params) {
  params->add.activation = kTfLiteActNone;
  params->sub.activation = kTfLiteActNone;
  params->mul.activation = kTfLiteActNone;
  switch (op) {
    case BinaryOp::kAdd:
      return &params->add;
    case BinaryOp::kSub:
      return &params->sub;
    default:
      return &params->mul;
  }
}

// dims arrays are in TfLiteIntArray form: {rank, d0, d1, ...}.
template <typename T>
TfLiteStatus RunBinary(BinaryOp op, int* dims1, const T* input1, int* dims2,
                       const T* input2, int* dims_out, T* output,
                       TfLiteType tensor_type) {
  TfLiteTensor tensors[] = {
      CreateTensor(input1, IntArrayFromInts(dims1), false, tensor_type),
      CreateTensor(input2, IntArrayFromInts(dims2), false, tensor_type),
      CreateTensor(output, IntArrayFromInts(dims_out), false, tensor_type),
  };
  int inputs_array_data[] = {2, 0, 1};
  int outputs_array_data[] = {1, 2};

  BuiltinParams params = {};
  const TFLMRegistration registration = OpRegistration(op);

  micro::KernelRunner runner(registration, tensors, 3,
                             IntArrayFromInts(inputs_array_data),
                             IntArrayFromInts(outputs_array_data),
                             BuiltinFor(op, &params));
  const TfLiteStatus prepare_status = runner.InitAndPrepare();
  if (prepare_status != kTfLiteOk) {
    return prepare_status;
  }
  return runner.Invoke();
}

RuntimeShape ShapeOf(const int* dims) {
  return RuntimeShape(dims[0], &dims[1]);
}

// The TFLM reference for the same op, over the same shapes. The 4DSlow
// variants also cover the equal-shape case, so one call serves every class.
void ReferenceBinary(BinaryOp op, const int* dims1, const float* input1,
                     const int* dims2, const float* input2,
                     const int* dims_out, float* output) {
  tflite::ArithmeticParams params;
  SetActivationParams(std::numeric_limits<float>::lowest(),
                      std::numeric_limits<float>::max(), &params);
  const RuntimeShape shape1 = ShapeOf(dims1);
  const RuntimeShape shape2 = ShapeOf(dims2);
  const RuntimeShape shape_out = ShapeOf(dims_out);

  switch (op) {
    case BinaryOp::kAdd:
      reference_ops::BroadcastAdd4DSlow(params, shape1, input1, shape2, input2,
                                        shape_out, output);
      break;
    case BinaryOp::kSub:
      reference_ops::BroadcastSubSlow(params, shape1, input1, shape2, input2,
                                      shape_out, output);
      break;
    default:
      reference_ops::BroadcastMul4DSlow(params, shape1, input1, shape2, input2,
                                        shape_out, output);
      break;
  }
}

// Elementwise reference for equal shapes at any rank; the 4DSlow walkers above
// are limited to rank 4.
void ReferenceFlat(BinaryOp op, const float* input1, const float* input2,
                   float* output, int count) {
  for (int i = 0; i < count; ++i) {
    output[i] = ApplyOp(op, input1[i], input2[i]);
  }
}

void CheckFloat32(BinaryOp op, int* dims1, const float* input1, int* dims2,
                  const float* input2, int* dims_out, int count) {
  float output[kWide] = {};
  float expected[kWide] = {};

  ReferenceBinary(op, dims1, input1, dims2, input2, dims_out, expected);
  EXPECT_EQ(kTfLiteOk, RunBinary(op, dims1, input1, dims2, input2, dims_out,
                                 output, kTfLiteFloat32));

  for (int i = 0; i < count; ++i) {
    if (std::fabs(expected[i] - output[i]) > kFloat32Tolerance) {
      MicroPrintf("%s element %d", OpName(op), i);
    }
    EXPECT_NEAR(expected[i], output[i], kFloat32Tolerance);
  }
}

#if ARM_NN_ENABLE_F16
void Narrow(const float* source, float16_t* destination, int count) {
  for (int i = 0; i < count; ++i) {
    destination[i] = static_cast<float16_t>(source[i]);
  }
}

// NHWC broadcast of the reference expression, evaluated in float32 on the
// narrowed operands. TFLM has no float16 reference to call instead.
void ExpectedFloat16(BinaryOp op, const int* dims1, const float16_t* input1,
                     const int* dims2, const float16_t* input2,
                     const int* dims_out, float* output) {
  const RuntimeShape shape1 = RuntimeShape::ExtendedShape(4, ShapeOf(dims1));
  const RuntimeShape shape2 = RuntimeShape::ExtendedShape(4, ShapeOf(dims2));
  const RuntimeShape shape_out =
      RuntimeShape::ExtendedShape(4, ShapeOf(dims_out));

  int index = 0;
  for (int n = 0; n < shape_out.Dims(0); ++n) {
    for (int h = 0; h < shape_out.Dims(1); ++h) {
      for (int w = 0; w < shape_out.Dims(2); ++w) {
        for (int c = 0; c < shape_out.Dims(3); ++c) {
          const int i1 = Offset(shape1, n % shape1.Dims(0), h % shape1.Dims(1),
                                w % shape1.Dims(2), c % shape1.Dims(3));
          const int i2 = Offset(shape2, n % shape2.Dims(0), h % shape2.Dims(1),
                                w % shape2.Dims(2), c % shape2.Dims(3));
          output[index++] = ApplyOp(op, static_cast<float>(input1[i1]),
                                    static_cast<float>(input2[i2]));
        }
      }
    }
  }
}

void CheckFloat16(BinaryOp op, int* dims1, const float* input1, int* dims2,
                  const float* input2, int* dims_out, int count1, int count2,
                  int count) {
  float16_t narrow1[kWide] = {};
  float16_t narrow2[kWide] = {};
  float16_t output[kWide] = {};
  float expected[kWide] = {};

  Narrow(input1, narrow1, count1);
  Narrow(input2, narrow2, count2);

  EXPECT_EQ(kTfLiteOk, RunBinary(op, dims1, narrow1, dims2, narrow2, dims_out,
                                 output, kTfLiteFloat16));
  ExpectedFloat16(op, dims1, narrow1, dims2, narrow2, dims_out, expected);

  for (int i = 0; i < count; ++i) {
    if (std::fabs(expected[i] - static_cast<float>(output[i])) >
        kFloat16Tolerance) {
      MicroPrintf("%s element %d", OpName(op), i);
    }
    EXPECT_NEAR(expected[i], static_cast<float>(output[i]),
                kFloat16Tolerance);
  }
}
#endif  // ARM_NN_ENABLE_F16

}  // namespace
}  // namespace testing
}  // namespace tflite

// kernels/helia/{add,sub,mul}.cc fall back to reference_ops and still return
// kTfLiteOk when a heliaCORE entry point declines, so the comparisons below
// would pass with the optimized path never taken. Assert the dispatch
// precondition directly. see AmbiqAI/helia-rt#234
TEST(HeliaFloatBroadcastTest, OptimizedFloat32PathIsReachable) {
#if ARM_NN_ENABLE_F32
  const cmsis_nn_dims dims1 = {1, 2, 3, 1};
  const cmsis_nn_dims dims2 = {1, 1, 1, 2};
  const cmsis_nn_dims dims_out = {1, 2, 3, 2};
  float out[tflite::testing::kWide] = {};
  EXPECT_EQ(ARM_CMSIS_NN_SUCCESS,
            arm_elementwise_add_broadcast_f32(
                tflite::testing::kNarrow, &dims1, tflite::testing::kChannel,
                &dims2, out, &dims_out, -3.4e38f, 3.4e38f));
  EXPECT_EQ(ARM_CMSIS_NN_SUCCESS,
            arm_elementwise_sub_broadcast_f32(
                tflite::testing::kNarrow, &dims1, tflite::testing::kChannel,
                &dims2, out, &dims_out, -3.4e38f, 3.4e38f));
  EXPECT_EQ(ARM_CMSIS_NN_SUCCESS,
            arm_elementwise_mul_broadcast_f32(
                tflite::testing::kNarrow, &dims1, tflite::testing::kChannel,
                &dims2, out, &dims_out, -3.4e38f, 3.4e38f));
  EXPECT_EQ(ARM_CMSIS_NN_SUCCESS,
            arm_elementwise_sub_f32(tflite::testing::kWideA,
                                    tflite::testing::kWideB, out, -3.4e38f,
                                    3.4e38f, tflite::testing::kWide));
#else
  MicroPrintf(
      "ARM_NN_ENABLE_F32 unset: heliaCORE float32 not compiled in; the TFLM "
      "reference kernels are under test here.");
#endif
}

TEST(HeliaFloatBroadcastTest, Float32SameShape) {
  for (int i = 0; i < tflite::testing::kNumOps; ++i) {
    int dims1[] = {4, 1, 2, 3, 2};
    int dims2[] = {4, 1, 2, 3, 2};
    int dims_out[] = {4, 1, 2, 3, 2};
    tflite::testing::CheckFloat32(tflite::testing::kOps[i], dims1,
                                  tflite::testing::kWideA, dims2,
                                  tflite::testing::kWideB, dims_out,
                                  tflite::testing::kWide);
  }
}

TEST(HeliaFloatBroadcastTest, Float32ScalarInput1) {
  for (int i = 0; i < tflite::testing::kNumOps; ++i) {
    int dims1[] = {4, 1, 1, 1, 1};
    int dims2[] = {4, 1, 2, 3, 2};
    int dims_out[] = {4, 1, 2, 3, 2};
    tflite::testing::CheckFloat32(tflite::testing::kOps[i], dims1,
                                  tflite::testing::kScalar, dims2,
                                  tflite::testing::kWideB, dims_out,
                                  tflite::testing::kWide);
  }
}

TEST(HeliaFloatBroadcastTest, Float32ScalarInput2) {
  for (int i = 0; i < tflite::testing::kNumOps; ++i) {
    int dims1[] = {4, 1, 2, 3, 2};
    int dims2[] = {4, 1, 1, 1, 1};
    int dims_out[] = {4, 1, 2, 3, 2};
    tflite::testing::CheckFloat32(tflite::testing::kOps[i], dims1,
                                  tflite::testing::kWideA, dims2,
                                  tflite::testing::kScalar, dims_out,
                                  tflite::testing::kWide);
  }
}

TEST(HeliaFloatBroadcastTest, Float32General) {
  for (int i = 0; i < tflite::testing::kNumOps; ++i) {
    int dims1[] = {4, 1, 2, 3, 1};
    int dims2[] = {4, 1, 1, 1, 2};
    int dims_out[] = {4, 1, 2, 3, 2};
    tflite::testing::CheckFloat32(tflite::testing::kOps[i], dims1,
                                  tflite::testing::kNarrow, dims2,
                                  tflite::testing::kChannel, dims_out,
                                  tflite::testing::kWide);
  }
}

// Rank 5 has no NHWC spelling, so this must come back from reference_ops.
TEST(HeliaFloatBroadcastTest, Float32Rank5FallsBackToReference) {
  for (int i = 0; i < tflite::testing::kNumOps; ++i) {
    const tflite::testing::BinaryOp op = tflite::testing::kOps[i];
    int dims1[] = {5, 1, 1, 2, 3, 2};
    int dims2[] = {5, 1, 1, 2, 3, 2};
    int dims_out[] = {5, 1, 1, 2, 3, 2};
    float output[tflite::testing::kWide] = {};
    float expected[tflite::testing::kWide] = {};

    tflite::testing::ReferenceFlat(op, tflite::testing::kWideA,
                                   tflite::testing::kWideB, expected,
                                   tflite::testing::kWide);
    EXPECT_EQ(kTfLiteOk, tflite::testing::RunBinary(
                             op, dims1, tflite::testing::kWideA, dims2,
                             tflite::testing::kWideB, dims_out, output,
                             kTfLiteFloat32));

    for (int e = 0; e < tflite::testing::kWide; ++e) {
      EXPECT_EQ(expected[e], output[e]);
    }
  }
}

#if ARM_NN_ENABLE_F16
TEST(HeliaFloatBroadcastTest, Float16SameShape) {
  for (int i = 0; i < tflite::testing::kNumOps; ++i) {
    int dims1[] = {4, 1, 2, 3, 2};
    int dims2[] = {4, 1, 2, 3, 2};
    int dims_out[] = {4, 1, 2, 3, 2};
    tflite::testing::CheckFloat16(tflite::testing::kOps[i], dims1,
                                  tflite::testing::kWideA, dims2,
                                  tflite::testing::kWideB, dims_out,
                                  tflite::testing::kWide,
                                  tflite::testing::kWide,
                                  tflite::testing::kWide);
  }
}

TEST(HeliaFloatBroadcastTest, Float16ScalarInput1) {
  for (int i = 0; i < tflite::testing::kNumOps; ++i) {
    int dims1[] = {4, 1, 1, 1, 1};
    int dims2[] = {4, 1, 2, 3, 2};
    int dims_out[] = {4, 1, 2, 3, 2};
    tflite::testing::CheckFloat16(tflite::testing::kOps[i], dims1,
                                  tflite::testing::kScalar, dims2,
                                  tflite::testing::kWideB, dims_out, 1,
                                  tflite::testing::kWide,
                                  tflite::testing::kWide);
  }
}

TEST(HeliaFloatBroadcastTest, Float16ScalarInput2) {
  for (int i = 0; i < tflite::testing::kNumOps; ++i) {
    int dims1[] = {4, 1, 2, 3, 2};
    int dims2[] = {4, 1, 1, 1, 1};
    int dims_out[] = {4, 1, 2, 3, 2};
    tflite::testing::CheckFloat16(tflite::testing::kOps[i], dims1,
                                  tflite::testing::kWideA, dims2,
                                  tflite::testing::kScalar, dims_out,
                                  tflite::testing::kWide, 1,
                                  tflite::testing::kWide);
  }
}

TEST(HeliaFloatBroadcastTest, Float16General) {
  for (int i = 0; i < tflite::testing::kNumOps; ++i) {
    int dims1[] = {4, 1, 2, 3, 1};
    int dims2[] = {4, 1, 1, 1, 2};
    int dims_out[] = {4, 1, 2, 3, 2};
    tflite::testing::CheckFloat16(tflite::testing::kOps[i], dims1,
                                  tflite::testing::kNarrow, dims2,
                                  tflite::testing::kChannel, dims_out, 6, 2,
                                  tflite::testing::kWide);
  }
}

// float16 has no reference fallback, so an unsupported shape pair has to fail
// at AllocateTensors rather than mid-inference.
TEST(HeliaFloatBroadcastTest, Float16Rank5IsRejectedAtPrepare) {
  for (int i = 0; i < tflite::testing::kNumOps; ++i) {
    int dims1[] = {5, 1, 1, 2, 3, 2};
    int dims2[] = {5, 1, 1, 2, 3, 2};
    int dims_out[] = {5, 1, 1, 2, 3, 2};
    float16_t narrow1[tflite::testing::kWide] = {};
    float16_t narrow2[tflite::testing::kWide] = {};
    float16_t output[tflite::testing::kWide] = {};
    tflite::testing::Narrow(tflite::testing::kWideA, narrow1,
                            tflite::testing::kWide);
    tflite::testing::Narrow(tflite::testing::kWideB, narrow2,
                            tflite::testing::kWide);

    EXPECT_EQ(kTfLiteError,
              tflite::testing::RunBinary(tflite::testing::kOps[i], dims1,
                                         narrow1, dims2, narrow2, dims_out,
                                         output, kTfLiteFloat16));
  }
}
#elif defined(__ARM_FEATURE_MVE) && ((__ARM_FEATURE_MVE) & 2)

// helia.inc defines ARM_NN_ENABLE_F16 for TARGET_ARCH=cortex-m55 only, and a
// silent compile-out would drop these cases while the leg still reported
// success. see AmbiqAI/helia-rt#231
TEST(HeliaFloatBroadcastTest, Float16CoverageMustNotSilentlyDisappear) {
  FAIL(
      "ARM_NN_ENABLE_F16 is not defined on a build with MVE floating point. "
      "The float16 broadcast coverage silently compiled out.");
}

#endif  // ARM_NN_ENABLE_F16

TF_LITE_MICRO_TESTS_MAIN
