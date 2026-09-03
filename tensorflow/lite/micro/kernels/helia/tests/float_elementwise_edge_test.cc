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

// NaN-propagation coverage for the helia float ADD and MUL kernels. The shared
// kernels/add_test.cc and mul_test.cc use only finite operands, so nothing else
// in the tree observes what arm_elementwise_add/mul_f32/f16 do with a NaN. see
// #227
//
// Asserted here are the cases whose IEEE-754 result is NaN regardless of the
// activation clamp:
//   * NaN op x  -> NaN
//   * (+Inf) + (-Inf) -> NaN
//   * 0 * (+Inf) -> NaN
// A finite control element rides in the same tensor so a kernel that gives up
// early on the non-finite lanes is still caught.
//
// Deliberately NOT asserted: Inf propagating through as Inf. With
// kTfLiteActNone the float activation range is +/-FLT_MAX (and the heliaCORE
// finite float16 sentinels for the float16 path), so a conforming kernel
// clamps a raw infinity to the finite bound. Only the NaN-producing cases are
// implementation-independent.
//
// These are TRUE CONTRACT assertions, unlike the tanh/logistic NaN cases in
// float_activation_edge_test.cc: the elementwise clamp helpers reclassify NaN
// on the integer bit pattern, which survives -Ofast, so this holds on every
// toolchain rather than only outside fast-math. see ns-cmsis-nn#380
//
// SUB is not covered: kernels/helia/sub.cc has no float dispatch into
// heliaCORE (float32 SUB runs the TFLM reference and there is no float16 SUB),
// so there is no optimized path to guard.

#include <cmath>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
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

// Four elements, equal shapes: the helia float16 ADD/MUL kernels have no
// broadcast support, so both operands must have the same shape for the
// optimized path to be selected.
constexpr int kCount = 4;

template <typename T>
void RunBinary(const TFLMRegistration& registration, const T* lhs, const T* rhs,
               T* output, TfLiteType tensor_type, void* builtin_data) {
  int dims_data[] = {2, 1, kCount};
  TfLiteTensor tensors[] = {
      CreateTensor(lhs, IntArrayFromInts(dims_data), false, tensor_type),
      CreateTensor(rhs, IntArrayFromInts(dims_data), false, tensor_type),
      CreateTensor(output, IntArrayFromInts(dims_data), false, tensor_type),
  };
  int inputs_array_data[] = {2, 0, 1};
  int outputs_array_data[] = {1, 2};

  micro::KernelRunner runner(registration, tensors, 3,
                             IntArrayFromInts(inputs_array_data),
                             IntArrayFromInts(outputs_array_data),
                             builtin_data);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

}  // namespace
}  // namespace testing
}  // namespace tflite

// kernels/helia/add.cc and mul.cc fall back to reference_ops and still return
// kTfLiteOk when the heliaCORE entry point declines, and the reference kernels
// propagate NaN, so a tightening of argument validation would flip the tests
// below GREEN while the code under test never ran. Assert the dispatch
// precondition directly; the link probe proves only that the symbol exists.
// see #234
TEST(HeliaFloatElementwiseEdgeTest, OptimizedFloat32PathIsReachable) {
#if ARM_NN_ENABLE_F32
  const float lhs[tflite::testing::kCount] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float rhs[tflite::testing::kCount] = {0.5f, 0.5f, 0.5f, 0.5f};
  float out[tflite::testing::kCount] = {};
  EXPECT_EQ(ARM_CMSIS_NN_SUCCESS,
            arm_elementwise_add_f32(lhs, rhs, out, -3.4e38f, 3.4e38f,
                                    tflite::testing::kCount));
  EXPECT_EQ(ARM_CMSIS_NN_SUCCESS,
            arm_elementwise_mul_f32(lhs, rhs, out, -3.4e38f, 3.4e38f,
                                    tflite::testing::kCount));
#else
  MicroPrintf(
      "ARM_NN_ENABLE_F32 unset: heliaCORE float32 not compiled in; the TFLM "
      "reference kernels are under test here.");
#endif
}

TEST(HeliaFloatElementwiseEdgeTest, AddFloat32PropagatesNan) {
  // lhs/rhs: NaN + 1, 1 + NaN, (+Inf) + (-Inf), 1.5 + 2.5 (finite control).
  const float lhs[tflite::testing::kCount] = {NAN, 1.0f, INFINITY, 1.5f};
  const float rhs[tflite::testing::kCount] = {1.0f, NAN, -INFINITY, 2.5f};
  float output[tflite::testing::kCount] = {};

  TfLiteAddParams params = {};
  params.activation = kTfLiteActNone;
  tflite::testing::RunBinary(tflite::Register_ADD(), lhs, rhs, output,
                             kTfLiteFloat32, &params);

  EXPECT_TRUE(std::isnan(output[0]));
  EXPECT_TRUE(std::isnan(output[1]));
  EXPECT_TRUE(std::isnan(output[2]));
  EXPECT_NEAR(4.0f, output[3], 1e-6f);
}

TEST(HeliaFloatElementwiseEdgeTest, MulFloat32PropagatesNan) {
  // lhs/rhs: NaN * 1, 1 * NaN, 0 * (+Inf), 1.5 * 2.0 (finite control).
  const float lhs[tflite::testing::kCount] = {NAN, 1.0f, 0.0f, 1.5f};
  const float rhs[tflite::testing::kCount] = {1.0f, NAN, INFINITY, 2.0f};
  float output[tflite::testing::kCount] = {};

  TfLiteMulParams params = {};
  params.activation = kTfLiteActNone;
  tflite::testing::RunBinary(tflite::Register_MUL(), lhs, rhs, output,
                             kTfLiteFloat32, &params);

  EXPECT_TRUE(std::isnan(output[0]));
  EXPECT_TRUE(std::isnan(output[1]));
  EXPECT_TRUE(std::isnan(output[2]));
  EXPECT_NEAR(3.0f, output[3], 1e-6f);
}

#if ARM_NN_ENABLE_F16
TEST(HeliaFloatElementwiseEdgeTest, AddFloat16PropagatesNan) {
  const float16_t lhs[tflite::testing::kCount] = {
      static_cast<float16_t>(NAN), static_cast<float16_t>(1.0f),
      static_cast<float16_t>(INFINITY), static_cast<float16_t>(1.5f)};
  const float16_t rhs[tflite::testing::kCount] = {
      static_cast<float16_t>(1.0f), static_cast<float16_t>(NAN),
      static_cast<float16_t>(-INFINITY), static_cast<float16_t>(2.5f)};
  float16_t output[tflite::testing::kCount] = {};

  TfLiteAddParams params = {};
  params.activation = kTfLiteActNone;
  tflite::testing::RunBinary(tflite::Register_ADD(), lhs, rhs, output,
                             kTfLiteFloat16, &params);

  EXPECT_TRUE(std::isnan(static_cast<float>(output[0])));
  EXPECT_TRUE(std::isnan(static_cast<float>(output[1])));
  EXPECT_TRUE(std::isnan(static_cast<float>(output[2])));
  EXPECT_NEAR(4.0f, static_cast<float>(output[3]), 1e-3f);
}

TEST(HeliaFloatElementwiseEdgeTest, MulFloat16PropagatesNan) {
  const float16_t lhs[tflite::testing::kCount] = {
      static_cast<float16_t>(NAN), static_cast<float16_t>(1.0f),
      static_cast<float16_t>(0.0f), static_cast<float16_t>(1.5f)};
  const float16_t rhs[tflite::testing::kCount] = {
      static_cast<float16_t>(1.0f), static_cast<float16_t>(NAN),
      static_cast<float16_t>(INFINITY), static_cast<float16_t>(2.0f)};
  float16_t output[tflite::testing::kCount] = {};

  TfLiteMulParams params = {};
  params.activation = kTfLiteActNone;
  tflite::testing::RunBinary(tflite::Register_MUL(), lhs, rhs, output,
                             kTfLiteFloat16, &params);

  EXPECT_TRUE(std::isnan(static_cast<float>(output[0])));
  EXPECT_TRUE(std::isnan(static_cast<float>(output[1])));
  EXPECT_TRUE(std::isnan(static_cast<float>(output[2])));
  EXPECT_NEAR(3.0f, static_cast<float>(output[3]), 1e-3f);
}
#elif defined(__ARM_FEATURE_MVE) && ((__ARM_FEATURE_MVE) & 2)

// See the matching guard in float_activation_edge_test.cc: helia.inc defines
// ARM_NN_ENABLE_F16 for TARGET_ARCH=cortex-m55 only, and a silent compile-out
// would drop these cases while the leg still reported success
// (helia-rt#231). Known gap: ATfE builds cortex-m55 with +nomve
// (helia-rt#225), so this guard cannot fire there.
TEST(HeliaFloatElementwiseEdgeTest, Float16CoverageMustNotSilentlyDisappear) {
  FAIL(
      "ARM_NN_ENABLE_F16 is not defined on a build with MVE floating point. "
      "The float16 elementwise coverage silently compiled out.");
}

#endif  // ARM_NN_ENABLE_F16

TF_LITE_MICRO_TESTS_MAIN
