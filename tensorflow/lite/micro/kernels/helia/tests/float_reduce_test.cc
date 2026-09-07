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

// Float MEAN and REDUCE_SUM coverage for the heliaCORE dispatch in
// kernels/helia/reduce_common.cc. see AmbiqAI/helia-rt#275
//
// arm_nn_mean_f32/_f16 and arm_reduce_sum_f32/_f16 take a 4-D NHWC shape and a
// 4-D axis mask, so kernels/helia/reduce_common.cc routes any axis-mask
// combination at rank up to 4 and leaves everything else on the reference. The
// rank-5 cases below therefore run the reference path in the same binary.

#include "tensorflow/lite/kernels/internal/reference/reduce.h"

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

constexpr int kElements = 24;

// Values chosen so no partial sum is exactly representable, which keeps the
// two accumulation orders from agreeing by construction.
constexpr float kInput[kElements] = {
    0.1f,  -0.2f, 0.3f,  -0.4f, 0.55f, -0.65f, 0.7f,  -0.8f,
    0.9f,  -1.1f, 1.2f,  -1.3f, 1.45f, -1.55f, 1.7f,  -1.8f,
    1.9f,  -2.1f, 2.2f,  -2.3f, 2.45f, -2.55f, 2.7f,  -2.8f};

// Both sides sum the same n <= 24 values with |x| <= 2.8 in some order.
// Sequential float32 summation of n terms carries an absolute error of at most
// (n - 1) * u * sum|x| with u = 2^-24, i.e. 23 * 5.96e-08 * 67.2 = 9.2e-05, and
// the two orders can sit on opposite sides of the exact value. MEAN divides by
// the reduction count, so the same argument gives a smaller bound there and
// this one covers both. Derived from the rounding, not measured: a failure is a
// heliaCORE finding for AmbiqAI/ns-cmsis-nn#428, not a bound to widen.
constexpr float kFloat32Tolerance = 2e-4f;

#if ARM_NN_ENABLE_F16
constexpr int kF16Elements = 12;

constexpr float kF16Input[kF16Elements] = {0.5f,  -1.5f, 2.0f,  -0.5f,
                                           1.0f,  -2.0f, 1.5f,  -1.0f,
                                           0.5f,  -0.5f, 2.0f,  -1.5f};

// The float16 reductions accumulate in float32 and round once to float16, so
// the spread against a float32 reference is one rounding of the result plus
// the float32 accumulation-order term above, which is far below half a
// float16 ulp here. Two float16 ulp at |output| <= 8 is 2 * 8 * 2^-10.
constexpr float kFloat16Tolerance = 2.0f * 8.0f * 9.765625e-04f;
#endif

template <typename T>
void RunReduce(const TFLMRegistration& registration, int* input_dims_data,
               const T* input, int* axis_dims_data, const int32_t* axis,
               int* output_dims_data, T* output, TfLiteType tensor_type,
               bool keep_dims) {
  TfLiteTensor tensors[] = {
      CreateTensor(input, IntArrayFromInts(input_dims_data), false,
                   tensor_type),
      CreateTensor(axis, IntArrayFromInts(axis_dims_data), false,
                   kTfLiteInt32),
      CreateTensor(output, IntArrayFromInts(output_dims_data), false,
                   tensor_type),
  };
  int inputs_array_data[] = {2, 0, 1};
  int outputs_array_data[] = {1, 2};

  TfLiteReducerParams params = {};
  params.keep_dims = keep_dims;

  micro::KernelRunner runner(registration, tensors, 3,
                             IntArrayFromInts(inputs_array_data),
                             IntArrayFromInts(outputs_array_data), &params);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

// Scratch for the reference reducers. reduce_common.cc sizes these the same
// way, which caps a reference comparison at two reduced axes.
int temp_index[8];
int resolved_axis[2];

void ReferenceMean(const float* input, const int* input_shape,
                   int input_num_dims, float* output, const int* output_shape,
                   int output_num_dims, const int32_t* axis, int num_axis,
                   bool keep_dims) {
  EXPECT_TRUE(reference_ops::Mean(input, input_shape, input_num_dims, output,
                                  output_shape, output_num_dims, axis, num_axis,
                                  keep_dims, temp_index, resolved_axis,
                                  output));
}

void ReferenceSum(const float* input, const int* input_shape,
                  int input_num_dims, float* output, const int* output_shape,
                  int output_num_dims, const int32_t* axis, int num_axis,
                  bool keep_dims) {
  EXPECT_TRUE(reference_ops::ReduceGeneric<float>(
      input, input_shape, input_num_dims, output, output_shape, output_num_dims,
      axis, num_axis, keep_dims, temp_index, resolved_axis, /*init_value=*/0.f,
      [](const float current, const float in) -> float { return in + current; }));
}

void ExpectNear(const float* expected, const float* actual, int count,
                float tolerance) {
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(expected[i], actual[i], tolerance);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

// kernels/helia/reduce_common.cc falls back to the reference and still returns
// kTfLiteOk when the heliaCORE entry point declines, so the comparisons below
// would pass with the optimized path never taken. Assert the dispatch
// precondition directly. see AmbiqAI/helia-rt#234
TEST(HeliaFloatReduceTest, OptimizedFloat32PathIsReachable) {
#if ARM_NN_ENABLE_F32
  const cmsis_nn_dims in_dims = {2, 2, 3, 2};
  const cmsis_nn_dims axis_dims = {0, 1, 1, 0};
  const cmsis_nn_dims out_dims = {2, 1, 1, 2};
  float out[4] = {};
  EXPECT_EQ(ARM_CMSIS_NN_SUCCESS,
            arm_nn_mean_f32(tflite::testing::kInput, &in_dims, &axis_dims, out,
                            &out_dims));
  EXPECT_EQ(ARM_CMSIS_NN_SUCCESS,
            arm_reduce_sum_f32(tflite::testing::kInput, &in_dims, &axis_dims,
                               out, &out_dims));
#else
  MicroPrintf(
      "ARM_NN_ENABLE_F32 unset: heliaCORE float32 not compiled in; the TFLM "
      "reference kernels are under test here.");
#endif
}

// Rank 4, spatial axes, keep_dims: the case reduce_common.cc used to hand to
// the specialized reference_ops::Mean overload.
TEST(HeliaFloatReduceTest, MeanFloat32Rank4SpatialAxes) {
  int input_dims_data[] = {4, 2, 2, 3, 2};
  int axis_dims_data[] = {1, 2};
  int output_dims_data[] = {4, 2, 1, 1, 2};
  const int32_t axis[] = {1, 2};
  const int input_shape[] = {2, 2, 3, 2};
  const int output_shape[] = {2, 1, 1, 2};

  float output[4] = {};
  float expected[4] = {};
  tflite::testing::ReferenceMean(tflite::testing::kInput, input_shape, 4,
                                 expected, output_shape, 4, axis, 2,
                                 /*keep_dims=*/true);
  tflite::testing::RunReduce(tflite::Register_MEAN(), input_dims_data,
                             tflite::testing::kInput, axis_dims_data, axis,
                             output_dims_data, output, kTfLiteFloat32,
                             /*keep_dims=*/true);

  tflite::testing::ExpectNear(expected, output, 4,
                              tflite::testing::kFloat32Tolerance);
}

// Rank 4, innermost axis, keep_dims off: a non-contiguous reduction that
// exercises the generic walk rather than the flattened-suffix fast path.
TEST(HeliaFloatReduceTest, MeanFloat32Rank4ChannelAxis) {
  int input_dims_data[] = {4, 2, 2, 3, 2};
  int axis_dims_data[] = {1, 1};
  int output_dims_data[] = {3, 2, 2, 3};
  const int32_t axis[] = {3};
  const int input_shape[] = {2, 2, 3, 2};
  const int output_shape[] = {2, 2, 3};

  float output[12] = {};
  float expected[12] = {};
  tflite::testing::ReferenceMean(tflite::testing::kInput, input_shape, 4,
                                 expected, output_shape, 3, axis, 1,
                                 /*keep_dims=*/false);
  tflite::testing::RunReduce(tflite::Register_MEAN(), input_dims_data,
                             tflite::testing::kInput, axis_dims_data, axis,
                             output_dims_data, output, kTfLiteFloat32,
                             /*keep_dims=*/false);

  tflite::testing::ExpectNear(expected, output, 12,
                              tflite::testing::kFloat32Tolerance);
}

// Rank 2, negative axis index.
TEST(HeliaFloatReduceTest, MeanFloat32Rank2NegativeAxis) {
  int input_dims_data[] = {2, 4, 6};
  int axis_dims_data[] = {1, 1};
  int output_dims_data[] = {1, 4};
  const int32_t axis[] = {-1};
  const int input_shape[] = {4, 6};
  const int output_shape[] = {4};

  float output[4] = {};
  float expected[4] = {};
  tflite::testing::ReferenceMean(tflite::testing::kInput, input_shape, 2,
                                 expected, output_shape, 1, axis, 1,
                                 /*keep_dims=*/false);
  tflite::testing::RunReduce(tflite::Register_MEAN(), input_dims_data,
                             tflite::testing::kInput, axis_dims_data, axis,
                             output_dims_data, output, kTfLiteFloat32,
                             /*keep_dims=*/false);

  tflite::testing::ExpectNear(expected, output, 4,
                              tflite::testing::kFloat32Tolerance);
}

// Rank 5: no NHWC spelling, so this must come back from reference_ops::Mean.
TEST(HeliaFloatReduceTest, MeanFloat32Rank5FallsBackToReference) {
  int input_dims_data[] = {5, 1, 2, 2, 3, 2};
  int axis_dims_data[] = {1, 1};
  int output_dims_data[] = {4, 1, 2, 2, 3};
  const int32_t axis[] = {4};
  const int input_shape[] = {1, 2, 2, 3, 2};
  const int output_shape[] = {1, 2, 2, 3};

  float output[12] = {};
  float expected[12] = {};
  tflite::testing::ReferenceMean(tflite::testing::kInput, input_shape, 5,
                                 expected, output_shape, 4, axis, 1,
                                 /*keep_dims=*/false);
  tflite::testing::RunReduce(tflite::Register_MEAN(), input_dims_data,
                             tflite::testing::kInput, axis_dims_data, axis,
                             output_dims_data, output, kTfLiteFloat32,
                             /*keep_dims=*/false);

  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(expected[i], output[i]);
  }
}

TEST(HeliaFloatReduceTest, SumFloat32Rank4SpatialAxes) {
  int input_dims_data[] = {4, 2, 2, 3, 2};
  int axis_dims_data[] = {1, 2};
  int output_dims_data[] = {4, 2, 1, 1, 2};
  const int32_t axis[] = {1, 2};
  const int input_shape[] = {2, 2, 3, 2};
  const int output_shape[] = {2, 1, 1, 2};

  float output[4] = {};
  float expected[4] = {};
  tflite::testing::ReferenceSum(tflite::testing::kInput, input_shape, 4,
                                expected, output_shape, 4, axis, 2,
                                /*keep_dims=*/true);
  tflite::testing::RunReduce(tflite::Register_SUM(), input_dims_data,
                             tflite::testing::kInput, axis_dims_data, axis,
                             output_dims_data, output, kTfLiteFloat32,
                             /*keep_dims=*/true);

  tflite::testing::ExpectNear(expected, output, 4,
                              tflite::testing::kFloat32Tolerance);
}

// Reduction over the outermost axis only: strided, not a contiguous suffix.
TEST(HeliaFloatReduceTest, SumFloat32Rank4BatchAxis) {
  int input_dims_data[] = {4, 2, 2, 3, 2};
  int axis_dims_data[] = {1, 1};
  int output_dims_data[] = {3, 2, 3, 2};
  const int32_t axis[] = {0};
  const int input_shape[] = {2, 2, 3, 2};
  const int output_shape[] = {2, 3, 2};

  float output[12] = {};
  float expected[12] = {};
  tflite::testing::ReferenceSum(tflite::testing::kInput, input_shape, 4,
                                expected, output_shape, 3, axis, 1,
                                /*keep_dims=*/false);
  tflite::testing::RunReduce(tflite::Register_SUM(), input_dims_data,
                             tflite::testing::kInput, axis_dims_data, axis,
                             output_dims_data, output, kTfLiteFloat32,
                             /*keep_dims=*/false);

  tflite::testing::ExpectNear(expected, output, 12,
                              tflite::testing::kFloat32Tolerance);
}

TEST(HeliaFloatReduceTest, SumFloat32Rank5FallsBackToReference) {
  int input_dims_data[] = {5, 1, 2, 2, 3, 2};
  int axis_dims_data[] = {1, 1};
  int output_dims_data[] = {4, 1, 2, 2, 3};
  const int32_t axis[] = {4};
  const int input_shape[] = {1, 2, 2, 3, 2};
  const int output_shape[] = {1, 2, 2, 3};

  float output[12] = {};
  float expected[12] = {};
  tflite::testing::ReferenceSum(tflite::testing::kInput, input_shape, 5,
                                expected, output_shape, 4, axis, 1,
                                /*keep_dims=*/false);
  tflite::testing::RunReduce(tflite::Register_SUM(), input_dims_data,
                             tflite::testing::kInput, axis_dims_data, axis,
                             output_dims_data, output, kTfLiteFloat32,
                             /*keep_dims=*/false);

  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(expected[i], output[i]);
  }
}

#if ARM_NN_ENABLE_F16
TEST(HeliaFloatReduceTest, MeanFloat16Rank4SpatialAxes) {
  int input_dims_data[] = {4, 1, 2, 3, 2};
  int axis_dims_data[] = {1, 2};
  int output_dims_data[] = {4, 1, 1, 1, 2};
  const int32_t axis[] = {1, 2};

  float16_t input[tflite::testing::kF16Elements] = {};
  for (int i = 0; i < tflite::testing::kF16Elements; ++i) {
    input[i] = static_cast<float16_t>(tflite::testing::kF16Input[i]);
  }
  float16_t output[2] = {};

  tflite::testing::RunReduce(tflite::Register_MEAN(), input_dims_data, input,
                             axis_dims_data, axis, output_dims_data, output,
                             kTfLiteFloat16, /*keep_dims=*/true);

  // Reference expression of reference_ops::Mean over the narrowed input,
  // evaluated in float32 as the kernel contract specifies.
  for (int c = 0; c < 2; ++c) {
    float sum = 0.0f;
    for (int i = 0; i < 6; ++i) {
      sum += static_cast<float>(input[i * 2 + c]);
    }
    EXPECT_NEAR(sum / 6.0f, static_cast<float>(output[c]),
                tflite::testing::kFloat16Tolerance);
  }
}

TEST(HeliaFloatReduceTest, SumFloat16Rank4SpatialAxes) {
  int input_dims_data[] = {4, 1, 2, 3, 2};
  int axis_dims_data[] = {1, 2};
  int output_dims_data[] = {4, 1, 1, 1, 2};
  const int32_t axis[] = {1, 2};

  float16_t input[tflite::testing::kF16Elements] = {};
  for (int i = 0; i < tflite::testing::kF16Elements; ++i) {
    input[i] = static_cast<float16_t>(tflite::testing::kF16Input[i]);
  }
  float16_t output[2] = {};

  tflite::testing::RunReduce(tflite::Register_SUM(), input_dims_data, input,
                             axis_dims_data, axis, output_dims_data, output,
                             kTfLiteFloat16, /*keep_dims=*/true);

  for (int c = 0; c < 2; ++c) {
    float sum = 0.0f;
    for (int i = 0; i < 6; ++i) {
      sum += static_cast<float>(input[i * 2 + c]);
    }
    EXPECT_NEAR(sum, static_cast<float>(output[c]),
                tflite::testing::kFloat16Tolerance);
  }
}
#elif defined(__ARM_FEATURE_MVE) && ((__ARM_FEATURE_MVE) & 2)

// helia.inc defines ARM_NN_ENABLE_F16 for TARGET_ARCH=cortex-m55 only, and a
// silent compile-out would drop these cases while the leg still reported
// success. see AmbiqAI/helia-rt#231
TEST(HeliaFloatReduceTest, Float16CoverageMustNotSilentlyDisappear) {
  FAIL(
      "ARM_NN_ENABLE_F16 is not defined on a build with MVE floating point. "
      "The float16 MEAN and SUM coverage silently compiled out.");
}

#endif  // ARM_NN_ENABLE_F16

TF_LITE_MICRO_TESTS_MAIN
