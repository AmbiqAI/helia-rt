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

// HELIA-only adapter coverage for GATHER and GATHER_ND. The wrappers used by
// GNU links prove whether a case called heliaCORE or stayed on the established
// LiteRT Micro reference path.

#include <cstdint>
#include <cstring>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

enum class Route {
  kGatherF32,
  kGatherF16,
  kGatherS8,
  kGatherS16,
  kGatherNdF32,
  kGatherNdF16,
  kGatherNdS8,
  kGatherNdS16,
  kCount,
};

#if HELIA_GATHER_LINK_WRAP
int g_calls[static_cast<int>(Route::kCount)] = {};
Route g_forced_error = Route::kCount;

bool Fail(Route route) {
  ++g_calls[static_cast<int>(route)];
  return g_forced_error == route;
}

extern "C" {

#if ARM_NN_ENABLE_F32
arm_cmsis_nn_status __real_arm_gather_f32(const float32_t*,
                                          const cmsis_nn_dims*, const int32_t*,
                                          const cmsis_nn_dims*,
                                          const cmsis_nn_gather_params*,
                                          float32_t*, const cmsis_nn_dims*);
arm_cmsis_nn_status __wrap_arm_gather_f32(const float32_t* input,
                                          const cmsis_nn_dims* input_dims,
                                          const int32_t* indices,
                                          const cmsis_nn_dims* indices_dims,
                                          const cmsis_nn_gather_params* params,
                                          float32_t* output,
                                          const cmsis_nn_dims* output_dims) {
  if (Fail(Route::kGatherF32)) return ARM_CMSIS_NN_ARG_ERROR;
  return __real_arm_gather_f32(input, input_dims, indices, indices_dims, params,
                               output, output_dims);
}

arm_cmsis_nn_status __real_arm_gather_nd_f32(const float32_t*,
                                             const cmsis_nn_dims*,
                                             const int32_t*,
                                             const cmsis_nn_dims*,
                                             const cmsis_nn_gather_nd_params*,
                                             float32_t*, const cmsis_nn_dims*);
arm_cmsis_nn_status __wrap_arm_gather_nd_f32(
    const float32_t* input, const cmsis_nn_dims* input_dims,
    const int32_t* indices, const cmsis_nn_dims* indices_dims,
    const cmsis_nn_gather_nd_params* params, float32_t* output,
    const cmsis_nn_dims* output_dims) {
  if (Fail(Route::kGatherNdF32)) return ARM_CMSIS_NN_ARG_ERROR;
  return __real_arm_gather_nd_f32(input, input_dims, indices, indices_dims,
                                  params, output, output_dims);
}
#endif

#if ARM_NN_ENABLE_F16
arm_cmsis_nn_status __real_arm_gather_f16(const float16_t*,
                                          const cmsis_nn_dims*, const int32_t*,
                                          const cmsis_nn_dims*,
                                          const cmsis_nn_gather_params*,
                                          float16_t*, const cmsis_nn_dims*);
arm_cmsis_nn_status __wrap_arm_gather_f16(const float16_t* input,
                                          const cmsis_nn_dims* input_dims,
                                          const int32_t* indices,
                                          const cmsis_nn_dims* indices_dims,
                                          const cmsis_nn_gather_params* params,
                                          float16_t* output,
                                          const cmsis_nn_dims* output_dims) {
  if (Fail(Route::kGatherF16)) return ARM_CMSIS_NN_ARG_ERROR;
  return __real_arm_gather_f16(input, input_dims, indices, indices_dims, params,
                               output, output_dims);
}

arm_cmsis_nn_status __real_arm_gather_nd_f16(const float16_t*,
                                             const cmsis_nn_dims*,
                                             const int32_t*,
                                             const cmsis_nn_dims*,
                                             const cmsis_nn_gather_nd_params*,
                                             float16_t*, const cmsis_nn_dims*);
arm_cmsis_nn_status __wrap_arm_gather_nd_f16(
    const float16_t* input, const cmsis_nn_dims* input_dims,
    const int32_t* indices, const cmsis_nn_dims* indices_dims,
    const cmsis_nn_gather_nd_params* params, float16_t* output,
    const cmsis_nn_dims* output_dims) {
  if (Fail(Route::kGatherNdF16)) return ARM_CMSIS_NN_ARG_ERROR;
  return __real_arm_gather_nd_f16(input, input_dims, indices, indices_dims,
                                  params, output, output_dims);
}
#endif

arm_cmsis_nn_status __real_arm_gather_s8(const int8_t*, const cmsis_nn_dims*,
                                         const int32_t*, const cmsis_nn_dims*,
                                         const cmsis_nn_gather_params*, int8_t*,
                                         const cmsis_nn_dims*);
arm_cmsis_nn_status __wrap_arm_gather_s8(const int8_t* input,
                                         const cmsis_nn_dims* input_dims,
                                         const int32_t* indices,
                                         const cmsis_nn_dims* indices_dims,
                                         const cmsis_nn_gather_params* params,
                                         int8_t* output,
                                         const cmsis_nn_dims* output_dims) {
  if (Fail(Route::kGatherS8)) return ARM_CMSIS_NN_ARG_ERROR;
  return __real_arm_gather_s8(input, input_dims, indices, indices_dims, params,
                              output, output_dims);
}

arm_cmsis_nn_status __real_arm_gather_s16(const int16_t*, const cmsis_nn_dims*,
                                          const int32_t*, const cmsis_nn_dims*,
                                          const cmsis_nn_gather_params*,
                                          int16_t*, const cmsis_nn_dims*);
arm_cmsis_nn_status __wrap_arm_gather_s16(const int16_t* input,
                                          const cmsis_nn_dims* input_dims,
                                          const int32_t* indices,
                                          const cmsis_nn_dims* indices_dims,
                                          const cmsis_nn_gather_params* params,
                                          int16_t* output,
                                          const cmsis_nn_dims* output_dims) {
  if (Fail(Route::kGatherS16)) return ARM_CMSIS_NN_ARG_ERROR;
  return __real_arm_gather_s16(input, input_dims, indices, indices_dims, params,
                               output, output_dims);
}

arm_cmsis_nn_status __real_arm_gather_nd_s8(const int8_t*, const cmsis_nn_dims*,
                                            const int32_t*,
                                            const cmsis_nn_dims*,
                                            const cmsis_nn_gather_nd_params*,
                                            int8_t*, const cmsis_nn_dims*);
arm_cmsis_nn_status __wrap_arm_gather_nd_s8(
    const int8_t* input, const cmsis_nn_dims* input_dims,
    const int32_t* indices, const cmsis_nn_dims* indices_dims,
    const cmsis_nn_gather_nd_params* params, int8_t* output,
    const cmsis_nn_dims* output_dims) {
  if (Fail(Route::kGatherNdS8)) return ARM_CMSIS_NN_ARG_ERROR;
  return __real_arm_gather_nd_s8(input, input_dims, indices, indices_dims,
                                 params, output, output_dims);
}

arm_cmsis_nn_status __real_arm_gather_nd_s16(
    const int16_t*, const cmsis_nn_dims*, const int32_t*, const cmsis_nn_dims*,
    const cmsis_nn_gather_nd_params*, int16_t*, const cmsis_nn_dims*);
arm_cmsis_nn_status __wrap_arm_gather_nd_s16(
    const int16_t* input, const cmsis_nn_dims* input_dims,
    const int32_t* indices, const cmsis_nn_dims* indices_dims,
    const cmsis_nn_gather_nd_params* params, int16_t* output,
    const cmsis_nn_dims* output_dims) {
  if (Fail(Route::kGatherNdS16)) return ARM_CMSIS_NN_ARG_ERROR;
  return __real_arm_gather_nd_s16(input, input_dims, indices, indices_dims,
                                  params, output, output_dims);
}

}  // extern "C"
#endif  // HELIA_GATHER_LINK_WRAP

void ResetRoutes() {
#if HELIA_GATHER_LINK_WRAP
  memset(g_calls, 0, sizeof(g_calls));
  g_forced_error = Route::kCount;
#endif
}

void ExpectCalls(Route route, int expected) {
#if HELIA_GATHER_LINK_WRAP
  EXPECT_EQ(expected, g_calls[static_cast<int>(route)]);
#else
  (void)route;
  (void)expected;
#endif
}

void ForceError(Route route) {
#if HELIA_GATHER_LINK_WRAP
  g_forced_error = route;
#else
  (void)route;
#endif
}

template <typename T>
TfLiteStatus RunGather(const T* input, int* input_dims, const int32_t* indices,
                       int* indices_dims, T* output, int* output_dims,
                       TfLiteType type, int axis = 0, int batch_dims = 0,
                       TfLiteStatus* prepare_status = nullptr,
                       TfLiteType indices_type = kTfLiteInt32,
                       TfLiteType output_type = kTfLiteNoType) {
  TfLiteTensor tensors[] = {
      tflite::testing::CreateTensor(
          input, tflite::testing::IntArrayFromInts(input_dims), false, type),
      tflite::testing::CreateTensor(
          indices, tflite::testing::IntArrayFromInts(indices_dims), false,
          indices_type),
      tflite::testing::CreateTensor(
          output, tflite::testing::IntArrayFromInts(output_dims), false,
          output_type == kTfLiteNoType ? type : output_type),
  };
  int inputs_data[] = {2, 0, 1};
  int outputs_data[] = {1, 2};
  TfLiteGatherParams params = {axis, batch_dims};
  TfLiteIntArray* declared_output_dims = tensors[2].dims;
  const TFLMRegistration registration = tflite::Register_GATHER();
  tflite::micro::KernelRunner runner(
      registration, tensors, 3, tflite::testing::IntArrayFromInts(inputs_data),
      tflite::testing::IntArrayFromInts(outputs_data), &params);
  const TfLiteStatus prepared = runner.InitAndPrepare();
  if (prepare_status != nullptr) *prepare_status = prepared;
  if (prepared != kTfLiteOk) {
    EXPECT_EQ(declared_output_dims, tensors[2].dims);
    return kTfLiteError;
  }

  int normalized_axis = axis;
  if (normalized_axis < 0) normalized_axis += input_dims[0];
  int normalized_batch_dims = batch_dims;
  if (normalized_batch_dims < 0) normalized_batch_dims += indices_dims[0];
  const int expected_rank =
      input_dims[0] + indices_dims[0] - normalized_batch_dims - 1;
  EXPECT_EQ(expected_rank, tensors[2].dims->size);
  if (tensors[2].dims->size == expected_rank) {
    int output_index = 0;
    for (int i = 0; i < normalized_axis; ++i) {
      EXPECT_EQ(input_dims[i + 1], tensors[2].dims->data[output_index++]);
    }
    for (int i = normalized_batch_dims; i < indices_dims[0]; ++i) {
      EXPECT_EQ(indices_dims[i + 1], tensors[2].dims->data[output_index++]);
    }
    for (int i = normalized_axis + 1; i < input_dims[0]; ++i) {
      EXPECT_EQ(input_dims[i + 1], tensors[2].dims->data[output_index++]);
    }
    EXPECT_EQ(expected_rank, output_index);
  }
  return runner.Invoke();
}

template <typename T>
TfLiteStatus RunGatherNd(const T* input, int* input_dims,
                         const int32_t* indices, int* indices_dims, T* output,
                         int* output_dims, TfLiteType type,
                         TfLiteStatus* prepare_status = nullptr,
                         TfLiteType indices_type = kTfLiteInt32,
                         TfLiteType output_type = kTfLiteNoType) {
  TfLiteTensor tensors[] = {
      tflite::testing::CreateTensor(
          input, tflite::testing::IntArrayFromInts(input_dims), false, type),
      tflite::testing::CreateTensor(
          indices, tflite::testing::IntArrayFromInts(indices_dims), false,
          indices_type),
      tflite::testing::CreateTensor(
          output, tflite::testing::IntArrayFromInts(output_dims), false,
          output_type == kTfLiteNoType ? type : output_type),
  };
  int inputs_data[] = {2, 0, 1};
  int outputs_data[] = {1, 2};
  TfLiteIntArray* declared_output_dims = tensors[2].dims;
  const TFLMRegistration registration = tflite::Register_GATHER_ND();
  tflite::micro::KernelRunner runner(
      registration, tensors, 3, tflite::testing::IntArrayFromInts(inputs_data),
      tflite::testing::IntArrayFromInts(outputs_data), nullptr);
  const TfLiteStatus prepared = runner.InitAndPrepare();
  if (prepare_status != nullptr) *prepare_status = prepared;
  if (prepared != kTfLiteOk) {
    EXPECT_EQ(declared_output_dims, tensors[2].dims);
    return kTfLiteError;
  }

  const int indices_rank = indices_dims[0];
  const int indices_nd = indices_dims[indices_rank];
  const int expected_rank = indices_rank - 1 + input_dims[0] - indices_nd;
  EXPECT_EQ(expected_rank, tensors[2].dims->size);
  if (tensors[2].dims->size == expected_rank) {
    int output_index = 0;
    for (int i = 0; i < indices_rank - 1; ++i) {
      EXPECT_EQ(indices_dims[i + 1], tensors[2].dims->data[output_index++]);
    }
    for (int i = indices_nd; i < input_dims[0]; ++i) {
      EXPECT_EQ(input_dims[i + 1], tensors[2].dims->data[output_index++]);
    }
    EXPECT_EQ(expected_rank, output_index);
  }
  return runner.Invoke();
}

template <typename T>
void ExpectArray(const T* expected, const T* actual, int count) {
  for (int i = 0; i < count; ++i) EXPECT_EQ(expected[i], actual[i]);
}

template <typename T>
void ExpectBits(const T* expected, const T* actual, int count) {
  for (int i = 0; i < count; ++i) {
    uint32_t expected_bits = 0;
    uint32_t actual_bits = 0;
    memcpy(&expected_bits, &expected[i], sizeof(T));
    memcpy(&actual_bits, &actual[i], sizeof(T));
    EXPECT_EQ(expected_bits, actual_bits);
  }
}

}  // namespace

TEST(HeliaGatherPairTest, Float32GatherShapesAndBits) {
  ResetRoutes();
  const uint32_t input_bits[] = {0x00000000, 0x80000000, 0x7f800000,
                                 0xff800000, 0x00000001, 0x7fc12345};
  float input[6];
  memcpy(input, input_bits, sizeof(input));

  int matrix[] = {2, 2, 3};
  int vector[] = {1, 2};
  int32_t indices[] = {2, 0};
  int output_dims[] = {2, 0, 0};
  float output[4] = {};
  EXPECT_EQ(kTfLiteOk, RunGather(input, matrix, indices, vector, output,
                                 output_dims, kTfLiteFloat32, /*axis=*/1));
  const float expected[] = {input[2], input[0], input[5], input[3]};
  ExpectBits(expected, output, 4);
  ExpectCalls(Route::kGatherF32, 1);

  int scalar_indices[] = {0};
  int scalar_output[] = {1, 0};
  int input_vector[] = {1, 6};
  int32_t scalar_index[] = {4};
  float scalar = 0;
  EXPECT_EQ(kTfLiteOk,
            RunGather(input, input_vector, scalar_index, scalar_indices,
                      &scalar, scalar_output, kTfLiteFloat32));
  ExpectBits(&input[4], &scalar, 1);
  ExpectCalls(Route::kGatherF32, 2);

  int batched_input_dims[] = {3, 2, 3, 1};
  int batched_indices_dims[] = {2, 2, 2};
  int batched_output_dims[] = {3, 0, 0, 0};
  int32_t batched_indices[] = {2, 0, 1, 2};
  float batched_output[4] = {};
  EXPECT_EQ(kTfLiteOk,
            RunGather(input, batched_input_dims, batched_indices,
                      batched_indices_dims, batched_output, batched_output_dims,
                      kTfLiteFloat32, /*axis=*/1,
                      /*batch_dims=*/1));
  const float batched_expected[] = {input[2], input[0], input[4], input[5]};
  ExpectBits(batched_expected, batched_output, 4);
  ExpectCalls(Route::kGatherF32, 3);

  float negative_output[4] = {};
  int negative_output_dims[] = {3, 0, 0, 0};
  EXPECT_EQ(kTfLiteOk,
            RunGather(input, batched_input_dims, batched_indices,
                      batched_indices_dims, negative_output,
                      negative_output_dims, kTfLiteFloat32, /*axis=*/-2,
                      /*batch_dims=*/-1));
  ExpectBits(batched_expected, negative_output, 4);
  ExpectCalls(Route::kGatherF32, 4);
}

TEST(HeliaGatherPairTest, Float32GatherNdShapesAndScalar) {
  ResetRoutes();
  const uint32_t input_bits[] = {0x80000000, 0x00000001, 0x7fc12345,
                                 0x7f800000, 0xff800000, 0x3f800000};
  float input[6];
  memcpy(input, input_bits, sizeof(input));

  int input_dims[] = {2, 2, 3};
  int slice_indices_dims[] = {2, 2, 1};
  int32_t slice_indices[] = {1, 0};
  int slice_output_dims[] = {2, 0, 0};
  float slice_output[6] = {};
  EXPECT_EQ(kTfLiteOk,
            RunGatherNd(input, input_dims, slice_indices, slice_indices_dims,
                        slice_output, slice_output_dims, kTfLiteFloat32));
  const float slice_expected[] = {input[3], input[4], input[5],
                                  input[0], input[1], input[2]};
  ExpectBits(slice_expected, slice_output, 6);
  ExpectCalls(Route::kGatherNdF32, 1);

  int scalar_indices_dims[] = {1, 2};
  int32_t scalar_indices[] = {1, 1};
  int scalar_output_dims[] = {1, 0};
  float scalar_output = 0;
  EXPECT_EQ(kTfLiteOk,
            RunGatherNd(input, input_dims, scalar_indices, scalar_indices_dims,
                        &scalar_output, scalar_output_dims, kTfLiteFloat32));
  ExpectBits(&input[4], &scalar_output, 1);
  ExpectCalls(Route::kGatherNdF32, 2);

  int batched_indices_dims[] = {3, 2, 1, 2};
  int32_t batched_indices[] = {0, 2, 1, 0};
  int batched_output_dims[] = {3, 0, 0, 0};
  float batched_output[2] = {};
  EXPECT_EQ(kTfLiteOk, RunGatherNd(input, input_dims, batched_indices,
                                   batched_indices_dims, batched_output,
                                   batched_output_dims, kTfLiteFloat32));
  const float batched_expected[] = {input[2], input[3]};
  ExpectBits(batched_expected, batched_output, 2);
  ExpectCalls(Route::kGatherNdF32, 3);
}

TEST(HeliaGatherPairTest, Float32BoundaryShapeClasses) {
  ResetRoutes();
  float input[120];
  for (int i = 0; i < 120; ++i) input[i] = static_cast<float>(i + 1);

  int gather_input_dims[] = {3, 2, 3, 4};
  int rank_two_indices_dims[] = {2, 2, 2};
  int32_t rank_two_indices[] = {3, 1, 0, 2};
  int rank_four_output_dims[] = {4, 0, 0, 0, 0};
  float rank_four_output[24] = {};
  EXPECT_EQ(kTfLiteOk,
            RunGather(input, gather_input_dims, rank_two_indices,
                      rank_two_indices_dims, rank_four_output,
                      rank_four_output_dims, kTfLiteFloat32, /*axis=*/2));
  float rank_four_expected[24];
  for (int outer = 0; outer < 6; ++outer) {
    for (int i = 0; i < 4; ++i) {
      rank_four_expected[outer * 4 + i] =
          input[outer * 4 + rank_two_indices[i]];
    }
  }
  ExpectBits(rank_four_expected, rank_four_output, 24);

  int middle_indices_dims[] = {1, 2};
  int32_t middle_indices[] = {2, 0};
  int middle_output_dims[] = {3, 0, 0, 0};
  float middle_output[16] = {};
  EXPECT_EQ(
      kTfLiteOk,
      RunGather(input, gather_input_dims, middle_indices, middle_indices_dims,
                middle_output, middle_output_dims, kTfLiteFloat32, /*axis=*/1));
  float middle_expected[16];
  for (int batch = 0; batch < 2; ++batch) {
    for (int i = 0; i < 2; ++i) {
      for (int inner = 0; inner < 4; ++inner) {
        middle_expected[(batch * 2 + i) * 4 + inner] =
            input[(batch * 3 + middle_indices[i]) * 4 + inner];
      }
    }
  }
  ExpectBits(middle_expected, middle_output, 16);
  ExpectCalls(Route::kGatherF32, 2);

  int nd_input_dims[] = {4, 3, 2, 2, 2};
  int width_one_indices_dims[] = {2, 2, 1};
  int32_t width_one_indices[] = {2, 0};
  int width_one_output_dims[] = {4, 0, 0, 0, 0};
  float width_one_output[16] = {};
  EXPECT_EQ(kTfLiteOk, RunGatherNd(input, nd_input_dims, width_one_indices,
                                   width_one_indices_dims, width_one_output,
                                   width_one_output_dims, kTfLiteFloat32));
  float width_one_expected[16];
  for (int i = 0; i < 8; ++i) {
    width_one_expected[i] = input[16 + i];
    width_one_expected[8 + i] = input[i];
  }
  ExpectBits(width_one_expected, width_one_output, 16);

  int nd_rank_four_input_dims[] = {4, 2, 3, 4, 5};
  int nd_rank_four_indices_dims[] = {3, 2, 2, 2};
  int32_t nd_rank_four_indices[] = {1, 2, 0, 1, 1, 0, 0, 2};
  int nd_rank_four_output_dims[] = {4, 0, 0, 0, 0};
  float nd_rank_four_output[80] = {};
  EXPECT_EQ(kTfLiteOk,
            RunGatherNd(input, nd_rank_four_input_dims, nd_rank_four_indices,
                        nd_rank_four_indices_dims, nd_rank_four_output,
                        nd_rank_four_output_dims, kTfLiteFloat32));
  float nd_rank_four_expected[80];
  for (int coordinate = 0; coordinate < 4; ++coordinate) {
    const int row = nd_rank_four_indices[2 * coordinate];
    const int column = nd_rank_four_indices[2 * coordinate + 1];
    const int source = (row * 3 + column) * 20;
    for (int i = 0; i < 20; ++i) {
      nd_rank_four_expected[coordinate * 20 + i] = input[source + i];
    }
  }
  ExpectBits(nd_rank_four_expected, nd_rank_four_output, 80);
  ExpectCalls(Route::kGatherNdF32, 2);
}

TEST(HeliaGatherPairTest, IntegerNativeAndFallbackRoutes) {
  ResetRoutes();
  int input_dims[] = {2, 2, 3};
  int indices_dims[] = {1, 2};
  int32_t indices[] = {2, 0};
  int output_dims[] = {2, 0, 0};

  const int8_t input_s8[] = {1, 2, 3, 4, 5, 6};
  int8_t output_s8[4] = {};
  EXPECT_EQ(kTfLiteOk,
            RunGather(input_s8, input_dims, indices, indices_dims, output_s8,
                      output_dims, kTfLiteInt8, /*axis=*/1));
  const int8_t expected_s8[] = {3, 1, 6, 4};
  ExpectArray(expected_s8, output_s8, 4);
  ExpectCalls(Route::kGatherS8, 1);

  const int16_t input_s16[] = {10, 20, 30, 40, 50, 60};
  int16_t output_s16[4] = {};
  int output_s16_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteOk,
            RunGather(input_s16, input_dims, indices, indices_dims, output_s16,
                      output_s16_dims, kTfLiteInt16, /*axis=*/1));
  const int16_t expected_s16[] = {30, 10, 60, 40};
  ExpectArray(expected_s16, output_s16, 4);
  ExpectCalls(Route::kGatherS16, 1);

  int scalar_dims[] = {0};
  int vector_dims[] = {1, 6};
  int scalar_output_dims[] = {1, 0};
  int32_t scalar_index[] = {3};
  int8_t scalar_output = 0;
  EXPECT_EQ(kTfLiteOk,
            RunGather(input_s8, vector_dims, scalar_index, scalar_dims,
                      &scalar_output, scalar_output_dims, kTfLiteInt8));
  EXPECT_EQ(input_s8[3], scalar_output);
  ExpectCalls(Route::kGatherS8, 1);

  TfLiteStatus prepared = kTfLiteOk;
  int16_t rejected_output = 0;
  int rejected_output_dims[] = {1, 0};
  EXPECT_EQ(kTfLiteError,
            RunGather(input_s16, vector_dims, scalar_index, scalar_dims,
                      &rejected_output, rejected_output_dims, kTfLiteInt16, 0,
                      0, &prepared));
  EXPECT_EQ(kTfLiteError, prepared);
  ExpectCalls(Route::kGatherS16, 1);

  int nd_indices_dims[] = {2, 2, 1};
  int32_t nd_indices[] = {1, 0};
  int nd_output_dims[] = {2, 0, 0};
  int8_t nd_output_s8[6] = {};
  EXPECT_EQ(kTfLiteOk,
            RunGatherNd(input_s8, input_dims, nd_indices, nd_indices_dims,
                        nd_output_s8, nd_output_dims, kTfLiteInt8));
  const int8_t nd_expected_s8[] = {4, 5, 6, 1, 2, 3};
  ExpectArray(nd_expected_s8, nd_output_s8, 6);
  ExpectCalls(Route::kGatherNdS8, 1);

  int16_t nd_output_s16[6] = {};
  int nd_output_s16_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteOk,
            RunGatherNd(input_s16, input_dims, nd_indices, nd_indices_dims,
                        nd_output_s16, nd_output_s16_dims, kTfLiteInt16));
  const int16_t nd_expected_s16[] = {40, 50, 60, 10, 20, 30};
  ExpectArray(nd_expected_s16, nd_output_s16, 6);
  ExpectCalls(Route::kGatherNdS16, 1);
}

TEST(HeliaGatherPairTest, HighRankAndZeroTupleStayReference) {
  ResetRoutes();
  int high_input_dims[] = {5, 1, 1, 1, 1, 4};
  int indices_dims[] = {1, 2};
  int32_t indices[] = {3, 1};
  int high_output_dims[] = {5, 0, 0, 0, 0, 0};
  const float input[] = {1, 2, 3, 4};
  float output[2] = {};
  EXPECT_EQ(kTfLiteOk,
            RunGather(input, high_input_dims, indices, indices_dims, output,
                      high_output_dims, kTfLiteFloat32, /*axis=*/4));
  const float expected[] = {4, 2};
  ExpectBits(expected, output, 2);
  ExpectCalls(Route::kGatherF32, 0);

  int32_t invalid_indices[] = {3, 4};
  float invalid_output[] = {55, 66};
  int invalid_output_dims[] = {5, 0, 0, 0, 0, 0};
  EXPECT_EQ(kTfLiteError,
            RunGather(input, high_input_dims, invalid_indices, indices_dims,
                      invalid_output, invalid_output_dims, kTfLiteFloat32,
                      /*axis=*/4));
  const float invalid_sentinel[] = {55, 66};
  ExpectBits(invalid_sentinel, invalid_output, 2);
  ExpectCalls(Route::kGatherF32, 0);

  int8_t input_s8[] = {1, 2, 3, 4};
  int8_t output_s8[2] = {};
  int high_output_s8_dims[] = {5, 0, 0, 0, 0, 0};
  EXPECT_EQ(kTfLiteOk,
            RunGather(input_s8, high_input_dims, indices, indices_dims,
                      output_s8, high_output_s8_dims, kTfLiteInt8,
                      /*axis=*/4));
  const int8_t expected_s8[] = {4, 2};
  ExpectArray(expected_s8, output_s8, 2);
  ExpectCalls(Route::kGatherS8, 0);

  int8_t invalid_output_s8[] = {55, 66};
  int invalid_output_s8_dims[] = {5, 0, 0, 0, 0, 0};
  EXPECT_EQ(kTfLiteError,
            RunGather(input_s8, high_input_dims, invalid_indices, indices_dims,
                      invalid_output_s8, invalid_output_s8_dims, kTfLiteInt8,
                      /*axis=*/4));
  const int8_t invalid_sentinel_s8[] = {55, 66};
  ExpectArray(invalid_sentinel_s8, invalid_output_s8, 2);
  ExpectCalls(Route::kGatherS8, 0);

  int high_nd_indices_dims[] = {2, 2, 1};
  int32_t high_nd_indices[] = {0, 0};
  int high_nd_output_dims[] = {5, 0, 0, 0, 0, 0};
  float high_nd_output[8] = {};
  EXPECT_EQ(kTfLiteOk, RunGatherNd(input, high_input_dims, high_nd_indices,
                                   high_nd_indices_dims, high_nd_output,
                                   high_nd_output_dims, kTfLiteFloat32));
  const float high_nd_expected[] = {1, 2, 3, 4, 1, 2, 3, 4};
  ExpectBits(high_nd_expected, high_nd_output, 8);
  ExpectCalls(Route::kGatherNdF32, 0);

  int32_t high_nd_invalid_indices[] = {0, 1};
  int high_nd_invalid_output_dims[] = {5, 0, 0, 0, 0, 0};
  float high_nd_invalid_output[8];
  for (float& value : high_nd_invalid_output) value = 77;
  EXPECT_EQ(kTfLiteError,
            RunGatherNd(input, high_input_dims, high_nd_invalid_indices,
                        high_nd_indices_dims, high_nd_invalid_output,
                        high_nd_invalid_output_dims, kTfLiteFloat32));
  const float high_nd_sentinel[] = {77, 77, 77, 77, 77, 77, 77, 77};
  ExpectBits(high_nd_sentinel, high_nd_invalid_output, 8);
  ExpectCalls(Route::kGatherNdF32, 0);

  int8_t high_nd_output_s8[8] = {};
  int high_nd_output_s8_dims[] = {5, 0, 0, 0, 0, 0};
  EXPECT_EQ(kTfLiteOk, RunGatherNd(input_s8, high_input_dims, high_nd_indices,
                                   high_nd_indices_dims, high_nd_output_s8,
                                   high_nd_output_s8_dims, kTfLiteInt8));
  const int8_t high_nd_expected_s8[] = {1, 2, 3, 4, 1, 2, 3, 4};
  ExpectArray(high_nd_expected_s8, high_nd_output_s8, 8);
  ExpectCalls(Route::kGatherNdS8, 0);

  int8_t high_nd_invalid_output_s8[8];
  for (int8_t& value : high_nd_invalid_output_s8) value = 77;
  int high_nd_invalid_output_s8_dims[] = {5, 0, 0, 0, 0, 0};
  EXPECT_EQ(kTfLiteError,
            RunGatherNd(input_s8, high_input_dims, high_nd_invalid_indices,
                        high_nd_indices_dims, high_nd_invalid_output_s8,
                        high_nd_invalid_output_s8_dims, kTfLiteInt8));
  const int8_t high_nd_sentinel_s8[] = {77, 77, 77, 77, 77, 77, 77, 77};
  ExpectArray(high_nd_sentinel_s8, high_nd_invalid_output_s8, 8);
  ExpectCalls(Route::kGatherNdS8, 0);

  int nd_input_dims[] = {2, 2, 2};
  int zero_tuple_dims[] = {2, 3, 0};
  int32_t unused_index = 0;
  int zero_output_dims[] = {3, 0, 0, 0};
  const float nd_input[] = {1, 2, 3, 4};
  float zero_output[12] = {};
  EXPECT_EQ(kTfLiteOk,
            RunGatherNd(nd_input, nd_input_dims, &unused_index, zero_tuple_dims,
                        zero_output, zero_output_dims, kTfLiteFloat32));
  const float zero_expected[] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
  ExpectBits(zero_expected, zero_output, 12);
  ExpectCalls(Route::kGatherNdF32, 0);
}

TEST(HeliaGatherPairTest, FloatErrorsAreAtomicAndPropagated) {
  ResetRoutes();
  int input_dims[] = {1, 4};
  int indices_dims[] = {1, 2};
  int output_dims[] = {1, 0};
  const float input[] = {1, 2, 3, 4};
  int32_t invalid_indices[] = {1, 4};
  float output[] = {77, 88};
  EXPECT_EQ(kTfLiteError,
            RunGather(input, input_dims, invalid_indices, indices_dims, output,
                      output_dims, kTfLiteFloat32));
  const float sentinel[] = {77, 88};
  ExpectBits(sentinel, output, 2);
  ExpectCalls(Route::kGatherF32, 1);

  int nd_input_dims[] = {2, 2, 2};
  int nd_indices_dims[] = {2, 2, 2};
  int nd_output_dims[] = {1, 0};
  int32_t dimension_invalid[] = {0, 0, 0, 2};
  float nd_output[] = {99, 88};
  EXPECT_EQ(kTfLiteError, RunGatherNd(input, nd_input_dims, dimension_invalid,
                                      nd_indices_dims, nd_output,
                                      nd_output_dims, kTfLiteFloat32));
  const float nd_sentinel[] = {99, 88};
  ExpectBits(nd_sentinel, nd_output, 2);
  ExpectCalls(Route::kGatherNdF32, 1);

#if HELIA_GATHER_LINK_WRAP
  ResetRoutes();
  ForceError(Route::kGatherF32);
  int32_t valid_indices[] = {0, 1};
  int propagated_output_dims[] = {1, 0};
  float propagated_output[] = {55, 66};
  EXPECT_EQ(kTfLiteError, RunGather(input, input_dims, valid_indices,
                                    indices_dims, propagated_output,
                                    propagated_output_dims, kTfLiteFloat32));
  ExpectCalls(Route::kGatherF32, 1);
  const float propagated_sentinel[] = {55, 66};
  ExpectBits(propagated_sentinel, propagated_output, 2);

  ResetRoutes();
  ForceError(Route::kGatherNdF32);
  int32_t valid_nd_indices[] = {0, 0};
  int valid_nd_indices_dims[] = {1, 2};
  int propagated_nd_output_dims[] = {0};
  float propagated_nd_output = 77;
  EXPECT_EQ(kTfLiteError,
            RunGatherNd(input, nd_input_dims, valid_nd_indices,
                        valid_nd_indices_dims, &propagated_nd_output,
                        propagated_nd_output_dims, kTfLiteFloat32));
  ExpectCalls(Route::kGatherNdF32, 1);
  EXPECT_EQ(77.0f, propagated_nd_output);
#endif
}

TEST(HeliaGatherPairTest, IntegerNativeLateInvalidIndicesAreAtomic) {
  ResetRoutes();
  int input_dims[] = {1, 4};
  int indices_dims[] = {1, 2};
  int output_dims[] = {1, 0};
  int32_t invalid_indices[] = {1, 4};

  const int8_t input_s8[] = {10, 20, 30, 40};
  int8_t output_s8[] = {77, 88};
  EXPECT_EQ(kTfLiteError,
            RunGather(input_s8, input_dims, invalid_indices, indices_dims,
                      output_s8, output_dims, kTfLiteInt8));
  const int8_t sentinel_s8[] = {77, 88};
  ExpectArray(sentinel_s8, output_s8, 2);
  ExpectCalls(Route::kGatherS8, 0);

  const int16_t input_s16[] = {10, 20, 30, 40};
  int16_t output_s16[] = {77, 88};
  int output_s16_dims[] = {1, 0};
  EXPECT_EQ(kTfLiteError,
            RunGather(input_s16, input_dims, invalid_indices, indices_dims,
                      output_s16, output_s16_dims, kTfLiteInt16));
  const int16_t sentinel_s16[] = {77, 88};
  ExpectArray(sentinel_s16, output_s16, 2);
  ExpectCalls(Route::kGatherS16, 0);

  int nd_input_dims[] = {2, 2, 2};
  int nd_indices_dims[] = {2, 2, 2};
  int nd_output_dims[] = {1, 0};
  int32_t invalid_nd_indices[] = {0, 0, 0, 2};

  int8_t nd_output_s8[] = {77, 88};
  EXPECT_EQ(
      kTfLiteError,
      RunGatherNd(input_s8, nd_input_dims, invalid_nd_indices, nd_indices_dims,
                  nd_output_s8, nd_output_dims, kTfLiteInt8));
  ExpectArray(sentinel_s8, nd_output_s8, 2);
  ExpectCalls(Route::kGatherNdS8, 0);

  int16_t nd_output_s16[] = {77, 88};
  int nd_output_s16_dims[] = {1, 0};
  EXPECT_EQ(
      kTfLiteError,
      RunGatherNd(input_s16, nd_input_dims, invalid_nd_indices, nd_indices_dims,
                  nd_output_s16, nd_output_s16_dims, kTfLiteInt16));
  ExpectArray(sentinel_s16, nd_output_s16, 2);
  ExpectCalls(Route::kGatherNdS16, 0);
}

TEST(HeliaGatherPairTest, EmptyAndNullBufferContracts) {
  ResetRoutes();
  int input_dims[] = {2, 0, 2};
  int indices_dims[] = {1, 0};
  int output_dims[] = {2, 0, 0};

  EXPECT_EQ(kTfLiteOk, RunGather(static_cast<const float*>(nullptr), input_dims,
                                 static_cast<const int32_t*>(nullptr),
                                 indices_dims, static_cast<float*>(nullptr),
                                 output_dims, kTfLiteFloat32));
  ExpectCalls(Route::kGatherF32, 0);

  int int8_output_dims[] = {2, 0, 0};
  EXPECT_EQ(
      kTfLiteOk,
      RunGather(static_cast<const int8_t*>(nullptr), input_dims,
                static_cast<const int32_t*>(nullptr), indices_dims,
                static_cast<int8_t*>(nullptr), int8_output_dims, kTfLiteInt8));
  ExpectCalls(Route::kGatherS8, 0);

  int int16_output_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteOk,
            RunGather(static_cast<const int16_t*>(nullptr), input_dims,
                      static_cast<const int32_t*>(nullptr), indices_dims,
                      static_cast<int16_t*>(nullptr), int16_output_dims,
                      kTfLiteInt16));
  ExpectCalls(Route::kGatherS16, 0);

  int nd_indices_dims[] = {2, 0, 1};
  int nd_output_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteOk,
            RunGatherNd(static_cast<const float*>(nullptr), input_dims,
                        static_cast<const int32_t*>(nullptr), nd_indices_dims,
                        static_cast<float*>(nullptr), nd_output_dims,
                        kTfLiteFloat32));
  ExpectCalls(Route::kGatherNdF32, 0);

  int nd_int8_output_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteOk,
            RunGatherNd(static_cast<const int8_t*>(nullptr), input_dims,
                        static_cast<const int32_t*>(nullptr), nd_indices_dims,
                        static_cast<int8_t*>(nullptr), nd_int8_output_dims,
                        kTfLiteInt8));
  ExpectCalls(Route::kGatherNdS8, 0);

  int nd_int16_output_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteOk,
            RunGatherNd(static_cast<const int16_t*>(nullptr), input_dims,
                        static_cast<const int32_t*>(nullptr), nd_indices_dims,
                        static_cast<int16_t*>(nullptr), nd_int16_output_dims,
                        kTfLiteInt16));
  ExpectCalls(Route::kGatherNdS16, 0);

  const float nonempty_input[] = {1, 2};
  int nonempty_input_dims[] = {1, 2};
  int nonempty_indices_dims[] = {1, 1};
  int32_t nonempty_indices[] = {0};
  int nonempty_output_dims[] = {1, 0};
  EXPECT_EQ(kTfLiteError,
            RunGather(nonempty_input, nonempty_input_dims, nonempty_indices,
                      nonempty_indices_dims, static_cast<float*>(nullptr),
                      nonempty_output_dims, kTfLiteFloat32));
  ExpectCalls(Route::kGatherF32, 1);

  int empty_trailing_input_dims[] = {2, 2, 0};
  int invalid_output_dims[] = {2, 0, 0};
  int32_t invalid_index[] = {2};
  EXPECT_EQ(kTfLiteError,
            RunGather(static_cast<const float*>(nullptr),
                      empty_trailing_input_dims, invalid_index,
                      nonempty_indices_dims, static_cast<float*>(nullptr),
                      invalid_output_dims, kTfLiteFloat32));
  ExpectCalls(Route::kGatherF32, 1);

  int32_t valid_index[] = {1};
  int valid_empty_output_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteOk,
            RunGather(static_cast<const float*>(nullptr),
                      empty_trailing_input_dims, valid_index,
                      nonempty_indices_dims, static_cast<float*>(nullptr),
                      valid_empty_output_dims, kTfLiteFloat32));
  ExpectCalls(Route::kGatherF32, 1);

  int valid_empty_int8_output_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteOk,
            RunGather(static_cast<const int8_t*>(nullptr),
                      empty_trailing_input_dims, valid_index,
                      nonempty_indices_dims, static_cast<int8_t*>(nullptr),
                      valid_empty_int8_output_dims, kTfLiteInt8));
  ExpectCalls(Route::kGatherS8, 0);

  int valid_empty_int16_output_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteOk,
            RunGather(static_cast<const int16_t*>(nullptr),
                      empty_trailing_input_dims, valid_index,
                      nonempty_indices_dims, static_cast<int16_t*>(nullptr),
                      valid_empty_int16_output_dims, kTfLiteInt16));
  ExpectCalls(Route::kGatherS16, 0);

#if ARM_NN_ENABLE_F16
  int valid_empty_f16_output_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteOk,
            RunGather(static_cast<const uint16_t*>(nullptr),
                      empty_trailing_input_dims, valid_index,
                      nonempty_indices_dims, static_cast<uint16_t*>(nullptr),
                      valid_empty_f16_output_dims, kTfLiteFloat16));
  ExpectCalls(Route::kGatherF16, 0);
#endif

  int nd_empty_trailing_input_dims[] = {3, 2, 2, 0};
  int nd_nonempty_indices_dims[] = {2, 1, 1};
  int nd_invalid_output_dims[] = {3, 0, 0, 0};
  EXPECT_EQ(kTfLiteError,
            RunGatherNd(static_cast<const int8_t*>(nullptr),
                        nd_empty_trailing_input_dims, invalid_index,
                        nd_nonempty_indices_dims, static_cast<int8_t*>(nullptr),
                        nd_invalid_output_dims, kTfLiteInt8));
  ExpectCalls(Route::kGatherNdS8, 0);

  int nd_valid_int8_output_dims[] = {3, 0, 0, 0};
  EXPECT_EQ(kTfLiteOk,
            RunGatherNd(static_cast<const int8_t*>(nullptr),
                        nd_empty_trailing_input_dims, valid_index,
                        nd_nonempty_indices_dims, static_cast<int8_t*>(nullptr),
                        nd_valid_int8_output_dims, kTfLiteInt8));
  ExpectCalls(Route::kGatherNdS8, 0);

  int nd_valid_int16_output_dims[] = {3, 0, 0, 0};
  EXPECT_EQ(kTfLiteOk, RunGatherNd(static_cast<const int16_t*>(nullptr),
                                   nd_empty_trailing_input_dims, valid_index,
                                   nd_nonempty_indices_dims,
                                   static_cast<int16_t*>(nullptr),
                                   nd_valid_int16_output_dims, kTfLiteInt16));
  ExpectCalls(Route::kGatherNdS16, 0);

  int nd_valid_float_output_dims[] = {3, 0, 0, 0};
  EXPECT_EQ(kTfLiteError,
            RunGatherNd(static_cast<const float*>(nullptr),
                        nd_empty_trailing_input_dims, invalid_index,
                        nd_nonempty_indices_dims, static_cast<float*>(nullptr),
                        nd_valid_float_output_dims, kTfLiteFloat32));
  ExpectCalls(Route::kGatherNdF32, 0);

  int nd_valid_float_output_dims_2[] = {3, 0, 0, 0};
  EXPECT_EQ(kTfLiteOk,
            RunGatherNd(static_cast<const float*>(nullptr),
                        nd_empty_trailing_input_dims, valid_index,
                        nd_nonempty_indices_dims, static_cast<float*>(nullptr),
                        nd_valid_float_output_dims_2, kTfLiteFloat32));
  ExpectCalls(Route::kGatherNdF32, 0);

#if ARM_NN_ENABLE_F16
  int nd_valid_f16_output_dims[] = {3, 0, 0, 0};
  EXPECT_EQ(kTfLiteOk, RunGatherNd(static_cast<const uint16_t*>(nullptr),
                                   nd_empty_trailing_input_dims, valid_index,
                                   nd_nonempty_indices_dims,
                                   static_cast<uint16_t*>(nullptr),
                                   nd_valid_f16_output_dims, kTfLiteFloat16));
  ExpectCalls(Route::kGatherNdF16, 0);
#endif
}

TEST(HeliaGatherPairTest, PrepareRejectsInvalidMetadata) {
  ResetRoutes();
  const float input[] = {1, 2, 3, 4};
  int input_dims[] = {2, 2, 2};
  int indices_dims[] = {1, 2};
  int32_t indices[] = {0, 1};
  float output[4] = {};
  TfLiteStatus prepared = kTfLiteOk;

  int short_output_dims[] = {1, 0};
  EXPECT_EQ(kTfLiteError,
            RunGather(input, input_dims, indices, indices_dims, output,
                      short_output_dims, kTfLiteFloat32, 0, 0, &prepared));
  EXPECT_EQ(kTfLiteError, prepared);

  int negative_width_indices_dims[] = {2, 1, -1};
  int negative_width_output_dims[] = {4, 0, 0, 0, 0};
  EXPECT_EQ(kTfLiteError,
            RunGatherNd(input, input_dims, indices, negative_width_indices_dims,
                        output, negative_width_output_dims, kTfLiteFloat32,
                        &prepared));
  EXPECT_EQ(kTfLiteError, prepared);

  int bad_axis_output_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteError,
            RunGather(input, input_dims, indices, indices_dims, output,
                      bad_axis_output_dims, kTfLiteFloat32, /*axis=*/2,
                      /*batch_dims=*/0, &prepared));
  EXPECT_EQ(kTfLiteError, prepared);

  int bad_batch_output_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteError,
            RunGather(input, input_dims, indices, indices_dims, output,
                      bad_batch_output_dims, kTfLiteFloat32, /*axis=*/1,
                      /*batch_dims=*/2, &prepared));
  EXPECT_EQ(kTfLiteError, prepared);

  int output_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteError, RunGather(input, input_dims, indices, indices_dims,
                                    output, output_dims, kTfLiteFloat32, 0, 0,
                                    &prepared, kTfLiteInt16));
  EXPECT_EQ(kTfLiteError, prepared);

  int mismatch_output_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteError,
            RunGather(input, input_dims, indices, indices_dims, output,
                      mismatch_output_dims, kTfLiteFloat32, 0, 0, &prepared,
                      kTfLiteInt32, kTfLiteInt8));
  EXPECT_EQ(kTfLiteError, prepared);

  int oversized_input_dims[] = {2, 32768, 16384};
  int oversized_output_dims[] = {2, 0, 0};
  int safe_input_dims[] = {2, 1, 1};
  TfLiteTensor oversized_tensors[] = {
      tflite::testing::CreateTensor(
          input, tflite::testing::IntArrayFromInts(safe_input_dims), false,
          kTfLiteFloat32),
      tflite::testing::CreateTensor(
          indices, tflite::testing::IntArrayFromInts(indices_dims)),
      tflite::testing::CreateTensor(
          output, tflite::testing::IntArrayFromInts(oversized_output_dims),
          false, kTfLiteFloat32),
  };
  oversized_tensors[0].dims =
      tflite::testing::IntArrayFromInts(oversized_input_dims);
  int oversized_inputs_data[] = {2, 0, 1};
  int oversized_outputs_data[] = {1, 2};
  TfLiteGatherParams oversized_params = {0, 0};
  const TFLMRegistration oversized_registration = tflite::Register_GATHER();
  tflite::micro::KernelRunner oversized_runner(
      oversized_registration, oversized_tensors, 3,
      tflite::testing::IntArrayFromInts(oversized_inputs_data),
      tflite::testing::IntArrayFromInts(oversized_outputs_data),
      &oversized_params);
  EXPECT_EQ(kTfLiteError, oversized_runner.InitAndPrepare());

  int rank_zero_indices[] = {0};
  int nd_output_dims[] = {1, 0};
  EXPECT_EQ(kTfLiteError,
            RunGatherNd(input, input_dims, indices, rank_zero_indices, output,
                        nd_output_dims, kTfLiteFloat32, &prepared));
  EXPECT_EQ(kTfLiteError, prepared);

  int zero_tuple_dims[] = {2, 2, 0};
  int zero_tuple_output_dims[] = {3, 0, 0, 0};
  const int16_t input_s16[] = {1, 2, 3, 4};
  int16_t output_s16[8] = {};
  EXPECT_EQ(
      kTfLiteError,
      RunGatherNd(input_s16, input_dims, indices, zero_tuple_dims, output_s16,
                  zero_tuple_output_dims, kTfLiteInt16, &prepared));
  EXPECT_EQ(kTfLiteError, prepared);

  int high_rank_input_dims[] = {5, 1, 1, 1, 1, 4};
  int high_rank_output_dims[] = {5, 0, 0, 0, 0, 0};
  EXPECT_EQ(kTfLiteError,
            RunGather(input_s16, high_rank_input_dims, indices, indices_dims,
                      output_s16, high_rank_output_dims, kTfLiteInt16,
                      /*axis=*/4, /*batch_dims=*/0, &prepared));
  EXPECT_EQ(kTfLiteError, prepared);

  int high_rank_nd_indices_dims[] = {2, 1, 1};
  int high_rank_nd_output_dims[] = {5, 0, 0, 0, 0, 0};
  EXPECT_EQ(kTfLiteError,
            RunGatherNd(input_s16, high_rank_input_dims, indices,
                        high_rank_nd_indices_dims, output_s16,
                        high_rank_nd_output_dims, kTfLiteInt16, &prepared));
  EXPECT_EQ(kTfLiteError, prepared);

  const uint16_t input_f16[] = {0x3c00, 0x4000, 0x4200, 0x4400};
  uint16_t output_f16[4] = {};
#if !ARM_NN_ENABLE_F16
  int f16_output_dims[] = {2, 0, 0};
  EXPECT_EQ(kTfLiteError,
            RunGather(input_f16, input_dims, indices, indices_dims, output_f16,
                      f16_output_dims, kTfLiteFloat16, 0, 0, &prepared));
  EXPECT_EQ(kTfLiteError, prepared);

  int f16_nd_output_dims[] = {1, 0};
  EXPECT_EQ(
      kTfLiteError,
      RunGatherNd(input_f16, input_dims, indices, indices_dims, output_f16,
                  f16_nd_output_dims, kTfLiteFloat16, &prepared));
  EXPECT_EQ(kTfLiteError, prepared);
#else
  EXPECT_EQ(kTfLiteError,
            RunGather(input_f16, high_rank_input_dims, indices, indices_dims,
                      output_f16, high_rank_output_dims, kTfLiteFloat16,
                      /*axis=*/4, /*batch_dims=*/0, &prepared));
  EXPECT_EQ(kTfLiteError, prepared);
  EXPECT_EQ(kTfLiteError,
            RunGatherNd(input_f16, high_rank_input_dims, indices,
                        high_rank_nd_indices_dims, output_f16,
                        high_rank_nd_output_dims, kTfLiteFloat16, &prepared));
  EXPECT_EQ(kTfLiteError, prepared);
  ExpectCalls(Route::kGatherF16, 0);
  ExpectCalls(Route::kGatherNdF16, 0);
#endif

  ExpectCalls(Route::kGatherF32, 0);
  ExpectCalls(Route::kGatherNdF32, 0);
}

TEST(HeliaGatherPairTest, EvalRefreshesAllTensorPointers) {
  ResetRoutes();
  float first_input[] = {1, 2, 3, 4};
  float second_input[] = {10, 20, 30, 40};
  int32_t first_indices[] = {0, 1};
  int32_t second_indices[] = {3, 2};
  float first_output[] = {0, 0};
  float second_output[] = {0, 0};
  int input_dims_data[] = {1, 4};
  int indices_dims_data[] = {1, 2};
  int output_dims_data[] = {1, 0};
  TfLiteTensor tensors[] = {
      tflite::testing::CreateTensor(
          first_input, tflite::testing::IntArrayFromInts(input_dims_data),
          false, kTfLiteFloat32),
      tflite::testing::CreateTensor(
          first_indices, tflite::testing::IntArrayFromInts(indices_dims_data)),
      tflite::testing::CreateTensor(
          first_output, tflite::testing::IntArrayFromInts(output_dims_data),
          false, kTfLiteFloat32),
  };
  int inputs_data[] = {2, 0, 1};
  int outputs_data[] = {1, 2};
  TfLiteGatherParams params = {0, 0};
  const TFLMRegistration gather_registration = tflite::Register_GATHER();
  tflite::micro::KernelRunner runner(
      gather_registration, tensors, 3,
      tflite::testing::IntArrayFromInts(inputs_data),
      tflite::testing::IntArrayFromInts(outputs_data), &params);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  const float first_expected[] = {1, 2};
  ExpectBits(first_expected, first_output, 2);

  tensors[0].data.f = second_input;
  tensors[1].data.i32 = second_indices;
  tensors[2].data.f = second_output;
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
  const float second_expected[] = {40, 30};
  ExpectBits(second_expected, second_output, 2);

  int32_t invalid_indices[] = {3, 4};
  float invalid_output[] = {55, 66};
  tensors[1].data.i32 = invalid_indices;
  tensors[2].data.f = invalid_output;
  EXPECT_EQ(kTfLiteError, runner.Invoke());
  const float invalid_sentinel[] = {55, 66};
  ExpectBits(invalid_sentinel, invalid_output, 2);
  ExpectCalls(Route::kGatherF32, 3);

  float first_nd_input[] = {1, 2, 3, 4};
  float second_nd_input[] = {10, 20, 30, 40};
  int32_t first_nd_indices[] = {0, 0};
  int32_t second_nd_indices[] = {1, 1};
  float first_nd_output[] = {0};
  float second_nd_output[] = {0};
  int nd_input_dims_data[] = {2, 2, 2};
  int nd_indices_dims_data[] = {1, 2};
  int nd_output_dims_data[] = {1, 0};
  TfLiteTensor nd_tensors[] = {
      tflite::testing::CreateTensor(
          first_nd_input, tflite::testing::IntArrayFromInts(nd_input_dims_data),
          false, kTfLiteFloat32),
      tflite::testing::CreateTensor(
          first_nd_indices,
          tflite::testing::IntArrayFromInts(nd_indices_dims_data)),
      tflite::testing::CreateTensor(
          first_nd_output,
          tflite::testing::IntArrayFromInts(nd_output_dims_data), false,
          kTfLiteFloat32),
  };
  const TFLMRegistration gather_nd_registration = tflite::Register_GATHER_ND();
  tflite::micro::KernelRunner nd_runner(
      gather_nd_registration, nd_tensors, 3,
      tflite::testing::IntArrayFromInts(inputs_data),
      tflite::testing::IntArrayFromInts(outputs_data), nullptr);
  EXPECT_EQ(kTfLiteOk, nd_runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, nd_runner.Invoke());
  EXPECT_EQ(1.0f, first_nd_output[0]);

  nd_tensors[0].data.f = second_nd_input;
  nd_tensors[1].data.i32 = second_nd_indices;
  nd_tensors[2].data.f = second_nd_output;
  EXPECT_EQ(kTfLiteOk, nd_runner.Invoke());
  EXPECT_EQ(40.0f, second_nd_output[0]);

  int32_t invalid_nd_indices[] = {0, 2};
  float invalid_nd_output = 77;
  nd_tensors[1].data.i32 = invalid_nd_indices;
  nd_tensors[2].data.f = &invalid_nd_output;
  EXPECT_EQ(kTfLiteError, nd_runner.Invoke());
  EXPECT_EQ(77.0f, invalid_nd_output);
  ExpectCalls(Route::kGatherNdF32, 3);
}

#if ARM_NN_ENABLE_F16
TEST(HeliaGatherPairTest, Float16PreservesPayloadBits) {
  ResetRoutes();
  const uint16_t input[] = {0x0000, 0x8000, 0x7c00, 0xfc00,
                            0x0001, 0x03ff, 0x7e01, 0x7d55};
  int input_dims[] = {2, 2, 4};
  int indices_dims[] = {1, 4};
  int32_t indices[] = {3, 0, 2, 1};
  int output_dims[] = {2, 0, 0};
  uint16_t output[8] = {};
  EXPECT_EQ(kTfLiteOk,
            RunGather(input, input_dims, indices, indices_dims, output,
                      output_dims, kTfLiteFloat16, /*axis=*/1));
  const uint16_t expected[] = {0xfc00, 0x0000, 0x7c00, 0x8000,
                               0x7d55, 0x0001, 0x7e01, 0x03ff};
  ExpectBits(expected, output, 8);
  ExpectCalls(Route::kGatherF16, 1);

  int vector_input_dims[] = {1, 8};
  int scalar_indices_dims[] = {0};
  int32_t scalar_index[] = {6};
  int scalar_output_dims[] = {0};
  uint16_t scalar_output = 0;
  EXPECT_EQ(kTfLiteOk, RunGather(input, vector_input_dims, scalar_index,
                                 scalar_indices_dims, &scalar_output,
                                 scalar_output_dims, kTfLiteFloat16));
  ExpectBits(&input[6], &scalar_output, 1);
  ExpectCalls(Route::kGatherF16, 2);

  int nd_indices_dims[] = {2, 2, 2};
  int32_t nd_indices[] = {1, 3, 0, 1};
  int nd_output_dims[] = {1, 0};
  uint16_t nd_output[2] = {};
  EXPECT_EQ(kTfLiteOk,
            RunGatherNd(input, input_dims, nd_indices, nd_indices_dims,
                        nd_output, nd_output_dims, kTfLiteFloat16));
  const uint16_t nd_expected[] = {0x7d55, 0x8000};
  ExpectBits(nd_expected, nd_output, 2);
  ExpectCalls(Route::kGatherNdF16, 1);

  uint16_t outer_input[12];
  for (int i = 0; i < 12; ++i) outer_input[i] = static_cast<uint16_t>(i + 1);
  int outer_input_dims[] = {3, 3, 2, 2};
  int outer_indices_dims[] = {1, 2};
  int32_t outer_indices[] = {2, 0};
  int outer_output_dims[] = {3, 0, 0, 0};
  uint16_t outer_output[8] = {};
  EXPECT_EQ(kTfLiteOk, RunGather(outer_input, outer_input_dims, outer_indices,
                                 outer_indices_dims, outer_output,
                                 outer_output_dims, kTfLiteFloat16));
  const uint16_t outer_expected[] = {9, 10, 11, 12, 1, 2, 3, 4};
  ExpectBits(outer_expected, outer_output, 8);

  int repeated_input_dims[] = {3, 2, 3, 4};
  int repeated_indices_dims[] = {1, 5};
  int32_t repeated_indices[] = {1, 0, 1, 1, 0};
  int repeated_output_dims[] = {3, 0, 0, 0};
  uint16_t repeated_input[24];
  uint16_t repeated_output[60] = {};
  uint16_t repeated_expected[60];
  for (int i = 0; i < 24; ++i) repeated_input[i] = static_cast<uint16_t>(i);
  for (int index = 0; index < 5; ++index) {
    for (int i = 0; i < 12; ++i) {
      repeated_expected[index * 12 + i] =
          repeated_input[repeated_indices[index] * 12 + i];
    }
  }
  EXPECT_EQ(kTfLiteOk,
            RunGather(repeated_input, repeated_input_dims, repeated_indices,
                      repeated_indices_dims, repeated_output,
                      repeated_output_dims, kTfLiteFloat16));
  ExpectBits(repeated_expected, repeated_output, 60);
  ExpectCalls(Route::kGatherF16, 4);
}

TEST(HeliaGatherPairTest, Float16LateInvalidIndicesAreAtomic) {
  ResetRoutes();
  const uint16_t input[] = {0x3c00, 0x4000, 0x4200, 0x4400};
  int input_dims[] = {1, 4};
  int indices_dims[] = {1, 3};
  int32_t indices[] = {0, 1, -1};
  int output_dims[] = {1, 0};
  uint16_t output[] = {0x5555, 0x5555, 0x5555};
  EXPECT_EQ(kTfLiteError, RunGather(input, input_dims, indices, indices_dims,
                                    output, output_dims, kTfLiteFloat16));
  const uint16_t sentinel[] = {0x5555, 0x5555, 0x5555};
  ExpectBits(sentinel, output, 3);
  ExpectCalls(Route::kGatherF16, 1);

  int32_t upper_indices[] = {0, 1, 4};
  int upper_output_dims[] = {1, 0};
  uint16_t upper_output[] = {0x5757, 0x5757, 0x5757};
  EXPECT_EQ(kTfLiteError,
            RunGather(input, input_dims, upper_indices, indices_dims,
                      upper_output, upper_output_dims, kTfLiteFloat16));
  const uint16_t upper_sentinel[] = {0x5757, 0x5757, 0x5757};
  ExpectBits(upper_sentinel, upper_output, 3);
  ExpectCalls(Route::kGatherF16, 2);

  int nd_input_dims[] = {2, 2, 2};
  int nd_indices_dims[] = {2, 2, 2};
  int32_t nd_indices[] = {0, 0, 0, 2};
  int nd_output_dims[] = {1, 0};
  uint16_t nd_output[] = {0x6666, 0x6666};
  EXPECT_EQ(kTfLiteError,
            RunGatherNd(input, nd_input_dims, nd_indices, nd_indices_dims,
                        nd_output, nd_output_dims, kTfLiteFloat16));
  const uint16_t nd_sentinel[] = {0x6666, 0x6666};
  ExpectBits(nd_sentinel, nd_output, 2);
  ExpectCalls(Route::kGatherNdF16, 1);

  int32_t nd_negative_indices[] = {0, 0, -1, 1};
  int nd_negative_output_dims[] = {1, 0};
  uint16_t nd_negative_output[] = {0x6868, 0x6868};
  EXPECT_EQ(kTfLiteError, RunGatherNd(input, nd_input_dims, nd_negative_indices,
                                      nd_indices_dims, nd_negative_output,
                                      nd_negative_output_dims, kTfLiteFloat16));
  const uint16_t nd_negative_sentinel[] = {0x6868, 0x6868};
  ExpectBits(nd_negative_sentinel, nd_negative_output, 2);
  ExpectCalls(Route::kGatherNdF16, 2);
}
#endif

TF_LITE_MICRO_TESTS_MAIN
