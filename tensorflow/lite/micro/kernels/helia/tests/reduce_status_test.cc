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

// MEAN and SUM must fail Invoke when the heliaCORE entry they call fails,
// for every dtype heliaCORE serves, rather than fall back to the reference.
// GCC and ATfE links wrap the entries to count each call and to make one
// fail; other toolchains run no checks.

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

#if ARM_NN_ENABLE_F32 || ARM_NN_ENABLE_F16
#include "Include/arm_nnfunctions_flt.h"
#endif

namespace {

enum class Leaf {
  kMeanS8,
  kMeanS16,
  kMeanF32,
  kSumF32,
  kMeanF16,
  kSumF16,
  kCount
};

[[maybe_unused]] int g_calls[static_cast<int>(Leaf::kCount)] = {};
[[maybe_unused]] Leaf g_fail = Leaf::kCount;

#if HELIA_REDUCE_STATUS_LINK_WRAP
// Counts the call; true when the test asked this entry to fail.
bool Fail(Leaf leaf) {
  ++g_calls[static_cast<int>(leaf)];
  return g_fail == leaf;
}
#endif

}  // namespace

#if HELIA_REDUCE_STATUS_LINK_WRAP
extern "C" {

#define HELIA_MEAN_Q_WRAP(NAME, T, LEAF)                                  \
  arm_cmsis_nn_status __real_##NAME(                                      \
      const T*, const cmsis_nn_dims*, int32_t, const cmsis_nn_dims*, T*,  \
      const cmsis_nn_dims*, int32_t, int32_t, int32_t);                   \
  arm_cmsis_nn_status __wrap_##NAME(const T* in, const cmsis_nn_dims* di, \
                                    int32_t io, const cmsis_nn_dims* da,  \
                                    T* out, const cmsis_nn_dims* dout,    \
                                    int32_t oo, int32_t m, int32_t s) {   \
    if (Fail(LEAF)) {                                                     \
      return ARM_CMSIS_NN_ARG_ERROR;                                      \
    }                                                                     \
    return __real_##NAME(in, di, io, da, out, dout, oo, m, s);            \
  }

#define HELIA_REDUCE_FLT_WRAP(NAME, T, LEAF)                              \
  arm_cmsis_nn_status __real_##NAME(const T*, const cmsis_nn_dims*,       \
                                    const cmsis_nn_dims*, T*,             \
                                    const cmsis_nn_dims*);                \
  arm_cmsis_nn_status __wrap_##NAME(const T* in, const cmsis_nn_dims* di, \
                                    const cmsis_nn_dims* da, T* out,      \
                                    const cmsis_nn_dims* dout) {          \
    if (Fail(LEAF)) {                                                     \
      return ARM_CMSIS_NN_ARG_ERROR;                                      \
    }                                                                     \
    return __real_##NAME(in, di, da, out, dout);                          \
  }

HELIA_MEAN_Q_WRAP(arm_mean_s8, int8_t, Leaf::kMeanS8)
HELIA_MEAN_Q_WRAP(arm_mean_s16, int16_t, Leaf::kMeanS16)
#if ARM_NN_ENABLE_F32
HELIA_REDUCE_FLT_WRAP(arm_nn_mean_f32, float32_t, Leaf::kMeanF32)
HELIA_REDUCE_FLT_WRAP(arm_reduce_sum_f32, float32_t, Leaf::kSumF32)
#endif
#if ARM_NN_ENABLE_F16
HELIA_REDUCE_FLT_WRAP(arm_nn_mean_f16, float16_t, Leaf::kMeanF16)
HELIA_REDUCE_FLT_WRAP(arm_reduce_sum_f16, float16_t, Leaf::kSumF16)
#endif

#undef HELIA_REDUCE_FLT_WRAP
#undef HELIA_MEAN_Q_WRAP

}  // extern "C"
#endif  // HELIA_REDUCE_STATUS_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

// Prepares and invokes `registration` on a [2, 3] input of `type` reduced
// over axis 1 into [2], with entry `fail` made to fail, and returns the
// invoke status. Prepare must succeed and `leaf` must be called once.
template <typename T>
[[maybe_unused]] TfLiteStatus InvokeWithFailure(
    const TFLMRegistration& registration, TfLiteType type, Leaf leaf,
    Leaf fail) {
  static T input[6] = {};
  static T output[2] = {};
  const int32_t axis[] = {1};
  int input_dims[] = {2, 2, 3};
  int axis_dims[] = {1, 1};
  int output_dims[] = {1, 2};
  TfLiteTensor tensors[3];
  if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>) {
    tensors[0] =
        CreateQuantizedTensor(input, IntArrayFromInts(input_dims), 0.5f, 0);
    tensors[2] =
        CreateQuantizedTensor(output, IntArrayFromInts(output_dims), 0.5f, 0);
  } else {
    tensors[0] = CreateTensor(input, IntArrayFromInts(input_dims), false, type);
    tensors[2] =
        CreateTensor(output, IntArrayFromInts(output_dims), false, type);
  }
  tensors[1] = CreateTensor(axis, IntArrayFromInts(axis_dims));
  tensors[1].allocation_type = kTfLiteMmapRo;
  TfLiteReducerParams params = {};
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  micro::KernelRunner runner(registration, tensors, 3, IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  memset(g_calls, 0, sizeof(g_calls));
  g_fail = fail;
  const TfLiteStatus status = runner.Invoke();
  g_fail = Leaf::kCount;
  EXPECT_EQ(1, g_calls[static_cast<int>(leaf)]);
  return status;
}

// MEAN calls `mean_leaf` and SUM `sum_leaf`: each succeeds as is and fails
// Invoke when its entry fails.
template <typename T>
[[maybe_unused]] void ExpectFailureFailsInvoke(TfLiteType type, Leaf mean_leaf,
                                               Leaf sum_leaf) {
  EXPECT_EQ(kTfLiteOk, InvokeWithFailure<T>(Register_MEAN(), type, mean_leaf,
                                            Leaf::kCount));
  EXPECT_EQ(kTfLiteError,
            InvokeWithFailure<T>(Register_MEAN(), type, mean_leaf, mean_leaf));
  EXPECT_EQ(kTfLiteOk,
            InvokeWithFailure<T>(Register_SUM(), type, sum_leaf, Leaf::kCount));
  EXPECT_EQ(kTfLiteError,
            InvokeWithFailure<T>(Register_SUM(), type, sum_leaf, sum_leaf));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(HeliaReduceStatusTest, Int8) {
#if HELIA_REDUCE_STATUS_LINK_WRAP
  tflite::testing::ExpectFailureFailsInvoke<int8_t>(kTfLiteInt8, Leaf::kMeanS8,
                                                    Leaf::kMeanS8);
#endif
}

TEST(HeliaReduceStatusTest, Int16) {
#if HELIA_REDUCE_STATUS_LINK_WRAP
  tflite::testing::ExpectFailureFailsInvoke<int16_t>(
      kTfLiteInt16, Leaf::kMeanS16, Leaf::kMeanS16);
#endif
}

// Without ARM_NN_ENABLE_F32 float32 MEAN and SUM take the reference only.
TEST(HeliaReduceStatusTest, Float32) {
#if HELIA_REDUCE_STATUS_LINK_WRAP && ARM_NN_ENABLE_F32
  tflite::testing::ExpectFailureFailsInvoke<float>(
      kTfLiteFloat32, Leaf::kMeanF32, Leaf::kSumF32);
#endif
}

TEST(HeliaReduceStatusTest, Float16) {
#if HELIA_REDUCE_STATUS_LINK_WRAP && ARM_NN_ENABLE_F16
  tflite::testing::ExpectFailureFailsInvoke<uint16_t>(
      kTfLiteFloat16, Leaf::kMeanF16, Leaf::kSumF16);
#endif
}

TF_LITE_MICRO_TESTS_MAIN
