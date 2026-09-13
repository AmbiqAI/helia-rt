/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kParams = 0;
constexpr int kIndices = 1;
constexpr int kOutputTensor = 0;
constexpr int kMaxIndicesNd = 5;
constexpr int kCoreMaxRank = 4;

class ScopedTempTensor {
 public:
  ScopedTempTensor(MicroContext* context, TfLiteTensor* tensor)
      : context_(context), tensor_(tensor) {}
  ~ScopedTempTensor() {
    if (tensor_ != nullptr) {
      context_->DeallocateTempTfLiteTensor(tensor_);
    }
  }

 private:
  MicroContext* context_;
  TfLiteTensor* tensor_;
};

bool ShapeFits(const TfLiteIntArray* shape, size_t element_size) {
  if (shape == nullptr || shape->size < 0 || element_size == 0) {
    return false;
  }
  bool has_zero = false;
  for (int i = 0; i < shape->size; ++i) {
    if (shape->data[i] < 0) {
      return false;
    }
    if (shape->data[i] == 0) {
      has_zero = true;
    }
  }
  if (has_zero) return true;
  size_t count = 1;
  const size_t limit = INT32_MAX / element_size;
  for (int i = 0; i < shape->size; ++i) {
    if (static_cast<size_t>(shape->data[i]) > limit / count) {
      return false;
    }
    count *= static_cast<size_t>(shape->data[i]);
  }
  return true;
}

bool RangeHasZero(const TfLiteIntArray* shape, int begin, int end) {
  for (int i = begin; i < end; ++i) {
    if (shape->data[i] == 0) {
      return true;
    }
  }
  return false;
}

bool AccumulateRange(const TfLiteIntArray* shape, int begin, int end,
                     size_t limit, size_t* count) {
  for (int i = begin; i < end; ++i) {
    const size_t extent = static_cast<size_t>(shape->data[i]);
    if (extent > limit / *count) {
      return false;
    }
    *count *= extent;
  }
  return true;
}

bool GatherNdOutputFits(const TfLiteIntArray* params_shape,
                        const TfLiteIntArray* indices_shape, int indices_nd,
                        size_t element_size) {
  if (RangeHasZero(indices_shape, 0, indices_shape->size - 1) ||
      RangeHasZero(params_shape, indices_nd, params_shape->size)) {
    return true;
  }
  size_t count = 1;
  const size_t limit = INT32_MAX / element_size;
  return AccumulateRange(indices_shape, 0, indices_shape->size - 1, limit,
                         &count) &&
         AccumulateRange(params_shape, indices_nd, params_shape->size, limit,
                         &count);
}

bool HasZeroExtent(const TfLiteIntArray* shape) {
  return RangeHasZero(shape, 0, shape->size);
}

bool FitsCoreGatherNd(const TfLiteIntArray* params_shape,
                      const TfLiteIntArray* indices_shape, int indices_nd,
                      int output_rank) {
  return params_shape->size >= 1 && params_shape->size <= kCoreMaxRank &&
         indices_shape->size >= 1 && indices_shape->size <= kCoreMaxRank &&
         indices_nd >= 1 && indices_nd <= params_shape->size &&
         output_rank >= 0 && output_rank <= kCoreMaxRank;
}

cmsis_nn_dims MakeCoreDims(const TfLiteIntArray* shape) {
  cmsis_nn_dims dims = {1, 1, 1, 1};
  if (shape->size > 0) dims.n = shape->data[0];
  if (shape->size > 1) dims.h = shape->data[1];
  if (shape->size > 2) dims.w = shape->data[2];
  if (shape->size > 3) dims.c = shape->data[3];
  return dims;
}

TfLiteStatus CoreStatus(const char* operation, arm_cmsis_nn_status status) {
  if (status == ARM_CMSIS_NN_SUCCESS) {
    return kTfLiteOk;
  }
  MicroPrintf("%s failed (%d).", operation, status);
  return kTfLiteError;
}

TfLiteStatus ValidateGatherNdIndices(const TfLiteEvalTensor* params,
                                     const TfLiteEvalTensor* indices) {
  const int indices_rank = indices->dims->size;
  const int indices_nd = indices->dims->data[indices_rank - 1];
  if (RangeHasZero(indices->dims, 0, indices_rank - 1) || indices_nd == 0) {
    return kTfLiteOk;
  }

  size_t n_slices = 1;
  for (int i = 0; i < indices_rank - 1; ++i) {
    n_slices *= static_cast<size_t>(indices->dims->data[i]);
  }
  const int32_t* indices_data = tflite::micro::GetTensorData<int32_t>(indices);
  if (indices_data == nullptr) {
    return kTfLiteError;
  }
  for (size_t i = 0; i < n_slices; ++i) {
    for (int j = 0; j < indices_nd; ++j) {
      const int32_t index = indices_data[i * indices_nd + j];
      if (index < 0 || index >= params->dims->data[j]) {
        return kTfLiteError;
      }
    }
  }
  return kTfLiteOk;
}

template <typename ParamsT, typename IndicesT>
TfLiteStatus GatherNdReference(const TfLiteEvalTensor* params,
                               const TfLiteEvalTensor* indices,
                               TfLiteEvalTensor* output) {
  const int indices_dims = indices->dims->size;
  const int indices_nd = indices->dims->data[indices_dims - 1];
  const int params_dims = params->dims->size;
  const IndicesT* index_data = tflite::micro::GetTensorData<IndicesT>(indices);
  const ParamsT* param_data = tflite::micro::GetTensorData<ParamsT>(params);
  ParamsT* output_data = tflite::micro::GetTensorData<ParamsT>(output);

  if (RangeHasZero(indices->dims, 0, indices_dims - 1)) {
    return kTfLiteOk;
  }
  const bool empty_slice = RangeHasZero(params->dims, indices_nd, params_dims);
  if (indices_nd == 0 && empty_slice) {
    return kTfLiteOk;
  }

  int n_slices = 1;
  for (int i = 0; i < indices_dims - 1; ++i) {
    n_slices *= indices->dims->data[i];
  }

  if (indices_nd > 0) {
    if (index_data == nullptr) {
      return kTfLiteError;
    }
    for (int i = 0; i < n_slices; ++i) {
      for (int j = 0; j < indices_nd; ++j) {
        const IndicesT index = index_data[i * indices_nd + j];
        if (index < 0 || index >= params->dims->data[j]) {
          return kTfLiteError;
        }
      }
    }
  }

  if (empty_slice) {
    return kTfLiteOk;
  }

  int slice_size = 1;
  for (int i = indices_nd; i < params_dims; ++i) {
    slice_size *= params->dims->data[i];
  }
  if (param_data == nullptr || output_data == nullptr) {
    return kTfLiteError;
  }

  int params_flat_size = ElementCount(*params->dims);
  int remain_flat_size = params_flat_size;
  int dims_to_count[kMaxIndicesNd];
  for (int i = 0; i < indices_nd; ++i) {
    dims_to_count[i] = remain_flat_size / params->dims->data[i];
    remain_flat_size = dims_to_count[i];
  }

  for (int i = 0; i < n_slices; ++i) {
    int from_pos = 0;
    for (int j = 0; j < indices_nd; ++j) {
      int offset = i * indices_nd + j;
      IndicesT index = index_data[offset];
      from_pos += index * dims_to_count[j];
    }
    if (from_pos < 0 || from_pos + slice_size > params_flat_size) {
      return kTfLiteError;
    }
    std::memcpy(output_data + i * slice_size, param_data + from_pos,
                sizeof(ParamsT) * slice_size);
  }
  return kTfLiteOk;
}

TfLiteStatus GatherNdPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* params = micro_context->AllocateTempInputTensor(node, kParams);
  ScopedTempTensor params_guard(micro_context, params);
  TF_LITE_ENSURE(context, params != nullptr);
  TfLiteTensor* indices =
      micro_context->AllocateTempInputTensor(node, kIndices);
  ScopedTempTensor indices_guard(micro_context, indices);
  TF_LITE_ENSURE(context, indices != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  ScopedTempTensor output_guard(micro_context, output);
  TF_LITE_ENSURE(context, output != nullptr);

  switch (params->type) {
    case kTfLiteFloat32:
    case kTfLiteInt8:
    case kTfLiteInt16:
      break;
    case kTfLiteFloat16:
#if ARM_NN_ENABLE_F16
      break;
#else
      MicroPrintf("Float16 gather_nd requires ARM_NN_ENABLE_F16.");
      return kTfLiteError;
#endif
    default:
      MicroPrintf("Params of type '%s' are not supported by gather_nd.",
                  TfLiteTypeGetName(params->type));
      return kTfLiteError;
  }
  if (indices->type != kTfLiteInt32) {
    MicroPrintf("Indices of type '%s' are not supported by gather_nd.",
                TfLiteTypeGetName(indices->type));
    return kTfLiteError;
  }
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, params->type);

  size_t element_size = 0;
  TF_LITE_ENSURE_OK(context, TfLiteTypeSizeOf(params->type, &element_size));
  TF_LITE_ENSURE(context, ShapeFits(params->dims, element_size));
  TF_LITE_ENSURE(context, ShapeFits(indices->dims, sizeof(int32_t)));
  TF_LITE_ENSURE(context, output->dims != nullptr && output->dims->size >= 0);

  const int params_rank = params->dims->size;
  const int indices_rank = indices->dims->size;
  TF_LITE_ENSURE(context, params_rank >= 1);
  TF_LITE_ENSURE(context, indices_rank >= 1);
  const int indices_nd = indices->dims->data[indices_rank - 1];
  TF_LITE_ENSURE(context, indices_nd >= 0);
  TF_LITE_ENSURE(context, indices_nd <= params_rank);
  TF_LITE_ENSURE(context, indices_nd <= kMaxIndicesNd);

  const int64_t output_rank_64 =
      static_cast<int64_t>(indices_rank) - 1 + params_rank - indices_nd;
  TF_LITE_ENSURE(
      context,
      output_rank_64 >= 0 && output_rank_64 <= std::numeric_limits<int>::max());
  const int output_rank = static_cast<int>(output_rank_64);
  TF_LITE_ENSURE(context, output->dims->size >= output_rank);
  TF_LITE_ENSURE(context, GatherNdOutputFits(params->dims, indices->dims,
                                             indices_nd, element_size));

  const bool native_contract =
      FitsCoreGatherNd(params->dims, indices->dims, indices_nd, output_rank);
  if (params->type == kTfLiteInt16 || params->type == kTfLiteFloat16) {
    TF_LITE_ENSURE(context, native_contract);
  }

  TfLiteEvalTensor* output_eval =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_OK(context, tflite::micro::CreateWritableTensorDimsWithCopy(
                                 context, output, output_eval));

  TfLiteIntArray* output_shape = output->dims;
  int output_index = 0;
  for (int i = 0; i < indices_rank - 1; ++i) {
    output_shape->data[output_index++] = indices->dims->data[i];
  }
  for (int i = indices_nd; i < params_rank; ++i) {
    output_shape->data[output_index++] = params->dims->data[i];
  }
  output_shape->size = output_index;
  return kTfLiteOk;
}

TfLiteStatus GatherNdEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* params =
      tflite::micro::GetEvalInput(context, node, kParams);
  const TfLiteEvalTensor* indices =
      tflite::micro::GetEvalInput(context, node, kIndices);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  const int indices_nd = indices->dims->data[indices->dims->size - 1];
  const int output_rank =
      indices->dims->size - 1 + params->dims->size - indices_nd;
  const bool native_contract =
      FitsCoreGatherNd(params->dims, indices->dims, indices_nd, output_rank);
  const bool integer_nonempty = !HasZeroExtent(params->dims) &&
                                !HasZeroExtent(indices->dims) &&
                                !HasZeroExtent(output->dims);

  cmsis_nn_gather_nd_params core_params = {params->dims->size,
                                           indices->dims->size, 0};
  cmsis_nn_dims params_dims;
  cmsis_nn_dims indices_dims;
  cmsis_nn_dims output_dims;
  if (native_contract) {
    params_dims = MakeCoreDims(params->dims);
    indices_dims = MakeCoreDims(indices->dims);
    output_dims = MakeCoreDims(output->dims);
  }
  const int32_t* indices_data = tflite::micro::GetTensorData<int32_t>(indices);
  if (HasZeroExtent(output->dims)) {
    return ValidateGatherNdIndices(params, indices);
  }

  switch (params->type) {
    case kTfLiteFloat32:
#if ARM_NN_ENABLE_F32
      if (native_contract) {
        return CoreStatus(
            "arm_gather_nd_f32",
            arm_gather_nd_f32(
                tflite::micro::GetTensorData<float>(params), &params_dims,
                indices_data, &indices_dims, &core_params,
                tflite::micro::GetTensorData<float>(output), &output_dims));
      }
#endif
      return GatherNdReference<float, int32_t>(params, indices, output);
    case kTfLiteInt8:
      if (native_contract && integer_nonempty) {
        if (ValidateGatherNdIndices(params, indices) != kTfLiteOk) {
          return kTfLiteError;
        }
        return CoreStatus(
            "arm_gather_nd_s8",
            arm_gather_nd_s8(
                tflite::micro::GetTensorData<int8_t>(params), &params_dims,
                indices_data, &indices_dims, &core_params,
                tflite::micro::GetTensorData<int8_t>(output), &output_dims));
      }
      return GatherNdReference<int8_t, int32_t>(params, indices, output);
    case kTfLiteInt16:
      if (ValidateGatherNdIndices(params, indices) != kTfLiteOk) {
        return kTfLiteError;
      }
      return CoreStatus(
          "arm_gather_nd_s16",
          arm_gather_nd_s16(
              tflite::micro::GetTensorData<int16_t>(params), &params_dims,
              indices_data, &indices_dims, &core_params,
              tflite::micro::GetTensorData<int16_t>(output), &output_dims));
    case kTfLiteFloat16:
#if ARM_NN_ENABLE_F16
      return CoreStatus(
          "arm_gather_nd_f16",
          arm_gather_nd_f16(
              tflite::micro::GetTensorData<float16_t>(params), &params_dims,
              indices_data, &indices_dims, &core_params,
              tflite::micro::GetTensorData<float16_t>(output), &output_dims));
#else
      return kTfLiteError;
#endif
    default:
      MicroPrintf("Params of type '%s' are not supported by gather_nd.",
                  TfLiteTypeGetName(params->type));
      return kTfLiteError;
  }
}
}  // namespace

TFLMRegistration Register_GATHER_ND() {
  return tflite::micro::RegisterOp(nullptr, GatherNdPrepare, GatherNdEval);
}

}  // namespace tflite
