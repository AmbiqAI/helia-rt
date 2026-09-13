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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kInputPositions = 1;
constexpr int kOutputTensor = 0;
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

bool GatherOutputFits(const TfLiteIntArray* input_shape,
                      const TfLiteIntArray* coords_shape, int axis,
                      int batch_dims, size_t element_size) {
  if (RangeHasZero(input_shape, 0, axis) ||
      RangeHasZero(coords_shape, batch_dims, coords_shape->size) ||
      RangeHasZero(input_shape, axis + 1, input_shape->size)) {
    return true;
  }
  size_t count = 1;
  const size_t limit = INT32_MAX / element_size;
  return AccumulateRange(input_shape, 0, axis, limit, &count) &&
         AccumulateRange(coords_shape, batch_dims, coords_shape->size, limit,
                         &count) &&
         AccumulateRange(input_shape, axis + 1, input_shape->size, limit,
                         &count);
}

bool HasZeroExtent(const TfLiteIntArray* shape) {
  return RangeHasZero(shape, 0, shape->size);
}

bool FitsCoreGather(const TfLiteIntArray* input_shape,
                    const TfLiteIntArray* coords_shape, int output_rank,
                    bool allow_scalar_coords) {
  return input_shape->size >= 1 && input_shape->size <= kCoreMaxRank &&
         coords_shape->size <= kCoreMaxRank && output_rank >= 0 &&
         output_rank <= kCoreMaxRank &&
         (allow_scalar_coords || coords_shape->size >= 1);
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

TfLiteStatus ValidateGatherIndices(const TfLiteEvalTensor* coords,
                                   int axis_size) {
  if (HasZeroExtent(coords->dims)) {
    return kTfLiteOk;
  }
  size_t coords_count = 1;
  for (int i = 0; i < coords->dims->size; ++i) {
    coords_count *= static_cast<size_t>(coords->dims->data[i]);
  }
  const int32_t* coords_data = tflite::micro::GetTensorData<int32_t>(coords);
  if (coords_data == nullptr) {
    return kTfLiteError;
  }
  for (size_t i = 0; i < coords_count; ++i) {
    if (coords_data[i] < 0 || coords_data[i] >= axis_size) {
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

template <typename InputT, typename CoordsT = int32_t>
TfLiteStatus GatherReference(const TfLiteGatherParams* params,
                             const TfLiteEvalTensor* input,
                             const TfLiteEvalTensor* coords,
                             TfLiteEvalTensor* output) {
  const InputT* input_data = tflite::micro::GetTensorData<InputT>(input);
  const CoordsT* coords_data = tflite::micro::GetTensorData<CoordsT>(coords);
  InputT* output_data = tflite::micro::GetTensorData<InputT>(output);
  const TfLiteIntArray* input_dims = input->dims;
  const int input_dims_size = input_dims->size;
  int axis = params->axis;
  if (axis < 0) {
    axis += input_dims_size;
  }
  TFLITE_DCHECK_GE(axis, 0);
  TFLITE_DCHECK_LT(axis, input_dims_size);

  int batch_dims = params->batch_dims;
  const TfLiteIntArray* coords_dims = coords->dims;
  const int coords_dims_size = coords_dims->size;
  if (batch_dims < 0) {
    batch_dims += coords_dims_size;
  }
  TFLITE_DCHECK_GE(batch_dims, 0);
  TFLITE_DCHECK_LT(batch_dims, input_dims_size);
  TFLITE_DCHECK_LE(batch_dims, coords_dims_size);
  TFLITE_DCHECK_GE(axis, batch_dims);
  for (int i = 0; i < batch_dims; ++i) {
    TFLITE_DCHECK_EQ(input_dims->data[i], coords_dims->data[i]);
  }

  const int axis_size = input_dims->data[axis];
  if (ValidateGatherIndices(coords, axis_size) != kTfLiteOk) {
    return kTfLiteError;
  }

  if (HasZeroExtent(output->dims)) {
    return kTfLiteOk;
  }
  if (input_data == nullptr || output_data == nullptr) {
    return kTfLiteError;
  }

  int batch_size = 1;
  for (int i = 0; i < batch_dims; ++i) {
    batch_size *= input_dims->data[i];
  }
  int outer_size = 1;
  for (int i = batch_dims; i < axis; ++i) {
    outer_size *= input_dims->data[i];
  }
  int inner_size = 1;
  for (int i = axis + 1; i < input_dims_size; ++i) {
    inner_size *= input_dims->data[i];
  }
  int coord_size = 1;
  for (int i = batch_dims; i < coords_dims_size; ++i) {
    coord_size *= coords_dims->data[i];
  }

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int outer = 0; outer < outer_size; ++outer) {
      for (int coord = 0; coord < coord_size; ++coord) {
        std::memcpy(output_data +
                        (((batch * outer_size) + outer) * coord_size + coord) *
                            inner_size,
                    input_data + (((batch * outer_size) + outer) * axis_size +
                                  coords_data[batch * coord_size + coord]) *
                                     inner_size,
                    sizeof(InputT) * inner_size);
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus GatherPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const auto* params =
      reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  ScopedTempTensor input_guard(micro_context, input);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* coords =
      micro_context->AllocateTempInputTensor(node, kInputPositions);
  ScopedTempTensor coords_guard(micro_context, coords);
  TF_LITE_ENSURE(context, coords != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  ScopedTempTensor output_guard(micro_context, output);
  TF_LITE_ENSURE(context, output != nullptr);

  if (coords->type != kTfLiteInt32) {
    MicroPrintf("Positions of type '%s' are not supported by gather.",
                TfLiteTypeGetName(coords->type));
    return kTfLiteError;
  }

  switch (input->type) {
    case kTfLiteFloat32:
    case kTfLiteInt8:
    case kTfLiteInt16:
      break;
    case kTfLiteFloat16:
#if ARM_NN_ENABLE_F16
      break;
#else
      MicroPrintf("Float16 gather requires ARM_NN_ENABLE_F16.");
      return kTfLiteError;
#endif
    default:
      MicroPrintf("Type '%s' is not supported by gather.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);

  size_t element_size = 0;
  TF_LITE_ENSURE_OK(context, TfLiteTypeSizeOf(input->type, &element_size));
  TF_LITE_ENSURE(context, ShapeFits(input->dims, element_size));
  TF_LITE_ENSURE(context, ShapeFits(coords->dims, sizeof(int32_t)));
  TF_LITE_ENSURE(context, output->dims != nullptr && output->dims->size >= 0);

  const int input_rank = input->dims->size;
  const int coords_rank = coords->dims->size;
  TF_LITE_ENSURE(context, input_rank >= 1);

  int axis = params->axis;
  if (axis < 0) {
    axis += input_rank;
  }
  TF_LITE_ENSURE(context, axis >= 0 && axis < input_rank);

  int batch_dims = params->batch_dims;
  if (batch_dims < 0) {
    batch_dims += coords_rank;
  }
  TF_LITE_ENSURE(context, batch_dims >= 0 && batch_dims <= coords_rank);
  TF_LITE_ENSURE(context, batch_dims < input_rank && batch_dims <= axis);
  for (int i = 0; i < batch_dims; ++i) {
    TF_LITE_ENSURE_EQ(context, input->dims->data[i], coords->dims->data[i]);
  }

  const int64_t output_rank_64 =
      static_cast<int64_t>(input_rank) + coords_rank - batch_dims - 1;
  TF_LITE_ENSURE(
      context,
      output_rank_64 >= 0 && output_rank_64 <= std::numeric_limits<int>::max());
  const int output_rank = static_cast<int>(output_rank_64);
  TF_LITE_ENSURE(context, output->dims->size >= output_rank);
  TF_LITE_ENSURE(context, GatherOutputFits(input->dims, coords->dims, axis,
                                           batch_dims, element_size));

  const bool float_contract =
      FitsCoreGather(input->dims, coords->dims, output_rank, true);
  const bool integer_contract =
      FitsCoreGather(input->dims, coords->dims, output_rank, false);
  if (input->type == kTfLiteInt16) {
    TF_LITE_ENSURE(context, integer_contract);
  }
  if (input->type == kTfLiteFloat16) {
    TF_LITE_ENSURE(context, float_contract);
  }

  TfLiteEvalTensor* output_eval =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_OK(context, tflite::micro::CreateWritableTensorDimsWithCopy(
                                 context, output, output_eval));

  TfLiteIntArray* output_shape = output->dims;
  int output_index = 0;
  for (int i = 0; i < axis; ++i) {
    output_shape->data[output_index++] = input->dims->data[i];
  }
  for (int i = batch_dims; i < coords_rank; ++i) {
    output_shape->data[output_index++] = coords->dims->data[i];
  }
  for (int i = axis + 1; i < input_rank; ++i) {
    output_shape->data[output_index++] = input->dims->data[i];
  }
  output_shape->size = output_index;
  return kTfLiteOk;
}

TfLiteStatus GatherEval(TfLiteContext* context, TfLiteNode* node) {
  const auto* params =
      reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* coords =
      tflite::micro::GetEvalInput(context, node, kInputPositions);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  int batch_dims = params->batch_dims;
  if (batch_dims < 0) {
    batch_dims += coords->dims->size;
  }
  const int output_rank =
      input->dims->size + coords->dims->size - batch_dims - 1;
  const bool float_contract =
      FitsCoreGather(input->dims, coords->dims, output_rank, true);
  const bool integer_contract =
      FitsCoreGather(input->dims, coords->dims, output_rank, false);
  const bool integer_nonempty = !HasZeroExtent(input->dims) &&
                                !HasZeroExtent(coords->dims) &&
                                !HasZeroExtent(output->dims);

  cmsis_nn_gather_params core_params = {params->axis, params->batch_dims,
                                        input->dims->size, coords->dims->size};
  cmsis_nn_dims input_dims;
  cmsis_nn_dims coords_dims;
  cmsis_nn_dims output_dims;
  if (float_contract || integer_contract) {
    input_dims = MakeCoreDims(input->dims);
    coords_dims = MakeCoreDims(coords->dims);
    output_dims = MakeCoreDims(output->dims);
  }
  const int32_t* coords_data = tflite::micro::GetTensorData<int32_t>(coords);
  int axis = params->axis;
  if (axis < 0) {
    axis += input->dims->size;
  }
  const int axis_size = input->dims->data[axis];
  if (HasZeroExtent(output->dims)) {
    return ValidateGatherIndices(coords, axis_size);
  }

  switch (input->type) {
    case kTfLiteFloat32:
#if ARM_NN_ENABLE_F32
      if (float_contract) {
        return CoreStatus(
            "arm_gather_f32",
            arm_gather_f32(tflite::micro::GetTensorData<float>(input),
                           &input_dims, coords_data, &coords_dims, &core_params,
                           tflite::micro::GetTensorData<float>(output),
                           &output_dims));
      }
#endif
      return GatherReference<float>(params, input, coords, output);
    case kTfLiteInt8:
      if (integer_contract && integer_nonempty) {
        if (ValidateGatherIndices(coords, axis_size) != kTfLiteOk) {
          return kTfLiteError;
        }
        return CoreStatus(
            "arm_gather_s8",
            arm_gather_s8(tflite::micro::GetTensorData<int8_t>(input),
                          &input_dims, coords_data, &coords_dims, &core_params,
                          tflite::micro::GetTensorData<int8_t>(output),
                          &output_dims));
      }
      return GatherReference<int8_t>(params, input, coords, output);
    case kTfLiteInt16:
      if (ValidateGatherIndices(coords, axis_size) != kTfLiteOk) {
        return kTfLiteError;
      }
      return CoreStatus(
          "arm_gather_s16",
          arm_gather_s16(tflite::micro::GetTensorData<int16_t>(input),
                         &input_dims, coords_data, &coords_dims, &core_params,
                         tflite::micro::GetTensorData<int16_t>(output),
                         &output_dims));
    case kTfLiteFloat16:
#if ARM_NN_ENABLE_F16
      return CoreStatus(
          "arm_gather_f16",
          arm_gather_f16(tflite::micro::GetTensorData<float16_t>(input),
                         &input_dims, coords_data, &coords_dims, &core_params,
                         tflite::micro::GetTensorData<float16_t>(output),
                         &output_dims));
#else
      return kTfLiteError;
#endif
    default:
      MicroPrintf("Type '%s' is not supported by gather.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}
}  // namespace

TFLMRegistration Register_GATHER() {
  return tflite::micro::RegisterOp(nullptr, GatherPrepare, GatherEval);
}

}  // namespace tflite
