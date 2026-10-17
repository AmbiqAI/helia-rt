/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "Include/arm_nnfunctions.h"

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/helia/helia_float_common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/mean.h"
#include "tensorflow/lite/micro/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/reduce.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace micro {

// MEAN and SUM iterate with fixed arrays: an input rank above this fails, and
// the reference axis resolution keeps at most one entry per input axis.
const int kMaxNumberOfAxis = 5;
constexpr int kCmsisRank = 4;

/**
 * Populate the CMSIS NN 4D shapes and axis mask from the TFLite input tensor
 * and axis information. The input shape is padded at the end, so a TFLite axis
 * index selects the same cmsis_nn_dims member; the output shape is derived
 * from the mask rather than read from the output tensor.
 *
 * @param input_t The input tensor to the reduce operation.
 * @param axis_data Pointer to the axis data array.
 * @param axis_count The number of axes to reduce.
 * @param in_dims Pointer to the cmsis_nn_dims structure for input dimensions.
 * @param axis_dims Pointer to the cmsis_nn_dims structure for axis dimensions.
 * @param out_dims Pointer to the cmsis_nn_dims structure for output dimensions.
 *
 * @return The number of input elements per output element.
 */
static int64_t PopulateCmsisShapeAndAxis(const TfLiteEvalTensor* input_t,
                                         const int32_t* axis_data,
                                         int axis_count,
                                         cmsis_nn_dims* in_dims,
                                         cmsis_nn_dims* axis_dims,
                                         cmsis_nn_dims* out_dims) {

  // --- 1) Pad input shape at end ---
  int32_t in_shape[kCmsisRank] = {1, 1, 1, 1};
  int in_rank = input_t->dims->size;
  for (int i = 0; i < in_rank && i < kCmsisRank; ++i) {
    in_shape[i] = input_t->dims->data[i];
  }
  in_dims->n = in_shape[0];
  in_dims->h = in_shape[1];
  in_dims->w = in_shape[2];
  in_dims->c = in_shape[3];

  // --- 2) Build axis mask & count elements ---
  int64_t num_elements_in_axis = 1;
  axis_dims->n = axis_dims->h = axis_dims->w = axis_dims->c = 0;
  for (int i = 0; i < axis_count; ++i) {
    int a = axis_data[i];
    if (a < 0) a += in_rank;
    switch (a) {
      case 0:
        if (!axis_dims->n) {
          axis_dims->n = 1;
          num_elements_in_axis *= in_shape[0];
        }
        break;
      case 1:
        if (!axis_dims->h) {
          axis_dims->h = 1;
          num_elements_in_axis *= in_shape[1];
        }
        break;
      case 2:
        if (!axis_dims->w) {
          axis_dims->w = 1;
          num_elements_in_axis *= in_shape[2];
        }
        break;
      case 3:
        if (!axis_dims->c) {
          axis_dims->c = 1;
          num_elements_in_axis *= in_shape[3];
        }
        break;
      default:
        break;  // out of range gets ignored
    }
  }

  // --- 3) Pad output shape at end ---
  // NOTE: We do not rely on provided output tensor shape because it could be flattened or a
  // different view (e.g. [1, 2] instead of [2, 1]). This is atypical but unit tests do this...
  out_dims->n = axis_dims->n ? 1 : in_dims->n;
  out_dims->h = axis_dims->h ? 1 : in_dims->h;
  out_dims->w = axis_dims->w ? 1 : in_dims->w;
  out_dims->c = axis_dims->c ? 1 : in_dims->c;

  return num_elements_in_axis;
}

/**
 * As PopulateCmsisShapeAndAxis, and additionally folds 1/count into the output
 * multiplier and shift for the quantized MEAN path.
 *
 * @param output_t The output tensor from the reduce operation.
 * @param output_mult Pointer to the output multiplier.
 * @param output_shift Pointer to the output shift value.
 * @param compute_sum Boolean indicating whether to compute the sum (true) or mean (false).
 */
static void PopulateCmsisParams(const TfLiteEvalTensor* input_t,
                                const TfLiteEvalTensor* output_t,
                                const int32_t* axis_data,
                                int axis_count,
                                cmsis_nn_dims* in_dims,
                                cmsis_nn_dims* axis_dims,
                                cmsis_nn_dims* out_dims,
                                int32_t *output_mult,
                                int32_t *output_shift,
                                bool compute_sum) {

  (void)output_t;  // Unused in this function, but may be used later.

  const int64_t num_elements_in_axis = PopulateCmsisShapeAndAxis(
      input_t, axis_data, axis_count, in_dims, axis_dims, out_dims);

  // Fold 1/count into output multiplier & shift (only for MEAN)
  if (!compute_sum) {
    TFLITE_DCHECK_GE(num_elements_in_axis, 0);
    int shift = 63 - CountLeadingZeros(static_cast<uint64_t>(num_elements_in_axis));
    shift = std::min(shift, 32);
    shift = std::min(shift, 31 + static_cast<int>(*output_shift));
    *output_mult = static_cast<int32_t>(
        (static_cast<int64_t>(*output_mult) << shift) /
        num_elements_in_axis);
    *output_shift = *output_shift - shift;
  }

}


namespace {

TfLiteStatus PrepareSimple(TfLiteContext* context, TfLiteNode* node,
                           int32_t* multiplier, int* shift) {
  MicroContext* micro_context = GetMicroContext(context);

  // Inputs Tensor (dtype depends on quantization):
  // [0] = Input
  // [1] = Axis
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);

  // Outputs Tensor (dtype depends on quantization):
  // [0] = Output

  // Validate number of inputs and outputs
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Validate axis type
  TfLiteTensor* axis = micro_context->AllocateTempInputTensor(node, 1);
  TF_LITE_ENSURE(context, axis != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, axis->type, kTfLiteInt32);

  if (input->type == kTfLiteInt8) {
    TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
    const double real_multiplier = static_cast<double>(input->params.scale) /
                                   static_cast<double>(output->params.scale);
    QuantizeMultiplier(real_multiplier, multiplier, shift);
    micro_context->DeallocateTempTfLiteTensor(output);
  }
  micro_context->DeallocateTempTfLiteTensor(axis);
  micro_context->DeallocateTempTfLiteTensor(input);
  return kTfLiteOk;
}

// MeanParams holds four axes, so a longer list keeps its first four and an
// axis_count of 4; only the two-axis 4-D special case reads them.
void ResolveAxis(const int* axis_data, int axis_count,
                 tflite::MeanParams* op_params) {
  const int count = axis_count < 4 ? axis_count : 4;
  int i = 0;
  for (; i < count; ++i) {
    op_params->axis[i] = static_cast<int16_t>(axis_data[i]);
  }
  for (; i < 4; ++i) {
    op_params->axis[i] = 1;
  }
  op_params->axis_count = count;
}

// heliaCORE's float reductions take a 4-D NHWC shape and a 4-D axis mask, so
// they cover any axis-mask combination and either keep_dims setting at rank up
// to 4; a higher rank, or an axis outside the input rank, has no such spelling.
// A scalar input ignores its axes, as the reference's axis resolution does.
// see AmbiqAI/ns-cmsis-nn#428
bool HeliaFloatReduceSupported(const TfLiteEvalTensor* input,
                               const int32_t* axis_data, int axis_count) {
  const int input_rank = input->dims->size;
  if (input_rank > kCmsisRank) {
    return false;
  }
  if (input_rank == 0) {
    return true;
  }
  for (int i = 0; i < axis_count; ++i) {
    int axis = axis_data[i];
    if (axis < 0) {
      axis += input_rank;
    }
    if (axis < 0 || axis >= input_rank) {
      return false;
    }
  }
  return true;
}

// Float16 reductions are optimized-only (TFLM has no float16 reference) and
// heliaCORE takes 4-D NHWC dims, so reject the rest at Prepare. A non-constant
// axis tensor holds no data until Invoke; Eval re-checks it either way.
TfLiteStatus PrepareFloat16Reduce(TfLiteContext* context,
                                  const TfLiteTensor* input,
                                  const TfLiteTensor* axis, const char* op) {
  if (!kHeliaFloat16Enabled) {
    MicroPrintf("Float16 %s requires ARM_NN_ENABLE_F16.", op);
    return kTfLiteError;
  }
  const int input_rank = NumDimensions(input);
  if (input_rank > kCmsisRank) {
    MicroPrintf("Float16 %s supports rank 4 and below.", op);
    return kTfLiteError;
  }
  if (input_rank > 0 && IsConstantTensor(axis)) {
    const int32_t* axis_data = GetTensorData<int32_t>(axis);
    for (int i = 0; i < NumElements(axis); ++i) {
      const int32_t reduce_axis =
          axis_data[i] < 0 ? axis_data[i] + input_rank : axis_data[i];
      if (reduce_axis < 0 || reduce_axis >= input_rank) {
        MicroPrintf(
            "Float16 %s requires every axis, after normalizing a negative "
            "value against the rank, to be inside [0, rank).",
            op);
        return kTfLiteError;
      }
    }
  }
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus QuantizedMeanOrSum(TfLiteContext* context, TfLiteNode* node,
                                int* temp_index, int* resolved_axis,
                                int32_t* temp_sum, OpDataReduce* op_data,
                                bool compute_sum) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TfLiteReducerParams* params =
      static_cast<TfLiteReducerParams*>(node->builtin_data);

  bool result = reference_ops::QuantizedMeanOrSumExtraArgs<T, int32_t>(
      tflite::micro::GetTensorData<T>(input), op_data->input_zp,
      op_data->input_scale, &input->dims->data[0], input->dims->size,
      tflite::micro::GetTensorData<T>(output), op_data->output_scale,
      op_data->multiplier, op_data->shift, op_data->output_zp,
      &output->dims->data[0], output->dims->size,
      tflite::micro::GetTensorData<int>(axis), op_data->num_axis,
      params->keep_dims, temp_index, resolved_axis, temp_sum, compute_sum);
  TF_LITE_ENSURE(context, result);

  return kTfLiteOk;
}

template <typename integer_type>
TfLiteStatus EvalIntegerMean(TfLiteContext* context, TfLiteNode* node,
                             int num_axis, OpDataReduce* op_data,
                             int* temp_index, int* resolved_axis) {
  int32_t* temp_sum = static_cast<int32_t*>(
      context->GetScratchBuffer(context, op_data->scratch_accumulator_idx));

  TF_LITE_ENSURE_OK(context, QuantizedMeanOrSum<integer_type>(
                                 context, node, temp_index, resolved_axis,
                                 temp_sum, op_data, /*compute_sum=*/false));

  return kTfLiteOk;
}

enum MinMaxEvalType { kEvalMin, kEvalMax };

template <typename T>
struct MinMaxReducerCompare {
  MinMaxReducerCompare() = delete;
  MinMaxReducerCompare(MinMaxEvalType evalType) : type_(evalType){};

  constexpr T initialValue() const {
    return (type_ == kEvalMin) ? std::numeric_limits<T>::max()
                               : std::numeric_limits<T>::lowest();
  }

  // should be able to use "auto" keyword here, but GCC and Clang blow a fuse
  T (*compare())(const T, const T) {
    if (type_ == kEvalMin) {
      return [](const T current, const T in) -> T {
        return (in < current) ? in : current;
      };
    } else {
      return [](const T current, const T in) -> T {
        return (in > current) ? in : current;
      };
    }
  }

 private:
  MinMaxEvalType type_;
};

// Element count of `dims`, or -1 for a negative dim or a nonzero count past
// INT32_MAX: such a count is not compared, since saturated counts could match.
int64_t CheckedElementCount(const int* dims, int count) {
  int64_t elements = 1;
  bool past_int32 = false;
  for (int i = 0; i < count; ++i) {
    if (dims[i] < 0) {
      return -1;
    }
    if (dims[i] == 0) {
      elements = 0;
    } else if (elements != 0 && !past_int32) {
      elements *= dims[i];
      past_int32 = elements > INT32_MAX;
    }
  }
  return elements != 0 && past_int32 ? -1 : elements;
}

// No rank limit beyond the input's own: REDUCE_MIN/MAX and REDUCE_ALL size
// their reference iteration scratch from the input rank in Prepare.
constexpr int kAnyReduceRank = INT32_MAX;

// Both heliaCORE and the reference reducers write the reduction by the input
// shape, and Prepare never compares that with the output tensor, so every
// route checks here that the output holds exactly the input's elements on the
// axes the reduction keeps. An input above `max_rank` or an axis outside the
// input rank fails, as the reference's axis resolution does; a scalar input
// ignores its axes, as that resolution does. The input's nonzero extents must
// multiply to at most INT32_MAX even when another extent empties it: the bound
// the kernels' int dims need on a nonempty input, kept for empty ones too.
TfLiteStatus EnsureReducedOutputSize(const TfLiteEvalTensor* input,
                                     const TfLiteEvalTensor* axis,
                                     const TfLiteEvalTensor* output,
                                     int max_rank, const char* op) {
  const int rank = input->dims->size;
  const int num_axis =
      rank == 0 ? 0 : static_cast<int>(ElementCount(*axis->dims));
  const int32_t* axis_data = tflite::micro::GetTensorData<int32_t>(axis);
  bool valid = rank <= max_rank;
  for (int i = 0; valid && i < num_axis; ++i) {
    const int32_t a = axis_data[i] < 0 ? axis_data[i] + rank : axis_data[i];
    valid = a >= 0 && a < rank;
  }
  int64_t nonzero_elements = 1;
  int64_t expected = 1;
  for (int d = 0; valid && d < rank; ++d) {
    const int32_t extent = input->dims->data[d];
    bool reduced = false;
    for (int i = 0; !reduced && i < num_axis; ++i) {
      reduced = (axis_data[i] < 0 ? axis_data[i] + rank : axis_data[i]) == d;
    }
    if (extent < 0) {
      valid = false;
    } else if (extent > 0) {
      nonzero_elements *= extent;
      valid = nonzero_elements <= INT32_MAX;
    }
    if (!reduced) {
      expected *= extent;
    }
  }
  const int64_t actual =
      CheckedElementCount(output->dims->data, output->dims->size);
  if (!valid || expected != actual) {
    MicroPrintf("%s: the output size does not match the reduction.", op);
    return kTfLiteError;
  }
  return kTfLiteOk;
}

bool HasZeroExtent(const TfLiteEvalTensor* input) {
  for (int i = 0; i < input->dims->size; ++i) {
    if (input->dims->data[i] == 0) {
      return true;
    }
  }
  return false;
}

// An empty input reaches neither heliaCORE nor a reference loop: every output
// element, as sized by EnsureReducedOutputSize, gets the reduction's start
// value, as the reference reducers write it.
template <typename T>
TfLiteStatus FillEmptyReduction(TfLiteEvalTensor* output, T start) {
  T* data = tflite::micro::GetTensorData<T>(output);
  const int count = ElementCount(*output->dims);
  for (int i = 0; i < count; ++i) {
    data[i] = start;
  }
  return kTfLiteOk;
}

TfLiteStatus MinMaxStatus(arm_cmsis_nn_status status, const char* entry) {
  if (status != ARM_CMSIS_NN_SUCCESS) {
    MicroPrintf("REDUCE_MAX/REDUCE_MIN: %s failed (%d).", entry,
                static_cast<int>(status));
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus EvalMinMaxHelper(TfLiteContext* context, TfLiteNode* node,
                              OpDataReduce* op_data, MinMaxEvalType evalType) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TfLiteReducerParams* params =
      static_cast<TfLiteReducerParams*>(node->builtin_data);

  // Interpret an axis tensor with null dimensions as a scalar
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  TF_LITE_ENSURE_OK(context,
                    EnsureReducedOutputSize(
                        input, axis, output, kAnyReduceRank,
                        evalType == kEvalMax ? "REDUCE_MAX" : "REDUCE_MIN"));
  int* temp_buffer = static_cast<int*>(
      context->GetScratchBuffer(context, op_data->scratch_input_iter_idx));
  int* resolved_axis = static_cast<int*>(
      context->GetScratchBuffer(context, op_data->scratch_resolved_axis_idx));
  const int32_t* axis_data = tflite::micro::GetTensorData<int32_t>(axis);
  const bool is_max = evalType == kEvalMax;
  switch (input->type) {
    case kTfLiteFloat32: {
      if (HasZeroExtent(input)) {
        return FillEmptyReduction<float>(
            output, MinMaxReducerCompare<float>(evalType).initialValue());
      }
#if ARM_NN_ENABLE_F32
      if (HeliaFloatReduceSupported(input, axis_data, num_axis)) {
        cmsis_nn_dims in_d, ax_d, out_d;
        PopulateCmsisShapeAndAxis(input, axis_data, num_axis, &in_d, &ax_d,
                                  &out_d);
        const float* input_data = tflite::micro::GetTensorData<float>(input);
        float* output_data = tflite::micro::GetTensorData<float>(output);
        const arm_cmsis_nn_status status =
            is_max ? arm_reduce_max_tflite_f32(input_data, &in_d, &ax_d,
                                               output_data, &out_d)
                   : arm_reduce_min_tflite_f32(input_data, &in_d, &ax_d,
                                               output_data, &out_d);
        return MinMaxStatus(status, is_max ? "arm_reduce_max_tflite_f32"
                                           : "arm_reduce_min_tflite_f32");
      }
#endif
      MinMaxReducerCompare<float> reducer(evalType);
      TF_LITE_ENSURE(
          context,
          reference_ops::ReduceGeneric<float>(
              tflite::micro::GetTensorData<float>(input), input->dims->data,
              input->dims->size, tflite::micro::GetTensorData<float>(output),
              output->dims->data, output->dims->size,
              tflite::micro::GetTensorData<int>(axis), num_axis,
              params->keep_dims, temp_buffer, resolved_axis,
              reducer.initialValue(), reducer.compare()));
    } break;
#if ARM_NN_ENABLE_F16
    case kTfLiteFloat16: {
      if (!HeliaFloatReduceSupported(input, axis_data, num_axis)) {
        MicroPrintf(
            "Float16 REDUCE_MAX/REDUCE_MIN requires rank <= 4 and axes inside "
            "the rank.");
        return kTfLiteError;
      }
      // float16's lowest and largest finite values.
      if (HasZeroExtent(input)) {
        return FillEmptyReduction<float16_t>(
            output, static_cast<float16_t>(is_max ? -65504.0f : 65504.0f));
      }
      cmsis_nn_dims in_d, ax_d, out_d;
      PopulateCmsisShapeAndAxis(input, axis_data, num_axis, &in_d, &ax_d,
                                &out_d);
      const float16_t* input_data =
          tflite::micro::GetTensorData<float16_t>(input);
      float16_t* output_data = tflite::micro::GetTensorData<float16_t>(output);
      const arm_cmsis_nn_status status =
          is_max ? arm_reduce_max_tflite_f16(input_data, &in_d, &ax_d,
                                             output_data, &out_d)
                 : arm_reduce_min_tflite_f16(input_data, &in_d, &ax_d,
                                             output_data, &out_d);
      return MinMaxStatus(status, is_max ? "arm_reduce_max_tflite_f16"
                                         : "arm_reduce_min_tflite_f16");
    }
#endif
    case kTfLiteInt8: {
      // The kernels copy values without requantizing, as the reference does.
      TF_LITE_ENSURE_EQ(context, static_cast<double>(op_data->input_scale),
                        static_cast<double>(op_data->output_scale));
      TF_LITE_ENSURE_EQ(context, op_data->input_zp, op_data->output_zp);
      if (HasZeroExtent(input)) {
        return FillEmptyReduction<int8_t>(
            output, MinMaxReducerCompare<int8_t>(evalType).initialValue());
      }
      if (input->dims->size <= kCmsisRank) {
        cmsis_nn_dims in_d, ax_d, out_d;
        PopulateCmsisShapeAndAxis(input, axis_data, num_axis, &in_d, &ax_d,
                                  &out_d);
        const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
        int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);
        const arm_cmsis_nn_status status =
            is_max ? arm_reduce_max_s8(input_data, &in_d, &ax_d, output_data,
                                       &out_d)
                   : arm_reduce_min_s8(input_data, &in_d, &ax_d, output_data,
                                       &out_d);
        return MinMaxStatus(status,
                            is_max ? "arm_reduce_max_s8" : "arm_reduce_min_s8");
      }

      MinMaxReducerCompare<int8_t> reducer(evalType);
      TF_LITE_ENSURE(
          context,
          reference_ops::ReduceGeneric<int8_t>(
              tflite::micro::GetTensorData<int8_t>(input), input->dims->data,
              input->dims->size, tflite::micro::GetTensorData<int8_t>(output),
              output->dims->data, output->dims->size,
              tflite::micro::GetTensorData<int>(axis), num_axis,
              params->keep_dims, temp_buffer, resolved_axis,
              reducer.initialValue(), reducer.compare()));
    } break;

    case kTfLiteInt16: {
      TF_LITE_ENSURE_EQ(context, static_cast<double>(op_data->input_scale),
                        static_cast<double>(op_data->output_scale));
      TF_LITE_ENSURE_EQ(context, op_data->input_zp, op_data->output_zp);
      if (HasZeroExtent(input)) {
        return FillEmptyReduction<int16_t>(
            output, MinMaxReducerCompare<int16_t>(evalType).initialValue());
      }
      if (input->dims->size <= kCmsisRank) {
        cmsis_nn_dims in_d, ax_d, out_d;
        PopulateCmsisShapeAndAxis(input, axis_data, num_axis, &in_d, &ax_d,
                                  &out_d);
        const int16_t* input_data =
            tflite::micro::GetTensorData<int16_t>(input);
        int16_t* output_data = tflite::micro::GetTensorData<int16_t>(output);
        const arm_cmsis_nn_status status =
            is_max ? arm_reduce_max_s16(input_data, &in_d, &ax_d, output_data,
                                        &out_d)
                   : arm_reduce_min_s16(input_data, &in_d, &ax_d, output_data,
                                        &out_d);
        return MinMaxStatus(status, is_max ? "arm_reduce_max_s16"
                                           : "arm_reduce_min_s16");
      }

      MinMaxReducerCompare<int16_t> reducer(evalType);
      TF_LITE_ENSURE(
          context,
          reference_ops::ReduceGeneric<int16_t>(
              tflite::micro::GetTensorData<int16_t>(input), input->dims->data,
              input->dims->size, tflite::micro::GetTensorData<int16_t>(output),
              output->dims->data, output->dims->size,
              tflite::micro::GetTensorData<int>(axis), num_axis,
              params->keep_dims, temp_buffer, resolved_axis,
              reducer.initialValue(), reducer.compare()));
    } break;

    default:
      MicroPrintf("Only float16, float32, int8, and int16 types are supported.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteStatus PrepareMinMaxHelper(TfLiteContext* context, TfLiteNode* node,
                                 OpDataReduce* op_data) {
  TF_LITE_ENSURE_OK(context, PrepareSimple(context, node, &op_data->multiplier,
                                           &op_data->shift));

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TfLiteTensor* axis = micro_context->AllocateTempInputTensor(node, 1);

  if (input->type == kTfLiteFloat16 &&
      PrepareFloat16Reduce(context, input, axis, "REDUCE_MAX/REDUCE_MIN") !=
          kTfLiteOk) {
    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(output);
    micro_context->DeallocateTempTfLiteTensor(axis);
    return kTfLiteError;
  }

  op_data->input_zp = input->params.zero_point;
  op_data->input_scale = input->params.scale;
  op_data->output_zp = output->params.zero_point;
  op_data->output_scale = output->params.scale;
  op_data->num_output_elements = NumElements(output);

  context->RequestScratchBufferInArena(context, sizeof(int) * input->dims->size,
                                       &op_data->scratch_input_iter_idx);
  context->RequestScratchBufferInArena(
      context, sizeof(int) * static_cast<int>(ElementCount(*axis->dims)),
      &op_data->scratch_resolved_axis_idx);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(axis);
  return kTfLiteOk;
}

TfLiteStatus PrepareMeanOrSumHelper(TfLiteContext* context, TfLiteNode* node,
                                    OpDataReduce* op_data) {
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TfLiteTensor* axis = micro_context->AllocateTempInputTensor(node, 1);

  if (input->type == kTfLiteFloat16 &&
      PrepareFloat16Reduce(context, input, axis, "MEAN/REDUCE_SUM") !=
          kTfLiteOk) {
    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(output);
    micro_context->DeallocateTempTfLiteTensor(axis);
    return kTfLiteError;
  }

  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    const double real_multiplier = static_cast<double>(input->params.scale) /
                                   static_cast<double>(output->params.scale);
    QuantizeMultiplier(real_multiplier, &op_data->multiplier, &op_data->shift);
  }

  int output_size = NumElements(output);
  op_data->num_axis = NumElements(axis);

  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    context->RequestScratchBufferInArena(context, output_size * sizeof(int32_t),
                                         &op_data->scratch_accumulator_idx);
    op_data->input_zp = input->params.zero_point;
    op_data->input_scale = input->params.scale;
    op_data->output_zp = output->params.zero_point;
    op_data->output_scale = output->params.scale;
  }

  TF_LITE_ENSURE_OK(
      context,
      PrepareSimple(context, node, &(op_data->multiplier), &(op_data->shift)));
  // TODO(b/144955155): Support uint8_t(b/144955155) and int8_t(b/144955018)
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(axis);
  return kTfLiteOk;
}

TfLiteStatus EvalMeanHelper(TfLiteContext* context, TfLiteNode* node,
                            OpDataReduce* op_data) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TfLiteReducerParams* params =
      reinterpret_cast<TfLiteReducerParams*>(node->builtin_data);

  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_OK(context, EnsureReducedOutputSize(input, axis, output,
                                                     kMaxNumberOfAxis, "MEAN"));
  int temp_index[kMaxNumberOfAxis];
  int resolved_axis[kMaxNumberOfAxis];

  switch (input->type) {
    case kTfLiteFloat16: {
#if ARM_NN_ENABLE_F16
      if (HasZeroExtent(input)) {
        return FillEmptyReduction<float16_t>(output,
                                             static_cast<float16_t>(0.0f));
      }
      if (HeliaFloatReduceSupported(
              input, tflite::micro::GetTensorData<int32_t>(axis), num_axis)) {
        cmsis_nn_dims in_d, ax_d, out_d;
        PopulateCmsisShapeAndAxis(input,
                                  tflite::micro::GetTensorData<int32_t>(axis),
                                  num_axis, &in_d, &ax_d, &out_d);
        if (arm_nn_mean_f16(tflite::micro::GetTensorData<float16_t>(input),
                            &in_d, &ax_d,
                            tflite::micro::GetTensorData<float16_t>(output),
                            &out_d) == ARM_CMSIS_NN_SUCCESS) {
          break;
        }
      }
#endif
      MicroPrintf(
          "Float16 MEAN: optimized kernel rejected the configuration; it "
          "requires rank <= 4, axes inside the rank, and ARM_NN_ENABLE_F16.");
      return kTfLiteError;
    }
    case kTfLiteFloat32: {
      if (HasZeroExtent(input)) {
        return FillEmptyReduction<float>(output, 0.0f);
      }
#if ARM_NN_ENABLE_F32
      if (HeliaFloatReduceSupported(
              input, tflite::micro::GetTensorData<int32_t>(axis), num_axis)) {
        cmsis_nn_dims in_d, ax_d, out_d;
        PopulateCmsisShapeAndAxis(input,
                                  tflite::micro::GetTensorData<int32_t>(axis),
                                  num_axis, &in_d, &ax_d, &out_d);
        if (arm_nn_mean_f32(tflite::micro::GetTensorData<float>(input), &in_d,
                            &ax_d, tflite::micro::GetTensorData<float>(output),
                            &out_d) == ARM_CMSIS_NN_SUCCESS) {
          break;
        }
      }
#endif
      tflite::MeanParams op_params;
      ResolveAxis(tflite::micro::GetTensorData<int>(axis), num_axis,
                  &op_params);

      // Special case mean implementation exists for 4D mean across axes 1
      // and 2.
      bool special_case_4d_axes_1_and_2 =
          input->dims->size == 4 && op_params.axis_count == 2 &&
          ((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
           (op_params.axis[0] == 2 && op_params.axis[1] == 1));

      // Defer to specialized implementation for 4D Mean across axes 1 & 2.
      if (params->keep_dims && special_case_4d_axes_1_and_2) {
        reference_ops::Mean(op_params, tflite::micro::GetTensorShape(input),
                            tflite::micro::GetTensorData<float>(input),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<float>(output));
      } else {
        TF_LITE_ENSURE(
            context,
            reference_ops::Mean(
                tflite::micro::GetTensorData<float>(input), input->dims->data,
                input->dims->size, tflite::micro::GetTensorData<float>(output),
                output->dims->data, output->dims->size,
                tflite::micro::GetTensorData<int>(axis), num_axis,
                params->keep_dims, temp_index, resolved_axis,
                tflite::micro::GetTensorData<float>(output)));
      }
    } break;
    case kTfLiteInt8: {
      // The reference writes T(), not the zero point, for an empty input.
      if (HasZeroExtent(input)) {
        return FillEmptyReduction<int8_t>(output, 0);
      }
      if (input->dims->size <= kCmsisRank) {
        cmsis_nn_dims in_d, ax_d, out_d;
        int32_t output_mult = op_data->multiplier;
        int32_t output_shift = op_data->shift;

        PopulateCmsisParams(
          input, output,
          tflite::micro::GetTensorData<int32_t>(axis),
          num_axis,
          &in_d, &ax_d, &out_d,
          &output_mult, &output_shift,
          /*compute_sum=*/false
        );

        arm_cmsis_nn_status rst = arm_mean_s8(
          tflite::micro::GetTensorData<int8_t>(input),
          &in_d,
          -op_data->input_zp,
          &ax_d,
          tflite::micro::GetTensorData<int8_t>(output),
          &out_d,
          op_data->output_zp,
          output_mult,
          output_shift
        );
        return rst == ARM_CMSIS_NN_SUCCESS ? kTfLiteOk : kTfLiteError;
      }
      TF_LITE_ENSURE_OK(
          context, EvalIntegerMean<int8_t>(context, node, num_axis, op_data,
                                           temp_index, resolved_axis));
    } break;
    case kTfLiteInt16: {
      if (HasZeroExtent(input)) {
        return FillEmptyReduction<int16_t>(output, 0);
      }
      if (input->dims->size <= kCmsisRank) {
        cmsis_nn_dims in_d, ax_d, out_d;
        int32_t output_mult = op_data->multiplier;
        int32_t output_shift = op_data->shift;

        PopulateCmsisParams(
          input, output,
          tflite::micro::GetTensorData<int32_t>(axis),
          num_axis,
          &in_d, &ax_d, &out_d,
          &output_mult, &output_shift,
          /*compute_sum=*/false
        );

        arm_cmsis_nn_status rst = arm_mean_s16(
          tflite::micro::GetTensorData<int16_t>(input),
          &in_d,
          -op_data->input_zp,
          &ax_d,
          tflite::micro::GetTensorData<int16_t>(output),
          &out_d,
          op_data->output_zp,
          output_mult,
          output_shift
        );
        return rst == ARM_CMSIS_NN_SUCCESS ? kTfLiteOk : kTfLiteError;
      }
      TF_LITE_ENSURE_OK(
          context, EvalIntegerMean<int16_t>(context, node, num_axis, op_data,
                                            temp_index, resolved_axis));
    } break;
    default:
      TF_LITE_ENSURE_MSG(context, false,
                         "Currently, only float16, float32, int8 or int16 "
                         "input type is supported.");
  }
  return kTfLiteOk;
}

TfLiteStatus EvalMaxHelper(TfLiteContext* context, TfLiteNode* node,
                           OpDataReduce* op_data) {
  return EvalMinMaxHelper(context, node, op_data, kEvalMax);
}

TfLiteStatus EvalMinHelper(TfLiteContext* context, TfLiteNode* node,
                           OpDataReduce* op_data) {
  return EvalMinMaxHelper(context, node, op_data, kEvalMin);
}

TfLiteStatus EvalSumHelper(TfLiteContext* context, TfLiteNode* node,
                           OpDataReduce* op_data) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TfLiteReducerParams* params =
      static_cast<TfLiteReducerParams*>(node->builtin_data);

  // Interpret an axis tensor with null dimensions as a scalar.
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  TF_LITE_ENSURE_OK(context, EnsureReducedOutputSize(input, axis, output,
                                                     kMaxNumberOfAxis, "SUM"));
  int temp_index[kMaxNumberOfAxis];
  int resolved_axis[kMaxNumberOfAxis];

  switch (input->type) {
    case kTfLiteFloat16: {
#if ARM_NN_ENABLE_F16
      if (HasZeroExtent(input)) {
        return FillEmptyReduction<float16_t>(output,
                                             static_cast<float16_t>(0.0f));
      }
      if (HeliaFloatReduceSupported(
              input, tflite::micro::GetTensorData<int32_t>(axis), num_axis)) {
        cmsis_nn_dims in_d, ax_d, out_d;
        PopulateCmsisShapeAndAxis(input,
                                  tflite::micro::GetTensorData<int32_t>(axis),
                                  num_axis, &in_d, &ax_d, &out_d);
        if (arm_reduce_sum_f16(tflite::micro::GetTensorData<float16_t>(input),
                               &in_d, &ax_d,
                               tflite::micro::GetTensorData<float16_t>(output),
                               &out_d) == ARM_CMSIS_NN_SUCCESS) {
          break;
        }
      }
#endif
      MicroPrintf(
          "Float16 REDUCE_SUM: optimized kernel rejected the configuration; it "
          "requires rank <= 4, axes inside the rank, and ARM_NN_ENABLE_F16.");
      return kTfLiteError;
    }
    case kTfLiteFloat32: {
      if (HasZeroExtent(input)) {
        return FillEmptyReduction<float>(output, 0.0f);
      }
#if ARM_NN_ENABLE_F32
      if (HeliaFloatReduceSupported(
              input, tflite::micro::GetTensorData<int32_t>(axis), num_axis)) {
        cmsis_nn_dims in_d, ax_d, out_d;
        PopulateCmsisShapeAndAxis(input,
                                  tflite::micro::GetTensorData<int32_t>(axis),
                                  num_axis, &in_d, &ax_d, &out_d);
        if (arm_reduce_sum_f32(tflite::micro::GetTensorData<float>(input),
                               &in_d, &ax_d,
                               tflite::micro::GetTensorData<float>(output),
                               &out_d) == ARM_CMSIS_NN_SUCCESS) {
          break;
        }
      }
#endif
      TF_LITE_ENSURE(
          context,
          reference_ops::ReduceGeneric<float>(
              tflite::micro::GetTensorData<float>(input), input->dims->data,
              input->dims->size, tflite::micro::GetTensorData<float>(output),
              output->dims->data, output->dims->size,
              tflite::micro::GetTensorData<int>(axis), num_axis,
              params->keep_dims, temp_index, resolved_axis, /*init_value=*/0.f,
              [](const float current, const float in) -> float {
                return in + current;
              }));
    } break;
    case kTfLiteInt8: {
      if (HasZeroExtent(input)) {
        return FillEmptyReduction<int8_t>(output, 0);
      }
      if (input->dims->size <= kCmsisRank) {
        cmsis_nn_dims in_d, ax_d, out_d;
        int32_t output_mult = op_data->multiplier;
        int32_t output_shift = op_data->shift;

        PopulateCmsisParams(
          input, output,
          tflite::micro::GetTensorData<int32_t>(axis),
          num_axis,
          &in_d, &ax_d, &out_d,
          &output_mult, &output_shift,
          /*compute_sum=*/true
        );

        arm_cmsis_nn_status rst = arm_mean_s8(
          tflite::micro::GetTensorData<int8_t>(input),
          &in_d,
          -op_data->input_zp,
          &ax_d,
          tflite::micro::GetTensorData<int8_t>(output),
          &out_d,
          op_data->output_zp,
          output_mult,
          output_shift
        );
        return rst == ARM_CMSIS_NN_SUCCESS ? kTfLiteOk : kTfLiteError;
      }

      int32_t* temp_sum = static_cast<int32_t*>(
          context->GetScratchBuffer(context, op_data->scratch_accumulator_idx));
      TF_LITE_ENSURE_OK(context, QuantizedMeanOrSum<int8_t>(
                                     context, node, temp_index, resolved_axis,
                                     temp_sum, op_data, /*compute_sum=*/true));
    } break;
    case kTfLiteInt16: {
      if (HasZeroExtent(input)) {
        return FillEmptyReduction<int16_t>(output, 0);
      }
      if (input->dims->size <= kCmsisRank) {
        cmsis_nn_dims in_d, ax_d, out_d;
        int32_t output_mult = op_data->multiplier;
        int32_t output_shift = op_data->shift;

        PopulateCmsisParams(
          input, output,
          tflite::micro::GetTensorData<int32_t>(axis),
          num_axis,
          &in_d, &ax_d, &out_d,
          &output_mult, &output_shift,
          /*compute_sum=*/true
        );

        arm_cmsis_nn_status rst = arm_mean_s16(
          tflite::micro::GetTensorData<int16_t>(input),
          &in_d,
          -op_data->input_zp,
          &ax_d,
          tflite::micro::GetTensorData<int16_t>(output),
          &out_d,
          op_data->output_zp,
          output_mult,
          output_shift
        );
        return rst == ARM_CMSIS_NN_SUCCESS ? kTfLiteOk : kTfLiteError;
      }

      int32_t* temp_sum = static_cast<int32_t*>(
          context->GetScratchBuffer(context, op_data->scratch_accumulator_idx));
      TF_LITE_ENSURE_OK(context, QuantizedMeanOrSum<int16_t>(
                                     context, node, temp_index, resolved_axis,
                                     temp_sum, op_data, /*compute_sum=*/true));
    } break;
    default:
      MicroPrintf(
          "Only float16, float32, int8, and int16 types are supported.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

// Reduce-all (boolean reduction) ported from upstream tflm reduce_common.cc:
// helia adopts the unmodified reference path because there is no CMSIS-NN
// accelerator for boolean reductions. Lives here (rather than upstream)
// because helia overrides the entire reduce_common.cc translation unit via
// `specialize_files.py`, so the upstream copy is not linked.
TfLiteStatus PrepareAllHelper(TfLiteContext* context, TfLiteNode* node,
                              OpDataReduce* op_data) {
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TfLiteTensor* axis = micro_context->AllocateTempInputTensor(node, 1);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteBool);

  TF_LITE_ENSURE_OK(context, PrepareSimple(context, node, &op_data->multiplier,
                                           &op_data->shift));

  op_data->input_zp = input->params.zero_point;
  op_data->input_scale = input->params.scale;
  op_data->output_zp = output->params.zero_point;
  op_data->output_scale = output->params.scale;
  op_data->num_output_elements = NumElements(output);
  context->RequestScratchBufferInArena(context, sizeof(int) * input->dims->size,
                                       &op_data->scratch_input_iter_idx);
  context->RequestScratchBufferInArena(
      context, sizeof(int) * static_cast<int>(ElementCount(*axis->dims)),
      &op_data->scratch_resolved_axis_idx);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(axis);
  return kTfLiteOk;
}

TfLiteStatus EvalAllHelper(TfLiteContext* context, TfLiteNode* node,
                           OpDataReduce* op_data) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TfLiteReducerParams* params =
      static_cast<TfLiteReducerParams*>(node->builtin_data);

  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  TF_LITE_ENSURE_OK(context,
                    EnsureReducedOutputSize(input, axis, output, kAnyReduceRank,
                                            "REDUCE_ALL"));
  int* input_iter = static_cast<int*>(
      context->GetScratchBuffer(context, op_data->scratch_input_iter_idx));
  int* resolved_axis = static_cast<int*>(
      context->GetScratchBuffer(context, op_data->scratch_resolved_axis_idx));
  switch (input->type) {
    case kTfLiteBool:
      TF_LITE_ENSURE(
          context,
          reference_ops::ReduceGeneric<bool>(
              tflite::micro::GetTensorData<bool>(input), input->dims->data,
              input->dims->size, tflite::micro::GetTensorData<bool>(output),
              output->dims->data, output->dims->size,
              tflite::micro::GetTensorData<int>(axis), num_axis,
              params->keep_dims, input_iter, resolved_axis, true,
              [](const bool current, const bool in) -> bool {
                return in && current;
              }));
      break;
    default:
      MicroPrintf("Only Bool is accepted as input type.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace micro
}  // namespace tflite
