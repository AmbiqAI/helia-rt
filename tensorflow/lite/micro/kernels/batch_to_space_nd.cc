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

#include "tensorflow/lite/kernels/internal/reference/batch_to_space_nd.h"

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

namespace {

constexpr int kInputTensor = 0;
constexpr int kBlockShapeTensor = 1;
constexpr int kCropsTensor = 2;
constexpr int kOutputTensor = 0;

// Currently, only 3D NHC and 4D NHWC input/output op_context are supported.
// In case of 3D input, it will be extended to 3D NHWC by adding W=1.
// The 4D array need to have exactly 2 spatial dimensions.
// TODO(b/149952582): Support arbitrary dimension in SpaceToBatchND.
const int kInputOutputMinDimensionNum = 3;
const int kInputOutputMaxDimensionNum = 4;

// helia: see AmbiqAI/helia-rt#407.
// TFLM sizes every tensor from the shape stored in the model. A model
// exported with a dynamic batch stores placeholder batches that do not follow
// from the graph, so it would compute wrong outputs; reject any output shape
// the input, block shape and crops do not give. With runtime crops (nullptr
// here), only the batch is checked.
TfLiteStatus CheckConstantOutputShape(const TfLiteTensor* input,
                                      const TfLiteTensor* block_shape,
                                      const TfLiteTensor* crops,
                                      const TfLiteTensor* output) {
  const int rank = NumDimensions(input);
  const int spatial_dims = rank - 2;
  if (NumDimensions(output) != rank || block_shape->type != kTfLiteInt32 ||
      NumElements(block_shape) != spatial_dims ||
      (crops != nullptr && (crops->type != kTfLiteInt32 ||
                            NumElements(crops) != 2 * spatial_dims))) {
    MicroPrintf("BATCH_TO_SPACE_ND: unsupported block shape or crops.");
    return kTfLiteError;
  }
  const int32_t* block = GetTensorData<int32_t>(block_shape);
  const int32_t* crop =
      crops != nullptr ? GetTensorData<int32_t>(crops) : nullptr;
  bool valid = input->dims->data[0] > 0;
  for (int i = 1; i < rank; ++i) {
    valid = valid && input->dims->data[i] >= 0;
  }
  int64_t expected[kInputOutputMaxDimensionNum];
  int64_t block_product = 1;
  for (int i = 0; valid && i < spatial_dims; ++i) {
    valid = block[i] > 0;
    block_product *= block[i];
    valid = valid && block_product <= INT32_MAX;
    if (valid && crop != nullptr) {
      expected[i + 1] =
          static_cast<int64_t>(input->dims->data[i + 1]) * block[i] -
          crop[2 * i] - crop[2 * i + 1];
      valid = crop[2 * i] >= 0 && crop[2 * i + 1] >= 0 &&
              expected[i + 1] >= 0 && expected[i + 1] <= INT32_MAX;
    }
  }
  if (!valid) {
    MicroPrintf(
        "BATCH_TO_SPACE_ND: invalid input shape, block shape or crops.");
    return kTfLiteError;
  }
  const int32_t product = static_cast<int32_t>(block_product);
  if (input->dims->data[0] % product != 0) {
    MicroPrintf(
        "BATCH_TO_SPACE_ND: input batch %d is not a multiple of the block "
        "product %d. The model was likely exported with a dynamic batch; "
        "re-export it with a fixed batch size (for example 1).",
        input->dims->data[0], static_cast<int>(product));
    return kTfLiteError;
  }
  expected[0] = input->dims->data[0] / product;
  expected[rank - 1] = input->dims->data[rank - 1];
  const int checked_dims = crop != nullptr ? rank : 1;
  for (int i = 0; i < checked_dims; ++i) {
    if (output->dims->data[i] != expected[i]) {
      MicroPrintf(
          "BATCH_TO_SPACE_ND: output dim %d is %d in the model, but the "
          "input, block shape and crops give %d.%s",
          i, output->dims->data[i], static_cast<int>(expected[i]),
          i == 0 ? " The model was likely exported with a dynamic batch; "
                   "re-export it with a fixed batch size (for example 1)."
                 : "");
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

// helia: see AmbiqAI/helia-rt#407.
// The shape is only checked when the block shape is constant; otherwise it is
// not known before Eval.
TfLiteStatus CheckOutputShape(TfLiteContext* context, TfLiteNode* node,
                              const TfLiteTensor* input,
                              const TfLiteTensor* output) {
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* block_shape =
      micro_context->AllocateTempInputTensor(node, kBlockShapeTensor);
  TfLiteTensor* crops =
      micro_context->AllocateTempInputTensor(node, kCropsTensor);
  TfLiteStatus status = kTfLiteOk;
  if (block_shape == nullptr || crops == nullptr) {
    status = kTfLiteError;
  } else if (IsConstantTensor(block_shape)) {
    status = CheckConstantOutputShape(
        input, block_shape, IsConstantTensor(crops) ? crops : nullptr, output);
  }
  if (block_shape != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(block_shape);
  }
  if (crops != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(crops);
  }
  return status;
}

TfLiteStatus BatchToSpaceNDPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, input != nullptr && output != nullptr);

  TF_LITE_ENSURE(context, NumDimensions(input) >= kInputOutputMinDimensionNum);
  TF_LITE_ENSURE(context, NumDimensions(output) >= kInputOutputMinDimensionNum);
  TF_LITE_ENSURE(context, NumDimensions(input) <= kInputOutputMaxDimensionNum);
  TF_LITE_ENSURE(context, NumDimensions(output) <= kInputOutputMaxDimensionNum);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  const TfLiteStatus status = CheckOutputShape(context, node, input, output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return status;
}

TfLiteStatus BatchToSpaceNDEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* block_shape =
      tflite::micro::GetEvalInput(context, node, kBlockShapeTensor);
  const TfLiteEvalTensor* crops =
      tflite::micro::GetEvalInput(context, node, kCropsTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      reference_ops::BatchToSpaceND(
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(block_shape),
          tflite::micro::GetTensorData<int32_t>(block_shape),
          tflite::micro::GetTensorShape(crops),
          tflite::micro::GetTensorData<int32_t>(crops),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      reference_ops::BatchToSpaceND(
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(block_shape),
          tflite::micro::GetTensorData<int32_t>(block_shape),
          tflite::micro::GetTensorShape(crops),
          tflite::micro::GetTensorData<int32_t>(crops),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
      break;
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace.

TFLMRegistration Register_BATCH_TO_SPACE_ND() {
  return tflite::micro::RegisterOp(nullptr, BatchToSpaceNDPrepare,
                                   BatchToSpaceNDEval);
}

}  // namespace tflite
