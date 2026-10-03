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
// Writes the computed batch into an output that stores the placeholder batch
// 1 of a dynamic-batch export; any other batch mismatch is rejected.
TfLiteStatus SetComputedBatch(TfLiteContext* context, TfLiteNode* node,
                              TfLiteTensor* output, int64_t batch) {
  TF_LITE_ENSURE(context, batch > 0 && batch <= INT32_MAX);
  if (output->dims->data[0] == batch) {
    return kTfLiteOk;
  }
  if (output->dims->data[0] != 1) {
    MicroPrintf("Output batch %d does not match the computed batch %d.",
                output->dims->data[0], static_cast<int>(batch));
    return kTfLiteError;
  }
  TfLiteEvalTensor* output_eval =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_OK(context, tflite::micro::CreateWritableTensorDimsWithCopy(
                                 context, output, output_eval));
  output->dims->data[0] = static_cast<int>(batch);
  return kTfLiteOk;
}

// helia: see AmbiqAI/helia-rt#407.
// With a constant block shape and crops, the output shape is fully known at
// Prepare: check every non-batch dim and set the batch. An input batch that
// the block product does not divide means an op between SPACE_TO_BATCH_ND
// and here did not carry the batch forward, so the model is rejected.
TfLiteStatus CheckConstantOutputShape(TfLiteContext* context, TfLiteNode* node,
                                      const TfLiteTensor* input,
                                      const TfLiteTensor* block_shape,
                                      const TfLiteTensor* crops,
                                      TfLiteTensor* output) {
  const int rank = NumDimensions(input);
  const int spatial_dims = rank - 2;
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), rank);
  TF_LITE_ENSURE_TYPES_EQ(context, block_shape->type, kTfLiteInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, crops->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumElements(block_shape), spatial_dims);
  TF_LITE_ENSURE_EQ(context, NumElements(crops), spatial_dims * 2);
  const int32_t* block = GetTensorData<int32_t>(block_shape);
  const int32_t* crop = GetTensorData<int32_t>(crops);
  int64_t block_product = 1;
  for (int i = 0; i < spatial_dims; ++i) {
    TF_LITE_ENSURE(context, block[i] > 0);
    TF_LITE_ENSURE(context, crop[2 * i] >= 0 && crop[2 * i + 1] >= 0);
    const int64_t uncropped =
        static_cast<int64_t>(input->dims->data[i + 1]) * block[i];
    TF_LITE_ENSURE_EQ(context, output->dims->data[i + 1],
                      uncropped - crop[2 * i] - crop[2 * i + 1]);
    block_product *= block[i];
  }
  TF_LITE_ENSURE_EQ(context, output->dims->data[rank - 1],
                    input->dims->data[rank - 1]);
  if (input->dims->data[0] % block_product != 0) {
    MicroPrintf(
        "BATCH_TO_SPACE_ND: input batch %d is not a multiple of the block "
        "product %d.",
        input->dims->data[0], static_cast<int>(block_product));
    return kTfLiteError;
  }
  return SetComputedBatch(context, node, output,
                          input->dims->data[0] / block_product);
}

// helia: see AmbiqAI/helia-rt#407.
// The shape is only checked when the block shape and crops are constant; it
// cannot be known at Prepare otherwise.
TfLiteStatus CheckOutputShape(TfLiteContext* context, TfLiteNode* node,
                              const TfLiteTensor* input, TfLiteTensor* output) {
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* block_shape =
      micro_context->AllocateTempInputTensor(node, kBlockShapeTensor);
  TfLiteTensor* crops =
      micro_context->AllocateTempInputTensor(node, kCropsTensor);
  TfLiteStatus status = kTfLiteError;
  if (block_shape != nullptr && crops != nullptr) {
    status = kTfLiteOk;
    if (IsConstantTensor(block_shape) && IsConstantTensor(crops)) {
      status = CheckConstantOutputShape(context, node, input, block_shape,
                                        crops, output);
    }
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

  // helia: see AmbiqAI/helia-rt#407.
  if (CheckOutputShape(context, node, input, output) != kTfLiteOk) {
    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(output);
    return kTfLiteError;
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
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
