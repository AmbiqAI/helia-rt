/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_DATA_MOVEMENT_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_DATA_MOVEMENT_H_

#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/helia/helia_float_common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

struct HeliaDataMovementData {
  int axis;
  int count;
  int32_t elements;
  int pointer_scratch;
  int32_t* split_sizes;
  int32_t* shape;
};

inline void* InitHeliaDataMovement(TfLiteContext* context, const char*,
                                   size_t) {
  auto* data =
      static_cast<HeliaDataMovementData*>(context->AllocatePersistentBuffer(
          context, sizeof(HeliaDataMovementData)));
  if (data != nullptr) *data = {0, 0, 0, -1, nullptr, nullptr};
  return data;
}

inline bool IsHeliaDataMovementFloat(TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteFloat16;
}

inline TfLiteStatus CheckHeliaDataMovementType(TfLiteContext* context,
                                               TfLiteType type) {
  TF_LITE_ENSURE_MSG(context, type != kTfLiteFloat16 || kHeliaFloat16Enabled,
                     "FP16 data movement requires ARM_NN_ENABLE_F16=1");
  return kTfLiteOk;
}

inline TfLiteStatus HeliaDataMovementElements(TfLiteContext* context,
                                              const TfLiteIntArray* dims,
                                              TfLiteType type, int32_t* count) {
  TF_LITE_ENSURE(context, dims != nullptr && dims->size >= 0);
  size_t elements = 1;
  const size_t width = type == kTfLiteFloat16 ? 2 : 4;
  const size_t max_elements_by_width =
      std::numeric_limits<size_t>::max() / width;
  const size_t max_elements =
      max_elements_by_width < static_cast<size_t>(INT32_MAX)
          ? max_elements_by_width
          : static_cast<size_t>(INT32_MAX);
  for (int i = 0; i < dims->size; ++i) {
    TF_LITE_ENSURE(context, dims->data[i] >= 0);
    const size_t extent = dims->data[i];
    TF_LITE_ENSURE(context, extent == 0 || elements <= max_elements / extent);
    elements *= extent;
  }
  *count = static_cast<int32_t>(elements);
  return kTfLiteOk;
}

inline TfLiteStatus HeliaDataMovementShape(TfLiteContext* context,
                                           const TfLiteIntArray* dims,
                                           HeliaDataMovementData* data) {
  if (data->elements == 0 || dims->size == 0) return kTfLiteOk;
  TF_LITE_ENSURE(context,
                 static_cast<size_t>(dims->size) <=
                     std::numeric_limits<size_t>::max() / sizeof(int32_t));
  data->shape = static_cast<int32_t*>(context->AllocatePersistentBuffer(
      context, static_cast<size_t>(dims->size) * sizeof(int32_t)));
  TF_LITE_ENSURE(context, data->shape != nullptr);
  // TfLite dimensions are int; some Arm toolchains define int32_t as long.
  for (int i = 0; i < dims->size; ++i) data->shape[i] = dims->data[i];
  return kTfLiteOk;
}

inline TfLiteStatus HeliaDataMovementPointers(TfLiteContext* context,
                                              HeliaDataMovementData* data) {
  if (data->elements == 0) return kTfLiteOk;
  TF_LITE_ENSURE(context, data->count > 0);
  TF_LITE_ENSURE(context,
                 static_cast<size_t>(data->count) <=
                     std::numeric_limits<size_t>::max() / sizeof(void*));
  return context->RequestScratchBufferInArena(
      context, static_cast<size_t>(data->count) * sizeof(void*),
      &data->pointer_scratch);
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_DATA_MOVEMENT_H_
