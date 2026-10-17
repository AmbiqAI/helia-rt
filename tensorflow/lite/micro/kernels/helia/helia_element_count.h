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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_ELEMENT_COUNT_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_ELEMENT_COUNT_H_

#include <climits>
#include <cstdint>

#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace micro {

// Element count of a tensor for a heliaCORE elementwise entry, which writes
// as many output elements as it reads. Fails on a null or negative-rank dims,
// a negative extent, or a count above INT32_MAX; a zero extent gives 0.
inline TfLiteStatus HeliaCheckedElementCount(TfLiteContext* context,
                                             const TfLiteIntArray* dims,
                                             int32_t* count) {
  TF_LITE_ENSURE(context, dims != nullptr && count != nullptr);
  TF_LITE_ENSURE(context, dims->size >= 0);
  bool empty = false;
  for (int i = 0; i < dims->size; ++i) {
    TF_LITE_ENSURE(context, dims->data[i] >= 0);
    empty = empty || dims->data[i] == 0;
  }
  if (empty) {
    *count = 0;
    return kTfLiteOk;
  }
  int32_t elements = 1;
  for (int i = 0; i < dims->size; ++i) {
    TF_LITE_ENSURE(context, dims->data[i] <= INT32_MAX / elements);
    elements *= dims->data[i];
  }
  *count = elements;
  return kTfLiteOk;
}

// Checked element count of an input and output that must hold the same
// number of elements; their shapes may differ.
inline TfLiteStatus HeliaCheckedMatchingElementCount(
    TfLiteContext* context, const TfLiteIntArray* input_dims,
    const TfLiteIntArray* output_dims, int32_t* count) {
  int32_t output_count;
  TF_LITE_ENSURE_OK(context,
                    HeliaCheckedElementCount(context, input_dims, count));
  TF_LITE_ENSURE_OK(
      context, HeliaCheckedElementCount(context, output_dims, &output_count));
  TF_LITE_ENSURE_EQ(context, *count, output_count);
  return kTfLiteOk;
}

// Checked element count of an input and output that must have the same rank
// and extents, as the float16 elementwise kernels require.
inline TfLiteStatus HeliaCheckedSameShapeElementCount(
    TfLiteContext* context, const TfLiteIntArray* input_dims,
    const TfLiteIntArray* output_dims, int32_t* count) {
  TF_LITE_ENSURE(context, input_dims != nullptr && output_dims != nullptr);
  TF_LITE_ENSURE_EQ(context, input_dims->size, output_dims->size);
  for (int i = 0; i < input_dims->size; ++i) {
    TF_LITE_ENSURE_EQ(context, input_dims->data[i], output_dims->data[i]);
  }
  return HeliaCheckedElementCount(context, input_dims, count);
}

}  // namespace micro
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_ELEMENT_COUNT_H_
