/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_MEMORY_HELPERS_H_
#define TENSORFLOW_LITE_MICRO_MEMORY_HELPERS_H_

#include <climits>
#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// Returns the next pointer address aligned to the given alignment.
uint8_t* AlignPointerUp(uint8_t* data, size_t alignment);

// Returns the previous pointer address aligned to the given alignment.
uint8_t* AlignPointerDown(uint8_t* data, size_t alignment);

// Returns an increased size that's a multiple of alignment.
size_t AlignSizeUp(size_t size, size_t alignment);

// Templated version of AlignSizeUp
// Returns an increased size that's a multiple of alignment.
template <typename T>
size_t AlignSizeUp(size_t count = 1) {
  return AlignSizeUp(sizeof(T) * count, alignof(T));
}

// Returns size in bytes for a given TfLiteType.
TfLiteStatus TfLiteTypeSizeOf(TfLiteType type, size_t* size);

// TODO(AmbiqAI/helia-rt#460): upstream does not bound tensor sizes; drop when upstream has it.
// The largest byte count BytesRequiredForTensor and TfLiteEvalTensorByteLength
// accept. The memory planner holds buffer sizes in int after rounding them up
// to MicroArenaBufferAlignment(), so a larger tensor cannot be planned.
constexpr size_t kMaxTensorBytes = INT_MAX - (MicroArenaBufferAlignment() - 1);

// Multiplies *size by factor in place. Returns kTfLiteError, leaving *size
// unchanged, if factor is negative or the product exceeds limit. The tensor
// sizing functions below pass kMaxTensorBytes for the element count and for
// the byte count. An element count never exceeds its byte count, so it also
// stays within the int that kernels hold it in.
TfLiteStatus MultiplyTensorSize(size_t* size, int64_t factor, size_t limit);

// How many bytes are needed to hold a tensor's contents.
// TODO(AmbiqAI/helia-rt#460): upstream does not check the rest of a shape that holds a -1; drop when upstream has it.
// A -1 dimension is one a kernel may resolve in Prepare (RESHAPE's stretch
// dimension, EMBEDDING_LOOKUP's output rows). The other dimensions and the
// element size are still checked, any other negative dimension fails, and
// *bytes is the product of every dimension and the element size in size_t, as
// upstream computes it, so kernels that read bytes before resolving the shape
// behave as upstream.
// The allocator re-sizes a TfLiteTensor once its dims are resolved.
TfLiteStatus BytesRequiredForTensor(const tflite::Tensor& flatbuffer_tensor,
                                    size_t* bytes, size_t* type_size);

// How many bytes are used in a TfLiteEvalTensor instance. The byte length is
// returned in out_bytes.
TfLiteStatus TfLiteEvalTensorByteLength(const TfLiteEvalTensor* eval_tensor,
                                        size_t* out_bytes);

// Deduce output dimensions from input and allocate given size.
// Useful for operators with two inputs where the largest input should equal the
// output dimension.
TfLiteStatus AllocateOutputDimensionsFromInput(TfLiteContext* context,
                                               const TfLiteTensor* input1,
                                               const TfLiteTensor* input2,
                                               TfLiteTensor* output);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MEMORY_HELPERS_H_
