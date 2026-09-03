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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_DEQUANTIZE_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_DEQUANTIZE_H_

#include <cstdint>
#include <cstring>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

struct DequantizeOpData {
  tflite::DequantizationParams quantization_params;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  int32_t output_zero_point;
};

TfLiteStatus DequantizePrepare(TfLiteContext* context, TfLiteNode* node);

// Widens an IEEE-754 binary16 bit pattern to float. Used on cores without
// float16 arithmetic, so it must not touch a float16 register.
inline float Float16BitsToFloat32(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  const uint32_t exponent = (bits >> 10) & 0x1Fu;
  const uint32_t mantissa = bits & 0x3FFu;

  uint32_t result;
  if (exponent == 0x1F) {
    result = sign | 0x7F800000u | (mantissa << 13);
  } else if (exponent != 0) {
    result = sign | ((exponent + 112) << 23) | (mantissa << 13);
  } else if (mantissa == 0) {
    result = sign;
  } else {
    // Subnormal: renormalize the mantissa, which float32 can always represent.
    uint32_t shifted = mantissa;
    uint32_t shifts = 0;
    while ((shifted & 0x400u) == 0) {
      shifted <<= 1;
      ++shifts;
    }
    result = sign | ((113 - shifts) << 23) | ((shifted & 0x3FFu) << 13);
  }

  float value;
  std::memcpy(&value, &result, sizeof(value));
  return value;
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_DEQUANTIZE_H_
