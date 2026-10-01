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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_FLOAT_COMMON_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_FLOAT_COMMON_H_

#include "Include/arm_nnfunctions.h"

namespace tflite {

// Compile-time mirrors of the ARM_NN_ENABLE_F32/F16 feature macros so kernel
// Prepare() type checks can accept float types only when the corresponding
// heliaCore API is actually compiled in. Rejecting an unsupported float type
// at Prepare (AllocateTensors) gives the user an actionable error instead of
// a bare kTfLiteError in the middle of an inference.
#if ARM_NN_ENABLE_F32
inline constexpr bool kHeliaFloat32Enabled = true;
#else
inline constexpr bool kHeliaFloat32Enabled = false;
#endif

#if ARM_NN_ENABLE_F16
inline constexpr bool kHeliaFloat16Enabled = true;
#else
inline constexpr bool kHeliaFloat16Enabled = false;
#endif

#if ARM_NN_ENABLE_F16
// Converts a float activation bound to float16_t, saturating to heliaCore's
// finite float16 range. CalculateActivationRange() reports "unbounded" as
// +/-FLT_MAX, and a plain static_cast of those to float16_t yields +/-inf;
// float16_t infinity behavior is not part of the heliaCore activation-clamp
// contract (see the ARM_NN_F16_FINITE_* sentinels), so saturate instead.
inline float16_t HeliaFloat16ActivationBound(float bound) {
  if (bound >= static_cast<float>(ARM_NN_F16_FINITE_MAX)) {
    return ARM_NN_F16_FINITE_MAX;
  }
  if (bound <= static_cast<float>(ARM_NN_F16_FINITE_LOWEST)) {
    return ARM_NN_F16_FINITE_LOWEST;
  }
  return static_cast<float16_t>(bound);
}
#endif

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_FLOAT_COMMON_H_
