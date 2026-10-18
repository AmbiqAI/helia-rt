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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_HELIA_DEPTHWISE_CONV_HELIA_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_HELIA_DEPTHWISE_CONV_HELIA_H_

#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {

// DEPTHWISE_CONV_2D for every type Register_DEPTHWISE_CONV_2D() takes, except
// that float16 layers call heliaCORE's arm_depthwise_conv_wrapper_f16_acc16,
// which accumulates in float16 lanes on MVE: the speed-oriented alternative to
// the default entry, which accumulates in float32 from the ns-cmsis-nn release
// with AmbiqAI/ns-cmsis-nn#830 (before it, only past 32 taps). Opt in per
// resolver with AddDepthwiseConv2D(Register_DEPTHWISE_CONV_2D_FLOAT16_ACC16()).
// HELIA backend only: the reference and CMSIS-NN backends do not define it.
TFLMRegistration Register_DEPTHWISE_CONV_2D_FLOAT16_ACC16();

}  // namespace micro

using micro::Register_DEPTHWISE_CONV_2D_FLOAT16_ACC16;

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_HELIA_DEPTHWISE_CONV_HELIA_H_
