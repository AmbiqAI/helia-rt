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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_HELIA_LOGISTIC_HELIA_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_HELIA_LOGISTIC_HELIA_H_

#include <cstdint>

#include "tensorflow/lite/micro/kernels/logistic.h"

namespace tflite {

// Upstream tflite-micro removed the int8 lookup-table field from
// `OpDataLogistic` in PR #308 in favour of a closed-form integer
// computation. The helia int8 logistic kernel still uses the
// table-based fast path (and the int16 path uses CMSIS-NN's
// `arm_logistic_s16`), so extend the upstream struct with the
// table here. The helia logistic kernel allocates
// `sizeof(OpDataLogisticHelia)` in `LogisticInit` so the extra
// 256 bytes are available; downstream code that only sees the
// upstream `OpDataLogistic` view (e.g. `LogisticPrepare`'s
// signature) keeps working unchanged because the helia struct
// is layout-compatible at the front.
struct OpDataLogisticHelia : OpDataLogistic {
  int8_t table[256];
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_HELIA_LOGISTIC_HELIA_H_
