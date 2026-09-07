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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_BROADCAST_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_BROADCAST_H_

#include <cstdint>

#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

// Which leg of the heliaCore NHWC broadcast walk a TFLite shape pair lands on.
// The walk dispatches identical shapes to the flat elementwise kernel, a
// single-element operand to the scalar kernel, and anything else to the
// strided per-run walk. see AmbiqAI/ns-cmsis-nn#415
enum class HeliaBroadcastClass : uint8_t {
  kUnsupported = 0,
  kSameShape,
  kScalarInput1,
  kScalarInput2,
  kGeneral,
};

// The heliaCore broadcast entry points take 4-D NHWC dims and require every
// n/h/w/c pair to be equal or 1, with the output the elementwise maximum;
// rank above 4 has no NHWC spelling and stays on the reference path unless the
// shapes are identical, which the flat kernel handles at any rank.
// see AmbiqAI/ns-cmsis-nn#415
inline HeliaBroadcastClass HeliaClassifyBroadcast(
    const RuntimeShape& unextended_input1_shape,
    const RuntimeShape& unextended_input2_shape,
    const RuntimeShape& unextended_output_shape) {
  // Identical shapes go to the flat kernel, which walks FlatSize and never
  // spells out n/h/w/c, so the 4-D ceiling below does not apply to them; an
  // empty tensor is a zero-length walk.
  if (unextended_input1_shape == unextended_input2_shape &&
      unextended_input1_shape == unextended_output_shape) {
    return HeliaBroadcastClass::kSameShape;
  }

  if (unextended_input1_shape.DimensionsCount() > 4 ||
      unextended_input2_shape.DimensionsCount() > 4 ||
      unextended_output_shape.DimensionsCount() > 4) {
    return HeliaBroadcastClass::kUnsupported;
  }

  const RuntimeShape input1_shape =
      RuntimeShape::ExtendedShape(4, unextended_input1_shape);
  const RuntimeShape input2_shape =
      RuntimeShape::ExtendedShape(4, unextended_input2_shape);
  const RuntimeShape output_shape =
      RuntimeShape::ExtendedShape(4, unextended_output_shape);

  bool same_shape = true;
  for (int i = 0; i < 4; ++i) {
    const int dim1 = input1_shape.Dims(i);
    const int dim2 = input2_shape.Dims(i);
    if (dim1 < 1 || dim2 < 1) {
      return HeliaBroadcastClass::kUnsupported;
    }
    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      return HeliaBroadcastClass::kUnsupported;
    }
    if (output_shape.Dims(i) != (dim1 > dim2 ? dim1 : dim2)) {
      return HeliaBroadcastClass::kUnsupported;
    }
    if (dim1 != dim2) {
      same_shape = false;
    }
  }

  if (same_shape) {
    return HeliaBroadcastClass::kSameShape;
  }
  if (input1_shape.FlatSize() == 1) {
    return HeliaBroadcastClass::kScalarInput1;
  }
  if (input2_shape.FlatSize() == 1) {
    return HeliaBroadcastClass::kScalarInput2;
  }
  return HeliaBroadcastClass::kGeneral;
}

// True when the class needs the broadcast entry point rather than the flat one.
inline bool HeliaBroadcastNeedsWalk(HeliaBroadcastClass broadcast_class) {
  return broadcast_class != HeliaBroadcastClass::kUnsupported &&
         broadcast_class != HeliaBroadcastClass::kSameShape;
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_HELIA_HELIA_BROADCAST_H_
