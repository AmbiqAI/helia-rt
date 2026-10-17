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

// RESIZE_NEAREST_NEIGHBOR through the helia kernel, compared bit for bit with
// the reference implementation for s8, s16, f32 and, where heliaCORE float16
// is built, f16, over every align_corners / half_pixel_centers combination,
// upscale and downscale ratios, 1-pixel inputs and outputs, batch > 1 and odd
// depths. Float inputs carry NaN payloads, both infinities, -0.0 and
// subnormals, and the f16 sweep covers every encoding. GCC and ATfE links wrap
// the heliaCORE entries to prove the route and the scratch context Prepare
// requested.

#include <cstdint>
#include <cstring>

#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/internal/reference/resize_nearest_neighbor.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

namespace {

enum class Entry { kS8, kS16, kF32, kF16, kCount };

int g_entry_calls[static_cast<int>(Entry::kCount)] = {};
int32_t g_ctx_size = -1;
bool g_ctx_buf = false;

#if HELIA_RESIZE_NN_LINK_WRAP
void Record(Entry entry, const cmsis_nn_context* ctx) {
  ++g_entry_calls[static_cast<int>(entry)];
  g_ctx_size = ctx != nullptr ? ctx->size : -1;
  g_ctx_buf = ctx != nullptr && ctx->buf != nullptr;
}
#endif

}  // namespace

#if HELIA_RESIZE_NN_LINK_WRAP
extern "C" {

#define HELIA_RESIZE_NN_WRAP(NAME, ENTRY, T)                                 \
  arm_cmsis_nn_status __real_##NAME(                                         \
      const cmsis_nn_context*, const cmsis_nn_resize_params*,                \
      const cmsis_nn_dims*, const T*, const cmsis_nn_dims*, const int32_t*,  \
      const cmsis_nn_dims*, T*);                                             \
  arm_cmsis_nn_status __wrap_##NAME(                                         \
      const cmsis_nn_context* ctx, const cmsis_nn_resize_params* params,     \
      const cmsis_nn_dims* input_shape, const T* input_data,                 \
      const cmsis_nn_dims* size_shape, const int32_t* size_data,             \
      const cmsis_nn_dims* output_shape, T* output_data) {                   \
    Record(ENTRY, ctx);                                                      \
    return __real_##NAME(ctx, params, input_shape, input_data, size_shape,   \
                         size_data, output_shape, output_data);              \
  }

HELIA_RESIZE_NN_WRAP(arm_resize_nearest_neighbor_s8, Entry::kS8, int8_t)
HELIA_RESIZE_NN_WRAP(arm_resize_nearest_neighbor_s16, Entry::kS16, int16_t)
HELIA_RESIZE_NN_WRAP(arm_resize_nearest_neighbor_f32, Entry::kF32, float)
#if ARM_NN_ENABLE_F16
HELIA_RESIZE_NN_WRAP(arm_resize_nearest_neighbor_f16, Entry::kF16, float16_t)
#endif

#undef HELIA_RESIZE_NN_WRAP

}  // extern "C"
#endif  // HELIA_RESIZE_NN_LINK_WRAP

namespace tflite {
namespace testing {
namespace {

struct Shape {
  int n, in_h, in_w, c, out_h, out_w;
};

constexpr int kMaxElements = 2048;

// Storage is the integer type of the same width; float tensors are compared
// as bit patterns.
template <typename Storage>
Storage Pattern(uint32_t index, bool is_float) {
  uint32_t bits = index * 2654435761u;
  if (is_float && sizeof(Storage) == 4) {
    const uint32_t special[] = {0x00000000, 0x80000000, 0x7f800000,
                                0xff800000, 0x7fc12345, 0xffc00001,
                                0x7f812345, 0x00000001, 0x807fffff};
    bits = special[index % 9] ^ (index / 9 % 2 ? 0 : 0x00001000);
  } else if (is_float) {
    bits = index;  // Every half encoding, in order.
  }
  Storage value;
  memcpy(&value, &bits, sizeof(Storage));
  return value;
}

template <typename Storage>
void ExpectMatchesReference(TfLiteType type, Entry entry, const Shape& s,
                            bool align_corners, bool half_pixel_centers,
                            uint32_t seed) {
  static Storage input[kMaxElements];
  static Storage output[kMaxElements + 2];
  static Storage expected[kMaxElements];

  const int input_count = s.n * s.in_h * s.in_w * s.c;
  const int output_count = s.n * s.out_h * s.out_w * s.c;
  EXPECT_LE(input_count, kMaxElements);
  EXPECT_LE(output_count, kMaxElements);
  if (input_count > kMaxElements || output_count > kMaxElements) {
    return;
  }
  const bool is_float = type == kTfLiteFloat32 || type == kTfLiteFloat16;
  for (int i = 0; i < input_count; ++i) {
    input[i] = Pattern<Storage>(seed + i, is_float);
  }
  Storage sentinel;
  memset(&sentinel, 0x5a, sizeof(sentinel));
  for (int i = 0; i < output_count + 2; ++i) {
    output[i] = sentinel;
  }

  int input_dims[] = {4, s.n, s.in_h, s.in_w, s.c};
  int size_dims[] = {1, 2};
  int output_dims[] = {4, s.n, s.out_h, s.out_w, s.c};
  int32_t size_data[] = {s.out_h, s.out_w};
  TfLiteTensor tensors[] = {
      CreateTensor(input, IntArrayFromInts(input_dims), false, type),
      CreateTensor(size_data, IntArrayFromInts(size_dims)),
      CreateTensor(output + 1, IntArrayFromInts(output_dims), false, type),
  };
  tensors[1].allocation_type = kTfLiteMmapRo;

  TfLiteResizeNearestNeighborParams params = {align_corners,
                                              half_pixel_centers};
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration =
      tflite::Register_RESIZE_NEAREST_NEIGHBOR();
  micro::KernelRunner runner(registration, tensors, 3,
                             IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  memset(g_entry_calls, 0, sizeof(g_entry_calls));
  g_ctx_size = -1;
  g_ctx_buf = false;
  EXPECT_EQ(kTfLiteOk, runner.Invoke());
#if HELIA_RESIZE_NN_LINK_WRAP
  for (int e = 0; e < static_cast<int>(Entry::kCount); ++e) {
    EXPECT_EQ(e == static_cast<int>(entry) ? 1 : 0, g_entry_calls[e]);
  }
  EXPECT_EQ(static_cast<int32_t>((s.out_h + s.out_w) * sizeof(int32_t)),
            g_ctx_size);
  EXPECT_TRUE(g_ctx_buf);
#else
  (void)entry;
#endif

  micro::ResizeNearestNeighborParams op_params;
  op_params.align_corners = align_corners;
  op_params.half_pixel_centers = half_pixel_centers;
  const int32_t in_shape[] = {s.n, s.in_h, s.in_w, s.c};
  const int32_t size_shape[] = {2};
  const int32_t out_shape[] = {s.n, s.out_h, s.out_w, s.c};
  micro::reference_ops::ResizeNearestNeighbor(
      op_params, RuntimeShape(4, in_shape), input, RuntimeShape(1, size_shape),
      size_data, RuntimeShape(4, out_shape), expected);

  int mismatches = 0;
  for (int i = 0; i < output_count; ++i) {
    mismatches += memcmp(&output[i + 1], &expected[i], sizeof(Storage)) != 0;
  }
  EXPECT_EQ(0, mismatches);
  EXPECT_EQ(0, memcmp(&output[0], &sentinel, sizeof(Storage)));
  EXPECT_EQ(0, memcmp(&output[output_count + 1], &sentinel, sizeof(Storage)));
}

constexpr Shape kShapes[] = {
    {1, 1, 1, 1, 1, 1},  {1, 1, 1, 3, 4, 5},    {1, 5, 7, 3, 1, 1},
    {2, 4, 6, 5, 8, 12}, {1, 7, 9, 3, 3, 4},    {1, 3, 5, 2, 7, 11},
    {3, 6, 4, 7, 9, 10}, {1, 16, 16, 4, 7, 7},  {1, 7, 7, 2, 16, 16},
    {1, 13, 17, 1, 5, 3}, {2, 9, 1, 3, 4, 1},   {1, 1, 12, 5, 1, 30},
};

template <typename Storage>
void Sweep(TfLiteType type, Entry entry) {
  uint32_t seed = 0;
  for (int corners = 0; corners < 2; ++corners) {
    for (int centers = 0; centers < 2; ++centers) {
      for (const Shape& s : kShapes) {
        ExpectMatchesReference<Storage>(type, entry, s, corners, centers,
                                        seed);
        seed += 977;
      }
      // Every input/output size pair from 1 to 8 on both axes.
      for (int in = 1; in <= 8; ++in) {
        for (int out = 1; out <= 8; ++out) {
          ExpectMatchesReference<Storage>(
              type, entry, Shape{1, in, in + 1, 3, out, out + 2}, corners,
              centers, seed);
          seed += 131;
        }
      }
    }
  }
}

// Resizes a 2x3 input to 3x2. With match_output false the output tensor is
// declared 2x3 instead, the same element count but not [N, size[0], size[1],
// C].
template <typename Storage>
TfLiteStatus RunSmallLayer(TfLiteType type, bool match_output,
                           bool* entry_called) {
  Storage input[2 * 3 * 2] = {};
  Storage output[6 * 2] = {};
  int input_dims[] = {4, 1, 2, 3, 2};
  int size_dims[] = {1, 2};
  int output_dims[] = {4, 1, match_output ? 3 : 2, match_output ? 2 : 3, 2};
  int32_t size_data[] = {3, 2};
  for (int i = 0; i < 12; ++i) {
    input[i] = static_cast<Storage>(i + 1);
  }
  TfLiteTensor tensors[] = {
      CreateTensor(input, IntArrayFromInts(input_dims), false, type),
      CreateTensor(size_data, IntArrayFromInts(size_dims)),
      CreateTensor(output, IntArrayFromInts(output_dims), false, type),
  };
  tensors[1].allocation_type = kTfLiteMmapRo;
  TfLiteResizeNearestNeighborParams params = {false, false};
  int inputs[] = {2, 0, 1};
  int outputs[] = {1, 2};
  const TFLMRegistration registration =
      tflite::Register_RESIZE_NEAREST_NEIGHBOR();
  micro::KernelRunner runner(registration, tensors, 3,
                             IntArrayFromInts(inputs),
                             IntArrayFromInts(outputs), &params);
  memset(g_entry_calls, 0, sizeof(g_entry_calls));
  TfLiteStatus status = runner.InitAndPrepare();
  if (status == kTfLiteOk) {
    status = runner.Invoke();
  }
  int calls = 0;
  for (int count : g_entry_calls) {
    calls += count;
  }
  *entry_called = calls != 0;
  return status;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

using tflite::testing::RunSmallLayer;
using tflite::testing::Sweep;

TEST(HeliaResizeNearestNeighborRouteTest, Int8MatchesReference) {
  Sweep<int8_t>(kTfLiteInt8, Entry::kS8);
}

TEST(HeliaResizeNearestNeighborRouteTest, Int16MatchesReference) {
  Sweep<int16_t>(kTfLiteInt16, Entry::kS16);
}

TEST(HeliaResizeNearestNeighborRouteTest, Float32SpecialValuesExactBits) {
  Sweep<int32_t>(kTfLiteFloat32, Entry::kF32);
}

TEST(HeliaResizeNearestNeighborRouteTest, MismatchedOutputShapeKeepsReference) {
  bool entry_called = true;
  EXPECT_EQ(kTfLiteOk, RunSmallLayer<int8_t>(kTfLiteInt8, false, &entry_called));
  EXPECT_FALSE(entry_called);
}

#if ARM_NN_ENABLE_F16
TEST(HeliaResizeNearestNeighborRouteTest, Float16EveryEncodingExactBits) {
  Sweep<uint16_t>(kTfLiteFloat16, Entry::kF16);
  // The sweep's first patterns stop well short of 0xffff; walk the rest.
  for (uint32_t base = 0; base < 65536; base += 1024) {
    tflite::testing::ExpectMatchesReference<uint16_t>(
        kTfLiteFloat16, Entry::kF16,
        tflite::testing::Shape{1, 16, 16, 4, 8, 16}, false, true, base);
  }
}

TEST(HeliaResizeNearestNeighborRouteTest,
     Float16MismatchedOutputShapeRejectedDuringPrepare) {
  bool entry_called = true;
  EXPECT_EQ(kTfLiteError,
            RunSmallLayer<uint16_t>(kTfLiteFloat16, false, &entry_called));
  EXPECT_FALSE(entry_called);
}
#else
TEST(HeliaResizeNearestNeighborRouteTest,
     UnavailableFloat16RejectedDuringPrepare) {
  bool entry_called = true;
  EXPECT_EQ(kTfLiteError,
            RunSmallLayer<uint16_t>(kTfLiteFloat16, true, &entry_called));
  EXPECT_FALSE(entry_called);
}
#endif

TF_LITE_MICRO_TESTS_MAIN
