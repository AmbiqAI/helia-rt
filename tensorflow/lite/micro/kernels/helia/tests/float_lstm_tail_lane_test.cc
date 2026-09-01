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

// UNIDIRECTIONAL_SEQUENCE_LSTM tail-lane coverage for the helia float kernels.
//
// Why this test exists (AmbiqAI/helia-rt#227, AmbiqAI/ns-cmsis-nn#315):
// heliaCORE's `arm_nn_lstm_step_f16` vectorises the gate activations over the
// hidden dimension with MVE and finishes the remainder with a scalar tail. In
// ns-cmsis-nn v7.29.2 the two halves do not use the same tanh approximation:
// the MVE body uses a lookup-table tanh and the scalar tail uses a rational
// approximation, so lanes 0..7 and lanes 8..9 of the *same* gate tensor are
// computed by different math. ns#315 reports a divergence of ~2.3e-2 in half
// precision on the same input: 0x3B3D == 0.90478515625 from one path against
// 0x3B6D == 0.92822265625 from the other. That reported figure is not this
// fixture's own measurement -- the worst divergence this test has observed is
// 1.10e-3 (CI run 33518826216), because it drives different operating points.
// Same defect, different magnitude; do not quote 2.3e-2 as something this
// test measured.
//
// The existing shared coverage cannot see this. The float16 case in
// kernels/unidirectional_sequence_lstm_test.cc uses state_dimension == 2,
// which is smaller than one MVE half-precision vector, so *every* lane goes
// down the scalar tail and the two implementations are never mixed within one
// tensor. Reproducing ns#315 requires state_dimension > 8 and
// state_dimension % 8 != 0; this test uses 10.
//
// The primary assertion is a *lane-uniformity invariant* rather than a
// tolerance comparison. Every gate row (input, recurrent and bias) is
// identical across the 10 hidden lanes and the initial state is zero, so all
// 10 lanes are mathematically the same number. A conforming kernel must
// therefore emit 10 numerically equal values per (batch, time step), and no
// tolerance has to be chosen.
//
// The strength of that claim differs by precision, so read the two cases
// differently:
//
//   * float16 (the ns#315 detector). The body and the tail run genuinely
//     different math -- LUT256 vs a Pade rational -- so the divergence is a
//     property of the implementation, not of the input. It cannot be explained
//     away by rounding or fusion.
//
//   * float32 (a weaker consistency check, and it needs a tolerance).
//     arm_nn_lstm_step_f32 has the same MVE-body/scalar-tail split, but both
//     halves interpolate the SAME LUT. The body uses explicit vfmaq while the
//     tail writes `y0 + (y1-y0)*frac`, whose fusion is an -ffp-contract
//     decision, so a body-vs-tail difference of a few ULP is permitted and
//     carries no information about correctness.
//
//     This is measured, not theoretical. In CI run 33518826216 the float32
//     lanes diverged on cortex-m55 gcc (MVE) while passing on cortex-m3 and
//     cortex-m4+fp gcc (no MVE), which isolates FMA contraction as the cause:
//     every divergent pair agreed to all six printed decimal places, e.g.
//     0.006203 vs 0.006203 and -0.039712 vs -0.039712. So the float32 case is
//     asserted with kFloat32LaneTolerance below, NOT with exact equality.
//     Only the float16 case is tolerance-free.
//
// A second, deliberately loose golden comparison against a double-precision
// reference LSTM guards against the degenerate case where every lane is
// uniformly wrong. Its tolerance is a magnitude sanity bound, NOT the ns#315
// detector -- see kFloat16GoldenTolerance below.
//
// This file lives under kernels/helia/tests/ (registered by
// ext_libs/helia_tests.inc) so the upstream shared test file keeps its
// upstream layout.

#include <cmath>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/testdata/lstm_test_data.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test_v2.h"

#if ARM_NN_ENABLE_F16
#include "arm_nnfunctions_flt.h"
#endif

namespace tflite {
namespace testing {
namespace {

// Shapes. kStateDimension is the whole point of the test: 10 > 8 and
// 10 % 8 == 2, so an MVE half-precision body (8 lanes) plus a 2 lane scalar
// tail is the natural decomposition of every gate.
constexpr int kBatchSize = 4;
constexpr int kTimeSteps = 2;
constexpr int kInputDimension = 4;
constexpr int kStateDimension = 10;

constexpr int kInputElements = kBatchSize * kTimeSteps * kInputDimension;
constexpr int kNumTensors = 24 + 1;
#if ARM_NN_ENABLE_F16
// Largest tensor in the node is the recurrent weight matrix (10 x 10). Only
// the float16 case needs side storage for the converted payload.
constexpr int kMaxTensorElements = kStateDimension * kStateDimension;
// The recurrent weight matrix and the output tensor are the two candidates for
// largest, and only the second grows with batch and time. Without these,
// raising kTimeSteps from 2 to 3 would write 120 float16 into a 100-element
// row and corrupt the next tensor's storage, silently.
static_assert(kBatchSize * kTimeSteps * kStateDimension <= kMaxTensorElements,
              "float16 conversion storage row is too small for the output "
              "tensor; raise kMaxTensorElements");
static_assert(kStateDimension * kStateDimension <= kMaxTensorElements,
              "float16 conversion storage row is too small for the recurrent "
              "weight matrix; raise kMaxTensorElements");
// The output tensor is the other candidate for largest, and it grows with
// batch and time while kMaxTensorElements does not. Without this guard,
// raising kTimeSteps or kBatchSize would silently overflow one row of the
// float16 conversion buffer into the next.
static_assert(kBatchSize * kTimeSteps * kStateDimension <= kMaxTensorElements,
              "float16 conversion storage row is too small for the output "
              "tensor; raise kMaxTensorElements");
static_assert(kInputElements <= kMaxTensorElements,
              "float16 conversion storage row is too small for the input "
              "tensor; raise kMaxTensorElements");
#endif  // ARM_NN_ENABLE_F16

constexpr double kCellClip = 6.0;

constexpr TfLiteUnidirectionalSequenceLSTMParams kBuiltinData = {
    /*.activation=*/kTfLiteActTanh,
    /*.cell_clip=*/static_cast<float>(kCellClip),
    /*.proj_clip=*/3.0f,
    /*.time_major=*/false,
    /*.asymmetric_quantize_inputs=*/true,
    /*.diagonal_recurrent_tensors=*/false};

// One row per gate. Every one of the kStateDimension hidden lanes gets this
// same row, which is what makes the lanes mathematically indistinguishable.
constexpr float kForgetInputRow[kInputDimension] = {-0.10f, 0.30f, 0.20f,
                                                    -0.05f};
constexpr float kForgetRecurrentValue = 0.04f;
constexpr float kForgetBias = 0.20f;

constexpr float kInputInputRow[kInputDimension] = {0.20f, -0.15f, 0.10f,
                                                   0.25f};
constexpr float kInputRecurrentValue = 0.05f;
constexpr float kInputBias = -0.10f;

constexpr float kCellInputRow[kInputDimension] = {0.35f, 0.15f, -0.25f, 0.20f};
constexpr float kCellRecurrentValue = 0.06f;
constexpr float kCellBias = 0.05f;

constexpr float kOutputInputRow[kInputDimension] = {0.25f, -0.20f, 0.30f,
                                                    0.10f};
constexpr float kOutputRecurrentValue = 0.03f;
constexpr float kOutputBias = -0.05f;

// [batch][time][input_dimension] (time_major == false). The four batches use
// different inputs so the run samples 4 x 2 x 2 == 16 distinct tanh arguments
// (one per cell gate, one per output activation), which makes it very unlikely
// that two different tanh approximations agree everywhere.
constexpr float kInputData[kInputElements] = {
    // batch 0
    1.00f, -2.00f, 0.50f, 1.50f,   //
    -1.00f, 2.00f, -0.50f, 0.50f,  //
    // batch 1
    2.00f, 1.00f, -1.50f, 0.00f,   //
    0.50f, -0.50f, 2.00f, -2.00f,  //
    // batch 2
    -2.00f, -1.00f, 1.00f, 2.00f,  //
    1.50f, 1.50f, -1.00f, -1.50f,  //
    // batch 3
    0.25f, 0.75f, -0.25f, 1.00f,   //
    -1.75f, 0.50f, 1.25f, -0.75f,  //
};

// Tolerances.
//
// Float32: the optimized and reference paths differ only by accumulation order
// and FMA contraction over 14-term dot products, which is well inside 1e-4.
constexpr float kFloat32GoldenTolerance = 1e-4f;

// Body-vs-tail agreement bound for float32. The two halves evaluate the same
// LUT expression with and without FMA contraction, so they may differ by a few
// ULP.
//
// Both margins, stated honestly, because they are very different sizes:
//
//   Headroom over legitimate error: large. The float32 divergence recorded on
//   m55-gcc printed equal to six decimal places at output magnitudes of 0.006
//   to 0.09, which bounds it below 1e-6 (six-decimal print equality bounds the
//   gap at <1e-6; it does NOT establish <5e-7, as an earlier revision of this
//   comment claimed). So 1e-5 is at least ~10x the observed FMA divergence,
//   and run 33521505536 confirms the float32 case passes with it.
//
//   Margin below a real ns#315-scale divergence: SMALL, about 1.6x. The
//   measured float16 divergence band in run 33518826216 was 1.6e-5 to 1.10e-3;
//   the low end came from a low-magnitude lane (output ~0.0074). So this
//   tolerance would catch the ns#315 signature at these operating points, but
//   it is not a comfortable margin, and a smaller-magnitude divergence could
//   slip under it. This float32 check is a consistency check, NOT the ns#315
//   detector -- that job belongs to the tolerance-free float16 assertion.
//
// Follow-up once the pin moves: ns#324 folds the float32 tail into the
// predicated MVE loop, which should make the lanes bit-exact and let this be
// tightened back toward exact equality.
constexpr float kFloat32LaneTolerance = 1e-5f;

#if ARM_NN_ENABLE_F16
//
// Float16: this bound is a *magnitude sanity check*, not the ns#315 detector.
// Half precision carries 11 significand bits, so one ULP near |y| == 1 is
// 2^-11 == 4.9e-4. Two time steps of 14-term dot products, four gate
// activations and the heliaCORE tanh/logistic table error accumulate a few
// ULP, and the shipped shared float16 LSTM test already documents deviations
// up to 2.7e-2 for its (much more aggressive) weights. 3e-2 therefore stays
// clear of false failures. It is deliberately LOOSER than the ~2.3e-2
// divergence ns#315 reports: the tolerance-free lane-uniformity assertion
// below is what catches that defect, so this check does not need to, and
// overloading it would make it flaky.
constexpr float kFloat16GoldenTolerance = 3e-2f;
#endif  // ARM_NN_ENABLE_F16

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Scalar double-precision reference for a single hidden lane. Because all
// lanes share the same weights and start from zero state, this one value is
// the reference for all kStateDimension lanes.
//
// Writes kBatchSize * kTimeSteps expected outputs (one per batch/time step).
void ComputeLaneReference(float* expected_output, float* expected_hidden,
                          float* expected_cell) {
  for (int b = 0; b < kBatchSize; ++b) {
    double hidden = 0.0;
    double cell = 0.0;
    for (int t = 0; t < kTimeSteps; ++t) {
      const float* x = &kInputData[(b * kTimeSteps + t) * kInputDimension];

      double forget_pre = static_cast<double>(kForgetBias);
      double input_pre = static_cast<double>(kInputBias);
      double cell_pre = static_cast<double>(kCellBias);
      double output_pre = static_cast<double>(kOutputBias);
      for (int i = 0; i < kInputDimension; ++i) {
        forget_pre += static_cast<double>(kForgetInputRow[i]) * static_cast<double>(x[i]);
        input_pre += static_cast<double>(kInputInputRow[i]) * static_cast<double>(x[i]);
        cell_pre += static_cast<double>(kCellInputRow[i]) * static_cast<double>(x[i]);
        output_pre += static_cast<double>(kOutputInputRow[i]) * static_cast<double>(x[i]);
      }
      // Every recurrent row is a constant, and every incoming hidden lane
      // holds the same value, so the recurrent contribution is
      // state_dimension * weight * hidden.
      forget_pre += kStateDimension * static_cast<double>(kForgetRecurrentValue) * hidden;
      input_pre += kStateDimension * static_cast<double>(kInputRecurrentValue) * hidden;
      cell_pre += kStateDimension * static_cast<double>(kCellRecurrentValue) * hidden;
      output_pre += kStateDimension * static_cast<double>(kOutputRecurrentValue) * hidden;

      const double forget_gate = Sigmoid(forget_pre);
      const double input_gate = Sigmoid(input_pre);
      const double cell_gate = std::tanh(cell_pre);
      const double output_gate = Sigmoid(output_pre);

      cell = forget_gate * cell + input_gate * cell_gate;
      if (cell > kCellClip) {
        cell = kCellClip;
      } else if (cell < -kCellClip) {
        cell = -kCellClip;
      }
      hidden = output_gate * std::tanh(cell);

      expected_output[b * kTimeSteps + t] = static_cast<float>(hidden);
    }
    expected_hidden[b] = static_cast<float>(hidden);
    expected_cell[b] = static_cast<float>(cell);
  }
}

// Fills a GateData whose rows are all identical.
void FillUniformGate(
    GateData<float, float, kInputDimension, kStateDimension>* gate,
    const float* input_row, float recurrent_value, float bias) {
  for (int s = 0; s < kStateDimension; ++s) {
    for (int i = 0; i < kInputDimension; ++i) {
      gate->activation_weight[s * kInputDimension + i] = input_row[i];
    }
    for (int r = 0; r < kStateDimension; ++r) {
      gate->recurrent_weight[s * kStateDimension + r] = recurrent_value;
    }
    gate->fused_bias[s] = bias;
    gate->activation_zp_folded_bias[s] = 0;
    gate->recurrent_zp_folded_bias[s] = 0;
  }
}

using UniformNodeContent =
    LstmNodeContent<float, float, float, float, kBatchSize, kTimeSteps,
                    kInputDimension, kStateDimension>;

UniformNodeContent CreateUniformLaneNodeContents() {
  GateData<float, float, kInputDimension, kStateDimension> forget_gate = {};
  GateData<float, float, kInputDimension, kStateDimension> input_gate = {};
  GateData<float, float, kInputDimension, kStateDimension> cell_gate = {};
  GateData<float, float, kInputDimension, kStateDimension> output_gate = {};
  FillUniformGate(&forget_gate, kForgetInputRow, kForgetRecurrentValue,
                  kForgetBias);
  FillUniformGate(&input_gate, kInputInputRow, kInputRecurrentValue,
                  kInputBias);
  FillUniformGate(&cell_gate, kCellInputRow, kCellRecurrentValue, kCellBias);
  FillUniformGate(&output_gate, kOutputInputRow, kOutputRecurrentValue,
                  kOutputBias);

  // Returned as a prvalue: LstmNodeContent stores pointers into its own member
  // arrays, so it must be constructed directly in the caller's storage rather
  // than copied. SetInputData() is called by the caller for the same reason.
  return UniformNodeContent(kBuiltinData, forget_gate, input_gate, cell_gate,
                            output_gate);
}

// Weights (indices 1..8) and biases (indices 12..15) must look like constant
// tensors to the kernel; mirrors SetConstTensors() in the shared test.
void SetConstTensors(TfLiteTensor* tensors) {
  for (int i = 1; i < 9; ++i) {
    tensors[i].allocation_type = kTfLiteMmapRo;
  }
  for (int i = 12; i < 16; ++i) {
    tensors[i].allocation_type = kTfLiteMmapRo;
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TEST(HeliaFloatLstmTailLaneTest, Float32TailLanesMatchVectorLanes) {
  auto contents = tflite::testing::CreateUniformLaneNodeContents();
  contents.SetInputData(tflite::testing::kInputData);
  tflite::testing::SetConstTensors(contents.GetTensors());

  auto builtin_data = contents.BuiltinData();
  const TFLMRegistration registration =
      tflite::Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  tflite::micro::KernelRunner runner(
      registration, contents.GetTensors(), tflite::testing::kNumTensors,
      contents.KernelInputs(), contents.KernelOutputs(),
      reinterpret_cast<void*>(&builtin_data));
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  const float* output = contents.GetOutputData();

  // Invariant: the kStateDimension lanes of one (batch, time step) are the
  // same number. Lanes 8..9 are the scalar tail; lanes 0..7 are the vector
  // body. Compared with kFloat32LaneTolerance rather than exactly, because the
  // two halves differ by FMA contraction on MVE builds -- see the file header.
  for (int b = 0; b < tflite::testing::kBatchSize; ++b) {
    for (int t = 0; t < tflite::testing::kTimeSteps; ++t) {
      const float* lane =
          &output[(b * tflite::testing::kTimeSteps + t) *
                  tflite::testing::kStateDimension];
      for (int s = 1; s < tflite::testing::kStateDimension; ++s) {
        EXPECT_NEAR(lane[0], lane[s],
                    tflite::testing::kFloat32LaneTolerance);
      }
    }
  }

  // Magnitude check against the double-precision reference.
  float expected_output[tflite::testing::kBatchSize *
                        tflite::testing::kTimeSteps] = {};
  float expected_hidden[tflite::testing::kBatchSize] = {};
  float expected_cell[tflite::testing::kBatchSize] = {};
  tflite::testing::ComputeLaneReference(expected_output, expected_hidden,
                                        expected_cell);
  for (int b = 0; b < tflite::testing::kBatchSize; ++b) {
    for (int t = 0; t < tflite::testing::kTimeSteps; ++t) {
      const float* lane =
          &output[(b * tflite::testing::kTimeSteps + t) *
                  tflite::testing::kStateDimension];
      for (int s = 0; s < tflite::testing::kStateDimension; ++s) {
        EXPECT_NEAR(expected_output[b * tflite::testing::kTimeSteps + t],
                    lane[s], tflite::testing::kFloat32GoldenTolerance);
      }
    }
  }

  const float* hidden = contents.GetHiddenStateData();
  const float* cell = contents.GetCellStateData();
  for (int b = 0; b < tflite::testing::kBatchSize; ++b) {
    for (int s = 0; s < tflite::testing::kStateDimension; ++s) {
      const int index = b * tflite::testing::kStateDimension + s;
      EXPECT_NEAR(hidden[b * tflite::testing::kStateDimension], hidden[index],
                  tflite::testing::kFloat32LaneTolerance);
      EXPECT_NEAR(cell[b * tflite::testing::kStateDimension], cell[index],
                  tflite::testing::kFloat32LaneTolerance);
      EXPECT_NEAR(expected_hidden[b], hidden[index],
                  tflite::testing::kFloat32GoldenTolerance);
      EXPECT_NEAR(expected_cell[b], cell[index],
                  tflite::testing::kFloat32GoldenTolerance);
    }
  }
}

#if ARM_NN_ENABLE_F16
// The ns#315 catcher. Same construction, float16 tensors.
TEST(HeliaFloatLstmTailLaneTest, Float16TailLanesMatchVectorLanes) {
  auto contents = tflite::testing::CreateUniformLaneNodeContents();
  contents.SetInputData(tflite::testing::kInputData);
  tflite::testing::SetConstTensors(contents.GetTensors());

  // Re-type every populated tensor as float16, converting the float32 payload
  // in place into local storage. Mirrors the conversion idiom in the shared
  // float16 LSTM test.
  static float16_t
      storage[tflite::testing::kNumTensors][tflite::testing::kMaxTensorElements];
  TfLiteTensor* tensors = contents.GetTensors();
  for (int i = 0; i < tflite::testing::kNumTensors; ++i) {
    if (tensors[i].data.raw == nullptr || tensors[i].dims == nullptr) {
      continue;
    }
    const int elements = tflite::ElementCount(*tensors[i].dims);
    for (int j = 0; j < elements; ++j) {
      storage[i][j] = static_cast<float16_t>(tensors[i].data.f[j]);
    }
    tensors[i].data.data = storage[i];
    tensors[i].type = kTfLiteFloat16;
    tensors[i].bytes = elements * sizeof(float16_t);
  }

  auto builtin_data = contents.BuiltinData();
  const TFLMRegistration registration =
      tflite::Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
  tflite::micro::KernelRunner runner(
      registration, tensors, tflite::testing::kNumTensors,
      contents.KernelInputs(), contents.KernelOutputs(),
      reinterpret_cast<void*>(&builtin_data));
  EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  EXPECT_EQ(kTfLiteOk, runner.Invoke());

  const int output_index = contents.KernelOutputs()->data[0];
  const float16_t* output =
      reinterpret_cast<const float16_t*>(tensors[output_index].data.raw);
  const float16_t* hidden = reinterpret_cast<const float16_t*>(
      tensors[tflite::kLstmOutputStateTensor].data.raw);
  const float16_t* cell = reinterpret_cast<const float16_t*>(
      tensors[tflite::kLstmCellStateTensor].data.raw);

  // ns#315: lanes 0..7 (MVE body) and lanes 8..9 (scalar tail) of the same
  // gate tensor must not be computed by different tanh approximations. All
  // lanes are mathematically identical here, so any difference at all is the
  // defect. No tolerance is involved. Unlike the float32 case above, this is
  // not sensitive to FMA contraction: the two halves evaluate different
  // functions (LUT256 vs the Pade rational), not the same function two ways.
  for (int b = 0; b < tflite::testing::kBatchSize; ++b) {
    for (int t = 0; t < tflite::testing::kTimeSteps; ++t) {
      const float16_t* lane =
          &output[(b * tflite::testing::kTimeSteps + t) *
                  tflite::testing::kStateDimension];
      for (int s = 1; s < tflite::testing::kStateDimension; ++s) {
        EXPECT_EQ(static_cast<float>(lane[0]), static_cast<float>(lane[s]));
      }
    }
    for (int s = 1; s < tflite::testing::kStateDimension; ++s) {
      const int index = b * tflite::testing::kStateDimension + s;
      const int base = b * tflite::testing::kStateDimension;
      EXPECT_EQ(static_cast<float>(hidden[base]),
                static_cast<float>(hidden[index]));
      EXPECT_EQ(static_cast<float>(cell[base]),
                static_cast<float>(cell[index]));
    }
  }

  // Magnitude sanity check only; see kFloat16GoldenTolerance.
  float expected_output[tflite::testing::kBatchSize *
                        tflite::testing::kTimeSteps] = {};
  float expected_hidden[tflite::testing::kBatchSize] = {};
  float expected_cell[tflite::testing::kBatchSize] = {};
  tflite::testing::ComputeLaneReference(expected_output, expected_hidden,
                                        expected_cell);
  for (int b = 0; b < tflite::testing::kBatchSize; ++b) {
    for (int t = 0; t < tflite::testing::kTimeSteps; ++t) {
      const float16_t* lane =
          &output[(b * tflite::testing::kTimeSteps + t) *
                  tflite::testing::kStateDimension];
      for (int s = 0; s < tflite::testing::kStateDimension; ++s) {
        EXPECT_NEAR(expected_output[b * tflite::testing::kTimeSteps + t],
                    static_cast<float>(lane[s]),
                    tflite::testing::kFloat16GoldenTolerance);
      }
    }
    for (int s = 0; s < tflite::testing::kStateDimension; ++s) {
      const int index = b * tflite::testing::kStateDimension + s;
      EXPECT_NEAR(expected_hidden[b], static_cast<float>(hidden[index]),
                  tflite::testing::kFloat16GoldenTolerance);
      EXPECT_NEAR(expected_cell[b], static_cast<float>(cell[index]),
                  tflite::testing::kFloat16GoldenTolerance);
    }
  }
}
#elif defined(__ARM_FEATURE_MVE) && ((__ARM_FEATURE_MVE) & 2)

// This is the guard that matters most. Float16TailLanesMatchVectorLanes is the
// ns#315 detector and cortex-m55 is the only configuration that has the MVE
// body / scalar tail split it looks for. If ARM_NN_ENABLE_F16 ever stops being
// defined on such a build, the detector would disappear while the binary still
// ran its one float32 case and the leg still reported success (helia-rt#231
// shows an empty suite scores green). Fail loudly instead.
//
// Known gap: ATfE builds cortex-m55 with +nomve (helia-rt#225), so
// __ARM_FEATURE_MVE is unset there and this cannot fire. Acceptable: without
// MVE there is no body/tail split, so there is no ns#315 coverage to lose.
TEST(HeliaFloatLstmTailLaneTest, Float16CoverageMustNotSilentlyDisappear) {
  FAIL(
      "ARM_NN_ENABLE_F16 is not defined on a build with MVE floating point. "
      "The ns#315 float16 LSTM tail-lane detector silently compiled out.");
}

#endif  // ARM_NN_ENABLE_F16

TF_LITE_MICRO_TESTS_MAIN
