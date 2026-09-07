# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Derives the float16 goldens for TestUnidirectionalLSTMFloat16.

Evaluates the 2x3x2x2 LSTM recurrence of Get2X2LstmEvalCheckData /
Create2x3x2X2FloatNodeContents (kernels/testdata/lstm_test_data.cc) in double
precision with libm tanh/exp, for two consecutive invokes with the hidden and
cell state carried across, then rounds each golden to the nearest float16 value
and prints the C literal arrays.

The reference is independent of the kernel under test: it is the textbook
recurrence, not a recording of what the f16 kernel produces. As a cross-check
it reproduces the float32 goldens in lstm_test_data.cc and
kExpectedSecondOutput/Hidden/Cell in unidirectional_sequence_lstm_test.cc to
within 4.5e-9.

Usage:
  python3 gen_lstm_f16_goldens.py
"""

import math

import numpy as np

# Create2x3x2X2FloatNodeContents: [state_dim][input_dim] activation weights,
# [state_dim][state_dim] recurrent weights, fused bias.
ACTIVATION_WEIGHT = {
    "forget": ((-10.0, -10.0), (-20.0, -20.0)),
    "input": ((10.0, 10.0), (20.0, 20.0)),
    "cell": ((1.0, 1.0), (1.0, 1.0)),
    "output": ((1.0, 1.0), (1.0, 1.0)),
}
RECURRENT_WEIGHT = ACTIVATION_WEIGHT
BIAS = {
    "forget": (1.0, 2.0),
    "input": (-1.0, -2.0),
    "cell": (0.0, 0.0),
    "output": (0.0, 0.0),
}

# Get2X2LstmEvalCheckData, batch major ([batch][time][input_dim]) because
# kDefaultBuiltinData sets time_major = false.
INPUT_DATA = (
    0.2, 0.3, 0.2, 0.3, 0.2, 0.3,
    -0.98, 0.62, 0.01, 0.99, 0.49, -0.32,
)
INITIAL_HIDDEN = (0.0, 0.0, 0.0, 0.0)
INITIAL_CELL = (0.0, 0.0, 0.0, 0.0)

BATCH_SIZE = 2
TIME_STEPS = 3
INPUT_DIM = 2
STATE_DIM = 2
CELL_CLIP = 6.0  # kDefaultBuiltinData


def sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x))


def gate(name, x, h, row):
  acc = BIAS[name][row]
  for d in range(INPUT_DIM):
    acc += ACTIVATION_WEIGHT[name][row][d] * x[d]
  for d in range(STATE_DIM):
    acc += RECURRENT_WEIGHT[name][row][d] * h[d]
  return acc


def invoke(hidden, cell):
  """Runs one three-step invoke, returning (output, hidden, cell)."""
  hidden = list(hidden)
  cell = list(cell)
  output = [0.0] * (BATCH_SIZE * TIME_STEPS * STATE_DIM)
  for t in range(TIME_STEPS):
    for b in range(BATCH_SIZE):
      x = INPUT_DATA[(b * TIME_STEPS + t) * INPUT_DIM:][:INPUT_DIM]
      h = hidden[b * STATE_DIM:][:STATE_DIM]
      for r in range(STATE_DIM):
        forget = sigmoid(gate("forget", x, h, r))
        remember = sigmoid(gate("input", x, h, r))
        candidate = math.tanh(gate("cell", x, h, r))
        out_gate = sigmoid(gate("output", x, h, r))
        c = forget * cell[b * STATE_DIM + r] + remember * candidate
        c = max(-CELL_CLIP, min(CELL_CLIP, c))
        cell[b * STATE_DIM + r] = c
        hidden[b * STATE_DIM + r] = out_gate * math.tanh(c)
      base = (b * TIME_STEPS + t) * STATE_DIM
      output[base:base + STATE_DIM] = hidden[b * STATE_DIM:][:STATE_DIM]
  return output, hidden, cell


def emit(name, values, per_line=4):
  """Prints a C float array of the float16-rounded values."""
  literals = [
      "%.8ff" % float(np.float16(v)) for v in values
  ]
  print("constexpr float %s[] = {" % name)
  for i in range(0, len(literals), per_line):
    print("    " + ", ".join(literals[i:i + per_line]) +
          ("," if i + per_line < len(literals) else "};"))


def main():
  first_output, hidden, cell = invoke(INITIAL_HIDDEN, INITIAL_CELL)
  second_output, second_hidden, second_cell = invoke(hidden, cell)

  emit("kExpectedFirstOutput", first_output)
  emit("kExpectedSecondOutput", second_output)
  emit("kExpectedSecondHidden", second_hidden)
  emit("kExpectedSecondCell", second_cell)


if __name__ == "__main__":
  main()
