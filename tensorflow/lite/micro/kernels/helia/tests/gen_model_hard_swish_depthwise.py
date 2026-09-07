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
"""Generates the model data for model_hard_swish_depthwise_test.

Writes model_hard_swish_depthwise_data.{cc,h} next to this script: three tiny
.tflite models as byte arrays, one fixed input vector each, and the goldens
captured from the TFLite reference interpreter.

The models are emitted as flatbuffer JSON and assembled with flatc rather than
converted from a trained Keras model. The int8 case needs exact control of the
input zero point (a converter picks it from a representative dataset), and the
byte-for-byte output has to be reproducible from a checkout that has no
TensorFlow installed.

Usage:
  python3 gen_model_hard_swish_depthwise.py [--flatc /path/to/flatc]

Requires flatc, clang-format, and an interpreter (ai_edge_litert, or
tensorflow.lite).
"""

import argparse
import json
import os
import subprocess
import tempfile

import numpy as np

try:
  from ai_edge_litert.interpreter import Interpreter, OpResolverType
except ImportError:  # pragma: no cover - depends on the host environment
  from tensorflow.lite.python.interpreter import Interpreter, OpResolverType

SEED = 20260906

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, *[os.pardir] * 6))
SCHEMA = os.path.join(REPO_ROOT, "tensorflow", "compiler", "mlir", "lite",
                      "schema", "schema.fbs")

LICENSE = """/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
"""

# Input quantization for the int8 HARD_SWISH model. The range is deliberately
# asymmetric so the zero point is not zero: hard swish is piecewise around
# x == 0 and x == -3, and a zero zero point hides an offset that is applied to
# the wrong side of the break. see AmbiqAI/ns-cmsis-nn#461
HARD_SWISH_INPUT_MIN = -3.0
HARD_SWISH_INPUT_MAX = 5.0
HARD_SWISH_OUTPUT_MIN = -0.5
HARD_SWISH_OUTPUT_MAX = 5.0


def affine_quant_params(real_min, real_max):
  """Return (scale, zero_point) for an int8 affine range covering [min, max]."""
  scale = (real_max - real_min) / 255.0
  zero_point = int(round(-128.0 - real_min / scale))
  return float(np.float32(scale)), zero_point


def build_model(flatc, json_model):
  """Run flatc over a flatbuffer JSON model and return the .tflite bytes."""
  with tempfile.TemporaryDirectory() as tmp:
    json_path = os.path.join(tmp, "model.json")
    with open(json_path, "w") as f:
      json.dump(json_model, f)
    subprocess.run(
        [flatc, "--binary", "-o", tmp, SCHEMA, json_path],
        check=True,
        cwd=tmp,
    )
    # flatc names the output from the schema's file_extension.
    out_path = os.path.join(tmp, "model.tflite")
    with open(out_path, "rb") as f:
      return f.read()


def format_in_place(clang_format, path):
  """Keep the emitted files in the same style as the other data files."""
  try:
    subprocess.run([clang_format, "-i", path], check=True)
  except (OSError, subprocess.CalledProcessError):
    print("warning: %s not run on %s" % (clang_format, path))


def float_buffer(values):
  """Return a flatbuffer JSON buffer holding little-endian float32 data."""
  return {"data": list(np.asarray(values, dtype=np.float32).tobytes())}


def hard_swish_model():
  """Single HARD_SWISH op, int8 in and out, non-zero input zero point."""
  in_scale, in_zp = affine_quant_params(HARD_SWISH_INPUT_MIN,
                                        HARD_SWISH_INPUT_MAX)
  out_scale, out_zp = affine_quant_params(HARD_SWISH_OUTPUT_MIN,
                                          HARD_SWISH_OUTPUT_MAX)
  assert in_zp != 0
  return {
      "version": 3,
      "operator_codes": [{
          "deprecated_builtin_code": 117,
          "version": 1,
          "builtin_code": "HARD_SWISH",
      }],
      "subgraphs": [{
          "tensors": [
              {
                  "shape": [1, 32],
                  "type": "INT8",
                  "buffer": 1,
                  "name": "input",
                  "quantization": {
                      "scale": [in_scale],
                      "zero_point": [in_zp],
                  },
              },
              {
                  "shape": [1, 32],
                  "type": "INT8",
                  "buffer": 2,
                  "name": "output",
                  "quantization": {
                      "scale": [out_scale],
                      "zero_point": [out_zp],
                  },
              },
          ],
          "inputs": [0],
          "outputs": [1],
          "operators": [{
              "opcode_index": 0,
              "inputs": [0],
              "outputs": [1],
          }],
      }],
      "description": "hard_swish_s8",
      "buffers": [{}, {}, {}],
  }, in_zp


def depthwise_model(in_channels, depth_multiplier, filter_values, bias_values,
                    description):
  """One float32 DEPTHWISE_CONV_2D, 3x3, stride 1, SAME padding, with bias."""
  out_channels = in_channels * depth_multiplier
  return {
      "version": 3,
      "operator_codes": [{
          "deprecated_builtin_code": 4,
          "version": 1,
          "builtin_code": "DEPTHWISE_CONV_2D",
      }],
      "subgraphs": [{
          "tensors": [
              {
                  "shape": [1, 8, 8, in_channels],
                  "type": "FLOAT32",
                  "buffer": 1,
                  "name": "input",
              },
              {
                  "shape": [1, 3, 3, out_channels],
                  "type": "FLOAT32",
                  "buffer": 2,
                  "name": "filter",
              },
              {
                  "shape": [out_channels],
                  "type": "FLOAT32",
                  "buffer": 3,
                  "name": "bias",
              },
              {
                  "shape": [1, 8, 8, out_channels],
                  "type": "FLOAT32",
                  "buffer": 4,
                  "name": "output",
              },
          ],
          "inputs": [0],
          "outputs": [3],
          "operators": [{
              "opcode_index": 0,
              "inputs": [0, 1, 2],
              "outputs": [3],
              "builtin_options_type": "DepthwiseConv2DOptions",
              "builtin_options": {
                  "padding": "SAME",
                  "stride_w": 1,
                  "stride_h": 1,
                  "depth_multiplier": depth_multiplier,
                  "fused_activation_function": "NONE",
                  "dilation_w_factor": 1,
                  "dilation_h_factor": 1,
              },
          }],
      }],
      "description": description,
      "buffers": [
          {},
          {},
          float_buffer(filter_values),
          float_buffer(bias_values),
          {},
      ],
  }


def run_reference(model_bytes, input_data):
  """Invoke the TFLite reference interpreter and return the output tensor."""
  # BUILTIN_REF, not the default: the default resolver delegates float ops to
  # XNNPACK, and these goldens are the reference kernels' output.
  interpreter = Interpreter(model_content=model_bytes,
                            experimental_op_resolver_type=
                            OpResolverType.BUILTIN_REF)
  interpreter.allocate_tensors()
  in_detail = interpreter.get_input_details()[0]
  out_detail = interpreter.get_output_details()[0]
  interpreter.set_tensor(in_detail["index"],
                         input_data.reshape(in_detail["shape"]))
  interpreter.invoke()
  return interpreter.get_tensor(out_detail["index"]).flatten()


def hard_swish_input(zero_point):
  """Fixed int8 input spanning the full range, straddling the zero point."""
  data = np.linspace(-128, 127, 32).round().astype(np.int8)
  # The three codes around the zero point are the ones the offset handling
  # gets wrong, so pin them rather than leaving them to the linear spread.
  # They replace the three nearest codes, which keeps the vector monotonic.
  nearest = np.argsort(np.abs(data.astype(np.int32) - zero_point))[:3]
  data[np.sort(nearest)] = [zero_point - 1, zero_point, zero_point + 1]
  return data


def format_bytes(data):
  """12 hex bytes per line, matching testing/test_conv_model.cc."""
  lines = []
  for i in range(0, len(data), 12):
    chunk = ", ".join("0x%02x" % b for b in data[i:i + 12])
    lines.append("    " + chunk + ("," if i + 12 < len(data) else ""))
  return "\n".join(lines)


def format_scalars(values, per_line, formatter):
  lines = []
  items = [formatter(v) for v in values]
  for i in range(0, len(items), per_line):
    chunk = ", ".join(items[i:i + per_line])
    lines.append("    " + chunk + ("," if i + per_line < len(items) else ""))
  return "\n".join(lines)


def int8_str(v):
  return str(int(v))


def float_str(v):
  # 9 significant digits round-trip float32 exactly.
  return "%.9gf" % np.float32(v)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--flatc", default="flatc")
  parser.add_argument("--clang-format", default="clang-format")
  args = parser.parse_args()

  rng = np.random.default_rng(SEED)

  models = []

  json_model, in_zp = hard_swish_model()
  hs_bytes = build_model(args.flatc, json_model)
  hs_input = hard_swish_input(in_zp)
  hs_golden = run_reference(hs_bytes, hs_input)
  models.append(("HardSwishS8", hs_bytes, ("int8_t", hs_input, int8_str, 12),
                 ("int8_t", hs_golden, int8_str, 12)))

  dw1_filter = rng.uniform(-0.5, 0.5, (1, 3, 3, 4)).astype(np.float32)
  dw1_bias = rng.uniform(-0.1, 0.1, (4,)).astype(np.float32)
  dw1_bytes = build_model(
      args.flatc,
      depthwise_model(4, 1, dw1_filter, dw1_bias, "depthwise_f32"))
  dw1_input = rng.uniform(-1.0, 1.0, (1, 8, 8, 4)).astype(np.float32)
  dw1_golden = run_reference(dw1_bytes, dw1_input)
  models.append(("DepthwiseF32", dw1_bytes,
                 ("float", dw1_input.flatten(), float_str, 4),
                 ("float", dw1_golden, float_str, 4)))

  dw2_filter = rng.uniform(-0.5, 0.5, (1, 3, 3, 4)).astype(np.float32)
  dw2_bias = rng.uniform(-0.1, 0.1, (4,)).astype(np.float32)
  dw2_bytes = build_model(
      args.flatc,
      depthwise_model(2, 2, dw2_filter, dw2_bias, "depthwise_dm2_f32"))
  dw2_input = rng.uniform(-1.0, 1.0, (1, 8, 8, 2)).astype(np.float32)
  dw2_golden = run_reference(dw2_bytes, dw2_input)
  models.append(("DepthwiseDm2F32", dw2_bytes,
                 ("float", dw2_input.flatten(), float_str, 4),
                 ("float", dw2_golden, float_str, 4)))

  header_path = os.path.join(SCRIPT_DIR, "model_hard_swish_depthwise_data.h")
  source_path = os.path.join(SCRIPT_DIR, "model_hard_swish_depthwise_data.cc")
  guard = ("TENSORFLOW_LITE_MICRO_KERNELS_HELIA_TESTS_"
           "MODEL_HARD_SWISH_DEPTHWISE_DATA_H_")

  hdr = [LICENSE.rstrip("\n"), ""]
  hdr.append("// Generated by gen_model_hard_swish_depthwise.py. The goldens")
  hdr.append("// come from the TFLite reference interpreter; do not hand-edit.")
  hdr.append("")
  hdr.append("#ifndef " + guard)
  hdr.append("#define " + guard)
  hdr.append("")
  hdr.append("#include <cstdint>")
  hdr.append("")
  for name, model_bytes, (in_type, in_data, _, _), (out_type, out_data, _,
                                                    _) in models:
    hdr.append("extern const unsigned char k%sModelData[];" % name)
    hdr.append("extern const unsigned int k%sModelDataSize;" % name)
    hdr.append("extern const %s k%sInput[];" % (in_type, name))
    hdr.append("constexpr unsigned int k%sInputSize = %d;" %
               (name, len(in_data)))
    hdr.append("extern const %s k%sGolden[];" % (out_type, name))
    hdr.append("constexpr unsigned int k%sGoldenSize = %d;" %
               (name, len(out_data)))
    hdr.append("")
  hdr.append("#endif  // " + guard)
  with open(header_path, "w") as f:
    f.write("\n".join(hdr) + "\n")

  src = [LICENSE.rstrip("\n"), ""]
  src.append('#include "tensorflow/lite/micro/kernels/helia/tests/'
             'model_hard_swish_depthwise_data.h"')
  src.append("")
  for name, model_bytes, (in_type, in_data, in_fmt,
                          in_per_line), (out_type, out_data, out_fmt,
                                         out_per_line) in models:
    src.append("alignas(16) const unsigned char k%sModelData[] = {" % name)
    src.append(format_bytes(model_bytes) + "};")
    src.append("const unsigned int k%sModelDataSize = %d;" %
               (name, len(model_bytes)))
    src.append("")
    src.append("const %s k%sInput[] = {" % (in_type, name))
    src.append(format_scalars(in_data, in_per_line, in_fmt) + "};")
    src.append("")
    src.append("const %s k%sGolden[] = {" % (out_type, name))
    src.append(format_scalars(out_data, out_per_line, out_fmt) + "};")
    src.append("")
    print("%s: model %d bytes, input %d, golden %d" %
          (name, len(model_bytes), len(in_data), len(out_data)))
  with open(source_path, "w") as f:
    f.write("\n".join(src))

  for path in (header_path, source_path):
    format_in_place(args.clang_format, path)

  print("hard swish input zero point: %d" % in_zp)


if __name__ == "__main__":
  main()
