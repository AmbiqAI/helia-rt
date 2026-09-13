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
"""Generate single-op FP16 ARG_MIN/ARG_MAX LiteRT model fixtures.

Requires ai-edge-litert==2.2.0 and flatbuffers.
"""

from importlib.metadata import version
from pathlib import Path
import struct

import flatbuffers
from ai_edge_litert import schema_py_generated as schema


CASES = {
    "ArgMinF16Model": (
        schema.BuiltinOperator.ARG_MIN,
        [2, 1, 3, 9],
        3,
        [2, 1, 3],
    ),
    "ArgMaxF16NegativeAxisModel": (
        schema.BuiltinOperator.ARG_MAX,
        [2, 1, 3, 9],
        -1,
        [2, 1, 3],
    ),
    "ArgMinF16PositiveAxisOutOfRangeModel": (
        schema.BuiltinOperator.ARG_MIN,
        [2, 1, 3, 9],
        4,
        [2, 1, 3],
    ),
    "ArgMaxF16NegativeAxisOutOfRangeModel": (
        schema.BuiltinOperator.ARG_MAX,
        [2, 1, 3, 9],
        -5,
        [2, 1, 3],
    ),
    "ArgMinF16RankFiveModel": (
        schema.BuiltinOperator.ARG_MIN,
        [1, 2, 1, 3, 9],
        4,
        [1, 2, 1, 3],
    ),
    "ArgMaxF16OutputMismatchModel": (
        schema.BuiltinOperator.ARG_MAX,
        [2, 1, 3, 9],
        3,
        [2, 1, 3, 1],
    ),
    "ArgMinF16ReducedEmptyModel": (
        schema.BuiltinOperator.ARG_MIN,
        [2, 1, 3, 0],
        3,
        [2, 1, 3],
    ),
    "ArgMaxF16ValidEmptyModel": (
        schema.BuiltinOperator.ARG_MAX,
        [9, 0],
        0,
        [0],
    ),
}


def make_model(builtin_code, input_shape, axis, output_shape):
    model = schema.ModelT()
    model.version = 3
    model.buffers = [schema.BufferT(), schema.BufferT()]
    model.buffers[1].data = bytearray(struct.pack("<i", axis))

    graph = schema.SubGraphT()
    input_tensor = schema.TensorT()
    input_tensor.name = "input"
    input_tensor.shape = input_shape
    input_tensor.type = schema.TensorType.FLOAT16
    input_tensor.buffer = 0
    axis_tensor = schema.TensorT()
    axis_tensor.name = "axis"
    axis_tensor.shape = [1]
    axis_tensor.type = schema.TensorType.INT32
    axis_tensor.buffer = 1
    output_tensor = schema.TensorT()
    output_tensor.name = "output"
    output_tensor.shape = output_shape
    output_tensor.type = schema.TensorType.INT32
    output_tensor.buffer = 0
    graph.tensors = [input_tensor, axis_tensor, output_tensor]
    graph.inputs = [0]
    graph.outputs = [2]

    operator = schema.OperatorT()
    operator.opcodeIndex = 0
    operator.inputs = [0, 1]
    operator.outputs = [2]
    if builtin_code == schema.BuiltinOperator.ARG_MIN:
        options = schema.ArgMinOptionsT()
        operator.builtinOptionsType = schema.BuiltinOptions.ArgMinOptions
    else:
        options = schema.ArgMaxOptionsT()
        operator.builtinOptionsType = schema.BuiltinOptions.ArgMaxOptions
    options.outputType = schema.TensorType.INT32
    operator.builtinOptions = options
    graph.operators = [operator]

    code = schema.OperatorCodeT()
    code.builtinCode = builtin_code
    code.deprecatedBuiltinCode = min(builtin_code, 127)
    code.version = 1
    model.operatorCodes = [code]
    model.subgraphs = [graph]

    builder = flatbuffers.Builder(4096)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def inspect_model(data):
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAs(data, 0))
    graph = model.subgraphs[0]
    operator = graph.operators[0]
    builtin_code = model.operatorCodes[operator.opcodeIndex].builtinCode
    axis = struct.unpack("<i", model.buffers[graph.tensors[1].buffer].data)[0]
    return (
        builtin_code,
        list(graph.tensors[0].shape),
        axis,
        list(graph.tensors[2].shape),
    )


def emit_bytes(name, values):
    lines = [f"constexpr unsigned char k{name}Data[] = {{"]
    for offset in range(0, len(values), 12):
        chunk = values[offset : offset + 12]
        lines.append("    " + ", ".join(f"0x{value:02x}" for value in chunk) + ",")
    lines.append("};")
    lines.append(f"constexpr unsigned int k{name}DataSize = {len(values)};")
    return lines


def main():
    litert_version = version("ai-edge-litert")
    if litert_version != "2.2.0":
        raise RuntimeError(f"Expected ai-edge-litert 2.2.0, got {litert_version}")

    models = {}
    for name, spec in CASES.items():
        data = make_model(*spec)
        if inspect_model(data) != spec:
            raise RuntimeError(f"Generated metadata mismatch for {name}")
        models[name] = data

    lines = [
        "// Generated by gen_arg_extrema_fp16_models.py using ai-edge-litert 2.2.0.",
        "#ifndef TENSORFLOW_LITE_MICRO_KERNELS_HELIA_TESTS_ARG_EXTREMA_FP16_MODELS_H_",
        "#define TENSORFLOW_LITE_MICRO_KERNELS_HELIA_TESTS_ARG_EXTREMA_FP16_MODELS_H_",
        "",
        "namespace {",
        "",
    ]
    for name, data in models.items():
        lines += emit_bytes(name, data)
        lines.append("")
    lines += [
        "}  // namespace",
        "",
        "#endif  // TENSORFLOW_LITE_MICRO_KERNELS_HELIA_TESTS_ARG_EXTREMA_FP16_MODELS_H_",
        "",
    ]

    license_end = Path(__file__).with_name("split_v_float_goldens.h").read_text().split("*/", 1)[0]
    output = Path(__file__).with_name("arg_extrema_fp16_models.h")
    output.write_text(license_end + "*/\n\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
