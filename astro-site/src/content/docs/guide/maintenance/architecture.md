---
title: Source architecture
description: Find runtime, backend, integration and generated-source responsibilities.
---

heliaRT maintains a downstream LiteRT for Microcontrollers runtime with Ambiq-specific adapters and integration paths. Keep upstream-derived layout and notices intact; place product-specific work in the available extension points.

## Source map

| Location | Responsibility |
| --- | --- |
| `tensorflow/lite/micro/` | Interpreter, resolver, allocator and runtime contracts |
| `tensorflow/lite/micro/kernels/` | Reference implementations and shared operator declarations |
| `tensorflow/lite/micro/kernels/helia/` | HELIA adapters to heliaCORE |
| `tensorflow/lite/micro/kernels/helia/tests/` | HELIA-specific test cases and deterministic model fixtures |
| `tensorflow/lite/micro/tools/make/ext_libs/helia.inc` | Make dependency, feature and profile integration |
| `tensorflow/lite/micro/tools/make/ext_libs/helia_tests.inc` | HELIA-specific Make test registration |
| `cmake/helia_rt_sources.cmake` | Shared runtime source manifest and backend selection |
| `zephyr/`, `nsx/` | Framework integration |
| `tools/cmsis_pack/` | CMSIS-Pack generation from the CMake manifest |
| `helia/patches/` | Documented upstream drift and build-time patches |
| `third_party_static/` | Exported dependency headers used by source integrations |

The [repository layout policy](https://github.com/AmbiqAI/helia-rt/blob/main/helia/docs/repository_layout.md) describes extension points and upstream drift.

## Kernel selection is not runtime fallback

The source selector substitutes a backend implementation when that basename exists in the selected backend directory. Otherwise it selects the reference source for that operator. Inside a selected adapter, runtime type/shape checks determine optimized dispatch, reference fallback or an error. These are separate decisions.

CMake, NSX, Zephyr and CMSIS-Pack consume the [shared manifest](https://github.com/AmbiqAI/helia-rt/blob/main/cmake/helia_rt_sources.cmake). Make has its own specialization mechanism. Bazel's upstream kernel configuration does not select the HELIA directory.

## Add or extend an adapter

1. Implement the LiteRT registration and prepare/invoke contract in the HELIA directory.
2. Validate types, shapes, axes, buffer sizes and error propagation. Preserve model state and tensor lifetimes where the operator contract requires them.
3. Ensure the basename is present in the shared CMake manifest and the relevant Make source lists. An override of an existing basename and a wholly new operator require different wiring checks.
4. Register HELIA tests in `helia_tests.inc`, including unsupported inputs, scratch allocation, repeated invocation and optimized-path dispatch where relevant.
5. Check every integration path affected by changed sources or dependency requirements. Update coverage and release notes for public capability changes.

A passing reference/Bazel test does not establish that the HELIA adapter ran. See [Testing and CI](/helia-rt/guide/maintenance/testing/).

## Generated models and data

Keep generator inputs and scripts with deterministic fixtures when adding runtime tests. The upstream [hello_world Makefile](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/examples/hello_world/Makefile.inc) and [Bazel target](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/examples/hello_world/BUILD) demonstrate model/data array generation. Generated files do not replace the model metadata and input/output contracts needed to understand a test.

For Python model inspection and fixtures, use the repository's declared dependencies and the target's build instructions. Bazel utility tests exercise their own source paths, not HELIA execution. Avoid renaming the repository or changing a global Python environment as a prerequisite for application integration.
