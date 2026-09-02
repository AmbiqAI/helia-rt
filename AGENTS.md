# Agent guidance for helia-rt

Ambiq's LiteRT-for-Micro (TFLM) fork with heliaCORE Ambiq-tuned kernels.
Drop-in: same `MicroInterpreter`/`Model`/`OpResolver` API as upstream.

- **Three build paths, and Bazel cannot see heliaCORE.** Bazel drives the
  upstream test infra (`tensorflow/lite/micro/tools/ci_build/test_bazel*.sh`)
  but builds **reference kernels only**: the `optimized_kernels` flag in
  `tensorflow/lite/micro/kernels/BUILD` accepts only `""` and xtensa values,
  there is no BUILD file under `kernels/helia/`, and no Bazel target
  references those sources. A green Bazel run says nothing about a helia
  kernel change. Verify helia kernel work with the make build
  (`OPTIMIZED_KERNEL_DIR=helia`, `tools/make/ext_libs/helia.inc`), which is
  what `.github/workflows/helia_test.yml` runs on the Corstone-300 FVP; note
  a native make build is host code, so MVE and armclang/ATfE are only
  exercised by those CI legs. Delivery is multi-path: Zephyr/west module,
  CMSIS-Pack, neuralSPOT, and source/CMake, plus prebuilt `.a` bundles
  published by `helia_release.yml` (built with `build_helia.sh`,
  `TARGET=cortex_m_generic`). Don't assume one build covers another.
- **Fork discipline.** Upstream-derived files keep upstream layout and style;
  no drive-by reformatting, it breaks merges. Ambiq-specific work lives in the
  heliaCORE kernel paths; copy the idiom of a neighboring kernel.
- Toolchains: GCC, Arm Compiler 6, ATfE, but the PR test matrix
  (`helia_test.yml`) covers gcc and ATfE only; armclang is exercised by
  `helia_release.yml` and by `cortex_m_arm_compiler.yml`, which is manual.
  MVE/Helium needs `TARGET_ARCH=cortex-m55` (a native make build is x86/arm64
  host code) and is orthogonal to the SPEED / SIZE kernel profiles; see the
  `AGENTS.md` in AmbiqAI/ns-cmsis-nn (present from v7.31.0) for the
  MVE-vs-NEON trap before writing intrinsics.
- FVP-backed tests gate PRs through `.github/workflows/helia_test.yml`
  (invoked by `tests_entry.yml`); the upstream `cortex_m*.yml` workflows are
  manual or label-gated and `cortex_m_virtual_hardware.yml` is dispatch-only.
  Local runs without an EVB or FVP cannot exercise them; say so rather than
  reporting them green.
- Release or status changes: call them out in the PR description so
  maintainers can update the internal product-status record.
- CI is the authority on style/lint/test gates; do not restate them here.
