# helia-rt → tflite-micro Replant Manifest

> **Status:** Draft / Phase 0
> **Freeze tag:** `pre-tflm-rebase-2026-05` → commit `01b125d5` (origin/main as of 2026-05-01)
> **Upstream baseline:** `tensorflow/tflite-micro` HEAD (currently `tflm/main` = `51bee03b`)
> **Last sync point of current main:** upstream commit `aa6e625` (Sep 9 2025) via helia-rt commit `23def27a`

This file is the **authoritative list of every path helia-rt owns** going forward. Anything not on this list will be replaced with the upstream version during the replant and must be obtainable mechanically from `tensorflow/tflite-micro` thereafter.

---

## Phase plan summary

| Phase | Action |
|---|---|
| 0 | **DONE.** Tag `pre-tflm-rebase-2026-05` pushed; this manifest written; cmsis_nn/xtensa/arc_mli/ceva/decompress drops confirmed. |
| 1 | Branch `chore/replant-on-tflm` from `tflm/main`; push to origin. |
| 2.1 | Migrate **repo metadata** (LICENSE, README, CODEOWNERS, etc.). |
| 2.2 | Migrate **CI infrastructure** (workflows, Dockerfiles, install scripts). |
| 2.3 | Migrate **helia kernel backend** (`kernels/helia/` + `ext_libs/helia.inc` + ns-cmsis-nn pin). GCC matrix should go green here. |
| 2.4 | Migrate **ATfE + Windows build glue** (linker script, target inc, test runner). ATfE matrix should go green here. |
| 2.5 | Migrate **vendored deps** (`third_party_static/`). |
| 2.6 | Migrate **neuralspot integration**. |
| 2.7 | Migrate **Zephyr module**. |
| 2.8 | Migrate **codegen**. |
| 2.9 | Migrate **NSX**. |
| 2.10 | Migrate **docs site**. |
| 2.11 | Migrate **versioning + release-please + CHANGELOG**. |
| 3 | Cutover PR: replace `main` with `chore/replant-on-tflm`. Old main archived as `archive/pre-replant-main`. |
| 4 | Build `ci/sync_from_tflite_micro.sh` + weekly bot workflow. |

---

## Explicit DROPS (intentionally NOT preserved)

We are abandoning every modification we made to these paths and taking the upstream version verbatim:

- `tensorflow/lite/micro/kernels/cmsis_nn/` — 13 modified files
- `tensorflow/lite/micro/kernels/xtensa/` — 51 modified files (+ 5 renames upstream did)
- `tensorflow/lite/micro/kernels/arc_mli/` — 17 modified files
- `tensorflow/lite/micro/kernels/ceva/` — 10 modified files
- `tensorflow/lite/micro/kernels/<top-level reference kernels>` — ~228 files; our diffs are dominated by:
  - 1144 chmod-only mode flips (`100644 → 100755`)
  - 334 `USE_TFLM_COMPRESSION` strip lines (we want compression *back*)
  - Copyright-year reverts (2024→2023)
  - Misc reformat noise
- `tensorflow/lite/micro/kernels/decompress*` (4 files we deleted) — restored by upstream import.
- `tensorflow/lite/micro/micro_allocator.{cc,h}`, `micro_context.{cc,h}`, `micro_interpreter_context.{cc,h}`, `test_helpers.{cc,h}`, `test_helper_custom_ops.{cc,h}`, `recording_micro_allocator*`, `micro_resource_variable.cc`, `micro_interpreter*`, `BUILD` — taken upstream verbatim.
- `signal/` — 129 modified files, all chmod / cosmetic.
- `tensorflow/lite/micro/tools/make/Makefile` — taken upstream **then re-patched minimally** (see Owned §C.1).
- `tensorflow/lite/micro/tools/make/helper_functions.inc`, `targets/cortex_m_generic_makefile.inc`, `targets/xtensa_makefile.inc`, `ext_libs/xtensa.inc`, `arm_gcc_download.sh`, `cmsis_download.sh`, `eyalroz_printf_download.sh`, `cmsis_nn_download.sh` — taken upstream verbatim. We had stray local edits but none are load-bearing.

---

## OWNED PATHS (replant manifest)

### A. Repo root metadata

| Path | Source | Notes |
|---|---|---|
| `LICENSE` | helia-rt | Ambiq Apollo SDK License — replaces upstream Apache 2.0 |
| `README.md` | helia-rt | helia-rt branding, install, examples |
| `AUTHORS` | helia-rt | |
| `CODEOWNERS` | helia-rt | |
| `CONTRIBUTING.md` | helia-rt | |
| `SECURITY.md` | helia-rt | |
| `CHANGELOG.md` | helia-rt | release-please managed |
| `.release-please-manifest.json` | helia-rt | |
| `release-please-config.json` | helia-rt | |
| `mkdocs.yaml` | helia-rt | |
| `pyproject.toml` | helia-rt | |
| `uv.lock` | helia-rt | |
| `zephyr_static_export.sh` | helia-rt | top-level Zephyr export helper |
| `.devcontainer/Dockerfile` | helia-rt | |
| `.devcontainer/devcontainer.json` | helia-rt | |
| `.devcontainer/install.sh` | helia-rt | |

### B. CI / GitHub Actions

#### B.1 Workflows kept (helia-owned, replace upstream where overlapping)

| Path | Notes |
|---|---|
| `.github/workflows/ci.yml` | helia main CI orchestrator |
| `.github/workflows/cortex_m.yml` | helia Cortex-M test matrix |
| `.github/workflows/cortex_m_arm_compiler.yml` | armclang (legacy ARMClang) coverage |
| `.github/workflows/cortex_m_virtual_hardware.yml` | AVH / FVP runs |
| `.github/workflows/docs.yml` | mkdocs build/publish |
| `.github/workflows/helia_build.yml` | NS-aware build |
| `.github/workflows/helia_build_docker_image.yml` | helia builder image |
| `.github/workflows/helia_release.yml` | release packaging |
| `.github/workflows/helia_test.yml` | core test matrix (GCC + ATfE × m3/m4+fp/m55) |
| `.github/workflows/release-please.yml` | |
| `.github/workflows/run_ci.yml` | |
| `.github/workflows/run_helia.yml` | |
| `.github/workflows/tests_entry.yml` | |
| `.github/workflows/tests_post.yml` | |
| `.github/workflows/zephyr_tflm_rt_assets.yml` | Zephyr prebuilt asset publisher |
| `.github/workflows/log_binary_size_pr.yml` | (modified upstream copy — keep our version) |
| `.github/workflows/check_tflite_files.yml` | |
| `.github/workflows/issue_on_error.yml` | |
| `.github/workflows/pypi_build.yml` | |
| `.github/workflows/stale_handler.yml` | |
| `.github/workflows/generate_integration_tests.yml` | |
| `.github/workflows/sync.yml` | (TF→TFLM-shared sync — re-enable cron in Phase 4) |

#### B.2 Workflows DROPPED (upstream-only, do not bring forward)

`check_bug_id.yml`, `merge_group.yml`, `pr_test.yml`, `run_core.yml`, `run_cortex_m.yml`, `run_hexagon.yml`, `run_riscv.yml`, `run_windows.yml`, `run_xtensa.yml`, `suite_core.yml`, `suite_cortex_m.yml`, `suite_hexagon.yml`, `suite_riscv.yml`, `suite_xtensa.yml`, `test_bazel.yml`, `test_hosted.yml`, `test_makefile.yml`, `test_misc.yml`, `test_windows.yml`.

#### B.3 CI scripts and Dockerfiles

| Path | Source | Notes |
|---|---|---|
| `ci/Dockerfile.micro` | helia-rt | |
| `ci/Dockerfile.hexagon` | helia-rt | |
| `ci/Dockerfile.xtensa_xplorer_11` | helia-rt | |
| `ci/Dockerfile.xtensa_xplorer_13` | helia-rt | |
| `ci/Dockerfile.xtensa_xplorer_solo` | helia-rt | |
| `ci/install_bazelisk.sh` | helia-rt | |
| `ci/install_buildifier.sh` | helia-rt | |
| `ci/install_cores_xplorer_11.sh` | helia-rt | |
| `ci/install_cores_xplorer_13.sh` | helia-rt | |
| `ci/install_cores_xplorer_solo.sh` | helia-rt | |
| `ci/install_qemu.sh` | helia-rt | |
| `ci/issue_on_error.py` | helia-rt | |
| `ci/check_tflite_files.py` | helia-rt | |
| `ci/sync_third_party_hexagon.sh` | helia-rt | |
| `ci/sync_from_upstream_tf.sh` | helia-rt | (TF→TFLM-shared sync — keep) |
| `ci/tflite_files.txt` | helia-rt | (companion to above) |
| **`ci/sync_from_tflite_micro.sh`** | **NEW (Phase 4)** | New TFLM→helia sync script |
| **`ci/tflite_micro_skip.txt`** | **NEW (Phase 4)** | Denylist mirroring this manifest |

### C. Build system additions and minimal patches

#### C.1 Top-level Makefile policy

Upstream `tensorflow/lite/micro/tools/make/Makefile` is taken **verbatim**. Our previous local diffs are split into:

- **ATfE rules** → folded into `tools/make/targets/cortex_m_corstone_300_makefile.inc` (already done in PR #115).
- **Windows-specific rules** → new `tools/make/helia_windows.inc` (TBD; survey Windows-related lines from current diff before Phase 2.4).
- **`OPTIMIZED_KERNEL_DIR=helia` glue** → already covered by upstream's existing `OPTIMIZED_KERNEL_DIR` mechanism + our `ext_libs/helia.inc`.

The upstream `Makefile` itself receives **at most one tiny patch** (e.g. an `-include` line) to wire in `helia_windows.inc`. Audit at Phase 2.4 to confirm this is feasible; if any helia-specific rule resists isolation, document why and add to this manifest as an explicit owned patch.

#### C.2 Owned ext_libs

| Path | Notes |
|---|---|
| `tensorflow/lite/micro/tools/make/ext_libs/helia.inc` | helia/ns-cmsis-nn integration |
| `tensorflow/lite/micro/tools/make/ext_libs/ns_cmsis_nn_download.sh` | ns-cmsis-nn fetcher |

#### C.3 Owned target makefiles (overlay onto upstream)

| Path | Notes |
|---|---|
| `tensorflow/lite/micro/tools/make/targets/cortex_m_corstone_300_makefile.inc` | ATfE branch (PR #115); review-then-overlay |
| (any `targets/*helia*.inc` / `*apollo*.inc`) | Confirm during Phase 2.4 |

#### C.4 Owned linker scripts

| Path | Notes |
|---|---|
| `tensorflow/lite/micro/cortex_m_corstone_300/corstone_300_atfe.ld` | PR #115 ATfE linker script |

#### C.5 Owned test runners

| Path | Notes |
|---|---|
| `tensorflow/lite/micro/testing/test_with_arm_corstone_300.sh` | semihosting, ATfE, FVP wiring (PR #115) |

#### C.6 Owned third_party pin

| Path | Notes |
|---|---|
| `tensorflow/lite/micro/tools/make/third_party_downloads.inc` | only the `NS_CMSIS_NN_COMMIT` line — apply as a focused patch |

#### C.7 Bazel / WORKSPACE

`MODULE.bazel`, `WORKSPACE` — taken upstream verbatim unless Phase 2 surfaces a real diff that's needed (audit at Phase 2.3).

### D. Helia kernel backend

#### D.1 Helia kernels (38 files, all under `tensorflow/lite/micro/kernels/helia/`)

```
README.md
helia_common.h
activations.cc
add.cc
batch_matmul.cc
comparisons.cc
concatenation.cc
conv.cc
depthwise_conv.cc
dequantize.cc
fill.cc
fully_connected.cc
hard_swish.cc
leaky_relu.cc
leaky_relu_common.cc
logistic.cc
logistic_common.cc
maximum_minimum.cc
mul.cc
pack.cc
pad.cc
pooling.cc
quantize_common.cc
reduce.cc
reduce_common.cc
reshape.cc
softmax.cc
split.cc
split_v.cc
squeeze.cc
strided_slice.cc
sub.cc
svdf.cc
tanh.cc
transpose.cc
transpose_conv.cc
unidirectional_sequence_lstm.cc
zeros_like.cc
```

#### D.2 Helia version header

| Path | Notes |
|---|---|
| `tensorflow/lite/micro/heliart_version.h` | release-please-managed version constant |

#### D.3 Helia docs

| Path | Notes |
|---|---|
| `tensorflow/lite/micro/docs/helia_notes.md` | helia-specific design notes |

### E. Integration tests

| Path | Notes |
|---|---|
| `tensorflow/lite/micro/integration_tests/nnaed/**` (261 files) | Generated test data + harnesses; verify regenerable from `generate_integration_tests.yml` before migrating — if regenerable, skip the data files and only port the generator config |

> **TODO (Phase 2.3):** confirm whether `nnaed/` is autogenerated and from what; if so, port the generator inputs only and rebuild artifacts in CI.

### F. Helia CI build/test scripts (under tflm tools dir)

| Path |
|---|
| `tensorflow/lite/micro/tools/ci_build/build_helia.sh` |
| `tensorflow/lite/micro/tools/ci_build/build_ambiq.sh` |
| `tensorflow/lite/micro/tools/ci_build/test_helia.sh` |
| `tensorflow/lite/micro/tools/ci_build/test_ambiq.sh` |
| `tensorflow/lite/micro/tools/ci_build/ns_local_build.sh` |
| `tensorflow/lite/micro/tools/ci_build/package_helia_bundle.sh` |
| `tensorflow/lite/micro/tools/ci_build/release_asset_helpers.sh` |
| `tensorflow/lite/micro/tools/ci_build/resolve_release_meta.sh` |
| `tensorflow/lite/micro/tools/ci_build/templates/zephyr_prebuilt/zephyr/CMakeLists.txt` |
| `tensorflow/lite/micro/tools/ci_build/templates/zephyr_prebuilt/zephyr/Kconfig` |
| `tensorflow/lite/micro/tools/ci_build/templates/zephyr_prebuilt/zephyr/module.yml` |

### G. Vendored static deps

`third_party_static/` (100 files): `flatbuffers/`, `gemmlowp/`, `kissfft/`, `ruy/`.

> **TODO (Phase 2.5):** verify each of these is still required given upstream uses `third_party/` with managed downloads. If upstream's mechanism suffices for any of these, drop it.

### H. Neuralspot integration

| Path |
|---|
| `neuralspot/build.sh` |
| `neuralspot/helia_build.py` |
| `neuralspot/module.mk` |

### I. Zephyr module

| Path |
|---|
| `zephyr/CMakeLists.txt` |
| `zephyr/Kconfig` |
| `zephyr/module.yml` |

### J. NSX

| Path |
|---|
| `nsx/CMakeLists.txt` |
| `nsx/nsx-module.yaml` |

### K. Codegen

`codegen/**` — full subtree (BUILD, build_def.bzl, code_generator.py, graph.py, inference_generator.py, tensor.py, utils.py, README.md, examples/, operators/, runtime/, templates/).

### L. Docs site (mkdocs)

`docs/**` (21 added files):
- `docs/index.md`
- `docs/assets/{helia-rt-banner-{dark,light}.png, helia-rt-icon{,-white}.png}`
- `docs/benchmarks/index.md`
- `docs/css/{custom,mkdocstrings,termynal}.css`
- `docs/examples/{index,neuralspot,source,zephyr}.md`
- `docs/features/index.md`
- `docs/js/{chart-init,custom,termynal}.js`
- `docs/usage/{index,neuralspot,source,zephyr}.md`
- `docs/automatically_generated_files.md`
- `docs/continuous_integration.md`
- `docs/python.md`
- `docs/internal/tflm-replant-manifest.md` ← **this file**

### M. Python packaging

- `python/tflite_micro/BUILD` (helia version)
- `python/tflite_micro/postinstall_check.py` (helia version)
- `python/py_namespace.bzl`
- `python/py_pkg_cc_deps.bzl`
- `python/tests/`

> **TODO:** audit which of these have content diff vs upstream and which are just chmod / cosmetic. Bring forward only real diffs.

### N. third_party/python_requirements.txt

| Path | Notes |
|---|---|
| `third_party/python_requirements.txt` | helia-specific pinned versions |
| `third_party/python_requirements.in` | helia-specific |

---

## Files explicitly verified as helia-owned

(Generated 2026-05-01 from `git diff aa6e625..origin/main`; cross-reference with upstream tree before Phase 2 actions.)

- `git diff --name-status aa6e625 origin/main | awk '$1=="A"' | wc -l` = **319 added files** under `tensorflow/`, **148** elsewhere.
- Renames (5): all under `kernels/xtensa/` or `tools/ci_build/` — drop along with §"Explicit DROPS".

---

## Open questions for the team

1. **Compression policy**: replant inherits upstream's compression code. Are we OK pulling in the `USE_TFLM_COMPRESSION` paths and the new `MicroInterpreter::SetCompressionMemory` API? (Recommended: yes — the strip in current main was probably accidental and we want the feature.)
2. **Integration tests `nnaed/`**: 261 files — generated or hand-curated? If generated, port generator only.
3. **`third_party_static/` vs upstream `third_party/`**: keep both, or migrate static consumers to upstream's download mechanism?
4. **Windows Makefile changes**: we believe there are some — needs audit at Phase 2.4. List specific lines/regions to preserve before that PR.
5. **`signal/` modifications (129 files)**: verified to be chmod/cosmetic only? If so, drop. If any have real Apollo-tuned changes, surface here.
6. **`MODULE.bazel` / `WORKSPACE`**: confirm no helia-specific deltas need preservation.

---

## Bookkeeping

- Freeze tag: `pre-tflm-rebase-2026-05` (commit `01b125d5`).
- Analysis branch: `chore/tflm-upstream-sync-analysis`.
- Replant branch (Phase 1): `chore/replant-on-tflm` (TBD).
- This manifest is the single source of truth — update it whenever a path is added/removed during Phase 2.
