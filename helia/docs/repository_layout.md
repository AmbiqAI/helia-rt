# Repository Layout — heliaRT Conventions

heliaRT is a downstream fork of [`tensorflow/tflite-micro`](https://github.com/tensorflow/tflite-micro)
that we periodically resync with upstream. The cost of every line of helia
code that lives inside an upstream file is paid again at every sync. This
document defines where helia-specific code is allowed to live so that cost
stays low.

## TL;DR — Decision Flow

```
Adding new code that is helia-only?
    └── Is there an extension hook?
          ├── Optimized kernel implementation       → kernels/helia/<op>.cc
          ├── Helia-only kernel test                → kernels/helia/tests/<op>_test.cc
          ├── Build glue / library deps             → tools/make/ext_libs/helia.inc
          ├── Extra test registration               → tools/make/ext_libs/helia_tests.inc
          ├── CI script (build/test/release/etc.)   → tools/ci_build/<verb>_helia.sh
          ├── GitHub workflow                       → .github/workflows/helia_<name>.yml
          ├── Docs / contributor guide / patches    → helia/{docs,patches}/
          └── Otherwise (no hook fits)              → helia/patches/ as a build-time patch
                                                      OR (last resort) in-place change with rationale
```

If the only feasible option is an in-place change to an upstream file, see
[Inline drift policy](#inline-drift-policy) below.

## Allowed locations for helia-specific files

### 1. Optimized kernels (`tensorflow/lite/micro/kernels/helia/`)

Upstream's [`OPTIMIZED_KERNEL_DIR`](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tools/make/specialize_files.py)
mechanism transparently swaps any file matching
`tensorflow/lite/micro/kernels/<op>.cc` for `kernels/helia/<op>.cc` when the
build is invoked with `OPTIMIZED_KERNEL_DIR=helia`. New helia kernel
implementations belong here. The same mechanism auto-defines `-DHELIA` for
the build (see [`tools/make/Makefile`](../../tensorflow/lite/micro/tools/make/Makefile)).

#### `HELIA` vs `CMSIS_NN` preprocessor macros

Helia builds define **both** macros:

- **`CMSIS_NN`** is added to `CCFLAGS`/`CXXFLAGS` from
  [`tools/make/ext_libs/helia.inc`](../../tensorflow/lite/micro/tools/make/ext_libs/helia.inc)
  because helia kernels are CMSIS-NN-backed (they `#include "Include/arm_nnfunctions.h"`
  and link NS-CMSIS-NN). Use `#if defined(CMSIS_NN)` for behavior helia
  inherits from CMSIS-NN — typically the `Register_*_INT8`/`INT16`/`INT4`
  variant declarations in `tensorflow/lite/micro/kernels/<op>.h`. This is
  the common case and creates **zero** inline drift.
- **`HELIA`** is auto-defined from `OPTIMIZED_KERNEL_DIR=helia`. Use
  `#if defined(HELIA)` only for heliaCORE-exclusive features that
  NS-CMSIS-NN supports but upstream CMSIS-NN does not (e.g., a future
  variant CMSIS-NN never adds).

Note that **kernel source selection is governed by `OPTIMIZED_KERNEL_DIR`,
not by these macros** — `kernels/cmsis_nn/*.cc` is never compiled into a
`OPTIMIZED_KERNEL_DIR=helia` build, regardless of `-DCMSIS_NN`.

When you need a HELIA-only declaration, **strongly prefer** adding it to a
helia-owned header under `kernels/helia/` (e.g.
[`kernels/helia/helia_common.h`](../../tensorflow/lite/micro/kernels/helia/helia_common.h))
over inline-patching an upstream header.

### 2. Helia kernel tests (`tensorflow/lite/micro/kernels/helia/tests/`)

For test coverage that exercises code paths that **only exist** in
`kernels/helia/` (e.g., int16 hard_swish), put tests under `kernels/helia/tests/`
and register them in [`tools/make/ext_libs/helia_tests.inc`](../../tensorflow/lite/micro/tools/make/ext_libs/helia_tests.inc).
That file is auto-included from `kernels/Makefile.inc` only when
`OPTIMIZED_KERNEL_DIR=helia`. Do **not** add tests to upstream test files for
helia-only behavior — it creates drift on every upstream sync.

### 3. Build extensions (`tensorflow/lite/micro/tools/make/ext_libs/`)

- `helia.inc` — declares CMSIS / NS_CMSIS_NN dependencies, additional
  third-party sources, include paths.
- `helia_tests.inc` — appends helia-only test sources to
  `MICROLITE_KERNEL_SIMPLE_TEST_SRCS`.

Both are auto-loaded only when `OPTIMIZED_KERNEL_DIR=helia` and require no
patches to upstream files.

### 4. CI scripts (`tensorflow/lite/micro/tools/ci_build/`)

helia-specific shell entrypoints follow the naming pattern `*_helia.sh`:

| Script | Used by |
| --- | --- |
| `build_helia.sh` | `helia_build.yml`, `helia_release.yml` |
| `test_helia.sh` | `helia_test.yml`, `run_helia.yml` |
| `package_helia_bundle.sh` | `helia_release.yml` |
| `resolve_release_meta.sh` | `helia_release.yml` |
| `release_asset_helpers.sh` | sourced by `helia_release.yml` |

New helia CI scripts must follow the same `*_helia.sh` (or
`<verb>_helia*.sh`) pattern. Avoid editing the upstream `test_*.sh`
scripts unless the behavior change is *also* useful upstream.

### 5. GitHub workflows (`.github/workflows/`)

helia-specific workflows are named `helia_*.yml` or `run_helia.yml` /
`zephyr_tflm_rt_assets.yml`. Upstream workflows (`ci.yml`, `cortex_m.yml`,
`docs.yml`, `release-please.yml`, etc.) should remain pristine wherever
possible — when they need helia behavior, prefer adding a sibling
`helia_*` workflow rather than patching the upstream one.

### 6. Top-level helia assets (`helia/`)

This very directory. Contains documentation, override patches, and assets
that don't belong in any of the above categories. **Do not** put runtime
source code under `helia/`.

### 7. Other approved helia-only locations

The following paths exist alongside upstream paths and are reserved for
helia-specific files:

| Path | Purpose |
| --- | --- |
| `tensorflow/lite/micro/heliart_version.h` | Single-file version header consumed by `release-please-config.json`. Do not add other headers here. |
| `tensorflow/lite/micro/cortex_m_corstone_300/corstone_300_atfe.ld` | ATfE-specific linker script for Corstone-300. Sibling files in this directory are upstream's; new helia files here must be `*_atfe.*` or `*_helia.*`. |
| `tensorflow/lite/micro/tools/make/targets/cortex_m_corstone_300_atfe.inc` | Externalized ATfE toolchain logic, included from `cortex_m_corstone_300_makefile.inc`. New target/toolchain combinations should follow the same `<target>_<toolchain>.inc` convention. |
| `tensorflow/lite/micro/tools/make/arm_clang_download.sh`, `arm_toolchain_embedded_download.sh` | helia-managed toolchain downloaders invoked from the cortex_m_* target makefiles. |
| `tensorflow/lite/micro/tools/make/ext_libs/ns_cmsis_nn_download.sh` | NS-CMSIS-NN downloader invoked from `helia.inc`. The `NS_CMSIS_NN_COMMIT` it consumes is exported from `helia.inc`. |
| `tensorflow/lite/micro/tools/ci_build/{ns_local_build,package_helia_bundle,resolve_release_meta,release_asset_helpers}.sh` | helia CI helpers. The `*_helia.sh` naming is preferred for new scripts. |
| `tensorflow/lite/micro/tools/github/arm_virtual_hardware/cortex_m_*_avh.yml` | Arm Virtual Hardware test configs consumed by `.github/workflows/cortex_m_virtual_hardware.yml`. |
| `tensorflow/lite/micro/integration_tests/nnaed/` | helia integration tests vendored from the nnaed test generator. New helia-only integration tests go here, not under `integration_tests/seanet/` (upstream). |
| `tensorflow/lite/micro/tools/ci_build/templates/zephyr_prebuilt/` | Zephyr module template assets used by `zephyr_static_export.sh` and `zephyr_tflm_rt_assets.yml`. |
| `.devcontainer/` | helia-rt VS Code dev container definition. Upstream has none. |
| `.github/stale.yml` | Probot-stale config (separate from `stale_handler.yml` which is upstream-owned). |
| `nsx/` | heliaRT NSX module manifest (`nsx-module.yaml`, `CMakeLists.txt`) consumed by neuralSPOT and bumped by `release-please-config.json`. |
| `zephyr/` | Top-level Zephyr module (`CMakeLists.txt`, `Kconfig`, `module.yml`) so the heliaRT repo can be west-imported as a Zephyr module. Distinct from `tensorflow/lite/micro/tools/ci_build/templates/zephyr_prebuilt/`, which is the prebuilt-asset template used by `zephyr_tflm_rt_assets.yml`. |
| `zephyr_static_export.sh` | Top-level helper that drives the static Zephyr export flow. |
| `pyproject.toml`, `uv.lock` | Python tooling (uv) for docs / release scripts. The lock file is committed so CI builds are reproducible. |
| `mkdocs.yaml` | MkDocs site configuration consumed by `.github/workflows/docs.yml`. |
| `release-please-config.json`, `.release-please-manifest.json` | release-please config + state. The `extra-files` block bumps `tensorflow/lite/micro/heliart_version.h` and `nsx/nsx-module.yaml`. |
| `ci/install_qemu.sh`, `ci/check_tflite_files.py`, `ci/issue_on_error.py` | helia-rt-only files in an upstream-owned directory; do not rename or move (referenced by helia workflows and by `ci/Dockerfile.micro`). |
| `codegen/`, `gen/`, `neuralspot/`, `data/`, `docs/`, `site/` | helia-rt-only top-level directories (not present upstream). New helia-only directories at the repo root must be approved here before being added. |

## Where you must NOT put helia-specific code

- Any file under `tensorflow/lite/`, `signal/`, `python/tflite_micro/`, or
  `third_party/` that exists upstream — unless the change is unavoidable
  (see [Inline drift policy](#inline-drift-policy)).
- Workflow files inherited from upstream (`ci.yml`, `cortex_m.yml`,
  `release-please.yml`, etc.).
- Upstream CI scripts (`test_cortex_m_corstone_300.sh`, `test_size.sh`,
  etc.) — exception: minimal localizations such as the size-comparison
  reference URL in `test_size.sh`.

## Inline drift policy

Some helia behavior **cannot** be expressed via an extension hook. Examples:

- Adding `|| defined(HELIA)` to existing `#if defined(CMSIS_NN)` guards in
  shared kernel headers (`tensorflow/lite/micro/kernels/<op>.h`). The
  helia kernel files won't link without these guards.
- Apollo-specific compiler flag wiring inside
  `tools/make/Makefile` and `tools/make/targets/cortex_m_generic_makefile.inc`
  (the `GLOBAL_KERNEL_OPTIMIZE` machinery, the `atfe` toolchain block,
  `CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY` define injection).

When inline drift is required:

1. **Keep the change minimal.** Add a single token to a list, not a new
   block of logic. If the change is more than ~5 lines, prefer a build-time
   patch under `helia/patches/`.
2. **Tag the line.** Add a comment containing the literal string
   `helia:` so it is grep-able during sync conflicts:
   ```cpp
   #if defined(CMSIS_NN) || defined(HELIA)  // helia: see kernels/helia/conv.cc
   ```
3. **Track it.** New inline drift goes into the inventory at
   [`helia/patches/inline_drift.md`](../patches/inline_drift.md).

## Build-time patches (`helia/patches/`)

For larger changes to upstream files that cannot be avoided, prefer a unified
diff under `helia/patches/applied/` over editing the upstream file directly.
The patch is applied by the helia build system (`tools/ci_build/build_helia.sh`)
via `helia/patches/apply.sh`. See `helia/patches/README.md` for authoring
conventions.

## Long-term reference: `pre-tflm-rebase-2026-05`

The annotated git tag `pre-tflm-rebase-2026-05` preserves the state of helia
main immediately before it was replanted on top of upstream
`tensorflow/tflite-micro`. Use it when investigating whether an Apollo-specific
customization was carried forward by the replant or needs to be re-applied as
a patch under `helia/patches/`.

```sh
git fetch --tags origin
git show pre-tflm-rebase-2026-05:<path>           # view file at tag
git diff pre-tflm-rebase-2026-05..main -- <path>  # diff vs current main
```
