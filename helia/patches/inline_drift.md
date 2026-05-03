# Inline Drift Inventory

This file tracks every helia-specific edit that lives **inside** an upstream
file (i.e., is not isolated under `kernels/helia/`, `tools/make/ext_libs/helia*.inc`,
`tools/ci_build/*_helia.sh`, `.github/workflows/helia_*.yml`, or `helia/`).

Whenever a sync from upstream lands, every entry here is a candidate for
review: did upstream take a similar change? Is the helia version still
needed? If yes, the entry stays. If no, the inline change can be removed.

Format per entry: file path, brief description, rationale for not using an
extension hook, and (if applicable) an upstream issue/PR link that would let
us drop the change.

## Shared kernel headers — `|| defined(HELIA)` guard tokens

**Resolved 2026-05.** All 14 kernel-header guards plus
`tensorflow/lite/micro/micro_profiler.cc` and
`tensorflow/lite/micro/kernels/unidirectional_sequence_lstm_test.cc` have
been reverted to upstream verbatim. The mechanism: helia kernels are
CMSIS-NN-backed (they `#include "Include/arm_nnfunctions.h"` and link
NS-CMSIS-NN), so `tools/make/ext_libs/helia.inc` now appends
`-DCMSIS_NN` to `CCFLAGS`/`CXXFLAGS`. The existing upstream
`#if defined(CMSIS_NN)` guards then do the right thing for helia builds
without per-header drift.

A side-effect: `tensorflow/lite/micro/kernels/fully_connected_test.cc:798`
(`#if !defined(XTENSA) && !defined(CMSIS_NN)`) now skips the Int16
PerChannel FC test for helia, matching the cmsis_nn-build behavior. helia
FC is a CMSIS-NN fork so this is the correct outcome.

## `tensorflow/lite/micro/micro_resource_variable.cc`

Wraps the four bulk copy/clear operations (Read, Allocate, Assign, ResetAll)
with a `#if defined(HELIA)` fast path that calls `arm_memcpy_s8` /
`arm_memset_s8` from CMSIS-NN. Preserves upstream's API.

Cannot be moved to `kernels/helia/` because it lives in core runtime.

Drop condition: upstream switches to `arm_memcpy_s8` / `arm_memset_s8`
unconditionally on Cortex-M55 (unlikely).

## `tensorflow/lite/micro/kernels/unidirectional_sequence_lstm_test.cc`

**Resolved 2026-05.** Reverted to upstream verbatim. The previous
`&& !defined(HELIA)` extension is now redundant: helia builds define
`-DCMSIS_NN` (see "Shared kernel headers" section above), so the existing
`#if !defined(CMSIS_NN)` guard already covers the helia case.

## `tensorflow/lite/micro/tools/make/Makefile`

Two minimal hooks (~19 lines of inline drift, down from ~80):

1. `GLOBAL_KERNEL_OPTIMIZE ?= SPEED` knob (defaults match upstream's
   static `KERNELS_OPTIMIZED_FOR_SPEED`) so the helia CI scripts
   (`build_helia.sh` / `test_helia.sh`) can flip SPEED↔SIZE without
   patching upstream. The per-kernel `CONV_OPT` / `FC_OPT` knobs and
   `CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY` opt-in live in
   [`tools/make/ext_libs/helia.inc`](../../tensorflow/lite/micro/tools/make/ext_libs/helia.inc),
   which appends `-D...` directly to `CCFLAGS` / `CXXFLAGS` (the
   `ADDITIONAL_DEFINES` capture in `COMMON_FLAGS` runs before
   `helia.inc` is sourced).
2. One-line `-Wno-error=nan-infinity-disabled` addition inside the
   existing armclang post-link branch so `-Werror` builds stay green
   on armclang. The same suppression for ATfE — plus
   `-Wno-error=unknown-attributes`, the `llvm-objcopy` override, the
   `$(BINDIR)%.bin` rule replacement, and the `test_helpers.o -O0`
   workaround for the clang-22 `BuildSimpleModelWithSubgraphsAndWhile`
   miscompile — lives in
   [`targets/atfe.inc`](../../tensorflow/lite/micro/tools/make/targets/atfe.inc),
   hooked via a 3-line `ifeq ($(TOOLCHAIN), atfe) … include … endif`
   block immediately after the upstream post-link block.

Drop condition: upstream introduces a per-`OPTIMIZED_KERNEL_DIR` Makefile
include that runs early enough to extend `ADDITIONAL_DEFINES`, **and**
upstream picks up first-class `atfe` toolchain support (at which point
both hooks can be deleted).

## `tensorflow/lite/micro/tools/make/targets/cortex_m_generic_makefile.inc`

Adds:

- Cross-platform `EXE := .exe` detection (Windows / Git Bash / WSL).
- armclang auto-download via `arm_clang_download.sh` and license activation
  via `armlm activate --code $(ARM_UBL_LICENSE_IDENTIFIER)`.
- `-fshort-enums` and `-gdwarf-4` for armclang.
- `atfe` (Arm Toolchain for Embedded — LLVM/Clang) toolchain block.

No upstream hook lets us add a third toolchain branch externally.

Drop condition: upstream adds first-class `atfe` and Windows-host support.

## `tensorflow/lite/micro/tools/make/targets/cortex_m_corstone_300_makefile.inc`

Reduced to three minimal hooks (the bulk of the atfe logic — ~85 lines —
lives in [`targets/cortex_m_corstone_300_atfe.inc`](../../tensorflow/lite/micro/tools/make/targets/cortex_m_corstone_300_atfe.inc),
which is helia-owned):

1. armclang auto-download / license activation block (~17 lines inside the
   existing `ifeq armclang` branch). Same pattern as `cortex_m_generic`.
2. `else ifeq ($(TOOLCHAIN), atfe)` branch that just `include`s the
   helia-owned `cortex_m_corstone_300_atfe.inc` (3 lines of inline drift).
3. `ifneq ($(TOOLCHAIN), atfe)` guard around
   `$(ETHOS_U_CORE_PLATFORM)/retarget.c` (4 lines). picolibc's `libsemihost`
   already provides stdio retargeting; adding `retarget.c` causes a link
   conflict under ATfE.

Drop condition: upstream adds first-class `atfe` toolchain support, picks
up the helia armclang download convention, and either drops `retarget.c`
or guards it against picolibc.

## `tensorflow/lite/micro/tools/benchmarking/show_meta_data.cc.template`

Adds `|| defined(AMBIQ)` to two pairs of `#if`/`#endif` guards so the
benchmarking metadata display path is enabled when downstream Ambiq Apollo
SDK consumers compile with `-DAMBIQ`.

Drop condition: upstream adds an `OPTIMIZED_KERNEL_DIR=helia`-aware
benchmarking template.

## `tensorflow/lite/micro/tools/ci_build/test_size.sh`

Single-line change: the size-comparison reference clones
`https://github.com/AmbiqAI/helia-rt.git` instead of upstream
`tensorflow/tflite-micro` so the size-regression baseline tracks helia main.

Drop condition: upstream parameterizes the reference URL.

## `tensorflow/lite/micro/kernels/Makefile.inc`

Adds an optional `-include $(MAKEFILE_DIR)/ext_libs/$(OPTIMIZED_KERNEL_DIR)_tests.inc`
hook so backends can register additional kernel tests without modifying
the upstream test list. helia's int16 hard_swish coverage is registered via
`tools/make/ext_libs/helia_tests.inc`. Strong upstream-PR candidate.

Drop condition: upstream merges the equivalent hook.

## `tensorflow/lite/micro/testing/test_with_arm_corstone_300.sh`

Adds `-C cpu0.semihosting-enable=1` to the FVP invocation so picolibc's
`libsemihost` (used by the ATfE toolchain) can route stdout/stderr through
SYS_WRITEC/SYS_WRITE0 to the FVP host. GCC and armclang builds use the
MPS3 UART and are unaffected. Five-line change. Strong upstream-PR candidate.

Drop condition: upstream enables semihosting unconditionally on Corstone-300.

## `.github/workflows/check_tflite_files.yml`

Replaces the upstream `tools/ci_build/check_tflite_files.sh` shell entry
point with an in-line `docker run … ghcr.io/ambiqai/helia-rt-ci:latest`
invocation so the file-allowlist check runs in the helia CI Docker image.

Cannot be moved: replacing with a sibling helia-named workflow would break
existing `pr_test.yml` PR-event wiring.

Drop condition: helia maintains its own `check_tflite_files.sh` and the
script auto-detects helia-rt vs tflite-micro at run time.

## `.github/workflows/issue_on_error.yml`

Two helia-specific changes:

1. Default `flag_label` changed from `bot:issue` to `ci:bot_issue` to
   match the helia-rt issue-tracker label scheme.
2. The error-reporting body calls `ci/issue_on_error.py` (a helia Python
   script) instead of upstream's inline `actions/github-script@v8` block.

Every helia and inherited workflow that calls `uses:
./.github/workflows/issue_on_error.yml` would need to be updated to point
at a sibling workflow before this could be moved.

Drop condition: helia switches all callers to a sibling
`helia_issue_on_error.yml`.

## `.github/workflows/sync.yml`

Disables the upstream-sync schedule (commented-out cron) and changes the
schedule-guard repo string from `tensorflow/tflite-micro` to
`AmbiqAI/helia-rt`. Action versions also pinned lower than upstream's
current. The workflow stays usable via `workflow_dispatch`.

Drop condition: helia replaces this with a sibling `helia_sync.yml`
(deferred — see Phase 4 plan).

## Top-level branding & policy files

These upstream-owned top-level files carry intentional helia rebrand /
licensing drift. They are tracked here so a future sync does not silently
re-apply the upstream copy.

| File | helia change | Drop condition |
| --- | --- | --- |
| `LICENSE` | Apache 2.0 replaced with the **Ambiq Apollo SDK License**. Required for distribution alongside the Ambiq Apollo SDK; cannot be reverted. | Never — keep helia version. |
| `README.md` | Full heliaRT rebrand (badges, intro, links, examples). | Never. |
| `CONTRIBUTING.md` | Heliart rebrand + Apollo SDK License preamble + redirected issue-tracker link. | Never. |
| `CODEOWNERS` | `/.github/` and `/ci/` reassigned from upstream `@veblush` to helia maintainers (`@advaitjain @rockyrhodes @suleshahid`). | Never. |
| `.gitignore` | Adds `build/`, `out/`, `.DS_Store`, `.aider*`, `neuralspot-*-local-*`, `neuralspot-*-local-*.zip`, `tflm-vanilla.zip`, `site/`. | Upstream adopts equivalents (won't happen for `neuralspot-*` / `tflm-vanilla.zip` — keep). |

## Top-level helia-only files in upstream-owned directories

These are **not drift inside an upstream file** but are listed here so a
sync conflict reviewer knows they are intentional. The canonical inventory
lives in [`helia/docs/repository_layout.md`](../docs/repository_layout.md)
under "Other approved helia-only locations".

- `nsx/` — heliaRT NSX module manifest (see repository_layout.md).
- `zephyr/` (top level) — Zephyr module manifest (see repository_layout.md).
- `zephyr_static_export.sh` — top-level Zephyr export driver.
- `pyproject.toml`, `uv.lock`, `mkdocs.yaml`, `release-please-config.json`, `.release-please-manifest.json`.
- `.devcontainer/`, `.github/stale.yml`.
- `ci/install_qemu.sh`, `ci/check_tflite_files.py`, `ci/issue_on_error.py`.

## `ci/` upstream-file drift

Resolved — all four `ci/` files (`Dockerfile.micro`, `install_bazelisk.sh`,
`install_buildifier.sh`, `sync_from_upstream_tf.sh`) and `ci/tflite_files.txt`
are now identical to `tflm/main`. Note that `ci/Dockerfile.micro` is dead
code in helia: the `helia-rt-ci` image is built from `.devcontainer/Dockerfile`
by `.github/workflows/helia_build_docker_image.yml`. We keep `Dockerfile.micro`
in sync with upstream solely to minimize sync conflicts.
