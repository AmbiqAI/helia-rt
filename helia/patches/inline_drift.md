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

Three minimal hooks (~34 lines of inline drift, down from ~80):

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

3. `MICROLITE_TEST_RUNTIME_SRCS` / `MICROLITE_TEST_RUNTIME_OBJS` (issue
   #239): an empty-by-default hook a target makefile can add
   target-owned test-runtime sources to. Its objects are appended to the
   link line of the `$(BINDIR)%_test` pattern rule and, via
   [`helper_functions.inc`](../../tensorflow/lite/micro/tools/make/helper_functions.inc),
   of every `microlite_test` binary — and are deliberately kept out of
   `MICROLITE_LIB_OBJS` so they never enter
   `libtensorflow-microlite.a` or a generated project. Upstream has no
   hook for "code that must exist when a binary runs on a simulator but
   must not ship in the library": `MICROLITE_CC_SRCS` archives it,
   `MICROLITE_LIBS` puts it on the link line without building it. The
   only consumer today is
   `targets/cortex_m_corstone_300_makefile.inc`, which uses it for the
   strong fault handlers in
   `cortex_m_corstone_300/fault_handlers.cc`. Ordering constraint,
   documented at the definition: the `_OBJS` assignment must stay above
   `include tests.inc` and `kernels/Makefile.inc`, because make captures
   a prerequisite list when the rule is defined but expands the recipe
   at execution time.

Drop condition: upstream introduces a per-`OPTIMIZED_KERNEL_DIR` Makefile
include that runs early enough to extend `ADDITIONAL_DEFINES`, **and**
upstream picks up first-class `atfe` toolchain support, **and** upstream
provides a target-owned test-runtime source hook (at which point all
three hooks can be deleted). The third hook is a strong upstream-PR
candidate on its own: it is target-agnostic and inert unless a target
sets the variable.

## `tensorflow/lite/micro/tools/make/helper_functions.inc`

One hook, two lines (issue #239): `$(MICROLITE_TEST_RUNTIME_OBJS)` is
added to the prerequisites and to the link command of the
`microlite_test` template, which is what builds every example,
integration test, benchmark and non-kernel test binary. Without it the
hook described above would cover only the `$(BINDIR)%_test` pattern rule
in `tools/make/Makefile`, i.e. the kernel tests, and every other binary
would link without the target's test runtime. First divergence in this
file; no upstream extension point exists for the template's link line.

Drop condition: same as the `MICROLITE_TEST_RUNTIME_SRCS` hook above —
upstream provides a target-owned test-runtime source hook.

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

Three changes:

1. Adds `-C cpu0.semihosting-enable=1` to the FVP invocation so picolibc's
   `libsemihost` (used by the ATfE toolchain) can route stdout/stderr through
   SYS_WRITEC/SYS_WRITE0 to the FVP host. GCC and armclang builds use the
   MPS3 UART and are unaffected. Five-line change. Strong upstream-PR
   candidate.

   Drop condition: upstream enables semihosting unconditionally on
   Corstone-300.

2. Calls `testing/assert_tests_executed.sh` on the captured log inside the
   `grep -q "$PASS_STRING"` success branch, before declaring PASS (issue
   #231). Cannot be a hook: the pass/fail decision is made in this script and
   there is no upstream extension point inside it. Also a comment explaining
   why the FVP's own process exit status is not consulted.

   Drop condition: upstream stops treating a bare pass-string match as
   sufficient evidence and asserts a positive executed-case count itself.

3. Gives each binary its own log file
   (`${RESULTS_DIRECTORY}/$(basename ${BINARY_TO_TEST}).txt` instead of the
   shared `logs.txt`), so binaries running concurrently under `make -j` do
   not interleave into one file and have their banners parsed against the
   wrong binary. Two-line change, needed for (2) to mean anything.

   Drop condition: upstream adopts a per-binary log path (worth an upstream
   PR on its own — the shared path is a latent bug there too).

## `tensorflow/lite/micro/testing/assert_tests_executed.sh`

Helia-only **new file** in an upstream-owned directory (so not drift inside
an upstream file, but listed here because the top-of-file exemptions —
`kernels/helia/`, `tools/make/ext_libs/helia*.inc`, `tools/ci_build/*_helia.sh`,
`.github/workflows/helia_*.yml`, `helia/` — do not cover
`tensorflow/lite/micro/testing/`, and a sync reviewer needs to know it is
intentional).

Asserts that a test binary actually executed cases: it rejects a zero
executed-case count, a failure marker, two concatenated runs, and a non-zero
`Application exit code:` line, and it appends the per-leg tally consumed by
`tools/ci_build/test_helia.sh`. Exists because both micro-test frameworks
print `~~~ALL TESTS PASSED~~~` whenever the failure count is zero, including
when the executed count is also zero — which is how the ATfE legs were
vacuously green (issue #231).

Why here and not under `helia/`: it is invoked by
`testing/test_with_arm_corstone_300.sh` via `$(dirname "${BASH_SOURCE[0]}")`,
so it has to sit next to its only caller. It is inert for upstream callers —
without `HELIA_TEST_TALLY_FILE` it writes no tally.

Drop condition: upstream makes the executed-case assertion part of its own
test runner, at which point this file and its call site go together.

### Per-binary hang and fault handling (issue #239)

Separate block deliberately: `fix/atfe-test-registration-231` (#236) is
rewriting the paragraph above at the same time, so keeping this apart keeps
the two branches to a textual merge.

About 70 lines, most of it the comment justifying the budget. Three changes,
none of which upstream offers a hook for — the pass/fail decision, the log
path and the FVP invocation are all made inline in this script:

1. The FVP invocation is wrapped in `timeout --kill-after=30 120`
   (`FVP_TIMEOUT_SECONDS` overrides it). A timeout becomes an explicit named
   FAIL, instead of one binary silently consuming the whole leg budget.
2. The log path is per binary (`<binary>.txt`) rather than one shared
   `logs.txt`, so the log of a binary that failed survives the ~90 binaries
   `make -k` runs after it. #236 makes the same change for its own reason
   (concurrent `make -j` interleaving); same expression, so the two agree.
3. A `^FAULT:` line in the log fails the binary, checked before the
   pass-string grep. This is what makes the report from
   `cortex_m_corstone_300/fault_handlers.cc` visible to the harness, and it
   catches a fault that happens after the pass string has been printed.

`tools/ci_build/test_cortex_m_corstone_300.sh` (the upstream cmsis_nn leg)
uses this same script and inherits all three.

Drop condition: upstream bounds each FVP run itself and gives each binary its
own log. The `^FAULT:` check drops together with the fault handlers, i.e.
when the CMSIS startup for this target stops defining the fault vectors as
`while(1);`.

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
| `CONTRIBUTING.md` | heliaRT rebrand + Apollo SDK License preamble + redirected issue-tracker link. | Never. |
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
