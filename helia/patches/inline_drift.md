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

The upstream commit these entries are measured against is recorded in
[`helia/UPSTREAM`](../UPSTREAM).

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

## `tensorflow/lite/micro/kernels/kernel_runner.h`

Stores the kernel registration **by value** (`const TFLMRegistration
registration_;`) instead of upstream's reference member
(`const TFLMRegistration& registration_;`). Three added comment lines
explain why; the constructor signature is unchanged
(`const TFLMRegistration&`), so every caller and the `.cc` are untouched.
A by-value member needs `TFLMRegistration` to be a complete type, which the
header previously got only transitively (`mock_micro_graph.h` ->
`micro_graph.h` -> `micro_common.h`), so the entry also adds a direct
`#include "tensorflow/lite/micro/micro_common.h"`. That include is part of
the same drift and goes away with it.

Rationale: a temporary bound to a *reference member* through a constructor
is not lifetime-extended, so `KernelRunner runner(Register_X(), ...)`
leaves the runner pointing at a dead stack slot. Under ATfE clang 22 for
cortex-m55 the compiler reuses that slot before `InitAndPrepare()`, which
corrupts `registration.init` and faults (see AmbiqAI/helia-rt#239).
`TFLMRegistration` is a 7-field POD of function pointers and ints, so the
copy is cheap and the class of bug cannot recur.

Cannot be moved to `kernels/helia/`: `KernelRunner` is upstream shared test
infrastructure used by every kernel test.

Drop condition: upstream tflite-micro adopts a by-value member (or otherwise
lifetime-extends the registration) in `kernel_runner.h`.

## `tensorflow/lite/micro/kernels/dequantize.h`, `tensorflow/lite/micro/kernels/dequantize.cc`, `tensorflow/lite/micro/kernels/dequantize_common.cc`, `tensorflow/lite/micro/kernels/dequantize_test.cc`, `tensorflow/lite/micro/kernels/xtensa/dequantize.cc`

Widens DEQUANTIZE to accept a `kTfLiteFloat16` input with a float32 output,
the form the LiteRT converter emits for fp16-PTQ weights (see
AmbiqAI/helia-rt#255). Five inline changes:

- `dequantize_common.cc`: `DequantizePrepare` admits `kTfLiteFloat16`. The
  scale/zero-point read stays unconditional; an f16 tensor carries `{0, 0}`
  params, which the widening path never uses.
- `dequantize.h`: adds the shared `Float16BitsToFloat32()` bit-expansion
  helper.
- `dequantize.cc`: adds the `kTfLiteFloat16` case to the reference Eval.
- `xtensa/dequantize.cc`: adds the same case to the xtensa Eval, using the
  same portable helper (there is no HiFi f16 widening primitive to call).
- `dequantize_test.cc`: adds the float16 golden and Prepare-rejection tests.

Prepare is shared with `kernels/helia/dequantize.cc`, so the type admission
cannot live under `kernels/helia/`; the helper and the reference Eval case
are kept inline so the helia kernel and the reference kernel produce
identical bits for every finite value, zero, infinity and NaN (signalling
NaNs are quieted on both paths), on hosts and on cores without f16
arithmetic. With `ARM_NN_ENABLE_F16` the helia kernel uses the hardware
convert instead, whose NaN and subnormal results follow FPSCR; the float16
golden test pins it on the cortex-m55 leg.

Drop condition: upstream TFLM accepts float16 DEQUANTIZE input.

## `tensorflow/lite/micro/kernels/cmsis_nn/svdf.cc`

First inline drift in `kernels/cmsis_nn/` (AmbiqAI/ns-cmsis-nn#312). Two
edits: `.size` is populated on the two SVDF scratch contexts with the exact
byte counts Prepare requested (upstream leaves the field indeterminate;
upstream ARM CMSIS-NN never reads it, an ns-cmsis-nn-backed build would),
and the `arm_svdf_s8` / `arm_svdf_state_s16_s8` calls are wrapped in
`TF_LITE_ENSURE_EQ(..., ARM_CMSIS_NN_SUCCESS)` instead of discarding the
status. The status half matches upstream's own `cmsis_nn/fully_connected.cc`
idiom, so on a sync conflict prefer keeping it and offering it upstream.

Cannot be moved to `kernels/helia/` because helia builds never compile this
directory; the caller-side fix has to live in the cmsis_nn kernel itself.

Drop condition: upstream tflite-micro takes an equivalent fix (the status
half is upstream-idiomatic today; the `.size` half matters upstream only if
ARM CMSIS-NN adopts a `ctx->size` contract).

## `tensorflow/lite/micro/tools/make/Makefile`

Four hooks, +158/-7 lines against upstream:

1. `GLOBAL_KERNEL_OPTIMIZE ?= SPEED` knob (defaults match upstream's
   static `KERNELS_OPTIMIZED_FOR_SPEED`) so the helia CI scripts
   (`build_helia.sh` / `test_helia.sh`) can flip SPEED↔SIZE without
   patching upstream. The per-kernel `CONV_OPT` / `FC_OPT` knobs and
   `CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY` opt-in live in
   [`tools/make/ext_libs/helia.inc`](../../tensorflow/lite/micro/tools/make/ext_libs/helia.inc),
   which appends `-D...` directly to `CCFLAGS` / `CXXFLAGS` (the
   `ADDITIONAL_DEFINES` capture in `COMMON_FLAGS` runs before
   `helia.inc` is sourced).
2. `-Werror=nan-infinity-disabled` inside the existing armclang
   post-link branch, so a NaN or infinity test that fast-math would fold
   away fails the armclang build instead. The same suppression for ATfE — plus
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

4. Per-toolchain optimization levels: `-Os`/`-O2`/`-O3` (core, kernels,
   third-party kernels) for gcc and ATfE, and `-Ofast -ffp-mode=full` for
   armclang. `-ffp-mode=full` must follow `-Ofast`, which otherwise
   disables NaN and infinity semantics; the comment at the definition
   records why.

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

Reduced to five minimal hooks (the bulk of the atfe logic — ~85 lines —
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
4. The gcc branch keeps the downloaded Arm GNU toolchain
   (`TARGET_TOOLCHAIN_ROOT := $(DOWNLOADS_DIR)/gcc_embedded/bin/`), as
   `cortex_m_generic` does. Upstream prefers an `arm-none-eabi-gcc` on
   `PATH`, and with none on `PATH` its `TARGET_TOOLCHAIN_ROOT ?=` is a no-op
   because the Makefile already defines the variable (empty), so the helia CI
   image would build with a bare `arm-none-eabi-gcc` that does not exist.
5. `ifeq` guard around the armlink `$(LIBDIR)/$(MICROLITE_LIB_NAME)(startup_$(ARM_CPU).o)`
   entry hint (3 lines), skipping it when `startup_$(ARM_CPU).c` is in
   `MICROLITE_TEST_RUNTIME_SRCS`. `test_helia_release_fp.sh` links a shipped
   archive with the startup file as a runtime object, so the library the hint
   names is never built and armlink stops with L6002U.

Drop condition: upstream adds first-class `atfe` toolchain support, picks
up the helia armclang download convention, and either drops `retarget.c`
or guards it against picolibc. Hook 4 goes when upstream's gcc branch
works without a compiler on `PATH`. Hook 5 goes when upstream only emits the
hint when the startup object is in the library.

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

Four changes:

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

4. Puts the log directory in the binary's build tree
   (`$(dirname ${BINARY_TO_TEST})/../logs`, i.e. `<gendir>/logs`) instead of
   `/tmp/${TARGET}_logs` (issue #368). The pass/fail decision is read back from
   the log, so two runs of one target on one machine (another toolchain, arch
   or checkout) must not write the same file. One code line; the comment above
   it is extended.

   Drop condition: upstream derives the log path from the build directory.

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

Resolved: identical to upstream. The workflow has no caller in helia-rt and is
no longer a CI-image pin point.

## `.github/workflows/issue_on_error.yml`

Two helia-specific changes:

1. Default `flag_label` changed from `bot:issue` to `ci:bot_issue` to
   match the helia-rt issue-tracker label scheme.
2. The error-reporting body calls `ci/issue_on_error.py` (a helia Python
   script) instead of upstream's inline `actions/github-script` block.

Every helia and inherited workflow that calls `uses:
./.github/workflows/issue_on_error.yml` would need to be updated to point
at a sibling workflow before this could be moved.

Drop condition: helia switches all callers to a sibling
`helia_issue_on_error.yml`.

## `.github/workflows/sync.yml`

Disables the upstream-sync schedule (commented-out cron) and changes the
schedule-guard repo string from `tensorflow/tflite-micro` to
`AmbiqAI/helia-rt`, and keeps helia's own Python and action steps (no
Bazel setup). The workflow stays usable via `workflow_dispatch`.

Drop condition: helia replaces this with a sibling `helia_sync.yml`
(deferred — see Phase 4 plan).

## `.github/workflows/ci.yml`

Upstream split this workflow into `run_*` / `suite_*` / `test_*` files
(tensorflow/tflite-micro#3307), which helia-rt does not carry; helia-rt keeps
the pre-split file as the `ci:run_full` workflow called by `tests_entry.yml`
and `run_ci.yml`. Beyond the general reshaping (every non-Bazel job runs in
the CI image and checks out `inputs.trigger-sha`, matrices are unrolled into
single jobs, `check_code_style` is disabled), helia-specific content includes:

1. The `ci-image` setup job pins `ghcr.io/ambiqai/helia-rt-ci` by digest
   (one of the pin points in `.github/workflows/README.md`).
2. `static_export_drift_check` (helia-only job) re-runs
   `zephyr_static_export.sh` and fails on a `third_party_static/` diff; it
   marks the container checkout as a git safe directory first.
3. `project_generation` puts the host `clang`/`clang++` ahead of the ATfE
   toolchain the CI image puts first on `PATH`, since the generated Makefile
   builds host code with `clang++`.

Drop condition: helia-rt adopts upstream's split workflows, or moves these
jobs to a sibling `helia_*.yml`.

## Action references in upstream-derived workflows

`log_binary_size_pr.yml`, `sync.yml` and `issue_on_error.yml` reference
actions by version tag; upstream pins them by commit SHA. Upstream's
read-only default `permissions` blocks are taken. `.github/dependabot.yml`
adds an `ignore` list for actions used only by upstream-vendored workflows
that helia disables, so dependabot does not open PRs against upstream YAML.

Drop condition: helia adopts SHA pinning for these workflows.

## Top-level branding & policy files

These upstream-owned top-level files carry intentional helia rebrand /
licensing drift. They are tracked here so a future sync does not silently
re-apply the upstream copy.

| File | helia change | Drop condition |
| --- | --- | --- |
| `LICENSE` | Apache 2.0 replaced with the **Ambiq Apollo SDK License**. Required for distribution alongside the Ambiq Apollo SDK; cannot be reverted. | Never — keep helia version. |
| `README.md` | Full heliaRT rebrand (badges, intro, links, examples). | Never. |
| `CONTRIBUTING.md` | heliaRT rebrand + Apollo SDK License preamble + redirected issue-tracker link. | Never. |
| `SECURITY.md` | Upstream redirect to TensorFlow's security policy replaced with the Ambiq reporting channel (`support.aitg@ambiq.com`, GitHub private vulnerability reporting once enabled). | Never. |
| `CODEOWNERS` | `/.github/` and `/ci/` reassigned from upstream `@veblush` to helia maintainers (`@advaitjain @rockyrhodes @suleshahid`). | Never. |
| `.gitignore` | Adds `build/`, `out/`, `.DS_Store`, `.aider*`, `neuralspot-*-local-*`, `neuralspot-*-local-*.zip`, `tflm-vanilla.zip`, `site/`. | Upstream adopts equivalents (won't happen for `neuralspot-*` / `tflm-vanilla.zip` — keep). |
| `pyproject.toml` | helia's `[project]` (uv tooling metadata) above upstream's `[tool.ruff]` configuration, plus an `__init__.py` F401 ignore. | Never for `[project]`; take upstream's ruff blocks on each sync. |

## Top-level helia-only files in upstream-owned directories

These are **not drift inside an upstream file** but are listed here so a
sync conflict reviewer knows they are intentional. The canonical inventory
lives in [`helia/docs/repository_layout.md`](../docs/repository_layout.md)
under "Other approved helia-only locations".

- `nsx/` — heliaRT NSX module manifest (see repository_layout.md).
- `zephyr/` (top level) — Zephyr module manifest (see repository_layout.md).
- `zephyr_static_export.sh` — top-level Zephyr export driver.
- `uv.lock`, `astro-site/`, `release-please-config.json`, `.release-please-manifest.json`.
- `tensorflow/lite/micro/tools/ci_build/test_cortex_m_generic.sh` — upstream
  deleted it (#3716); kept because the manual `cortex_m.yml` and
  `cortex_m_arm_compiler.yml` workflows call it.
- `tensorflow/lite/micro/tools/github/arm_virtual_hardware/` — upstream
  deleted it; used only by the dispatch-only `cortex_m_virtual_hardware.yml`.
- `.devcontainer/`, `.github/stale.yml`.
- `ci/install_qemu.sh`, `ci/check_tflite_files.py`, `ci/issue_on_error.py`.

## Upstream files helia-rt does not carry

The 21 upstream CI workflows (`check_bug_id`, `check_maintainer_edits`,
`generate_integration_tests`, `merge_group`, `pr_test`, `pypi_build`,
`run_*`, `suite_*`, `test_*`), `.github/pull_request_template.md` and the
three `docs/*.md` files are not imported; helia-rt runs its own workflows
and documentation. An upstream sync skips them.

## `ci/` upstream-file drift

Resolved — the upstream `ci/` files and `ci/tflite_files.txt` are identical
to the recorded upstream commit. Note that `ci/Dockerfile.micro` is dead
code in helia: the `helia-rt-ci` image is built from `.devcontainer/Dockerfile`
by `.github/workflows/helia_build_docker_image.yml`. We keep `Dockerfile.micro`
in sync with upstream solely to minimize sync conflicts.

## `tensorflow/lite/micro/kernels/space_to_batch_nd.cc` and test

Prepare initializes real-zero padding (INT8 output zero point, FP32 zero) and
requires matching INT8 input/output quantization. Regression tests cover padded
3D/4D inputs and nonzero persistent storage. See AmbiqAI/helia-rt#317.

This common correctness fix stays in the shared kernel: a helia-only override
or build_helia.sh patch would leave reference/CMSIS-NN, direct CMake, Bazel and
source consumers unfixed. The shared reference header is unchanged.
ci/sync_from_upstream_tf.sh preserves micro/, but a TFLM sync must reconcile
these two files. Drop this drift when the selected upstream TFLM pin includes
equivalent initialization, quantization checks and regression coverage.

## `tensorflow/lite/micro/kernels/ethos_u/ethosu.cc`

Compiles the Ethos-U kernel only under `HELIA_RT_ENABLE_ETHOSU`, which
`ETHOS_U` sets unless `HELIA_RT_DISABLE_ETHOSU` is defined; otherwise
`Register_ETHOSU()` is a stub returning `nullptr`. The kernel declares the
driver entry points it calls instead of including `ethosu_driver.h`. The
source-list builds (CMake, Zephyr, NSX) compile this file without the
Ethos-U driver. See AmbiqAI/helia-rt#172 and #213.

Drop condition: upstream gates the kernel on driver availability.

## `tensorflow/lite/micro/testing/micro_test.h`, `micro_test_v2.h`

The near and float-equal checks treat NaN as a mismatch unless both values
are NaN, so a kernel that returns NaN cannot pass a tolerance check. See
AmbiqAI/helia-rt#338.

Drop condition: upstream's macros become NaN-aware.

## Kernel tests with helia cases

`activations`, `add`, `batch_matmul`, `concatenation`, `conv`,
`depthwise_conv`, `fully_connected`, `logistic`, `maximum_minimum`, `mul`,
`pad`, `pooling`, `reshape`, `softmax`, `svdf`, `tanh`, `transpose`,
`transpose_conv` and `unidirectional_sequence_lstm` `_test.cc` add
`#if ARM_NN_ENABLE_F16` float16 cases, keep the kernel registration alive
for the test's lifetime, and (LSTM) add stateful streaming coverage. These
run against the helia kernels on the M55 legs.

Drop condition: none; re-merge on each sync.

## Download scripts

`bash_helpers.sh` adds `wget_with_retries` and the seed helpers
(`check_seed`, `write_seed`). `download_and_extract.sh` reuses a download only
when it is complete and its URL and checksum match the seed; the seed is
written into the staging directory before the completion marker, so it
survives the move into place. `ext_libs/cmsis_download.sh` and
`cmsis_nn_download.sh` use https, the retrying download and the seed check,
so a pin change re-downloads instead of reusing a stale tree.
`xtensa_download.sh` uses https and the retrying download;
`xtensa_ndsp_download.sh`, `arm_gcc_download.sh`,
`corstone_300_download.sh` and `renode_download.sh` use the retrying
download. `corstone_300_download.sh` also drops a stray `fi` that makes
upstream's copy fail to parse when the FVP has to be downloaded.

Drop condition: upstream re-downloads on pin changes and retries transient
failures; the `fi` fix drops when upstream's script parses.

## `tensorflow/lite/micro/tools/make/targets/bluepill_makefile.inc`

Writes the link map to `$(GENDIR)` instead of `gen/`, so parallel builds with
different `BASE_GENDIR` values do not share one map.

Drop condition: upstream writes target maps under `GENDIR`.

## `tensorflow/lite/micro/kernels/cmsis_nn/maximum_minimum.cc`

MAXIMUM/MINIMUM with any operand above rank 4 run the reference loop. The
CMSIS-NN kernels take 4-D dims: before upstream #3563 the int8 path collapsed
rank 5 to one element and returned wrong values; after it, `ExtendedShape(4,
...)` aborts. See AmbiqAI/helia-rt#356.

Drop condition: upstream's CMSIS-NN MAXIMUM/MINIMUM handle rank above 4.

## `tensorflow/lite/micro/kernels/arg_min_max.cc`

Floating-point ARG_MAX/ARG_MIN use a NaN-aware comparator with the semantics
of upstream's `reference_ops::GetComparefunction` (a NaN never wins, as in
TensorFlow/LiteRT), as a functor rather than a `std::function`; integer types
keep the plain comparators. Upstream's micro kernel passes `GreaterFn`/`LessFn`
for every type, which leave a leading NaN in place.
`kernels/arg_min_max_test.cc` adds the float32 NaN cases. See
AmbiqAI/helia-rt#359.

This correctness fix stays in the shared kernel: a build_helia.sh patch would
leave reference/CMSIS-NN, direct CMake, Bazel and source consumers unfixed.

Drop condition: upstream's micro ARG_MAX/ARG_MIN use the NaN-aware comparator.

## `tensorflow/lite/micro/kernels/elementwise.cc`

`GenericPrepare` and `PrepareAbsRsqrt` name their type-predicate template
parameter and call it. Upstream leaves it unnamed, so `IsSupportedType(type)`
is a function-pointer cast and every dtype passes Prepare. On a rejected dtype
both release their input and output temps before returning, as the success
path does. `IsRsqrtSupportedType` admits int16, which `RsqrtEval` already
handles. `kernels/elementwise_test.cc`
adds a rejected dtype for SIN, ABS, RSQRT and LOGICAL_NOT. The helia copy has
the same change (AmbiqAI/helia-rt#342). See AmbiqAI/helia-rt#376.

This correctness fix stays in the shared kernel: a build_helia.sh patch would
leave reference/CMSIS-NN, direct CMake, Bazel and source consumers unfixed.

Drop condition: upstream's elementwise Prepare calls its type predicate and
releases its temps on that path.

## `tensorflow/lite/micro/kernels/pad_common.cc`

PAD/PADV2 Prepare rejects input and output ranks that differ before it
indexes `input->dims` by the output rank and the paddings by the input rank;
upstream's loop reads past either array for a malformed model. It also
rejects a negative padding, as TensorFlow Lite's `kernels/pad.cc` does
("Pad value has to be greater than equal to 0"); one turns into out-of-range
indices and copy sizes at Eval. `kernels/pad_test.cc` adds a case for each
rank direction and a negative padding in each slot of a rank-2 PAD. Both
rejections release Prepare's temporaries first. See AmbiqAI/helia-rt#350 and
AmbiqAI/helia-rt#372.

This correctness fix stays in the shared kernel: a build_helia.sh patch would
leave reference/CMSIS-NN, direct CMake, Bazel and source consumers unfixed.

Drop condition: upstream's PAD Prepare checks that the ranks match and that
paddings are non-negative.

## Dynamic-batch shapes: `kernels/space_to_batch_nd.cc`, `kernels/batch_to_space_nd.cc`, `kernels/depthwise_conv_common.cc`, `kernels/conv_common.cc`, `kernels/kernel_util.{h,cc}`, `micro_allocator.cc`, `micro_allocation_info.cc` and tests

A model exported with a dynamic batch stores batch 1 for tensors whose real
batch comes from the graph: the SPACE_TO_BATCH_ND output of a dilated
convolution, and the DEPTHWISE_CONV_2D or CONV_2D output after it. TensorFlow
Lite resizes these at runtime; TFLM computed into the stored shape, so S2B
filled one batch, the convolution ran on batch 1 (its batch check is a DCHECK)
and BATCH_TO_SPACE_ND wrote nothing, giving silent wrong outputs. See
AmbiqAI/helia-rt#407.

- SPACE_TO_BATCH_ND and BATCH_TO_SPACE_ND Prepare compute the output shape from
  constant block shape and paddings/crops, reject any non-batch mismatch, and
  write the computed batch into an output that stores the placeholder batch 1
  (`CreateWritableTensorDimsWithCopy`, as depth_to_space and gather do).
  BATCH_TO_SPACE_ND rejects an input batch the block product does not divide,
  so an op between the pair that kept the stored batch is caught. Non-constant
  block shape or paddings keep upstream's behaviour.
- `micro::MatchOutputBatchToInput` makes a convolution's output batch follow its
  input batch when the output stores batch 1, and rejects any other mismatch;
  the reference DEPTHWISE_CONV_2D and CONV_2D Prepare call it, as do the helia
  overrides.
- A batch rewrite also scales the tensor's `bytes`, and `MicroAllocator` sizes
  the persistent and temp `TfLiteTensor`s it rebuilds from rewritten eval dims,
  so `output()->bytes` matches a resized graph output.
- `AllocationInfoBuilder` rejects an offline-planned tensor whose dims were
  rewritten at Prepare: the offline plan was computed for the stored shape.
- The four kernel tests cover the placeholder batch, the rejections, the new
  byte size, temporary release and untouched model dims (the conv and depthwise
  cases are skipped on the upstream cmsis_nn backend);
  `micro_allocator_test.cc` covers the byte sizing and the offline-plan
  rejection; `depthwise_conv_test.cc` gains an optional validation length on
  its per-channel helper and a helia-only case for the depthwise batch limit
  (heliaCORE takes the batch as `uint16_t`).

This correctness fix stays in the shared kernels: a helia-only override would
leave reference, direct CMake, Bazel and source consumers wrong. The cmsis_nn
backend's own convolution Prepare is upstream and unchanged.

Drop condition: upstream TFLM checks SPACE_TO_BATCH_ND/BATCH_TO_SPACE_ND output
shapes at Prepare and propagates a dynamic batch through the convolution.
