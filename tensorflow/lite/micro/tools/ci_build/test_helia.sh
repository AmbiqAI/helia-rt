#!/usr/bin/env bash
# Build (and optionally test) TFLM for a given target arch + toolchain.
# Defaults: arch=cortex-m55, toolchain=gcc, optimize=BOTH, tests=ON
# Examples:
#   ./test_helia.sh                         # build + test, both SPEED and SIZE
#   ./test_helia.sh --no-tests              # build only (both variants)
#   ./test_helia.sh -a cortex-m4 -t gcc     # build + test for m4
#   ./test_helia.sh -O SPEED --build-only   # build only, SPEED variant

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/../../../../.."
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

# ----------------------------- Defaults ---------------------------------------
TARGET_ARCH="cortex-m55"
TOOLCHAIN="gcc"
OPT_CHOICE="BOTH"     # SPEED | SIZE | BOTH
RUN_TESTS=1           # 1=enabled (default), 0=disabled

ARM_UBL_LICENSE_IDENTIFIER="${ARM_UBL_LICENSE_IDENTIFIER:-}"

CO_PROCESSOR=
OPTIMIZED_KERNEL_DIR=helia
TARGET=cortex_m_corstone_300

# --------------------------- Arg parsing --------------------------------------
usage() {
  cat <<'USAGE'
Usage: test_helia.sh [options]

Options:
  -a, --arch <cortex-m55|cortex-m4|cortex-m4+fp|...>   Target CPU arch (default: cortex-m55)
  -t, --toolchain <gcc|armclang|atfe>                  Toolchain (default: gcc)
  -O, --opt <SPEED|SIZE|BOTH>                          Kernel optimization (default: BOTH)
      --no-tests | --build-only                        Disable tests; build only
  -L, --arm-ubl-license-id, --arm-ubl-license-identifier <VALUE>
  -h, --help                                           Show this help

Examples:
  ./test_helia.sh
  ./test_helia.sh -a cortex-m4 -t gcc
  ./test_helia.sh --no-tests
  ./test_helia.sh -O SPEED --build-only
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--arch)       TARGET_ARCH="${2:?missing value for --arch}"; shift 2 ;;
    -t|--toolchain)  TOOLCHAIN="${2:?missing value for --toolchain}"; shift 2 ;;
    -O|--opt)
      OPT_CHOICE="${2:?missing value for --opt}"; shift 2 ;;
    --no-tests|--build-only)
      RUN_TESTS=0; shift ;;
     -L|--arm-ubl-license-id|--arm-ubl-license-identifier)
      ARM_UBL_LICENSE_IDENTIFIER="${2:?missing value for --arm-ubl-license-id}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

case "${OPT_CHOICE}" in
  SPEED|SIZE|BOTH) : ;;
  *) echo "Invalid --opt '${OPT_CHOICE}'. Use SPEED|SIZE|BOTH." >&2; exit 2 ;;
esac

case "${TOOLCHAIN}" in
  gcc|armclang|atfe) ;;
  *) echo "Invalid --toolchain '${TOOLCHAIN}'. Use gcc|armclang|atfe." >&2; exit 2 ;;
esac

# ATfE: prefer pre-installed copy at /opt/atfe (CI image) over runtime download.
ATFE_OVERRIDE=()
if [[ "${TOOLCHAIN}" == "atfe" ]]; then
  ATFE_PREINSTALL="${ATFE_PREINSTALL:-/opt/atfe}"
  if [[ -x "${ATFE_PREINSTALL}/bin/clang" ]]; then
    echo "Using pre-installed ATfE at ${ATFE_PREINSTALL}"
    ATFE_OVERRIDE=("TARGET_TOOLCHAIN_ROOT=${ATFE_PREINSTALL}/bin/")
  fi
fi

# ------------------------- Inline-asm gating ----------------------------------
enable_requantize_inline_asm=false
case "${TARGET_ARCH}" in
  cortex-m55|cortex-m4+fp|cortex-m4) enable_requantize_inline_asm=true ;;
  *)                                 enable_requantize_inline_asm=false ;;
esac

# ------------------------- License credential ---------------------------------
# Hand the Arm user-based license identifier to make through the environment
# rather than the command line: readable_run echoes every make invocation, so
# a command-line argument put the credential straight into the log. make reads
# environment variables as make variables and no makefile assigns this one, so
# the value still reaches the activation check.
#
# One deliberate difference from passing it per-invocation: the export also
# reaches the `third_party_downloads` make call below, so with --toolchain
# armclang `armlm activate` now runs there too. Activation is idempotent, so
# this costs one extra call and changes nothing else.
export ARM_UBL_LICENSE_IDENTIFIER

# ------------------------- Make args ------------------------------------------
MAKEFILE=tensorflow/lite/micro/tools/make/Makefile
common_args=(
  -f "${MAKEFILE}"
  CO_PROCESSOR="${CO_PROCESSOR}"
  OPTIMIZED_KERNEL_DIR="${OPTIMIZED_KERNEL_DIR}"
  TARGET="${TARGET}"
  TARGET_ARCH="${TARGET_ARCH}"
  TOOLCHAIN="${TOOLCHAIN}"
  ${ATFE_OVERRIDE[@]+"${ATFE_OVERRIDE[@]}"}
)

# Ensure third_party deps are present (download step)
readable_run make "${common_args[@]}" third_party_downloads

# ------------------------- ccache wrappers (optional) -------------------------
# When CCACHE_WRAP_DIR is set in the environment (typically by CI), generate
# wrapper scripts that route the toolchain through ccache, and pass the
# resulting CC/CXX to make.
#
# Wrappers are necessary because helia's collect_meta_data.sh execs $CC as a
# single-token argv[0], so passing  CC="ccache /path/to/gcc"  via make would
# fail with "No such file or directory". The wrappers keep CC/CXX as a single
# absolute path while still routing through ccache.
#
# Local benchmarking on the helia-rt-ci dev container (32-core) showed:
#   - cold (cache empty)  build:  ~39 s
#   - hot  (cache present) build: ~13 s
#   - direct cache-hit rate on hot rebuild: 100% (1164/1164)
# i.e. ~3x wall-clock speedup once the cache is warm.
if [[ -n "${CCACHE_WRAP_DIR:-}" ]] && command -v ccache >/dev/null 2>&1; then
  mkdir -p "${CCACHE_WRAP_DIR}"
  case "${TOOLCHAIN}" in
    gcc)
      _real_cc="${ROOT_DIR}/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi-gcc"
      _real_cxx="${ROOT_DIR}/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi-g++"
      _wrap_cc="${CCACHE_WRAP_DIR}/arm-none-eabi-gcc"
      _wrap_cxx="${CCACHE_WRAP_DIR}/arm-none-eabi-g++"
      ;;
    atfe)
      _real_cc="${ATFE_PREINSTALL:-/opt/atfe}/bin/clang"
      _real_cxx="${ATFE_PREINSTALL:-/opt/atfe}/bin/clang++"
      _wrap_cc="${CCACHE_WRAP_DIR}/clang"
      _wrap_cxx="${CCACHE_WRAP_DIR}/clang++"
      ;;
    *)
      echo "ccache: TOOLCHAIN=${TOOLCHAIN} unsupported, skipping wrapper" >&2
      _wrap_cc=
      ;;
  esac
  if [[ -n "${_wrap_cc:-}" ]]; then
    cat > "${_wrap_cc}" <<EOF
#!/usr/bin/env bash
exec ccache "${_real_cc}" "\$@"
EOF
    cat > "${_wrap_cxx}" <<EOF
#!/usr/bin/env bash
exec ccache "${_real_cxx}" "\$@"
EOF
    chmod +x "${_wrap_cc}" "${_wrap_cxx}"
    common_args+=( CC="${_wrap_cc}" CXX="${_wrap_cxx}" )
    echo "==> ccache: routing $(basename "${_real_cc}")/$(basename "${_real_cxx}") through ccache" >&2
    ccache --version | head -1 >&2 || true
  fi
fi

# Helper to produce argument list with (optional) kernel optimization + asm flag.
build_args_with_opts() {
  local opt="$1"
  local -a args=( "${common_args[@]}" )
  [[ -n "${opt}" ]] && args+=( GLOBAL_KERNEL_OPTIMIZE="${opt}" )
  if [[ "${enable_requantize_inline_asm}" == "true" ]]; then
    args+=( CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY=1 )
  fi
  # ARM_UBL_LICENSE_IDENTIFIER is deliberately NOT added here. These args are
  # echoed by readable_run (and by any caller that logs the make command), and
  # the identifier is a credential. It is exported instead, which make picks up
  # as a regular variable — see the export near the top of this script.
  printf '%q\n' "${args[@]}"
}

# Decide optimization variants to build
variants=()
if [[ "${OPT_CHOICE}" == "BOTH" ]]; then
  variants=(SPEED SIZE)
else
  variants=("${OPT_CHOICE}")
fi

# nproc fallback (e.g., mac w/ gnu coreutils not present)
JOBS="$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)"

echo "==> TARGET_ARCH=${TARGET_ARCH} TOOLCHAIN=${TOOLCHAIN} OPT=${OPT_CHOICE} TESTS=$([[ $RUN_TESTS -eq 1 ]] && echo ON || echo OFF)"

for OPTIMIZE_KERNELS_FOR in "${variants[@]}"; do
  echo "=== Building with ${OPTIMIZE_KERNELS_FOR} (TARGET_ARCH=${TARGET_ARCH}) ==="

  readable_run make -f "${MAKEFILE}" clean

  # Build
  mapfile -t ARGS < <(build_args_with_opts "${OPTIMIZE_KERNELS_FOR}")
  readable_run make -j"${JOBS}" "${ARGS[@]}" build

  # ---- FP symbol link-probe (issue #227) -------------------------------------
  # Runs on every leg, right after the library exists and before the test
  # binaries. It links a generated program against the archive that references
  # every ns-cmsis-nn float entry point the helia kernels call, so a missing,
  # renamed, or itself-unresolvable FP symbol fails here even on the legs whose
  # executed FP coverage is still thin. Cheap (one compile + one link) and
  # needs no FVP, which is why it can be unconditional.
  GENDIR="$(make "${ARGS[@]}" list_gendir 2>/dev/null | tail -1)"
  readable_run tensorflow/lite/micro/tools/ci_build/fp_symbol_link_probe.sh \
    --lib "${GENDIR}lib/libtensorflow-microlite.a" \
    --arch "${TARGET_ARCH}" \
    --toolchain "${TOOLCHAIN}" \
    --target "${TARGET}" \
    --label "${TARGET_ARCH}-${TOOLCHAIN}-${OPTIMIZE_KERNELS_FOR}"

  if [[ "${RUN_TESTS}" -eq 1 ]]; then
    # ---- executed-case tally (issue #231) ------------------------------------
    # testing/assert_tests_executed.sh fails any individual binary that
    # executes 0 test cases. This file collects what each binary actually ran
    # so the leg can also report -- and optionally floor-check -- its total.
    # That catches the other half of the failure mode: binaries silently
    # dropping out of the suite entirely, which no per-binary check can see.
    # An explicit template is required: BSD/macOS mktemp has no default
    # template, so a bare `mktemp` fails there and would take this script
    # down under `set -e` on a local run.
    HELIA_TEST_TALLY_FILE="$(mktemp "${TMPDIR:-/tmp}/helia_test_tally.XXXXXX")"
    export HELIA_TEST_TALLY_FILE
    : > "${HELIA_TEST_TALLY_FILE}"

    # Individual tests (keep as-is; fast failures, clearer logs)
    readable_run make -j"${JOBS}" "${ARGS[@]}" test_integration_tests_nnaed_conv_test
    readable_run make -j"${JOBS}" "${ARGS[@]}" test_integration_tests_nnaed_pad_test
    readable_run make -j"${JOBS}" "${ARGS[@]}" test_integration_tests_nnaed_leaky_relu_test
    readable_run make -j"${JOBS}" "${ARGS[@]}" test_integration_tests_nnaed_fully_connected_test

    # Full suite.
    #
    # -k (keep-going): report every failing test binary instead of stopping at
    # the first one. Without it make aborts the whole stream at the first
    # non-zero exit, so a known-failing test silently prevents every later test
    # binary from running at all -- in run 33515504619 a red activation test
    # stopped the suite before the LSTM and elementwise tests were ever
    # invoked, and their results were indistinguishable from "passed" in the
    # log. -k does not change the leg's verdict: make still exits non-zero if
    # anything failed, so the job is still red. It only makes the failure set
    # complete.
    #
    # The cost of -k is that failures are now scattered through a ~12,700-line
    # log instead of sitting at the end, which makes it easy to misattribute a
    # one-off FVP timeout to the same cause as a real assertion failure. Tee
    # the run and print a consolidated list of failing targets afterwards.
    mapfile -t ARGS2 < <(build_args_with_opts "${OPTIMIZE_KERNELS_FOR}")
    suite_log="$(mktemp -t helia_suite_XXXXXX.log)"

    # errexit is disabled only around the pipeline so a failing suite reaches
    # the summary below instead of aborting the script. make's own status is
    # read from PIPESTATUS[0] immediately, before any other command can reset
    # it -- notably before `set -e`, which would clobber it.
    set +e
    readable_run make -k "${ARGS2[@]}" test 2>&1 | tee "${suite_log}"
    suite_status=${PIPESTATUS[0]}
    set -e

    echo "==> Failing test targets (${TARGET_ARCH}/${TOOLCHAIN}/${OPTIMIZE_KERNELS_FOR}):"
    grep -E '^make(\[[0-9]+\])?: \*\*\* \[.*\] Error ' "${suite_log}" \
      | sed -E 's/^make(\[[0-9]+\])?: \*\*\* \[[^]]*: ([^]]*)\] Error.*/  \2/' \
      | sort -u \
      || echo "  (none)"
    rm -f "${suite_log}"

    if [[ "${suite_status}" -ne 0 ]]; then
      echo "ERROR: helia test suite failed (make exit ${suite_status})" >&2
      exit "${suite_status}"
    fi
    # ---- per-leg executed-case floor (issue #231) ----------------------------
    # Zero is always a failure: a leg that ran no test binary, or whose
    # binaries collectively executed no case, is not a passing leg however
    # green make looked. HELIA_MIN_TEST_BINARIES / HELIA_MIN_EXECUTED_CASES
    # are optional and unset by default -- set them in the workflow to pin a
    # real floor once CI has reported the actual per-leg numbers. They are
    # deliberately not hard-coded here: a guessed floor is either useless or
    # a false alarm waiting to happen.
    # Tally line format: <binary-name><TAB><executed-cases><TAB><kind>.
    # 'frameworkless' binaries (see FRAMEWORKLESS_BINARIES in
    # testing/assert_tests_executed.sh) ran but have no case count, so they
    # count as binaries and contribute zero cases. Report them separately so
    # "N binaries, M cases" cannot be read as "every binary reported cases".
    tally_binaries="$(wc -l < "${HELIA_TEST_TALLY_FILE}" | tr -d '[:space:]')"
    tally_frameworkless="$(awk -F'\t' '$3 == "frameworkless" {n += 1} \
                           END {print n + 0}' "${HELIA_TEST_TALLY_FILE}")"
    tally_cases="$(awk -F'\t' '{s += $2} END {print s + 0}' \
                   "${HELIA_TEST_TALLY_FILE}")"
    echo "==> executed-case tally for ${TARGET_ARCH}/${TOOLCHAIN}/${OPTIMIZE_KERNELS_FOR}:" \
         "${tally_binaries} binaries (${tally_frameworkless} framework-less," \
         "no case count), ${tally_cases} test cases"

    if [[ "${tally_binaries}" -eq 0 || "${tally_cases}" -eq 0 ]]; then
      echo "::error ::${TARGET_ARCH}/${TOOLCHAIN}/${OPTIMIZE_KERNELS_FOR}:" \
           "the suite executed ${tally_cases} test cases across" \
           "${tally_binaries} binaries. A green leg that ran nothing is a" \
           "harness failure, not a pass (issue #231)."
      exit 1
    fi
    if [[ -n "${HELIA_MIN_TEST_BINARIES:-}" \
          && "${tally_binaries}" -lt "${HELIA_MIN_TEST_BINARIES}" ]]; then
      echo "::error ::only ${tally_binaries} test binaries ran; floor is" \
           "${HELIA_MIN_TEST_BINARIES} (HELIA_MIN_TEST_BINARIES)."
      exit 1
    fi
    if [[ -n "${HELIA_MIN_EXECUTED_CASES:-}" \
          && "${tally_cases}" -lt "${HELIA_MIN_EXECUTED_CASES}" ]]; then
      echo "::error ::only ${tally_cases} test cases executed; floor is" \
           "${HELIA_MIN_EXECUTED_CASES} (HELIA_MIN_EXECUTED_CASES)."
      exit 1
    fi
    rm -f "${HELIA_TEST_TALLY_FILE}"
    unset HELIA_TEST_TALLY_FILE
  else
    echo ">>> Skipping tests for ${OPTIMIZE_KERNELS_FOR} (build-only mode)."
  fi
done
