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
Usage: helia_test.sh [options]

Options:
  -a, --arch <cortex-m55|cortex-m4|cortex-m4+fp|...>   Target CPU arch (default: cortex-m55)
  -t, --toolchain <gcc|clang|...>                      Toolchain (default: gcc)
  -O, --opt <SPEED|SIZE|BOTH>                          Kernel optimization (default: BOTH)
      --no-tests | --build-only                        Disable tests; build only
  -L, --arm-ubl-license-id, --arm-ubl-license-identifier <VALUE>
  -h, --help                                           Show this help

Examples:
  ./helia_test.sh
  ./helia_test.sh -a cortex-m4 -t gcc
  ./helia_test.sh --no-tests
  ./helia_test.sh -O SPEED --build-only
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

# ------------------------- Inline-asm gating ----------------------------------
enable_requantize_inline_asm=false
case "${TARGET_ARCH}" in
  cortex-m55|cortex-m4+fp|cortex-m4) enable_requantize_inline_asm=true ;;
  *)                                 enable_requantize_inline_asm=false ;;
esac

# ------------------------- Make args ------------------------------------------
MAKEFILE=tensorflow/lite/micro/tools/make/Makefile
common_args=(
  -f "${MAKEFILE}"
  CO_PROCESSOR="${CO_PROCESSOR}"
  OPTIMIZED_KERNEL_DIR="${OPTIMIZED_KERNEL_DIR}"
  TARGET="${TARGET}"
  TARGET_ARCH="${TARGET_ARCH}"
  TOOLCHAIN="${TOOLCHAIN}"
)

# Ensure third_party deps are present (download step)
readable_run make "${common_args[@]}" third_party_downloads

# Helper to produce argument list with (optional) kernel optimization + asm flag.
build_args_with_opts() {
  local opt="$1"
  local -a args=( "${common_args[@]}" )
  [[ -n "${opt}" ]] && args+=( GLOBAL_KERNEL_OPTIMIZE="${opt}" )
  if [[ "${enable_requantize_inline_asm}" == "true" ]]; then
    args+=( CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY=1 )
  fi
  if [[ -n "${ARM_UBL_LICENSE_IDENTIFIER}" ]]; then
    args+=( ARM_UBL_LICENSE_IDENTIFIER="${ARM_UBL_LICENSE_IDENTIFIER}" )
  fi
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

  if [[ "${RUN_TESTS}" -eq 1 ]]; then
    # Individual tests (keep as-is; fast failures, clearer logs)
    readable_run make -j"${JOBS}" "${ARGS[@]}" test_integration_tests_nnaed_conv_test
    readable_run make -j"${JOBS}" "${ARGS[@]}" test_integration_tests_nnaed_pad_test
    readable_run make -j"${JOBS}" "${ARGS[@]}" test_integration_tests_nnaed_leaky_relu_test
    readable_run make -j"${JOBS}" "${ARGS[@]}" test_integration_tests_nnaed_fully_connected_test

    # Full suite
    mapfile -t ARGS2 < <(build_args_with_opts "${OPTIMIZE_KERNELS_FOR}")
    readable_run make "${ARGS2[@]}" test
  else
    echo ">>> Skipping tests for ${OPTIMIZE_KERNELS_FOR} (build-only mode)."
  fi
done
