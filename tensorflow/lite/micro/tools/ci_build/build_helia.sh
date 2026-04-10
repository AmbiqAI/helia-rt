#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# build_helia.sh
#
# Build static library for a single {arch, build, toolchain} combination.
# Outputs to the given --outdir:
#   <OUTDIR>/
#     lib/libhelia-rt-<cmX>-<toolchain>-<build>.a
#     tflm/                      # reduced headers/sources for type-hinting
#
# Examples:
#   1) GCC, M55, release:
#      build_helia.sh -a cortex-m55 -b release -t gcc -o ./out/m55/gcc/release
#
#   2) ArmClang, M4+FP, debug:
#      build_helia.sh --arch cortex-m4+fp --build debug \
#                     --toolchain armclang --outdir ./out/m4/armclang/debug
#
#   3) GCC, M55, release using CMSIS-NN kernels:
#      build_helia.sh -a cortex-m55 -b release -t gcc -k cmsis_nn \
#                     -o ./out/m55/gcc/release
#
# Notes:
#   - For --toolchain armclang you must export ARM_UBL_LICENSE_IDENTIFIER.
#   - Designed for CI matrix runs: each combo writes to a unique OUTDIR.
# ------------------------------------------------------------------------------

set -euo pipefail

# ----------------------------- Defaults ---------------------------------------
ARCH="cortex-m55"
BUILD=""
TOOLCHAIN="gcc"
OUTDIR=""

ARM_UBL_LICENSE_IDENTIFIER="${ARM_UBL_LICENSE_IDENTIFIER:-}"

CO_PROCESSOR=
OPTIMIZED_KERNEL_DIR="helia"
TARGET="cortex_m_generic"

# --------------------------- Arg parsing --------------------------------------

usage() {
  cat <<EOF
Usage: $0 -a <arch> -b <build> -t <toolchain> -o <outdir>
       $0 --arch <arch> --build <build> --toolchain <toolchain> --outdir <outdir> \
          [--optimized-kernel-dir <dir>]

Required:
  -a, --arch        cortex-m4+fp | cortex-m55
  -b, --build       debug | release | release_with_logs
  -t, --toolchain   gcc | armclang
  -o, --outdir      Output directory for this build (unique per combo)

Other:
  -k, --optimized-kernel-dir  Kernel specialization dir (default: helia; e.g. cmsis_nn)
  -h, --help        Show this help

Environment:
  ARM_UBL_LICENSE_IDENTIFIER   Required when --toolchain armclang
  NS_CMSIS_NN_SSH_KEY          Optional; forwarded to make if present
EOF
}

# --- Parse args (short + long) ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--arch)
      [[ $# -ge 2 && ! "${2:-}" =~ ^- ]] || { echo "ERROR: $1 requires a value"; usage; exit 2; }
      ARCH="$2"; shift 2 ;;
    -b|--build)
      [[ $# -ge 2 && ! "${2:-}" =~ ^- ]] || { echo "ERROR: $1 requires a value"; usage; exit 2; }
      BUILD="$2"; shift 2 ;;
    -t|--toolchain)
      [[ $# -ge 2 && ! "${2:-}" =~ ^- ]] || { echo "ERROR: $1 requires a value"; usage; exit 2; }
      TOOLCHAIN="$2"; shift 2 ;;
    -o|--outdir)
      [[ $# -ge 2 && ! "${2:-}" =~ ^- ]] || { echo "ERROR: $1 requires a value"; usage; exit 2; }
      OUTDIR="$2"; shift 2 ;;
    -k|--optimized-kernel-dir)
      [[ $# -ge 2 && ! "${2:-}" =~ ^- ]] || { echo "ERROR: $1 requires a value"; usage; exit 2; }
      OPTIMIZED_KERNEL_DIR="$2"; shift 2 ;;
    -L|--arm-ubl-license-id|--arm-ubl-license-identifier)
      ARM_UBL_LICENSE_IDENTIFIER="${2:?missing value for --arm-ubl-license-id}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*)
      echo "Unknown option: $1"; usage; exit 2 ;;
    *)
      echo "Unexpected argument: $1"; usage; exit 2 ;;
  esac
done

# ---------------------------- Validate args -----------------------------------

[[ -n "${ARCH}"      ]] || { echo "Missing --arch"; usage; exit 2; }
[[ -n "${BUILD}"     ]] || { echo "Missing --build"; usage; exit 2; }
[[ -n "${TOOLCHAIN}" ]] || { echo "Missing --toolchain"; usage; exit 2; }
[[ -n "${OUTDIR}"    ]] || { echo "Missing --outdir"; usage; exit 2; }

# ArmClang license check (fail fast)
if [[ "${TOOLCHAIN}" == "armclang" && -z "${ARM_UBL_LICENSE_IDENTIFIER:-}" ]]; then
  echo "ERROR: --toolchain armclang requires env ARM_UBL_LICENSE_IDENTIFIER to be set." >&2
  exit 3
fi


# ---------------------------- Paths -------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM_SRC_DIR=${SCRIPT_DIR}/../../../../..
MAKE_DIR="${TFLM_SRC_DIR}/tensorflow/lite/micro/tools/make"
GEN_DIR="${TFLM_SRC_DIR}/gen"

# rm -rf "${OUTDIR}"
mkdir -p "${OUTDIR}/lib" "${OUTDIR}/tflm"


# ---- Inline requantize asm gating (enable only on M55 and M4+FP) ----
CMSIS_FLAG=""
case "${ARCH}" in
  cortex-m55|cortex-m4|cortex-m4+fp) CMSIS_FLAG="CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY=1" ;;
esac

echo "== Building Library =="
echo "   ARCH      : ${ARCH}"
echo "   BUILD     : ${BUILD}"
echo "   TOOLCHAIN : ${TOOLCHAIN}"
echo "   TARGET    : ${TARGET}"
echo "   KERNEL DIR: ${OPTIMIZED_KERNEL_DIR}"
echo "   OUTDIR    : ${OUTDIR}"
if [[ -n "${CMSIS_FLAG}" ]]; then
  echo "   CMSIS flag: ${CMSIS_FLAG}"
fi

# ---------------------------- Third-party downloads ---------------------------
make -f "${MAKE_DIR}/Makefile" \
  TARGET="${TARGET}" \
  TARGET_ARCH="${ARCH}" \
  TOOLCHAIN="${TOOLCHAIN}" \
  OPTIMIZED_KERNEL_DIR="${OPTIMIZED_KERNEL_DIR}" \
  third_party_downloads

# ---------------------------- Clean previous build outputs -------------------
make -f "${MAKE_DIR}/Makefile" clean

# ---------------------------- Build library -----------------------------------
JOBS="$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)"
make -j"${JOBS}" -f "${MAKE_DIR}/Makefile" \
  TARGET="${TARGET}" \
  TARGET_ARCH="${ARCH}" \
  TOOLCHAIN="${TOOLCHAIN}" \
  ARM_UBL_LICENSE_IDENTIFIER="${ARM_UBL_LICENSE_IDENTIFIER}" \
  OPTIMIZED_KERNEL_DIR="${OPTIMIZED_KERNEL_DIR}" \
  ${CMSIS_FLAG} \
  BUILD_TYPE="${BUILD}" \
  microlite

# ---------------------------- Copy outputs --------------------------------------
TARGET_SHORT="$(echo "${ARCH}" | sed -E 's/^cortex-m([0-9]+).*$/cm\1/')"
BUILD_NAME="$(echo "${BUILD}" | tr _ -)"
LIB_DIR_SUFFIX="${TARGET}_${ARCH}_${BUILD}"
if [[ -n "${OPTIMIZED_KERNEL_DIR}" ]]; then
  LIB_DIR_SUFFIX+="_${OPTIMIZED_KERNEL_DIR}"
fi
LIB_DIR_SUFFIX+="_${TOOLCHAIN}"
LIB_PATH="${GEN_DIR}/${LIB_DIR_SUFFIX}/lib/libtensorflow-microlite.a"

if [[ ! -f "${LIB_PATH}" ]]; then
  echo "ERROR: Missing library at ${LIB_PATH}" >&2
  exit 4
fi

cp "${LIB_PATH}" "${OUTDIR}/lib/libhelia-rt-${TARGET_SHORT}-${TOOLCHAIN}-${BUILD_NAME}.a"
echo "Copied lib -> ${OUTDIR}/lib/libhelia-rt-${TARGET_SHORT}-${TOOLCHAIN}-${BUILD_NAME}.a"

# ---- Create reduced TFLM tree into OUTDIR/tflm ----
python3 "${TFLM_SRC_DIR}/tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py" \
  --makefile_options "TARGET=${TARGET} TARGET_ARCH=${ARCH} OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR}" \
  "${OUTDIR}/tflm"

# ---- Summary ----
echo "== Build complete =="
echo "Contents:"
find "${OUTDIR}" -maxdepth 2 -type f -print
