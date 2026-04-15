#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# ns_local_build.sh
#
# Local sequential builder/packager that mirrors the NeuralSPOT asset workflows
# (same arch/toolchain/build combos) without GitHub Actions matrix.
#
# Flavors:
#   - helia-rt  -> OPTIMIZED_KERNEL_DIR=helia
#   - tflm      -> OPTIMIZED_KERNEL_DIR=cmsis_nn
#
# Examples:
#   tensorflow/lite/micro/tools/ci_build/ns_local_build.sh \
#     --flavor tflm --tag local-test --toolchains gcc
#
#   tensorflow/lite/micro/tools/ci_build/ns_local_build.sh \
#     --flavor helia-rt --tag v0.0.0-local
# ------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TFLM_SRC_DIR="${SCRIPT_DIR}/../../../../.."
BUILD_SCRIPT="${SCRIPT_DIR}/build_helia.sh"
NEURALSPOT_MODULE_MK="${TFLM_SRC_DIR}/neuralspot/module.mk"

DEFAULT_ARCHES=(cortex-m4+fp cortex-m55)
DEFAULT_BUILDS=(debug release release_with_logs)
DEFAULT_TOOLCHAINS=(gcc armclang)

FLAVOR="tflm"
TAG="local"
OUTDIR="${TFLM_SRC_DIR}/out/ns_local_build"
BUNDLE_PREFIX_OVERRIDE=""

ARM_UBL_LICENSE_IDENTIFIER="${ARM_UBL_LICENSE_IDENTIFIER:-}"

usage() {
  cat <<'USAGE'
Usage: ns_local_build.sh [options]

Options:
  --flavor <helia-rt|tflm>   Build flavor. helia-rt=helia, tflm=cmsis_nn (default: tflm)
  --tag <value>              Tag suffix used in bundle/zip names (default: local)
  -o, --outdir <path>        Output root for combos, bundle, and zip
  --arches a,b               Comma-separated arches (default: cortex-m4+fp,cortex-m55)
  --builds a,b               Comma-separated builds (default: debug,release,release_with_logs)
  --toolchains a,b           Comma-separated toolchains (default: gcc,armclang)
  --bundle-prefix <value>    Override bundle prefix (default from flavor)
  -h, --help                 Show this help

Outputs:
  <outdir>/combos/<arch>/<toolchain>/<build>/...
  <outdir>/<bundle-prefix>-<tag>/...
  <outdir>/<bundle-prefix>-<tag>.zip

Environment:
  ARM_UBL_LICENSE_IDENTIFIER   Required if toolchains include armclang
USAGE
}

parse_csv_into_array() {
  local value="$1"
  local out_var="$2"
  local parsed=()
  local item
  local item_quoted

  IFS=',' read -r -a parsed <<< "${value}"

  eval "${out_var}=()"
  for item in "${parsed[@]}"; do
    printf -v item_quoted '%q' "${item}"
    eval "${out_var}+=(${item_quoted})"
  done
}

ARCHES=("${DEFAULT_ARCHES[@]}")
BUILDS=("${DEFAULT_BUILDS[@]}")
TOOLCHAINS=("${DEFAULT_TOOLCHAINS[@]}")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flavor)
      FLAVOR="${2:?missing value for --flavor}"; shift 2 ;;
    --tag)
      TAG="${2:?missing value for --tag}"; shift 2 ;;
    -o|--outdir)
      OUTDIR="${2:?missing value for --outdir}"; shift 2 ;;
    --arches)
      parse_csv_into_array "${2:?missing value for --arches}" ARCHES; shift 2 ;;
    --builds)
      parse_csv_into_array "${2:?missing value for --builds}" BUILDS; shift 2 ;;
    --toolchains)
      parse_csv_into_array "${2:?missing value for --toolchains}" TOOLCHAINS; shift 2 ;;
    --bundle-prefix)
      BUNDLE_PREFIX_OVERRIDE="${2:?missing value for --bundle-prefix}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2 ;;
  esac
done

[[ -x "${BUILD_SCRIPT}" ]] || { echo "ERROR: Missing build script: ${BUILD_SCRIPT}" >&2; exit 3; }
[[ -n "${TAG}" ]] || { echo "ERROR: --tag cannot be empty" >&2; exit 2; }
[[ -n "${ARCHES[*]}" ]] || { echo "ERROR: --arches produced an empty list." >&2; exit 2; }
[[ -n "${BUILDS[*]}" ]] || { echo "ERROR: --builds produced an empty list." >&2; exit 2; }
[[ -n "${TOOLCHAINS[*]}" ]] || { echo "ERROR: --toolchains produced an empty list." >&2; exit 2; }

OPTIMIZED_KERNEL_DIR=""
BUNDLE_PREFIX=""
case "${FLAVOR}" in
  helia-rt|helios-rt|helia)
    OPTIMIZED_KERNEL_DIR="helia"
    BUNDLE_PREFIX="neuralspot-helia-rt"
    ;;
  tflm|cmsis-nn|cmsis_nn)
    OPTIMIZED_KERNEL_DIR="cmsis_nn"
    BUNDLE_PREFIX="neuralspot-tflm-rt"
    ;;
  *)
    echo "ERROR: Unsupported --flavor '${FLAVOR}'. Use helia-rt or tflm." >&2
    exit 2
    ;;
esac

if [[ -n "${BUNDLE_PREFIX_OVERRIDE}" ]]; then
  BUNDLE_PREFIX="${BUNDLE_PREFIX_OVERRIDE}"
fi

if [[ " ${TOOLCHAINS[*]} " == *" armclang " ]] && [[ -z "${ARM_UBL_LICENSE_IDENTIFIER}" ]]; then
  echo "ERROR: armclang builds require ARM_UBL_LICENSE_IDENTIFIER to be set." >&2
  exit 4
fi

COMBO_ROOT="${OUTDIR}/combos"
BUNDLE_REL="${BUNDLE_PREFIX}-${TAG}"
BUNDLE_DIR="${OUTDIR}/${BUNDLE_REL}"
ZIP_REL="${BUNDLE_REL}.zip"
ZIP_PATH="${OUTDIR}/${ZIP_REL}"

mkdir -p "${COMBO_ROOT}" "${OUTDIR}"
rm -rf "${BUNDLE_DIR}" "${ZIP_PATH}"

TOTAL=$(( ${#ARCHES[@]} * ${#TOOLCHAINS[@]} * ${#BUILDS[@]} ))
COUNT=0

echo "== ns_local_build =="
echo "  flavor     : ${FLAVOR}"
echo "  kernel dir : ${OPTIMIZED_KERNEL_DIR}"
echo "  outdir     : ${OUTDIR}"
echo "  total jobs : ${TOTAL}"

for arch in "${ARCHES[@]}"; do
  for toolchain in "${TOOLCHAINS[@]}"; do
    for build_type in "${BUILDS[@]}"; do
      COUNT=$((COUNT + 1))
      OUT_COMBO="${COMBO_ROOT}/${arch}/${toolchain}/${build_type}"
      echo "[${COUNT}/${TOTAL}] build_helia.sh -a ${arch} -t ${toolchain} -b ${build_type} -k ${OPTIMIZED_KERNEL_DIR}"
      "${BUILD_SCRIPT}" \
        -a "${arch}" \
        -b "${build_type}" \
        -t "${toolchain}" \
        -k "${OPTIMIZED_KERNEL_DIR}" \
        -o "${OUT_COMBO}"
    done
  done
done

mkdir -p "${BUNDLE_DIR}/lib"

echo "== Packaging bundle: ${BUNDLE_DIR} =="
find "${COMBO_ROOT}" -type f -path "*/lib/*.a" -print -exec cp -v {} "${BUNDLE_DIR}/lib/" \;

CAND="$(find "${COMBO_ROOT}" -type d -path '*/tflm' | sort | head -n1 || true)"
[[ -z "${CAND}" ]] && { echo "ERROR: No tflm tree found in ${COMBO_ROOT}" >&2; exit 5; }
cp -a "${CAND}/." "${BUNDLE_DIR}/"

NS_CMSIS_NN_DIR="${BUNDLE_DIR}/third_party/ns_cmsis_nn"
if [[ -d "${NS_CMSIS_NN_DIR}" ]]; then
  find "${NS_CMSIS_NN_DIR}" -type f \
    ! \( \
      \( -path "${NS_CMSIS_NN_DIR}/Include/*" -a \( -name '*.h' -o -name '*.hpp' \) \) \
      -o -name 'LICENSE' \
    \) \
    -delete
  find "${NS_CMSIS_NN_DIR}" -depth -type d -empty -delete
fi

if [[ -f "${NEURALSPOT_MODULE_MK}" ]]; then
  cp "${NEURALSPOT_MODULE_MK}" "${BUNDLE_DIR}/module.mk"
else
  echo "ERROR: Missing ${NEURALSPOT_MODULE_MK}" >&2
  exit 6
fi

{
  echo "${BUNDLE_PREFIX} ${TAG}"
  echo "Flavor: ${FLAVOR}"
  echo "KernelDir: ${OPTIMIZED_KERNEL_DIR}"
  echo
  echo "Libraries:"
  ls -1 "${BUNDLE_DIR}/lib"
} > "${BUNDLE_DIR}/MANIFEST.txt"

(
  mkdir -p "$(dirname "${ZIP_PATH}")"
  cd "${BUNDLE_DIR}"
  zip -r "${ZIP_PATH}" . >/dev/null
)

echo "== Done =="
echo "Bundle: ${BUNDLE_DIR}"
echo "Zip   : ${ZIP_PATH}"
find "${BUNDLE_DIR}" -maxdepth 2 -type f -print | sort
