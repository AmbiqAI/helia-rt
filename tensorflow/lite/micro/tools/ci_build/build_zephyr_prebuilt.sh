#!/usr/bin/env bash
# Build a single prebuilt Zephyr archive, embedding the Ambiq-optimized backend.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_AMBIQ="${SCRIPT_DIR}/build_ambiq.sh"

ARCH=""
BUILD=""
TOOLCHAIN=""
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--arch)
      ARCH="${2:?missing value for --arch}"
      shift 2
      ;;
    -b|--build)
      BUILD="${2:?missing value for --build}"
      shift 2
      ;;
    -t|--toolchain)
      TOOLCHAIN="${2:?missing value for --toolchain}"
      shift 2
      ;;
    -o|--outdir)
      OUTDIR="${2:?missing value for --outdir}"
      shift 2
      ;;
    -L|--arm-ubl-license-id|--arm-ubl-license-identifier)
      export ARM_UBL_LICENSE_IDENTIFIER="${2:?missing value for --arm-ubl-license-id}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

[[ -n "${ARCH}" ]] || { echo "Missing --arch" >&2; exit 2; }
[[ -n "${BUILD}" ]] || { echo "Missing --build" >&2; exit 2; }
[[ -n "${TOOLCHAIN}" ]] || { echo "Missing --toolchain" >&2; exit 2; }
[[ -n "${OUTDIR}" ]] || { echo "Missing --outdir" >&2; exit 2; }

"${BUILD_AMBIQ}" \
  --arch "${ARCH}" \
  --build "${BUILD}" \
  --toolchain "${TOOLCHAIN}" \
  --optimized-kernel-dir cmsis_nn \
  --outdir "${OUTDIR}"

case "${ARCH}" in
  cortex-m4+fp)
    ARCH_TOKEN="cm4"
    ;;
  cortex-m55)
    ARCH_TOKEN="cm55"
    ;;
  *)
    echo "Unsupported arch for Zephyr prebuilt bundle: ${ARCH}" >&2
    exit 3
    ;;
esac

BUILD_TOKEN="$(echo "${BUILD}" | tr '_' '-')"
SRC_LIB="${OUTDIR}/lib/libtensorflow-microlite-${ARCH_TOKEN}-${TOOLCHAIN}-${BUILD_TOKEN}.a"
DST_LIB="${OUTDIR}/lib/libhelia-rt-zephyr-${ARCH_TOKEN}-${TOOLCHAIN}-${BUILD_TOKEN}.a"

[[ -f "${SRC_LIB}" ]] || { echo "Missing source library: ${SRC_LIB}" >&2; exit 4; }
mv "${SRC_LIB}" "${DST_LIB}"
echo "Renamed ${SRC_LIB} -> ${DST_LIB}"
