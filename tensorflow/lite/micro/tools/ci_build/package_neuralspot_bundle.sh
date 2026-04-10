#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tensorflow/lite/micro/tools/ci_build/release_asset_helpers.sh
source "${SCRIPT_DIR}/release_asset_helpers.sh"

ARTIFACTS_DIR=""
CHECKOUT_DIR=""
TAG=""
SHA=""
BUNDLE_PREFIX=""
UPLOAD_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifacts-dir)
      ARTIFACTS_DIR="${2:?missing value for --artifacts-dir}"
      shift 2
      ;;
    --checkout-dir)
      CHECKOUT_DIR="${2:?missing value for --checkout-dir}"
      shift 2
      ;;
    --tag)
      TAG="${2:?missing value for --tag}"
      shift 2
      ;;
    --sha)
      SHA="${2:?missing value for --sha}"
      shift 2
      ;;
    --bundle-prefix)
      BUNDLE_PREFIX="${2:?missing value for --bundle-prefix}"
      shift 2
      ;;
    --upload-dir)
      UPLOAD_DIR="${2:?missing value for --upload-dir}"
      shift 2
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[[ -d "${ARTIFACTS_DIR}" ]] || die "Artifacts dir not found: ${ARTIFACTS_DIR}"
[[ -d "${CHECKOUT_DIR}" ]] || die "Checkout dir not found: ${CHECKOUT_DIR}"
[[ -n "${TAG}" ]] || die "Tag is required."
[[ -n "${SHA}" ]] || die "SHA is required."
[[ -n "${BUNDLE_PREFIX}" ]] || die "Bundle prefix is required."
[[ -n "${UPLOAD_DIR}" ]] || die "Upload dir is required."

BUNDLE_DIR="${BUNDLE_PREFIX}-${TAG}"

rm -rf "${BUNDLE_DIR}"
mkdir -p "${BUNDLE_DIR}/lib"

echo "== Collecting libraries into ${BUNDLE_DIR}/lib =="
copy_archives_from_artifacts "${ARTIFACTS_DIR}" "${BUNDLE_DIR}/lib"

echo "== Selecting one tflm tree into ${BUNDLE_DIR}/ =="
copy_first_tflm_tree "${ARTIFACTS_DIR}" "${BUNDLE_DIR}"

echo "== Pruning cmsis_nn to headers only =="
prune_cmsis_nn_to_headers "${BUNDLE_DIR}"

echo "== Copying neuralSPOT module.mk =="
if [[ -f "${CHECKOUT_DIR}/neuralspot/module.mk" ]]; then
  cp "${CHECKOUT_DIR}/neuralspot/module.mk" "${BUNDLE_DIR}/module.mk"
else
  die "neuralspot/module.mk not found"
fi

if [[ -f "${CHECKOUT_DIR}/LICENSE" ]]; then
  cp "${CHECKOUT_DIR}/LICENSE" "${BUNDLE_DIR}/LICENSE"
fi

echo "== Writing MANIFEST =="
write_manifest "${BUNDLE_PREFIX}" "${TAG}" "${SHA}" "${BUNDLE_DIR}/lib" "${BUNDLE_DIR}/MANIFEST.txt"

echo "== Final bundle structure =="
find "${BUNDLE_DIR}" -maxdepth 2 -type f -print

ZIP_NAME="${BUNDLE_PREFIX}-${TAG}.zip"
zip_bundle_into_upload_dir "${BUNDLE_DIR}" "${ZIP_NAME}" "${UPLOAD_DIR}"
