#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tensorflow/lite/micro/tools/ci_build/release_asset_helpers.sh
source "${SCRIPT_DIR}/release_asset_helpers.sh"

REF=""
TAG=""
UPLOAD=""
SANITIZE_TAG="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)
      REF="${2:?missing value for --ref}"
      shift 2
      ;;
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --upload)
      UPLOAD="${2:-}"
      shift 2
      ;;
    --sanitize-tag)
      SANITIZE_TAG="true"
      shift
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

emit_release_meta "${REF}" "${TAG}" "${UPLOAD}" "${SANITIZE_TAG}"
