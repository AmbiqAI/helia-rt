#!/usr/bin/env bash
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Checks that every workflow pinning the CI image pins the same digest.
#
# The test matrix and the release build pull the image from separate literals
# in separate workflows. If a bump updates some and not others, release
# artifacts are built by an image nothing tested, and nothing else fails.
# see AmbiqAI/helia-rt#219 and .github/workflows/README.md
#
# Rules:
#   * each pin point below carries exactly one
#     ghcr.io/ambiqai/helia-rt-ci@sha256:<64 hex digits>;
#   * all pin points carry the same digest;
#   * no other workflow pins the image, so a new pin point is added to
#     PIN_POINTS and to the README table on purpose.
#
# Usage: check_ci_image_pins.sh [root]
#   root defaults to the repository root inferred from this script's location.
#
# Exit codes: 0 consistent, 1 inconsistent pins, 2 usage error.

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
WORKFLOWS="${ROOT}/.github/workflows"

if [[ ! -d "${WORKFLOWS}" ]]; then
  echo "usage: $(basename "$0") [repo-root]" >&2
  echo "error: '${ROOT}' has no .github/workflows directory" >&2
  exit 2
fi

PIN_POINTS=(
  ci.yml
  helia_build.yml
  helia_release.yml
  helia_test.yml
  check_tflite_files.yml
)

IMAGE='ghcr.io/ambiqai/helia-rt-ci@sha256:'
PIN_RE='ghcr\.io/ambiqai/helia-rt-ci@sha256:[0-9a-f]*'

status=0
digests=()

for file in "${PIN_POINTS[@]}"; do
  path="${WORKFLOWS}/${file}"
  if [[ ! -f "${path}" ]]; then
    echo "error: pin point ${file} is missing" >&2
    status=1
    continue
  fi
  mapfile -t found < <(grep -o "${PIN_RE}" "${path}" || true)
  if [[ "${#found[@]}" -ne 1 ]]; then
    echo "error: ${file} pins the CI image ${#found[@]} times; expected 1" >&2
    status=1
    continue
  fi
  digest="${found[0]#"${IMAGE}"}"
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "error: ${file} pins a malformed digest '${digest}'" >&2
    status=1
    continue
  fi
  digests+=("${digest}")
done

while IFS= read -r path; do
  file="$(basename "${path}")"
  listed=0
  for pin in "${PIN_POINTS[@]}"; do
    [[ "${file}" == "${pin}" ]] && listed=1
  done
  if [[ "${listed}" -eq 0 ]]; then
    echo "error: ${file} pins the CI image but is not a listed pin point" >&2
    status=1
  fi
done < <(grep -l "${PIN_RE}" "${WORKFLOWS}"/*.yml "${WORKFLOWS}"/*.yaml \
           2>/dev/null || true)

if [[ "${#digests[@]}" -gt 0 ]]; then
  distinct="$(printf '%s\n' "${digests[@]}" | sort -u | grep -c .)"
  if [[ "${distinct}" -ne 1 ]]; then
    echo "error: pin points carry ${distinct} different digests:" >&2
    for i in "${!PIN_POINTS[@]}"; do
      grep -o "${PIN_RE}" "${WORKFLOWS}/${PIN_POINTS[$i]}" 2>/dev/null \
        | sed "s|^|  ${PIN_POINTS[$i]}: |" >&2 || true
    done
    status=1
  fi
fi

if [[ "${status}" -eq 0 ]]; then
  echo "check_ci_image_pins: ${#PIN_POINTS[@]} pin points share digest" \
       "${digests[0]}"
fi
exit "${status}"
