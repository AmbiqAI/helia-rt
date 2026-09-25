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
# Checks that the workflows use the CI image only through one shared digest.
#
# The test matrix and the release build pull the image from separate literals
# in separate workflows. If a bump updates some and not others, or a workflow
# uses a tag, release artifacts can be built by an image nothing tested.
# see AmbiqAI/helia-rt#219 and .github/workflows/README.md
#
# Rules, applied to every .yml/.yaml workflow outside comments. Comments are
# found line by line (a line starting with #, or a # after whitespace), which
# matches YAML except inside block scalars and quoted strings: there a " #"
# before the image on the same line hides the reference. Rules:
#   * each pin point below references the image exactly once, as
#     ghcr.io/ambiqai/helia-rt-ci@sha256:<64 lowercase hex digits>;
#   * all pin points carry the same digest;
#   * no other reference to the image exists, by digest, tag or bare name,
#     except the bare repository name in the image publisher. A new pin point
#     is added to PIN_POINTS and to the README table on purpose.
#
# Usage: check_helia_ci_image_pins.sh [root]
#   root defaults to the repository root inferred from this script's location.
#
# Exit codes: 0 consistent, 1 inconsistent references, 2 usage error.

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
PUBLISHER=helia_build_docker_image.yml

IMAGE='ghcr.io/ambiqai/helia-rt-ci'
DIGEST_RE='^@sha256:[0-9a-f]{64}$'

is_pin_point() {
  local pin
  for pin in "${PIN_POINTS[@]}"; do
    [[ "$1" == "${pin}" ]] && return 0
  done
  return 1
}

# One "file<TAB>suffix" line per image reference outside a YAML comment, where
# suffix is whatever follows the repository name (empty for the bare name).
# The image name is matched case-insensitively; the digest check is exact.
references() {
  local path
  for path in "${WORKFLOWS}"/*.yml "${WORKFLOWS}"/*.yaml; do
    [[ -f "${path}" ]] || continue
    awk -v file="$(basename "${path}")" -v image="${IMAGE}" '
      {
        line = $0
        sub(/\r$/, "", line)
        if (line ~ /^[[:space:]]*#/) next
        sub(/[[:space:]]#.*$/, "", line)
        lower = tolower(line)
        while ((i = index(lower, image)) > 0) {
          rest = substr(line, i + length(image))
          match(rest, /^[@:][A-Za-z0-9._:-]*/)
          suffix = RSTART == 1 ? substr(rest, 1, RLENGTH) : ""
          print file "\t" suffix
          line = substr(line, i + length(image))
          lower = substr(lower, i + length(image))
        }
      }' "${path}" || return 1
  done
}

if ! refs="$(references)"; then
  echo "error: could not read the workflows under ${WORKFLOWS}" >&2
  exit 2
fi

status=0
declare -A pin_count=()
digests=()

while IFS=$'\t' read -r file suffix; do
  [[ -n "${file}" ]] || continue
  if is_pin_point "${file}"; then
    pin_count["${file}"]=$(( ${pin_count["${file}"]:-0} + 1 ))
    if [[ "${suffix}" =~ ${DIGEST_RE} ]]; then
      digests+=("${file}: ${suffix#@sha256:}")
    else
      echo "error: ${file} references the CI image as '${IMAGE}${suffix}';" \
           "expected @sha256:<64 lowercase hex>" >&2
      status=1
    fi
  elif [[ "${file}" == "${PUBLISHER}" && -z "${suffix}" ]]; then
    continue
  else
    echo "error: ${file} references the CI image as '${IMAGE}${suffix}'" \
         "but is not a listed pin point" >&2
    status=1
  fi
done <<< "${refs}"

for file in "${PIN_POINTS[@]}"; do
  if [[ ! -f "${WORKFLOWS}/${file}" ]]; then
    echo "error: pin point ${file} is missing" >&2
    status=1
  elif [[ "${pin_count["${file}"]:-0}" -ne 1 ]]; then
    echo "error: ${file} references the CI image" \
         "${pin_count["${file}"]:-0} times; expected 1" >&2
    status=1
  fi
done

if [[ "${#digests[@]}" -gt 0 ]]; then
  distinct="$(printf '%s\n' "${digests[@]}" | sed 's/^[^ ]* //' \
              | sort -u | grep -c .)"
  if [[ "${distinct}" -ne 1 ]]; then
    echo "error: pin points carry ${distinct} different digests:" >&2
    printf '  %s\n' "${digests[@]}" >&2
    status=1
  fi
fi

if [[ "${status}" -eq 0 ]]; then
  echo "check_helia_ci_image_pins: ${#PIN_POINTS[@]} pin points share digest" \
       "${digests[0]#* }"
fi
exit "${status}"
