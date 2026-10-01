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
# Rules, applied to every .yml/.yaml workflow line by line outside comments
# (a line starting with #, or a # after whitespace, which matches YAML except
# inside block scalars and quoted strings). A line refers to the image when it
# contains helia-rt-ci in any case, so a name built from an expression or
# spelled with another host, owner, case or tag still counts. Rules:
#   * each pin point below has exactly one such line, and it is, after
#     trimming, exactly
#       run: echo "image=ghcr.io/ambiqai/helia-rt-ci@sha256:<64 lowercase hex>" >> "$GITHUB_OUTPUT"
#     so nothing can precede the host or follow the digest inside the word;
#   * all pin points carry the same digest;
#   * the publisher's one such line is exactly
#       IMAGE_NAME: ghcr.io/ambiqai/helia-rt-ci
#   * any other line that refers to the image fails, in any workflow. A new
#     pin point is added to PIN_POINTS and to the README table on purpose.
# A name assembled without the literal helia-rt-ci is not seen.
#
# Usage: check_helia_ci_image_pins.sh [root]
#   root defaults to the repository root inferred from this script's location.
#
# Exit codes: 0 consistent, 1 inconsistent references, 2 usage error or an
# unreadable workflow.

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
)
PUBLISHER=helia_build_docker_image.yml

# shellcheck disable=SC2016  # $GITHUB_OUTPUT is matched literally.
PIN_LINE_RE='^run: echo "image=ghcr\.io/ambiqai/helia-rt-ci@sha256:([0-9a-f]{64})" >> "\$GITHUB_OUTPUT"$'
PUBLISHER_LINE='IMAGE_NAME: ghcr.io/ambiqai/helia-rt-ci'

is_pin_point() {
  local pin
  for pin in "${PIN_POINTS[@]}"; do
    [[ "$1" == "${pin}" ]] && return 0
  done
  return 1
}

# One "file<US>line-number<US>text" line (US = \037) per workflow line that
# contains helia-rt-ci in any case, outside comments, with a trailing comment
# and surrounding whitespace removed.
references() {
  local path
  for path in "${WORKFLOWS}"/*.yml "${WORKFLOWS}"/*.yaml; do
    [[ -f "${path}" ]] || continue
    awk -v file="$(basename "${path}")" '
      {
        line = $0
        sub(/\r$/, "", line)
        if (line ~ /^[[:space:]]*#/) next
        sub(/[[:space:]]#.*$/, "", line)
        if (index(tolower(line), "helia-rt-ci") == 0) next
        sub(/^[[:space:]]+/, "", line)
        sub(/[[:space:]]+$/, "", line)
        print file "\037" FNR "\037" line
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

while IFS=$'\037' read -r file lineno text; do
  [[ -n "${file}" ]] || continue
  if is_pin_point "${file}" && [[ "${text}" =~ ${PIN_LINE_RE} ]]; then
    pin_count["${file}"]=$(( ${pin_count["${file}"]:-0} + 1 ))
    digests+=("${file}: ${BASH_REMATCH[1]}")
  elif [[ "${file}" == "${PUBLISHER}" && "${text}" == "${PUBLISHER_LINE}" ]]; then
    continue
  elif is_pin_point "${file}"; then
    echo "error: ${file}:${lineno} refers to the CI image as '${text}';" \
         "a pin point has only the line" \
         "run: echo \"image=ghcr.io/ambiqai/helia-rt-ci@sha256:<64 lowercase hex>\" >> \"\$GITHUB_OUTPUT\"" >&2
    status=1
  else
    echo "error: ${file}:${lineno} refers to the CI image as '${text}'" \
         "but is not a listed pin point" >&2
    status=1
  fi
done <<< "${refs}"

for file in "${PIN_POINTS[@]}"; do
  if [[ ! -f "${WORKFLOWS}/${file}" ]]; then
    echo "error: pin point ${file} is missing" >&2
    status=1
  elif [[ "${pin_count["${file}"]:-0}" -ne 1 ]]; then
    echo "error: ${file} has ${pin_count["${file}"]:-0} pin lines for the CI" \
         "image; expected 1" >&2
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
