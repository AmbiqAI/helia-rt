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
# Self-test for check_helia_ci_image_pins.sh. Each case copies this
# repository's workflows into a scratch tree, edits one line, and expects the
# check's exit status: 0 for a valid tree, 1 for an invalid one.
# see AmbiqAI/helia-rt#369
#
# Usage: check_helia_ci_image_pins_test.sh [check-script]
#   check-script defaults to check_helia_ci_image_pins.sh next to this file.
#
# Exit codes: 0 every case as expected, 1 otherwise.

# shellcheck disable=SC2016  # $ in perl programs and workflow text is literal.
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
CHECK="${1:-${SCRIPT_DIR}/check_helia_ci_image_pins.sh}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

PIN='ghcr.io/ambiqai/helia-rt-ci@sha256:'
DIGEST="$(grep -ohE "${PIN}[0-9a-f]{64}" "${ROOT}/.github/workflows/helia_test.yml" \
          | head -n 1 | sed "s|${PIN}||")"
if [[ ${#DIGEST} -ne 64 ]]; then
  echo "error: no pin found in .github/workflows/helia_test.yml" >&2
  exit 1
fi
OTHER="$(printf '%064d' 0)"

failures=0
cases=0

# run_case <name> <expected-exit> <perl program> [files]
# The perl program edits each of the space-separated workflow files (default
# helia_test.yml); an empty program leaves the tree unchanged.
run_case() {
  local name="$1" expected="$2" edit="$3" files="${4:-helia_test.yml}" file
  local tree="${WORK}/${cases}"
  cases=$((cases + 1))
  mkdir -p "${tree}/.github"
  cp -R "${ROOT}/.github/workflows" "${tree}/.github/"
  if [[ -n "${edit}" ]]; then
    for file in ${files}; do
      PIN="${PIN}" DIGEST="${DIGEST}" OTHER="${OTHER}" \
        perl -0pi -e "${edit}" "${tree}/.github/workflows/${file}"
      # An edit that matched nothing would make the case pass vacuously.
      if cmp -s "${ROOT}/.github/workflows/${file}" \
                "${tree}/.github/workflows/${file}"; then
        echo "FAIL ${name}: the edit did not change ${file}"
        failures=$((failures + 1))
        rm -rf "${tree}"
        return
      fi
    done
  fi
  "${CHECK}" "${tree}" >/dev/null 2>&1
  local actual=$?
  if [[ ${actual} -ne ${expected} ]]; then
    echo "FAIL ${name}: exit ${actual}, expected ${expected}"
    failures=$((failures + 1))
  fi
  rm -rf "${tree}"
}

# Appends a step line to a workflow.
add_line() {
  printf '%s' "\$_ .= \"\\n      - run: echo $1\\n\";"
}

P='\Q$ENV{PIN}$ENV{DIGEST}\E'

run_case "unchanged tree" 0 ''
run_case "digest bumped at every pin point" 0 \
  "s/$P/\$ENV{PIN}\$ENV{OTHER}/g" "ci.yml helia_build.yml helia_release.yml helia_test.yml"

run_case "uneven digest" 1 "s/$P/\$ENV{PIN}\$ENV{OTHER}/g"
run_case "percent after digest" 1 "s/($P)/\$1%/"
run_case "comma after digest" 1 "s/($P)/\$1,/"
run_case "brace after digest" 1 "s/($P)/\$1}/"
run_case "quote and text after digest" 1 "s/($P)\"/\$1\"x\"/"
run_case "dollar before host" 1 "s/image=($P)/image=\\\$\$1/"
run_case "hash before host" 1 "s/image=($P)/image=#\$1/"
run_case "brace before host" 1 "s/image=($P)/image=}\$1/"
run_case "quote before host" 1 "s/\"image=($P)/x\"image=\$1/"
run_case "tag instead of digest" 1 "s/($P)/ghcr.io\\/ambiqai\\/helia-rt-ci:latest/"
run_case "uppercase name" 1 "s/ambiqai\\/helia-rt-ci\@/AmbiqAI\\/helia-rt-ci\@/"
run_case "other host" 1 "s/ghcr\\.io\\/ambiqai\\/helia-rt-ci\@/docker.io\\/ambiqai\\/helia-rt-ci\@/"
run_case "uppercase hex" 1 "s/$P/\$ENV{PIN}\\U\$ENV{DIGEST}/"
run_case "short digest" 1 "s/($P)/substr(\$1,0,-1)/e"
run_case "pin line removed" 1 "s/^.*\\Q\$ENV{PIN}\\E.*\\n//m"
run_case "second reference in a pin point" 1 "$(add_line 'ghcr.io/ambiqai/helia-rt-ci:latest')"
run_case "digest reference in another workflow" 1 \
  "$(add_line "\$ENV{PIN}\$ENV{DIGEST}")" ns_cmsis_nn_canary.yml
run_case "expression-built owner" 1 \
  "$(add_line 'ghcr.io/\${{ github.repository_owner }}/helia-rt-ci:latest')" ns_cmsis_nn_canary.yml
run_case "expression-built tag" 1 \
  "$(add_line 'ghcr.io/ambiqai/helia-rt-ci:\${{ env.TAG }}')" ns_cmsis_nn_canary.yml
run_case "publisher tag" 1 \
  "s/(IMAGE_NAME: ghcr\\.io\\/ambiqai\\/helia-rt-ci)/\$1:latest/" helia_build_docker_image.yml
run_case "trailing comment kept out" 0 \
  "s/(\\\$GITHUB_OUTPUT\"\\s*)#[^\\n]*/\$1# a note/"

if [[ ${failures} -ne 0 ]]; then
  echo "check_helia_ci_image_pins_test: ${failures} of ${cases} cases failed"
  exit 1
fi
echo "check_helia_ci_image_pins_test: ${cases} cases passed"
