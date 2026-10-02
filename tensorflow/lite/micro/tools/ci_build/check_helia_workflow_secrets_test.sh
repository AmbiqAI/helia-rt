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
# Self-test for check_helia_workflow_secrets.sh. Each case writes one workflow
# into a scratch tree and expects the check's exit status and, when it fails,
# the line it reports.
# see AmbiqAI/helia-rt#366
#
# Usage: check_helia_workflow_secrets_test.sh [check-script]
#   check-script defaults to check_helia_workflow_secrets.sh next to this file.
#
# Exit codes: 0 every case as expected, 1 otherwise.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK="${1:-${SCRIPT_DIR}/check_helia_workflow_secrets.sh}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
cases=0

# run_case <name> <expected-exit> <reported-line-or-empty> <workflow text>
run_case() {
  local name="$1" expected="$2" line="$3" text="$4"
  local tree="${WORK}/${cases}"
  cases=$((cases + 1))
  mkdir -p "${tree}/.github/workflows"
  printf '%s\n' "${text}" > "${tree}/.github/workflows/case.yml"
  local errors actual
  errors="$("${CHECK}" "${tree}" 2>&1 >/dev/null)"
  actual=$?
  if [[ ${actual} -ne ${expected} ]]; then
    echo "FAIL ${name}: exit ${actual}, expected ${expected}"
    failures=$((failures + 1))
  elif [[ -n "${line}" && "${errors}" != *"case.yml:${line}: secrets: inherit"* ]]; then
    echo "FAIL ${name}: no report for line ${line} in: ${errors}"
    failures=$((failures + 1))
  fi
  rm -rf "${tree}"
}

HEAD=$'jobs:\n  call:\n    uses: ./.github/workflows/called.yml'

run_case "secret passed by name" 0 '' \
  "${HEAD}"$'\n    secrets:\n      TOKEN: ${{ secrets.TOKEN }}'
run_case "bare inherit" 1 4 "${HEAD}"$'\n    secrets: inherit'
run_case "quoted inherit" 1 4 "${HEAD}"$'\n    secrets: "inherit"'
run_case "single-quoted inherit" 1 4 "${HEAD}"$'\n    secrets: \'inherit\''
run_case "trailing comment" 1 4 "${HEAD}"$'\n    secrets: inherit  # all of them'
run_case "extra spaces and tab" 1 4 "${HEAD}"$'\n    secrets:  \tinherit'
run_case "flow mapping" 1 2 $'jobs:\n  call: { uses: ./x.yml, secrets: inherit }'
run_case "list item" 1 3 $'jobs:\n  call:\n    - secrets: inherit'
run_case "value on the next line" 1 4 "${HEAD}"$'\n    secrets:\n      inherit'
run_case "quoted value after a comment and a blank line" 1 4 \
  "${HEAD}"$'\n    secrets:\n      # every secret\n\n      "inherit"'
run_case "CRLF line ends" 1 4 "${HEAD}"$'\r\n    secrets: inherit\r'
run_case "double-quoted key" 1 4 "${HEAD}"$'\n    "secrets": inherit'
run_case "single-quoted key" 1 4 "${HEAD}"$'\n    \'secrets\': inherit'
run_case "anchored inherit" 1 4 "${HEAD}"$'\n    secrets: &all inherit'
run_case "alias" 1 4 "${HEAD}"$'\n    secrets: *all'
run_case "alias on the next line" 1 4 "${HEAD}"$'\n    secrets:\n      *all'
run_case "quoted key in a flow mapping" 1 2 $'jobs:\n  call: { uses: ./x.yml, "secrets": inherit }'
run_case "anchored mapping of named secrets" 0 '' \
  "${HEAD}"$'\n    secrets: &named\n      TOKEN: ${{ secrets.TOKEN }}'
run_case "commented out" 0 '' "${HEAD}"$'\n    # secrets: inherit'
run_case "another key named like it" 0 '' "${HEAD}"$'\n    mysecrets: inherit'
run_case "a value that only starts with inherit" 0 '' "${HEAD}"$'\n    secrets: inheritance'
run_case "a secret named inherit-like" 0 '' \
  "${HEAD}"$'\n    secrets:\n      INHERIT_TOKEN: x'
run_case "text that mentions it" 0 '' \
  "${HEAD}"$'\n    env:\n      NOTE: "secrets: inherit is not allowed"'
run_case "commented out at column 0" 0 '' "${HEAD}"$'\n# secrets: inherit'
run_case "another key, value on the next line" 0 '' "${HEAD}"$'\n    mysecrets:\n      inherit'
run_case "next-line value that only starts with inherit" 0 '' \
  "${HEAD}"$'\n    secrets:\n      inheritance'
run_case "a lone inherit after a mapping entry" 0 '' \
  "${HEAD}"$'\n    secrets:\n      TOKEN: x\n  other:\n    run: |\n      inherit'
run_case "two calls, both reported" 1 7 \
  "${HEAD}"$'\n    secrets: inherit\n  again:\n    uses: ./x.yml\n    secrets: inherit'

# An unreadable workflow is a usage error, not a pass.
tree="${WORK}/unreadable"
mkdir -p "${tree}/.github/workflows"
ln -s missing.yml "${tree}/.github/workflows/case.yml"
cases=$((cases + 1))
"${CHECK}" "${tree}" >/dev/null 2>&1
actual=$?
if [[ ${actual} -ne 2 ]]; then
  echo "FAIL unreadable workflow: exit ${actual}, expected 2"
  failures=$((failures + 1))
fi

if [[ ${failures} -ne 0 ]]; then
  echo "check_helia_workflow_secrets_test: ${failures} of ${cases} cases failed"
  exit 1
fi
echo "check_helia_workflow_secrets_test: ${cases} cases passed"
