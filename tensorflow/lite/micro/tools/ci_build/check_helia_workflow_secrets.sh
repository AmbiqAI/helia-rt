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
# Checks that no workflow passes every repository secret to a called workflow.
#
# A reusable workflow declares the secrets it reads under
# on.workflow_call.secrets, and each caller passes those by name.
# "secrets: inherit" would hand every repository secret to jobs that may build
# a pull request's code.
# see AmbiqAI/helia-rt#366
#
# The rule applies to every top-level .yml/.yaml workflow, line by line: a
# "secrets:" key whose value is inherit, bare or quoted, in a block or flow
# mapping, fails unless the line is a comment. Line matching also flags the
# text inside a block scalar (run: |) and after a trailing #, and does not
# see a quoted key, a value on the next line, or a YAML tag or alias.
#
# Usage: check_helia_workflow_secrets.sh [root]
#   root defaults to the repository root inferred from this script's location.
#
# Exit codes: 0 none found, 1 a workflow passes secrets: inherit, 2 usage error
# or an unreadable workflow.

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

shopt -s nullglob
files=("${WORKFLOWS}"/*.yml "${WORKFLOWS}"/*.yaml)

PATTERN='(^|[{,[:space:]])secrets[[:space:]]*:[[:space:]]*["'"'"']?inherit["'"'"']?[[:space:]]*([,}#]|$)'

status=0
for file in "${files[@]}"; do
  rc=0
  matches="$(grep -nE "${PATTERN}" "${file}")" || rc=$?
  if [[ ${rc} -gt 1 ]]; then
    echo "error: could not read ${file#"${ROOT}/"}" >&2
    exit 2
  fi
  while IFS= read -r hit; do
    [[ -n "${hit}" ]] || continue
    echo "${file#"${ROOT}/"}:${hit}: secrets: inherit; pass each secret the called workflow declares by name" >&2
    status=1
  done < <(printf '%s\n' "${matches}" | grep -vE '^[0-9]+:[[:space:]]*#' | cut -d: -f1)
done

if [[ ${status} -eq 0 ]]; then
  echo "check_helia_workflow_secrets: no workflow passes secrets: inherit (${#files[@]} files)"
fi
exit "${status}"
