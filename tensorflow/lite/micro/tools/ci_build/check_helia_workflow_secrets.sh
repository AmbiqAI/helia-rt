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
# a pull request's code. see AmbiqAI/helia-rt#366
#
# The rule applies to every .yml/.yaml workflow outside comments: a
# "secrets:" key whose value is inherit, bare or quoted, fails.
#
# Usage: check_helia_workflow_secrets.sh [root]
#   root defaults to the repository root inferred from this script's location.
#
# Exit codes: 0 none found, 1 a workflow passes secrets: inherit, 2 usage error.

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

status=0
for file in "${files[@]}"; do
  # A flow mapping ({ ..., secrets: inherit }) is matched as well as a block key.
  while IFS= read -r hit; do
    echo "${file#"${ROOT}/"}:${hit}: secrets: inherit; pass each secret the called workflow declares by name" >&2
    status=1
  done < <(grep -nE '(^|[{,[:space:]])secrets[[:space:]]*:[[:space:]]*["'"'"']?inherit["'"'"']?[[:space:]]*([,}#]|$)' "${file}" \
             | grep -vE '^[0-9]+:[[:space:]]*#' \
             | cut -d: -f1)
done

if [[ ${status} -eq 0 ]]; then
  echo "check_helia_workflow_secrets: no workflow passes secrets: inherit (${#files[@]} files)"
fi
exit "${status}"
