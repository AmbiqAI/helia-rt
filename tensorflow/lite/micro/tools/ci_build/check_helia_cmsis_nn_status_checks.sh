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
# Checks that the helia kernels never discard the status a heliaCORE entry
# point returns.
#
# Every ns-cmsis-nn function declared to return arm_cmsis_nn_status must have
# its status captured and mapped to kTfLiteError; a discarded one reports a
# failed or skipped kernel as success.
# see AmbiqAI/helia-rt#377 and AmbiqAI/helia-rt#238
#
# The status-returning names come from the headers under the given Include
# directory, with the return type on the same line as the name or alone on the
# line before it. In each kernel source, outside comments, a call to one of
# them fails when it starts a statement: it is the first thing on its line,
# optionally behind (void), and the previous code line ends with ; { or } or
# is else or do. A call nested in an expression, an assignment or a macro
# argument is not a discard. The check is line based: a call that follows
# another statement or an unbraced if on the same line is not seen.
#
# Usage: check_helia_cmsis_nn_status_checks.sh [include-dir [source-dir]]
#   include-dir defaults to ${NS_CMSIS_NN_PATH}/Include, or the downloaded
#   ns-cmsis-nn under tensorflow/lite/micro/tools/make/downloads.
#   source-dir defaults to tensorflow/lite/micro/kernels/helia.
#
# Exit codes: 0 no discarded status, 1 one or more found, 2 usage error,
# unreadable input, or no status-returning declaration found.

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
DEFAULT_CORE="${NS_CMSIS_NN_PATH:-${ROOT}/tensorflow/lite/micro/tools/make/downloads/ns_cmsis_nn}"
INCLUDE_DIR="${1:-${DEFAULT_CORE}/Include}"
SOURCE_DIR="${2:-${ROOT}/tensorflow/lite/micro/kernels/helia}"

if [[ ! -d "${INCLUDE_DIR}" || ! -d "${SOURCE_DIR}" ]]; then
  echo "usage: $(basename "$0") [include-dir [source-dir]]" >&2
  echo "error: '${INCLUDE_DIR}' or '${SOURCE_DIR}' is not a directory" >&2
  exit 2
fi

shopt -s nullglob
headers=("${INCLUDE_DIR}"/*.h)
sources=("${SOURCE_DIR}"/*.cc "${SOURCE_DIR}"/*.h)

# Removes /* */ and // comments; one code line out per input line.
# shellcheck disable=SC2016  # awk program text: $ is awk's, not the shell's.
STRIP='
  {
    line = $0
    sub(/\r$/, "", line)
    out = ""
    while (line != "") {
      if (in_block) {
        end = index(line, "*/")
        if (end == 0) { line = ""; break }
        line = substr(line, end + 2)
        in_block = 0
        continue
      }
      b = index(line, "/*")
      s = index(line, "//")
      if (s > 0 && (b == 0 || s < b)) { out = out substr(line, 1, s - 1); line = ""; break }
      if (b == 0) { out = out line; line = ""; break }
      out = out substr(line, 1, b - 1)
      line = substr(line, b + 2)
      in_block = 1
    }
    code = out
  }
'

rc=0
names="$(awk "${STRIP}"'
  {
    if (pending) {
      if (code ~ /^[[:space:]]*$/) next
      if (match(code, /^[[:space:]]*arm_[A-Za-z0-9_]+[[:space:]]*\(/)) {
        name = substr(code, RSTART, RLENGTH)
        sub(/^[[:space:]]*/, "", name); sub(/[[:space:]]*\($/, "", name)
        print name
      }
      pending = 0
    }
    if (code ~ /^[[:space:]]*arm_cmsis_nn_status[[:space:]]*$/) { pending = 1; next }
    rest = code
    while (match(rest, /arm_cmsis_nn_status[[:space:]]+arm_[A-Za-z0-9_]+[[:space:]]*\(/)) {
      name = substr(rest, RSTART, RLENGTH)
      sub(/^arm_cmsis_nn_status[[:space:]]+/, "", name); sub(/[[:space:]]*\($/, "", name)
      print name
      rest = substr(rest, RSTART + RLENGTH)
    }
  }' "${headers[@]}" | sort -u)" || rc=$?
if [[ ${rc} -ne 0 ]]; then
  echo "error: could not read the headers under ${INCLUDE_DIR}" >&2
  exit 2
fi
if [[ -z "${names}" ]]; then
  echo "error: no arm_cmsis_nn_status declaration under ${INCLUDE_DIR}" >&2
  exit 2
fi

status=0
for file in "${sources[@]}"; do
  rc=0
  hits="$(printf '%s\n' "${names}" | awk -v names_file=/dev/stdin "${STRIP}"'
    BEGIN { while ((getline n < names_file) > 0) known[n] = 1 }
    {
      text = code
      sub(/^[[:space:]]+/, "", text)
      sub(/[[:space:]]+$/, "", text)
      if (match(text, /^(\(void\)[[:space:]]*)?arm_[A-Za-z0-9_]+[[:space:]]*\(/)) {
        name = substr(text, RSTART, RLENGTH)
        sub(/^\(void\)[[:space:]]*/, "", name); sub(/[[:space:]]*\($/, "", name)
        if ((name in known) && (prev == "" || prev ~ /[;{}]$/ || prev ~ /(^|[^A-Za-z0-9_])(else|do)$/))
          print FNR ": " name
      }
      if (text != "") prev = text
    }' "${file}")" || rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "error: could not read ${file}" >&2
    exit 2
  fi
  while IFS= read -r hit; do
    [[ -n "${hit}" ]] || continue
    echo "${file#"${ROOT}/"}:${hit} returns arm_cmsis_nn_status, which is discarded;" \
         "map a failure to kTfLiteError" >&2
    status=1
  done <<< "${hits}"
done

if [[ ${status} -eq 0 ]]; then
  echo "check_helia_cmsis_nn_status_checks: $(grep -c . <<< "${names}") status-returning" \
       "entry points, none discarded in ${#sources[@]} files"
fi
exit "${status}"
