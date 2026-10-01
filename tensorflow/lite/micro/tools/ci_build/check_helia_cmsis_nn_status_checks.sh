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
# The status-returning names come from the top-level headers under the given
# Include directory (not Include/Internal), with the return type on the same
# line as the name or alone on the line before it; names built by macros are
# not seen. In each kernel source directly under the source directory (not
# tests/), outside comments and string literals, a call to one of them fails
# when its status is discarded:
#   * the call starts a statement: it is the first thing on its line,
#     optionally behind (void), and the previous code line ends with ; { or }
#     (not a brace initializer), ends a label (case X: or default:), is an
#     unbraced if, for or while header, or is else or do. Preprocessor lines do
#     not count as the previous line;
#   * the call is an argument of a TFLITE_DCHECK* macro, which is compiled out
#     of a release build.
# A call nested in an expression, an assignment or another macro is not a
# discard. The check is line based: a call after another statement on the
# same line, a control header spread over several lines, and spellings such as
# static_cast<void>(...) or ::arm_x are not seen.
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
if [[ ${#headers[@]} -eq 0 || ${#sources[@]} -eq 0 ]]; then
  echo "error: no headers under '${INCLUDE_DIR}' or no sources under '${SOURCE_DIR}'" >&2
  exit 2
fi

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
      if (!match(line, /\/\*|\/\/|"|\047/)) { out = out line; break }
      tok = substr(line, RSTART, RLENGTH)
      out = out substr(line, 1, RSTART - 1)
      line = substr(line, RSTART + RLENGTH)
      if (tok == "//") break
      if (tok == "/*") { in_block = 1; continue }
      # A string or character literal: keep a placeholder, skip its contents.
      out = out tok tok
      while (line != "") {
        c = substr(line, 1, 1)
        line = substr(line, 2)
        if (c == "\\") { line = substr(line, 2); continue }
        if (c == tok) break
      }
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
    function first_call(text,    rest, name) {
      rest = text
      while (match(rest, /arm_[A-Za-z0-9_]+[[:space:]]*\(/)) {
        name = substr(rest, RSTART, RLENGTH)
        sub(/[[:space:]]*\($/, "", name)
        if ((name in known) && (RSTART == 1 || substr(rest, RSTART - 1, 1) !~ /[A-Za-z0-9_]/)) return name
        rest = substr(rest, RSTART + RLENGTH)
      }
      return ""
    }
    {
      text = code
      sub(/^[[:space:]]+/, "", text)
      sub(/[[:space:]]+$/, "", text)
      if (text == "" || text ~ /^#/) next
      if (text ~ /TFLITE_DCHECK[A-Z_]*[[:space:]]*\(/) { in_dcheck = 1; dline = FNR }
      if (in_dcheck) {
        name = first_call(text)
        if (name != "") print dline ": " name " in TFLITE_DCHECK"
        if (text ~ /;$/) in_dcheck = 0
      } else if (match(text, /^(\(void\)[[:space:]]*)?arm_[A-Za-z0-9_]+[[:space:]]*\(/)) {
        name = substr(text, RSTART, RLENGTH)
        sub(/^\(void\)[[:space:]]*/, "", name); sub(/[[:space:]]*\($/, "", name)
        starts = (prev == "" || prev ~ /[;}]$/ || (prev ~ /\{$/ && prev !~ /=[[:space:]]*\{$/) ||
                  prev ~ /(^|[^A-Za-z0-9_])(else|do)$/ || prev ~ /[^:]:$/ ||
                  prev ~ /^(\}[[:space:]]*)?(else[[:space:]]+)?(if|for|while)[[:space:]]*\(.*\)$/)
        if ((name in known) && starts) print FNR ": " name
      }
      prev = text
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
