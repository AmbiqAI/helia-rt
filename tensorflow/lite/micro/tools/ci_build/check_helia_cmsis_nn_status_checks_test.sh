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
# Self-test for check_helia_cmsis_nn_status_checks.sh. Each case writes a
# header and one kernel source into a scratch tree and expects the check's exit
# status and, when it fails, the line it reports.
# see AmbiqAI/helia-rt#377
#
# Usage: check_helia_cmsis_nn_status_checks_test.sh [check-script]
#
# Exit codes: 0 every case as expected, 1 otherwise.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK="${1:-${SCRIPT_DIR}/check_helia_cmsis_nn_status_checks.sh}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

HEADER='typedef int arm_cmsis_nn_status;
/* arm_cmsis_nn_status arm_in_comment(int); */
arm_cmsis_nn_status arm_one_line(int a);
arm_cmsis_nn_status
arm_split_line(int a,
               int b);
void arm_returns_void(int a);
int32_t arm_returns_size(int a);'

failures=0
cases=0

# run_case <name> <expected-exit> <reported-line-or-empty> <source> [header]
run_case() {
  local name="$1" expected="$2" line="$3" source="$4" header="${5-${HEADER}}"
  local tree="${WORK}/${cases}"
  cases=$((cases + 1))
  mkdir -p "${tree}/include" "${tree}/src"
  printf '%s\n' "${header}" > "${tree}/include/arm_nnfunctions.h"
  printf '%s\n' "${source}" > "${tree}/src/kernel.cc"
  local errors actual
  errors="$("${CHECK}" "${tree}/include" "${tree}/src" 2>&1 >/dev/null)"
  actual=$?
  if [[ ${actual} -ne ${expected} ]]; then
    echo "FAIL ${name}: exit ${actual}, expected ${expected}: ${errors}"
    failures=$((failures + 1))
  elif [[ -n "${line}" && "${errors}" != *"kernel.cc:${line}: "* ]]; then
    echo "FAIL ${name}: no report for line ${line} in: ${errors}"
    failures=$((failures + 1))
  fi
  rm -rf "${tree}"
}

OPEN=$'void Eval() {\n  int x = 0;'

run_case "status captured" 0 '' "${OPEN}"$'\n  const arm_cmsis_nn_status s = arm_one_line(x);\n}'
run_case "status captured on the next line" 0 '' \
  "${OPEN}"$'\n  const arm_cmsis_nn_status s =\n      arm_split_line(x,\n                     x);\n}'
run_case "status checked in a macro" 0 '' \
  "${OPEN}"$'\n  TF_LITE_ENSURE_EQ(context, arm_one_line(x), ARM_CMSIS_NN_SUCCESS);\n}'
run_case "status returned" 0 '' "${OPEN}"$'\n  return arm_one_line(x);\n}'
run_case "status compared in an if" 0 '' "${OPEN}"$'\n  if (arm_one_line(x) != 0) {\n  }\n}'
run_case "void entry point" 0 '' "${OPEN}"$'\n  arm_returns_void(x);\n}'
run_case "size query used" 0 '' "${OPEN}"$'\n  arm_returns_size(x);\n}'
run_case "call in a line comment" 0 '' "${OPEN}"$'\n  // arm_one_line(x);\n}'
run_case "call in a block comment" 0 '' "${OPEN}"$'\n  /*\n  arm_one_line(x);\n  */\n}'
run_case "name only in a header comment" 0 '' "${OPEN}"$'\n  arm_in_comment(x);\n}'

run_case "discarded after a statement" 1 3 "${OPEN}"$'\n  arm_one_line(x);\n}'
run_case "discarded after a brace" 1 2 $'void Eval() {\n  arm_one_line(0);\n}'
run_case "discarded with (void)" 1 3 "${OPEN}"$'\n  (void)arm_one_line(x);\n}'
run_case "discarded, return type on its own line" 1 3 \
  "${OPEN}"$'\n  arm_split_line(x,\n                 x);\n}'
run_case "discarded in an else branch" 1 6 \
  "${OPEN}"$'\n  if (x) {\n    x = 1;\n  } else\n    arm_one_line(x);\n}'
run_case "discarded after a closing comment" 1 4 \
  "${OPEN}"$'\n  x = 1;  /* note */\n  arm_one_line(x);\n}'
run_case "CRLF line ends" 1 3 "${OPEN//$'\n'/$'\r\n'}"$'\r\n  arm_one_line(x);\r\n}'

run_case "discarded after a case label" 1 5 \
  "${OPEN}"$'\n  switch (x) {\n    case 1:\n      arm_one_line(x);\n      break;\n  }\n}'
run_case "discarded after default" 1 5 \
  "${OPEN}"$'\n  switch (x) {\n    default:\n      arm_one_line(x);\n  }\n}'
run_case "discarded under an unbraced if" 1 4 "${OPEN}"$'\n  if (x > 0)\n    arm_one_line(x);\n}'
run_case "discarded under an unbraced else if" 1 5 \
  "${OPEN}"$'\n  if (x > 1) x = 2;\n  else if (x > 0)\n    arm_one_line(x);\n}'
run_case "discarded under an unbraced for" 1 4 "${OPEN}"$'\n  for (int i = 0; i < 2; ++i)\n    arm_one_line(i);\n}'
run_case "discarded after a preprocessor line" 1 5 \
  "${OPEN}"$'\n  x = 1;\n#if defined(X)\n  arm_one_line(x);\n#endif\n}'
run_case "discarded in TFLITE_DCHECK_EQ" 1 3 \
  "${OPEN}"$'\n  TFLITE_DCHECK_EQ(arm_one_line(x), ARM_CMSIS_NN_SUCCESS);\n}'
run_case "discarded in a multi-line TFLITE_DCHECK_EQ" 1 3 \
  "${OPEN}"$'\n  TFLITE_DCHECK_EQ(\n      arm_split_line(x, x),\n      ARM_CMSIS_NN_SUCCESS);\n}'
run_case "discarded after a string with comment markers" 1 4 \
  "${OPEN}"$'\n  const char* s = "a/*b//c";\n  arm_one_line(x);\n}'
run_case "return type mid-line in the header" 1 3 "${OPEN}"$'\n  arm_mid_line(x);\n}' \
  "${HEADER}"$'\nstatic inline arm_cmsis_nn_status arm_mid_line(int a) { return 0; }'
run_case "argument continuation" 0 '' \
  "${OPEN}"$'\n  TF_LITE_ENSURE_EQ(context,\n      arm_one_line(x), ARM_CMSIS_NN_SUCCESS);\n}'
run_case "call opening an argument list" 0 '' \
  "${OPEN}"$'\n  TF_LITE_ENSURE_EQ(\n      arm_one_line(x), ARM_CMSIS_NN_SUCCESS);\n}'
run_case "brace initializer" 0 '' \
  "${OPEN}"$'\n  const arm_cmsis_nn_status statuses[] = {\n      arm_one_line(x)};\n}'
run_case "no status declarations" 2 '' "${OPEN}"$'\n}' 'void arm_only_void(int a);'

for empty in include src; do
  tree="${WORK}/empty-${empty}"
  mkdir -p "${tree}/include" "${tree}/src"
  [[ ${empty} == include ]] || printf '%s\n' "${HEADER}" > "${tree}/include/arm_nnfunctions.h"
  [[ ${empty} == src ]] || printf 'void f() {}\n' > "${tree}/src/kernel.cc"
  cases=$((cases + 1))
  "${CHECK}" "${tree}/include" "${tree}/src" </dev/null >/dev/null 2>&1
  if [[ $? -ne 2 ]]; then
    echo "FAIL empty ${empty} directory: expected exit 2"
    failures=$((failures + 1))
  fi
done

tree="${WORK}/missing"
cases=$((cases + 1))
"${CHECK}" "${tree}/include" "${tree}/src" >/dev/null 2>&1
if [[ $? -ne 2 ]]; then
  echo "FAIL missing directories: expected exit 2"
  failures=$((failures + 1))
fi

if [[ ${failures} -ne 0 ]]; then
  echo "check_helia_cmsis_nn_status_checks_test: ${failures} of ${cases} cases failed"
  exit 1
fi
echo "check_helia_cmsis_nn_status_checks_test: ${cases} cases passed"
