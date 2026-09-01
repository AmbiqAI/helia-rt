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
# helia-rt: assert that a micro-test binary actually EXECUTED test cases.
#
# Why this exists (issue #231): both test frameworks in this tree print the
# CI pass string, '~~~ALL TESTS PASSED~~~', whenever the failure count is
# zero -- including when the executed count is also zero. Under the ATfE
# toolchain no static constructor ran, so micro_test_v2's TEST()
# self-registration list was empty and all ~127 binaries printed
#
#     [==========] 0 tests ran.
#     [  PASSED  ] 0 tests.
#     ~~~ALL TESTS PASSED~~~
#
# and the leg went green. Four required status contexts were vacuous for as
# long as that held. The pass string alone is therefore NOT evidence that
# anything ran; a positive executed-case count is. This script is the check
# that turns "all 0 of 0 passed" into a failure.
#
# Usage:
#   assert_tests_executed.sh <log-file> <binary-path>
#
# Exit status: 0 if the log reports at least one executed test case (or the
# binary is a documented framework-less exception), non-zero otherwise.
#
# Side effect: when HELIA_TEST_TALLY_FILE is set in the environment, appends
# one '<binary-name><TAB><executed-cases>' line so the calling CI script can
# report and floor-check a per-leg total. Unset (the upstream CI scripts) it
# does nothing.

set -euo pipefail

LOG_FILE="${1:?usage: assert_tests_executed.sh <log-file> <binary-path>}"
BINARY_PATH="${2:?usage: assert_tests_executed.sh <log-file> <binary-path>}"
BINARY_NAME="$(basename "${BINARY_PATH}")"

if [[ ! -f "${LOG_FILE}" ]]; then
  echo "ERROR: ${BINARY_NAME}: test log '${LOG_FILE}' not found; cannot" \
       "confirm that any test case executed."
  exit 1
fi

# Binaries that legitimately report no case count. These are hand-rolled
# main() programs that print the pass string directly and never link either
# micro-test framework, so there is no registry to be empty and no count to
# assert. Keep this list exact and short: anything added here stops being
# covered by this guard, so a new entry needs the same justification.
#
#   hello_world_test  - tensorflow/lite/micro/examples/hello_world/
#                       hello_world_test.cc: main() calls three inference
#                       helpers under TF_LITE_ENSURE_STATUS and prints the
#                       pass string itself. No TEST()/TF_LITE_MICRO_TEST.
#
# Every other pass-string emitter in the tree is one of the two frameworks
# (testing/micro_test.h, testing/micro_test_v2.h), both of which always print
# a count banner ahead of the pass string.
FRAMEWORKLESS_BINARIES=(
  hello_world_test
)

for exempt in "${FRAMEWORKLESS_BINARIES[@]}"; do
  if [[ "${BINARY_NAME}" == "${exempt}" ]]; then
    echo "${BINARY_NAME}: no case count expected (framework-less test binary)"
    exit 0
  fi
done

# micro_test_v2.h: "[==========] %d tests ran."
# Matched with grep -oE rather than an anchored sed so that FVP/UART output
# that prefixes or wraps the line still parses. Take the last occurrence.
executed="$(grep -oE '\[==========\] [0-9]+ tests ran' "${LOG_FILE}" \
            | tail -n 1 | grep -oE '[0-9]+' || true)"

if [[ -z "${executed}" ]]; then
  # micro_test.h (v1): "%d/%d tests passed" -- passed/total. The total is the
  # executed count.
  executed="$(grep -oE '[0-9]+/[0-9]+ tests passed' "${LOG_FILE}" \
              | tail -n 1 | cut -d/ -f2 | grep -oE '[0-9]+' || true)"
fi

if [[ -z "${executed}" ]]; then
  echo "--------------------------------------------------------"
  echo "ERROR: ${BINARY_NAME}: no executed-test-case count found in the log."
  echo "Neither the micro_test_v2 banner ('[==========] N tests ran.') nor"
  echo "the micro_test banner ('N/M tests passed') was printed, so the pass"
  echo "string cannot be trusted as evidence that any test ran. See issue"
  echo "#231. If this binary genuinely has no test cases by design, add it"
  echo "to FRAMEWORKLESS_BINARIES in $(basename "${BASH_SOURCE[0]}") with a"
  echo "justification -- do not relax this check."
  echo "--------------------------------------------------------"
  exit 1
fi

if [[ -n "${HELIA_TEST_TALLY_FILE:-}" ]]; then
  printf '%s\t%s\n' "${BINARY_NAME}" "${executed}" \
    >> "${HELIA_TEST_TALLY_FILE}"
fi

if [[ "${executed}" -eq 0 ]]; then
  echo "--------------------------------------------------------"
  echo "ERROR: ${BINARY_NAME}: 0 test cases executed."
  echo "The binary linked and ran but registered no test cases, so its"
  echo "'~~~ALL TESTS PASSED~~~' proves nothing. This is a harness or link"
  echo "failure, not a passing test. See issue #231."
  echo "--------------------------------------------------------"
  exit 1
fi

echo "${BINARY_NAME}: ${executed} test cases executed"
