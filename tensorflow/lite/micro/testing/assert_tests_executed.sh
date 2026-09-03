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
# Why this exists: both test frameworks in this tree print the CI pass string,
# '~~~ALL TESTS PASSED~~~', whenever the failure count is zero -- including
# when the executed count is also zero. The pass string alone is therefore not
# evidence that anything ran; a positive executed-case count is.
# see AmbiqAI/helia-rt#231
#
# Usage:
#   assert_tests_executed.sh <log-file> <binary-path>
#
# Exit status: 0 if the log reports at least one executed test case (or the
# binary is a documented framework-less exception), non-zero otherwise.
#
# Side effect: when HELIA_TEST_TALLY_FILE is set in the environment, appends
# one '<binary-name><TAB><executed-cases><TAB><kind>' line so the calling CI
# script can report and floor-check a per-leg total. <kind> is 'counted' or
# 'frameworkless'; both are recorded, so the per-leg binary count stays honest.
# Unset (the upstream CI scripts) it does nothing.

set -euo pipefail

LOG_FILE="${1:?usage: assert_tests_executed.sh <log-file> <binary-path>}"
BINARY_PATH="${2:?usage: assert_tests_executed.sh <log-file> <binary-path>}"
BINARY_NAME="$(basename "${BINARY_PATH}")"

if [[ ! -f "${LOG_FILE}" ]]; then
  echo "ERROR: ${BINARY_NAME}: test log '${LOG_FILE}' not found; cannot" \
       "confirm that any test case executed."
  exit 1
fi

# The tested program's own return value, when the log carries it. The FVP's
# process exit status cannot be used: the program spins after printing
# "Application exit code: %d." and the model stops on uart0.shutdown_on_eot,
# which carries no status. The ATfE path prints no such line, so the assertion
# is conditional on it being present. see AmbiqAI/helia-rt#231
app_exit="$({ grep -aoE 'Application exit code: -?[0-9]+' "${LOG_FILE}" \
              || true; } | tail -n 1 | sed -E 's/^.*: //')"
if [[ -n "${app_exit}" && "${app_exit}" != "0" ]]; then
  echo "--------------------------------------------------------"
  echo "ERROR: ${BINARY_NAME}: the program returned ${app_exit}."
  echo "The log reports 'Application exit code: ${app_exit}.', so the test"
  echo "binary itself failed however the rest of the output reads. Both"
  echo "frameworks return kTfLiteError from main() when a case fails."
  echo "--------------------------------------------------------"
  exit 1
fi

# Append one line to the per-leg tally, if the caller asked for one.
# Arguments: <executed-cases> <kind>.
record_tally() {
  if [[ -n "${HELIA_TEST_TALLY_FILE:-}" ]]; then
    printf '%s\t%s\t%s\n' "${BINARY_NAME}" "${1}" "${2}" \
      >> "${HELIA_TEST_TALLY_FILE}"
  fi
}

# Binaries that legitimately report no case count: hand-rolled main() programs
# that print the pass string themselves and never link either micro-test
# framework, so there is no registry to be empty and no count to assert. Keep
# this list exact -- an entry here stops being covered by this guard. Every
# other pass-string emitter prints a count banner ahead of the pass string.
FRAMEWORKLESS_BINARIES=(
  hello_world_test
)

for exempt in "${FRAMEWORKLESS_BINARIES[@]}"; do
  if [[ "${BINARY_NAME}" == "${exempt}" ]]; then
    echo "${BINARY_NAME}: no case count expected (framework-less test binary)"
    # It still ran, so it counts as a binary for the per-leg tally, with zero
    # cases and flagged, so a HELIA_MIN_TEST_BINARIES floor stays honest.
    record_tally 0 frameworkless
    exit 0
  fi
done

# A count banner is necessary but not sufficient, because the caller reaches
# this script via a bare `grep -q '~~~ALL TESTS PASSED~~~'`:
#  1. A binary can print the pass string itself from inside a test body, so a
#     log can hold both that string and a real framework failure marker
#     ('~~~SOME TESTS FAILED~~~' or '[  FAILED  ]').
#  2. Two runs concatenated: the count taken below is the last one, so a
#     failing or empty first run could hide behind a later good one.
# see AmbiqAI/helia-rt#231
if grep -aqE '~~~SOME TESTS FAILED~~~|\[ +FAILED +\]' "${LOG_FILE}"; then
  echo "--------------------------------------------------------"
  echo "ERROR: ${BINARY_NAME}: the log contains a test-failure marker."
  grep -aoE '~~~SOME TESTS FAILED~~~|\[ +FAILED +\].*' "${LOG_FILE}" \
    | head -n 5
  echo "The pass string is not evidence when the framework reported a"
  echo "failure: a binary can print '~~~ALL TESTS PASSED~~~' itself from"
  echo "inside a test body. Treat this run as FAILED. See issue #231."
  echo "--------------------------------------------------------"
  exit 1
fi

summaries="$({ grep -aoE '\[==========\] [0-9]+ tests ran|[0-9]+/[0-9]+ tests passed' \
                 "${LOG_FILE}" || true; } | wc -l | tr -d '[:space:]')"
if [[ "${summaries}" -gt 1 ]]; then
  echo "--------------------------------------------------------"
  echo "ERROR: ${BINARY_NAME}: ${summaries} framework summaries in one log."
  echo "The log looks like more than one run concatenated, so no single"
  echo "executed-case count describes this binary and a bad run could be"
  echo "masked by a good one. Give each binary its own log. See issue #231."
  echo "--------------------------------------------------------"
  exit 1
fi

# micro_test_v2.h: "[==========] %d tests ran." Matched with grep -oE rather
# than an anchored sed so UART output that wraps the line still parses; take
# the last occurrence. -a is load-bearing: some tests emit raw bytes into the
# log, and grep -o on a file it classifies as binary prints nothing at all, so
# the count would come back empty for a binary that ran fine.
executed="$(grep -aoE '\[==========\] [0-9]+ tests ran' "${LOG_FILE}" \
            | tail -n 1 | grep -oE '[0-9]+' || true)"

if [[ -z "${executed}" ]]; then
  # micro_test.h (v1): "%d/%d tests passed" -- passed/total. The total is the
  # executed count.
  executed="$(grep -aoE '[0-9]+/[0-9]+ tests passed' "${LOG_FILE}" \
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

record_tally "${executed}" counted

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
