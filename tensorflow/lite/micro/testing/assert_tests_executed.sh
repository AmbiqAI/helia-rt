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
# one '<binary-name><TAB><executed-cases><TAB><kind>' line so the calling CI
# script can report and floor-check a per-leg total. <kind> is 'counted' for a
# binary that reported a case count and 'frameworkless' for one of the exempt
# binaries below, which ran but has no count to contribute; both are real
# binaries and both are recorded, so the per-leg binary count stays honest.
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

# The tested program's own return value, when the log carries it.
#
# The FVP's process exit status cannot be used for this. On the GCC and
# armclang paths the binary terminates inside ethos-u-core-platform's
# retarget.c _exit(), which prints "Application exit code: %d." followed by
# 0x04 (end-of-transmission) and an "EXITTHESIM" shutdown tag to the MPS3
# UART and then spins in `while (1) {}`. The model stops because of
# `-C mps3_board.uart0.shutdown_on_eot=1`, a UART-model shutdown that carries
# no status -- the sample run in tools/benchmarking/README.md shows exactly
# that pairing: "Application exit code: 0." followed by "Info: /OSCI/SystemC:
# Simulation stopped by user". So the log, not $? of the FVP, is where the
# program's return value survives, and this is the check that reads it.
#
# The ATfE path links picolibc + libsemihost instead of retarget.c and prints
# no such line, so the assertion is conditional on the line being present.
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
    # It still ran, so it counts as a binary for the per-leg tally -- with
    # zero executed cases and flagged so the leg summary can say how many of
    # its binaries carry no count. Dropping it here would undercount the
    # binaries the leg actually executed, and any HELIA_MIN_TEST_BINARIES
    # floor would be measured against an incomplete list.
    record_tally 0 frameworkless
    exit 0
  fi
done

# A count banner is necessary but not sufficient. Two more log shapes must
# not be allowed through, because the caller reaches this script via a bare
# `grep -q '~~~ALL TESTS PASSED~~~'`:
#
#  1. A failure marker anywhere in the log. Both frameworks print the pass
#     string only when their own failure count is zero, but a binary can
#     print that string itself from inside a test body
#     (examples/network_tester/network_tester_test.cc does), so a log can
#     hold both the pass string and a real framework failure.
#     micro_test.h prints '~~~SOME TESTS FAILED~~~'; micro_test_v2.h prints
#     '[  FAILED  ] ...' per failing case and in its summary.
#  2. More than one framework summary in one log, i.e. two runs concatenated.
#     The count taken below is the last one, so a failing or empty first run
#     could hide behind a later good one, and the tally would credit the leg
#     with cases from a run that is not this binary's.
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

# micro_test_v2.h: "[==========] %d tests ran."
# Matched with grep -oE rather than an anchored sed so that FVP/UART output
# that prefixes or wraps the line still parses. Take the last occurrence.
#
# -a (--text) is load-bearing, not defensive. Some tests emit raw bytes into
# the captured log -- micro_log_test deliberately prints badly-formed format
# strings -- and GNU grep then classifies the whole file as binary. With -o
# that suppresses the matched text entirely: grep writes "binary file matches"
# to stderr, exits 0, and prints NOTHING on stdout. The count would come back
# empty and this guard would fail a binary that ran its cases perfectly well.
# The pass-string check in test_with_arm_corstone_300.sh does not hit this
# because grep -q only needs the exit status.
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
