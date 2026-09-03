#!/usr/bin/env bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
#
# Parameters:
#  ${1} - path to a binary to test or directory (all *_test will be run).
#  ${2} - String that is checked for pass/fail.
#  ${3} - target (e.g. cortex_m_generic.)

set -e

BINARY_TO_TEST=${1}
PASS_STRING=${2}
TARGET=${3}

RESULTS_DIRECTORY=/tmp/${TARGET}_logs
# helia-rt: one log per binary, not a single shared logs.txt. The test rules
# can run under `make -j`, and a shared path also loses the log of a binary
# that hung or faulted as soon as the next one starts. `tee` truncates on
# open, so a silent binary leaves an empty log and still fails.
# see AmbiqAI/helia-rt#231, AmbiqAI/helia-rt#239
MICRO_LOG_FILENAME=${RESULTS_DIRECTORY}/$(basename "${BINARY_TO_TEST}").txt
mkdir -p ${RESULTS_DIRECTORY}

# helia-rt: bound every individual FVP run, so one binary that never returns
# fails by name instead of burning the whole `make test` timeout that
# readable_run wraps around the leg. The budget is set by the aggregate case
# (many binaries hanging), not by the slowest binary; it is a hang detector,
# not a performance gate. Override with FVP_TIMEOUT_SECONDS.
# see AmbiqAI/helia-rt#239
FVP_TIMEOUT_SECONDS="${FVP_TIMEOUT_SECONDS:-120}"
# Grace period between SIGTERM and SIGKILL. The FVP spawns subprocesses and
# does not always die on the first signal.
FVP_TIMEOUT_KILL_AFTER_SECONDS="${FVP_TIMEOUT_KILL_AFTER_SECONDS:-30}"

# GNU coreutils `timeout`; on macOS/Homebrew it is installed as `gtimeout`.
TIMEOUT_CMD=""
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_CMD="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_CMD="gtimeout"
else
  echo "WARNING: no 'timeout' or 'gtimeout' on PATH; running ${BINARY_TO_TEST}"
  echo "WARNING: unbounded. A hanging binary will take the whole leg down."
fi

FVP="FVP_Corstone_SSE-300_Ethos-U55 "
FVP+="-C ethosu.num_macs=256 "
FVP+="-C mps3_board.visualisation.disable-visualisation=1 "
FVP+="-C mps3_board.telnetterminal0.start_telnet=0 "
FVP+='-C mps3_board.uart0.out_file="-" '
FVP+='-C mps3_board.uart0.unbuffered_output=1 '
FVP+='-C mps3_board.uart0.shutdown_on_eot=1 '
# Semihosting is required on every toolchain: picolibc's libsemihost (ATfE)
# routes stdout/stderr through it, and cortex_m_corstone_300/fault_handlers.cc,
# linked by architecture on all of them, reports faults through it.
FVP+='-C cpu0.semihosting-enable=1 '
FVP+='--stat'
# helia-rt: the FVP's exit status is not the PROGRAM's status -- the program
# spins after printing its exit code and the model stops on
# uart0.shutdown_on_eot -- so the log is the authority for the program result.
# The status is consulted for one thing only: whether the `timeout` wrapper
# ended the run. `set -e` does not fire on a failing pipeline element and the
# pipeline status is tee's, so read the FVP's out of PIPESTATUS.
# see AmbiqAI/helia-rt#231, AmbiqAI/helia-rt#239
set +e
if [[ -n "${TIMEOUT_CMD}" ]]; then
  ${TIMEOUT_CMD} --kill-after="${FVP_TIMEOUT_KILL_AFTER_SECONDS}" \
    "${FVP_TIMEOUT_SECONDS}" \
    ${FVP} ${BINARY_TO_TEST} | tee ${MICRO_LOG_FILENAME}
else
  ${FVP} ${BINARY_TO_TEST} | tee ${MICRO_LOG_FILENAME}
fi
FVP_STATUS=${PIPESTATUS[0]}
set -e

# helia-rt: cortex_m_corstone_300/fault_handlers.cc prints one '^FAULT:' line.
# The check runs FIRST, ahead of the timeout classification and the pass
# string: the faulting PC/LR is the actionable diagnosis, and a fault after
# '~~~ALL TESTS PASSED~~~' (in teardown, a static destructor, _exit) is still a
# failure -- for non_test_binary targets too. see AmbiqAI/helia-rt#239
if grep -aq '^FAULT:' "${MICRO_LOG_FILENAME}"
then
  echo "--------------------------------------------------------"
  echo "$BINARY_TO_TEST: FAIL - the program took a CPU fault."
  grep -a '^FAULT:' "${MICRO_LOG_FILENAME}"
  echo "Full log: ${MICRO_LOG_FILENAME}. PC/LR in the line above are the"
  echo "faulting instruction and its caller. See issue #239."
  echo "--------------------------------------------------------"
  exit 1
fi

# Both statuses mean "timeout ended this run": 124 from the SIGTERM sent at
# the deadline, 137 (128 + SIGKILL) when --kill-after had to escalate, which
# `timeout` propagates rather than collapsing to 124. Gated on TIMEOUT_CMD,
# because without it FVP_STATUS is the FVP's own status and 137 there is an
# ordinary signal death, not a timeout. see AmbiqAI/helia-rt#239
if [[ -n "${TIMEOUT_CMD}" && ( ${FVP_STATUS} -eq 124 || ${FVP_STATUS} -eq 137 ) ]]
then
  echo "--------------------------------------------------------"
  echo "$BINARY_TO_TEST: FAIL - timed out after ${FVP_TIMEOUT_SECONDS}s."
  echo "The FVP did not return. Its output is in ${MICRO_LOG_FILENAME}; the"
  echo "last '[ RUN ]' line there names the test case that hung. See #239."
  echo "--------------------------------------------------------"
  exit 1
fi

# Unbounded fallback (no timeout/gtimeout on PATH; the WARNING above fired).
# Nothing here can tell a hang from a crash, so any non-zero FVP status is a
# failure and deliberately does not fall through to the pass-string grep.
# Stricter than the bounded path above, which is what CI runs.
# see AmbiqAI/helia-rt#239
if [[ -z "${TIMEOUT_CMD}" && ${FVP_STATUS} -ne 0 ]]
then
  echo "--------------------------------------------------------"
  echo "$BINARY_TO_TEST: FAIL - the FVP exited with status ${FVP_STATUS}."
  echo "This run was UNBOUNDED (no 'timeout' or 'gtimeout' on PATH), so the"
  echo "status is the FVP's own and carries no timeout meaning: 124/137 here"
  echo "are an ordinary exit or signal death, not a per-binary timeout."
  echo "Full log: ${MICRO_LOG_FILENAME}."
  echo "--------------------------------------------------------"
  exit 1
fi


if [[ ${2} != "non_test_binary" ]]
then
  if grep -q "$PASS_STRING" ${MICRO_LOG_FILENAME}
  then
    # helia-rt: both micro-test frameworks print the pass string whenever the
    # FAILURE count is zero, including when the EXECUTED count is zero too, so
    # a positive executed-case count is required as well.
    # see AmbiqAI/helia-rt#231
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    "${SCRIPT_DIR}/assert_tests_executed.sh" \
      "${MICRO_LOG_FILENAME}" "${BINARY_TO_TEST}"
    echo "$BINARY_TO_TEST: PASS"
    exit 0
  else
    echo "$BINARY_TO_TEST: FAIL - '$PASS_STRING' not found in logs."
    exit 1
  fi
fi
