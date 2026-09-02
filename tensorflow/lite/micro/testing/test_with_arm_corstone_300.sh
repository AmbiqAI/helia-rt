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
# helia-rt (issues #231 and #239): one log per binary, not a single shared
# logs.txt. Two independent reasons for the same change:
#  - #231: the test rules can run under `make -j`, and with one path two
#    concurrent binaries interleave into it -- binary A's assertion would then
#    parse binary B's banner, or see two framework summaries and fail a run
#    that was fine.
#  - #239: with one path every binary truncates and rewrites the same file, so
#    the log of a binary that hung or faulted is gone as soon as the next one
#    starts, and `make -k` runs roughly ninety more after it.
# `tee` still truncates on open, so a binary that never produces output leaves
# an empty log and still fails the assertion below.
MICRO_LOG_FILENAME=${RESULTS_DIRECTORY}/$(basename "${BINARY_TO_TEST}").txt
mkdir -p ${RESULTS_DIRECTORY}

# helia-rt (issue #239): bound every individual FVP run.
#
# Why: ci_build/helper_functions.sh:readable_run wraps the WHOLE `make test`
# invocation in `timeout 60m`. One binary that never returns therefore burns
# the entire leg budget and kills the run, so nothing after the first hang is
# ever observed. That is exactly what the cortex-m55 + ATfE leg does today:
# kernel_transpose_conv_test printed '[ RUN ] ...Float16Stride1Golden' and
# never came back, so only 37 of ~130 binaries have ever completed there and
# the remaining ~93 have never been executed even once.
#
# A per-binary budget converts that hang into one named FAIL and lets the
# rest of the suite run, which is what makes the failure diagnosable.
#
# Budget: 120 s. The binding constraint is not the slowest binary, it is the
# AGGREGATE when many binaries hang, because readable_run's 60m wrap around
# `make test` is unchanged and `make -k` (on main since #229) keeps going.
#
# Evidence, GitHub Actions run 33557473930 (the run whose ATfE m55 leg hung):
#   - gcc / cortex-m55 / SPEED completed 130 binaries; slowest was
#     integration_tests_nnaed_conv_test at 5.14 s, then person_detection_test
#     at 4.14 s; the bulk sit at ~1.1 s.
#   - atfe / cortex-m55 / SPEED completed 37 binaries before the hang;
#     slowest was integration_tests_nnaed_conv_test at 2.16 s.
#
# Headroom: 120 s is 23x the slowest binary ever measured (5.14 s), so a
# shared CI runner having a bad day does not trip it. This is a hang
# detector, not a performance gate.
#
# Aggregate: ~19 binaries carry the float16 goldens implicated in #239, i.e.
# the worst realistic case is 19 binaries each burning the full budget.
#   19 x 120 s = 38 min, plus ~7 min for the ~130 binaries that do return,
#   = ~45 min, inside readable_run's 60m.
# At 600 s (the first revision of this commit) the same case is
#   19 x 600 s = 190 min, which re-exhausts the 60m leg budget and loses the
# aggregate result -- the per-binary logs would survive, but the leg would be
# killed before most binaries ran, which is the failure this commit exists to
# remove. Hence 120 s.
#
# Override with FVP_TIMEOUT_SECONDS for a local bisect or a slower host.
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
# Enable semihosting so picolibc's libsemihost (used by the ATfE
# toolchain) can route stdout/stderr through SYS_WRITEC/SYS_WRITE0
# to the FVP host. GCC builds use the MPS3 UART instead and are
# unaffected by this setting.
FVP+='-C cpu0.semihosting-enable=1 '
FVP+='--stat'
# helia-rt (issue #231): the FVP's exit status is not the PROGRAM's status.
# The program does not terminate through it -- ethos-u-core-platform's
# retarget.c _exit() prints "Application exit code: N.", 0x04 and EXITTHESIM
# to the UART and then spins, so the model stops on uart0.shutdown_on_eot and
# reports "Simulation stopped by user" (see the sample run in
# tools/benchmarking/README.md). The log remains the authority for the
# program's result: the assertion below checks that exit-code line, the
# executed-case count and the failure markers.
#
# helia-rt (issue #239): the status IS now consulted, for one thing only --
# whether the `timeout` wrapper ended the run. 124 and 137 come from timeout,
# not from the model, and they are the only statuses interpreted below, and
# only when timeout actually ran. Everything about the program's own result
# still comes from the log.
#
# `set -e` does not fire on a failing pipeline element, and the exit status of
# the pipeline is tee's, so the FVP status has to be read out of PIPESTATUS.
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

# helia-rt (issue #239): the fault check runs FIRST, ahead of the timeout and
# status classification below. If a binary both faults and then fails to
# terminate -- which is exactly what happens if the FVP does not honour the
# handler's semihosting SYS_EXIT -- the fault is the actionable diagnosis: it
# names the faulting PC and LR, where a timeout only says the model never came
# back. Reporting the timeout instead would bury the cause.
# helia-rt (issue #239): a fault report fails the binary unconditionally, and
# it is checked BEFORE the pass string. cortex_m_corstone_300/fault_handlers.cc
# prints one '^FAULT: ...' line from the fault handlers. A fault that happens
# after micro_test has already printed '~~~ALL TESTS PASSED~~~' -- in teardown,
# in a static destructor, inside _exit or the semihosting path -- would
# otherwise leave both lines in the log and be reported as a PASS. This applies
# to non_test_binary targets too: an example that faults on the way out is a
# failure whether or not it has a pass string.
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

# Both statuses mean "timeout ended this run", and both must be handled:
#   124 - the FVP died on the SIGTERM timeout sent at the deadline.
#   137 - (128 + SIGKILL) the FVP did not die on SIGTERM and --kill-after had
#         to escalate. timeout propagates the child's signal death here; it
#         does NOT collapse this case to 124.
# Verified on GNU coreutils 9.11: a child with `trap '' TERM` piped through
# tee yields PIPESTATUS[0]=137, while one with the default disposition yields
# 124. This matters because, as the --kill-after comment above says, the FVP
# spawns subprocesses and does not always die on the first signal -- the
# escalating case is the likely one for a wedged model, so dropping 137 would
# have let the exact failure this commit targets fall through to the
# pass-string check and be reported as a plain FAIL with no timeout diagnosis.
#
# The classification is gated on TIMEOUT_CMD. 124 and 137 only carry a timeout
# meaning when `timeout` was the process that produced the status. With no
# timeout on PATH, FVP_STATUS is the FVP's OWN exit code, and 137 there is an
# ordinary SIGKILL death -- an OOM kill is the obvious way to get one. Calling
# that "timed out after ${FVP_TIMEOUT_SECONDS}s" would be a fabricated
# diagnosis, and it would name a budget that was never applied.
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
# Nothing here can tell a hang from a crash, so the conservative reading wins:
# any non-zero status from the FVP is a failure, reported with the raw number
# and no interpretation. It deliberately does NOT fall through to the
# pass-string grep -- a binary that printed the pass string and then died on a
# signal must not be graded PASS on the strength of the string alone.
#
# Note the asymmetry with the bounded path above, which still leaves non-124/137
# statuses to the pass-string check. That is deliberate: the bounded path is
# what CI runs, and tightening it would change the verdict for legs unrelated
# to #239. This branch only affects a host with no coreutils timeout, i.e. a
# local developer run, where being stricter costs nothing.
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
    # helia-rt (issue #231): both micro-test frameworks print the pass string
    # whenever the FAILURE count is zero, which includes the case where the
    # EXECUTED count is also zero. Require a positive executed-case count
    # before calling this a pass, so an empty test registry fails the leg
    # instead of going green.
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
