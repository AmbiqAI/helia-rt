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
# Rejects `micro::KernelRunner runner(Register_X(), ...)`.
#
# Why: a temporary bound to a reference *member* through a constructor is NOT
# lifetime-extended, so every later call reads a dead stack slot. KernelRunner
# stores the registration by value now, so this keeps the *source* idiom honest
# across upstream syncs (see helia/patches/inline_drift.md).
# see AmbiqAI/helia-rt#239
#
# The correct idiom, which every other test in these files already uses:
#
#   const TFLMRegistration registration = Register_X();
#   micro::KernelRunner runner(registration, tensors, ...);
#
# What is flagged: a KernelRunner constructed from a *call expression*, on the
# constructor line or wrapped onto the next line. Comment lines are skipped,
# and passing an existing object is fine.
#
# Opt-out: mark the constructor line with a trailing
#
#   // NOLINT(kernelrunner-temporary)
#
# when the referent genuinely outlives the runner (a function-local static, a
# file-scope constant).
#
# Usage: check_helia_kernel_runner_temporaries.sh [root]
#   root defaults to the repository root inferred from this script's location.
#
# Exit codes: 0 clean, 1 offending construction(s) found, 2 usage error.

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"

if [[ ! -d "${ROOT}/tensorflow/lite/micro/kernels" ]]; then
  echo "usage: $(basename "$0") [repo-root]" >&2
  echo "error: '${ROOT}' does not look like a helia-rt checkout" >&2
  exit 2
fi

cd "${ROOT}"

# Test sources plus the shared test helpers that also build runners
# (conv_test.h, decode_test_helpers.h, conv_test_common.cc, ...).
# shellcheck disable=SC2016  # $0 / FILENAME below are awk's, not the shell's.
HITS="$(find tensorflow/lite/micro \
    \( -name '*_test.cc' -o -name '*_test*.h' -o -name '*_test_common.cc' \) \
    -print0 \
  | xargs -0 awk '
      BEGIN {
        id      = "[A-Za-z_][A-Za-z0-9_]*"
        # A call expression: optional namespace qualifiers, then Name(args).
        # [^()] keeps it to a single level, which is all the idiom needs.
        call    = "(" id "::)*" id "\\([^()]*\\)"
        ctor    = "KernelRunner[[:space:]]+" id "[[:space:]]*\\("
        comment = "^[[:space:]]*(//|/\\*|\\*)"
        nolint  = "NOLINT\\(kernelrunner-temporary\\)"
      }
      FNR == 1 { prev = "" }
      {
        if ($0 ~ comment) { prev = ""; next }

        if ($0 ~ ctor "[[:space:]]*" call) {
          if ($0 !~ nolint) {
            printf "%s:%d: %s\n", FILENAME, FNR, $0
          }
        } else if (prev ~ ctor "[[:space:]]*$" &&
                   $0 ~ "^[[:space:]]*" call "[[:space:]]*,") {
          if (prev !~ nolint && $0 !~ nolint) {
            printf "%s:%d: %s\n", FILENAME, FNR - 1, prev
          }
        }
        prev = $0
      }
    ')"

if [[ -n "${HITS}" ]]; then
  COUNT="$(printf '%s\n' "${HITS}" | wc -l | tr -d ' ')"
  echo "ERROR: ${COUNT} KernelRunner construction(s) from a temporary registration:"
  printf '%s\n' "${HITS}"
  cat <<'EOF'

KernelRunner keeps the registration past the constructing statement, so the
argument must outlive the runner. Replace each hit with:

  const TFLMRegistration registration = Register_X();
  micro::KernelRunner runner(registration, ...);

If the argument is an accessor returning a reference that genuinely outlives
the runner, mark the constructor line // NOLINT(kernelrunner-temporary).

See AmbiqAI/helia-rt#239.
EOF
  exit 1
fi

echo "check_helia_kernel_runner_temporaries: no KernelRunner temporaries found."
exit 0
