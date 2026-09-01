#!/bin/bash
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
# Called with following arguments:
# 1 - Path to the downloads folder which is typically
#     ${TENSORFLOW_ROOT}/tensorflow/lite/micro/tools/make/downloads
# 2 - (optional) TENSORFLOW_ROOT: path to root of the TFLM tree (relative to directory from where the script is called).
#
# This script is called from the Makefile and uses the following convention to
# enable determination of sucess/failure:
#
#   - If the script is successful, the only output on stdout should be SUCCESS.
#     The makefile checks for this particular string.
#
#   - Any string on stdout that is not SUCCESS will be shown in the makefile as
#     the cause for the script to have failed.
#
#   - Any other informational prints should be on stderr.

set -e

TENSORFLOW_ROOT=${2}
source ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/bash_helpers.sh

DOWNLOADS_DIR=${1}
DOWNLOADED_NS_CMSIS_NN_PATH="${DOWNLOADS_DIR}/ns_cmsis_nn"

# AmbiqAI/ns-cmsis-nn is a public repository, so it clones anonymously over
# https and needs no credential. Never put a token in this URL: git echoes the
# remote in "fatal: unable to access '<url>'" errors, so any network failure
# would print the credential into the build log.
NS_CMSIS_NN_URL="https://github.com/AmbiqAI/ns-cmsis-nn.git"

# Some environments carry a global or system git rewrite such as
#
#     [url "git@github.com:"]
#         insteadOf = https://github.com/
#
# which silently converts the anonymous https URL above into an ssh one. A
# machine with no ssh key or known_hosts entry then fails with "Host key
# verification failed" / "Could not read from remote repository" even though
# the repository is public and https connectivity is fine. Other downloads in
# the same build keep working because they use wget, which no git config can
# rewrite -- so the failure looks specific to ns-cmsis-nn and reads like an
# access problem rather than a local config one.
#
# Pin the URL to itself for this clone. git applies at most one insteadOf
# rewrite -- remote.c:alias_url makes a single pass and never re-scans -- and
# keeps the entry with the longest matching prefix, so pinning the full URL to
# itself outranks a shorter rule such as "https://github.com/".
#
# Two properties bound what this can do, worth stating so the protection is not
# assumed to be broader than it is:
#
#   - Displacement requires a *strictly* longer prefix, and command-line -c is
#     parsed last, so a rule in a config file exactly as long as this pin still
#     wins. Anything shorter loses wherever it lives. Defeating this therefore
#     takes a rule written against this specific repository -- deliberate
#     configuration, not the accidental host-wide rewrite described above.
#   - The pin is ${NS_CMSIS_NN_URL} itself rather than a hand-copied
#     substring, so the two cannot drift apart on a repository rename and the
#     prefix is as long as it can be. A shorter literal would silently stop
#     matching, leaving this comment claiming a protection that no longer
#     exists.
#
# Passing it with -c (rather than blanking GIT_CONFIG_GLOBAL/GIT_CONFIG_SYSTEM)
# leaves legitimate proxy, TLS, and credential settings untouched. Note that
# git clone does not read the enclosing repository's local config, so
# local-scope rules are not part of the threat model either way.

# Set GIT_COMMIT to NS_CMSIS_NN_COMMIT if set, otherwise use default.
# Default tracks AmbiqAI/ns-cmsis-nn tag v7.31.0. Keep in sync with
# NS_CMSIS_NN_COMMIT in ext_libs/helia.inc, which is what make actually passes;
# this fallback only applies when the script is run directly.
#
# Quote every use of GIT_COMMIT below. It is no longer only a literal from
# helia.inc: ns_cmsis_nn_canary.yml routes a workflow_dispatch string into
# NS_CMSIS_NN_COMMIT, so an unquoted expansion would let a crafted ref split
# into extra argv words for `git checkout` (option injection, not shell
# injection -- there is no eval here). The canary also validates the input
# against ^[A-Za-z0-9._/-]+$ before it gets this far; this is the second layer.
GIT_COMMIT=${NS_CMSIS_NN_COMMIT:-9884d5fccab884c90c3d5e8865d5babbb1cabc63}

# clone_ns_cmsis_nn: attempt git clone and surface a clear error on failure.
clone_ns_cmsis_nn() {
  local dest="${1}"
  if git -c url."${NS_CMSIS_NN_URL}".insteadOf="${NS_CMSIS_NN_URL}" \
         clone "${NS_CMSIS_NN_URL}" "${dest}" >&2 2>&1; then
    return 0
  fi
  cat >&2 <<'EOF'

================================================================================
ERROR: Failed to clone the ns-cmsis-nn repository.

The HELIA optimized-kernel backend (OPTIMIZED_KERNEL_DIR=helia) builds against
the public AmbiqAI/ns-cmsis-nn repository, cloned over https with no
credential. This failure is therefore usually environmental: check
connectivity and any proxy or firewall that filters github.com.

If the error above mentions ssh, "Host key verification failed", "Could not
read from remote repository", or a host you did not expect, the clone URL was
rewritten by a git "insteadOf" rule. Inspect it with:

  git config --show-origin --get-regexp '^url\..*\.insteadof$'

This script pins its own URL to outrank host-wide rules of that kind. If you
deliberately redirect github.com to an internal mirror, the pin overrides it;
point the build at a local checkout instead:

  make ... NS_CMSIS_NN_PATH=/path/to/ns-cmsis-nn

You can also build with the open-source CMSIS-NN backend or the reference
kernels instead:

  make ... OPTIMIZED_KERNEL_DIR=cmsis_nn   # Arm CMSIS-NN (open source)
  make ... OPTIMIZED_KERNEL_DIR=           # Reference kernels only

For Zephyr builds, select the backend in prj.conf:

  CONFIG_HELIA_RT_BACKEND_CMSIS_NN=y       # Arm CMSIS-NN (open source)
  CONFIG_HELIA_RT_BACKEND_REFERENCE=y      # Reference kernels only

For help with the HELIA backend, contact support.aitg@ambiq.com.
================================================================================

EOF
  exit 1
}

# checkout_ref: `git checkout` a ref and, on failure, say plainly that the REF
# did not resolve. Without this the bare `git checkout` error ("error: pathspec
# '<ref>' did not match ...") scrolls past inside a make download step and reads
# like infra breakage -- but the common cause now is a canary dispatched with a
# charset-valid but nonexistent ref (a `v7.30.O`-for-`v7.30.0` typo). Must run
# inside the already-cloned repo dir (the callers `pushd` first).
checkout_ref() {
  local ref="${1}"
  if ! git checkout "${ref}" >&2; then
    echo >&2 "ERROR: ns-cmsis-nn ref '${ref}' did not resolve to a commit in AmbiqAI/ns-cmsis-nn."
    echo >&2 "       Check the ref passed via NS_CMSIS_NN_COMMIT (or the ns_cmsis_nn_canary"
    echo >&2 "       dispatch input). A valid full SHA, tag (e.g. v7.30.1), or branch is required."
    exit 1
  fi
}

should_download=$(check_should_download "${DOWNLOADS_DIR}")

if [[ ${should_download} == "no" ]]; then
  show_download_url_md5 "${NS_CMSIS_NN_URL}" "${GIT_COMMIT}"
elif [ ! -d "${DOWNLOADS_DIR}" ]; then
  echo "The top-level downloads directory: ${DOWNLOADS_DIR} does not exist."
  exit 1
elif [ -d "${DOWNLOADED_NS_CMSIS_NN_PATH}" ]; then
  if [[ "${TFLM_FORCE_REDOWNLOAD:-0}" == "1" ]]; then
    echo >&2 "TFLM_FORCE_REDOWNLOAD set, re-downloading ns-cmsis-nn."
    rm -rf "${DOWNLOADED_NS_CMSIS_NN_PATH}"
    clone_ns_cmsis_nn "${DOWNLOADED_NS_CMSIS_NN_PATH}"
    pushd "${DOWNLOADED_NS_CMSIS_NN_PATH}" > /dev/null
    checkout_ref "${GIT_COMMIT}"
    popd > /dev/null
  else
    # Check that the existing clone is at the right commit. Only trust the
    # skip when GIT_COMMIT is a full SHA: resolving a branch or tag name
    # inside the stale local clone would compare against the stale ref and
    # could keep an outdated tree (branches move; tags can be re-cut).
    # Non-SHA pins always redownload, matching the historical behavior.
    pushd "${DOWNLOADED_NS_CMSIS_NN_PATH}" > /dev/null
    CURRENT_COMMIT=$(git rev-parse HEAD)
    if [[ "${GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
      EXPECTED_COMMIT=$(git rev-parse --verify "${GIT_COMMIT}^{commit}" 2>/dev/null || true)
    else
      EXPECTED_COMMIT=""
    fi
    popd > /dev/null

    if [ -n "${EXPECTED_COMMIT}" ] && [ "${CURRENT_COMMIT}" = "${EXPECTED_COMMIT}" ]; then
      echo >&2 "ns-cmsis-nn is already at ${GIT_COMMIT}, skipping download."
    else
      echo >&2 "ns-cmsis-nn is at ${CURRENT_COMMIT} but expected ${GIT_COMMIT}, redownloading."
      rm -rf "${DOWNLOADED_NS_CMSIS_NN_PATH}"
      clone_ns_cmsis_nn "${DOWNLOADED_NS_CMSIS_NN_PATH}"
      pushd "${DOWNLOADED_NS_CMSIS_NN_PATH}" > /dev/null
      checkout_ref "${GIT_COMMIT}"
      popd > /dev/null
    fi
  fi

else
  clone_ns_cmsis_nn "${DOWNLOADED_NS_CMSIS_NN_PATH}"
  pushd "${DOWNLOADED_NS_CMSIS_NN_PATH}" > /dev/null
  checkout_ref "${GIT_COMMIT}"
  popd > /dev/null
fi

echo "SUCCESS"
