#!/usr/bin/env bash
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

# Check the download path argument
#
# Parameter(s):
#   ${1} - path to the download directory or --no-downloads
#
# Outputs:
# "yes" or "no"
function check_should_download() {
  if [[ ${1} == "--no-downloads" ]]; then
    echo "no"
  else
    echo "yes"
  fi
}

# Show the download URL and MD5 checksum
#
# Parameter(s):
#   ${1} - download URL
#   ${2} - download MD5 checksum
#
# Download scripts require informational output should be on stderr.
function show_download_url_md5() {
  echo >&2 "LIBRARY_URL=${1}"
  echo >&2 "LIBRARY_MD5=${2}"
}

# Compute the MD5 sum.
#
# Parameter(s):
#   ${1} - path to the file
function compute_md5() {
  UNAME_S=`uname -s`
  if [ ${UNAME_S} == Linux ]; then
    tflm_md5sum=md5sum
  elif [ ${UNAME_S} == Darwin ]; then
    tflm_md5sum='md5 -r'
  else
    tflm_md5sum=md5sum
  fi
  ${tflm_md5sum} ${1} | awk '{print $1}'
}

# Check that MD5 sum matches expected value.
#
# Parameter(s):
#   ${1} - path to the file
#   ${2} - expected md5
function check_md5() {
  MD5=`compute_md5 ${1}`

  if [[ ${MD5} != ${2} ]]
  then
    echo "Bad checksum. Expected: ${2}, Got: ${MD5}"
    exit 1
  fi

}

# Create a git repo in a folder.
#
# Parameter(s):
#   $[1} - relative path to folder
create_git_repo() {
  pushd ${1} > /dev/null
  git init . > /dev/null
  git config user.email "tflm@google.com" --local
  git config user.name "TFLM" --local
  git add . >&2 2> /dev/null
  git commit -a -m "Commit for a temporary repository." > /dev/null
  git checkout -b tflm > /dev/null
  popd > /dev/null
}

# Create a new commit with a patch in a folder that has a git repo.
#
# Parameter(s):
#   $[1} - relative path to folder
#   ${2} - path to patch file (relative to ${1})
#   ${3} - commit nessage for the patch
function apply_patch_to_folder() {
  pushd ${1} > /dev/null
  echo >&2 "Applying ${PWD}/${1}/${2} to ${PWD}/${1}"
  git apply --ignore-space-change --ignore-whitespace ${2}
  git commit -a -m "${3}" > /dev/null
  popd > /dev/null
}

# Download a URL to a file, retrying on transient server errors.
#
# GitHub's archive/release endpoints intermittently return 5xx responses or
# drop connections mid-header. wget treats HTTP errors as fatal by default,
# so a single 503 fails the whole build even though an immediate retry would
# succeed. Retry both connection failures and retryable HTTP status codes,
# with wget's built-in linear backoff (--waitretry) between attempts.
#
# Parameter(s):
#   ${1} - download URL
#   ${2} - path to the output file
#   remaining arguments are passed through to wget (e.g. -4)
function wget_with_retries() {
  local url="${1}"
  local output="${2}"
  shift 2
  # 15 tries with linear backoff capped at 15s waits ~105s in total before
  # giving up — GitHub's 503 spells have been observed to outlast a ~30s
  # window, so a minute-scale ceiling is deliberate.
  wget --tries=15 --waitretry=15 --retry-connrefused \
      --retry-on-http-error=429,500,502,503,504 \
      "$@" "${url}" -O "${output}" >&2
}

# ---------------------------------------------------------------------------
# Seed-file helpers for reproducible third-party downloads.
#
# Each download directory stores a small stamp file (.tflm_seed) that records
# the URL+MD5 (or commit hash) used to populate it.  On subsequent runs the
# stamp is compared against the expected value; a mismatch triggers automatic
# re-download so stale caches can never silently break the build.
#
# Set  TFLM_FORCE_REDOWNLOAD=1  to unconditionally re-download everything
# (useful for experimentation or cache-busting).
# ---------------------------------------------------------------------------
TFLM_SEED_FILE=".tflm_seed"

# Write a seed stamp into a download directory.
#   $1 - download directory
#   $2 - seed string (URL + MD5, commit hash, etc.)
function write_seed() {
  echo "${2}" > "${1}/${TFLM_SEED_FILE}"
}

# Check whether a download directory matches its expected seed.
# Returns 0 (up-to-date) or 1 (stale / missing / force-redownload).
#   $1 - download directory
#   $2 - expected seed string
function check_seed() {
  if [[ "${TFLM_FORCE_REDOWNLOAD:-0}" == "1" ]]; then
    return 1
  fi
  local seed_file="${1}/${TFLM_SEED_FILE}"
  if [[ -f "${seed_file}" ]] && [[ "$(cat "${seed_file}")" == "${2}" ]]; then
    return 0
  fi
  return 1
}

# Verify an existing download directory against its expected seed.  If the
# directory exists but is stale, remove it so the caller can re-download.
# Prints a human-readable status message to stderr.
# After this function returns:
#   - directory exists  → it is verified, caller should skip the download
#   - directory absent  → caller should proceed with the download
#   $1 - download directory
#   $2 - expected seed string
#   $3 - human-readable library name (for log messages)
function verify_or_remove() {
  if [ -d "${1}" ]; then
    if check_seed "${1}" "${2}"; then
      echo >&2 "${3}: ${1} is up-to-date, skipping download."
      return 0
    fi
    echo >&2 "${3}: stale download in ${1} (seed mismatch), removing for re-download."
    rm -rf "${1}"
    return 1
  fi
  return 1
}


