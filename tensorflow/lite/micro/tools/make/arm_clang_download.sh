#!/usr/bin/env bash
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

readonly ARM_COMPILER_VERSION=6.23
readonly ARM_COMPILER_REL=32

set -e

if [ -z "$1" ]; then
  echo "Error: Missing required argument for downloads directory." >&2
  exit 1
fi

TENSORFLOW_ROOT=${2:-}

if [ -z "${TENSORFLOW_ROOT}" ]; then
  echo "Warning: TENSORFLOW_ROOT not set. Assuming current directory." >&2
  TENSORFLOW_ROOT="./"
fi

source "${TENSORFLOW_ROOT}/tensorflow/lite/micro/tools/make/bash_helpers.sh"

DOWNLOADS_DIR=${1}
if [ ! -d "${DOWNLOADS_DIR}" ]; then
  echo "The top-level downloads directory: ${DOWNLOADS_DIR} does not exist."
  exit 1
fi

DOWNLOADED_ARM_COMPILER_PATH="${DOWNLOADS_DIR}/arm_compiler"

if [ -d "${DOWNLOADED_ARM_COMPILER_PATH}" ]; then
  echo >&2 "${DOWNLOADED_ARM_COMPILER_PATH} already exists, skipping the download."
else

  HOST_OS=
  if [ "${OS}" == "Windows_NT" ]; then
    HOST_OS=win
  else
    UNAME_S=$(uname -s)
    if [ "${UNAME_S}" == "Linux" ]; then
      HOST_OS=linux
    elif [ "${UNAME_S}" == "Darwin" ]; then
      HOST_OS=darwin
    fi
  fi

  if [ "${HOST_OS}" == "linux" ]; then
    UNAME_M=$(uname -m)
    if [ "${UNAME_M}" == "x86_64" ]; then
      ARM_COMPILER_ARCHIVE="standalone-linux-x86_64-rel.tar.gz"
    elif [ "${UNAME_M}" == "aarch64" ]; then
      ARM_COMPILER_ARCHIVE="standalone-linux-armv8l_64-rel.tar.gz"
    else
      echo "Unsupported host architecture: ${UNAME_M}"
      exit 1
    fi

  elif [ "${HOST_OS}" == "darwin" ]; then
    ARM_COMPILER_ARCHIVE="standalone-darwin-x86_64-rel.tar.gz"

  elif [ "${HOST_OS}" == "win" ]; then
    ARM_COMPILER_ARCHIVE="standalone-win-x86_64-rel.zip"
  else
    echo "Unsupported host OS: ${HOST_OS}"
    exit 1
  fi

  ARM_COMPILER_URL="https://artifacts.tools.arm.com/arm-compiler/$ARM_COMPILER_VERSION/$ARM_COMPILER_REL/$ARM_COMPILER_ARCHIVE"
  tempdir=$(mktemp -d)
  trap 'rm -rf "${tempdir}"' EXIT
  tempfile=${tempdir}/temp_file

  wget -4 "${ARM_COMPILER_URL}" -O "${tempfile}" >&2

  mkdir "${DOWNLOADED_ARM_COMPILER_PATH}"

  if [ "${HOST_OS}" == "win" ]; then
    unzip -q "${tempfile}" -d "${DOWNLOADED_ARM_COMPILER_PATH}" >&2
  else
    tar -xzf "${tempfile}" -C "${DOWNLOADED_ARM_COMPILER_PATH}" >&2
  fi
  echo >&2 "ARM Compiler setup complete at: ${DOWNLOADED_ARM_COMPILER_PATH}"
fi

echo "SUCCESS"
