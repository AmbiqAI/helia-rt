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
#
# Downloads and extracts the Arm Toolchain for Embedded (ATfE; LLVM/Clang based)
# from https://github.com/arm/arm-toolchain/releases. The result lives at
# ${DOWNLOADS_DIR}/arm_toolchain_embedded with the standard layout:
#
#   arm_toolchain_embedded/
#     bin/   (clang, clang++, llvm-ar, llvm-objcopy, ld.lld, ...)
#     lib/
#     include/
#     ...

readonly ATFE_VERSION=22.1.0

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

DOWNLOADED_ATFE_PATH="${DOWNLOADS_DIR}/arm_toolchain_embedded"

if [ -d "${DOWNLOADED_ATFE_PATH}" ]; then
  echo >&2 "${DOWNLOADED_ATFE_PATH} already exists, skipping the download."
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
      ATFE_ARCHIVE="ATfE-${ATFE_VERSION}-Linux-x86_64.tar.xz"
    elif [ "${UNAME_M}" == "aarch64" ] || [ "${UNAME_M}" == "arm64" ]; then
      ATFE_ARCHIVE="ATfE-${ATFE_VERSION}-Linux-AArch64.tar.xz"
    else
      echo "Unsupported host architecture: ${UNAME_M}"
      exit 1
    fi

  elif [ "${HOST_OS}" == "darwin" ]; then
    ATFE_ARCHIVE="ATfE-${ATFE_VERSION}-Darwin-universal.dmg"

  elif [ "${HOST_OS}" == "win" ]; then
    ATFE_ARCHIVE="ATfE-${ATFE_VERSION}-Windows-x86_64.zip"
  else
    echo "Unsupported host OS: ${HOST_OS}"
    exit 1
  fi

  ATFE_URL="https://github.com/arm/arm-toolchain/releases/download/release-${ATFE_VERSION}-ATfE/${ATFE_ARCHIVE}"
  tempdir=$(mktemp -d)
  trap 'rm -rf "${tempdir}"' EXIT
  tempfile=${tempdir}/${ATFE_ARCHIVE}

  echo >&2 "Downloading ATfE ${ATFE_VERSION} from ${ATFE_URL}"
  wget -4 "${ATFE_URL}" -O "${tempfile}" >&2

  mkdir -p "${DOWNLOADED_ATFE_PATH}"

  case "${ATFE_ARCHIVE}" in
    *.tar.xz)
      tar -xJf "${tempfile}" -C "${DOWNLOADED_ATFE_PATH}" --strip-components=1 >&2
      ;;
    *.zip)
      tmp_extract=${tempdir}/extract
      mkdir -p "${tmp_extract}"
      unzip -q "${tempfile}" -d "${tmp_extract}" >&2
      # Flatten one level (top-level dir is typically ATfE-${ATFE_VERSION}/)
      shopt -s dotglob
      mv "${tmp_extract}"/*/* "${DOWNLOADED_ATFE_PATH}/" >&2
      ;;
    *.dmg)
      # On macOS, mount the dmg and copy the toolchain payload out.
      mountpoint=${tempdir}/mnt
      mkdir -p "${mountpoint}"
      hdiutil attach "${tempfile}" -nobrowse -quiet -mountpoint "${mountpoint}" >&2
      # The dmg contains a top-level ATfE-<version> directory.
      src_dir=$(find "${mountpoint}" -maxdepth 1 -type d -name "ATfE-*" | head -n1)
      if [ -z "${src_dir}" ]; then
        hdiutil detach "${mountpoint}" -quiet >&2 || true
        echo "Could not locate ATfE-* directory inside ${ATFE_ARCHIVE}."
        exit 1
      fi
      cp -R "${src_dir}/." "${DOWNLOADED_ATFE_PATH}/" >&2
      hdiutil detach "${mountpoint}" -quiet >&2 || true
      ;;
    *)
      echo "Unsupported ATfE archive format: ${ATFE_ARCHIVE}"
      exit 1
      ;;
  esac

  if [ ! -x "${DOWNLOADED_ATFE_PATH}/bin/clang" ]; then
    echo "ATfE extraction failed: ${DOWNLOADED_ATFE_PATH}/bin/clang not found."
    exit 1
  fi

  echo >&2 "ATfE ${ATFE_VERSION} setup complete at: ${DOWNLOADED_ATFE_PATH}"
fi

echo "SUCCESS"
