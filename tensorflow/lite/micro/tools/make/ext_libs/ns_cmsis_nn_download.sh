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
DOWNLOADED_NS_CMSIS_NN_PATH=${DOWNLOADS_DIR}/ns_cmsis_nn

if [ -n "${NS_CMSIS_NN_SSH_KEY}" ]; then
  NS_CMSIS_NN_URL="https://${NS_CMSIS_NN_SSH_KEY}@github.com/AmbiqAI/ns-cmsis-nn.git"
else
  NS_CMSIS_NN_URL="git@github.com:AmbiqAI/ns-cmsis-nn.git"
fi

# Set GIT_COMMIT to NS_CMSIS_NN_COMMIT if set, otherwise use default
GIT_COMMIT=${NS_CMSIS_NN_COMMIT:-a03574150e3faa7cbe154e66c93bb9e05926a64d}

should_download=$(check_should_download ${DOWNLOADS_DIR})

if [[ ${should_download} == "no" ]]; then
  show_download_url_md5 ${NS_CMSIS_NN_URL} ${GIT_COMMIT}
elif [ ! -d ${DOWNLOADS_DIR} ]; then
  echo "The top-level downloads directory: ${DOWNLOADS_DIR} does not exist."
  exit 1
elif [ -d ${DOWNLOADED_NS_CMSIS_NN_PATH} ]; then
  # Check that the existing clone is at the right commit
  pushd ${DOWNLOADED_NS_CMSIS_NN_PATH} > /dev/null
  CURRENT_COMMIT=$(git rev-parse HEAD)
  popd > /dev/null

  if [ "${CURRENT_COMMIT}" = "${GIT_COMMIT}" ]; then
    echo >&2 "ns-cmsis-nn is already at ${GIT_COMMIT}, skipping download."
  else
    echo >&2 "ns-cmsis-nn is at ${CURRENT_COMMIT} but expected ${GIT_COMMIT}, redownloading."
    rm -rf ${DOWNLOADED_NS_CMSIS_NN_PATH}
    git clone ${NS_CMSIS_NN_URL} ${DOWNLOADED_NS_CMSIS_NN_PATH} >&2
    pushd ${DOWNLOADED_NS_CMSIS_NN_PATH} > /dev/null
    git checkout ${GIT_COMMIT} >&2
    popd > /dev/null
  fi

else
  git clone ${NS_CMSIS_NN_URL} ${DOWNLOADED_NS_CMSIS_NN_PATH} >&2
  pushd ${DOWNLOADED_NS_CMSIS_NN_PATH} > /dev/null
  git checkout ${GIT_COMMIT} >&2
  popd > /dev/null
fi

echo "SUCCESS"
