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
GIT_COMMIT="22080c68d040c98139e6cb1549473e3149735f4d"

should_download=$(check_should_download ${DOWNLOADS_DIR})

if [[ ${should_download} == "no" ]]; then
  show_download_url_md5 ${NS_CMSIS_NN_URL} ${GIT_COMMIT}
elif [ ! -d ${DOWNLOADS_DIR} ]; then
  echo "The top-level downloads directory: ${DOWNLOADS_DIR} does not exist."
  exit 1
elif [ -d ${DOWNLOADED_NS_CMSIS_NN_PATH} ]; then
  echo >&2 "${DOWNLOADED_NS_CMSIS_NN_PATH} already exists, skipping the download."
else
  rm -rf /tmp/ns-cmsis-nn
  git clone ${NS_CMSIS_NN_URL} ${DOWNLOADED_NS_CMSIS_NN_PATH} >&2
  pushd ${DOWNLOADED_ETHOS_U_CORE_PLATFORM_PATH} > /dev/null
  git checkout ${GIT_COMMIT} >&2
  rm -rf .git
  popd > /dev/null
fi

echo "SUCCESS"
