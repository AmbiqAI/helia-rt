set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${SCRIPT_DIR}/../../../../..
cd "${ROOT_DIR}"

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

# If "armclang" passed as 1st arg, use it. Otherwise default to gcc.
if [ "$1" = "armclang" ]; then
    TOOLCHAIN=armclang
else
    TOOLCHAIN=gcc
fi

TARGET=cortex_m_corstone_300
OPTIMIZED_KERNEL_DIR=ambiq

# Download third-party dependencies (one-time).
readable_run make -f tensorflow/lite/micro/tools/make/Makefile \
  OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
  TARGET=${TARGET} \
  TARGET_ARCH=cortex-m4 \
  TOOLCHAIN=${TOOLCHAIN} \
  third_party_downloads

readable_run make -f tensorflow/lite/micro/tools/make/Makefile clean

readable_run make -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile \
  OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
  TARGET=${TARGET} \
  TARGET_ARCH=cortex-m4+fp \
  TOOLCHAIN=${TOOLCHAIN} \
  build

# Now run the 3 integration tests
readable_run make -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile \
  OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
  TARGET=${TARGET} \
  TARGET_ARCH=cortex-m4+fp \
  TOOLCHAIN=${TOOLCHAIN} \
  test_integration_tests_nnaed_conv_test

readable_run make -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile \
  OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
  TARGET=${TARGET} \
  TARGET_ARCH=cortex-m4+fp \
  TOOLCHAIN=${TOOLCHAIN} \
  test_integration_tests_nnaed_pad_test

readable_run make -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile \
  OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
  TARGET=${TARGET} \
  TARGET_ARCH=cortex-m4+fp \
  TOOLCHAIN=${TOOLCHAIN} \
  test_integration_tests_nnaed_leaky_relu_test

readable_run make -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
    TARGET=${TARGET} \
    TARGET_ARCH=cortex-m4+fp \
    TOOLCHAIN=${TOOLCHAIN} \
    test