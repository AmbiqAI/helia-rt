#!/usr/bin/env bash

set -e
pwd

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
TARGET_ARCH=cortex-m4+fp
CO_PROCESSOR=
OPTIMIZED_KERNEL_DIR=ambiq
TOOLCHAINS=(gcc armclang)

# Download third-party dependencies (one-time).
readable_run make -f tensorflow/lite/micro/tools/make/Makefile \
    CO_PROCESSOR=${CO_PROCESSOR} \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    TOOLCHAIN=${TOOLCHAIN} \
    third_party_downloads

# Avoid running tests in parallel.
readable_run make -f tensorflow/lite/micro/tools/make/Makefile \
    clean

readable_run make -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile \
    CO_PROCESSOR=${CO_PROCESSOR} \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    TOOLCHAIN=${TOOLCHAIN} \
    build

readable_run make  -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile \
    CO_PROCESSOR=${CO_PROCESSOR} \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    TOOLCHAIN=${TOOLCHAIN} \
    test_integration_tests_nnaed_conv_test

readable_run make  -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile \
    CO_PROCESSOR=${CO_PROCESSOR} \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    TOOLCHAIN=${TOOLCHAIN} \
    test_integration_tests_nnaed_pad_test

readable_run make  -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile \
    CO_PROCESSOR=${CO_PROCESSOR} \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    TOOLCHAIN=${TOOLCHAIN} \
    test_integration_tests_nnaed_leaky_relu_test

readable_run make -f tensorflow/lite/micro/tools/make/Makefile \
    CO_PROCESSOR=${CO_PROCESSOR} \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    TOOLCHAIN=${TOOLCHAIN} \
    test