#!/usr/bin/env bash

set -e

# 1. Download dependencies (one-time).
make -f tensorflow/lite/micro/tools/make/Makefile \
CO_PROCESSOR= \
OPTIMIZED_KERNEL_DIR=ambiq \
TARGET=cortex_m_corstone_300 \
TARGET_ARCH=cortex-m55 \
TOOLCHAIN=gcc \
third_party_downloads

# 2. Clean, then build for Cortex-M4+fp
make -f tensorflow/lite/micro/tools/make/Makefile clean
make -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile \
CO_PROCESSOR= \
OPTIMIZED_KERNEL_DIR=ambiq \
TARGET=cortex_m_corstone_300 \
TARGET_ARCH=cortex-m4+fp \
TOOLCHAIN=gcc \
build

# 3. Clean, then build for Cortex-M55
make -f tensorflow/lite/micro/tools/make/Makefile clean
make -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile \
CO_PROCESSOR= \
OPTIMIZED_KERNEL_DIR=ambiq \
TARGET=cortex_m_corstone_300 \
TARGET_ARCH=cortex-m55 \
TOOLCHAIN=gcc \
build

