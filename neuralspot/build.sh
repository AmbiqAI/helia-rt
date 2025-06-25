#!/usr/bin/env bash

set -e

echo "Starting to build TFLM"

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TFLM_SRC_DIR="$DIR/.."
TFLM_MAKE_DIR="$TFLM_SRC_DIR/tensorflow/lite/micro/tools/make"
DOWNLOADS_DIR="$TFLM_MAKE_DIR/downloads"

BUILD_DIR="$TFLM_SRC_DIR/build"
BUILD_LIB_DIR="$BUILD_DIR/lib"

mkdir -p "$BUILD_DIR"
mkdir -p "$BUILD_LIB_DIR"

readonly TARGET="cortex_m_generic"
readonly OPTIM_KERNEL="ambiq"
readonly TARGET_ARCHS=("cortex-m4+fp" "cortex-m55")
readonly BUILDS=("debug" "release" "release_with_logs")
ARM_UBL_LICENSE_IDENTIFIER=${ARM_UBL_LICENSE_IDENTIFIER:-}

TOOLCHAINS=("gcc" "armclang")

cd "$DIR"

# Iterate over all build variants and toolchains (gcc always, armclang if license is present)
for BUILD in "${BUILDS[@]}"; do
  for TARGET_ARCH in "${TARGET_ARCHS[@]}"; do
    TARGET_NAME=$(echo "$TARGET_ARCH" | sed -E 's/^cortex-m([0-9]+).*$/cm\1/')
    BUILD_NAME=$(echo "$BUILD" | tr _ -)

    for TOOLCHAIN in "${TOOLCHAINS[@]}"; do
      echo "Building TFLM with COMPILER=$TOOLCHAIN, BUILD=$BUILD, TARGET_ARCH=$TARGET_ARCH" >&2

      cd "$TFLM_SRC_DIR"

      make -f "$TFLM_SRC_DIR/tensorflow/lite/micro/tools/make/Makefile" \
        TARGET="$TARGET" \
        TARGET_ARCH="$TARGET_ARCH" \
        TOOLCHAIN="$TOOLCHAIN" \
        OPTIMIZED_KERNEL_DIR="$OPTIM_KERNEL" \
        ARM_UBL_LICENSE_IDENTIFIER="$ARM_UBL_LICENSE_IDENTIFIER" \
        CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY=1 \
        BUILD_TYPE="$BUILD" \
        microlite -j8

      LIB_PATH="$TFLM_SRC_DIR/gen/${TARGET}_${TARGET_ARCH}_${BUILD}_${OPTIM_KERNEL}_${TOOLCHAIN}/lib/libtensorflow-microlite.a"
      if [ ! -f "$LIB_PATH" ]; then
        echo "Error: Build failed or missing library at $LIB_PATH" >&2
        exit 1
      fi

      cp "$LIB_PATH" "$BUILD_LIB_DIR/libtensorflow-microlite-${TARGET_NAME}-${TOOLCHAIN}-${BUILD_NAME}.a"
    done
  done
done

echo "Creating TFLM tree"
sudo python3 "$TFLM_SRC_DIR/tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py" \
  --makefile_options "TARGET=$TARGET TARGET_ARCH=$TARGET_ARCH OPTIMIZED_KERNEL_DIR=$OPTIM_KERNEL" \
  "$BUILD_DIR"

cp "$DIR/module.mk" "$BUILD_DIR/module.mk"

echo "TFLM build completed successfully" >&2

echo "SUCCESS"
