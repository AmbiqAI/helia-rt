#!/usr/bin/env bash

set -e

CURRENT_USER=$(whoami)
if [ "$CURRENT_USER" != "ambiqai" ]; then
  echo "Current user is '$CURRENT_USER'. Attempting to switch to 'ambiqai' using sudo..." >&2
  SCRIPT_PATH="$(readlink -f "$0")"
  exec sudo -u ambiqai env ARM_UBL_LICENSE_IDENTIFIER="$ARM_UBL_LICENSE_IDENTIFIER" "$SCRIPT_PATH" "$@"
fi

echo "Starting to build TFLM"

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TFLM_SRC_DIR="$DIR/../"
DOWNLOADS_DIR="$TFLM_SRC_DIR/tensorflow/lite/micro/tools/make/downloads"

BUILD_DIR="$TFLM_SRC_DIR/build"
BUILD_LIB_DIR="$BUILD_DIR/lib"

mkdir -p "$BUILD_DIR"
mkdir -p "$BUILD_LIB_DIR"

readonly TARGET="cortex_m_generic"
readonly OPTIM_KERNEL="ambiq"
readonly TARGET_ARCHS=("cortex-m4+fp" "cortex-m55")
readonly BUILDS=("debug" "release" "release_with_logs")
ARM_UBL_LICENSE_IDENTIFIER=${ARM_UBL_LICENSE_IDENTIFIER:-}


# Always run GCC build
TOOLCHAINS=("gcc")

# If ARM UBL license is available, prepare to run armclang builds too
if [ -n "$ARM_UBL_LICENSE_IDENTIFIER" ]; then
  echo "ARM UBL License detected. Adding armclang builds." >&2

  # if [ "$(whoami)" != "ambiqai" ]; then
  #     echo "Error: This script must be run as the 'ambiqai' user when building for arm clang." >&2
  #     exit 1
  # fi

  ARM_COMPILER_INSTALLER="$DOWNLOADS_DIR/arm_compiler_download.sh"
  ARM_COMPILER_DIR="$DOWNLOADS_DIR/arm_compiler"
  ARM_COMPILER_BIN="$ARM_COMPILER_DIR/bin/" # need trailing slash for makefile

  # 1. Install ARM compiler (if not already installed)
  if [ ! -d "$ARM_COMPILER_BIN" ]; then
    echo "Installing ARM Compiler to $ARM_COMPILER_DIR" >&2
    bash "$ARM_COMPILER_INSTALLER" "$DOWNLOADS_DIR" "$TFLM_SRC_DIR"
  else
    echo "ARM Compiler already installed at $ARM_COMPILER_DIR" >&2
  fi

  # 2. Activate UBL license if not set
  echo "Activating UBL license..." >&2
  "${ARM_COMPILER_BIN}armlm" activate --code "$ARM_UBL_LICENSE_IDENTIFIER"

  # 3. Check armclang version reports success
  "${ARM_COMPILER_BIN}armclang" --version
  if [ $? -ne 0 ]; then
    echo "Error: ARM Compiler activation failed. Please check the license identifier." >&2
    exit 1
  fi

  # 4. Add armclang to toolchains to build
  TOOLCHAINS+=("armclang")
fi

cd "$DIR"

# Iterate over all build variants and toolchains (gcc always, armclang if license is present)
for BUILD in "${BUILDS[@]}"; do
  for TARGET_ARCH in "${TARGET_ARCHS[@]}"; do
    TARGET_NAME=$(echo "$TARGET_ARCH" | sed -E 's/^cortex-m([0-9]+).*$/cm\1/')
    BUILD_NAME=$(echo "$BUILD" | tr _ -)

    for TOOLCHAIN in "${TOOLCHAINS[@]}"; do
      echo "Building TFLM with COMPILER=$TOOLCHAIN, BUILD=$BUILD, TARGET_ARCH=$TARGET_ARCH" >&2

      cd "$TFLM_SRC_DIR"

      if [ "$TOOLCHAIN" == "armclang" ]; then
        TARGET_TOOLCHAIN_ROOT="$ARM_COMPILER_BIN"
      else
        TARGET_TOOLCHAIN_ROOT="${DOWNLOADS_DIR}/gcc_embedded/bin/"
      fi

      make -f "$TFLM_SRC_DIR/tensorflow/lite/micro/tools/make/Makefile" \
        TARGET="$TARGET" \
        TARGET_ARCH="$TARGET_ARCH" \
        TOOLCHAIN="$TOOLCHAIN" \
        OPTIMIZED_KERNEL_DIR="$OPTIM_KERNEL" \
        BUILD_TYPE="$BUILD" \
        TARGET_TOOLCHAIN_ROOT="$TARGET_TOOLCHAIN_ROOT" \
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
