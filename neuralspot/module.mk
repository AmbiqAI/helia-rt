# ---------------------------------------------------------------------------
# DEPRECATED — neuralspot/module.mk (legacy AmbiqSuite Makefile glue)
# ---------------------------------------------------------------------------
# This file is the AmbiqSuite Makefile bridge to the prebuilt heliaRT static
# library matrix (lib/libhelia-rt-<core>-<tc>-<variant>.a). With Phase 2 of
# issue #147 (NSX source mode), all CMake-based consumers — including NSX
# via nsx/CMakeLists.txt — now build heliaRT from source via the canonical
# CMake source-of-truth at cmake/helia_rt_sources.cmake.
#
# This file is preserved during Phase 2 so that:
#   * tensorflow/lite/micro/tools/ci_build/package_helia_bundle.sh and
#     ns_local_build.sh continue to work for the prebuilt release pipeline.
#   * AmbiqSuite Makefile-only consumers can keep linking the prebuilt .a
#     produced by build_helia.sh.
#
# Phase 5 of #147 deprecates the prebuilt pipeline and removes this file
# together with build_helia.sh, package_helia_bundle.sh, and the prebuilt
# Zephyr template. Track that work on the milestone for #147.
# ---------------------------------------------------------------------------

# Include paths
includes_api += $(subdirectory)/.
includes_api += $(subdirectory)/third_party
includes_api += $(subdirectory)/third_party/flatbuffers/include
includes_api += $(subdirectory)/third_party/gemmlowp

# Preprocessor defines
DEFINES += NS_TFSTRUCTURE_RECENT
DEFINES += NS_TFLM_NEW_MICRO_PROFILER

# Determine short architecture tag
TFP := $(if $(filter apollo5,$(ARCH)),cm55,cm4)

# Determine build type suffix
ifeq ($(MLDEBUG),1)
  BUILD_TYPE := debug
else ifeq ($(MLPROFILE),1)
  BUILD_TYPE := release-with-logs
else
  BUILD_TYPE := release
endif

# Determine toolchain name (neuralSPOT uses 'arm' for armclang and 'atfe' for ATfE)
ifeq ($(TOOLCHAIN),arm)
  TOOLCHAIN_NAME := armclang
else ifeq ($(TOOLCHAIN),atfe)
  TOOLCHAIN_NAME := atfe
else
  TOOLCHAIN_NAME := gcc
endif

# Construct final static library path
lib_prebuilt += $(subdirectory)/lib/libhelia-rt-$(TFP)-$(TOOLCHAIN_NAME)-$(BUILD_TYPE).a
