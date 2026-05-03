# Enable u-arch specfic behaviours
ifneq (,$(filter $(TARGET_ARCH), x86_64))
  # NS-CMSIS-NN optimizations not supported
endif

# Tracks AmbiqAI/ns-cmsis-nn tag v7.24.0. Consumed by ns_cmsis_nn_download.sh
# via the environment.
NS_CMSIS_NN_COMMIT ?= 8d62c8ce81a3e26d764fdea44f519fb2d4626781
export NS_CMSIS_NN_COMMIT

ifneq (,$(NS_CMSIS_NN_LIBS))
  ifeq (,$(CMSIS_PATH))
    $(error NS_CMSIS_NN_LIBS provided but not CMSIS_PATH)
  endif
  ifeq (,$(NS_CMSIS_NN_PATH))
    $(error NS_CMSIS_NN_LIBS provided but not NS_CMSIS_NN_PATH)
  endif
endif

# Unless an external path is provided we force a download during the first
# phase of make.
CMSIS_DEFAULT_DOWNLOAD_PATH := $(DOWNLOADS_DIR)/cmsis
CMSIS_PATH := $(CMSIS_DEFAULT_DOWNLOAD_PATH)
ifeq ($(CMSIS_PATH), $(CMSIS_DEFAULT_DOWNLOAD_PATH))
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/cmsis_download.sh ${DOWNLOADS_DIR} ${TENSORFLOW_ROOT})
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the CMSIS download: $(DOWNLOAD_RESULT))
  endif
endif
NS_CMSIS_NN_DEFAULT_DOWNLOAD_PATH := $(DOWNLOADS_DIR)/ns_cmsis_nn
NS_CMSIS_NN_PATH := $(NS_CMSIS_NN_DEFAULT_DOWNLOAD_PATH)
ifeq ($(NS_CMSIS_NN_PATH), $(NS_CMSIS_NN_DEFAULT_DOWNLOAD_PATH))
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/ext_libs/ns_cmsis_nn_download.sh ${DOWNLOADS_DIR} ${TENSORFLOW_ROOT})
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the CMSIS-NN download: $(DOWNLOAD_RESULT))
  endif
endif

ifeq (,$(NS_CMSIS_NN_LIBS))
  THIRD_PARTY_KERNEL_CC_SRCS += $(shell find $(NS_CMSIS_NN_PATH)/Source -name "*.c")
else
  MICROLITE_LIBS += $(NS_CMSIS_NN_LIBS)
endif
THIRD_PARTY_CC_HDRS += $(shell find $(NS_CMSIS_NN_PATH)/Include -name "*.h")

# Note all the headers from CMSIS/Core/Include are needed to ensure that the
# project generation scripts copy over the compiler specific implementations of
# the various intrinisics.
THIRD_PARTY_CC_HDRS += \
  $(CMSIS_PATH)/LICENSE \
  $(NS_CMSIS_NN_PATH)/LICENSE \
  $(wildcard $(CMSIS_PATH)/CMSIS/Core/Include/*.h)

# We add -I$(CMSIS_PATH) to enable the code in the TFLM repo (mostly in the
# tensorflow/lite/micro/kernels/ns_cmsis_nn) to use include paths relative to
# the CMSIS code-base.
#
# The CMSIS code itself uses includes such as #include "arm_math.h" and so
# we add $(CMSIS_PATH)/CMSIS/Core/Include etc. to be able to build the CMSIS
# code without any modifications.
INCLUDES += \
  -I$(CMSIS_PATH) \
  -I$(NS_CMSIS_NN_PATH) \
  -I$(CMSIS_PATH)/CMSIS/Core/Include \
  -I$(NS_CMSIS_NN_PATH)/Include

# Helia kernels are CMSIS-NN-backed (linked against NS-CMSIS-NN, including
# arm_nnfunctions.h). Defining CMSIS_NN in addition to HELIA lets the
# upstream `#if defined(CMSIS_NN)` guards in shared kernel headers do the
# right thing for helia builds without any per-header drift. See
# helia/patches/inline_drift.md ("Shared kernel headers" section, removed
# 2026-05) for the history.
#
# Macro semantics (use the right one when adding new code):
#   * CMSIS_NN  - "the linked NN library exposes the CMSIS-NN API surface".
#                 Use `#if defined(CMSIS_NN)` for behavior helia inherits
#                 from CMSIS-NN (the common case for Register_*_INT8/16/4
#                 variant declarations in tensorflow/lite/micro/kernels/*.h).
#   * HELIA     - "this build uses OPTIMIZED_KERNEL_DIR=helia". Use
#                 `#if defined(HELIA)` only for heliaCORE-exclusive
#                 features that NS-CMSIS-NN supports but upstream
#                 CMSIS-NN does not.
# Strongly prefer adding HELIA-only declarations under
# tensorflow/lite/micro/kernels/helia/ (e.g. helia_common.h) over
# inline-patching upstream headers.
#
# Note: ADDITIONAL_DEFINES is folded into COMMON_FLAGS with immediate (:=)
# expansion in tools/make/Makefile *before* this file is included, so we
# append directly to CCFLAGS/CXXFLAGS (which are referenced via deferred
# $(...) expansion in the compile recipes).
CCFLAGS  += -DCMSIS_NN
CXXFLAGS += -DCMSIS_NN

# helia per-kernel optimization knobs.
#
# kernels/helia/{conv,depthwise_conv,transpose_conv,fully_connected,svdf}.cc
# select between speed-optimized and size-optimized implementations using
# defines like CONV_KERNEL_OPTIMIZED_FOR_SPEED and FC_KERNEL_OPTIMIZED_FOR_SPEED.
# These are driven by CONV_OPT / FC_OPT, which default to the project-wide
# GLOBAL_KERNEL_OPTIMIZE knob (declared in tools/make/Makefile so it can
# also drive the upstream-shared OPTIMIZE_KERNELS_FOR variable).
#
# Same CCFLAGS/CXXFLAGS-append pattern as -DCMSIS_NN above: the upstream
# ADDITIONAL_DEFINES capture has already happened by the time this file
# is included, so we bypass it.
CONV_OPT ?= $(GLOBAL_KERNEL_OPTIMIZE)
FC_OPT   ?= $(GLOBAL_KERNEL_OPTIMIZE)
CCFLAGS  += -DCONV_KERNEL_OPTIMIZED_FOR_$(CONV_OPT)
CXXFLAGS += -DCONV_KERNEL_OPTIMIZED_FOR_$(CONV_OPT)
CCFLAGS  += -DFC_KERNEL_OPTIMIZED_FOR_$(FC_OPT)
CXXFLAGS += -DFC_KERNEL_OPTIMIZED_FOR_$(FC_OPT)

# Optional flag that selects an inline-assembly requantize path inside the
# Ambiq-vendored CMSIS-NN. Activated by build_helia.sh / test_helia.sh on
# Cortex-M55 / M4 only. Empty by default to keep upstream behavior.
CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY ?=
ifneq ($(CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY),)
  CCFLAGS  += -DCMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY=$(CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY)
  CXXFLAGS += -DCMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY=$(CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY)
endif

MICROLITE_CC_HDRS += \
  tensorflow/lite/micro/kernels/helia/helia_common.h
