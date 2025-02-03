# # Directories where sources are located
# KERNELS_DIR  := tensorflow/lite/micro/kernels
# KERNELS_AMBIQ := $(KERNELS_DIR)/ambiq

# # Collect all .cc files
# SRCS := $(wildcard $(KERNELS_DIR)/*.cc) \
#         $(wildcard $(KERNELS_AMBIQ)/*.cc)

# # Add them to the "sources" variable the top-level uses
# sources += $(SRCS)

# # If you have any public includes or necessary -I paths:
# includes_api += tensorflow/lite/micro/kernels
# includes_api += tensorflow/lite/micro/kernels/ambiq

# # If you are building a library for these object files:
# myLibName := $(BINDIR)/libtensorflow-microlite-$(TFP)-gcc$(GCC_VERSION).a

# # Let the top-level know that we want to build this library
# libraries += $(myLibName)

# # In order for the top-level Makefile to actually build the library, we declare a rule:
# $(myLibName): $(filter %.o,$(call source-to-object, $(SRCS)))
# 	@echo "Creating static library $@"
# 	@$(AR) rcs $@ $^




# module.mk for ns-tflm
# This file is included by the toplevel Makefile.

# 1) Gather all the sources in your submodule
local_src := $(wildcard $(subdirectory)/tensorflow/lite/micro/kernels/*.cc)
local_src += $(wildcard $(subdirectory)/tensorflow/lite/micro/kernels/ambiq/*.cc)
# If you have .c, .cpp, or .s files, include those too:
# local_src += $(wildcard $(subdirectory)/tensorflow/lite/micro/kernels/*.c)
# local_src += $(wildcard $(subdirectory)/tensorflow/lite/micro/kernels/*.s)
# etc.

# 2) If you have any public headers you want other modules to see, add them:
includes_api += $(subdirectory)/includes-api
# Or if you want direct TFLM kernel headers:
includes_api += $(subdirectory)/tensorflow/lite/micro/kernels
includes_api += $(subdirectory)/tensorflow/lite/micro/kernels/ambiq

# If you need custom preprocessor defines, do:
# pp_defines += SOMETHING=1

# 3) Create a local build directory for this submodule
local_bin := $(BINDIR)/$(subdirectory)
bindirs   += $(local_bin)

# 4) Build a static library (the typical pattern for NeuralSPOT modules)
# The name is up to you. Example: ns-tflm.a
$(eval $(call make-library, $(local_bin)/ns-tflm.a, $(local_src)))
