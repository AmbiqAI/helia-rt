# Toolchain commands (for GCC on ARM)
TOOLCHAIN_PATH ?= ../tensorflow/lite/micro/tools/make/downloads/gcc_embedded

CC ?= $(TOOLCHAIN_PATH)/bin/arm-none-eabi-gcc
CXX ?= $(TOOLCHAIN_PATH)/bin/arm-none-eabi-g++
AR ?= $(TOOLCHAIN_PATH)/bin/arm-none-eabi-ar
OBJCOPY ?= arm-none-eabi-objcopy

TARGET_ARCH := cortex-m55

CORE_OPTIMIZATION_LEVEL     := -Os
KERNEL_OPTIMIZATION_LEVEL   := -O2
COMMON_FLAGS := \
  -Werror \
  -Wall -Wextra -Wno-unused-parameter \
  -Wsign-compare -Wdouble-promotion -Wunused-variable -Wswitch -Wvla \
  -fno-unwind-tables -ffunction-sections -fdata-sections -fmessage-length=0 \
  -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON

CXXFLAGS += -std=c++17 -fno-rtti -fno-exceptions $(COMMON_FLAGS)
CFLAGS   += -std=c17 $(COMMON_FLAGS)

# Remove -Werror from CFLAGS and CXXFLAGS
CFLAGS   += $(filter-out -Werror,$(CFLAGS))
CXXFLAGS += $(filter-out -Werror,$(CXXFLAGS))

# Set optimized kernel folder name:
OPTIMIZED_KERNEL_DIR := cmsis_nn

ifneq ($(OPTIMIZED_KERNEL_DIR),)
	ADDITIONAL_DEFINES += -D$(shell echo $(OPTIMIZED_KERNEL_DIR) | tr [a-z] [A-Z])
endif

# Add ADDITIONAL_DEFINES to CFLAGS and CXXFLAGS
CFLAGS   += $(ADDITIONAL_DEFINES)
CXXFLAGS += $(ADDITIONAL_DEFINES)

# 3) Include paths.
CFLAGS   += -I. -I./third_party/$(OPTIMIZED_KERNEL_DIR)
CXXFLAGS += -I. -I./third_party/$(OPTIMIZED_KERNEL_DIR)

CFLAGS   += -I. -I./third_party/flatbuffers/include
CXXFLAGS += -I. -I./third_party/flatbuffers/include

CFLAGS   += -I. -I./third_party/gemmlowp
CXXFLAGS += -I. -I./third_party/gemmlowp

CFLAGS   += -I. -I./third_party/kissfft
CXXFLAGS += -I. -I./third_party/kissfft

CFLAGS   += -I. -I./third_party/ruy
CXXFLAGS += -I. -I./third_party/ruy

# 4) Paths to precompiled third-party libraries (adjust as needed).
PRECOMPILED_LIB_DIR := ../neuralspot

THIRD_PARTY_STATIC_LIBS := \
  $(PRECOMPILED_LIB_DIR)/libtensorflow-microlite.a

# 5) Dynamic collection of TFLM source files (skips anything with 'third_party').
TFLM_SRC_FILES := \
  tensorflow/lite/micro/cortex_m_generic/debug_log.cc \
  tensorflow/lite/micro/fake_micro_context.cc \
  tensorflow/lite/micro/flatbuffer_utils.cc \
  tensorflow/lite/micro/memory_helpers.cc \
  tensorflow/lite/micro/micro_allocation_info.cc \
  tensorflow/lite/micro/micro_allocator.cc \
  tensorflow/lite/micro/micro_context.cc \
  tensorflow/lite/micro/micro_interpreter.cc \
  tensorflow/lite/micro/micro_interpreter_context.cc \
  tensorflow/lite/micro/micro_interpreter_graph.cc \
  tensorflow/lite/micro/micro_log.cc \
  tensorflow/lite/micro/micro_op_resolver.cc \
  tensorflow/lite/micro/micro_profiler.cc \
  tensorflow/lite/micro/micro_resource_variable.cc \
  tensorflow/lite/micro/cortex_m_generic/micro_time.cc \
  tensorflow/lite/micro/micro_utils.cc \
  tensorflow/lite/micro/mock_micro_graph.cc \
  tensorflow/lite/micro/recording_micro_allocator.cc \
  tensorflow/lite/micro/system_setup.cc \
  tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc \
  tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cc \
  tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.cc \
  tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc \
  tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc \
  tensorflow/lite/micro/memory_planner/linear_memory_planner.cc \
  tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.cc \
  tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc \
  tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc \
  tensorflow/lite/kernels/kernel_util.cc \
  tensorflow/lite/kernels/internal/tensor_utils.cc \
  tensorflow/lite/kernels/internal/common.cc \
  tensorflow/lite/kernels/internal/portable_tensor_utils.cc \
  tensorflow/lite/kernels/internal/tensor_ctypes.cc \
  tensorflow/lite/kernels/internal/runtime_shape.cc \
  tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc \
  tensorflow/lite/kernels/internal/reference/comparisons.cc \
  tensorflow/lite/kernels/internal/quantization_util.cc \
  tensorflow/lite/core/api/tensor_utils.cc \
  tensorflow/lite/core/api/flatbuffer_conversions.cc \
  tensorflow/lite/core/c/common.cc \
  tensorflow/compiler/mlir/lite/core/api/error_reporter.cc \
  tensorflow/compiler/mlir/lite/schema/schema_utils.cc \
  tensorflow/lite/micro/kernels/activations.cc \
  tensorflow/lite/micro/kernels/activations_common.cc \
  tensorflow/lite/micro/kernels/cmsis_nn/add.cc \
  tensorflow/lite/micro/kernels/add_common.cc \
  tensorflow/lite/micro/kernels/add_n.cc \
  tensorflow/lite/micro/kernels/arg_min_max.cc \
  tensorflow/lite/micro/kernels/assign_variable.cc \
  tensorflow/lite/micro/kernels/cmsis_nn/batch_matmul.cc \
  tensorflow/lite/micro/kernels/batch_to_space_nd.cc \
  tensorflow/lite/micro/kernels/broadcast_args.cc \
  tensorflow/lite/micro/kernels/broadcast_to.cc \
  tensorflow/lite/micro/kernels/call_once.cc \
  tensorflow/lite/micro/kernels/cast.cc \
  tensorflow/lite/micro/kernels/ceil.cc \
  tensorflow/lite/micro/kernels/circular_buffer.cc \
  tensorflow/lite/micro/kernels/circular_buffer_common.cc \
  tensorflow/lite/micro/kernels/comparisons.cc \
  tensorflow/lite/micro/kernels/concatenation.cc \
  tensorflow/lite/micro/kernels/cmsis_nn/conv.cc \
  tensorflow/lite/micro/kernels/conv_common.cc \
  tensorflow/lite/micro/kernels/cumsum.cc \
  tensorflow/lite/micro/kernels/depth_to_space.cc \
  tensorflow/lite/micro/kernels/cmsis_nn/depthwise_conv.cc \
  tensorflow/lite/micro/kernels/depthwise_conv_common.cc \
  tensorflow/lite/micro/kernels/dequantize.cc \
  tensorflow/lite/micro/kernels/dequantize_common.cc \
  tensorflow/lite/micro/kernels/detection_postprocess.cc \
  tensorflow/lite/micro/kernels/div.cc \
  tensorflow/lite/micro/kernels/elementwise.cc \
  tensorflow/lite/micro/kernels/elu.cc \
  tensorflow/lite/micro/kernels/embedding_lookup.cc \
  tensorflow/lite/micro/kernels/ethosu.cc \
  tensorflow/lite/micro/kernels/exp.cc \
  tensorflow/lite/micro/kernels/expand_dims.cc \
  tensorflow/lite/micro/kernels/fill.cc \
  tensorflow/lite/micro/kernels/floor.cc \
  tensorflow/lite/micro/kernels/floor_div.cc \
  tensorflow/lite/micro/kernels/floor_mod.cc \
  tensorflow/lite/micro/kernels/cmsis_nn/fully_connected.cc \
  tensorflow/lite/micro/kernels/fully_connected_common.cc \
  tensorflow/lite/micro/kernels/gather.cc \
  tensorflow/lite/micro/kernels/gather_nd.cc \
  tensorflow/lite/micro/kernels/hard_swish.cc \
  tensorflow/lite/micro/kernels/hard_swish_common.cc \
  tensorflow/lite/micro/kernels/if.cc \
  tensorflow/lite/micro/kernels/kernel_runner.cc \
  tensorflow/lite/micro/kernels/kernel_util.cc \
  tensorflow/lite/micro/kernels/l2norm.cc \
  tensorflow/lite/micro/kernels/l2_pool_2d.cc \
  tensorflow/lite/micro/kernels/ambiq/leaky_relu.cc \
  tensorflow/lite/micro/kernels/ambiq/leaky_relu_common.cc \
  tensorflow/lite/micro/kernels/logical.cc \
  tensorflow/lite/micro/kernels/logical_common.cc \
  tensorflow/lite/micro/kernels/ambiq/logistic.cc \
  tensorflow/lite/micro/kernels/ambiq/logistic_common.cc \
  tensorflow/lite/micro/kernels/log_softmax.cc \
  tensorflow/lite/micro/kernels/lstm_eval.cc \
  tensorflow/lite/micro/kernels/lstm_eval_common.cc \
  tensorflow/lite/micro/kernels/maximum_minimum.cc \
  tensorflow/lite/micro/kernels/micro_tensor_utils.cc \
  tensorflow/lite/micro/kernels/mirror_pad.cc \
  tensorflow/lite/micro/kernels/cmsis_nn/mul.cc \
  tensorflow/lite/micro/kernels/mul_common.cc \
  tensorflow/lite/micro/kernels/neg.cc \
  tensorflow/lite/micro/kernels/pack.cc \
  tensorflow/lite/micro/kernels/ambiq/pad.cc \
  tensorflow/lite/micro/kernels/cmsis_nn/pooling.cc \
  tensorflow/lite/micro/kernels/pooling_common.cc \
  tensorflow/lite/micro/kernels/prelu.cc \
  tensorflow/lite/micro/kernels/prelu_common.cc \
  tensorflow/lite/micro/kernels/quantize.cc \
  tensorflow/lite/micro/kernels/quantize_common.cc \
  tensorflow/lite/micro/kernels/read_variable.cc \
  tensorflow/lite/micro/kernels/reduce.cc \
  tensorflow/lite/micro/kernels/reduce_common.cc \
  tensorflow/lite/micro/kernels/reshape.cc \
  tensorflow/lite/micro/kernels/reshape_common.cc \
  tensorflow/lite/micro/kernels/resize_bilinear.cc \
  tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc \
  tensorflow/lite/micro/kernels/round.cc \
  tensorflow/lite/micro/kernels/select.cc \
  tensorflow/lite/micro/kernels/shape.cc \
  tensorflow/lite/micro/kernels/slice.cc \
  tensorflow/lite/micro/kernels/cmsis_nn/softmax.cc \
  tensorflow/lite/micro/kernels/softmax_common.cc \
  tensorflow/lite/micro/kernels/space_to_batch_nd.cc \
  tensorflow/lite/micro/kernels/space_to_depth.cc \
  tensorflow/lite/micro/kernels/split.cc \
  tensorflow/lite/micro/kernels/split_v.cc \
  tensorflow/lite/micro/kernels/squared_difference.cc \
  tensorflow/lite/micro/kernels/squeeze.cc \
  tensorflow/lite/micro/kernels/strided_slice.cc \
  tensorflow/lite/micro/kernels/strided_slice_common.cc \
  tensorflow/lite/micro/kernels/sub.cc \
  tensorflow/lite/micro/kernels/sub_common.cc \
  tensorflow/lite/micro/kernels/cmsis_nn/svdf.cc \
  tensorflow/lite/micro/kernels/svdf_common.cc \
  tensorflow/lite/micro/kernels/ambiq/tanh.cc \
  tensorflow/lite/micro/kernels/transpose.cc \
  tensorflow/lite/micro/kernels/cmsis_nn/transpose_conv.cc \
  tensorflow/lite/micro/kernels/cmsis_nn/unidirectional_sequence_lstm.cc \
  tensorflow/lite/micro/kernels/unpack.cc \
  tensorflow/lite/micro/kernels/var_handle.cc \
  tensorflow/lite/micro/kernels/while.cc \
  tensorflow/lite/micro/kernels/zeros_like.cc \
  signal/src/kiss_fft_wrappers/kiss_fft_float.cc \
  signal/src/kiss_fft_wrappers/kiss_fft_int16.cc \
  signal/src/kiss_fft_wrappers/kiss_fft_int32.cc \
  signal/micro/kernels/delay.cc \
  signal/micro/kernels/energy.cc \
  signal/micro/kernels/fft_auto_scale_kernel.cc \
  signal/micro/kernels/fft_auto_scale_common.cc \
  signal/micro/kernels/filter_bank.cc \
  signal/micro/kernels/filter_bank_log.cc \
  signal/micro/kernels/filter_bank_square_root.cc \
  signal/micro/kernels/filter_bank_square_root_common.cc \
  signal/micro/kernels/filter_bank_spectral_subtraction.cc \
  signal/micro/kernels/framer.cc \
  signal/micro/kernels/irfft.cc \
  signal/micro/kernels/rfft.cc \
  signal/micro/kernels/stacker.cc \
  signal/micro/kernels/overlap_add.cc \
  signal/micro/kernels/pcan.cc \
  signal/micro/kernels/window.cc \
  signal/src/circular_buffer.cc \
  signal/src/energy.cc \
  signal/src/fft_auto_scale.cc \
  signal/src/filter_bank.cc \
  signal/src/filter_bank_log.cc \
  signal/src/filter_bank_square_root.cc \
  signal/src/filter_bank_spectral_subtraction.cc \
  signal/src/irfft_float.cc \
  signal/src/irfft_int16.cc \
  signal/src/irfft_int32.cc \
  signal/src/log.cc \
  signal/src/max_abs.cc \
  signal/src/msb_32.cc \
  signal/src/msb_64.cc \
  signal/src/overlap_add.cc \
  signal/src/pcan_argc_fixed.cc \
  signal/src/rfft_float.cc \
  signal/src/rfft_int16.cc \
  signal/src/rfft_int32.cc \
  signal/src/square_root_32.cc \
  signal/src/square_root_64.cc \
  signal/src/window.cc \

# Include the Cortex M generic makefile
include cortex_m_generic_makefile.inc

