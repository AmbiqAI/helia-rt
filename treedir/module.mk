TARGET_ARCH := cortex-m55

CORE_OPTIMIZATION_LEVEL     := -Os
KERNEL_OPTIMIZATION_LEVEL   := -O2
COMMON_FLAGS := \
  -Wall -Wextra -Wno-unused-parameter \
  -Wsign-compare -Wdouble-promotion -Wunused-variable -Wswitch -Wvla \
  -fno-unwind-tables -ffunction-sections -fdata-sections -fmessage-length=0 \
  -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON

CXXFLAGS += -std=c++17 -fno-rtti -fno-exceptions $(COMMON_FLAGS)
CFLAGS   += $(COMMON_FLAGS)

# Set optimized kernel folder name:
OPTIMIZED_KERNEL_DIR := cmsis_nn

ifneq ($(OPTIMIZED_KERNEL_DIR),)
	ADDITIONAL_DEFINES += -D$(shell echo $(OPTIMIZED_KERNEL_DIR) | tr [a-z] [A-Z])
endif

# Add ADDITIONAL_DEFINES to CFLAGS and CXXFLAGS
CFLAGS   += $(ADDITIONAL_DEFINES)
CXXFLAGS += $(ADDITIONAL_DEFINES)

# Include paths.
CFLAGS   += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/$(OPTIMIZED_KERNEL_DIR)
CXXFLAGS += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/$(OPTIMIZED_KERNEL_DIR)

CFLAGS   += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/flatbuffers/include
CXXFLAGS += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/flatbuffers/include

CFLAGS   += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/gemmlowp
CXXFLAGS += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/gemmlowp

CFLAGS   += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/kissfft
CXXFLAGS += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/kissfft

CFLAGS   += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/ruy
CXXFLAGS += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/ruy

CFLAGS   += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/cmsis/CMSIS/Core/Include
CXXFLAGS += -Ins-tflm/treedir -Ins-tflm/treedir/third_party/cmsis/CMSIS/Core/Include

# Paths to precompiled third-party libraries.
lib_prebuilt += ns-tflm/treedir/libtensorflow-microlite.a

# Dynamic collection of TFLM source files (skips 'third_party').
local_src := \
ns-tflm/treedir/tensorflow/lite/micro/cortex_m_generic/debug_log.cc \
ns-tflm/treedir/tensorflow/lite/micro/fake_micro_context.cc \
ns-tflm/treedir/tensorflow/lite/micro/flatbuffer_utils.cc \
ns-tflm/treedir/tensorflow/lite/micro/memory_helpers.cc \
ns-tflm/treedir/tensorflow/lite/micro/micro_allocation_info.cc \
ns-tflm/treedir/tensorflow/lite/micro/micro_allocator.cc \
ns-tflm/treedir/tensorflow/lite/micro/micro_context.cc \
ns-tflm/treedir/tensorflow/lite/micro/micro_interpreter.cc \
ns-tflm/treedir/tensorflow/lite/micro/micro_interpreter_context.cc \
ns-tflm/treedir/tensorflow/lite/micro/micro_interpreter_graph.cc \
ns-tflm/treedir/tensorflow/lite/micro/micro_log.cc \
ns-tflm/treedir/tensorflow/lite/micro/micro_op_resolver.cc \
ns-tflm/treedir/tensorflow/lite/micro/micro_profiler.cc \
ns-tflm/treedir/tensorflow/lite/micro/micro_resource_variable.cc \
ns-tflm/treedir/tensorflow/lite/micro/cortex_m_generic/micro_time.cc \
ns-tflm/treedir/tensorflow/lite/micro/micro_utils.cc \
ns-tflm/treedir/tensorflow/lite/micro/mock_micro_graph.cc \
ns-tflm/treedir/tensorflow/lite/micro/recording_micro_allocator.cc \
ns-tflm/treedir/tensorflow/lite/micro/system_setup.cc \
ns-tflm/treedir/tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc \
ns-tflm/treedir/tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cc \
ns-tflm/treedir/tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.cc \
ns-tflm/treedir/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc \
ns-tflm/treedir/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc \
ns-tflm/treedir/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc \
ns-tflm/treedir/tensorflow/lite/micro/memory_planner/non_persistent_buffer_planner_shim.cc \
ns-tflm/treedir/tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc \
ns-tflm/treedir/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc \
ns-tflm/treedir/tensorflow/lite/kernels/kernel_util.cc \
ns-tflm/treedir/tensorflow/lite/kernels/internal/tensor_utils.cc \
ns-tflm/treedir/tensorflow/lite/kernels/internal/common.cc \
ns-tflm/treedir/tensorflow/lite/kernels/internal/portable_tensor_utils.cc \
ns-tflm/treedir/tensorflow/lite/kernels/internal/tensor_ctypes.cc \
ns-tflm/treedir/tensorflow/lite/kernels/internal/runtime_shape.cc \
ns-tflm/treedir/tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc \
ns-tflm/treedir/tensorflow/lite/kernels/internal/reference/comparisons.cc \
ns-tflm/treedir/tensorflow/lite/kernels/internal/quantization_util.cc \
ns-tflm/treedir/tensorflow/lite/core/api/tensor_utils.cc \
ns-tflm/treedir/tensorflow/lite/core/api/flatbuffer_conversions.cc \
ns-tflm/treedir/tensorflow/lite/core/c/common.cc \
ns-tflm/treedir/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc \
ns-tflm/treedir/tensorflow/compiler/mlir/lite/schema/schema_utils.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/activations.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/activations_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/cmsis_nn/add.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/add_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/add_n.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/arg_min_max.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/assign_variable.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/cmsis_nn/batch_matmul.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/batch_to_space_nd.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/broadcast_args.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/broadcast_to.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/call_once.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/cast.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/ceil.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/circular_buffer.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/circular_buffer_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/comparisons.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/concatenation.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/cmsis_nn/conv.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/conv_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/cumsum.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/depth_to_space.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/cmsis_nn/depthwise_conv.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/depthwise_conv_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/dequantize.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/dequantize_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/detection_postprocess.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/div.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/elementwise.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/elu.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/embedding_lookup.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/ethosu.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/exp.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/expand_dims.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/fill.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/floor.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/floor_div.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/floor_mod.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/cmsis_nn/fully_connected.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/fully_connected_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/gather.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/gather_nd.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/hard_swish.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/hard_swish_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/if.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/kernel_runner.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/kernel_util.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/l2norm.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/l2_pool_2d.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/ambiq/leaky_relu.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/ambiq/leaky_relu_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/logical.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/logical_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/ambiq/logistic.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/ambiq/logistic_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/log_softmax.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/lstm_eval.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/lstm_eval_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/maximum_minimum.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/micro_tensor_utils.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/mirror_pad.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/cmsis_nn/mul.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/mul_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/neg.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/pack.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/ambiq/pad.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/cmsis_nn/pooling.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/pooling_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/prelu.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/prelu_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/quantize.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/quantize_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/read_variable.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/reduce.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/reduce_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/reshape.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/reshape_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/resize_bilinear.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/round.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/select.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/shape.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/slice.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/cmsis_nn/softmax.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/softmax_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/space_to_batch_nd.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/space_to_depth.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/split.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/split_v.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/squared_difference.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/squeeze.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/strided_slice.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/strided_slice_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/sub.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/sub_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/cmsis_nn/svdf.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/svdf_common.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/ambiq/tanh.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/transpose.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/cmsis_nn/transpose_conv.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/cmsis_nn/unidirectional_sequence_lstm.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/unpack.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/var_handle.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/while.cc \
ns-tflm/treedir/tensorflow/lite/micro/kernels/zeros_like.cc \
ns-tflm/treedir/signal/src/kiss_fft_wrappers/kiss_fft_float.cc \
ns-tflm/treedir/signal/src/kiss_fft_wrappers/kiss_fft_int16.cc \
ns-tflm/treedir/signal/src/kiss_fft_wrappers/kiss_fft_int32.cc \
ns-tflm/treedir/signal/micro/kernels/delay.cc \
ns-tflm/treedir/signal/micro/kernels/energy.cc \
ns-tflm/treedir/signal/micro/kernels/fft_auto_scale_kernel.cc \
ns-tflm/treedir/signal/micro/kernels/fft_auto_scale_common.cc \
ns-tflm/treedir/signal/micro/kernels/filter_bank.cc \
ns-tflm/treedir/signal/micro/kernels/filter_bank_log.cc \
ns-tflm/treedir/signal/micro/kernels/filter_bank_square_root.cc \
ns-tflm/treedir/signal/micro/kernels/filter_bank_square_root_common.cc \
ns-tflm/treedir/signal/micro/kernels/filter_bank_spectral_subtraction.cc \
ns-tflm/treedir/signal/micro/kernels/framer.cc \
ns-tflm/treedir/signal/micro/kernels/irfft.cc \
ns-tflm/treedir/signal/micro/kernels/rfft.cc \
ns-tflm/treedir/signal/micro/kernels/stacker.cc \
ns-tflm/treedir/signal/micro/kernels/overlap_add.cc \
ns-tflm/treedir/signal/micro/kernels/pcan.cc \
ns-tflm/treedir/signal/micro/kernels/window.cc \
ns-tflm/treedir/signal/src/circular_buffer.cc \
ns-tflm/treedir/signal/src/energy.cc \
ns-tflm/treedir/signal/src/fft_auto_scale.cc \
ns-tflm/treedir/signal/src/filter_bank.cc \
ns-tflm/treedir/signal/src/filter_bank_log.cc \
ns-tflm/treedir/signal/src/filter_bank_square_root.cc \
ns-tflm/treedir/signal/src/filter_bank_spectral_subtraction.cc \
ns-tflm/treedir/signal/src/irfft_float.cc \
ns-tflm/treedir/signal/src/irfft_int16.cc \
ns-tflm/treedir/signal/src/irfft_int32.cc \
ns-tflm/treedir/signal/src/log.cc \
ns-tflm/treedir/signal/src/max_abs.cc \
ns-tflm/treedir/signal/src/msb_32.cc \
ns-tflm/treedir/signal/src/msb_64.cc \
ns-tflm/treedir/signal/src/overlap_add.cc \
ns-tflm/treedir/signal/src/pcan_argc_fixed.cc \
ns-tflm/treedir/signal/src/rfft_float.cc \
ns-tflm/treedir/signal/src/rfft_int16.cc \
ns-tflm/treedir/signal/src/rfft_int32.cc \
ns-tflm/treedir/signal/src/square_root_32.cc \
ns-tflm/treedir/signal/src/square_root_64.cc \
ns-tflm/treedir/signal/src/window.cc \

# Include the Cortex M generic makefile
include ns-tflm/treedir/cortex_m_generic_makefile.inc

$(eval $(call make-library, $(local_bin)/ns-tflm.a, $(local_src)))
