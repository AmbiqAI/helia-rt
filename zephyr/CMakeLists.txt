if(CONFIG_HELIA_RT)

  set(HELIA_RT_DIR ${ZEPHYR_CURRENT_MODULE_DIR})
  set(HELIA_RT_KERNEL_ROOT ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels)

  zephyr_library_named(helia_rt)

  zephyr_include_directories(
    ${HELIA_RT_DIR}/.
    ${HELIA_RT_DIR}/include
    ${HELIA_RT_DIR}/third_party_static/gemmlowp
    ${HELIA_RT_DIR}/third_party_static/flatbuffers/include
    ${HELIA_RT_DIR}/third_party_static/ruy
    ${HELIA_RT_DIR}/third_party_static/kissfft
  )

  # Select reference, open CMSIS-NN, or HELIA kernel sources.
  set(HELIA_RT_OPTIMIZED_KERNEL_DIR "")
  set(HELIA_RT_EXTERNAL_KERNEL_MODULE_DIR "")
  set(HELIA_RT_BACKEND_EXTRA_SOURCES "")

  if(CONFIG_HELIA_RT_BACKEND_CMSIS_NN)
    if(DEFINED ZEPHYR_CMSIS_NN_MODULE_DIR AND EXISTS "${ZEPHYR_CMSIS_NN_MODULE_DIR}")
      set(HELIA_RT_EXTERNAL_KERNEL_MODULE_DIR "${ZEPHYR_CMSIS_NN_MODULE_DIR}")
    else()
      message(FATAL_ERROR
        "CONFIG_HELIA_RT_BACKEND_CMSIS_NN requires an open CMSIS-NN Zephyr module. "
        "Add the CMSIS-NN module to your workspace or select the REFERENCE backend.")
    endif()

    set(HELIA_RT_OPTIMIZED_KERNEL_DIR cmsis_nn)
    zephyr_library_compile_definitions(CMSIS_NN)
  elseif(CONFIG_HELIA_RT_BACKEND_HELIA)
    if(DEFINED ZEPHYR_NS_CMSIS_NN_MODULE_DIR AND EXISTS "${ZEPHYR_NS_CMSIS_NN_MODULE_DIR}")
      set(HELIA_RT_EXTERNAL_KERNEL_MODULE_DIR "${ZEPHYR_NS_CMSIS_NN_MODULE_DIR}")
    else()
      message(FATAL_ERROR
        "CONFIG_HELIA_RT_BACKEND_HELIA requires Ambiq's HELIA acceleration module, "
        "currently distributed as ns-cmsis-nn. Add that module to your workspace "
        "or select the REFERENCE or CMSIS-NN backend.")
    endif()

    set(HELIA_RT_OPTIMIZED_KERNEL_DIR helia)
    zephyr_library_compile_definitions(NS_CMSIS_NN)
    zephyr_library_compile_definitions(HELIA)
  endif()

  if(HELIA_RT_EXTERNAL_KERNEL_MODULE_DIR)
    # Fixes imports that start with Include/ and pulls in CMSIS-Core headers.
    zephyr_library_include_directories(${HELIA_RT_EXTERNAL_KERNEL_MODULE_DIR})
    if(EXISTS "${HELIA_RT_EXTERNAL_KERNEL_MODULE_DIR}/Include")
      zephyr_library_include_directories(${HELIA_RT_EXTERNAL_KERNEL_MODULE_DIR}/Include)
    endif()
    if(EXISTS "${HELIA_RT_EXTERNAL_KERNEL_MODULE_DIR}/CMSIS/Core/Include")
      zephyr_library_include_directories(${HELIA_RT_EXTERNAL_KERNEL_MODULE_DIR}/CMSIS/Core/Include)
    endif()
  endif()

  function(helia_rt_select_kernel_source out_var relative_path)
    if(HELIA_RT_OPTIMIZED_KERNEL_DIR)
      set(candidate "${HELIA_RT_KERNEL_ROOT}/${HELIA_RT_OPTIMIZED_KERNEL_DIR}/${relative_path}")
      if(EXISTS "${candidate}")
        set(${out_var} "${candidate}" PARENT_SCOPE)
        return()
      endif()
    endif()

    set(${out_var} "${HELIA_RT_KERNEL_ROOT}/${relative_path}" PARENT_SCOPE)
  endfunction()

  helia_rt_select_kernel_source(HELIA_RT_KERNEL_ACTIVATIONS activations.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_ADD add.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_BATCH_MATMUL batch_matmul.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_COMPARISONS comparisons.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_CONCATENATION concatenation.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_CONV conv.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_DEPTHWISE_CONV depthwise_conv.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_DEQUANTIZE dequantize.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_FILL fill.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_FULLY_CONNECTED fully_connected.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_HARD_SWISH hard_swish.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_LEAKY_RELU leaky_relu.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_LEAKY_RELU_COMMON leaky_relu_common.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_LOGISTIC logistic.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_MAXIMUM_MINIMUM maximum_minimum.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_MUL mul.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_PACK pack.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_PAD pad.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_POOLING pooling.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_QUANTIZE_COMMON quantize_common.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_REDUCE reduce.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_REDUCE_COMMON reduce_common.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_RESHAPE reshape.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_SOFTMAX softmax.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_SPLIT split.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_SPLIT_V split_v.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_SQUEEZE squeeze.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_STRIDED_SLICE strided_slice.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_SUB sub.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_SVDF svdf.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_TANH tanh.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_TRANSPOSE transpose.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_TRANSPOSE_CONV transpose_conv.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_UNIDIRECTIONAL_SEQUENCE_LSTM unidirectional_sequence_lstm.cc)
  helia_rt_select_kernel_source(HELIA_RT_KERNEL_ZEROS_LIKE zeros_like.cc)

  if (CONFIG_ETHOS_U)
    set(ETHOSU_CO_PROCESSOR ethos_u)
  endif()

  zephyr_library_sources(
    ${HELIA_RT_DIR}/python/tflite_micro/python_ops_resolver.cc
    ${HELIA_RT_DIR}/signal/micro/kernels/fft_flexbuffers_generated_data.cc
    ${HELIA_RT_DIR}/signal/micro/kernels/rfft.cc
    ${HELIA_RT_DIR}/signal/micro/kernels/window.cc
    ${HELIA_RT_DIR}/signal/micro/kernels/window_flexbuffers_generated_data.cc
    ${HELIA_RT_DIR}/signal/src/rfft_float.cc
    ${HELIA_RT_DIR}/signal/src/rfft_int16.cc
    ${HELIA_RT_DIR}/signal/src/rfft_int32.cc
    ${HELIA_RT_DIR}/signal/src/window.cc
    ${HELIA_RT_DIR}/signal/src/kiss_fft_wrappers/kiss_fft_float.cc
    ${HELIA_RT_DIR}/signal/src/kiss_fft_wrappers/kiss_fft_int16.cc
    ${HELIA_RT_DIR}/signal/src/kiss_fft_wrappers/kiss_fft_int32.cc
    ${HELIA_RT_DIR}/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc
    ${HELIA_RT_DIR}/tensorflow/compiler/mlir/lite/schema/schema_utils.cc
    ${HELIA_RT_DIR}/tensorflow/lite/array.cc
    ${HELIA_RT_DIR}/tensorflow/lite/core/c/common.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/debug_log.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/hexdump.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/fake_micro_context.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/memory_helpers.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/micro_allocation_info.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/test_helpers.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/test_helper_custom_ops.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/recording_micro_allocator.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/micro_time.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/micro_profiler.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/micro_utils.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/flatbuffer_utils.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/mock_micro_graph.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/micro_interpreter.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/micro_interpreter_context.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/micro_interpreter_graph.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/micro_allocator.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/micro_context.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/micro_log.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/micro_op_resolver.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/micro_resource_variable.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/system_setup.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc
    ${HELIA_RT_DIR}/tensorflow/lite/kernels/internal/common.cc
    ${HELIA_RT_DIR}/tensorflow/lite/kernels/internal/quantization_util.cc
    ${HELIA_RT_DIR}/tensorflow/lite/kernels/internal/portable_tensor_utils.cc
    ${HELIA_RT_DIR}/tensorflow/lite/kernels/internal/tensor_ctypes.cc
    ${HELIA_RT_DIR}/tensorflow/lite/kernels/internal/tensor_utils.cc
    ${HELIA_RT_DIR}/tensorflow/lite/kernels/internal/reference/comparisons.cc
    ${HELIA_RT_DIR}/tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc
    ${HELIA_RT_DIR}/tensorflow/lite/kernels/kernel_util.cc
    ${HELIA_RT_DIR}/tensorflow/lite/core/api/flatbuffer_conversions.cc
    ${HELIA_RT_DIR}/tensorflow/lite/core/api/tensor_utils.cc
    ${HELIA_RT_KERNEL_ACTIVATIONS}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/activations_common.cc
    ${HELIA_RT_KERNEL_ADD}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/add_common.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/add_n.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/arg_min_max.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/assign_variable.cc
    ${HELIA_RT_KERNEL_BATCH_MATMUL}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/batch_matmul_common.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/batch_to_space_nd.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/broadcast_args.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/broadcast_to.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/call_once.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/cast.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/ceil.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/circular_buffer.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/circular_buffer_common.cc
    ${HELIA_RT_KERNEL_COMPARISONS}
    ${HELIA_RT_KERNEL_CONCATENATION}
    ${HELIA_RT_KERNEL_CONV}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/conv_common.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/cumsum.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/depth_to_space.cc
    ${HELIA_RT_KERNEL_DEPTHWISE_CONV}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/depthwise_conv_common.cc
    ${HELIA_RT_KERNEL_DEQUANTIZE}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/dequantize_common.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/detection_postprocess.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/div.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/elementwise.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/elu.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/embedding_lookup.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/${ETHOSU_CO_PROCESSOR}/ethosu.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/exp.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/expand_dims.cc
    ${HELIA_RT_KERNEL_FILL}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/floor.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/floor_div.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/floor_mod.cc
    ${HELIA_RT_KERNEL_FULLY_CONNECTED}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/fully_connected_common.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/gather.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/gather_nd.cc
    ${HELIA_RT_KERNEL_HARD_SWISH}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/hard_swish_common.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/if.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/kernel_runner.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/kernel_util.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/l2_pool_2d.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/l2norm.cc
    ${HELIA_RT_KERNEL_LEAKY_RELU}
    ${HELIA_RT_KERNEL_LEAKY_RELU_COMMON}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/log_softmax.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/logical.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/logical_common.cc
    ${HELIA_RT_KERNEL_LOGISTIC}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/logistic_common.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/lstm_eval.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/lstm_eval_common.cc
    ${HELIA_RT_KERNEL_MAXIMUM_MINIMUM}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/micro_tensor_utils.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/mirror_pad.cc
    ${HELIA_RT_KERNEL_MUL}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/mul_common.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/neg.cc
    ${HELIA_RT_KERNEL_PACK}
    ${HELIA_RT_KERNEL_PAD}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/pad_common.cc
    ${HELIA_RT_KERNEL_POOLING}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/pooling_common.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/prelu.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/prelu_common.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/quantize.cc
    ${HELIA_RT_KERNEL_QUANTIZE_COMMON}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/read_variable.cc
    ${HELIA_RT_KERNEL_REDUCE}
    ${HELIA_RT_KERNEL_REDUCE_COMMON}
    ${HELIA_RT_KERNEL_RESHAPE}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/reshape_common.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/resize_bilinear.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/resize_nearest_neighbor.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/reverse.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/round.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/select.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/shape.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/slice.cc
    ${HELIA_RT_KERNEL_SOFTMAX}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/softmax_common.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/space_to_batch_nd.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/space_to_depth.cc
    ${HELIA_RT_KERNEL_SPLIT}
    ${HELIA_RT_KERNEL_SPLIT_V}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/squared_difference.cc
    ${HELIA_RT_KERNEL_SQUEEZE}
    ${HELIA_RT_KERNEL_STRIDED_SLICE}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/strided_slice_common.cc
    ${HELIA_RT_KERNEL_SUB}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/sub_common.cc
    ${HELIA_RT_KERNEL_SVDF}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/svdf_common.cc
    ${HELIA_RT_KERNEL_TANH}
    ${HELIA_RT_KERNEL_TRANSPOSE}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/transpose_common.cc
    ${HELIA_RT_KERNEL_TRANSPOSE_CONV}
    ${HELIA_RT_KERNEL_UNIDIRECTIONAL_SEQUENCE_LSTM}
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/unpack.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/var_handle.cc
    ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/while.cc
    ${HELIA_RT_KERNEL_ZEROS_LIKE}
    ${HELIA_RT_BACKEND_EXTRA_SOURCES}
  )

endif()
