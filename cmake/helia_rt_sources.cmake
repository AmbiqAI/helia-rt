# ---------------------------------------------------------------------------
# cmake/helia_rt_sources.cmake — canonical source manifest for heliaRT.
#
# Single source of truth for what files compose the heliaRT TFLM runtime,
# the per-kernel reference / CMSIS-NN / HELIA selection rule, and the
# default compile-options layering.
#
# Consumed by:
#   * Top-level CMakeLists.txt        — generic CMake / FetchContent users
#   * zephyr/CMakeLists.txt           — Zephyr module (Phase 4 migration)
#   * nsx/CMakeLists.txt (src mode)   — NSX source-mode consumer (Phase 2)
#   * cmake/dump_manifest.cmake       — JSON export for the CMSIS-Pack builder
#
# Hard constraint (see issue #147): every path referenced from here lives
# inside the heliaRT repo. We never patch upstream files under tensorflow/,
# so ci/sync_from_upstream_tf.sh stays a no-op for these lists.
#
# Adding a new kernel:
#   1. Drop the .cc into tensorflow/lite/micro/kernels/ (and any backend
#      variant siblings under cmsis_nn/ or helia/).
#   2. Add the basename to HELIA_RT_KERNEL_BASENAMES below.
#   3. Done. The selector function picks the right variant per backend.
#
# Adding a new third-party header dir:
#   1. Run ./zephyr_static_export.sh to refresh third_party_static/.
#   2. Append the new path to HELIA_RT_INCLUDE_DIRS below.
# ---------------------------------------------------------------------------

# Guard against double-inclusion. Some consumers (e.g. zephyr modules) end up
# including this file twice through different paths; the variable definitions
# below are idempotent but we still skip the second pass for clarity.
if(DEFINED _HELIA_RT_SOURCES_INCLUDED)
    return()
endif()
set(_HELIA_RT_SOURCES_INCLUDED TRUE)

# The repo root, derived from this file's location.
get_filename_component(HELIA_RT_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(HELIA_RT_KERNEL_ROOT "${HELIA_RT_ROOT}/tensorflow/lite/micro/kernels")

# Pack version — sourced from the release-please-managed public header.
set(_HELIA_RT_VERSION_HEADER
    "${HELIA_RT_ROOT}/tensorflow/lite/micro/helia_rt_version.h")
file(READ "${_HELIA_RT_VERSION_HEADER}" _HELIA_RT_VERSION_HEADER_CONTENTS)
if(NOT _HELIA_RT_VERSION_HEADER_CONTENTS MATCHES
        "#[ \t]*define[ \t]+HELIA_RT_VERSION[ \t]+\"v?([^\"]+)\"")
    message(FATAL_ERROR
        "Could not parse HELIA_RT_VERSION from ${_HELIA_RT_VERSION_HEADER}")
endif()
set(HELIA_RT_VERSION "${CMAKE_MATCH_1}")

# ---------------------------------------------------------------------------
# Include directories. PUBLIC: propagate to consumers.
# ---------------------------------------------------------------------------
set(HELIA_RT_INCLUDE_DIRS
    "${HELIA_RT_ROOT}"
    "${HELIA_RT_ROOT}/third_party_static/gemmlowp"
    "${HELIA_RT_ROOT}/third_party_static/flatbuffers/include"
    "${HELIA_RT_ROOT}/third_party_static/ruy"
    "${HELIA_RT_ROOT}/third_party_static/kissfft"
)

# ---------------------------------------------------------------------------
# Common (non-kernel) sources, partitioned into profiles. Backend-independent.
#
# The source set is split into three buckets so production consumers can opt
# out of the recording/introspection and test/mock helpers that are not needed
# to run inference:
#
#   HELIA_RT_RUNTIME_COMMON_SOURCES  always-on production runtime
#   HELIA_RT_RECORDING_SOURCES       arena recording / introspection (opt-in)
#   HELIA_RT_TEST_SUPPORT_SOURCES    test / mock / fake helpers (opt-in)
#
# HELIA_RT_COMMON_SOURCES is preserved below as the union of all three, so
# existing consumers (zephyr/CMakeLists.txt, nsx/CMakeLists.txt,
# cmake/dump_manifest.cmake) keep their current behavior unchanged.
# ---------------------------------------------------------------------------
set(HELIA_RT_RUNTIME_COMMON_SOURCES
    # python_ops_resolver — full reference op set registration
    python/tflite_micro/python_ops_resolver.cc

    # signal/
    signal/micro/kernels/fft_flexbuffers_generated_data.cc
    signal/micro/kernels/rfft.cc
    signal/micro/kernels/window.cc
    signal/micro/kernels/window_flexbuffers_generated_data.cc
    signal/src/rfft_float.cc
    signal/src/rfft_int16.cc
    signal/src/rfft_int32.cc
    signal/src/window.cc
    signal/src/kiss_fft_wrappers/kiss_fft_float.cc
    signal/src/kiss_fft_wrappers/kiss_fft_int16.cc
    signal/src/kiss_fft_wrappers/kiss_fft_int32.cc

    # tensorflow/compiler/mlir
    tensorflow/compiler/mlir/lite/core/api/error_reporter.cc
    tensorflow/compiler/mlir/lite/schema/schema_utils.cc

    # tensorflow/lite (core)
    tensorflow/lite/array.cc
    tensorflow/lite/core/c/common.cc
    tensorflow/lite/core/api/flatbuffer_conversions.cc
    tensorflow/lite/core/api/tensor_utils.cc

    # tensorflow/lite/micro (runtime)
    tensorflow/lite/micro/debug_log.cc
    tensorflow/lite/micro/hexdump.cc
    tensorflow/lite/micro/memory_helpers.cc
    tensorflow/lite/micro/micro_allocation_info.cc
    tensorflow/lite/micro/micro_time.cc
    tensorflow/lite/micro/micro_profiler.cc
    tensorflow/lite/micro/micro_utils.cc
    tensorflow/lite/micro/flatbuffer_utils.cc
    tensorflow/lite/micro/micro_interpreter.cc
    tensorflow/lite/micro/micro_interpreter_context.cc
    tensorflow/lite/micro/micro_interpreter_graph.cc
    tensorflow/lite/micro/micro_allocator.cc
    tensorflow/lite/micro/micro_context.cc
    tensorflow/lite/micro/micro_log.cc
    tensorflow/lite/micro/micro_op_resolver.cc
    tensorflow/lite/micro/micro_resource_variable.cc
    tensorflow/lite/micro/system_setup.cc

    # arena allocator
    tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc
    tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cc
    tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc

    # tflite_bridge
    tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc
    tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc

    # memory planners
    tensorflow/lite/micro/memory_planner/linear_memory_planner.cc
    tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc

    # tensorflow/lite/kernels/internal
    tensorflow/lite/kernels/internal/common.cc
    tensorflow/lite/kernels/internal/quantization_util.cc
    tensorflow/lite/kernels/internal/portable_tensor_utils.cc
    tensorflow/lite/kernels/internal/tensor_ctypes.cc
    tensorflow/lite/kernels/internal/tensor_utils.cc
    tensorflow/lite/kernels/internal/reference/comparisons.cc
    tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc
    tensorflow/lite/kernels/kernel_util.cc
)

# Arena recording / introspection. Only needed for RecordingMicroInterpreter /
# RecordingMicroAllocator (memory-usage auditing). Not on the inference path.
set(HELIA_RT_RECORDING_SOURCES
    tensorflow/lite/micro/recording_micro_allocator.cc
    tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.cc
)

# Test / mock / fake helpers. Used by unit tests and KernelRunner-based test
# harnesses, not by production inference.
set(HELIA_RT_TEST_SUPPORT_SOURCES
    tensorflow/lite/micro/test_helpers.cc
    tensorflow/lite/micro/test_helper_custom_ops.cc
    tensorflow/lite/micro/mock_micro_graph.cc
    tensorflow/lite/micro/fake_micro_context.cc
)

# Backward-compatible aggregate: the full common source set (runtime +
# recording + test support). Existing consumers that reference
# HELIA_RT_COMMON_SOURCES keep their current behavior.
set(HELIA_RT_COMMON_SOURCES
    ${HELIA_RT_RUNTIME_COMMON_SOURCES}
    ${HELIA_RT_RECORDING_SOURCES}
    ${HELIA_RT_TEST_SUPPORT_SOURCES}
)

# ---------------------------------------------------------------------------
# Kernel basenames. Each name is resolved by helia_rt_select_kernel_sources()
# to one of:
#   ${HELIA_RT_KERNEL_ROOT}/<backend>/<name>    if it exists for the backend
#   ${HELIA_RT_KERNEL_ROOT}/<name>              otherwise (the reference impl)
#
# This curated list mirrors the existing zephyr/CMakeLists.txt allowlist.
# Kernels not in this list (decode/decompress/dynamic_update_slice/ethosu/...)
# are deliberately omitted today; add a line below to opt them in.
# ---------------------------------------------------------------------------
set(HELIA_RT_KERNEL_BASENAMES
    activations.cc
    activations_common.cc
    add.cc
    add_common.cc
    add_n.cc
    arg_min_max.cc
    assign_variable.cc
    batch_matmul.cc
    batch_matmul_common.cc
    batch_to_space_nd.cc
    broadcast_args.cc
    broadcast_to.cc
    call_once.cc
    cast.cc
    ceil.cc
    circular_buffer.cc
    circular_buffer_common.cc
    comparisons.cc
    concatenation.cc
    conv.cc
    conv_common.cc
    cumsum.cc
    depth_to_space.cc
    depthwise_conv.cc
    depthwise_conv_common.cc
    dequantize.cc
    dequantize_common.cc
    detection_postprocess.cc
    div.cc
    elementwise.cc
    elu.cc
    embedding_lookup.cc
    exp.cc
    expand_dims.cc
    fill.cc
    floor.cc
    floor_div.cc
    floor_mod.cc
    fully_connected.cc
    fully_connected_common.cc
    gather.cc
    gather_nd.cc
    hard_swish.cc
    hard_swish_common.cc
    if.cc
    kernel_runner.cc
    kernel_util.cc
    l2_pool_2d.cc
    l2norm.cc
    leaky_relu.cc
    leaky_relu_common.cc
    log_softmax.cc
    logical.cc
    logical_common.cc
    logistic.cc
    logistic_common.cc
    lstm_eval.cc
    lstm_eval_common.cc
    maximum_minimum.cc
    micro_tensor_utils.cc
    mirror_pad.cc
    mul.cc
    mul_common.cc
    neg.cc
    pack.cc
    pad.cc
    pad_common.cc
    pooling.cc
    pooling_common.cc
    prelu.cc
    prelu_common.cc
    quantize.cc
    quantize_common.cc
    read_variable.cc
    reduce.cc
    reduce_common.cc
    reshape.cc
    reshape_common.cc
    resize_bilinear.cc
    resize_nearest_neighbor.cc
    reverse.cc
    round.cc
    select.cc
    shape.cc
    slice.cc
    softmax.cc
    softmax_common.cc
    space_to_batch_nd.cc
    space_to_depth.cc
    split.cc
    split_v.cc
    squared_difference.cc
    squeeze.cc
    strided_slice.cc
    strided_slice_common.cc
    sub.cc
    sub_common.cc
    svdf.cc
    svdf_common.cc
    tanh.cc
    transpose.cc
    transpose_common.cc
    transpose_conv.cc
    unidirectional_sequence_lstm.cc
    unpack.cc
    var_handle.cc
    while.cc
    zeros_like.cc
)

# Supported backend names. Order matters for diagnostics only.
set(HELIA_RT_BACKENDS reference cmsis_nn helia)

# ---------------------------------------------------------------------------
# Per-kernel optimization knobs for the `helia` backend.
#
# Mirrors the Makefile contract documented in
# tensorflow/lite/micro/kernels/helia/README.md and implemented in
# tools/make/Makefile + ext_libs/helia.inc:
#
#   HELIA_RT_GLOBAL_KERNEL_OPTIMIZE   project-wide default (SPEED|SIZE)
#   HELIA_RT_CONV_OPT                 override for conv / depthwise_conv /
#                                     transpose_conv kernels (empty = global)
#   HELIA_RT_FC_OPT                   override for fully_connected (empty =
#                                     global)
#
# These drive the CONV_KERNEL_OPTIMIZED_FOR_*, FC_KERNEL_OPTIMIZED_FOR_*,
# and KERNELS_OPTIMIZED_FOR_* compile defines emitted by
# helia_rt_backend_compile_definitions() below. Without them the precomputed
# weight-sum fast paths in kernels/helia/{fully_connected,svdf}.cc stay
# disabled and the runtime regresses by 80–150%+ on MLPerf Tiny vs the
# Makefile/prebuilt path. svdf.cc additionally `#error`s out unless one of
# KERNELS_OPTIMIZED_FOR_{SPEED,SIZE} is defined, so SPEED is the safe default.
# ---------------------------------------------------------------------------
set(HELIA_RT_GLOBAL_KERNEL_OPTIMIZE "SPEED" CACHE STRING
    "Default per-kernel optimization target for the helia backend (SPEED|SIZE).")
set_property(CACHE HELIA_RT_GLOBAL_KERNEL_OPTIMIZE PROPERTY STRINGS SPEED SIZE)

set(HELIA_RT_CONV_OPT "" CACHE STRING
    "Override conv-family kernel optimization (SPEED|SIZE); empty inherits HELIA_RT_GLOBAL_KERNEL_OPTIMIZE.")
set_property(CACHE HELIA_RT_CONV_OPT PROPERTY STRINGS "" SPEED SIZE)

set(HELIA_RT_FC_OPT "" CACHE STRING
    "Override fully-connected kernel optimization (SPEED|SIZE); empty inherits HELIA_RT_GLOBAL_KERNEL_OPTIMIZE.")
set_property(CACHE HELIA_RT_FC_OPT PROPERTY STRINGS "" SPEED SIZE)

# ---------------------------------------------------------------------------
# helia_rt_select_kernel_sources(OUT_VAR BACKEND <name>)
#
# Resolves HELIA_RT_KERNEL_BASENAMES into absolute paths under the requested
# backend, falling back to the top-level reference implementation when a
# backend-specific variant does not exist on disk. The EXISTS probe is the
# selector — no manual maintenance is required when a new variant lands.
#
# Arguments:
#   OUT_VAR         (positional)  Name of the parent-scope variable to fill.
#   BACKEND <name>  (kwarg)       One of HELIA_RT_BACKENDS.
# ---------------------------------------------------------------------------
function(helia_rt_select_kernel_sources OUT_VAR)
    cmake_parse_arguments(_ARG "" "BACKEND" "" ${ARGN})

    if(NOT _ARG_BACKEND)
        message(FATAL_ERROR "helia_rt_select_kernel_sources: BACKEND <name> is required")
    endif()

    list(FIND HELIA_RT_BACKENDS "${_ARG_BACKEND}" _idx)
    if(_idx EQUAL -1)
        message(FATAL_ERROR
            "helia_rt_select_kernel_sources: unknown BACKEND '${_ARG_BACKEND}'. "
            "Supported: ${HELIA_RT_BACKENDS}")
    endif()

    # Reference backend: never probes a subdirectory.
    set(_variant_dir "")
    if(NOT _ARG_BACKEND STREQUAL "reference")
        set(_variant_dir "${HELIA_RT_KERNEL_ROOT}/${_ARG_BACKEND}")
    endif()

    set(_out)
    foreach(_base IN LISTS HELIA_RT_KERNEL_BASENAMES)
        set(_candidate "")
        if(_variant_dir)
            set(_candidate "${_variant_dir}/${_base}")
        endif()
        if(_candidate AND EXISTS "${_candidate}")
            list(APPEND _out "${_candidate}")
        else()
            list(APPEND _out "${HELIA_RT_KERNEL_ROOT}/${_base}")
        endif()
    endforeach()

    set(${OUT_VAR} ${_out} PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# helia_rt_backend_compile_definitions(OUT_VAR BACKEND <name>)
#
# Returns the compile definitions a backend variant of heliaRT requires.
# Matches the historical Zephyr behavior:
#   reference  → (none)
#   cmsis_nn   → CMSIS_NN
#   helia      → CMSIS_NN, HELIA
#                plus per-kernel optimization defines driven by
#                HELIA_RT_{GLOBAL_KERNEL_OPTIMIZE,CONV_OPT,FC_OPT}:
#                    CONV_KERNEL_OPTIMIZED_FOR_<SPEED|SIZE>
#                    FC_KERNEL_OPTIMIZED_FOR_<SPEED|SIZE>
#                    KERNELS_OPTIMIZED_FOR_<SPEED|SIZE>
# ---------------------------------------------------------------------------
function(helia_rt_backend_compile_definitions OUT_VAR)
    cmake_parse_arguments(_ARG "" "BACKEND" "" ${ARGN})
    if(NOT _ARG_BACKEND)
        message(FATAL_ERROR "helia_rt_backend_compile_definitions: BACKEND <name> is required")
    endif()

    set(_defs "")
    if(_ARG_BACKEND STREQUAL "cmsis_nn")
        set(_defs CMSIS_NN)
    elseif(_ARG_BACKEND STREQUAL "helia")
        set(_defs CMSIS_NN HELIA)

        # Resolve per-kernel overrides, falling back to the global default.
        set(_global "${HELIA_RT_GLOBAL_KERNEL_OPTIMIZE}")
        set(_conv "${HELIA_RT_CONV_OPT}")
        if(_conv STREQUAL "")
            set(_conv "${_global}")
        endif()
        set(_fc "${HELIA_RT_FC_OPT}")
        if(_fc STREQUAL "")
            set(_fc "${_global}")
        endif()

        # Validate. SPEED|SIZE only; anything else is a fast-fail.
        if(NOT _global MATCHES "^(SPEED|SIZE)$")
            message(FATAL_ERROR
                "helia_rt_backend_compile_definitions: "
                "HELIA_RT_GLOBAL_KERNEL_OPTIMIZE='${_global}' "
                "is invalid — must be SPEED or SIZE.")
        endif()
        if(NOT _conv MATCHES "^(SPEED|SIZE)$")
            message(FATAL_ERROR
                "helia_rt_backend_compile_definitions: "
                "HELIA_RT_CONV_OPT='${_conv}' "
                "is invalid — must be SPEED or SIZE.")
        endif()
        if(NOT _fc MATCHES "^(SPEED|SIZE)$")
            message(FATAL_ERROR
                "helia_rt_backend_compile_definitions: "
                "HELIA_RT_FC_OPT='${_fc}' "
                "is invalid — must be SPEED or SIZE.")
        endif()

        list(APPEND _defs
            CONV_KERNEL_OPTIMIZED_FOR_${_conv}
            FC_KERNEL_OPTIMIZED_FOR_${_fc}
            KERNELS_OPTIMIZED_FOR_${_global})
    endif()
    set(${OUT_VAR} ${_defs} PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# helia_rt_build_type_compile_definitions(OUT_VAR BUILD_TYPE <type>)
#
# Returns the compile definitions implied by a HELIA_RT_BUILD_TYPE value.
# Mirrors the upstream TFLM Makefile's BUILD_TYPE knob.
#   debug              → (none — keep asserts, do not strip strings)
#   release_with_logs  → NDEBUG
#   release            → NDEBUG, TF_LITE_STRIP_ERROR_STRINGS
# ---------------------------------------------------------------------------
function(helia_rt_build_type_compile_definitions OUT_VAR)
    cmake_parse_arguments(_ARG "" "BUILD_TYPE" "" ${ARGN})
    if(NOT _ARG_BUILD_TYPE)
        message(FATAL_ERROR "helia_rt_build_type_compile_definitions: BUILD_TYPE <type> is required")
    endif()

    set(_defs "")
    if(_ARG_BUILD_TYPE STREQUAL "release_with_logs")
        list(APPEND _defs NDEBUG)
    elseif(_ARG_BUILD_TYPE STREQUAL "release")
        list(APPEND _defs NDEBUG TF_LITE_STRIP_ERROR_STRINGS)
    elseif(NOT _ARG_BUILD_TYPE STREQUAL "debug")
        message(FATAL_ERROR
            "helia_rt_build_type_compile_definitions: unknown BUILD_TYPE "
            "'${_ARG_BUILD_TYPE}'. Supported: debug | release_with_logs | release")
    endif()
    set(${OUT_VAR} ${_defs} PARENT_SCOPE)
endfunction()
