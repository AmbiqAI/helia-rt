# ---------------------------------------------------------------------------
# nsx-heliart — NSX module for heliaRT prebuilt static libraries
# ---------------------------------------------------------------------------
# This module wraps heliaRT prebuilt libraries as an NSX module so that any
# NSX application can link nsx::heliart and get the full TFLM runtime with
# Ambiq-optimized (ns-cmsis-nn) kernels.
#
# The prebuilt .a files are expected under lib/ relative to this file:
#   lib/libhelia-rt-<core>-<toolchain>-<variant>.a
#
# CMake options:
#   HELIART_VARIANT   — debug | release | release-with-logs (default)
# ---------------------------------------------------------------------------

if(NOT DEFINED NSX_BOARD_FLAGS_TARGET)
    message(FATAL_ERROR
        "Board must define NSX_BOARD_FLAGS_TARGET before nsx-heliart is added.")
endif()

if(NOT DEFINED NSX_SOC_FAMILY)
    message(FATAL_ERROR
        "NSX_SOC_FAMILY must be defined before nsx-heliart is added.")
endif()

# --- Resolve Cortex-M core tag from SoC family ---
set(_HELIART_CORE "")
if(NSX_SOC_FAMILY MATCHES "^apollo5|^apollo330|^apollo510")
    set(_HELIART_CORE "cm55")
elseif(NSX_SOC_FAMILY MATCHES "^apollo4|^apollo3")
    set(_HELIART_CORE "cm4")
else()
    message(FATAL_ERROR
        "nsx-heliart: unsupported NSX_SOC_FAMILY '${NSX_SOC_FAMILY}'")
endif()

# --- Resolve toolchain tag ---
set(_HELIART_TOOLCHAIN "gcc")
if(CMAKE_C_COMPILER_ID STREQUAL "ARMClang" OR
   CMAKE_C_COMPILER MATCHES "armclang")
    set(_HELIART_TOOLCHAIN "armclang")
elseif(CMAKE_C_COMPILER_ID STREQUAL "Clang" OR
       CMAKE_C_COMPILER MATCHES "/atfe/|arm_toolchain_embedded")
    # Arm Toolchain for Embedded (ATfE) is LLVM/Clang-based but, unlike
    # armclang, does NOT define __ARMCC_VERSION. CMake reports it as "Clang".
    set(_HELIART_TOOLCHAIN "atfe")
endif()

# --- Build variant (user-configurable) ---
set(HELIART_VARIANT "release-with-logs" CACHE STRING
    "heliaRT build variant: debug | release-with-logs | release")
set_property(CACHE HELIART_VARIANT PROPERTY STRINGS
    "debug" "release-with-logs" "release")

# --- Locate the prebuilt static library ---
# v1.11.x naming: libhelia-rt-{core}-{tc}-{variant}.a
#
# Library search order:
#   1. ${HELIART_LIB_DIR}     — explicit override from the consumer
#   2. ${CMAKE_CURRENT_LIST_DIR}/lib    — co-located (some in-tree setups)
#   3. ${CMAKE_CURRENT_LIST_DIR}/../lib — release bundle layout (lib/ at root)
set(_HELIART_LIB_NAME
    "libhelia-rt-${_HELIART_CORE}-${_HELIART_TOOLCHAIN}-${HELIART_VARIANT}.a")

set(_HELIART_LIB_SEARCH_DIRS
    "${HELIART_LIB_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/lib"
    "${CMAKE_CURRENT_LIST_DIR}/../lib"
)

set(_HELIART_LIB_PATH "")
foreach(_dir IN LISTS _HELIART_LIB_SEARCH_DIRS)
    if(_dir AND EXISTS "${_dir}/${_HELIART_LIB_NAME}")
        set(_HELIART_LIB_PATH "${_dir}/${_HELIART_LIB_NAME}")
        break()
    endif()
endforeach()

if(NOT _HELIART_LIB_PATH)
    message(FATAL_ERROR
        "nsx-heliart: prebuilt library not found:\n"
        "  ${_HELIART_LIB_NAME}\n"
        "Searched: ${_HELIART_LIB_SEARCH_DIRS}\n"
        "Check HELIART_VARIANT (${HELIART_VARIANT}) and toolchain (${_HELIART_TOOLCHAIN}).")
endif()

message(STATUS "nsx-heliart: using ${_HELIART_LIB_NAME}")

# --- Import the prebuilt static library ---
add_library(nsx_heliart_prebuilt STATIC IMPORTED GLOBAL)
set_target_properties(nsx_heliart_prebuilt PROPERTIES
    IMPORTED_LOCATION "${_HELIART_LIB_PATH}"
)

# --- Locate the TFLM source tree ---
# In the source repo,   nsx/ and tensorflow/ are siblings (root = ../).
# In release bundles,   tflm/ is copied to the bundle root by
#                       package_helia_bundle.sh, so tensorflow/, third_party/
#                       are still siblings of nsx/ (root = ../).
# HELIART_TFLM_ROOT can be overridden if a consumer places the tree
# somewhere else.
set(HELIART_TFLM_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH
    "Root directory containing tensorflow/ and third_party/ for nsx-heliart")

# --- Public-facing target ---
# The prebuilt .a contains the TFLM runtime but leaves MicroPrintf()
# undefined.  We compile the platform glue here so the linker resolves it.
add_library(nsx_heliart STATIC
    ${HELIART_TFLM_ROOT}/tensorflow/lite/micro/micro_log.cc
    ${HELIART_TFLM_ROOT}/tensorflow/lite/micro/cortex_m_generic/debug_log.cc
)
set_target_properties(nsx_heliart PROPERTIES EXPORT_NAME heliart)
add_library(nsx::heliart ALIAS nsx_heliart)

target_link_libraries(nsx_heliart
    PUBLIC
        ${NSX_BOARD_FLAGS_TARGET}
        nsx_heliart_prebuilt
    PRIVATE
        nsx_soc_hal
        nsx_core
)

target_include_directories(nsx_heliart
    PUBLIC
        $<BUILD_INTERFACE:${HELIART_TFLM_ROOT}>
        $<BUILD_INTERFACE:${HELIART_TFLM_ROOT}/third_party>
        $<BUILD_INTERFACE:${HELIART_TFLM_ROOT}/third_party/flatbuffers/include>
        $<BUILD_INTERFACE:${HELIART_TFLM_ROOT}/third_party/gemmlowp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nsx/modules/nsx-heliart>
)

target_compile_definitions(nsx_heliart
    PUBLIC
        TF_LITE_STATIC_MEMORY
        NS_TFSTRUCTURE_RECENT
        NS_TFLM_NEW_MICRO_PROFILER
)

install(TARGETS nsx_heliart nsx_heliart_prebuilt EXPORT nsxTargets)
