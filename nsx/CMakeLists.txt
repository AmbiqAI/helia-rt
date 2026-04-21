# ---------------------------------------------------------------------------
# nsx-heliart — NSX module for heliaRT prebuilt static libraries
# ---------------------------------------------------------------------------
# This module wraps heliaRT prebuilt libraries as an NSX module so that any
# NSX application can link nsx::heliart and get the full TFLM runtime with
# Ambiq-optimized (ns-cmsis-nn) kernels.
#
# The prebuilt .a files are expected under lib/ relative to this file:
#   lib/libtensorflow-microlite-<core>-<toolchain>-<variant>.a
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
endif()

# --- Build variant (user-configurable) ---
set(HELIART_VARIANT "release-with-logs" CACHE STRING
    "heliaRT build variant: debug | release-with-logs | release")
set_property(CACHE HELIART_VARIANT PROPERTY STRINGS
    "debug" "release-with-logs" "release")

# --- Locate the prebuilt static library ---
set(_HELIART_LIB_NAME
    "libtensorflow-microlite-${_HELIART_CORE}-${_HELIART_TOOLCHAIN}-${HELIART_VARIANT}.a")
set(_HELIART_LIB_PATH
    "${CMAKE_CURRENT_LIST_DIR}/lib/${_HELIART_LIB_NAME}")

if(NOT EXISTS "${_HELIART_LIB_PATH}")
    message(FATAL_ERROR
        "nsx-heliart: prebuilt library not found:\n"
        "  ${_HELIART_LIB_PATH}\n"
        "Check HELIART_VARIANT (${HELIART_VARIANT}) and toolchain (${_HELIART_TOOLCHAIN}).")
endif()

message(STATUS "nsx-heliart: using ${_HELIART_LIB_NAME}")

# --- Import the prebuilt static library ---
add_library(nsx_heliart_prebuilt STATIC IMPORTED GLOBAL)
set_target_properties(nsx_heliart_prebuilt PROPERTIES
    IMPORTED_LOCATION "${_HELIART_LIB_PATH}"
)

# --- Public-facing INTERFACE target ---
# The prebuilt .a already contains all TFLM code including platform glue
# (debug_log, micro_log).  This INTERFACE library adds the include paths,
# compile definitions, and link dependencies that consumers need.
add_library(nsx_heliart INTERFACE)
set_target_properties(nsx_heliart PROPERTIES EXPORT_NAME heliart)
add_library(nsx::heliart ALIAS nsx_heliart)

target_link_libraries(nsx_heliart
    INTERFACE
        ${NSX_BOARD_FLAGS_TARGET}
        nsx_heliart_prebuilt
        nsx_soc_hal
        nsx_core
)

target_include_directories(nsx_heliart
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/third_party>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/third_party/flatbuffers/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/third_party/gemmlowp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/nsx/modules/nsx-heliart>
)

target_compile_definitions(nsx_heliart
    INTERFACE
        TF_LITE_STATIC_MEMORY
        NS_TFSTRUCTURE_RECENT
        NS_TFLM_NEW_MICRO_PROFILER
)

install(TARGETS nsx_heliart nsx_heliart_prebuilt EXPORT nsxTargets)
