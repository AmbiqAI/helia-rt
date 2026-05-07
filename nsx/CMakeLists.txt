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
# apollo5* / apollo330* / apollo510* are all Cortex-M55. The apollo510 alt
# would otherwise be redundant with apollo5, but is kept implicit (covered
# by ^apollo5) — listing it separately previously was just shadowed.
set(_HELIART_CORE "")
if(NSX_SOC_FAMILY MATCHES "^apollo5|^apollo330")
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
elseif(CMAKE_C_COMPILER MATCHES "arm_toolchain_embedded|[Aa][Tt][Ff][Ee]")
    # Arm Toolchain for Embedded (ATfE). The installer paths we support are
    # the in-tree download (`<helia>/.../arm_toolchain_embedded/...`) and
    # the system install (e.g. `/Applications/ATfE-x.y.z/...`). We match by
    # path rather than CMAKE_C_COMPILER_ID alone because that is also
    # `Clang` for many other LLVM-based embedded toolchains, which would
    # otherwise be misclassified as ATfE here.
    set(_HELIART_TOOLCHAIN "atfe")
endif()

# --- Build variant (user-configurable) ---
set(HELIART_VARIANT "release-with-logs" CACHE STRING
    "heliaRT build variant: debug | release-with-logs | release")
set_property(CACHE HELIART_VARIANT PROPERTY STRINGS
    "debug" "release-with-logs" "release")

# --- Locate the prebuilt static library ---
# Bundle naming: libhelia-rt-{core}-{tc}-{variant}.a
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
# Search order (mirrors the library lookup above):
#   1. ${HELIART_TFLM_ROOT}              — explicit override from the consumer
#   2. ${CMAKE_CURRENT_LIST_DIR}         — flattened layout (some NSX module
#                                          consumers copy nsx/CMakeLists.txt
#                                          to the bundle root and place
#                                          tensorflow/ alongside it)
#   3. ${CMAKE_CURRENT_LIST_DIR}/..      — source repo + release bundle
#                                          (nsx/ is a sibling of tensorflow/)
#
# HELIART_TFLM_ROOT is exposed as a CACHE PATH so a consumer can override
# it explicitly without editing this file.
set(_HELIART_TFLM_SEARCH_DIRS
    "${HELIART_TFLM_ROOT}"
    "${CMAKE_CURRENT_LIST_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/.."
)
set(_HELIART_TFLM_RESOLVED "")
foreach(_dir IN LISTS _HELIART_TFLM_SEARCH_DIRS)
    if(_dir AND EXISTS "${_dir}/tensorflow/lite/micro/micro_log.cc")
        set(_HELIART_TFLM_RESOLVED "${_dir}")
        break()
    endif()
endforeach()
if(NOT _HELIART_TFLM_RESOLVED)
    message(FATAL_ERROR
        "nsx-heliart: TFLM source tree not found.\n"
        "Looked in: ${_HELIART_TFLM_SEARCH_DIRS}\n"
        "Set -DHELIART_TFLM_ROOT=<dir> to a directory containing\n"
        "  tensorflow/lite/micro/micro_log.cc")
endif()
set(HELIART_TFLM_ROOT "${_HELIART_TFLM_RESOLVED}" CACHE PATH
    "Root directory containing tensorflow/ and third_party/ for nsx-heliart"
    FORCE)

# --- Platform glue (OBJECT library) ---
# The prebuilt .a contains the TFLM runtime but leaves MicroPrintf() and
# DebugLog() undefined. We compile the platform glue here so the linker
# resolves it.
#
# Using an OBJECT library (not STATIC) lets us inject the resulting .o
# files directly onto consumer link lines (see target_sources below).
# Plain object files are always pulled in unconditionally, so single-pass
# static linkers (GNU ld, lld) cannot skip them the way they would skip
# unused members of a sibling archive — MicroPrintf / DebugLog are
# guaranteed to be defined no matter where the prebuilt is in the link
# order.
add_library(nsx_heliart_glue OBJECT
    "${HELIART_TFLM_ROOT}/tensorflow/lite/micro/micro_log.cc"
    "${HELIART_TFLM_ROOT}/tensorflow/lite/micro/cortex_m_generic/debug_log.cc"
)

# Glue compilation needs board flags + soc/core deps.
target_link_libraries(nsx_heliart_glue PRIVATE
    ${NSX_BOARD_FLAGS_TARGET}
    nsx_soc_hal
    nsx_core
)

target_include_directories(nsx_heliart_glue PRIVATE
    "${HELIART_TFLM_ROOT}"
    "${HELIART_TFLM_ROOT}/third_party"
    "${HELIART_TFLM_ROOT}/third_party/flatbuffers/include"
    "${HELIART_TFLM_ROOT}/third_party/gemmlowp"
)

target_compile_definitions(nsx_heliart_glue PRIVATE
    TF_LITE_STATIC_MEMORY
    NS_TFSTRUCTURE_RECENT
    NS_TFLM_NEW_MICRO_PROFILER
)

# --- Public-facing INTERFACE target ---
# This module is intended to be consumed via add_subdirectory() from the
# bundle/source tree (no install/export contract).
add_library(nsx_heliart INTERFACE)
add_library(nsx::heliart ALIAS nsx_heliart)

# Inject glue .o files into consumer link lines via INTERFACE_SOURCES.
# NOTE: putting nsx_heliart_glue in INTERFACE_LINK_LIBRARIES does NOT
# propagate its .o files (CMake treats OBJECT libs in INTERFACE link
# lists as interface-only). $<TARGET_OBJECTS:...> via target_sources is
# the documented way to forward the actual object files.
target_sources(nsx_heliart INTERFACE
    $<TARGET_OBJECTS:nsx_heliart_glue>
)

target_link_libraries(nsx_heliart INTERFACE
    ${NSX_BOARD_FLAGS_TARGET}
    nsx_heliart_prebuilt
)

target_include_directories(nsx_heliart INTERFACE
    "${HELIART_TFLM_ROOT}"
    "${HELIART_TFLM_ROOT}/third_party"
    "${HELIART_TFLM_ROOT}/third_party/flatbuffers/include"
    "${HELIART_TFLM_ROOT}/third_party/gemmlowp"
)

target_compile_definitions(nsx_heliart INTERFACE
    TF_LITE_STATIC_MEMORY
    NS_TFSTRUCTURE_RECENT
    NS_TFLM_NEW_MICRO_PROFILER
)
