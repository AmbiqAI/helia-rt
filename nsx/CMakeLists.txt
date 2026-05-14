# ---------------------------------------------------------------------------
# nsx-heliart — NSX module for the heliaRT TFLM runtime (source build).
# ---------------------------------------------------------------------------
# Phase 2 of issue #147: previously this module wrapped a prebuilt static
# library matrix (libhelia-rt-<core>-<tc>-<variant>.a). It now compiles
# heliaRT from source via the canonical CMake source-of-truth at
# ../cmake/helia_rt_sources.cmake, linking ns-cmsis-nn (provided by the
# nsx-cmsis-nn module — see nsx-module.yaml depends.required).
#
# Public-facing target — unchanged contract:
#   nsx::heliart    INTERFACE library that pulls in the full TFLM runtime
#                   with Ambiq-optimized kernels.
#
# Cache options — back-compat with the old prebuilt module:
#   HELIART_VARIANT      debug | release-with-logs | release   (default: release-with-logs)
#                        Mapped onto HELIA_RT_BUILD_TYPE (- replaced with _).
#   HELIART_TFLM_ROOT    Override repo root. Defaults to the parent dir of nsx/.
#
# Required upstream targets (provided by NSX board package + nsx-cmsis-nn):
#   ${NSX_BOARD_FLAGS_TARGET}    board CFLAGS interface
#   nsx_soc_hal                  SoC HAL (transitive deps for nsx-cmsis-nn)
#   nsx::cmsis_nn (or alias)     ns-cmsis-nn kernels (the helia_rt::helia
#                                target probes nsx::cmsis_nn -> nsx_cmsis_nn
#                                -> ns-cmsis-nn -> ns_cmsis_nn).
# ---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED NSX_BOARD_FLAGS_TARGET)
    message(FATAL_ERROR
        "nsx-heliart: NSX_BOARD_FLAGS_TARGET must be defined before this module is added.")
endif()

if(NOT DEFINED NSX_SOC_FAMILY)
    message(FATAL_ERROR
        "nsx-heliart: NSX_SOC_FAMILY must be defined before this module is added.")
endif()

# --- Build variant (back-compat with prior prebuilt module) ---
set(HELIART_VARIANT "release-with-logs" CACHE STRING
    "heliaRT build variant: debug | release-with-logs | release")
set_property(CACHE HELIART_VARIANT PROPERTY STRINGS
    "debug" "release-with-logs" "release")

# Map HELIART_VARIANT (dash form) -> HELIA_RT_BUILD_TYPE (underscore form
# expected by helia_rt_sources.cmake).
string(REPLACE "-" "_" _heliart_build_type "${HELIART_VARIANT}")
set(HELIA_RT_BUILD_TYPE "${_heliart_build_type}" CACHE STRING
    "heliaRT build flavor (set by nsx-heliart from HELIART_VARIANT)" FORCE)

# --- Locate the heliaRT repo root ---
# Default: nsx/ is a sibling of CMakeLists.txt at the repo root. Allow an
# explicit override (HELIART_TFLM_ROOT) for out-of-tree consumers (e.g. an
# NSX bundle where nsx/ ships at the bundle root with the rest of the repo
# alongside it).
set(_HELIART_REPO_DEFAULT "${CMAKE_CURRENT_LIST_DIR}/..")
set(_HELIART_REPO_SEARCH_DIRS
    "${HELIART_TFLM_ROOT}"
    "${_HELIART_REPO_DEFAULT}"
    "${CMAKE_CURRENT_LIST_DIR}"     # flat bundle layout
)
set(_HELIART_REPO_RESOLVED "")
foreach(_dir IN LISTS _HELIART_REPO_SEARCH_DIRS)
    if(_dir AND EXISTS "${_dir}/cmake/helia_rt_sources.cmake")
        get_filename_component(_HELIART_REPO_RESOLVED "${_dir}" ABSOLUTE)
        break()
    endif()
endforeach()
if(NOT _HELIART_REPO_RESOLVED)
    message(FATAL_ERROR
        "nsx-heliart: could not locate heliaRT repo root (looking for "
        "cmake/helia_rt_sources.cmake).\nSearched: ${_HELIART_REPO_SEARCH_DIRS}\n"
        "Set -DHELIART_TFLM_ROOT=<dir> to a directory containing cmake/helia_rt_sources.cmake.")
endif()
set(HELIART_TFLM_ROOT "${_HELIART_REPO_RESOLVED}" CACHE PATH
    "heliaRT repo root (contains cmake/helia_rt_sources.cmake)" FORCE)

# --- Configure heliaRT for embedded use BEFORE add_subdirectory ---
# Cache settings must be in place before the heliaRT CMakeLists processes
# its option() / set(... CACHE ...) calls.
set(HELIA_RT_ENABLE_HELIA       ON  CACHE BOOL "" FORCE)
set(HELIA_RT_ENABLE_CMSIS_NN    OFF CACHE BOOL "" FORCE)
set(HELIA_RT_STATIC_MEMORY      ON  CACHE BOOL "" FORCE)
# Embedded targets: skip the host-only NEON guard.
set(HELIA_RT_DISABLE_X86_NEON   OFF CACHE BOOL "" FORCE)
# We provide our own Cortex-M debug_log.cc below; suppress the default stub
# so we don't get duplicate-symbol link errors.
set(HELIA_RT_PROVIDE_DEBUG_LOG  OFF CACHE BOOL "" FORCE)

# Tell helia_rt::helia which ns-cmsis-nn target to link. Matches the
# nsx-cmsis-nn module's public alias (nsx::cmsis_nn). Override per-build
# with -DHELIA_RT_NSCMSISNN_TARGET=<name> if the contract changes.
if(NOT HELIA_RT_NSCMSISNN_TARGET)
    set(HELIA_RT_NSCMSISNN_TARGET "nsx::cmsis_nn" CACHE STRING
        "ns-cmsis-nn target to link from helia_rt::helia (nsx-heliart default)" FORCE)
endif()

message(STATUS "nsx-heliart: source build, "
               "variant=${HELIART_VARIANT} (HELIA_RT_BUILD_TYPE=${HELIA_RT_BUILD_TYPE}), "
               "ns-cmsis-nn target=${HELIA_RT_NSCMSISNN_TARGET}, "
               "repo=${HELIART_TFLM_ROOT}")

# --- Bring in heliaRT ---
# EXCLUDE_FROM_ALL: don't materialize generic test/utility targets in the
# parent build. nsx::heliart explicitly depends on helia_rt::helia, so the
# library still gets built when needed.
add_subdirectory("${HELIART_TFLM_ROOT}" heliart-src EXCLUDE_FROM_ALL)

# --- Apply NSX board flags to the source build ---
# helia_rt_helia is created by the helia-rt CMakeLists and needs the
# board's CFLAGS (target arch, FPU, MVE, etc.) to compile. Inject them
# privately so consumers don't see them on their own compile lines via
# nsx::heliart.
target_link_libraries(helia_rt_helia PRIVATE ${NSX_BOARD_FLAGS_TARGET})

# --- ns-cmsis-nn header-prefix shim ---
# heliaRT helia kernels include the cmsis-nn API as
#     #include "Include/arm_nnfunctions.h"
# (with the literal `Include/` prefix). The nsx::cmsis_nn target exposes
# only its `Include/` directory on its INTERFACE_INCLUDE_DIRECTORIES, so the
# parent dir must be added explicitly to resolve the prefixed path. We pull
# it from the cmsis_nn target's SOURCE_DIR via a generator expression — no
# assumption about the on-disk location. (See ns-cmsis-nn issue tracking
# the root-include exposure; once the upstream module exports the parent
# automatically, this block becomes a harmless no-op.)
get_target_property(_nsx_heliart_aliased nsx_cmsis_nn ALIASED_TARGET)
if(_nsx_heliart_aliased)
    set(_nsx_heliart_cmsis_nn_target ${_nsx_heliart_aliased})
else()
    set(_nsx_heliart_cmsis_nn_target nsx_cmsis_nn)
endif()
target_include_directories(helia_rt_helia PRIVATE
    "$<TARGET_PROPERTY:${_nsx_heliart_cmsis_nn_target},SOURCE_DIR>/..")

# --- Cortex-M debug_log glue ---
# OBJECT library so plain object files land directly on the consumer's
# link line; this matches the prior nsx-heliart pattern and guarantees the
# Cortex-M-specific DebugLog() implementation is pulled in regardless of
# link order. Sourced from the upstream TFLM cortex_m_generic backend
# (ITM/SWO output).
add_library(nsx_heliart_glue OBJECT
    "${HELIART_TFLM_ROOT}/tensorflow/lite/micro/cortex_m_generic/debug_log.cc"
)
target_link_libraries(nsx_heliart_glue PRIVATE
    ${NSX_BOARD_FLAGS_TARGET}
    nsx_soc_hal
)
target_include_directories(nsx_heliart_glue PRIVATE
    "${HELIART_TFLM_ROOT}"
    "${HELIART_TFLM_ROOT}/third_party_static/flatbuffers/include"
    "${HELIART_TFLM_ROOT}/third_party_static/gemmlowp"
)
target_compile_definitions(nsx_heliart_glue PRIVATE
    TF_LITE_STATIC_MEMORY
)

# --- Public-facing INTERFACE target ---
add_library(nsx_heliart INTERFACE)
add_library(nsx::heliart ALIAS nsx_heliart)

# Inject the cortex-m debug_log object into consumers' link lines. Plain
# OBJECTs in INTERFACE_LINK_LIBRARIES are not propagated by CMake; the
# documented mechanism is $<TARGET_OBJECTS:...> via target_sources.
target_sources(nsx_heliart INTERFACE
    $<TARGET_OBJECTS:nsx_heliart_glue>
)

target_link_libraries(nsx_heliart INTERFACE
    helia_rt::helia
)
