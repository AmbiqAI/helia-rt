# ---------------------------------------------------------------------------
# nsx-helia-rt — NSX module for the heliaRT TFLM runtime (source build).
# ---------------------------------------------------------------------------
# Phase 2 of issue #147: previously this module wrapped a prebuilt static
# library matrix (libhelia-rt-<core>-<tc>-<variant>.a). It now compiles
# heliaRT from source via the canonical CMake source-of-truth at
# ../cmake/helia_rt_sources.cmake.
#
# Public-facing target — stable consumer contract:
#   nsx::helia_rt    INTERFACE library that pulls in the full TFLM runtime
#                   with the selected kernel backend.
#
# Cache options:
#   NSX_HELIA_RT_BACKEND  helia | cmsis_nn | reference  (default: helia)
#                        Selects which helia_rt::* backend nsx::helia_rt wraps.
#   HELIA_RT_VARIANT      debug | release-with-logs | release   (default: release-with-logs)
#                        Mapped onto HELIA_RT_BUILD_TYPE (- replaced with _).
#   HELIA_RT_TFLM_ROOT    Override repo root. Defaults to the parent dir of nsx/.
#
# Required upstream targets (provided by NSX board package):
#   ${NSX_BOARD_FLAGS_TARGET}    board CFLAGS interface
#   nsx_soc_hal                  SoC HAL (transitive deps for debug_log glue)
#
# Backend-specific upstream targets:
#   helia:    nsx::cmsis_nn (or alias) — ns-cmsis-nn kernels, with Ambiq's
#             additive weight-sum / fast-path symbols actively referenced.
#   cmsis_nn: User-supplied upstream CMSIS-NN target. ns-cmsis-nn is NOT
#             compatible (modified API signatures). Supply the target via
#             -DHELIA_RT_CMSISNN_TARGET=<name> (e.g. CMSIS::NN from an
#             upstream fetch). A future nsx-cmsis-nn-compat module may
#             provide this natively.
#   reference: (none)
#
# Advanced-user note: switching NSX_HELIA_RT_BACKEND in an existing build
# directory unsets the inactive backend's kernel-target var. For helia,
# the active var resets to nsx::cmsis_nn. For cmsis_nn, you must re-pass
# -DHELIA_RT_CMSISNN_TARGET=<name> on every reconfigure to that backend.
# ---------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED NSX_BOARD_FLAGS_TARGET)
    message(FATAL_ERROR
        "nsx-helia-rt: NSX_BOARD_FLAGS_TARGET must be defined before this module is added.")
endif()

if(NOT DEFINED NSX_SOC_FAMILY)
    message(FATAL_ERROR
        "nsx-helia-rt: NSX_SOC_FAMILY must be defined before this module is added.")
endif()

# --- Backend selector (issue #155) ---
set(NSX_HELIA_RT_BACKEND "helia" CACHE STRING
    "heliaRT kernel backend: helia | cmsis_nn | reference")
set_property(CACHE NSX_HELIA_RT_BACKEND PROPERTY STRINGS
    "helia" "cmsis_nn" "reference")

# Validate immediately.
set(_nsx_helia_rt_valid_backends helia cmsis_nn reference)
if(NOT NSX_HELIA_RT_BACKEND IN_LIST _nsx_helia_rt_valid_backends)
    message(FATAL_ERROR
        "nsx-helia-rt: invalid NSX_HELIA_RT_BACKEND='${NSX_HELIA_RT_BACKEND}'. "
        "Accepted values: helia, cmsis_nn, reference.")
endif()

# --- Build variant (back-compat with prior prebuilt module) ---
set(HELIA_RT_VARIANT "release-with-logs" CACHE STRING
    "heliaRT build variant: debug | release-with-logs | release")
set_property(CACHE HELIA_RT_VARIANT PROPERTY STRINGS
    "debug" "release-with-logs" "release")

# Map HELIA_RT_VARIANT (dash form) -> HELIA_RT_BUILD_TYPE (underscore form
# expected by helia_rt_sources.cmake).
string(REPLACE "-" "_" _helia_rt_build_type "${HELIA_RT_VARIANT}")
set(HELIA_RT_BUILD_TYPE "${_helia_rt_build_type}" CACHE STRING
    "heliaRT build flavor (set by nsx-helia-rt from HELIA_RT_VARIANT)" FORCE)

# --- Locate the heliaRT repo root ---
# Default: nsx/ is a sibling of CMakeLists.txt at the repo root. Allow an
# explicit override (HELIA_RT_TFLM_ROOT) for out-of-tree consumers (e.g. an
# NSX bundle where nsx/ ships at the bundle root with the rest of the repo
# alongside it).
set(_HELIA_RT_REPO_DEFAULT "${CMAKE_CURRENT_LIST_DIR}/..")
set(_HELIA_RT_REPO_SEARCH_DIRS
    "${HELIA_RT_TFLM_ROOT}"
    "${_HELIA_RT_REPO_DEFAULT}"
    "${CMAKE_CURRENT_LIST_DIR}"     # flat bundle layout
)
set(_HELIA_RT_REPO_RESOLVED "")
foreach(_dir IN LISTS _HELIA_RT_REPO_SEARCH_DIRS)
    if(_dir AND EXISTS "${_dir}/cmake/helia_rt_sources.cmake")
        get_filename_component(_HELIA_RT_REPO_RESOLVED "${_dir}" ABSOLUTE)
        break()
    endif()
endforeach()
if(NOT _HELIA_RT_REPO_RESOLVED)
    message(FATAL_ERROR
        "nsx-helia-rt: could not locate heliaRT repo root (looking for "
        "cmake/helia_rt_sources.cmake).\nSearched: ${_HELIA_RT_REPO_SEARCH_DIRS}\n"
        "Set -DHELIA_RT_TFLM_ROOT=<dir> to a directory containing cmake/helia_rt_sources.cmake.")
endif()
set(HELIA_RT_TFLM_ROOT "${_HELIA_RT_REPO_RESOLVED}" CACHE PATH
    "heliaRT repo root (contains cmake/helia_rt_sources.cmake)" FORCE)

# --- Configure heliaRT for embedded use BEFORE add_subdirectory ---
# Cache settings must be in place before the heliaRT CMakeLists processes
# its option() / set(... CACHE ...) calls.

# Map NSX_HELIA_RT_BACKEND -> repo-root HELIA_RT_ENABLE_* toggles.
if(NSX_HELIA_RT_BACKEND STREQUAL "helia")
    set(HELIA_RT_ENABLE_HELIA    ON  CACHE BOOL "" FORCE)
    set(HELIA_RT_ENABLE_CMSIS_NN OFF CACHE BOOL "" FORCE)
elseif(NSX_HELIA_RT_BACKEND STREQUAL "cmsis_nn")
    set(HELIA_RT_ENABLE_HELIA    OFF CACHE BOOL "" FORCE)
    set(HELIA_RT_ENABLE_CMSIS_NN ON  CACHE BOOL "" FORCE)
elseif(NSX_HELIA_RT_BACKEND STREQUAL "reference")
    set(HELIA_RT_ENABLE_HELIA    OFF CACHE BOOL "" FORCE)
    set(HELIA_RT_ENABLE_CMSIS_NN OFF CACHE BOOL "" FORCE)
endif()

set(HELIA_RT_STATIC_MEMORY      ON  CACHE BOOL "" FORCE)
# Embedded targets: skip the host-only NEON guard.
set(HELIA_RT_DISABLE_X86_NEON   OFF CACHE BOOL "" FORCE)
# We provide our own Cortex-M debug_log.cc below; suppress the default stub
# so we don't get duplicate-symbol link errors.
set(HELIA_RT_PROVIDE_DEBUG_LOG  OFF CACHE BOOL "" FORCE)

# Wire the kernel-library target for the selected backend. The helia
# backend uses nsx::cmsis_nn (ns-cmsis-nn). The cmsis_nn backend requires
# an upstream-CMSIS-NN-compatible target — ns-cmsis-nn has modified API
# signatures and won't compile with the vanilla cmsis_nn wrappers. The
# user must supply -DHELIA_RT_CMSISNN_TARGET=<name> explicitly.
# The cache vars are kept consistent across reconfigures (e.g. switching
# NSX_HELIA_RT_BACKEND in the same build dir) by unsetting the inactive one
# rather than leaving a stale FORCE entry behind.
if(NSX_HELIA_RT_BACKEND STREQUAL "helia")
    if(NOT HELIA_RT_NSCMSISNN_TARGET)
        set(HELIA_RT_NSCMSISNN_TARGET "nsx::cmsis_nn" CACHE STRING
            "ns-cmsis-nn target to link from helia_rt::helia (nsx-helia-rt default)" FORCE)
    endif()
    unset(HELIA_RT_CMSISNN_TARGET CACHE)
elseif(NSX_HELIA_RT_BACKEND STREQUAL "cmsis_nn")
    if(NOT HELIA_RT_CMSISNN_TARGET)
        message(FATAL_ERROR
            "nsx-helia-rt: NSX_HELIA_RT_BACKEND=cmsis_nn requires an upstream "
            "CMSIS-NN target. Set -DHELIA_RT_CMSISNN_TARGET=<name> (e.g. CMSIS::NN). "
            "ns-cmsis-nn (nsx::cmsis_nn) is NOT compatible — its API signatures "
            "differ from upstream CMSIS-NN.")
    endif()
    unset(HELIA_RT_NSCMSISNN_TARGET CACHE)
else() # reference
    unset(HELIA_RT_CMSISNN_TARGET CACHE)
    unset(HELIA_RT_NSCMSISNN_TARGET CACHE)
endif()

# Compute backend target names for later use.
set(_nsx_helia_rt_alias_target   "helia_rt::${NSX_HELIA_RT_BACKEND}")
set(_nsx_helia_rt_concrete_target "helia_rt_${NSX_HELIA_RT_BACKEND}")

# Backend-relevant kernel-target diagnostic for the status line. Only print
# the variable actually consulted by the active backend so the log stays
# uncluttered for cmsis_nn / reference builds.
if(NSX_HELIA_RT_BACKEND STREQUAL "helia")
    set(_nsx_helia_rt_kernel_diag ", ns-cmsis-nn target=${HELIA_RT_NSCMSISNN_TARGET}")
elseif(NSX_HELIA_RT_BACKEND STREQUAL "cmsis_nn")
    set(_nsx_helia_rt_kernel_diag ", cmsis-nn target=${HELIA_RT_CMSISNN_TARGET}")
else()
    set(_nsx_helia_rt_kernel_diag "")
endif()

message(STATUS "nsx-helia-rt: source build, "
               "backend=${NSX_HELIA_RT_BACKEND}, "
               "variant=${HELIA_RT_VARIANT} (HELIA_RT_BUILD_TYPE=${HELIA_RT_BUILD_TYPE}), "
               "target=${_nsx_helia_rt_alias_target}"
               "${_nsx_helia_rt_kernel_diag}, "
               "repo=${HELIA_RT_TFLM_ROOT}")

# --- Bring in heliaRT ---
# EXCLUDE_FROM_ALL: don't materialize generic test/utility targets in the
# parent build. nsx::helia_rt explicitly depends on helia_rt::helia, so the
# library still gets built when needed.
add_subdirectory("${HELIA_RT_TFLM_ROOT}" helia-rt-src EXCLUDE_FROM_ALL)

# --- Apply NSX board flags to the source build ---
# The selected backend target needs the board's CFLAGS (target arch, FPU,
# MVE, etc.) to compile. Inject them privately so consumers don't see them
# on their own compile lines via nsx::helia_rt.
target_link_libraries(${_nsx_helia_rt_concrete_target} PRIVATE ${NSX_BOARD_FLAGS_TARGET})

# --- ns-cmsis-nn header-prefix shim (helia + cmsis_nn backends) ---
# heliaRT kernel wrappers include the cmsis-nn API as
#     #include "Include/arm_nnfunctions.h"
# (with the literal `Include/` prefix). The nsx::cmsis_nn target exposes
# only its `Include/` directory on its INTERFACE_INCLUDE_DIRECTORIES, so the
# parent dir must be added explicitly to resolve the prefixed path. We pull
# it from the cmsis_nn target's SOURCE_DIR via a generator expression — no
# assumption about the on-disk location. (See ns-cmsis-nn issue tracking
# the root-include exposure; once the upstream module exports the parent
# automatically, this block becomes a harmless no-op.)
if(NSX_HELIA_RT_BACKEND STREQUAL "helia")
    get_target_property(_nsx_helia_rt_aliased nsx_cmsis_nn ALIASED_TARGET)
    if(_nsx_helia_rt_aliased)
        set(_nsx_helia_rt_cmsis_nn_target ${_nsx_helia_rt_aliased})
    else()
        set(_nsx_helia_rt_cmsis_nn_target nsx_cmsis_nn)
    endif()
    target_include_directories(helia_rt_helia PRIVATE
        "$<TARGET_PROPERTY:${_nsx_helia_rt_cmsis_nn_target},SOURCE_DIR>/..")
endif()

# --- Cortex-M debug_log glue ---
# OBJECT library so plain object files land directly on the consumer's
# link line; this matches the prior nsx-helia-rt pattern and guarantees the
# Cortex-M-specific DebugLog() implementation is pulled in regardless of
# link order. Sourced from the upstream TFLM cortex_m_generic backend
# (ITM/SWO output).
add_library(nsx_helia_rt_glue OBJECT
    "${HELIA_RT_TFLM_ROOT}/tensorflow/lite/micro/cortex_m_generic/debug_log.cc"
)
target_link_libraries(nsx_helia_rt_glue PRIVATE
    ${NSX_BOARD_FLAGS_TARGET}
    nsx_soc_hal
)
target_include_directories(nsx_helia_rt_glue PRIVATE
    "${HELIA_RT_TFLM_ROOT}"
    "${HELIA_RT_TFLM_ROOT}/third_party_static/flatbuffers/include"
    "${HELIA_RT_TFLM_ROOT}/third_party_static/gemmlowp"
)
target_compile_definitions(nsx_helia_rt_glue PRIVATE
    TF_LITE_STATIC_MEMORY
)

# --- Public-facing INTERFACE target ---
add_library(nsx_helia_rt INTERFACE)
add_library(nsx::helia_rt ALIAS nsx_helia_rt)

# Inject the cortex-m debug_log object into consumers' link lines. Plain
# OBJECTs in INTERFACE_LINK_LIBRARIES are not propagated by CMake; the
# documented mechanism is $<TARGET_OBJECTS:...> via target_sources.
target_sources(nsx_helia_rt INTERFACE
    $<TARGET_OBJECTS:nsx_helia_rt_glue>
)

target_link_libraries(nsx_helia_rt INTERFACE
    ${_nsx_helia_rt_alias_target}
)
