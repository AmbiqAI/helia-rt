# ---------------------------------------------------------------------------
# nsx-heliart — NSX module for the heliaRT TFLM runtime (source build).
# ---------------------------------------------------------------------------
# Phase 2 of issue #147: previously this module wrapped a prebuilt static
# library matrix (libhelia-rt-<core>-<tc>-<variant>.a). It now compiles
# heliaRT from source via the canonical CMake source-of-truth at
# ../cmake/helia_rt_sources.cmake.
#
# Public-facing target — stable consumer contract:
#   nsx::heliart    INTERFACE library that pulls in the full TFLM runtime
#                   with the selected kernel backend.
#
# Cache options:
#   NSX_HELIART_BACKEND  helia | cmsis_nn | reference  (default: helia)
#                        Selects which helia_rt::* backend nsx::heliart wraps.
#   HELIART_VARIANT      debug | release-with-logs | release   (default: release-with-logs)
#                        Mapped onto HELIA_RT_BUILD_TYPE (- replaced with _).
#   HELIART_TFLM_ROOT    Override repo root. Defaults to the parent dir of nsx/.
#
# Required upstream targets (provided by NSX board package):
#   ${NSX_BOARD_FLAGS_TARGET}    board CFLAGS interface
#   nsx_soc_hal                  SoC HAL (transitive deps for debug_log glue)
#
# Backend-specific upstream targets:
#   helia:    nsx::cmsis_nn (or alias) — ns-cmsis-nn kernels, with Ambiq's
#             additive weight-sum / fast-path symbols actively referenced.
#   cmsis_nn: nsx::cmsis_nn — same kernel library, but the TFLM cmsis_nn
#             wrappers only call the upstream arm_* API. The Ambiq-added
#             symbols sit unreferenced and --gc-sections drops them, giving
#             apples-to-apples vanilla-CMSIS-NN behavior. Override with
#             -DHELIA_RT_CMSISNN_TARGET=<name> to swap in a different
#             CMSIS-NN provider.
#   reference: (none)
#
# Advanced-user note: switching NSX_HELIART_BACKEND in an existing build
# directory resets HELIA_RT_CMSISNN_TARGET / HELIA_RT_NSCMSISNN_TARGET to
# the nsx-heliart defaults (nsx::cmsis_nn). If you supplied a custom -D
# override and want it across a backend switch, re-pass it on reconfigure.
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

# --- Backend selector (issue #155) ---
set(NSX_HELIART_BACKEND "helia" CACHE STRING
    "heliaRT kernel backend: helia | cmsis_nn | reference")
set_property(CACHE NSX_HELIART_BACKEND PROPERTY STRINGS
    "helia" "cmsis_nn" "reference")

# Validate immediately.
set(_nsx_heliart_valid_backends helia cmsis_nn reference)
if(NOT NSX_HELIART_BACKEND IN_LIST _nsx_heliart_valid_backends)
    message(FATAL_ERROR
        "nsx-heliart: invalid NSX_HELIART_BACKEND='${NSX_HELIART_BACKEND}'. "
        "Accepted values: helia, cmsis_nn, reference.")
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

# Map NSX_HELIART_BACKEND -> repo-root HELIA_RT_ENABLE_* toggles.
if(NSX_HELIART_BACKEND STREQUAL "helia")
    set(HELIA_RT_ENABLE_HELIA    ON  CACHE BOOL "" FORCE)
    set(HELIA_RT_ENABLE_CMSIS_NN OFF CACHE BOOL "" FORCE)
elseif(NSX_HELIART_BACKEND STREQUAL "cmsis_nn")
    set(HELIA_RT_ENABLE_HELIA    OFF CACHE BOOL "" FORCE)
    set(HELIA_RT_ENABLE_CMSIS_NN ON  CACHE BOOL "" FORCE)
elseif(NSX_HELIART_BACKEND STREQUAL "reference")
    set(HELIA_RT_ENABLE_HELIA    OFF CACHE BOOL "" FORCE)
    set(HELIA_RT_ENABLE_CMSIS_NN OFF CACHE BOOL "" FORCE)
endif()

set(HELIA_RT_STATIC_MEMORY      ON  CACHE BOOL "" FORCE)
# Embedded targets: skip the host-only NEON guard.
set(HELIA_RT_DISABLE_X86_NEON   OFF CACHE BOOL "" FORCE)
# We provide our own Cortex-M debug_log.cc below; suppress the default stub
# so we don't get duplicate-symbol link errors.
set(HELIA_RT_PROVIDE_DEBUG_LOG  OFF CACHE BOOL "" FORCE)

# Wire the kernel-library target for the selected backend. Both helia and
# cmsis_nn point at nsx::cmsis_nn (the nsx-cmsis-nn module's public alias)
# — ns-cmsis-nn is vanilla CMSIS-NN plus additive Ambiq-authored symbols,
# so the cmsis_nn backend (whose TFLM wrappers only reference the upstream
# arm_* API) gets apples-to-apples vanilla-CMSIS-NN behavior from the same
# kernel library, making it a valid A/B baseline against HELIA. Override
# per-build with -DHELIA_RT_{CMSISNN,NSCMSISNN}_TARGET=<name> if the
# contract changes. The cache vars are kept consistent across reconfigures
# (e.g. switching NSX_HELIART_BACKEND in the same build dir) by unsetting
# the inactive one rather than leaving a stale FORCE entry behind.
if(NSX_HELIART_BACKEND STREQUAL "helia")
    if(NOT HELIA_RT_NSCMSISNN_TARGET)
        set(HELIA_RT_NSCMSISNN_TARGET "nsx::cmsis_nn" CACHE STRING
            "ns-cmsis-nn target to link from helia_rt::helia (nsx-heliart default)" FORCE)
    endif()
    unset(HELIA_RT_CMSISNN_TARGET CACHE)
elseif(NSX_HELIART_BACKEND STREQUAL "cmsis_nn")
    if(NOT HELIA_RT_CMSISNN_TARGET)
        set(HELIA_RT_CMSISNN_TARGET "nsx::cmsis_nn" CACHE STRING
            "CMSIS-NN target to link from helia_rt::cmsis_nn (nsx-heliart default)" FORCE)
    endif()
    unset(HELIA_RT_NSCMSISNN_TARGET CACHE)
else() # reference
    unset(HELIA_RT_CMSISNN_TARGET CACHE)
    unset(HELIA_RT_NSCMSISNN_TARGET CACHE)
endif()

# Compute backend target names for later use.
set(_nsx_heliart_alias_target   "helia_rt::${NSX_HELIART_BACKEND}")
set(_nsx_heliart_concrete_target "helia_rt_${NSX_HELIART_BACKEND}")

# Backend-relevant kernel-target diagnostic for the status line. Only print
# the variable actually consulted by the active backend so the log stays
# uncluttered for cmsis_nn / reference builds.
if(NSX_HELIART_BACKEND STREQUAL "helia")
    set(_nsx_heliart_kernel_diag ", ns-cmsis-nn target=${HELIA_RT_NSCMSISNN_TARGET}")
elseif(NSX_HELIART_BACKEND STREQUAL "cmsis_nn")
    set(_nsx_heliart_kernel_diag ", cmsis-nn target=${HELIA_RT_CMSISNN_TARGET}")
else()
    set(_nsx_heliart_kernel_diag "")
endif()

message(STATUS "nsx-heliart: source build, "
               "backend=${NSX_HELIART_BACKEND}, "
               "variant=${HELIART_VARIANT} (HELIA_RT_BUILD_TYPE=${HELIA_RT_BUILD_TYPE}), "
               "target=${_nsx_heliart_alias_target}"
               "${_nsx_heliart_kernel_diag}, "
               "repo=${HELIART_TFLM_ROOT}")

# --- Bring in heliaRT ---
# EXCLUDE_FROM_ALL: don't materialize generic test/utility targets in the
# parent build. nsx::heliart explicitly depends on helia_rt::helia, so the
# library still gets built when needed.
add_subdirectory("${HELIART_TFLM_ROOT}" heliart-src EXCLUDE_FROM_ALL)

# --- Apply NSX board flags to the source build ---
# The selected backend target needs the board's CFLAGS (target arch, FPU,
# MVE, etc.) to compile. Inject them privately so consumers don't see them
# on their own compile lines via nsx::heliart.
target_link_libraries(${_nsx_heliart_concrete_target} PRIVATE ${NSX_BOARD_FLAGS_TARGET})

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
if(NSX_HELIART_BACKEND STREQUAL "helia" OR NSX_HELIART_BACKEND STREQUAL "cmsis_nn")
    get_target_property(_nsx_heliart_aliased nsx_cmsis_nn ALIASED_TARGET)
    if(_nsx_heliart_aliased)
        set(_nsx_heliart_cmsis_nn_target ${_nsx_heliart_aliased})
    else()
        set(_nsx_heliart_cmsis_nn_target nsx_cmsis_nn)
    endif()
    target_include_directories(${_nsx_heliart_concrete_target} PRIVATE
        "$<TARGET_PROPERTY:${_nsx_heliart_cmsis_nn_target},SOURCE_DIR>/..")
endif()

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
    ${_nsx_heliart_alias_target}
)
