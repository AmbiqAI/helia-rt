# ---------------------------------------------------------------------------
# zephyr/CMakeLists.txt — heliaRT Zephyr module entry point.
# ---------------------------------------------------------------------------
# Phase 4 of issue #147: this file used to maintain its own per-kernel source
# list, which drifted from the canonical TFLM source set every time a kernel
# was added or removed. It now consumes the single source-of-truth at
# cmake/helia_rt_sources.cmake — adding/removing files there propagates here
# automatically, no zephyr-side edit required.
#
# Backend selection (Kconfig in zephyr/Kconfig):
#   HELIA_RT_BACKEND_REFERENCE   → reference TFLM kernels only
#   HELIA_RT_BACKEND_CMSIS_NN    → Arm open-source CMSIS-NN module
#   HELIA_RT_BACKEND_HELIA       → Ambiq's HELIA / ns-cmsis-nn module
#
# Required external Zephyr modules:
#   CMSIS_NN backend → west module `cmsis-nn` (ZEPHYR_CMSIS_NN_MODULE_DIR)
#   HELIA   backend → west module `ns-cmsis-nn` (ZEPHYR_NS_CMSIS_NN_MODULE_DIR)
# ---------------------------------------------------------------------------

if(NOT CONFIG_HELIA_RT)
    return()
endif()

set(HELIA_RT_DIR ${ZEPHYR_CURRENT_MODULE_DIR})

# Translate the Kconfig HELIA_RT_KERNEL_OPTIMIZE choice into the SSoT cache
# variable BEFORE including helia_rt_sources.cmake, so its default-of-SPEED
# does not win over the user's Kconfig selection. Only meaningful for the
# HELIA backend; the cache var is harmless for reference / cmsis_nn builds.
if(CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SIZE)
    set(HELIA_RT_GLOBAL_KERNEL_OPTIMIZE "SIZE" CACHE STRING
        "Default per-kernel optimization target (set from Kconfig)" FORCE)
elseif(CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SPEED)
    set(HELIA_RT_GLOBAL_KERNEL_OPTIMIZE "SPEED" CACHE STRING
        "Default per-kernel optimization target (set from Kconfig)" FORCE)
endif()

# Pull in the canonical source manifest.
include(${HELIA_RT_DIR}/cmake/helia_rt_sources.cmake)

zephyr_library_named(helia_rt)

# --- Includes from SSoT ---
zephyr_include_directories(${HELIA_RT_INCLUDE_DIRS})

# --- Backend selection ---
if(CONFIG_HELIA_RT_BACKEND_HELIA)
    set(_HELIA_RT_BACKEND helia)
    if(NOT DEFINED ZEPHYR_NS_CMSIS_NN_MODULE_DIR
       OR NOT EXISTS "${ZEPHYR_NS_CMSIS_NN_MODULE_DIR}")
        message(FATAL_ERROR
            "CONFIG_HELIA_RT_BACKEND_HELIA requires the ns-cmsis-nn Zephyr "
            "module (ZEPHYR_NS_CMSIS_NN_MODULE_DIR). Add it to your west "
            "manifest or select REFERENCE / CMSIS-NN.")
    endif()
    set(_HELIA_RT_KERNEL_MODULE_DIR "${ZEPHYR_NS_CMSIS_NN_MODULE_DIR}")
elseif(CONFIG_HELIA_RT_BACKEND_CMSIS_NN)
    set(_HELIA_RT_BACKEND cmsis_nn)
    if(NOT DEFINED ZEPHYR_CMSIS_NN_MODULE_DIR
       OR NOT EXISTS "${ZEPHYR_CMSIS_NN_MODULE_DIR}")
        message(FATAL_ERROR
            "CONFIG_HELIA_RT_BACKEND_CMSIS_NN requires the cmsis-nn Zephyr "
            "module (ZEPHYR_CMSIS_NN_MODULE_DIR). Add it to your west "
            "manifest or select REFERENCE.")
    endif()
    set(_HELIA_RT_KERNEL_MODULE_DIR "${ZEPHYR_CMSIS_NN_MODULE_DIR}")
else()
    set(_HELIA_RT_BACKEND reference)
    set(_HELIA_RT_KERNEL_MODULE_DIR "")
endif()

# --- Backend compile definitions from SSoT ---
helia_rt_backend_compile_definitions(_HELIA_RT_BACKEND_DEFS
    BACKEND ${_HELIA_RT_BACKEND})
if(_HELIA_RT_BACKEND_DEFS)
    zephyr_library_compile_definitions(${_HELIA_RT_BACKEND_DEFS})
endif()

# --- External CMSIS-NN / ns-cmsis-nn header dirs ---
# heliaRT helia & cmsis_nn kernels include the cmsis-nn API as
#     #include "Include/arm_nnfunctions.h"
# (with the literal `Include/` prefix). The external module's repo root must
# be on the include path so the prefixed path resolves; the bare `Include/`
# dir is added too for sources that drop the prefix.
if(_HELIA_RT_KERNEL_MODULE_DIR)
    zephyr_library_include_directories(${_HELIA_RT_KERNEL_MODULE_DIR})
    if(EXISTS "${_HELIA_RT_KERNEL_MODULE_DIR}/Include")
        zephyr_library_include_directories(${_HELIA_RT_KERNEL_MODULE_DIR}/Include)
    endif()
    if(EXISTS "${_HELIA_RT_KERNEL_MODULE_DIR}/CMSIS/Core/Include")
        zephyr_library_include_directories(${_HELIA_RT_KERNEL_MODULE_DIR}/CMSIS/Core/Include)
    endif()
endif()

# --- Resolve kernel sources from SSoT ---
helia_rt_select_kernel_sources(_HELIA_RT_KERNEL_SOURCES
    BACKEND ${_HELIA_RT_BACKEND})

# --- Common (non-kernel) sources from SSoT (relative → absolute) ---
# Match the generic CMake profile split: always compile the production
# runtime, and gate recording / test-support sources behind Zephyr Kconfig.
set(_HELIA_RT_COMMON_REL ${HELIA_RT_RUNTIME_COMMON_SOURCES})
if(CONFIG_HELIA_RT_ENABLE_RECORDING)
    list(APPEND _HELIA_RT_COMMON_REL ${HELIA_RT_RECORDING_SOURCES})
endif()
if(CONFIG_HELIA_RT_ENABLE_TEST_HELPERS)
    list(APPEND _HELIA_RT_COMMON_REL ${HELIA_RT_TEST_SUPPORT_SOURCES})
endif()

set(_HELIA_RT_COMMON_ABS "")
foreach(_rel IN LISTS _HELIA_RT_COMMON_REL)
    list(APPEND _HELIA_RT_COMMON_ABS "${HELIA_RT_DIR}/${_rel}")
endforeach()

# --- Zephyr-specific extras the SSoT does not own ---
# Ethos-U op kernel (Zephyr-only conditional). The SSoT deliberately omits
# this kernel because its inclusion is gated on a board feature.
# When CONFIG_ETHOS_U is set we ship the real implementation; otherwise we
# ship the no-op stub at kernels/ethosu.cc so that consumers calling
# Register_ETHOSU() / MicroMutableOpResolver::AddEthosU() still link.
set(_HELIA_RT_EXTRA_SOURCES "")
if(CONFIG_ETHOS_U)
    list(APPEND _HELIA_RT_EXTRA_SOURCES
        ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/ethos_u/ethosu.cc)
else()
    list(APPEND _HELIA_RT_EXTRA_SOURCES
        ${HELIA_RT_DIR}/tensorflow/lite/micro/kernels/ethosu.cc)
endif()

zephyr_library_sources(
    ${_HELIA_RT_COMMON_ABS}
    ${_HELIA_RT_KERNEL_SOURCES}
    ${_HELIA_RT_EXTRA_SOURCES}
)
