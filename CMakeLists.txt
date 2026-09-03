# ---------------------------------------------------------------------------
# Top-level CMakeLists.txt — heliaRT as a CMake package.
#
# This file makes the heliaRT repository directly consumable from generic
# CMake projects via either:
#
#     add_subdirectory(path/to/helia-rt)
#
# or:
#
#     include(FetchContent)
#     FetchContent_Declare(helia_rt
#         GIT_REPOSITORY https://github.com/AmbiqAI/helia-rt.git
#         GIT_TAG        helia-rt-v1.19.0) # x-release-please-version
#     FetchContent_MakeAvailable(helia_rt)
#
# It exposes one STATIC library per backend, depending on what external
# kernel dependencies the caller has wired up:
#
#     helia_rt::reference   — always available, zero external deps
#     helia_rt::cmsis_nn    — requires target `cmsis-nn`        (consumer-provided)
#     helia_rt::helia       — requires target `ns-cmsis-nn`     (consumer-provided)
#
# Zephyr and NSX have their own CMakeLists; they consume the same source
# manifest at cmake/helia_rt_sources.cmake — this top-level file is for
# generic CMake / FetchContent users.
#
# Tracking: https://github.com/AmbiqAI/helia-rt/issues/147
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.21)

# Allow the file to be included via add_subdirectory() *without* its parent
# having called project(...). We do not declare project() here so that
# embedding into Zephyr/NSX/CMSIS-Toolbox doesn't override their project()
# settings. Instead we use a sentinel: if no language has been enabled,
# this is a standalone configure and we declare one.
get_property(_helia_rt_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if(NOT _helia_rt_languages)
    project(helia_rt
        # NOTE: release-please rewrites the version on the line below with the
        # release's SemVer string. CMake's project(VERSION) accepts only numeric
        # X.Y.Z[.W]; a prerelease or build-metadata form (X.Y.Z-rc.N) is NOT
        # valid and would hard-fail configure. This repo cuts no prereleases
        # through release-please today -- see issue #200 before changing that.
        VERSION 1.19.0 # x-release-please-version
        DESCRIPTION "heliaRT — Ambiq's TensorFlow Lite for Microcontrollers (TFLM) runtime"
        LANGUAGES C CXX)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/helia_rt_sources.cmake)

# ---------------------------------------------------------------------------
# User-facing cache options. Defaults mirror the upstream TFLM Makefile.
# ---------------------------------------------------------------------------
set(HELIA_RT_BUILD_TYPE "release_with_logs" CACHE STRING
    "heliaRT build flavor: debug | release_with_logs | release")
set_property(CACHE HELIA_RT_BUILD_TYPE PROPERTY STRINGS
    debug release_with_logs release)

option(HELIA_RT_STATIC_MEMORY      "Define TF_LITE_STATIC_MEMORY"        ON)
option(HELIA_RT_USE_COMPRESSION    "Define USE_TFLM_COMPRESSION"         OFF)
option(HELIA_RT_DISABLE_X86_NEON   "Define TF_LITE_DISABLE_X86_NEON"     OFF)

set(HELIA_RT_CORE_OPT        "-Os" CACHE STRING "Optimization flag for runtime core sources")
set(HELIA_RT_KERNEL_OPT      "-O2" CACHE STRING "Optimization flag for kernel sources")
set(HELIA_RT_THIRD_PARTY_OPT "-O2" CACHE STRING "Optimization flag for third_party_static sources")

set(HELIA_RT_CXX_STANDARD "17" CACHE STRING "C++ standard (14|17)")
set_property(CACHE HELIA_RT_CXX_STANDARD PROPERTY STRINGS 14 17)

set(HELIA_RT_EXTRA_DEFINES "" CACHE STRING
    "Free-form list of extra compile definitions to apply to all heliaRT targets")

# Platform glue. The default tensorflow/lite/micro/debug_log.cc uses
# fprintf(stderr, ...). Embedded targets (Cortex-M / NSX boards) typically
# want a backend-specific implementation (e.g. ITM/SWO). Set OFF and
# provide the override implementation as a separate target/source.
option(HELIA_RT_PROVIDE_DEBUG_LOG
    "Compile the default tensorflow/lite/micro/debug_log.cc into helia_rt targets" ON)

# Source profiles. By default heliaRT targets contain only the production
# inference runtime. The recording/introspection allocators and the
# test/mock/fake helpers are debug-time utilities that bloat the archive and
# are not on the inference path, so they are opt-in:
#
#   HELIA_RT_ENABLE_RECORDING    -> RecordingMicroAllocator / -Interpreter
#                                   (arena-usage auditing)
#   HELIA_RT_ENABLE_TEST_HELPERS -> test_helpers / mock_micro_graph /
#                                   fake_micro_context (unit-test & KernelRunner
#                                   support)
option(HELIA_RT_ENABLE_RECORDING
    "Compile the recording/introspection allocators into helia_rt targets" OFF)
option(HELIA_RT_ENABLE_TEST_HELPERS
    "Compile the test/mock/fake helpers into helia_rt targets" OFF)

# Backend dependency target names. Defensive default lists let us absorb
# upstream renames in cmsis-nn / ns-cmsis-nn without breaking consumers:
# the first existing TARGET in each list wins. Override with a single
# explicit name (e.g. -DHELIA_RT_NSCMSISNN_TARGET=nsx::cmsis_nn) when the
# upstream contract stabilizes.
set(HELIA_RT_CMSISNN_TARGET   "" CACHE STRING
    "Override CMSIS-NN target name for helia_rt::cmsis_nn linkage")
set(HELIA_RT_NSCMSISNN_TARGET "" CACHE STRING
    "Override ns-cmsis-nn target name for helia_rt::helia linkage")
option(HELIA_RT_ENABLE_ETHOSU
    "Enable Ethos-U operator support in heliaRT targets" OFF)

# ---------------------------------------------------------------------------
# Helper: reconcile heliaRT's ARM_NN_ENABLE_F32/F16 definitions with the ones
# actually exported by the linked ns-cmsis-nn target.
#
# The dependency's exported definitions are the ground truth: they reflect
# which float kernels were compiled into its archive. If heliaRT were to
# advertise a float API the dependency does not ship, helia kernels would
# reference symbols that only surface at final link time (and never at all in
# archive-only builds). Conversely, if the dependency ships float kernels we
# did not enable, we would silently fall back to the reference paths.
#
# We therefore adopt the dependency's values and warn on any mismatch.
# ---------------------------------------------------------------------------
function(_helia_rt_sync_float_definitions target dep)
    # Shared resolver; see cmake/helia_rt_sources.cmake.
    helia_rt_float_flags_from_target(${dep} _dep_F32 _dep_F16)

    foreach(_prop COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
        get_target_property(_cur ${target} ${_prop})
        if(NOT _cur)
            set(_cur "")
        endif()
        set(_next "")
        foreach(_d IN LISTS _cur)
            if(NOT _d MATCHES "^ARM_NN_ENABLE_F(32|16)=")
                list(APPEND _next "${_d}")
            endif()
        endforeach()
        foreach(_feature F32 F16)
            if(_dep_${_feature})
                list(APPEND _next "ARM_NN_ENABLE_${_feature}=1")
            endif()
        endforeach()
        set_property(TARGET ${target} PROPERTY ${_prop} "${_next}")
    endforeach()

    helia_rt_float_feature_flags(_want_F32 _want_F16)
    foreach(_feature F32 F16)
        if(NOT _want_${_feature} STREQUAL _dep_${_feature})
            message(WARNING
                "helia_rt_helia: requested ARM_NN_ENABLE_${_feature}="
                "${_want_${_feature}} but the linked target '${dep}' exports "
                "${_dep_${_feature}}. Using the dependency's value. Configure "
                "NSX_CMSIS_NN_ENABLE_${_feature} (NSX module), "
                "ARM_NN_ENABLE_${_feature} (ns-cmsis-nn root CMakeLists), or "
                "CONFIG_NS_CMSIS_NN_ENABLE_${_feature} (Zephyr) *before* "
                "ns-cmsis-nn is added — it selects its sources at that point. "
                "Requires ns-cmsis-nn v7.28.0 or later.")
        endif()
    endforeach()
endfunction()

# ---------------------------------------------------------------------------
# Helper: add one backend target (reference / cmsis_nn / helia).
# ---------------------------------------------------------------------------
function(_helia_rt_add_backend_target backend)
    set(_target helia_rt_${backend})

    helia_rt_select_kernel_sources(_kernels BACKEND ${backend})
    helia_rt_backend_compile_definitions(_backend_defs BACKEND ${backend})
    helia_rt_build_type_compile_definitions(_buildtype_defs BUILD_TYPE ${HELIA_RT_BUILD_TYPE})

    # Select the common (non-kernel) sources for this profile: production
    # runtime always, recording / test-support only when opted in.
    set(_common_rel ${HELIA_RT_RUNTIME_COMMON_SOURCES})
    if(HELIA_RT_ENABLE_RECORDING)
        list(APPEND _common_rel ${HELIA_RT_RECORDING_SOURCES})
    endif()
    if(HELIA_RT_ENABLE_TEST_HELPERS)
        list(APPEND _common_rel ${HELIA_RT_TEST_SUPPORT_SOURCES})
    endif()

    # Resolve common-source paths relative to repo root (they are stored
    # relative in helia_rt_sources.cmake for portability).
    set(_common_abs)
    foreach(_s IN LISTS _common_rel)
        if(NOT HELIA_RT_PROVIDE_DEBUG_LOG AND
           _s STREQUAL "tensorflow/lite/micro/debug_log.cc")
            continue()
        endif()
        list(APPEND _common_abs "${HELIA_RT_ROOT}/${_s}")
    endforeach()

    add_library(${_target} STATIC
        ${_common_abs}
        ${_kernels}
        ${HELIA_RT_ROOT}/tensorflow/lite/micro/kernels/ethos_u/ethosu.cc)
    add_library(helia_rt::${backend} ALIAS ${_target})

    target_include_directories(${_target} PUBLIC ${HELIA_RT_INCLUDE_DIRS})
    if(HELIA_RT_ENABLE_ETHOSU)
        target_compile_definitions(${_target} PUBLIC HELIA_RT_ENABLE_ETHOSU=1)
    else()
        # Suppress the ETHOS_U -> HELIA_RT_ENABLE_ETHOSU implication inside
        # kernels/ethos_u/ethosu.cc, so an explicit OFF on this option (or one
        # forced by the nsx wrapper) wins over an ambient -DETHOS_U from a BSP
        # or toolchain file.
        #
        # PRIVATE, unlike the ENABLE define above: ENABLE is a capability
        # signal consumers may legitimately test with #ifdef before calling
        # AddEthosU(); DISABLE is an internal build-control knob read only by
        # this library's ethosu.cc, and propagating it would suppress the
        # implication in downstream translation units that are none of our
        # business.
        target_compile_definitions(${_target} PRIVATE HELIA_RT_DISABLE_ETHOSU=1)
    endif()

    set_target_properties(${_target} PROPERTIES
        CXX_STANDARD          ${HELIA_RT_CXX_STANDARD}
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS        NO
        POSITION_INDEPENDENT_CODE OFF)

        # TODO: Remove this once upstream TFLM fixes the placement-new sites in
        # micro_allocator.cc / recording_micro_allocator.cc to avoid class-scope
        # operator delete lookup when TF_LITE_STATIC_MEMORY is enabled.
        target_compile_options(${_target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>)

    # Compile definitions: backend + build-type + feature toggles + extras.
    set(_defs ${_backend_defs} ${_buildtype_defs} ${HELIA_RT_EXTRA_DEFINES})
    if(HELIA_RT_STATIC_MEMORY)
        list(APPEND _defs TF_LITE_STATIC_MEMORY)
    endif()
    if(HELIA_RT_USE_COMPRESSION)
        list(APPEND _defs USE_TFLM_COMPRESSION)
    endif()
    if(HELIA_RT_DISABLE_X86_NEON)
        list(APPEND _defs TF_LITE_DISABLE_X86_NEON)
    endif()
    target_compile_definitions(${_target} PUBLIC ${_defs})

    # Per-file optimization layering (mirrors TFLM Makefile):
    #   * core sources       → HELIA_RT_CORE_OPT
    #   * kernel sources     → HELIA_RT_KERNEL_OPT
    #   * third_party_static → HELIA_RT_THIRD_PARTY_OPT (no .cc today, but
    #     reserved for future when third-party sources land in the build)
    # CMake's COMPILE_OPTIONS on the source files is a no-op for the
    # interpreter / compile-flags-only test in CI, but is honored by real
    # toolchains. We use set_source_files_properties so consumers can still
    # override at the target level via target_compile_options(... PRIVATE ...).
    set_source_files_properties(${_common_abs}
        TARGET_DIRECTORY ${_target}
        PROPERTIES COMPILE_OPTIONS "${HELIA_RT_CORE_OPT}")
    set_source_files_properties(${_kernels}
        TARGET_DIRECTORY ${_target}
        PROPERTIES COMPILE_OPTIONS "${HELIA_RT_KERNEL_OPT}")

    # External kernel deps. Consumers must provide these targets *before*
    # add_subdirectory(helia-rt) — FetchContent makes this natural.
    if(backend STREQUAL "cmsis_nn")
        if(HELIA_RT_CMSISNN_TARGET)
            set(_candidates "${HELIA_RT_CMSISNN_TARGET}")
        else()
            set(_candidates cmsis-nn CMSIS::NN)
        endif()
        set(_resolved "")
        foreach(_t IN LISTS _candidates)
            if(TARGET ${_t})
                set(_resolved "${_t}")
                break()
            endif()
        endforeach()
        if(_resolved)
            target_link_libraries(${_target} PUBLIC ${_resolved})
        else()
            message(WARNING
                "helia_rt_cmsis_nn: no CMSIS-NN target found among [${_candidates}]. "
                "Provide it via add_subdirectory or FetchContent before adding helia-rt, "
                "or set -DHELIA_RT_CMSISNN_TARGET=<name>, or link it manually after "
                "this CMakeLists is processed.")
        endif()
    elseif(backend STREQUAL "helia")
        if(HELIA_RT_NSCMSISNN_TARGET)
            set(_candidates "${HELIA_RT_NSCMSISNN_TARGET}")
        else()
            # Defensive default order — current ns-cmsis-nn ships nsx::cmsis_nn
            # as its public alias; older/alt names kept as fallbacks until
            # the public contract stabilizes (issue #147 Phase 2).
            set(_candidates nsx::cmsis_nn nsx_cmsis_nn ns-cmsis-nn ns_cmsis_nn)
        endif()
        set(_resolved "")
        foreach(_t IN LISTS _candidates)
            if(TARGET ${_t})
                set(_resolved "${_t}")
                break()
            endif()
        endforeach()
        if(_resolved)
            target_link_libraries(${_target} PUBLIC ${_resolved})
            target_include_directories(${_target} PRIVATE "$<TARGET_PROPERTY:${_resolved},SOURCE_DIR>")
            _helia_rt_sync_float_definitions(${_target} ${_resolved})
        else()
            message(WARNING
                "helia_rt_helia: no ns-cmsis-nn target found among [${_candidates}]. "
                "Provide it via add_subdirectory or FetchContent before adding helia-rt, "
                "or set -DHELIA_RT_NSCMSISNN_TARGET=<name>, or link it manually after "
                "this CMakeLists is processed.")
        endif()
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Materialize targets.
# ---------------------------------------------------------------------------
_helia_rt_add_backend_target(reference)

# Only build the optimized variants when the user opts in. Otherwise we'd
# error/warn on every configure in environments without the kernel deps.
option(HELIA_RT_ENABLE_CMSIS_NN "Build helia_rt::cmsis_nn target" OFF)
option(HELIA_RT_ENABLE_HELIA    "Build helia_rt::helia target"    OFF)

if(HELIA_RT_ENABLE_CMSIS_NN)
    _helia_rt_add_backend_target(cmsis_nn)
endif()
if(HELIA_RT_ENABLE_HELIA)
    _helia_rt_add_backend_target(helia)
endif()

set(_helia_rt_target_list "helia_rt::reference")
if(HELIA_RT_ENABLE_CMSIS_NN)
    list(APPEND _helia_rt_target_list "helia_rt::cmsis_nn")
endif()
if(HELIA_RT_ENABLE_HELIA)
    list(APPEND _helia_rt_target_list "helia_rt::helia")
endif()
list(JOIN _helia_rt_target_list ", " _helia_rt_target_summary)
message(STATUS "heliaRT ${HELIA_RT_VERSION}: "
               "build_type=${HELIA_RT_BUILD_TYPE} "
               "targets=[${_helia_rt_target_summary}]")
