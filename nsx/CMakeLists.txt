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
#   NSX_HELIA_RT_ENABLE_ETHOSU  ON | OFF  (default: OFF)
#                        Enables Ethos-U custom-op dispatch in the runtime.
#                        The app must link an Ethos-U core-driver provider
#                        (e.g. the nsx-npu module) and initialize it.
#   HELIA_RT_TFLM_ROOT    Override repo root. Defaults to the parent dir of nsx/.
#
# Required upstream targets (provided by NSX board package):
#   ${NSX_BOARD_FLAGS_TARGET}    board CFLAGS interface
#   nsx_soc_hal                  SoC HAL (transitive deps for debug_log glue)
#
# Backend-specific upstream targets:
#   helia:    nsx::cmsis_nn (or alias) — ns-cmsis-nn kernels, with Ambiq's
#             additive weight-sum / fast-path symbols actively referenced.
#             fp32/fp16 are opt-in: set NSX_CMSIS_NN_ENABLE_F32/F16 before
#             add_subdirectory(<ns-cmsis-nn>/nsx ...), which is where
#             ns-cmsis-nn reads them. Publishes what it observed as
#             HELIA_RT_FLOAT32_ENABLED / HELIA_RT_FLOAT16_ENABLED.
#             see #253, #254, docs/guides/floating-point.md
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

# --- Ethos-U custom-op dispatch (opt-in) ---
# Enables the real ethos-u kernel implementation in the heliaRT build
# (HELIA_RT_ENABLE_ETHOSU). The kernel externs the Ethos-U core-driver entry
# points (ethosu_reserve_driver / ethosu_invoke_v3 / ...), so the final app
# must link an Ethos-U core-driver provider module and initialize the driver
# before invoking a Vela-compiled model.
option(NSX_HELIA_RT_ENABLE_ETHOSU
    "Enable Ethos-U custom-op dispatch (requires an Ethos-U core-driver provider module)" OFF)

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

# Map NSX_HELIA_RT_ENABLE_ETHOSU -> repo-root HELIA_RT_ENABLE_ETHOSU toggle.
#
# The repo-root option is owned by this module while the nsx wrapper is in
# use: we FORCE the cache entry below, so a consumer who sets that entry
# directly (-DHELIA_RT_ENABLE_ETHOSU=ON/OFF) has their value overwritten.
# Reject that up front rather than leaving them to discover it by
# disassembling the archive.
#
# The guard compares normalized ON/OFF *values*, and rejects disagreement in
# either direction with anything we did not write ourselves:
#
#  * Our own FORCE write persists, so on a reconfigure of an existing build
#    dir the entry holds our value rather than the consumer's.
#    _NSX_HELIA_RT_OWNS_ETHOSU records the exact value we last wrote, and an
#    entry equal to it is treated as ours -- so toggling the wrapper back and
#    forth in one build dir is accepted instead of telling the user to enable
#    the thing they just disabled. Recording only a boolean "we wrote
#    something" would latch: after the first configure the guard would be
#    disarmed for the life of the build dir, and a later
#    -DHELIA_RT_ENABLE_ETHOSU=ON with the wrapper OFF would be silently
#    clobbered -- the exact override the guard exists to prevent.
#
#  * Both polarities are checked. A truthiness-only test ("entry set but
#    wrapper off") cannot see an explicit -DHELIA_RT_ENABLE_ETHOSU=OFF
#    against a wrapper ON, and would silently compile Ethos-U in for a
#    consumer who explicitly opted out.
#
#  * A normal (non-cache) variable of the same name in an enclosing scope
#    shadows the cache entry and SURVIVES a FORCE cache write -- FORCE
#    updates the cache, not the shadow. Left alone, the build would compile
#    with the parent's value while we report ours. Drop the shadow before
#    writing so the cache entry is what actually takes effect. The guard runs
#    first and reads the shadow (a plain reference sees it ahead of the cache
#    entry), so a shadowed value that disagrees is still reported.
#
# Not detectable: a consumer -D value identical to the one we wrote on the
# previous configure of the same build dir leaves no trace in the cache to
# distinguish it from our own write. That case is accepted and clobbered.
if(NSX_HELIA_RT_ENABLE_ETHOSU)
    set(_nsx_ethosu_want ON)
else()
    set(_nsx_ethosu_want OFF)
endif()
set(_nsx_ethosu_have "")
if(DEFINED HELIA_RT_ENABLE_ETHOSU OR DEFINED CACHE{HELIA_RT_ENABLE_ETHOSU})
    if(HELIA_RT_ENABLE_ETHOSU)
        set(_nsx_ethosu_have ON)
    else()
        set(_nsx_ethosu_have OFF)
    endif()
endif()
set(_nsx_ethosu_wrote "")
if(DEFINED CACHE{_NSX_HELIA_RT_OWNS_ETHOSU})
    if(_NSX_HELIA_RT_OWNS_ETHOSU)
        set(_nsx_ethosu_wrote ON)
    else()
        set(_nsx_ethosu_wrote OFF)
    endif()
endif()
if(NOT "${_nsx_ethosu_have}" STREQUAL ""
        AND NOT "${_nsx_ethosu_have}" STREQUAL "${_nsx_ethosu_want}"
        AND NOT "${_nsx_ethosu_have}" STREQUAL "${_nsx_ethosu_wrote}")
    message(FATAL_ERROR
        "nsx-helia-rt: HELIA_RT_ENABLE_ETHOSU is owned by this module; "
        "set -DNSX_HELIA_RT_ENABLE_ETHOSU=${_nsx_ethosu_have} instead")
endif()
unset(HELIA_RT_ENABLE_ETHOSU)
set(HELIA_RT_ENABLE_ETHOSU ${_nsx_ethosu_want} CACHE BOOL "" FORCE)
set(_NSX_HELIA_RT_OWNS_ETHOSU "${_nsx_ethosu_want}" CACHE INTERNAL
    "HELIA_RT_ENABLE_ETHOSU value last written by nsx-helia-rt in this build dir")
unset(_nsx_ethosu_want)
unset(_nsx_ethosu_have)
unset(_nsx_ethosu_wrote)

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

    # --- Effective float kernel set ---
    # fp32/fp16 are opt-in. ns-cmsis-nn fixed its source list before this
    # module was added, so observe the resolved target, never assign.
    # see #253, #254
    include("${HELIA_RT_TFLM_ROOT}/cmake/helia_rt_sources.cmake")
    helia_rt_resolve_float_flags(
        _nsx_helia_rt_f32 _nsx_helia_rt_f16 _nsx_helia_rt_float_source
        DEP "${HELIA_RT_NSCMSISNN_TARGET}")

    message(STATUS
        "nsx-helia-rt: float kernels: "
        "fp32=${_nsx_helia_rt_f32} fp16=${_nsx_helia_rt_f16} "
        "(source: ${_nsx_helia_rt_float_source})")

    # The option is only a request. Flag a target that shipped something
    # else, which almost always means the option was set too late.
    if(_nsx_helia_rt_float_source STREQUAL "ns-cmsis-nn target")
        foreach(_nsx_f IN ITEMS 32 16)
            if(DEFINED NSX_CMSIS_NN_ENABLE_F${_nsx_f})
                if(NSX_CMSIS_NN_ENABLE_F${_nsx_f})
                    set(_nsx_req ON)
                else()
                    set(_nsx_req OFF)
                endif()
                if(NOT _nsx_req STREQUAL "${_nsx_helia_rt_f${_nsx_f}}")
                    message(WARNING
                        "nsx-helia-rt: NSX_CMSIS_NN_ENABLE_F${_nsx_f}="
                        "${_nsx_req} was ignored -- the linked ns-cmsis-nn "
                        "target '${HELIA_RT_NSCMSISNN_TARGET}' shipped "
                        "fp${_nsx_f}=${_nsx_helia_rt_f${_nsx_f}}, and heliaRT "
                        "compiles against what it shipped. ns-cmsis-nn reads "
                        "that option while its own CMakeLists runs, so it "
                        "must be set *before* "
                        "add_subdirectory(<ns-cmsis-nn>/nsx ...) -- setting "
                        "it afterwards, or on a build dir that already "
                        "configured ns-cmsis-nn, has no effect.")
                endif()
            endif()
        endforeach()
        unset(_nsx_f)
        unset(_nsx_req)
    endif()

    # NOTICE, not WARNING: int8-only is a supported configuration and a
    # WARNING would break consumers configuring with -Werror=dev.
    if(NOT _nsx_helia_rt_f32)
        message(NOTICE
            "nsx-helia-rt: fp32 helia kernels are OFF, so no arm_*_f32 kernel "
            "is linked. FLOAT32 operators still run, on the slower reference "
            "float path; int8 models are unaffected. To get the optimized "
            "fp32 kernels, add\n"
            "    set(NSX_CMSIS_NN_ENABLE_F32 ON CACHE BOOL \"\" FORCE)\n"
            "to the parent CMakeLists *before* "
            "add_subdirectory(<ns-cmsis-nn>/nsx ...) -- ns-cmsis-nn selects "
            "its float sources at that point, so setting it afterwards has no "
            "effect. Requires ns-cmsis-nn v7.28.0 or later.")
    endif()

    # --- MVE-F capability probe ---
    # Ask the toolchain, not CMAKE_SYSTEM_PROCESSOR. Flags come from a
    # transitive walk: boards put -mcpu on a linked SoC flags target.
    # Compile-only; a cross build may have no linkable runtime yet.
    # see #254
    helia_rt_collect_interface_compile_flags(_nsx_helia_rt_mve_flags
        TARGET "${NSX_BOARD_FLAGS_TARGET}")

    # check_<lang>_source_compiles() keys its cache on the result variable
    # name alone, so tie the answer to the flags it was probed with.
    string(SHA256 _nsx_helia_rt_mve_key
        "${_nsx_helia_rt_mve_flags}|${CMAKE_C_FLAGS}|${CMAKE_CXX_FLAGS}")
    # Both sides expanded so an undefined side cannot compare as a literal name.
    if(DEFINED HELIA_RT_MVE_FP_PROBE_KEY AND NOT
       "${HELIA_RT_MVE_FP_PROBE_KEY}" STREQUAL "${_nsx_helia_rt_mve_key}")
        unset(HELIA_RT_TARGET_HAS_MVE_FP CACHE)
    endif()
    set(HELIA_RT_MVE_FP_PROBE_KEY "${_nsx_helia_rt_mve_key}" CACHE INTERNAL
        "Flag string HELIA_RT_TARGET_HAS_MVE_FP was last probed with")

    set(_nsx_helia_rt_mve_src
"#if !(defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 2))
#error no MVE-F
#endif
int main(void){return 0;}
")
    get_property(_nsx_helia_rt_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    # Save and restore exactly, undefined included: both belong to the
    # consumer and are inherited by everything this module adds below.
    set(_nsx_helia_rt_had_req_flags OFF)
    if(DEFINED CMAKE_REQUIRED_FLAGS)
        set(_nsx_helia_rt_had_req_flags ON)
        set(_nsx_helia_rt_saved_req_flags "${CMAKE_REQUIRED_FLAGS}")
    endif()
    set(_nsx_helia_rt_had_try_type OFF)
    if(DEFINED CMAKE_TRY_COMPILE_TARGET_TYPE)
        set(_nsx_helia_rt_had_try_type ON)
        set(_nsx_helia_rt_saved_try_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
    endif()
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    set(CMAKE_REQUIRED_FLAGS "${_nsx_helia_rt_mve_flags}")
    if("C" IN_LIST _nsx_helia_rt_languages)
        include(CheckCSourceCompiles)
        check_c_source_compiles("${_nsx_helia_rt_mve_src}"
            HELIA_RT_TARGET_HAS_MVE_FP)
    elseif("CXX" IN_LIST _nsx_helia_rt_languages)
        include(CheckCXXSourceCompiles)
        check_cxx_source_compiles("${_nsx_helia_rt_mve_src}"
            HELIA_RT_TARGET_HAS_MVE_FP)
    elseif(NOT DEFINED HELIA_RT_TARGET_HAS_MVE_FP)
        # No compiler to ask (languages not enabled yet / script mode).
        set(HELIA_RT_TARGET_HAS_MVE_FP OFF CACHE INTERNAL
            "Target has MVE floating point (__ARM_FEATURE_MVE & 2)")
    endif()
    if(_nsx_helia_rt_had_req_flags)
        set(CMAKE_REQUIRED_FLAGS "${_nsx_helia_rt_saved_req_flags}")
    else()
        unset(CMAKE_REQUIRED_FLAGS)
    endif()
    if(_nsx_helia_rt_had_try_type)
        set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_nsx_helia_rt_saved_try_type}")
    else()
        unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
    endif()
    unset(_nsx_helia_rt_had_req_flags)
    unset(_nsx_helia_rt_saved_req_flags)
    unset(_nsx_helia_rt_had_try_type)
    unset(_nsx_helia_rt_saved_try_type)

    # NOTICE for the same reason as fp32, plus this module cannot see
    # whether the model has a FLOAT16 operator. see #254
    if(HELIA_RT_TARGET_HAS_MVE_FP AND NOT _nsx_helia_rt_f16)
        message(NOTICE
            "nsx-helia-rt: this target has MVE floating point but the fp16 "
            "helia kernels are OFF. A FLOAT16 operator will fail at run time "
            "with kTfLiteError and a message naming the type -- most helia "
            "kernels have no reference float16 path. To enable fp16, add\n"
            "    set(NSX_CMSIS_NN_ENABLE_F16 ON CACHE BOOL \"\" FORCE)\n"
            "to the parent CMakeLists *before* "
            "add_subdirectory(<ns-cmsis-nn>/nsx ...). On GCC 14 this needs "
            "ns-cmsis-nn v7.30.0 or later, which fixed the fp16 sources that "
            "ICE the compiler (GCC PR 118460); the -fno-ssa-phiopt workaround "
            "in tools/make/ext_libs/helia.inc covers the Make path only and "
            "has no effect on this build.")
    endif()

    # Outputs, not knobs: what the build actually has, for consumers to
    # branch on instead of re-deriving it.
    set(HELIA_RT_FLOAT32_ENABLED ${_nsx_helia_rt_f32} CACHE BOOL
        "Effective fp32 helia kernel availability, resolved by nsx-helia-rt from the linked ns-cmsis-nn target (output, not a knob)"
        FORCE)
    set(HELIA_RT_FLOAT16_ENABLED ${_nsx_helia_rt_f16} CACHE BOOL
        "Effective fp16 helia kernel availability, resolved by nsx-helia-rt from the linked ns-cmsis-nn target (output, not a knob)"
        FORCE)

    # Mirror onto the ARM_NN_ENABLE_F32/F16 cache names for consumers that
    # read those. Ours to update when the value matches the companion or the
    # resolved truth; anything else is consumer-owned and left alone.
    # TODO(#254): drop once ns-cmsis-nn publishes this mirror itself
    # (cross-repo consumer: AmbiqAI/helia-aot#349).
    foreach(_nsx_f IN ITEMS 32 16)
        set(_nsx_want "${_nsx_helia_rt_f${_nsx_f}}")
        set(_nsx_ours OFF)
        if(NOT DEFINED ARM_NN_ENABLE_F${_nsx_f})
            set(_nsx_ours ON)
        else()
            if(ARM_NN_ENABLE_F${_nsx_f})
                set(_nsx_have ON)
            else()
                set(_nsx_have OFF)
            endif()
            if(_nsx_have STREQUAL _nsx_want)
                # Matches reality: nothing to disagree about, so adopt it
                # and refresh the companion.
                set(_nsx_ours ON)
            elseif(DEFINED HELIA_RT_ARM_NN_MIRROR_F${_nsx_f})
                if(HELIA_RT_ARM_NN_MIRROR_F${_nsx_f})
                    set(_nsx_last ON)
                else()
                    set(_nsx_last OFF)
                endif()
                if(_nsx_have STREQUAL _nsx_last)
                    set(_nsx_ours ON)
                endif()
            endif()
        endif()

        if(_nsx_ours)
            set(ARM_NN_ENABLE_F${_nsx_f} ${_nsx_want} CACHE BOOL
                "Mirror of the effective fp${_nsx_f} kernel availability, for consumers that read the ARM_NN_ENABLE_* names (set by nsx-helia-rt)"
                FORCE)
            set(HELIA_RT_ARM_NN_MIRROR_F${_nsx_f} ${_nsx_want} CACHE INTERNAL
                "Value nsx-helia-rt last wrote to ARM_NN_ENABLE_F${_nsx_f}")
        else()
            message(WARNING
                "nsx-helia-rt: ARM_NN_ENABLE_F${_nsx_f}=${_nsx_have} "
                "disagrees with the float set this build actually has "
                "(fp${_nsx_f}=${_nsx_want}, source: "
                "${_nsx_helia_rt_float_source}). Leaving your value alone; "
                "heliaRT compiles against the real one. Set "
                "NSX_CMSIS_NN_ENABLE_F${_nsx_f}=${_nsx_have} before the "
                "ns-cmsis-nn module is added if that is what you meant.")
        endif()
    endforeach()
    unset(_nsx_f)
    unset(_nsx_want)
    unset(_nsx_have)
    unset(_nsx_last)
    unset(_nsx_ours)
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
               # Effective post-mapping value, not the requested wrapper
               # value, so the log reflects what actually compiles.
               "ethos-u=${HELIA_RT_ENABLE_ETHOSU}, "
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
        "$<TARGET_PROPERTY:${_nsx_helia_rt_cmsis_nn_target},SOURCE_DIR>"
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
