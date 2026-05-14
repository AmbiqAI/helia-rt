# ---------------------------------------------------------------------------
# cmake/dump_manifest.cmake — emit the resolved heliaRT source manifest as JSON.
#
# Invoke in script mode (no project context required):
#
#     cmake -DBACKEND=reference -P cmake/dump_manifest.cmake
#
# Writes the JSON manifest to stdout. To capture into a file, either redirect
# (`> manifest.json`) or pass an explicit path:
#
#     cmake -DBACKEND=reference -DMANIFEST_OUT=manifest.json \
#         -P cmake/dump_manifest.cmake
#
# Output goes to stdout; the CMSIS-Pack builder pipes it through python json.
# This is the only bridge between the CMake source-of-truth and any
# non-CMake consumer (e.g. tools/cmsis_pack/build_pack.py in Phase 3).
# ---------------------------------------------------------------------------

if(NOT DEFINED BACKEND)
    set(BACKEND reference)
endif()
if(NOT DEFINED MANIFEST_OUT)
    set(MANIFEST_OUT "/dev/stdout")
endif()

# Pretend we're a project so the lists file's get_filename_component(... ABSOLUTE)
# resolves CMAKE_CURRENT_LIST_DIR from this script's location, not the cwd.
include("${CMAKE_CURRENT_LIST_DIR}/helia_rt_sources.cmake")

helia_rt_select_kernel_sources(_kernel_sources BACKEND "${BACKEND}")
helia_rt_backend_compile_definitions(_backend_defs BACKEND "${BACKEND}")

# --- emit JSON ------------------------------------------------------------
# Hand-roll the JSON; CMake has no native JSON-writer. Lists become arrays
# of strings, all paths are POSIX-style relative to HELIA_RT_ROOT.

function(_relpath_list OUT_VAR ROOT)
    set(_rel)
    foreach(_p IN LISTS ARGN)
        file(RELATIVE_PATH _r "${ROOT}" "${_p}")
        # file(RELATIVE_PATH x x) returns empty; surface that as "." so
        # consumers know the repo root itself is on the include path.
        if(_r STREQUAL "")
            set(_r ".")
        endif()
        list(APPEND _rel "${_r}")
    endforeach()
    set(${OUT_VAR} ${_rel} PARENT_SCOPE)
endfunction()

function(_emit_json_array INDENT)
    set(_first TRUE)
    foreach(_item IN LISTS ARGN)
        if(_first)
            set(_first FALSE)
        else()
            string(APPEND _OUT ",\n")
        endif()
        string(APPEND _OUT "${INDENT}\"${_item}\"")
    endforeach()
    set(_OUT "${_OUT}" PARENT_SCOPE)
endfunction()

_relpath_list(_inc_rel  "${HELIA_RT_ROOT}" ${HELIA_RT_INCLUDE_DIRS})
_relpath_list(_kern_rel "${HELIA_RT_ROOT}" ${_kernel_sources})
# COMMON_SOURCES are already repo-relative in helia_rt_sources.cmake.
set(_common_rel ${HELIA_RT_COMMON_SOURCES})

set(_OUT "{\n")
string(APPEND _OUT "  \"schema\": \"helia-rt-manifest-v1\",\n")
string(APPEND _OUT "  \"version\": \"${HELIA_RT_VERSION}\",\n")
string(APPEND _OUT "  \"backend\": \"${BACKEND}\",\n")

string(APPEND _OUT "  \"include_dirs\": [\n")
_emit_json_array("    " ${_inc_rel})
string(APPEND _OUT "\n  ],\n")

string(APPEND _OUT "  \"common_sources\": [\n")
_emit_json_array("    " ${_common_rel})
string(APPEND _OUT "\n  ],\n")

string(APPEND _OUT "  \"kernel_sources\": [\n")
_emit_json_array("    " ${_kern_rel})
string(APPEND _OUT "\n  ],\n")

string(APPEND _OUT "  \"backend_defines\": [")
set(_first TRUE)
foreach(_d IN LISTS _backend_defs)
    if(_first)
        set(_first FALSE)
    else()
        string(APPEND _OUT ", ")
    endif()
    string(APPEND _OUT "\"${_d}\"")
endforeach()
string(APPEND _OUT "]\n")

string(APPEND _OUT "}\n")

# message() writes to stderr in script mode, which breaks `> manifest.json`
# redirection. file(WRITE) honors /dev/stdout on POSIX and accepts an
# arbitrary path otherwise.
file(WRITE "${MANIFEST_OUT}" "${_OUT}")
