# cmake/ — heliaRT source-of-truth (SSoT)

The CMake module under this directory is the **single source of truth** for
what files compose the heliaRT TFLM runtime, the per-kernel
reference / CMSIS-NN / HELIA selection rule, and the default compile-option
layering. It is consumed by:

| Channel | Consumer |
|---|---|
| Generic CMake / FetchContent | [`CMakeLists.txt`](../CMakeLists.txt) at the repo root |
| Zephyr module | [`zephyr/CMakeLists.txt`](../zephyr/CMakeLists.txt) *(Phase 4 — refactor pending)* |
| NSX module (source mode) | [`nsx/CMakeLists.txt`](../nsx/CMakeLists.txt) *(Phase 2 — pending)* |
| CMSIS-Pack builder | [`tools/cmsis_pack/build_pack.py`](../tools/cmsis_pack/build_pack.py) via `cmake -P cmake/dump_manifest.cmake` |

Tracked in [issue #147](https://github.com/AmbiqAI/helia-rt/issues/147).

## Files

| File | Purpose |
|---|---|
| `helia_rt_sources.cmake` | Lists (`HELIA_RT_INCLUDE_DIRS`, `HELIA_RT_COMMON_SOURCES`, `HELIA_RT_KERNEL_BASENAMES`) + helper functions (`helia_rt_select_kernel_sources`, `helia_rt_backend_compile_definitions`, `helia_rt_build_type_compile_definitions`). Pure data + logic, no `add_library` calls. Safe to `include()` from any context including `cmake -P` script mode. |
| `dump_manifest.cmake` | Emits the resolved manifest as JSON on stdout. Runnable in script mode: `cmake -DBACKEND=reference -P cmake/dump_manifest.cmake`. |

## Adding a new kernel

1. Drop the `.cc` into [`tensorflow/lite/micro/kernels/`](../tensorflow/lite/micro/kernels/) and any backend variant siblings under `cmsis_nn/` or `helia/`.
2. Add the basename (one line) to `HELIA_RT_KERNEL_BASENAMES` in [`helia_rt_sources.cmake`](helia_rt_sources.cmake).
3. **Done.** Zephyr, NSX (source mode), generic CMake, and CMSIS-Pack pick it up on the next build. The selector function probes for backend-specific variants via `EXISTS`; no manual maintenance is required.

## Adding a new third-party header directory

1. If you bumped an upstream pin, run [`./zephyr_static_export.sh`](../zephyr_static_export.sh) to refresh `third_party_static/`. CI's `static_export_drift_check` will fail the PR if these go out of sync.
2. Append the new path to `HELIA_RT_INCLUDE_DIRS` in [`helia_rt_sources.cmake`](helia_rt_sources.cmake).
3. **Done.**

## Hard constraint: hands off `tensorflow/`

heliaRT periodically re-syncs from `tensorflow/tflite-micro` via [`ci/sync_from_upstream_tf.sh`](../ci/sync_from_upstream_tf.sh). To keep that sync clean:

- The SSoT files live **only** under `cmake/`, top-level `CMakeLists.txt`, and `tools/cmsis_pack/` — never under `tensorflow/`, `signal/`, or `python/tflite_micro/`.
- Per-source compile options (e.g. `-O2`) are applied via `set_source_files_properties(... TARGET_DIRECTORY ...)` on the *target side*. Source files are never edited.
- Kernel selection is path-based (`${KERNEL_ROOT}/<backend>/<base>` vs `${KERNEL_ROOT}/<base>`); no patches.

After `ci/sync_from_upstream_tf.sh` runs on a clean checkout, **no SSoT files should be modified**.

## Customer-facing CMake options

When consuming via the top-level `CMakeLists.txt` (or NSX/Zephyr source-mode shells), the following cache variables tune the runtime. Defaults mirror upstream TFLM's `Makefile`.

| Option | Default | Effect |
|---|---|---|
| `HELIA_RT_BUILD_TYPE` | `release_with_logs` | `debug` / `release_with_logs` (defines `NDEBUG`) / `release` (also `TF_LITE_STRIP_ERROR_STRINGS`) |
| `HELIA_RT_ENABLE_CMSIS_NN` | `OFF` | Materialize `helia_rt::cmsis_nn`. Requires consumer-provided `cmsis-nn` target. |
| `HELIA_RT_ENABLE_HELIA` | `OFF` | Materialize `helia_rt::helia`. Requires consumer-provided `ns-cmsis-nn` target. |
| `HELIA_RT_CORE_OPT` | `-Os` | Optimization flag for runtime core (`micro_allocator.cc`, etc.) |
| `HELIA_RT_KERNEL_OPT` | `-O2` | Optimization flag for kernel `.cc` files |
| `HELIA_RT_THIRD_PARTY_OPT` | `-O2` | Reserved — applied when `third_party_static/` ships compile units |
| `HELIA_RT_STATIC_MEMORY` | `ON` | Defines `TF_LITE_STATIC_MEMORY` (heap-free TFLM runtime) |
| `HELIA_RT_USE_COMPRESSION` | `OFF` | Defines `USE_TFLM_COMPRESSION` |
| `HELIA_RT_DISABLE_X86_NEON` | `OFF` | Defines `TF_LITE_DISABLE_X86_NEON` (host builds only) |
| `HELIA_RT_CXX_STANDARD` | `17` | C++ standard (14 or 17) |
| `HELIA_RT_EXTRA_DEFINES` | (empty) | Free-form compile-definitions escape hatch |

## Example: consuming via `add_subdirectory`

```cmake
# In your project:
set(HELIA_RT_BUILD_TYPE release CACHE STRING "" FORCE)
add_subdirectory(third_party/helia-rt)
target_link_libraries(my_app PRIVATE helia_rt::reference)
```

## Example: consuming via `FetchContent`

```cmake
include(FetchContent)
FetchContent_Declare(helia_rt
    GIT_REPOSITORY https://github.com/AmbiqAI/helia-rt.git
    GIT_TAG        v1.13.1)
FetchContent_MakeAvailable(helia_rt)
target_link_libraries(my_app PRIVATE helia_rt::reference)
```
