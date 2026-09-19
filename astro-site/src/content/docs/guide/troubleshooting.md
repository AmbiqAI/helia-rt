---
title: Troubleshooting
description: Diagnose configuration, link, allocation and invocation failures in order.
---

Keep diagnostics enabled while bringing up the application. Start with `release_with_logs` or `debug` and retain the first failing status and message. Debug the earliest failing stage before interpreting later output.

## The HELIA dependency is missing

HELIA requires `ns-cmsis-nn` in addition to the runtime. In Zephyr, check that both modules were fetched and that the generated configuration selects `CONFIG_HELIA_RT_BACKEND_HELIA=y`. In CMake, add the kernel dependency before heliaRT and pass its target through `HELIA_RT_NSCMSISNN_TARGET` when discovery does not find it.

CMSIS-NN is a different backend with a different dependency API. Substituting `ns-cmsis-nn` for an upstream CMSIS-NN target does not repair an incorrectly selected backend. See [Build configuration](/helia-rt/guide/build-configuration/).

## Float options have no effect

CMake source selection happens while the dependency is configured. Set `ARM_NN_ENABLE_F32/F16` before its `add_subdirectory()` or the NSX bootstrap. Check the resolved feature report and remove obsolete `NSX_CMSIS_NN_ENABLE_F32/F16` inputs from the cache. In Zephyr, inspect the final `.config` because an explicit value can override an implied default.

A prebuilt archive cannot gain kernels from application compiler flags. Use the matching archive and feature manifest or rebuild the dependency. See [Floating point](/helia-rt/guide/floating-point/).

## Downloads fail

Identify the failed URL and the build's selected dependency revision before retrying. Check network/proxy configuration and the downloader's error. A Make source build can use `NS_CMSIS_NN_PATH` to select an existing kernel checkout; preserve its revision and inspect the [HELIA build fragment](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/make/ext_libs/helia.inc) for the separate external-library path requirements. Do not replace a pinned dependency with an arbitrary checkout to bypass a download failure.

## The final link fails

| Symptom | What to check |
| --- | --- |
| Duplicate `Register_*` or interpreter symbols | Runtime/kernel sources and a prebuilt runtime archive may both be linked. Keep one implementation of each. |
| Unresolved `arm_*_f32` / `arm_*_f16` | Kernel archive feature set, dependency revision, transitive link targets and application link command. |
| Unresolved logging implementation | Whether the integration provides `DebugLog` and whether `HELIA_RT_PROVIDE_DEBUG_LOG` matches that choice. |
| Unresolved Ethos-U driver symbols | Driver provider and initialization contract; enabling RT dispatch does not supply a driver. |
| Code or memory region overflow | Linker map, registered operators, enabled optional sources, model placement and [kernel profile](/helia-rt/guide/kernel-profiles/). |

A CMake static archive does not absorb the objects of libraries listed in its link interface. Link the exported CMake target so the final executable receives its dependencies.

## Operator registration fails

Check every `Add...()` return value. Verify resolver capacity and duplicate registrations. Register the operator set the model actually uses; increasing capacity does not add support for a new tensor type or operator option.

## AllocateTensors fails

Allocation includes graph preparation. Read the diagnostic before enlarging the arena:

1. Check model schema compatibility and registrations.
2. Check each node's types, dimensions, axes, quantization and options.
3. Check FP16/FP32 kernel availability and dependency versions.
4. If the allocator reports insufficient space, increase and align the arena, then audit its use after allocation succeeds.

See [Model compatibility](/helia-rt/guide/model-compatibility/) and [Memory and profiling](/helia-rt/guide/memory-and-profiling/). `arena_used_bytes()` after a failed allocation is not a reliable production arena-size prescription.

## Invoke fails or outputs differ

Do not consume outputs after a failed invocation as valid inference results. Verify that inputs were populated after successful allocation, using the exact tensor type, shape and quantization parameters. Retain model and buffer lifetimes. For stateful models, reproduce the input sequence and reset behavior.

Use known inputs and baseline outputs to distinguish preprocessing mistakes from numeric or adapter differences. FP32 and FP16 support is operator-specific; a floating-point model is not inherently invalid. NaN handling, rounding and unsupported shapes can differ across paths. Compare against the version-matched adapter contract, not just an operator-name checklist.

## Timing is zero or misleading

Confirm the platform timer works, profiling has not been compiled out by stripped diagnostic strings, and events are cleared between runs. A host build measures host execution. A Cortex-M source build that merely compiles does not establish MVE execution or board performance.

## Report a reproducible issue

Include runtime and kernel revisions, integration path, compiler/version, target flags, backend/profile, float feature set, first error, model operator/type requirements and the smallest safe reproducer. State whether the failure is during configuration, final link, allocation or invocation and whether it was reproduced on hardware or simulation.

Use [GitHub issues](https://github.com/AmbiqAI/helia-rt/issues/new/choose) for ordinary reports. Follow the [support policy](/helia-rt/guide/maintenance/support/) for security reports.
