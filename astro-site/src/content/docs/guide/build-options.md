---
title: Build option reference
description: Public heliaRT configuration options across CMake, neuralSPOT-X, Zephyr and Make.
---

This page lists the heliaRT integration options defined by the repository. Compiler flags, board configuration and options belonging to heliaCORE or upstream build tools have their own contracts. Start with [Build configuration](/helia-rt/guide/build-configuration/) for the decisions behind these settings.

## Root CMake options

Set cache options before `add_subdirectory(helia-rt)`. The [root CMake file](https://github.com/AmbiqAI/helia-rt/blob/main/CMakeLists.txt) and [source manifest](https://github.com/AmbiqAI/helia-rt/blob/main/cmake/helia_rt_sources.cmake) are authoritative.

| Option | Default | Purpose |
| --- | --- | --- |
| `HELIA_RT_ENABLE_HELIA` | `OFF` | Build `helia_rt::helia`; provide an ns-cmsis-nn dependency target. |
| `HELIA_RT_ENABLE_CMSIS_NN` | `OFF` | Build `helia_rt::cmsis_nn`; provide an upstream CMSIS-NN target. |
| `HELIA_RT_NSCMSISNN_TARGET` | Empty, discover target | Explicit kernel dependency target for HELIA. |
| `HELIA_RT_CMSISNN_TARGET` | Empty, discover target | Explicit kernel dependency target for CMSIS-NN. |
| `HELIA_RT_BUILD_TYPE` | `release_with_logs` | `debug`, `release_with_logs`, or `release`; controls assertions and error strings. |
| `HELIA_RT_GLOBAL_KERNEL_OPTIMIZE` | `SPEED` | Global HELIA profile: `SPEED` or `SIZE`. |
| `HELIA_RT_CONV_OPT` | Empty, inherit global | Convolution family override: `SPEED` or `SIZE`. |
| `HELIA_RT_FC_OPT` | Empty, inherit global | Fully connected override: `SPEED` or `SIZE`. |
| `HELIA_RT_STATIC_MEMORY` | `ON` | Export `TF_LITE_STATIC_MEMORY`. |
| `HELIA_RT_USE_COMPRESSION` | `OFF` | Export `USE_TFLM_COMPRESSION`; enabling the flag alone does not convert a model to a supported compressed representation. |
| `HELIA_RT_DISABLE_X86_NEON` | `OFF` | Export `TF_LITE_DISABLE_X86_NEON` for applicable host builds. |
| `HELIA_RT_CORE_OPT` | `-Os` | Compiler optimization option applied to common runtime sources. |
| `HELIA_RT_KERNEL_OPT` | `-O2` | Compiler optimization option applied to selected kernel sources. |
| `HELIA_RT_THIRD_PARTY_OPT` | `-O2` | Declared for third-party optimization; the root build does not apply it to a separate third-party source set. |
| `HELIA_RT_CXX_STANDARD` | `17` | Requested C++ standard; cache choices are `14` and `17`. |
| `HELIA_RT_EXTRA_DEFINES` | Empty | Extra compile definitions exported by the runtime target. |
| `HELIA_RT_PROVIDE_DEBUG_LOG` | `ON` | Compile the default logging implementation; turn off when the platform supplies it. |
| `HELIA_RT_ENABLE_RECORDING` | `OFF` | Include recording allocators for arena auditing. |
| `HELIA_RT_ENABLE_TEST_HELPERS` | `OFF` | Include mock/test helper sources for test harnesses. |
| `HELIA_RT_ENABLE_ETHOSU` | `OFF` | Enable Ethos-U custom-op dispatch; the application must supply and initialize the driver. |

`helia_rt::reference` is always available. Enabling multiple backend targets makes multiple libraries available; link the one selected for an application rather than linking duplicate runtime implementations.

Float inputs `ARM_NN_ENABLE_F32/F16` belong to the kernel dependency. Set them before that dependency is configured. RT reads the resolved features; see [Floating point](/helia-rt/guide/floating-point/).

## neuralSPOT-X wrapper

The [NSX wrapper](https://github.com/AmbiqAI/helia-rt/blob/main/nsx/CMakeLists.txt) supplies the application target `nsx::helia_rt` and owns several root settings.

| Option | Default | Purpose |
| --- | --- | --- |
| `NSX_HELIA_RT_BACKEND` | `helia` | `helia`, `cmsis_nn` or `reference`. |
| `HELIA_RT_VARIANT` | `release-with-logs` | `debug`, `release-with-logs` or `release`; maps to root build type. |
| `NSX_HELIA_RT_ENABLE_ETHOSU` | `OFF` | Wrapper-owned Ethos-U enable switch. Use this instead of the root option. |
| `HELIA_RT_TFLM_ROOT` | Resolved repository root | Override runtime source location for bundle layouts. |
| `HELIA_RT_NSCMSISNN_TARGET` | `nsx::cmsis_nn` when unset | HELIA dependency target. |
| `HELIA_RT_CMSISNN_TARGET` | Required for CMSIS-NN backend | An upstream-compatible CMSIS-NN target, not the HELIA kernel target. |

The wrapper forces static memory on, disables the root default logging source and supplies platform logging glue. It selects the root backend and maps the build flavor. Root profile and recording settings remain useful source-build controls.

`NSX_BOARD_FLAGS_TARGET` and `NSX_SOC_FAMILY` are required board-provided integration inputs. `HELIA_RT_FLOAT32_ENABLED`, `HELIA_RT_FLOAT16_ENABLED` and `HELIA_RT_TARGET_HAS_MVE_FP` report resolved capabilities; they are not feature requests.

## Zephyr Kconfig

The [source module's Kconfig](https://github.com/AmbiqAI/helia-rt/blob/main/zephyr/Kconfig) exposes the following application settings. Generated prebuilt bundles have their own module configuration; do not assume source-module options can alter their archive.

| Setting | Meaning |
| --- | --- |
| `CONFIG_HELIA_RT` | Enable the runtime source module and its C++ requirements. |
| `CONFIG_HELIA_RT_BACKEND_HELIA` | HELIA backend; requires Ambiq SoC family and ns-cmsis-nn. |
| `CONFIG_HELIA_RT_BACKEND_CMSIS_NN` | Upstream CMSIS-NN backend; requires Cortex-M and CMSIS-NN. |
| `CONFIG_HELIA_RT_BACKEND_REFERENCE` | Reference backend. |
| `CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SPEED` | Global HELIA SPEED profile, the profile default. |
| `CONFIG_HELIA_RT_KERNEL_OPTIMIZE_SIZE` | Global HELIA SIZE profile. |
| `CONFIG_HELIA_RT_ENABLE_RECORDING` | Recording allocator sources. |
| `CONFIG_HELIA_RT_ENABLE_TEST_HELPERS` | Test/mock helper sources. |

The backend choice defaults to HELIA for `SOC_FAMILY_AMBIQ && NS_CMSIS_NN`, then CMSIS-NN for `SOC_FAMILY_AMBIQ && CMSIS_NN`, otherwise Reference. Inspect the final configuration when depending on defaults. Float options `CONFIG_NS_CMSIS_NN_ENABLE_F32/F16` belong to the dependency; HELIA provides weak implied defaults.

## Make integration

These variables are consumed by the [Makefile](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/make/Makefile) and [HELIA extension](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/make/ext_libs/helia.inc).

| Variable | Purpose |
| --- | --- |
| `TARGET` | Build platform, for example `cortex_m_generic`; a native build is a host build. |
| `TARGET_ARCH` | Architecture configuration such as `cortex-m4+fp` or `cortex-m55`. |
| `TOOLCHAIN` | Compiler path; the release builder accepts `gcc`, `armclang`, `atfe`. |
| `OPTIMIZED_KERNEL_DIR` | `helia`, `cmsis_nn` or empty/reference specialization. |
| `BUILD_TYPE` | `debug`, `release_with_logs` or `release`. |
| `GLOBAL_KERNEL_OPTIMIZE` | Global profile; default `SPEED`. |
| `CONV_OPT`, `FC_OPT` | Family overrides, each defaulting to global. |
| `NS_CMSIS_NN_COMMIT` | Kernel source revision used by the downloader; default is the pin in `helia.inc`. |
| `NS_CMSIS_NN_PATH` | Existing kernel checkout instead of the default download location. |
| `NS_CMSIS_NN_LIBS` | External kernel link libraries instead of compiling the dependency sources. Requires kernel and CMSIS header paths. |
| `CMSIS_PATH` | CMSIS header location; pass explicitly as a Make command-line assignment when using the external-library path. |
| `CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY` | Optional kernel requantization implementation switch; the HELIA extension forwards a non-empty value to C/C++ compilation. |

Float definitions in the HELIA Make path are derived by `helia.inc` from the target. They are not the CMake feature-selection interface. External libraries must match those definitions.

For profile examples, see [Kernel profiles](/helia-rt/guide/kernel-profiles/). For compiler/ABI selection and the release script's flags, see [Toolchains and artifacts](/helia-rt/guide/toolchains/).
