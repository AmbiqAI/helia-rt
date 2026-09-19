---
title: neuralSPOT-X
description: Add the heliaRT source module to an NSX application and configure its backend.
---

The `nsx-helia-rt` module builds heliaRT from source and exports the CMake target `nsx::helia_rt`. It uses the board and SDK configuration of your neuralSPOT-X application. This is separate from the older neuralSPOT `module.mk` integration included in release bundles.

## Create or open an application

Install neuralSPOT-X and check its toolchain setup, then create an application for your board:

```bash
pipx install neuralspotx
nsx doctor
nsx create-app rt_app --board apollo510_evb
cd rt_app
nsx module add nsx-helia-rt
```

For an existing NSX application, run the module command from its root instead. NSX resolves the module's dependencies, including `nsx-cmsis-nn`, and vendors them into the application. Keep `nsx.yml` and `nsx.lock` under version control. Inspect their resolved heliaRT and heliaCORE revisions before following configuration instructions for a different release.

See the [NSX application workflow](https://github.com/AmbiqAI/neuralspotx/blob/main/README.md) and [module management](https://github.com/AmbiqAI/neuralspotx/blob/main/docs/user-guide/modules.md) for installation and dependency updates.

## Select the backend and build flavor

Place runtime settings in the application's `CMakeLists.txt` **before** `nsx_bootstrap_app()`:

```cmake
set(NSX_HELIA_RT_BACKEND "helia" CACHE STRING "Runtime backend")
set(HELIA_RT_VARIANT "release-with-logs" CACHE STRING "Runtime build flavor")
```

| Setting | Values |
|---|---|
| `NSX_HELIA_RT_BACKEND` | `helia` (default), `reference`, `cmsis_nn` |
| `HELIA_RT_VARIANT` | `debug`, `release-with-logs` (default), `release` |

The `helia` backend consumes `nsx::cmsis_nn`. The `cmsis_nn` backend instead needs a separately supplied **upstream Arm CMSIS-NN** target selected with `HELIA_RT_CMSISNN_TARGET`; the heliaCORE module is not an interchangeable substitute. Use a fresh build directory when changing backend dependencies.

Add the runtime target to your application's existing link list:

```cmake
target_link_libraries(rt_app PRIVATE nsx::helia_rt)
```

Replace `rt_app` with the executable target in your generated project. Retain the generated board, runtime and other application links.

## Enable floating-point kernels when needed

For a float model, configure the heliaCORE features before `nsx_bootstrap_app()` creates its targets:

```cmake
set(ARM_NN_ENABLE_F32 ON CACHE BOOL "Enable FP32 kernels" FORCE)
```

For FP16 arithmetic, also set `ARM_NN_ENABLE_F16` to `ON` only with a compatible target and toolchain supporting MVE floating point. These options are opt-in on the NSX source path. Leave unused float features disabled for an integer-only application.

Setting application compiler flags after bootstrap does not change which heliaCORE sources were compiled. The module reports the resolved features through `HELIA_RT_FLOAT32_ENABLED` and `HELIA_RT_FLOAT16_ENABLED`. Read configure diagnostics if an option was ignored. See [Model compatibility](/helia-rt/guide/model-compatibility/).

## Add inference and run

Add your model data and inference code to the application target using [First inference](/helia-rt/getting-started/first-inference/). Then configure and build:

```bash
nsx lock --app-dir .
nsx configure --app-dir .
nsx build --app-dir .
```

With your board connected and the programmer configured:

```bash
nsx flash --app-dir .
nsx view --app-dir .
```

Check allocation and invocation status and compare output against known model inputs. A successful build establishes link compatibility, not inference correctness on hardware.

For a larger example, the NSX [keyword-spotting application](https://github.com/AmbiqAI/neuralspotx/tree/main/examples/kws_infer) demonstrates model embedding, an operator resolver, memory placement and per-layer profiling. Its manifest pins its own dependency versions; treat those as an example's tested configuration rather than a pin for every application.

## Integration contract

The authoritative settings and target requirements are in [nsx/CMakeLists.txt](https://github.com/AmbiqAI/helia-rt/blob/main/nsx/CMakeLists.txt) and [module metadata](https://github.com/AmbiqAI/helia-rt/blob/main/nsx/nsx-module.yaml). Ethos-U dispatch is a separate opt-in integration requiring a driver provider, driver initialization and a compatible compiled model; enabling the runtime switch alone is insufficient.
