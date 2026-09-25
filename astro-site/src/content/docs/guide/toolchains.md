---
title: Toolchains and artifacts
description: Match compiler, target and ABI across source builds and release libraries.
---

heliaRT's release builder accepts GCC (`gcc`), Arm Compiler 6 (`armclang`) and Arm Toolchain for Embedded (`atfe`). Choose a compiler that fits your firmware, then keep the application, runtime and kernel library compatible.

## Match the complete configuration

A compiler name alone is insufficient. Record its version, CPU/architecture flags, FPU and floating-point ABI, C++ configuration, linker/runtime libraries, runtime build flavor and dependency versions. The `cortex-m55` target must actually enable the required MVE features to use the corresponding optimized paths.

The [release matrix](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/helia_release.yml) contains `cortex-m4+fp` and `cortex-m55`, each with the three compilers and three build flavors. Use the matching bundle entry and its headers. Do not infer a SPEED/SIZE choice from a filename or apply a source-build flag to an already built archive.

## Build with the release entry point

The [release build script](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/ci_build/build_helia.sh) accepts:

```sh
tensorflow/lite/micro/tools/ci_build/build_helia.sh \
  --arch cortex-m55 \
  --toolchain gcc \
  --build release_with_logs \
  --outdir /tmp/heliart-m55-gcc
```

Its target is `cortex_m_generic` and its default backend is HELIA. Use a unique output directory for each configuration. The Arm Compiler path requires the license configuration documented by the script; do not put license values in committed files.

For precise compiler provisioning, inspect the [CI image definition](https://github.com/AmbiqAI/helia-rt/blob/main/.devcontainer/Dockerfile) and workflow used by the build. A toolchain version listed in an old measurement is not a compatibility guarantee for a different release.

## Compiler optimization and kernel profiles

The CMake source integration controls runtime and kernel compiler optimization using `HELIA_RT_CORE_OPT` and `HELIA_RT_KERNEL_OPT`. These do not replace the [SPEED/SIZE kernel profiles](/helia-rt/guide/kernel-profiles/), which select code paths. See the [option reference](/helia-rt/guide/build-options/) for defaults.

## Known issue: ATfE 22.1.0 MVE auto-vectorization

With MVE enabled (`-mcpu=cortex-m55`) and optimization on, ATfE 22.1.0 (clang 22.1.0) can miscompile auto-vectorized loops in well-defined code:

- At `-O2`, `-O3` or `-Os`, a loop that converts integers to `float` and multiplies them by a constant scale of `0.0625f` (2⁻⁴) can be emitted as a fixed-point `vcvt` with the wrong shift, so the results are wrong ([llvm-project#226591](https://github.com/llvm/llvm-project/issues/226591)).
- At `-Os`, a loop that gathers through an index array can be emitted without its exit test, so it runs past its arrays until the core faults ([llvm-project#226592](https://github.com/llvm/llvm-project/issues/226592)).

The ATfE Cortex-M55 test legs build the heliaRT library with the same MVE and vectorizer settings as the release archive. Application code you compile yourself with this compiler, and a heliaRT library you build from source with it (for example through CMake or Zephyr), are outside that coverage. Until a fixed compiler is available, compile those sources with `-fno-vectorize -fno-slp-vectorize`. MVE stays available to hand-written intrinsics and to the prebuilt library. See [helia-rt#225](https://github.com/AmbiqAI/helia-rt/issues/225).

## What validation establishes

The HELIA PR execution matrix uses GCC and ATfE; the Arm Compiler path has separate workflow coverage. A release archive's successful build and FP symbol link probe establish compilation and symbol availability, not model correctness or board performance. See [Testing and CI](/helia-rt/guide/maintenance/testing/).

Compare compilers with the same model, input data, target, placement and build choices. Check outputs before comparing latency or memory. There is no universal performance percentage for choosing one compiler over another.
