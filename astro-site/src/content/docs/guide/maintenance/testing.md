---
title: Testing and CI
description: Understand what each validation path covers and where executed evidence comes from.
---

Different builds exercise different runtime paths. Use a workflow result as evidence only for the commit, target and configuration it actually ran.

## Validation paths

| Path | What it covers | Boundary |
| --- | --- | --- |
| HELIA Make tests | HELIA adapters with selected kernel dependencies and profiles | A native run uses the host; Cortex-M/MVE execution requires the corresponding target runner. |
| Corstone-300 FVP legs | Target runtime execution for configurations in the HELIA matrix | Simulation does not establish physical-board power or peripheral behavior. |
| Bazel / upstream tests | Upstream test infrastructure and configured reference/other upstream backends | Bazel does not select the HELIA kernel directory. |
| CMake smoke and link probes | Manifest integrity, source integration, cross-compilation and symbol resolution | Successful static compilation/linking does not prove model outputs. |
| Release builds | Architecture/toolchain/flavor artifact creation and release-side probes | Artifact creation is not a benchmark or a complete execution test. |
| Documentation checks | Site content/build/link contracts | Documentation CI is independent of runtime validation. |

## Workflow entry points

[tests_entry.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/tests_entry.yml) invokes the HELIA test workflow on pull requests. The `ci:run_full` label also invokes the broader upstream CI workflow. Inspect both the trigger and job conditions when requesting additional coverage.

[helia_test.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/helia_test.yml) defines the HELIA GCC and ATfE matrix, including target/profile choices and executed-test floors. [smoke_cmake.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/smoke_cmake.yml) covers manifest and integration checks. Arm Compiler 6 has separate [Cortex-M workflow](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/cortex_m_arm_compiler.yml) and [release workflow](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/helia_release.yml) paths.

Manual entry points include [run_helia.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/run_helia.yml) and [run_ci.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/run_ci.yml). The post-merge [tests_post.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/tests_post.yml) has its own job guards; a reusable workflow appearing there does not mean all its jobs execute for every event.

## Reproduce a failure

Open the failed job and record its commit, container/toolchain revision, exact build arguments and first failing test. Follow its script under `tensorflow/lite/micro/tools/ci_build/` rather than substituting a different host build. Preserve the generated log, test counts and any model fixture used by the run.

Report separately whether you configured, compiled, linked, executed on a host, executed in simulation or ran on physical hardware. For kernel changes, verify the test actually dispatches through the changed adapter and does not only exercise a reference fallback.

## Public claims

A measurement needs its model, inputs, runtime/kernel revisions, target, compiler, profile and methodology. Keep single-operator kernel measurements separate from complete-model results. The [profiling guide](/helia-rt/guide/memory-and-profiling/) describes what to record when comparing applications.
