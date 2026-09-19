---
title: Benchmark an application
description: Use the repository's benchmark harnesses and report reproducible model-level measurements.
---

Benchmark the configuration you intend to deploy. Preserve a correctness baseline before changing a backend, kernel profile, compiler or memory placement. A kernel result and a complete-model result answer different questions.

## Available harnesses

| Harness | Use | Boundary |
| --- | --- | --- |
| [Generic model benchmark](https://github.com/AmbiqAI/helia-rt/tree/main/tensorflow/lite/micro/tools/benchmarking) | Run a supplied model, collect metadata, profiling events, input/output CRCs and arena use | Its default input filler writes pseudorandom bytes; adapt input preparation for the model's valid values. |
| [Keyword benchmark](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/benchmarks/keyword_benchmark.cc) | Repeat a fixed workload for platform comparison | The benchmark model has scrambled weights; its output is not an accuracy metric. |
| [Person detection benchmark](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/benchmarks/person_detection_benchmark.cc) | Exercise the included example workload | Use the exact model and target configuration represented by the run. |
| Application plus `MicroProfiler` | Measure your real model and operator mix | Verify timer implementation and account for profiling overhead. |

The [benchmark readme](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/benchmarks/README.md) describes workload boundaries. Neither random inputs nor output CRCs establish accuracy on representative data.

## Build the generic harness

The Make target is `tflm_benchmark`; `run_tflm_benchmark` uses the selected target's runner. Set `GENERIC_BENCHMARK_MODEL_PATH` to compile a model into the program, and `GENERIC_BENCHMARK_ARENA_SIZE` to its arena capacity in bytes. The [Makefile fragment](https://github.com/AmbiqAI/helia-rt/blob/main/tensorflow/lite/micro/tools/benchmarking/Makefile.inc) defines these inputs.

For example, this builds a host reference harness with the repository's person-detection model:

```sh
make -f tensorflow/lite/micro/tools/make/Makefile \
  BUILD_TYPE=release_with_logs \
  GENERIC_BENCHMARK_MODEL_PATH=tensorflow/lite/micro/models/person_detect.tflite \
  GENERIC_BENCHMARK_ARENA_SIZE=153600 \
  tflm_benchmark
```

The arena value is an example harness capacity, not a target's memory size. Check allocation and execution for the selected model. The generated executable is under the selected target's `gen/` output. A host result says nothing about MVE execution or target-board latency.

For HELIA measurements, use the appropriate target runner or board application with `OPTIMIZED_KERNEL_DIR=helia`, the intended architecture/compiler and profile. Keep logs enabled for the supplied profiling instrumentation. See [Memory and profiling](/helia-rt/guide/memory-and-profiling/).

## Run a controlled comparison

1. Fix model bytes, input tensors and expected outputs. Include state/reset sequences for recurrent models.
2. Record RT and kernel revisions, toolchain/version, architecture/FPU/MVE flags, backend, profile and build flavor.
3. Check allocation/invocation status and outputs for every compared build.
4. Separate initialization, input preparation, invocation and output processing when reporting their costs.
5. State warmup policy, iteration count and summary statistic. Retain raw measurements and the measurement harness.
6. Report final firmware code size, arena use and other relevant memory separately. Record model/arena placement.

Use a physical board with a verified timer for CPU performance measurements. The FVP benchmark documentation explicitly cautions against using simulated CPU timing, even for relative comparisons. NPU counter measurements have a separate contract; do not present them as complete CPU-plus-NPU application latency.

Energy requires an energy measurement with a defined interval and setup. Do not convert a cycle reduction directly into a battery-life claim.

## Historical measurements

The repository contains [historical operator results for heliaRT v1.16.0](https://github.com/AmbiqAI/helia-rt/blob/helia-rt-v1.21.0/docs/reference/benchmarks/index.md#test-environment). That page describes an int8, single-operator comparison against an upstream snapshot with GCC and its stated hardware/methodology. Treat the data as belonging to that recorded setup, not as a guarantee for another release, tensor shape, compiler or model. Its small near-parity differences should be interpreted from the raw cycle values rather than rounded speedup labels.

No benchmark number on that historical page substitutes for validating and measuring your own application.
