# heliaRT

heliaRT is Ambiq's optimized TensorFlow Lite for Microcontrollers runtime for Apollo platforms. It is designed to help developers bring efficient inference to ultra-low-power Ambiq silicon, with tuned kernels that take advantage of Apollo CPU, DSP, and MVE capabilities where available.

[![CI](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_ci.yml/badge.svg)](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_ci.yml)
[![Unit Tests](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_helia.yml/badge.svg)](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_helia.yml)


## Why heliaRT?

heliaRT focuses on efficient inference for Ambiq edge devices. By aligning the runtime with Apollo hardware capabilities, heliaRT helps reduce integration friction while improving performance and energy efficiency on supported Ambiq targets.

## Key Features

- **Optimized Performance**: Utilizes MVE and DSP hardware capabilities to enhance computational efficiency and speed.
- **Energy Efficiency**: Designed to minimize power usage, extending the battery life of edge devices.
- **Broad Ambiq Coverage**: Supports a range of Ambiq Apollo SoCs through source and prebuilt integration paths.

Start with the [Getting Started guide](docs/usage/index.md) to choose a neuralSPOT, Zephyr, or source-build path for heliaRT on Ambiq hardware.

## Getting Started

The recommended getting-started paths are:

- profile a model with `ns_autodeploy`
- integrate heliaRT into a Zephyr application
- build heliaRT from source for a custom integration

See the full [Getting Started documentation](docs/usage/index.md) for step-by-step instructions.

The main setup guides are:

- [Zephyr setup](docs/usage/zephyr.md)
- [neuralSPOT setup](docs/usage/neuralspot.md)
- [Source builds](docs/usage/source.md)
- [Features overview](docs/features/index.md)


## Supported SoCs

heliaRT is specifically optimized to leverage the advanced features of Ambiq's ultra-low-power SoCs. Below is the list of SoCs that are fully supported:

- **Apollo3**: Ideal for battery-operated mobile devices with its highly efficient power management capabilities.
- **Apollo4**: Enhances performance with higher processing capabilities and improved memory architecture.
- **Apollo4 Plus**: Features a Cortex-M4 core, offering a balance of power and performance for complex processing tasks.
- **Apollo510**: Equipped with a Cortex-M55 core and MVE Helium capabilities, designed for next-level computation needs and edge AI applications.

These optimizations ensure that heliaRT can provide excellent performance and energy efficiency on Ambiq's cutting-edge hardware platforms.


## Official Build Status

This table provides a summary of the build status for heliaRT across various platforms and configurations, ensuring both compatibility and optimal performance.

| Build Type         | Status |
| ------------------ | ------ |
| **CI on Linux**    | [![CI Status](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_ci.yml/badge.svg)](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_ci.yml) |
| **Apollo3/4 (CM4)** | [![Apollo4 Tests](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_helia.yml/badge.svg)](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_helia.yml) |
| **Apollo510 (CM55)** | [![Apollo510 Tests](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_helia.yml/badge.svg)](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_helia.yml) |

Each badge links directly to the detailed results of the respective builds, allowing for quick access to the latest test outcomes and build logs.


## Getting Help

If you encounter issues or need assistance, the following resources are available:

- **Primary Support**: [Submit a GitHub Issue](https://github.com/AmbiqAI/helia-rt/issues/new/choose) for direct support on heliaRT related queries.
- **Community and Discussions**:
  - Contact Ambiq AITG [email group](mailto:support.aitg@ambiq.com)
- **TensorFlow Community**:
  - Pose general questions on the [TensorFlow Discourse forum](https://discuss.tensorflow.org).
  - Engage with the broader TensorFlow community via the [TensorFlow Lite mailing list](https://groups.google.com/a/tensorflow.org/g/tflite).
  - Report broader TensorFlow issues on the [official TensorFlow GitHub page](https://github.com/tensorflow/tensorflow/issues/new/choose).
  - Discuss optimization techniques on the [Model Optimization Toolkit GitHub page](https://github.com/tensorflow/model-optimization).

## Documentation

Explore the main documentation entry points:

- [Getting Started](docs/usage/index.md): Step-by-step guide to begin with heliaRT.
- [Continuous Integration](docs/continuous_integration.md): Details on our CI processes and infrastructure.
- [Benchmarks](docs/benchmarks/index.md): Performance-focused documentation for supported Ambiq targets.
- [Profiling](tensorflow/lite/micro/docs/profiling.md): Techniques to profile and optimize your TFLM applications.
- [Memory Management](tensorflow/lite/micro/docs/memory_management.md): Strategies for effective memory use in constrained environments.
- [Logging](tensorflow/lite/micro/docs/logging.md): How to implement and utilize logging within heliaRT projects.
- [Porting Reference Kernels from TfLite to TFLM](tensorflow/lite/micro/docs/porting_reference_ops.md): Guide on adapting TensorFlow Lite kernels for microcontrollers.
- [Optimized Kernel Implementations](tensorflow/lite/micro/docs/optimized_kernel_implementations.md): Discusses the optimized kernels specific to various architectures.
- [New Platform Support](tensorflow/lite/micro/docs/new_platform_support.md): Instructions for adding heliaRT support to new hardware platforms.
- [heliaRT Python Development Guide](docs/python.md): Insights into using Python for heliaRT development.
- [Automatically Generated Files](docs/automatically_generated_files.md): Information about the files generated during the build process.
- [Python Interpreter Guide](python/tflite_micro/README.md): Detailed guide for using the Python interpreter with TFLM.


## Operator Support Matrix

Below is the operator support matrix for heliaRT's three kernel backends. Each operator is available in all backends at the Reference level. The **CMSIS-NN** and **HELIA** columns indicate where optimized implementations replace the generic reference kernels.

The three backends correspond to Zephyr Kconfig choices:

- **Reference** (`HELIA_RT_BACKEND_REFERENCE`): Generic TFLM C kernels. Works on any architecture.
- **CMSIS-NN** (`HELIA_RT_BACKEND_CMSIS_NN`): Open-source Arm CMSIS-NN optimized kernels. Cortex-M only.
- **HELIA** (`HELIA_RT_BACKEND_HELIA`): Ambiq-optimized kernels (heliaCORE / ns-cmsis-nn). Cortex-M only. Requires Ambiq-provided module.

Data type key: **i8** = int8 activations/weights, **i16** = int16 activations, **i4** = int4 weights, **f32** = float32.

Operators without an optimized variant in a backend fall through to the Reference implementation.

### Compute-Heavy Ops

| Operator | Reference | CMSIS-NN | HELIA | Notes |
| --- | --- | --- | --- | --- |
| conv | f32, i8, i16, i4 | f32, i8, i16, i4 | f32, i8, i16, i4 | HELIA: weight repacking at Prepare |
| depthwise_conv | f32, i8, i16, i4 | f32, i8, i16, i4 | f32, i8, i16, i4 | HELIA: weight repacking at Prepare |
| fully_connected | f32, i8, i16, i4 | f32, i8, i16, i4 | f32, i8, **i16(w8+w16)**, i4 | HELIA uniquely supports A16W16 |
| batch_matmul | f32, i8, i16 | f32, i8, i16 | f32, i8, i16 | |
| transpose_conv | f32, i8, i16 | f32, i8, i16 | f32, i8, i16 | |
| svdf | f32, i8 | f32, i8 | f32, i8 | |
| unidirectional_sequence_lstm | f32, i8 | f32, i8 | f32, i8 | |

### Pooling and Reduce

| Operator | Reference | CMSIS-NN | HELIA | Notes |
| --- | --- | --- | --- | --- |
| avg_pool | f32, i8, i16 | f32, i8, i16 | f32, i8, i16 | |
| max_pool | f32, i8, i16 | f32, i8, i16 | f32, i8, i16 | |
| softmax | f32, i8, i16 | f32, i8, i16 | f32, i8, i16 | |
| reduce (mean/max) | f32, i8 | *(Reference)* | f32, i8 | HELIA-only optimized |

### Activations

| Operator | Reference | CMSIS-NN | HELIA | Notes |
| --- | --- | --- | --- | --- |
| relu / relu6 | f32, i8, i16 | *(Reference)* | f32, i8, i16 | HELIA-only optimized |
| logistic | f32, i8, i16 | *(Reference)* | f32, i8, i16 | HELIA-only optimized |
| tanh | f32, i8, i16 | *(Reference)* | f32, i8, i16 | HELIA-only optimized |
| hard_swish | f32, i8 | *(Reference)* | f32, i8, i16 | HELIA extends to i16 |
| leaky_relu | f32, i8, i16 | *(Reference)* | f32, i8, i16 | HELIA-only optimized |

### Elementwise Arithmetic

| Operator | Reference | CMSIS-NN | HELIA | Notes |
| --- | --- | --- | --- | --- |
| add | f32, i8, i16 | f32, i8, i16 | f32, i8, i16 | |
| mul | f32, i8, i16 | f32, i8, i16 | f32, i8, i16 | |
| sub | f32, i8, i16 | *(Reference)* | f32, i8, i16 | HELIA-only optimized |
| maximum / minimum | f32, i8 | f32, i8 | f32, i8 | |
| comparisons | f32, i8 | *(Reference)* | f32, i8 | HELIA-only optimized |

### Data Movement / Shape

| Operator | Reference | CMSIS-NN | HELIA | Notes |
| --- | --- | --- | --- | --- |
| pad | f32, i8, i16 | f32, i8, i16 | f32, i8, i16 | |
| transpose | f32, i8, i16 | f32, i8, i16 | f32, i8, i16 | |
| concatenation | f32, i8, i16 | *(Reference)* | f32, i8, i16 | HELIA-only optimized |
| reshape | all | *(Reference)* | all | HELIA-only optimized |
| split / split_v | all | *(Reference)* | all | HELIA-only optimized |
| pack | all | *(Reference)* | all | HELIA-only optimized |
| squeeze | all | *(Reference)* | all | HELIA-only optimized |
| strided_slice | all | *(Reference)* | all | HELIA-only optimized |
| dequantize | i8→f32, i16→f32 | *(Reference)* | i8→f32, i16→f32 | HELIA-only optimized |
| fill | all | *(Reference)* | all | HELIA-only optimized |
| zeros_like | f32, i8, i16 | *(Reference)* | f32, i8, i16 | HELIA-only optimized |

All operators not listed above (e.g., cast, elu, gather, slice, unpack, etc.) are available via the Reference backend on all data types they support. See the TFLM documentation for the full reference operator list.
