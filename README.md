# heliaRT

heliaRT is Ambiq's optimized TensorFlow Lite for Microcontrollers runtime for Apollo platforms. It is designed to help developers bring efficient inference to ultra-low-power Ambiq silicon, with tuned kernels that take advantage of Apollo CPU, DSP, and MVE capabilities where available.

[![CI](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_ci.yml/badge.svg)](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_ci.yml)
[![Unit Tests](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_ambiq.yml/badge.svg)](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_ambiq.yml)


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
| **Apollo3/4 (CM4)** | [![Apollo4 Tests](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_ambiq.yml/badge.svg)](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_ambiq.yml) |
| **Apollo510 (CM55)** | [![Apollo510 Tests](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_ambiq.yml/badge.svg)](https://github.com/AmbiqAI/helia-rt/actions/workflows/run_ambiq.yml) |

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

Below is a detailed matrix that outlines the support for various operators across different backends and data types within heliaRT. Each entry indicates whether an operator is supported (`Yes`), not supported (`No`), or not applicable (`N/A`) for the specified data type and backend. "N/A" is used where the operation does not logically apply to the data type, whereas "No" indicates that the operation could be supported but currently isn't.

The implementations are categorized under three main technologies:
- **C**: Standard C implementation.
- **DSP**: Utilizes Digital Signal Processing instructions.
- **MVE**: Uses Matrix Vector Extensions.

| Operator          | C <br> int8 | C<br>int16 | C<br>int4* | DSP<br>int8 | DSP<br>int16 | DSP<br>int4* | MVE<br>int8 | MVE<br>int16 | MVE<br>int4* |
| ----------------- | ----------- | ---------- |------------|-------------| -------------|--------------|-------------| -------------|--------------|
| add               | Yes         | Yes        | N/A        | Yes         | Yes          | N/A          | Yes         | Yes          | N/A          |
| batch_matmul      | Yes         | Yes        | No         | Yes         | Yes          | No           | Yes         | Yes          | No           |
| batch_to_space_nd | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| cast              | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| comparisons       | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| concatenation     | Yes         | Yes        | No         | Yes         | Yes          | No           | Yes         | Yes          | No           |
| conv              | Yes         | Yes        | Yes        | Yes         | Yes          | Yes          | Yes         | Yes          | Yes          |
| cumsum            | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| depth_to_space    | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| depthwise_conv    | Yes         | Yes        | Yes        | Yes         | Yes          | Yes          | Yes         | Yes          | Yes          |
| dequantize        | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| elementwise       | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| elu               | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| embedding_lookup  | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| expand_dims       | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| fill              | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| fully_connected   | Yes         | Yes        | Yes        | Yes         | Yes          | Yes          | Yes         | Yes          | Yes          |
| gather_nd         | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| gather            | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| hard_swish        | Yes         | No         | No         | No          | No           | No           | No          | No           | No           |
| l2norm            | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| leaky_relu        | Yes         | Yes        | No         | Yes         | No           | No           | Yes         | Yes          | No           |
| log_softmax       | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| logistic          | Yes         | Yes        | No         | Yes         | No           | No           | Yes         | Yes          | No           |
| lstm              | Yes         | Yes        | No         | Yes         | Yes          | No           | Yes         | Yes          | No           |
| minimum           | Yes         | Yes        | N/A        | No          | No           | N/A          | Yes         | Yes          | N/A          |
| maximum           | Yes         | Yes        | N/A        | No          | No           | N/A          | Yes         | Yes          | N/A          |
| mirror_pad        | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| mul               | Yes         | Yes        | N/A        | Yes         | Yes          | N/A          | Yes         | Yes          | N/A          |
| neg               | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| pack              | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| pad               | Yes         | Yes        | N/A        | No          | No           | N/A          | Yes         | Yes          | N/A          |
| max_pooling       | Yes         | Yes        | N/A        | Yes         | Yes          | N/A          | Yes         | Yes          | N/A          |
| avg_pooling       | Yes         | Yes        | N/A        | Yes         | Yes          | N/A          | Yes         | Yes          | N/A          |
| prelu             | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| resize_bilinear   | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| select            | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| slice             | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| softmax           | Yes         | Yes        | No         | Yes         | Yes          | No           | Yes         | Yes          | No           |
| space_to_batch_nd | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| space_to_depth    | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| split             | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| square_difference | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| strided_slice     | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| sub               | Yes         | Yes        | No         | Yes         | No           | N/A          | No          | No           | No           |
| svdf              | Yes         | Yes        | No         | Yes         | Yes          | No           | Yes         | Yes          | No           |
| tanh              | Yes         | Yes        | No         | Yes         | No           | No           | Yes         | No           | No           |
| transpose_conv    | Yes         | No         | No         | Yes         | No           | No           | Yes         | No           | No           |
| tranpose          | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| unpack            | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |
| zeros_like        | Yes         | Yes        | No         | Yes         | Yes          | No           | Yes         | Yes          | No           |
