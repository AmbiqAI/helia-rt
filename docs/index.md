#

[![](./assets/helios-rt-banner-light.png#only-light)](https://ambiqai.github.io/helios-rt/)
[![](./assets/helios-rt-banner-dark.png#only-dark)](https://ambiqai.github.io/helios-rt/)

## 📖 Overview

Welcome to HeliosRT, a specialized fork of TensorFlow Lite for Microcontrollers (TFLM), tailored for Ambiq's Apollo family of ultra-low-power System-on-Chips (SoCs). This project enhances TFLM by optimizing it to leverage the advanced hardware intrinsics of the Apollo series, such as Matrix Vector Extensions (MVE) and Digital Signal Processing (DSP) instructions.

## Why HeliosRT?

HeliosRT addresses the unique demands of ultra-low-power devices, offering developers the tools to build AI applications that require minimal energy consumption without sacrificing performance. By harnessing the specific capabilities of Ambiq's SoCs, HeliosRT enables more efficient neural network inference on devices like smart watches, fitness trackers, and other smart IoT devices.

## 🚀  Key Features

- **Optimized Performance**: Utilizes MVE and DSP hardware capabilities to enhance computational efficiency and speed.
- **Energy Efficiency**: Designed to minimize power usage, extending the battery life of edge devices.
- **Broad Compatibility**: Supports a wide range of Ambiq's Apollo SoCs, ensuring versatile applications across different hardware.

Explore our [Getting Started guide](usage/index.md) to dive into the development with HeliosRT, or check out the [Benchmarks section](benchmarks/index.md) to see how HeliosRT performs under various conditions. Whether you are developing for wearable technology or other smart devices, HeliosRT provides a robust framework for embedding AI into your projects.

## 📚 Quick Links

- **Install HeliosRT** and getting up and running in minutes. &nbsp; [:material-clock-fast: Install HeliosRT](usage/index.md){ .md-button }

- **Usage Examples** showcasing real-world applications and best practices. &nbsp; [:material-book-open-page-variant: Usage Examples](examples/index.md){ .md-button }

- **Performance Benchmarks** comparing HeliosRT to other frameworks. &nbsp; [:material-chart-bar: Performance Benchmarks](benchmarks/index.md){ .md-button }


## Getting Started

Jumpstart your development with [neuralSPOT](https://github.com/AmbiqAI/neuralSPOT), a robust AI SDK optimized for Ambiq's ultra-low-power Apollo SoCs. This toolkit provides comprehensive resources, including the latest stable releases of HeliosRT, fully configured for both GCC and Arm Clang across debug and release builds. For a detailed step-by-step guide, refer to our [Getting Started documentation](usage/index.md).


## Operator Support Matrix

Below is a detailed matrix that outlines the support for various operators across different backends and data types within HeliosRT. Each entry indicates whether an operator is supported (`Yes`), not supported (`No`), or not applicable (`N/A`) for the specified data type and backend. "N/A" is used where the operation does not logically apply to the data type, whereas "No" indicates that the operation could be supported but currently isn't.

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
| zeros_like        | Yes         | Yes        | No         | No          | No           | No           | No          | No           | No           |


---

> **Ready to dive in?**
> Head over to the [Getting Started](./usage/index.md) guide and generate your first module in minutes.

## 📜 License
