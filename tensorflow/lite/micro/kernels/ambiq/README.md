# Ambiq

This directory contains custom kernel operations for TFLM optimized for Ambiq SoCs. This is intended to supplement CMSIS-NN optimized kernels.

## Target SoCs

* Apollo3
* Apollo4
* Apollo4 Plus w/ Cortex-M4
* Apollo510 w/ Cortex-M55

## Operator Support Matrix

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


## Notes

### Testing full tflite models

We can test full tflite models in 1 of 2 ways:

1. Split the operations into separate tflite and generated random stimulus data for each operation. This is the most straightforward way to test each operation in isolation.

To generate tests for leaky_relu on a model, we can run the following command:

```bash

bazel build tensorflow/lite/micro/integration_tests:generate_per_layer_tests

bazel-bin/tensorflow/lite/micro/integration_tests/generate_per_layer_tests \
  --input_tflite_file=my-model.tflite \
  --output_dir=tensorflow/lite/micro/integration_tests/my-model/conv

```

A couple things to note:
- The input_tflite_file should be the full model that you want to test.
- The output_dir should be a directory where the test files will be generated.
- The final part of the output_dir should be the operation you are testing. This is important because the test runner will look for tests in this directory.
- Run the command from the root of the tensorflow directory so that we can use a relative path for output_dir which gets baked into the test.


To run the tests, you can run the following command:

```bash

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

TOOLCHAIN=gcc
TARGET=cortex_m_corstone_300
TARGET_ARCH=cortex-m55
CO_PROCESSOR=ambiq
OPTIMIZED_KERNEL_DIR=cmsis_nn

readable_run make  -j$(nproc) -f tensorflow/lite/micro/tools/make/Makefile \
    CO_PROCESSOR=${CO_PROCESSOR} \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    TOOLCHAIN=${TOOLCHAIN} \
    test_integration_tests_my-model_conv_test

```

2. Convert tflite into header along with input data and run the full model. This is more complex but allows us to test the full model.
