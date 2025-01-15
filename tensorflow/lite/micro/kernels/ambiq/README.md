# Ambiq

This directory contains custom kernel operations for TFLM optimized for Ambiq SoCs. This is intended to supplement CMSIS-NN optimized kernels.

## Target SoCs

* Apollo510 w/ Cortex-M55

## Completed Operations

* Tanh
* Logistic
* Leaky ReLU
* Pad

## In Progress Operations

* Sub: Should be essentially duplicate of Add
* AvgPool: Already implemented but we might be able to further optimize core loop
* Conv2D: Already implemented but look to optimize 1D convolutions w/ dilations


## Notes

### Testing full tflite models.

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
