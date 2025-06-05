# Getting Started with HeliosRT

Welcome to the HeliosRT Getting Started guide. HeliosRT, a derivative of TensorFlow Lite for Microcontrollers (TFLM), is specially optimized for Ambiq's Apollo platforms using NS-CMSIS-NN, enhancing performance on these devices. This guide outlines similarities with TFLM and provides detailed instructions on leveraging HeliosRT's capabilities through neuralSPOT. Read on to learn how to deploy TFLite models, run examples, and integrate HeliosRT into your development projects with ease.


## Latest Release w/ neuralSPOT

The latest release of HeliosRT is now available as a static module within the neuralSPOT framework. Explore the following options to effectively utilize HeliosRT with neuralSPOT:

### Option 1: Run a TFLite Model via Autodeploy

The `ns_autodeploy` tool is a Python command-line utility designed to streamline the process of compiling, flashing, and profiling TensorFlow Lite models. This tool also generates CSV and Excel files containing detailed profiling results and identifies any known sub-optimal network architectures.

```bash
ns_autodeploy --tflite-filename=mymodel.tflite --model-name mymodel
```
Learn more and access the tool [here](https://ambiqai.github.io/neuralSPOT/tools/index.html).

### Option 2: Run an Example in neuralSPOT

neuralSPOT includes a variety of AI examples tailored for Ambiq's Apollo platforms. Each example comes with its own README.md, providing detailed instructions on setup and execution. These examples can be found in the `examples` directory of the neuralSPOT repository.
Access the examples [here](https://ambiqai.github.io/neuralSPOT/examples/).

### Option 3: Create a Nest Application in neuralSPOT

Utilize the neuralSPOT makefile system to create custom applications, known as `nest` applications, which can operate on Ambiq's Apollo platforms. These self-contained applications are customizable and can include any number of static modules, such as HeliosRT.

```bash
make nest
```

For detailed makefile instructions, visit [this page](https://ambiqai.github.io/neuralSPOT/docs/makefile-details.html).


## Bleeding Edge w/ neuralSPOT

The bleeding edge version of HeliosRT is available as a neuralSPOT static module and can be integrated into your projects to utilize the latest features and optimizations. Follow these steps to build and configure the module:

### 1. Clone the HeliosRT Repository

Start by cloning the HeliosRT repository and checking out the latest commit:

```bash
git clone https://github.com/AmbiqAI/helios-rt
```

### 2. Build the NeuralSPOT Module

Navigate to the cloned directory and execute the build script. This script prepares both GCC and Arm Clang toolchain versions, which are automatically downloaded and configured. Specify a local toolchain by setting `$TARGET_TOOLCHAIN_ROOT` or adjust the toolchain used with the `build.sh` script's `$TOOLCHAIN` variable.

```bash
cd helios-rt
./neuralspot/build.sh
```

This process generates a static module in the `build` directory structured as follows:

```plaintext
build/
├── LICENSE
├── lib
│   ├── libtensorflow-microlite-cm4-armclang-debug.a
│   ├── libtensorflow-microlite-cm4-armclang-release-with-logs.a
│   ├── libtensorflow-microlite-cm4-armclang-release.a
│   ├── libtensorflow-microlite-cm4-gcc-debug.a
│   ├── libtensorflow-microlite-cm4-gcc-release-with-logs.a
│   ├── libtensorflow-microlite-cm4-gcc-release.a
│   ├── libtensorflow-microlite-cm55-armclang-debug.a
│   ├── libtensorflow-microlite-cm55-armclang-release-with-logs.a
│   ├── libtensorflow-microlite-cm55-armclang-release.a
│   ├── libtensorflow-microlite-cm55-gcc-debug.a
│   ├── libtensorflow-microlite-cm55-gcc-release-with-logs.a
│   └── libtensorflow-microlite-cm55-gcc-release.a
├── module.mk
├── signal
│   ├── micro
│   └── src
├── tensorflow
│   ├── compiler
│   └── lite
└── third_party
    ├── cmsis
    ├── flatbuffers
    ├── gemmlowp
    ├── kissfft
    ├── ns_cmsis_nn
    └── ruy
```

The `lib` directory contains static libraries for various configurations, and `module.mk` describes the module and its dependencies within neuralSPOT.

### 3. Integrate the Module into Your NeuralSPOT Project

To integrate the built module with your project, copy the `build` directory to the neuralSPOT project directory:

```bash
cp -r helios-rt/build neuralspot/extern/ns_tflm_bleeding_edge
```

### 4. Configure NeuralSPOT to Use the Bleeding Edge Module

Define the module version in your project's makefile settings to use the bleeding edge version:

```bash
$TF_VERSION := ns_tflm_bleeding_edge
```

Alternatively, specify this version directly when using the `ns_autodeploy` tool:

```bash
ns_autodeploy --tensorflow-version=ns_tflm_bleeding_edge
```

These steps ensure that your project utilizes the latest HeliosRT features, enhancing functionality and performance on supported Ambiq SoCs.


## Zephyr Integration

Coming soon...


## Building HeliosRT from Source

To integrate HeliosRT into AmbiqSuite or a third-party project, you can build TFLM from source. The following steps will guide you through the high-level process of building TFLM from source into a static library.

1. **Clone the HeliosRT repository and checkout the latest commit.**

```bash
git clone https://github.com/AmbiqAI/helios-rt
```

2. **(Optional) Open the repository as a VSCode devcontainer.**

This will install all the required dependencies and set up the environment for building TFLM.


3. **Configure the build environment.**

Be sure to configure the variables below to match your target architecture and toolchain.


```bash

cd helios-rt

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

TARGET_ARCH=cortex-m55  # one of cortex-m4+fp, cortex-m55
TOOLCHAIN=gcc  # one of gcc, armclang
BUILD_TYPE=release # one of debug, release, release-with-logs

TARGET=cortex_m_generic
OPTIMIZED_KERNEL=ambiq # <- use ambiq optimized kernels

```

4. **Download the third-party dependencies.**

```bash

readable_run make -f tensorflow/lite/micro/tools/make/Makefile \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    TOOLCHAIN=${TOOLCHAIN} \
    third_party_downloads

```

5. **Build the static library.**

```bash
readable_run make -f tensorflow/lite/micro/tools/make/Makefile \
    TARGET=${TARGET} \
    TARGET_ARCH="${TARGET_ARCH} \
    TOOLCHAIN=${TOOLCHAIN} \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL} \
    BUILD_TYPE=${BUILD} \
    microlite -j8
```

The static library will be generated to `gen/${TARGET}_${TARGET_ARCH}_${BUILD}_${OPTIM_KERNEL}_${TOOLCHAIN}/lib/libtensorflow-microlite.a`. The library can be linked into your project using the standard linker flags.

6. **(Optional) Generate HeliosRT tree for intellisense and debugging.**

```bash
python3 /tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  --makefile_options "TARGET=$TARGET TARGET_ARCH=$TARGET_ARCH OPTIMIZED_KERNEL_DIR=$OPTIM_KERNEL" \
  "gen/${TARGET}_${TARGET_ARCH}_${BUILD}_${OPTIM_KERNEL}_${TOOLCHAIN}"
```

This will generate `signal`, `tensorflow`, and `third_party` directories in the `gen/${TARGET}_${TARGET_ARCH}_${BUILD}_${OPTIM_KERNEL}_${TOOLCHAIN}` directory. These directories contain the source code for the module for intellisense and debugging.
