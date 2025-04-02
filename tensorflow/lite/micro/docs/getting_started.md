# Getting Started with NS-TFLM

Welcome to the NS-TFLM Getting Started guide. NS-TFLM, a derivative of TensorFlow Lite for Microcontrollers (TFLM), is specially optimized for Ambiq's Apollo platforms using NS-CMSIS-NN, enhancing performance on these devices. This guide outlines similarities with TFLM and provides detailed instructions on leveraging NS-TFLM's capabilities through neuralSPOT. Read on to learn how to deploy TFLite models, run examples, and integrate NS-TFLM into your development projects with ease.


## Latest Release w/ neuralSPOT

The latest release of NS-TFLM is packaged as a neuralSPOT static module in neuralSPOT. Refer to the following three options to use NS-TFLM in neuralSPOT:

**Option 1. [Run a TFLite model via autodeploy](https://ambiqai.github.io/neuralSPOT/tools/index.html)**: ns_autodeploy is a python command line tool that automatically compiles, flashes, and profiles a supplied TFLite model. The autodeploy tool produces csv and excel files with the profiling results and also detects for known sub-optimal network architectures.

```bash
ns_autodeploy --tflite-filename=mymodel.tflite --model-name mymodel
```

**Option 2. [Run an example in neuralSPOT](https://ambiqai.github.io/neuralSPOT/examples/basic_tf_stub/index.html)**: neuralSPOT comes included with several AI examples that can run across Ambiq's Apollo platforms. Each example includes a README.md file that describes how to run the example. The examples are located in the `examples` directory of the neuralSPOT repository.

**Option 3. [Create a nest application in neuralSPOT](https://ambiqai.github.io/neuralSPOT/docs/makefile-details.html)**: The neuralSPOT makefile system allows you to create a custom application that can run on Ambiq's Apollo platforms. A `nest` application is a self-contained application that can be built and flashed to the target device. The `nest` application can include any number of static modules, including NS-TFLM. The `nest` application can be created using the `make nest` command line tool.


## Bleeding Edge w/ neuralSPOT

The bleeding edge version of NS-TFLM can be built and packaged as a neuralSPOT static module. The following steps will guide you through creating the static module and adding it into neuralSPOT. Once configured, the above three options can be used to run a TFLite model.

1. **Clone the NS-TFLM repository and checkout the latest commit.**

```bash
git clone https://github.com/AmbiqAI/ns-tflm
```

2. **Build the static neuralSPOT module.**

The included neuralspot build script will generate both gcc and armclang toolchain versions. These toolchains are automatically downloaded and configured. To use a local toolchain, you can specify `$TARGET_TOOLCHAIN_ROOT`. If only one toolchain is required, you can modify the `build.sh` scripts `$TOOLCHAIN` variable to the desired toolchain (e.g `TOOLCHAINS=("gcc" "armclang")`)

```bash
cd ns-tflm
./neuralspot/build.sh
```

This will generate a static module in the `build` directory with the following structure:

```bash
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

The `lib` directory contains the static libraries for different configurations. The `module.mk` is a neuralSPOT module makefile that describes the module and its dependencies. The signal, tensorflow, and third_party directories contain the source code for the module for intellisense and debugging.

3. **Add the static module to your neuralSPOT project.**

Copy the `build` directory to your neuralSPOT project directory. Assuming neuralSPOT is installed alongside the ns-tflm repository, the following command will copy the build directory to your neuralSPOT project directory.

```bash
cp -r ns-tflm/build neuralspot/extern/ns_tflm_bleeding_edge
```

4. **Configure neuralSPOT to use the static module.**

You can define `$TF_VERSION := ns_tflm_bleeding_edge` in `make/local_overrides.mk` or as makefile argument. For autodeploy, you can pass the commandline argument `--tensorflow-version=ns_tflm_bleeding_edge` to the `ns_autodeploy` command.


## Building TFLM from Source

To integrate NS-TFLM into AmbiqSuite or a third-party project, you can build TFLM from source. The following steps will guide you through the high-level process of building TFLM from source into a static library.

1. **Clone the NS-TFLM repository and checkout the latest commit.**

```bash
git clone https://github.com/AmbiqAI/ns-tflm
```

2. **(Optional) Open the repository as a VSCode devcontainer. This will install all the required dependencies and set up the environment for building TFLM.**


3. **Configure the build environment**

```bash

cd ns-tflm

source tensorflow/lite/micro/tools/ci_build/helper_functions.sh

TARGET_ARCH=cortex-m55  # one of cortex-m4+fp, cortex-m55
TOOLCHAIN=gcc  # one of gcc, armclang
BUILD_TYPE=release # one of debug, release, release-with-logs

TARGET=cortex_m_generic
OPTIMIZED_KERNEL=ambiq

```

4. **Download the third-party dependencies.**

```bash

readable_run make -f tensorflow/lite/micro/tools/make/Makefile \
    CO_PROCESSOR=${CO_PROCESSOR} \
    OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL} \
    TARGET=${TARGET} \
    TARGET_ARCH=${TARGET_ARCH} \
    TOOLCHAIN=${TOOLCHAIN} \
    third_party_downloads

```

5. **Build the static library.**

```bash
readable_run make -f tensorflow/lite/micro/tools/make/Makefile \
    TARGET="$TARGET" \
    TARGET_ARCH="$TARGET_ARCH" \
    TOOLCHAIN="$TOOLCHAIN" \
    OPTIMIZED_KERNEL_DIR="$OPTIMIZED_KERNEL" \
    BUILD_TYPE="$BUILD" \
    microlite -j8
```

The static library will be generated to `gen/${TARGET}_${TARGET_ARCH}_${BUILD}_${OPTIM_KERNEL}_${TOOLCHAIN}/lib/libtensorflow-microlite.a`. The library can be linked into your project using the standard linker flags.

6. **(Optional) Generate NS-TFLM tree for intellisense and debugging.**

```bash
python3 /tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  --makefile_options "TARGET=$TARGET TARGET_ARCH=$TARGET_ARCH OPTIMIZED_KERNEL_DIR=$OPTIM_KERNEL" \
  "gen/${TARGET}_${TARGET_ARCH}_${BUILD}_${OPTIM_KERNEL}_${TOOLCHAIN}"
```

This will generate `signal`, `tensorflow`, and `third_party` directories in the `gen/${TARGET}_${TARGET_ARCH}_${BUILD}_${OPTIM_KERNEL}_${TOOLCHAIN}` directory. These directories contain the source code for the module for intellisense and debugging.
