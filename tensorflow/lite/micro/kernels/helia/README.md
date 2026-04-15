<!-- mdformat off(b/169948621#comment2) -->

# General Info
NS-CMSIS-NN is a library containing kernel optimizations for Arm(R) Cortex(R)-M
processors, developed by Ambiq. To use NS-CMSIS-NN optimized kernels instead of reference kernels, add
`OPTIMIZED_KERNEL_DIR=helia` to the make command line. See examples below.

For more information about the optimizations, check out
[CMSIS-NN documentation](https://github.com/ARM-software/CMSIS-NN/blob/main/README.md),

## Example - FVP based on Arm Corstone-300 software.
In this example, the kernel conv unit test is built. For more information about
this specific target, check out the [Corstone-300 readme](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/cortex_m_corstone_300/README.md).

Downloaded NS-CMSIS-NN code is built:
```
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=helia TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 kernel_conv_test
```

# Build for speed or size
It is possible to build for speed or size. The size option may be required for a large model on an embedded system with limited memory. Where applicable, building for size would result in higher latency paired with a smaller scratch buffer, whereas building for speed would result in lower latency with a larger scratch buffer. Currently only transpose conv supports this.  See examples below.
Per‑kernel overrides – The global flag (GLOBAL_KERNEL_OPTIMIZE) sets the default for all kernels, but any individual kernel can be overridden by appending <KERNEL>_OPT=<SPEED|SIZE> to the make invocation (e.g. FC_OPT=SPEED, CONV_OPT=SIZE). Per‑kernel values always win over the global setting. 

## Example - building a static library with CMSIS-NN optimized kernels
More info on the target used in this example: https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/cortex_m_generic/README.md

Bulding for speed (default):
Note that speed is default so if leaving out GLOBAL_KERNEL_OPTIMIZE completely that will be the default.
```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 OPTIMIZED_KERNEL_DIR=helia GLOBAL_KERNEL_OPTIMIZE=SPEED microlite

```

Bulding for size:
```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 OPTIMIZED_KERNEL_DIR=helia GLOBAL_KERNEL_OPTIMIZE=SIZE microlite

```
Building for size, but overriding the FC kernel to be built for speed:

```
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 \
  OPTIMIZED_KERNEL_DIR=helia \
  GLOBAL_KERNEL_OPTIMIZE=SIZE \   # default all kernels to size‑optimized
  FC_OPT=SPEED \                  # but build fully‑connected for speed
  microlite
  
```
