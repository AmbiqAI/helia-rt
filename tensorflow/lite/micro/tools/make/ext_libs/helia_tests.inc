# Helia-specific kernel test extensions. Loaded via the optional
# `-include $(MAKEFILE_DIR)/ext_libs/$(OPTIMIZED_KERNEL_DIR)_tests.inc`
# hook in tensorflow/lite/micro/kernels/Makefile.inc when
# OPTIMIZED_KERNEL_DIR=helia.

# Coverage for kernel paths that exist only in kernels/helia/ (e.g. int16
# variants of operators that the upstream reference implementation does
# not provide). Keeping these tests under kernels/helia/tests/ avoids
# patching the upstream test files for purely additive coverage.
MICROLITE_KERNEL_SIMPLE_TEST_SRCS += \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/helia/tests/hard_swish_int16_test.cc
