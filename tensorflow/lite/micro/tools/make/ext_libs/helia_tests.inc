# Helia-specific kernel test extensions. Loaded via the optional
# `-include $(MAKEFILE_DIR)/ext_libs/$(OPTIMIZED_KERNEL_DIR)_tests.inc`
# hook in tensorflow/lite/micro/kernels/Makefile.inc when
# OPTIMIZED_KERNEL_DIR=helia.

# Coverage for kernel paths that exist only in kernels/helia/ (e.g. int16
# variants of operators that the upstream reference implementation does
# not provide). Keeping these tests under kernels/helia/tests/ avoids
# patching the upstream test files for purely additive coverage.
MICROLITE_KERNEL_SIMPLE_TEST_SRCS += \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/helia/tests/hard_swish_int16_test.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/helia/tests/quantize_zero_point_test.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/helia/tests/transpose_conv_optional_bias_test.cc

# Executed floating-point coverage (helia-rt#227). The shared upstream kernel
# tests already run float32 and float16 goldens for the helia dispatch, but
# nothing there feeds a non-finite value to a float kernel, and the shared
# float16 LSTM case uses state_dimension == 2, which is too small to mix the
# MVE body with the scalar tail. These tests close both gaps:
#   float_activation_edge_test  - tanh/logistic NaN and +/-Inf propagation,
#                                 float16 goldens inside the stable table
#                                 window, and sign/saturation-only assertions
#                                 outside it (ns#314 / ns#303 classes).
#   float_elementwise_edge_test - ADD/MUL NaN propagation (ns#333 class).
#   float_lstm_tail_lane_test   - UNIDIRECTIONAL_SEQUENCE_LSTM with
#                                 state_dimension == 10, asserting that the
#                                 vector lanes and the scalar tail lanes of one
#                                 gate agree (ns#315 class).
#
# More than one of these is expected to fail on the current ns-cmsis-nn pin.
# `make test` walks MICROLITE_TEST_TARGETS in list order, so a failing binary
# would stop every later one from running at all; tools/ci_build/test_helia.sh
# therefore runs the suite with `make -k`. Do not rely on list order to decide
# which defect classes get observed.
MICROLITE_KERNEL_SIMPLE_TEST_SRCS += \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/helia/tests/float_activation_edge_test.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/helia/tests/float_elementwise_edge_test.cc \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/helia/tests/float_lstm_tail_lane_test.cc
