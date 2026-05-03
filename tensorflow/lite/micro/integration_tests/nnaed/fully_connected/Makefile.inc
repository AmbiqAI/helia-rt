integration_tests_nnaed_fully_connected_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/nnaed/fully_connected/fully_connected0.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/nnaed/fully_connected/fully_connected1.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/nnaed/fully_connected/fully_connected2.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/nnaed/fully_connected/fully_connected3.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/nnaed/fully_connected/fully_connected0_input0_int8.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/nnaed/fully_connected/fully_connected0_golden_int8.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/nnaed/fully_connected/fully_connected1_input0_int8.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/nnaed/fully_connected/fully_connected1_golden_int8.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/nnaed/fully_connected/fully_connected2_input0_int8.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/nnaed/fully_connected/fully_connected2_golden_int8.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/nnaed/fully_connected/fully_connected3_input0_int8.csv \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/nnaed/fully_connected/fully_connected3_golden_int8.csv \

integration_tests_nnaed_fully_connected_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/integration_tests/nnaed/fully_connected/integration_tests.cc  \
$(TENSORFLOW_ROOT)python/tflite_micro/python_ops_resolver.cc \


integration_tests_nnaed_fully_connected_HDR := \
$(TENSORFLOW_ROOT)python/tflite_micro/python_ops_resolver.h \


$(eval $(call microlite_test,integration_tests_nnaed_fully_connected_test,\
$(integration_tests_nnaed_fully_connected_SRCS),$(integration_tests_nnaed_fully_connected_HDR),$(integration_tests_nnaed_fully_connected_GENERATOR_INPUTS)))