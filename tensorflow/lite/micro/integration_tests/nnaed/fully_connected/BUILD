# Description:
#   generated integration test for one specific kernel in a model.
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "generate_cc_arrays",
    "tflm_cc_library",
    "tflm_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

generate_cc_arrays(name = "generated_fully_connected0_model_data_cc",src = "fully_connected0.tflite",out = "fully_connected0_model_data.cc",)
generate_cc_arrays(name = "generated_fully_connected0_model_data_hdr",src = "fully_connected0.tflite",out = "fully_connected0_model_data.h",)
generate_cc_arrays(name = "generated_fully_connected1_model_data_cc",src = "fully_connected1.tflite",out = "fully_connected1_model_data.cc",)
generate_cc_arrays(name = "generated_fully_connected1_model_data_hdr",src = "fully_connected1.tflite",out = "fully_connected1_model_data.h",)
generate_cc_arrays(name = "generated_fully_connected2_model_data_cc",src = "fully_connected2.tflite",out = "fully_connected2_model_data.cc",)
generate_cc_arrays(name = "generated_fully_connected2_model_data_hdr",src = "fully_connected2.tflite",out = "fully_connected2_model_data.h",)
generate_cc_arrays(name = "generated_fully_connected3_model_data_cc",src = "fully_connected3.tflite",out = "fully_connected3_model_data.cc",)
generate_cc_arrays(name = "generated_fully_connected3_model_data_hdr",src = "fully_connected3.tflite",out = "fully_connected3_model_data.h",)

generate_cc_arrays(
  name = "generated_fully_connected0_input0_int8_test_data_cc",
  src = "fully_connected0_input0_int8.csv",
  out = "fully_connected0_input0_int8_test_data.cc",
)

generate_cc_arrays(
  name = "generated_fully_connected0_input0_int8_test_data_hdr",
  src = "fully_connected0_input0_int8.csv",
  out = "fully_connected0_input0_int8_test_data.h",
)

generate_cc_arrays(
  name = "generated_fully_connected0_golden_int8_test_data_cc",
  src = "fully_connected0_golden_int8.csv",
  out = "fully_connected0_golden_int8_test_data.cc",
)

generate_cc_arrays(
  name = "generated_fully_connected0_golden_int8_test_data_hdr",
  src = "fully_connected0_golden_int8.csv",
  out = "fully_connected0_golden_int8_test_data.h",
)
generate_cc_arrays(
  name = "generated_fully_connected1_input0_int8_test_data_cc",
  src = "fully_connected1_input0_int8.csv",
  out = "fully_connected1_input0_int8_test_data.cc",
)

generate_cc_arrays(
  name = "generated_fully_connected1_input0_int8_test_data_hdr",
  src = "fully_connected1_input0_int8.csv",
  out = "fully_connected1_input0_int8_test_data.h",
)

generate_cc_arrays(
  name = "generated_fully_connected1_golden_int8_test_data_cc",
  src = "fully_connected1_golden_int8.csv",
  out = "fully_connected1_golden_int8_test_data.cc",
)

generate_cc_arrays(
  name = "generated_fully_connected1_golden_int8_test_data_hdr",
  src = "fully_connected1_golden_int8.csv",
  out = "fully_connected1_golden_int8_test_data.h",
)
generate_cc_arrays(
  name = "generated_fully_connected2_input0_int8_test_data_cc",
  src = "fully_connected2_input0_int8.csv",
  out = "fully_connected2_input0_int8_test_data.cc",
)

generate_cc_arrays(
  name = "generated_fully_connected2_input0_int8_test_data_hdr",
  src = "fully_connected2_input0_int8.csv",
  out = "fully_connected2_input0_int8_test_data.h",
)

generate_cc_arrays(
  name = "generated_fully_connected2_golden_int8_test_data_cc",
  src = "fully_connected2_golden_int8.csv",
  out = "fully_connected2_golden_int8_test_data.cc",
)

generate_cc_arrays(
  name = "generated_fully_connected2_golden_int8_test_data_hdr",
  src = "fully_connected2_golden_int8.csv",
  out = "fully_connected2_golden_int8_test_data.h",
)
generate_cc_arrays(
  name = "generated_fully_connected3_input0_int8_test_data_cc",
  src = "fully_connected3_input0_int8.csv",
  out = "fully_connected3_input0_int8_test_data.cc",
)

generate_cc_arrays(
  name = "generated_fully_connected3_input0_int8_test_data_hdr",
  src = "fully_connected3_input0_int8.csv",
  out = "fully_connected3_input0_int8_test_data.h",
)

generate_cc_arrays(
  name = "generated_fully_connected3_golden_int8_test_data_cc",
  src = "fully_connected3_golden_int8.csv",
  out = "fully_connected3_golden_int8_test_data.cc",
)

generate_cc_arrays(
  name = "generated_fully_connected3_golden_int8_test_data_hdr",
  src = "fully_connected3_golden_int8.csv",
  out = "fully_connected3_golden_int8_test_data.h",
)

tflm_cc_library(
    name = "models_and_testdata",
    srcs = [
        "generated_fully_connected0_model_data_cc",
        "generated_fully_connected0_input0_int8_test_data_cc",
        "generated_fully_connected0_golden_int8_test_data_cc",
        "generated_fully_connected1_model_data_cc",
        "generated_fully_connected1_input0_int8_test_data_cc",
        "generated_fully_connected1_golden_int8_test_data_cc",
        "generated_fully_connected2_model_data_cc",
        "generated_fully_connected2_input0_int8_test_data_cc",
        "generated_fully_connected2_golden_int8_test_data_cc",
        "generated_fully_connected3_model_data_cc",
        "generated_fully_connected3_input0_int8_test_data_cc",
        "generated_fully_connected3_golden_int8_test_data_cc",
    ],
    hdrs = [
        "generated_fully_connected0_model_data_hdr",
        "generated_fully_connected0_input0_int8_test_data_hdr",
        "generated_fully_connected0_golden_int8_test_data_hdr",
        "generated_fully_connected1_model_data_hdr",
        "generated_fully_connected1_input0_int8_test_data_hdr",
        "generated_fully_connected1_golden_int8_test_data_hdr",
        "generated_fully_connected2_model_data_hdr",
        "generated_fully_connected2_input0_int8_test_data_hdr",
        "generated_fully_connected2_golden_int8_test_data_hdr",
        "generated_fully_connected3_model_data_hdr",
        "generated_fully_connected3_input0_int8_test_data_hdr",
        "generated_fully_connected3_golden_int8_test_data_hdr",
    ],
)

tflm_cc_test(
    name = "integration_test",
    srcs = [
        "integration_tests.cc",
    ],
    deps = [
        ":models_and_testdata",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_resource_variable",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "//python/tflite_micro:python_ops_resolver",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)
