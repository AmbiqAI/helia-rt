# Currently only need cmsis_nn.inc
MICROLITE_CC_HDRS += \
  tensorflow/lite/micro/kernels/ambiq/ambiq_common.h


MICROLITE_CC_SRCS += \
  tensorflow/lite/micro/kernels/ambiq/arm_fully_connected_per_channel_s16.cc \
  tensorflow/lite/micro/kernels/ambiq/arm_fully_connected_wrapper_s16.cc \
  tensorflow/lite/micro/kernels/ambiq/arm_nn_vec_mat_mult_t_per_ch_s16.cc \
