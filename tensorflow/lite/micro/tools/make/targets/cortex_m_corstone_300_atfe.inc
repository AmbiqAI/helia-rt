# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Helia-rt: Arm Toolchain for Embedded (ATfE / LLVM-Clang) support for the
# Corstone-300 target. This file is included from
# targets/cortex_m_corstone_300_makefile.inc when TOOLCHAIN=atfe.
#
# Lives outside the upstream block so that the inline drift in
# cortex_m_corstone_300_makefile.inc is reduced to a single `include` line
# plus the `retarget.c` toolchain guard. See helia/patches/inline_drift.md.

TARGET_DEFAULT_TOOLCHAIN_ROOT := $(DOWNLOADS_DIR)/arm_toolchain_embedded/bin/
TARGET_TOOLCHAIN_ROOT := $(TARGET_DEFAULT_TOOLCHAIN_ROOT)
ifeq ($(TARGET_TOOLCHAIN_ROOT), $(TARGET_DEFAULT_TOOLCHAIN_ROOT))
  DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_toolchain_embedded_download.sh $(DOWNLOADS_DIR) $(TENSORFLOW_ROOT))
  ifneq ($(DOWNLOAD_RESULT), SUCCESS)
    $(error Something went wrong with the ATfE download: $(DOWNLOAD_RESULT))
  endif
endif

CXX_TOOL := clang++
CC_TOOL  := clang
AR_TOOL  := llvm-ar
# Use the clang driver as the linker so it pulls in picolibc and the
# picolibc crt0/semihost libraries via -l flags below; LDFLAGS pass
# through with -Wl,...

# ATfE rejects -mfpu=auto; let -mcpu drive FPU/MVE feature selection.
# Disable MVE auto-vectorization on cortex-m55: clang 22's MVE vectorizer
# miscompiles `(int)int8_value * float_scale` loops by emitting
# `vcvt.f32.s32 qN, qN, #12` (i.e., dividing by 2^12) when the correct
# shift count is -log2(scale). It also matches the picolibc multilib
# selection (armv8.1m.main_hard_fpdp_nomve_unaligned).
ATFE_MCPU := $(MCPU_OPTION)
ifneq ($(filter cortex-m55%,$(TARGET_ARCH)),)
  ATFE_MCPU := $(MCPU_OPTION)+nomve
endif
FLAGS_ATFE = \
  --target=arm-none-eabi \
  -mcpu=$(ATFE_MCPU) \
  -fshort-enums

CXXFLAGS += $(FLAGS_ATFE)
CCFLAGS += $(FLAGS_ATFE)

# Linker setup for ATfE / picolibc on Corstone-300.
#
# `-nostartfiles` suppresses clang's automatic crt0.o (whose default
# `_start` clobbers the CMSIS Reset_Handler vector). The CMSIS
# `Reset_Handler` (from startup_ARMCM55.c) remains the ELF entry and
# invokes picolibc's `_start` via `__PROGRAM_START()`. We then pull in
# the semihost-flavored picolibc startup (`libcrt0-semihost.a`) and
# the semihost syscall stubs (`libsemihost.a`, which also provides
# `stderr` / `stdin` / `stdout` consumed by debug_log.cc). `-lc -lm`
# keep picolibc itself in the link group.
#
# The ATfE-specific linker script adds a PT_TLS phdr (required by
# picolibc's TLS errno) plus PROVIDE() aliases for the symbols crt0
# and CMSIS' clang startup expect (`__data_*`, `__bss_*`,
# `__stack_limit`, `__tls_base`, `__arm32_tls_tcb_offset`, ...).
ATFE_LINKER_SCRIPT := $(TENSORFLOW_ROOT)tensorflow/lite/micro/cortex_m_corstone_300/corstone_300_atfe.ld

# Note: we intentionally do NOT pass `-nostdlib++`. TFLM uses a few
# libc++ symbols (`operator new`, `std::__throw_*`, ...) that ATfE
# provides via the multilib's `libc++.a` / `libc++abi.a`. clang++
# auto-links these and `--gc-sections` strips them when unused, so
# the TCM/DDR cost is zero for tests that don't reference libc++.
LDFLAGS += \
  -nostartfiles \
  -T $(ATFE_LINKER_SCRIPT) \
  -Wl,-Map=gen/$(TARGET).map,--cref \
  -Wl,--gc-sections \
  -Wl,--entry=Reset_Handler

# Order matters: -lcrt0-semihost provides _start; -lc / -lm reference
# _write/_read/etc. from -lsemihost, so semihost must come last (or be
# inside a group). Use -Wl,--start-group to be safe with picolibc's
# cross-references.
MICROLITE_LIBS += -Wl,--start-group -lcrt0-semihost -lc -lm -lsemihost -Wl,--end-group

ldflags_to_remove = -Wl,--fatal-warnings
LDFLAGS := $(filter-out $(ldflags_to_remove),$(LDFLAGS))

# flatbuffer_utils_test.cc exercises std::stoi / std::stof / std::stod from
# libc++.  Under ATfE 22 + picolibc on Cortex-M (semihosting), libc++'s
# `__as_integer`/`__as_float` helpers read/write TLS `errno` via
# `__aeabi_read_tp` and the LLVM TLS GD->LE relaxations end up hanging the
# core when invoked from this test (the wrappers parse correctly but never
# return).  No other TFLM test relies on these C++ numeric-conversion
# wrappers, and `tflite::FlexbufferWrapper` itself is also exercised by the
# kernel tests that use flexbuffer-encoded custom op data, so dropping this
# stand-alone test for the atfe toolchain keeps full coverage of the runtime
# code paths while avoiding the toolchain-specific hang.
#
# Lives here (not in sources.inc) because the filter only applies when
# TOOLCHAIN=atfe. sources.inc is included before the target makefile, but
# MICROLITE_TEST_SRCS is consumed after the target makefile loads, so a
# late filter-out is safe.
MICROLITE_TEST_SRCS := $(filter-out \
  $(TENSORFLOW_ROOT)tensorflow/lite/micro/flatbuffer_utils_test.cc, \
  $(MICROLITE_TEST_SRCS))
