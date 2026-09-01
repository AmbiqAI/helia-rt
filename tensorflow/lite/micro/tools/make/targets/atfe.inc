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
# ==============================================================================
#
# Toolchain-level shim for the Arm Toolchain for Embedded (ATfE), an
# LLVM/Clang bare-metal toolchain. Included from the cortex_m_*_makefile.inc
# (or cortex_m_corstone_300_atfe.inc) ATfE branches so the toolchain-shared
# bits (warning suppressions, llvm-objcopy, post-link `.bin` rule,
# test-only -O0 workaround) live in one place.
#
# Optimization-level note: ATfE clang 22 deprecated -Ofast and rejects it
# under -Werror. `-O3 -ffast-math` was the documented replacement, but
# `-ffast-math` enables `-funsafe-math-optimizations` /
# `-ffinite-math-only` / `-fassociative-math`, which alters the precision
# of `std::frexp` + `TfLiteRound(q * (1LL << 31))` inside
# `QuantizeMultiplier()`. For per-channel int16x8 kernels (e.g.
# `transpose_conv`), that perturbs the per-channel `output_multiplier[]`
# enough to drift several LSBs in the quantized output and trip kernel
# unit tests. Clang's `-O3` also miscompiles a few activation kernels
# (e.g. ELU) versus GCC. We therefore inherit the upstream non-armclang
# defaults of CORE=-Os, KERNEL=-O2, THIRD_PARTY_KERNEL=-O2 (no override
# needed -- upstream's `else` fall-through already produces these
# values for TOOLCHAIN=atfe).

# ATfE clang shares armclang's diagnostic for inf/nan use under fast-math
# options.
#
# Status of these two lines, stated accurately because the old wording ("keep
# -Werror builds green") was wrong and the same wrong assumption is what hid
# issue #228: NO global `-Werror` is set anywhere in this build -- not in
# tools/make/Makefile, not in any targets/*.inc -- so `-Wno-error=` currently
# downgrades nothing and these are no-ops. They are kept as forward
# compatibility for a build that does enable `-Werror`, and because ATfE's
# optimization levels (CORE=-Os, KERNEL=-O2, THIRD_PARTY_KERNEL=-O2 -- see the
# note above) carry no `-ffast-math`, so if this diagnostic ever fires under
# ATfE it means something different than it does under armclang.
#
# Contrast with armclang, which DOES use `-Ofast`: tools/make/Makefile
# deliberately does the opposite there and promotes this exact diagnostic with
# `-Werror=nan-infinity-disabled`, because under `-Ofast` it is a precise
# oracle for fast-math having been re-enabled by accident (issue #228).
CXXFLAGS += -Wno-error=nan-infinity-disabled
CCFLAGS  += -Wno-error=nan-infinity-disabled

# ns-cmsis-nn (and other GCC-targeted upstream sources) use GCC-only
# attributes such as __attribute__((optimize(...))). Clang ignores these and
# warns. Same status as the pair above: no global `-Werror` is set today, so
# this is a no-op kept for forward compatibility rather than something
# currently holding the build green.
CXXFLAGS += -Wno-error=unknown-attributes
CCFLAGS  += -Wno-error=unknown-attributes

# ATfE / clang 22 miscompiles `BuildSimpleModelWithSubgraphsAndWhile()` in
# `test_helpers.cc` at any optimization level >= -O1 on cortex-m55+nomve:
# the resulting flatbuffer puts `kernel_while_test` into an infinite loop
# inside `MicroInterpreter::Invoke()`. Compiling this single test-only
# translation unit at -O0 dodges the bug without affecting runtime perf
# (test_helpers.cc is only linked into kernel/integration tests).
$(CORE_OBJDIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/test_helpers.o: CORE_OPTIMIZATION_LEVEL := -O0

# ATfE ships llvm-objcopy under <root>/bin (no target prefix). Override
# the OBJCOPY variable and the `.bin` post-link rule so the upstream
# `else` branch (which uses $(TARGET_TOOLCHAIN_PREFIX)objcopy) does not
# execute for ATfE builds.
OBJCOPY := $(TARGET_TOOLCHAIN_ROOT)llvm-objcopy
$(BINDIR)%.bin: $(BINDIR)%
	$(Q)$(MKDIR_P) $(dir $@)
	$(Q)$(OBJCOPY) $< $@ -O binary
