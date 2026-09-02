/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// helia-rt (issue #239): fault diagnostics for the Corstone-300 test runtime.
//
// Why this file exists
// --------------------
// The CMSIS startup this target links (Cortex_DFP/Device/ARMCM55/Source/
// startup_ARMCM55.c, added by targets/cortex_m_corstone_300_makefile.inc)
// defines every fault vector as `while(1);`:
//
//     void HardFault_Handler(void) __attribute__ ((weak));
//     void HardFault_Handler(void) { while(1); }
//     void MemManage_Handler(void) __attribute__((weak, alias("Default_Handler")));
//     ... and Default_Handler(void) { while(1); }
//
// So on this target a fault and a genuine infinite loop look identical from
// the outside: the FVP keeps running and prints nothing more. That is exactly
// the state ConvTest.Float16MultiChannelGolden and
// TransposeConvTest.Float16Stride1Golden are in under cortex-m55 + ATfE --
// '[ RUN ]' is printed and nothing follows -- and it is why the cause is not
// currently knowable from CI. These strong definitions replace the weak ones
// so a fault prints its cause and stops the simulation instead.
//
// Scope: test runtime only. This file is wired in through
// MICROLITE_TEST_RUNTIME_SRCS, which the makefile links into test/example
// binaries and deliberately keeps out of libtensorflow-microlite.a, so it
// cannot reach a release library or pack artifact. It is compiled for every
// toolchain on this target (gcc/newlib, ATfE/picolibc, armclang).
//
// Architecture: targets/cortex_m_corstone_300_makefile.inc adds this file for
// v7-M and later only, because Armv6-M (the cortex-m0 leg of helia_build.yml)
// has neither the configurable fault vectors nor CFSR/HFSR/MMFAR/BFAR. The
// trampolines below are nonetheless written in the v6-M instruction subset so
// that an accidental future inclusion is a no-op rather than a build break.
//
// Output and exit path
// --------------------
// Everything here is raw semihosting (BKPT 0xAB), not DebugLog/MicroPrintf:
//
//  - DebugLog on this target is the generic tensorflow/lite/micro/debug_log.cc
//    implementation, i.e. vfprintf(stderr, ...). stdio is not safe to call
//    from handler context -- it can take locks and allocate, and the fault may
//    well have happened inside libc -- so the report would risk faulting
//    again and giving us the same silence we are trying to remove.
//  - Semihosting is enabled for every toolchain on this target: the harness
//    testing/test_with_arm_corstone_300.sh passes
//    '-C cpu0.semihosting-enable=1' unconditionally, and the FVP traps
//    BKPT 0xAB regardless of which C library the binary was linked against.
//    SYS_WRITE0 output goes to the FVP's stdout, which the harness captures.
//  - The exit is SYS_EXIT rather than a return, so a fault ends the run.
//    That turns the hang into a FAIL the harness can see (no pass string in
//    the log) instead of something only the per-binary timeout can end.
//
// Deliberately not overridden: Default_Handler and the interrupt vectors that
// alias it. Unlike the fault handlers, Default_Handler has a NON-weak
// definition in the CMSIS startup file, so a second definition here would be
// a duplicate-symbol link error on every toolchain.
//
// Note on which handler actually runs: MemManage, BusFault and UsageFault are
// disabled at reset (SCB->SHCSR), so in practice these faults escalate to
// HardFault. That loses nothing -- the escalation sets HFSR.FORCED and the
// original cause still lands in CFSR, which is printed either way. The three
// specific handlers are provided so the report stays correct if something
// later enables them.
//
// Known limitation: a main-stack overflow is not reportable from here.
// Everything this file owns that could need memory is at file scope (the
// report buffer and the semihosting exit block) precisely so the reporter
// itself adds no stack, but that is not sufficient, and switching MSP to a
// dedicated scratch stack inside the trampoline would not fix it either:
//  - If MSPLIM is armed and MSP overflows, the exception ENTRY stacking for
//    this handler hits the limit too, which is a lockup. Control never
//    reaches the trampoline, so no instruction it could execute helps.
//  - If MSPLIM is 0, which is the state the CMSIS startup on this target
//    leaves it in, no STKOF is raised at all: the overflow silently walks
//    into whatever is below the stack and surfaces later, if ever, as some
//    other fault, which this file does report.
// Reporting the armed case needs the limit registers reset before the
// stacking, i.e. work in the startup file, not here. It is deliberately out
// of scope: it would trade a specific, currently untriggerable case for
// changes to code shared with every non-test build of this target.

#include <stdint.h>

namespace {

// System Control Block fault status registers, addressed directly rather than
// through the CMSIS headers: this file has to build identically under three
// toolchains and must not depend on the device header having been configured.
// Values are from the Armv8-M architecture reference manual.
constexpr uint32_t kSCB_CFSR = 0xE000ED28;   // Configurable Fault Status
constexpr uint32_t kSCB_HFSR = 0xE000ED2C;   // HardFault Status
constexpr uint32_t kSCB_MMFAR = 0xE000ED34;  // MemManage Fault Address
constexpr uint32_t kSCB_BFAR = 0xE000ED38;   // BusFault Address

// Semihosting operation numbers (Arm semihosting specification).
constexpr uint32_t kSysWrite0 = 0x04;
constexpr uint32_t kSysExit = 0x18;
constexpr uint32_t kSysExitExtended = 0x20;

// Semihosting exit reason codes.
constexpr uint32_t kApplicationExit = 0x20026;
constexpr uint32_t kRunTimeErrorUnknown = 0x20023;

// Offsets, in words, into the exception stack frame pushed by the core.
// Layout: r0, r1, r2, r3, r12, lr, return address, xPSR. An extended
// (floating-point) frame appends to this, it does not reorder it.
constexpr uint32_t kFrameLrIndex = 5;
constexpr uint32_t kFramePcIndex = 6;

inline uint32_t ReadRegister(uint32_t address) {
  return *reinterpret_cast<volatile uint32_t*>(address);
}

// One semihosting call. r0 holds the operation, r1 the argument (a pointer
// for SYS_WRITE0 and SYS_EXIT_EXTENDED, an immediate reason code for the
// 32-bit form of SYS_EXIT).
inline uint32_t SemihostingCall(uint32_t operation, uint32_t argument) {
  register uint32_t r0 __asm__("r0") = operation;
  register uint32_t r1 __asm__("r1") = argument;
  __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory", "cc");
  return r0;
}

void WriteString(const char* text) {
  SemihostingCall(kSysWrite0, reinterpret_cast<uint32_t>(text));
}

// Minimal, allocation-free '0x' + 8 hex digits formatter. Writes into the
// caller's buffer and returns the position after the last character written.
char* AppendHex(char* out, uint32_t value) {
  static const char kDigits[] = "0123456789abcdef";
  *out++ = '0';
  *out++ = 'x';
  for (int shift = 28; shift >= 0; shift -= 4) {
    *out++ = kDigits[(value >> shift) & 0xF];
  }
  return out;
}

char* AppendString(char* out, const char* text) {
  while (*text != '\0') {
    *out++ = *text++;
  }
  return out;
}

char* AppendField(char* out, const char* label, uint32_t value) {
  out = AppendString(out, label);
  out = AppendHex(out, value);
  *out++ = ' ';
  return out;
}

// File scope, not a local: the fault may well be a stack overflow, so the
// report must not need another ~200 bytes of the stack that just ran out.
// Nothing here is re-entrant, which is fine -- this path ends the program.
char g_report[256];

}  // namespace

extern "C" {

// Called from the naked trampolines below with:
//   frame       - the exception stack frame, on whichever stack was active
//   exc_return  - the EXC_RETURN value the core put in LR on entry
//   exception   - which vector we came in through (see kExceptionNames)
//
// Prints one line and terminates the simulation. Never returns.
__attribute__((used)) void HeliaCorstoneReportFault(const uint32_t* frame,
                                                    uint32_t exc_return,
                                                    uint32_t exception) {
  static const char* const kExceptionNames[] = {"HardFault", "MemManage",
                                                "BusFault", "UsageFault"};
  const char* name = (exception < 4) ? kExceptionNames[exception] : "Unknown";

  char* out = g_report;
  out = AppendString(out, "FAULT: ");
  out = AppendField(out, "HFSR=", ReadRegister(kSCB_HFSR));
  out = AppendField(out, "CFSR=", ReadRegister(kSCB_CFSR));
  out = AppendField(out, "PC=", frame[kFramePcIndex]);
  out = AppendField(out, "LR=", frame[kFrameLrIndex]);
  out = AppendField(out, "MMFAR=", ReadRegister(kSCB_MMFAR));
  out = AppendField(out, "BFAR=", ReadRegister(kSCB_BFAR));
  out = AppendField(out, "EXC_RETURN=", exc_return);
  out = AppendField(out, "SP=", reinterpret_cast<uint32_t>(frame));
  out = AppendString(out, "EXC=");
  out = AppendString(out, name);
  *out++ = '\n';
  *out = '\0';
  WriteString(g_report);

  // Terminate with a non-zero status so the run ends in a failure rather than
  // in the hang this file exists to remove. SYS_EXIT_EXTENDED is the form
  // that can carry an exit code on a 32-bit target; fall back to the plain
  // 32-bit SYS_EXIT with a run-time-error reason if it is not honoured, and
  // to a spin only if neither call ends the simulation (the per-binary
  // timeout in the harness bounds that last case).
  //
  // static, for the same reason as g_report: SYS_EXIT_EXTENDED takes a
  // pointer to a two-word block, and a fault that ran the stack out must not
  // need two more words of it to say so.
  static uint32_t exit_block[2] = {kApplicationExit, 1};
  SemihostingCall(kSysExitExtended, reinterpret_cast<uint32_t>(exit_block));
  SemihostingCall(kSysExit, kRunTimeErrorUnknown);
  while (true) {
  }
}

// Naked trampolines. They must be naked: a compiler-generated prologue would
// move SP before we can capture the frame, and could clobber the EXC_RETURN
// value in LR that tells us which stack the frame is on. Bit 2 of EXC_RETURN
// is 0 for MSP and 1 for PSP.
//
// Every instruction here is in the Armv6-M subset, so the body assembles for
// any Cortex-M even though the makefile only adds this file for v7-M and
// later (see the header comment). Two consequences worth stating:
//  - `lsls r1, r1, #29` + `bmi`, not `tst r1, #4` + `bne`. TST with an
//    immediate is a 32-bit Armv7-M encoding with no v6-M form; the shift
//    moves EXC_RETURN bit 2 into the N flag and is a 16-bit T1 instruction
//    on every M-profile core. r1 is a scratch copy of LR, so destroying it
//    costs nothing.
//  - The tail `b` to the reporter is a plain branch. On v6-M that is a
//    +/-2 KB range; if this file is ever genuinely built for a v6-M target
//    the link may need a veneer or an ldr/bx pair. Nothing is affected today
//    because the makefile does not add it there.
//
// The exception index is passed as a bare immediate rather than a string
// pointer so that no literal pool is needed inside the naked body, which
// keeps the sequence identical across the GNU and LLVM assemblers.
#define HELIA_FAULT_TRAMPOLINE(handler, index)         \
  __attribute__((naked)) void handler(void) {          \
    __asm__ volatile(                                  \
        "mov  r1, lr                        \n"        \
        "lsls r1, r1, #29                   \n"        \
        "bmi  1f                            \n"        \
        "mrs  r0, msp                       \n"        \
        "b    2f                            \n"        \
        "1:                                 \n"        \
        "mrs  r0, psp                       \n"        \
        "2:                                 \n"        \
        "movs r2, #" #index                 "\n"       \
        "b    HeliaCorstoneReportFault      \n");      \
  }

HELIA_FAULT_TRAMPOLINE(HardFault_Handler, 0)
HELIA_FAULT_TRAMPOLINE(MemManage_Handler, 1)
HELIA_FAULT_TRAMPOLINE(BusFault_Handler, 2)
HELIA_FAULT_TRAMPOLINE(UsageFault_Handler, 3)

#undef HELIA_FAULT_TRAMPOLINE

}  // extern "C"
