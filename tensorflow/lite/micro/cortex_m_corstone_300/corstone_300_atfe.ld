/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * ============================================================================
 *
 * Linker script for the Arm Corstone-300 FVP target when building with
 * the Arm Toolchain for Embedded (ATfE; LLVM/Clang + picolibc).
 *
 * This script is derived from the GCC-targeted
 * `tensorflow/lite/micro/tools/make/downloads/ethos_u_core_platform/
 * targets/corstone-300/platform_parsed.ld` script that ships with
 * ethos-u-core-platform (and is patched by `ethos_u_core_platform.patch`).
 *
 * Differences from `platform_parsed.ld`:
 *   - Adds (NOLOAD) `.tdata` / `.tbss` anchor sections so picolibc TLS
 *     objects (e.g. `errno`, declared `__thread` in libc.a) link
 *     cleanly. Without an explicit `.tbss` lld errors with
 *     `STT_TLS without PT_TLS`. lld auto-generates the corresponding
 *     `PT_TLS` program header.
 *   - Drops the explicit `PHDRS { ... }` block (and `:rom_exec`,
 *     `:rom_dram`, `:null` per-section phdr selectors). The original
 *     script was authored for ld.bfd; lld mishandles its layout
 *     (resulting in `output file too large`). Letting lld auto-generate
 *     PT_LOAD segments produces the correct ELF.
 *   - Switches `.heap` / `.stack` from `(COPY)` to `(NOLOAD)` so the
 *     location counter advances correctly under lld.
 *   - Adds `PROVIDE(...)` aliases for the symbols picolibc's `crt0.o`
 *     and CMSIS' clang startup expect (`__data_start`, `__data_size`,
 *     `__data_source`, `__bss_start`, `__bss_size`, `__stack`,
 *     `__stack_limit`, `__stack_seal`, `__tls_base`,
 *     `__arm32_tls_tcb_offset`, `__heap_start`, `__heap_end`).
 *
 * Memory map and section placement remain identical to the GCC build so
 * that the resulting ELF still boots on FVP_Corstone_SSE-300_Ethos-U55.
 */

__STACK_SIZE = 0x00030000;
__HEAP_SIZE = 0x00008000;

MEMORY
{
  ITCM (rx) : ORIGIN = 0x10000000, LENGTH = 0x00080000
  BRAM (rw) : ORIGIN = 0x11000000, LENGTH = 0x00400000
  DTCM (rw) : ORIGIN = 0x30000000, LENGTH = 0x00080000
  SRAM (rw) : ORIGIN = 0x31000000, LENGTH = 0x02000000
  DDR (rwx) : ORIGIN = 0x70000000, LENGTH = 0x60000000
}

ENTRY(Reset_Handler)

SECTIONS
{
  .text :
  {
    KEEP(*(.vectors))
    *(.text*)
    KEEP(*(.init))
    KEEP(*(.fini))
    *crtbegin.o(.ctors)
    *crtbegin?.o(.ctors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
    *(SORT(.ctors.*))
    *(.ctors)
    *crtbegin.o(.dtors)
    *crtbegin?.o(.dtors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
    *(SORT(.dtors.*))
    *(.dtors)
    KEEP(*(.eh_frame*))
  } > ITCM

  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > ITCM

  .ARM.exidx :
  {
    __exidx_start = .;
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
    __exidx_end = .;
  } > ITCM

  .copy.table :
  {
    . = ALIGN(4);
    __copy_table_start__ = .;
    LONG (__etext)
    LONG (__data_start__)
    LONG (__data_end__ - __data_start__)
    LONG (__eddr_data)
    LONG (__sram_data_start__)
    LONG (__sram_data_end__ - __sram_data_start__ )
    __copy_table_end__ = .;
  } > ITCM

  .zero.table :
  {
    . = ALIGN(4);
    __zero_table_start__ = .;
    LONG (__bss_start__)
    LONG (__bss_end__ - __bss_start__)
    __zero_table_end__ = .;
    __etext = ALIGN (4);
  } > ITCM

  .data : AT(__etext)
  {
    __data_start__ = .;
    *(vtable)
    *(.data)
    *(.data.*)
    . = ALIGN(4);
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);
    . = ALIGN(4);
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN (__init_array_end = .);
    . = ALIGN(4);
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array))
    PROVIDE_HIDDEN (__fini_array_end = .);
    KEEP(*(.jcr*))
    . = ALIGN(4);
    __data_end__ = .;
  } > DTCM

  /* picolibc puts thread-local data in .tdata / .tbss. TFLM declares no
   * `__thread` variables of its own but picolibc's libc.a still defines
   * `errno` (and a few other state vars) as TLS, so we must provide
   * .tdata / .tbss anchors or lld errors with `STT_TLS without PT_TLS`.
   *
   * Both sections are NOLOAD in DTCM so they do not contribute to any
   * PT_LOAD segment. lld auto-generates the PT_TLS program header from
   * them. picolibc's `_set_tls` allocates the runtime TLS block
   * separately. */
  .tdata (NOLOAD) : ALIGN(8)
  {
    __tdata_start = .;
    *(.tdata .tdata.* .gnu.linkonce.td.*)
    __tdata_end = .;
  } > DTCM

  .tbss (NOLOAD) : ALIGN(8)
  {
    __tbss_start = .;
    *(.tbss .tbss.* .gnu.linkonce.tb.*)
    *(.tcommon)
    __tbss_end = .;
  } > DTCM

  .sram.bss :
  {
    . = ALIGN(16);
    *(.bss.ethosu_scratch);
    *.(output_data_sec)
  } > SRAM

  .ddr :
  {
    . = ALIGN(16);
    *(.bss.tensor_arena)
    . = ALIGN(4);
    *(input_data_sec)
    . = ALIGN(16);
    *(network_model_sec)
    * (expected_output_data_sec)
    *(.rodata*)
  } > DDR

  __eddr_data = ALIGN (16) ;
  .sram.data : {
    __sram_data_start__ = .;
    *(.sram.data)
    __sram_data_end__ = .;
  } > BRAM AT >DDR

  .bss :
  {
    . = ALIGN(4);
    __bss_start__ = .;
    *(.bss)
    *(.bss.*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
  } > SRAM

  .heap (NOLOAD) :
  {
    . = ALIGN(8);
    __end__ = .;
    PROVIDE(end = .);
    . = . + __HEAP_SIZE;
    . = ALIGN(8);
    __HeapLimit = .;
  } > DTCM

  .stack (ORIGIN(DTCM) + LENGTH(DTCM) - __STACK_SIZE) (NOLOAD) :
  {
    . = ALIGN(8);
    __StackLimit = .;
    . = . + __STACK_SIZE;
    . = ALIGN(8);
    __StackTop = .;
  } > DTCM

  /* CMSIS clang startup (cmsis_clang_m.h) expects the lowercase aliases. */
  PROVIDE(__stack       = __StackTop);
  PROVIDE(__stack_limit = __StackLimit);
  /* TrustZone stack seal is unused on this Cortex-M55 build; point it at the
   * top of the stack so taking its address is harmless. */
  PROVIDE(__stack_seal  = __StackTop);

  /* picolibc's crt0._start (pulled in via -lcrt0-semihost) needs these. */
  PROVIDE(__data_start  = __data_start__);
  PROVIDE(__data_end    = __data_end__);
  PROVIDE(__data_size   = __data_end__ - __data_start__);
  PROVIDE(__data_source = __etext);
  PROVIDE(__data_source_end  = __etext + (__data_end__ - __data_start__));
  PROVIDE(__data_source_size = __data_end__ - __data_start__);

  PROVIDE(__bss_start   = __bss_start__);
  PROVIDE(__bss_end     = __bss_end__);
  PROVIDE(__bss_size    = __bss_end__ - __bss_start__);

  /* Heap pointers consumed by picolibc's sbrk implementation. */
  PROVIDE(__heap_start  = __end__);
  PROVIDE(__heap_end    = __HeapLimit);

  /* TLS image consumed by picolibc's _set_tls. The TFLM build doesn't
   * use any application TLS so __tls_base just points at our (empty)
   * .tdata anchor. */
  PROVIDE(__tls_base            = ADDR(.tdata));
  PROVIDE(__tdata_source        = LOADADDR(.tdata));
  PROVIDE(__tdata_source_end    = LOADADDR(.tdata) + SIZEOF(.tdata));
  PROVIDE(__tdata_size          = SIZEOF(.tdata));
  PROVIDE(__tbss_size           = SIZEOF(.tbss));
  PROVIDE(__tls_size            = SIZEOF(.tdata) + SIZEOF(.tbss));
  PROVIDE(__tls_align           = 8);
  PROVIDE(__arm32_tls_tcb_offset = 8);

  ASSERT(__StackLimit >= __HeapLimit, "region DTCM overflowed with stack")
}
