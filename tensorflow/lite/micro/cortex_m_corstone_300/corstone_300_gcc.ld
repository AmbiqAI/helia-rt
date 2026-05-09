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
 * helia-rt: Linker script for the Arm Corstone-300 FVP target when building
 * with arm-none-eabi-gcc.
 *
 * Derived verbatim from upstream `tensorflow/lite/micro/tools/make/downloads/
 * ethos_u_core_platform/targets/corstone-300/platform_parsed.ld` (the script
 * patched into ethos-u-core-platform by `ethos_u_core_platform.patch`).
 *
 * Sole intentional divergence from upstream:
 *
 *   - `.vectors` is split out of `.text` and KEEP'd at the start of ITCM so
 *     the CPU can still fetch the initial SP and Reset_Handler from VTOR=0
 *     (ITCM is aliased at 0x00000000 on Corstone-300).
 *   - The bulk of `.text`, `.ARM.extab`, and `.ARM.exidx` is moved from
 *     ITCM to DDR (`:rom_dram`). Cortex-M55 caches DDR, so this is
 *     functionally transparent for the integration tests we run on FVP.
 *
 * Why we need this:
 *   ITCM is 512 KiB. With `THIRD_PARTY_KERNEL_OPTIMIZATION_LEVEL := -O3`
 *   on GCC 14 the CMSIS-NN inner loops unroll/vectorise enough that the
 *   `tensorflow/lite/micro/integration_tests/{nnaed,seanet}/...` link
 *   images overflow ITCM by 4-6 KiB. Moving `.text` to DDR matches what
 *   the ATfE linker script already does (see `corstone_300_atfe.ld`) so
 *   GCC- and ATfE-built integration tests share the same memory layout
 *   and the same kernel optimisation level.
 *
 * Everything else (memory map, PHDRS, .data/.bss/.heap/.stack placement,
 * symbol names) tracks `platform_parsed.ld` verbatim so future ethos-u
 * syncs are easy to diff against this file.
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
PHDRS
{
    rom_exec PT_LOAD;
    rom_dram PT_LOAD;
    null PT_NULL;
}
ENTRY(Reset_Handler)
SECTIONS
{
  /* helia-rt: split .vectors out of .text so it stays in ITCM. The CPU
   * needs to read the initial SP and Reset_Handler address from VTOR=0
   * (ITCM is aliased at 0x00000000 on Corstone-300).
   */
  .vectors :
  {
    KEEP(*(.vectors))
  } > ITCM :rom_exec
  /* helia-rt: bulk of .text moved from ITCM to DDR to fit -O3 CMSIS-NN
   * builds. See file header for rationale.
   */
  .text :
  {
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
  } > DDR :rom_dram
  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > DDR :rom_dram
  .ARM.exidx :
  {
  __exidx_start = .;
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  __exidx_end = .;
  } > DDR :rom_dram
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
  } > ITCM :rom_exec
  .zero.table :
  {
    . = ALIGN(4);
    __zero_table_start__ = .;
    LONG (__bss_start__)
    LONG (__bss_end__ - __bss_start__)
    __zero_table_end__ = .;
  __etext = ALIGN (4);
  } > ITCM :rom_exec
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
  } > DTCM :rom_exec
  .sram.bss :
  {
    . = ALIGN(16);
    *(.bss.ethosu_scratch);
    *.(output_data_sec)
  } > SRAM :null
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
  } > DDR :rom_dram
  __eddr_data = ALIGN (16) ;
  .sram.data : {
    __sram_data_start__ = .;
    *(.sram.data)
    __sram_data_end__ = .;
  } > BRAM AT >DDR :rom_dram
  .bss :
  {
    . = ALIGN(4);
    __bss_start__ = .;
    *(.bss)
    *(.bss.*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
  } > SRAM :null
  .heap (COPY) :
  {
    . = ALIGN(8);
    __end__ = .;
    PROVIDE(end = .);
    . = . + __HEAP_SIZE;
    . = ALIGN(8);
    __HeapLimit = .;
  } > DTCM :null
  .stack (ORIGIN(DTCM) + LENGTH(DTCM) - __STACK_SIZE) (COPY) :
  {
    . = ALIGN(8);
    __StackLimit = .;
    . = . + __STACK_SIZE;
    . = ALIGN(8);
    __StackTop = .;
  } > DTCM :null
  PROVIDE(__stack = __StackTop);
  ASSERT(__StackLimit >= __HeapLimit, "region DTCM overflowed with stack")
}
