# Placing Model and Arena on Apollo + Zephyr

This page shows how to control where the TFLite model flatbuffer and
tensor arena are placed in memory on Ambiq Apollo SoCs when building with
Zephyr.

## Principle

helia-rt receives pointers (`GetModel()`, tensor arena pointer). It never
controls where those objects live in memory — placement is purely an
application-side and board-level concern.

Use Zephyr-native primitives directly instead of custom wrapper macros:

| Primitive | Header / Mechanism |
|---|---|
| `__itcm_section` | `<zephyr/linker/section_tags.h>` |
| `__dtcm_bss_section`, `__dtcm_data_section`, `__dtcm_noinit_section` | `<zephyr/linker/section_tags.h>` |
| `Z_GENERIC_SECTION(name)` | `<zephyr/toolchain.h>` |
| Custom linker fragments | `zephyr_linker_sources(...)` in CMake |
| Devicetree SRAM partitioning | Board `.dts` / `.overlay` files |

## Apollo Memory Region Summary

| SoC | TCM regions in devicetree | `zephyr,itcm` | `zephyr,dtcm` |
|---|---|---|---|
| Apollo3 / Apollo3p | Single TCM (64 KB at 0x10000000) | `&tcm` (labelled "ITCM") | *not defined* |
| Apollo4 / Apollo4p | Single TCM (64 KB at 0x10000000) | `&tcm` (labelled "ITCM") | *not defined* |
| Apollo510 | Separate ITCM + DTCM | `&itcm` | `&dtcm` |

> **Key point:** Apollo3 and Apollo4 boards expose their TCM as an **ITCM**
> region only. `__dtcm_bss_section` and `__dtcm_data_section` require
> `zephyr,dtcm` in the devicetree and therefore **only work on Apollo510**.
> On Apollo3/4, use `__itcm_section` instead.

## Canonical Patterns

### Model in MRAM / Flash (all boards)

Declare the model flatbuffer as `const` so the linker places it in the
read-only flash / MRAM region:

```c
#include <zephyr/toolchain.h>

__aligned(8)
static const unsigned char g_model[] = { /* flatbuffer bytes */ };
```

### Model copy in SRAM (all boards)

If you need a writable copy (e.g. for in-place model patching), force it
into `.data`:

```c
#include <zephyr/toolchain.h>

__aligned(8)
static Z_GENERIC_SECTION(.data) unsigned char g_model_copy[] = {
    /* flatbuffer bytes — will be copied from flash to SRAM at boot */
};
```

### Arena in default SRAM (all boards)

An uninitialized global or static array lands in `.bss`, which is placed in
the default `zephyr,sram` region:

```c
__aligned(16)
static uint8_t tensor_arena[96 * 1024];
```

### Arena in TCM — Apollo3 / Apollo4

On Apollo3 and Apollo4, the TCM is exposed as an ITCM linker region.
Use `__itcm_section` to place the arena there. Because ITCM is a
load-at-boot section, the array **must be initialized** (even if to zero):

```c
#include <zephyr/linker/section_tags.h>

__itcm_section __aligned(16)
static uint8_t tensor_arena[96 * 1024] = {0};
```

### Arena in DTCM — Apollo510

Apollo510 has a dedicated DTCM region. Use `__dtcm_bss_section` for
zero-initialized (NOLOAD) placement:

```c
#include <zephyr/linker/section_tags.h>

__dtcm_bss_section __aligned(16)
static uint8_t tensor_arena[96 * 1024];
```

Or use `__dtcm_noinit_section` if you do not need the arena zeroed at boot.

### Custom Section with Linker Fragment

Use this when default `.data` / `.bss` placement is not enough, or when you
want fine-grained control over which memory bank receives the data.

```cmake
# CMakeLists.txt
zephyr_linker_sources(SECTIONS app_memory_placement.ld)
```

```ld
/* app_memory_placement.ld */
SECTION_DATA_PROLOGUE(.app_model,,)
{
    KEEP(*(.app_model*))
} > SRAM

SECTION_PROLOGUE(.app_arena, (NOLOAD),)
{
    KEEP(*(.app_arena*))
} > DTCM   /* or ITCM, depending on board */
```

```c
#include <zephyr/toolchain.h>

static Z_GENERIC_SECTION(.app_model) unsigned char model_copy[] = { /* ... */ };
static Z_GENERIC_SECTION(.app_arena) uint8_t tensor_arena[96 * 1024];
```

## Devicetree and SRAM Partitioning

For multi-bank SRAM layouts (e.g. Apollo510 with a separate non-cached SRAM
region), make memory ownership explicit in devicetree and have custom linker
sections target the matching memory region.

Example overlay (illustrative):

```dts
/ {
    chosen {
        zephyr,sram = &sram0;
    };
};

&sram0 {
    reg = <0x10000000 0x00080000>;
};

&sram1 {
    reg = <0x10080000 0x00080000>;
};
```

Then map custom sections to the intended region in your linker fragment.

## Verifying Placement

Always check the linker map after building to confirm your buffers ended
up in the intended memory region:

```bash
# After building, inspect the map file:
grep -E "model|arena" build/zephyr/zephyr.map
```

Look for your symbol names under the expected section (`.itcm`, `.dtcm_bss`,
`.rodata`, `.data`, `.bss`) and verify the address falls within the
expected memory range for your board.
