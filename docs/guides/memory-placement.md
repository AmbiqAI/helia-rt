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
| `zephyr_linker_sources(...)` | CMake linker fragment API |
| Devicetree SRAM partitioning | Board `.dts` / `.overlay` files |

## Apollo Memory Region Summary

| SoC | TCM layout | `zephyr,itcm` | `zephyr,dtcm` |
|---|---|---|---|
| Apollo3 / Apollo3p | Single TCM (64 KB at 0x10000000) | `&tcm` (labelled "ITCM") | *not defined* |
| Apollo4 / Apollo4p | Single TCM (384 KB at 0x10000000) | `&tcm` (labelled "ITCM") | *not defined* |
| Apollo510 | Separate ITCM (256 KB) + DTCM (512 KB) | `&itcm` | `&dtcm` |


> **Key point:** Apollo3 and Apollo4 boards expose their TCM as an **ITCM**
> region only. `__dtcm_bss_section` and `__dtcm_data_section` require
> `zephyr,dtcm` in the devicetree and therefore **only work on Apollo510**.
> On Apollo3/4, use `__itcm_section` instead.

## Canonical Patterns

### Model in MRAM / Flash (all boards)

Declare the model flatbuffer as `const` so the linker places it in
`.rodata`, which maps to **MRAM** on Apollo4/5 and **flash** on Apollo3:

```c
#include <zephyr/toolchain.h>

__aligned(8)
static const unsigned char g_model[] = { /* flatbuffer bytes */ };
```

### Model copy in SRAM (all boards)

If you need a writable copy (e.g. for in-place model patching), remove the
`const` qualifier. A non-`const` initialized array at file scope is
automatically placed in `.data` and copied from flash/MRAM to SRAM at boot:

```c
__aligned(8)
static unsigned char g_model_copy[] = {
    /* flatbuffer bytes — copied from flash/MRAM to SRAM at boot */
};
```

If you need the copy in a **specific** SRAM bank, use a custom section name
wired to a linker fragment (see [Custom Section with Linker Fragment](#custom-section-with-linker-fragment) below):

```c
#include <zephyr/toolchain.h>

__aligned(8)
static Z_GENERIC_SECTION(.app_model_sram) unsigned char g_model_copy[] = {
    /* flatbuffer bytes */
};
```

### Arena in default SRAM (all boards)

An uninitialized global or static array lands in `.bss`, which is placed in
the default `zephyr,sram` region:

```c
__aligned(16)
static uint8_t tensor_arena[96 * 1024];
```

> On Apollo3 and Apollo4, the TCM is exposed as ITCM only. Zephyr's
> `__itcm_section` copies data from flash at boot, so it is suited for
> hot kernel code or small lookup tables — not large scratch buffers.
> Use default SRAM for the tensor arena on these SoCs.

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

#### How it works

You create a custom **linker section** (a named bucket of data) and assign
it to a **memory region** (a physical address range like SRAM, DTCM, or
`SRAM_NO_CACHE`). Three pieces are needed:

1. **C source** — `Z_GENERIC_SECTION(.app_arena)` tags a variable so the
   compiler emits it into a section called `.app_arena` (name is arbitrary).

2. **Linker fragment** (`.ld` file in your app directory) — maps that
   section to a memory region using Zephyr-provided macros:
   - `SECTION_PROLOGUE(name, (NOLOAD),)` — uninitialized buffers (no
     flash copy at boot).
   - `SECTION_DATA_PROLOGUE(name,,)` — initialized data (copied from
     flash at boot).

   These macros come from `<zephyr/linker/linker-tool.h>`, which
   dispatches to the correct toolchain backend (GCC, LLD/armclang, MWDT)
   automatically.

3. **CMake** — `zephyr_linker_sources(SECTIONS my_fragment.ld)` injects
   your fragment into Zephyr's main linker script at build time. You never
   edit the main linker script directly.

#### Example

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
} > DTCM   /* or ITCM, SRAM_NO_CACHE, etc. */
```

```c
#include <zephyr/toolchain.h>

static Z_GENERIC_SECTION(.app_model) unsigned char model_copy[] = { /* ... */ };
static Z_GENERIC_SECTION(.app_arena) uint8_t tensor_arena[96 * 1024];
```

## Devicetree and SRAM Partitioning

On SoCs with a large SSRAM (e.g. Apollo510 with 3 MB), the board DTS
typically assigns a portion to `zephyr,sram` (Zephyr's default heap /
stack / `.bss` region) and exposes additional portions as named memory
regions. A named region declared with `compatible = "zephyr,memory-region"`
causes Zephyr's build system to emit a matching linker region, which custom
linker sections can then target.

The Apollo510 EVB board DTS demonstrates this pattern:

```dts
/* From apollo510_evb.dts (simplified) */
/ {
    chosen {
        zephyr,sram = &sram0;   /* default heap/stack/bss — 2 MB */
    };

    /* Main SRAM: first 2 MB of the 3 MB SSRAM */
    sram0: memory@20080000 {
        compatible = "mmio-sram";
        reg = <0x20080000 0x200000>;   /* 2 MB */
    };

    /* Remaining 1 MB carved out as a non-cacheable region */
    sram_no_cache: memory@20280000 {
        compatible = "zephyr,memory-region", "mmio-sram";
        reg = <0x20280000 0x100000>;   /* 1 MB */
        zephyr,memory-region = "SRAM_NO_CACHE";
        zephyr,memory-attr = <DT_MEM_ARM(ATTR_MPU_RAM_NOCACHE)>;
    };
};
```

Key points:

- `sram0` uses `compatible = "mmio-sram"` and is selected via
  `chosen { zephyr,sram = &sram0; }` — Zephyr places `.bss`, `.data`,
  heap, and stacks here by default.
- `sram_no_cache` adds `"zephyr,memory-region"` to its compatible list
  and sets `zephyr,memory-region = "SRAM_NO_CACHE"`. This tells Zephyr's
  linker generator to create a linker region named `SRAM_NO_CACHE`.
- `zephyr,memory-attr` sets MPU attributes (here: non-cacheable RAM).
  See the [Memory Attributes](https://docs.zephyrproject.org/latest/services/mem_mgmt/index.html)
  documentation for the full list of attribute flags.

To place a tensor arena in such a region, follow the
[Custom Section with Linker Fragment](#custom-section-with-linker-fragment)
pattern and set the output region to `SRAM_NO_CACHE` (i.e.
`} > SRAM_NO_CACHE` in the `.ld` fragment). Any `zephyr,memory-region`
name in the devicetree can be used as a linker region target this way.

You can also carve out your own region in an application overlay — just
ensure the `reg` range does not overlap with `sram0` or any other defined
region.

## Verifying Placement

Always check the linker map after building to confirm your buffers ended
up in the intended memory region:

```bash
# After building, search for your exact symbol names:
grep -E "g_model|tensor_arena" build/zephyr/zephyr.map

# Or inspect the relevant linker sections:
grep -E "^\.(itcm|dtcm|bss|rodata|data) " build/zephyr/zephyr.map
```

Look for your symbol names under the expected section (`.itcm`, `.dtcm_bss`,
`.rodata`, `.data`, `.bss`) and verify the address falls within the
expected memory range for your board.

## Next Steps

- [Zephyr setup](../getting-started/zephyr.md) — choose the module path for your product
- [Zephyr example](../examples/zephyr.md) — see a complete application flow
- [Troubleshooting](troubleshooting.md) — debug arena sizing, backend selection, and build issues
