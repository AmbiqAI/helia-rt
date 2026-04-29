#ifndef HELIA_RT_MEMORY_REGIONS_H_
#define HELIA_RT_MEMORY_REGIONS_H_

#include <zephyr/linker/section_tags.h>
#include <zephyr/toolchain.h>

/*
 * Apollo3x and Apollo4x expose a single TCM region through Zephyr's ITCM
 * linker section. Apollo5x provides a dedicated DTCM linker region.
 */
#if defined(CONFIG_SOC_SERIES_APOLLO3X) || defined(CONFIG_SOC_SERIES_APOLLO4X)
#define HELIA_RT_TCM_DATA_SECTION __itcm_section
#define HELIA_RT_TCM_BSS_SECTION __itcm_section
#else
#define HELIA_RT_TCM_DATA_SECTION __dtcm_data_section
#define HELIA_RT_TCM_BSS_SECTION __dtcm_bss_section
#endif

/*
 * Model/flatbuffer placement options.
 *
 * Example:
 *   alignas(8) HELIA_RT_MODEL_REGION_MRAM const unsigned char g_model[] = { ... };
 *   alignas(8) HELIA_RT_MODEL_REGION_SRAM unsigned char model_copy[] = { ... };
 */
#define HELIA_RT_MODEL_REGION_MRAM const
#define HELIA_RT_MODEL_REGION_SRAM Z_GENERIC_SECTION(.data)
#define HELIA_RT_MODEL_REGION_DTCM HELIA_RT_TCM_DATA_SECTION

/*
 * Writable tensor arena placement options.
 *
 * Example:
 *   alignas(16) HELIA_RT_ARENA_REGION_SRAM uint8_t tensor_arena[kTensorArenaSize];
 *   alignas(16) HELIA_RT_ARENA_REGION_DTCM uint8_t tensor_arena[kTensorArenaSize];
 */
#define HELIA_RT_ARENA_REGION_SRAM
#define HELIA_RT_ARENA_REGION_DTCM HELIA_RT_TCM_BSS_SECTION

#endif /* HELIA_RT_MEMORY_REGIONS_H_ */
