#ifndef HELIA_RT_ZEPHYR_MEMORY_REGIONS_H_
#define HELIA_RT_ZEPHYR_MEMORY_REGIONS_H_

#include <zephyr/linker/section_tags.h>
#include <zephyr/toolchain.h>

/*
 * This header is currently intended only for Zephyr applications that consume
 * helia-rt, either from source modules or from the prebuilt release bundle.
 */

/*
 * Model/flatbuffer placement options.
 *
 * Example:
 *   alignas(8) HELIA_RT_MODEL_REGION_MRAM const unsigned char g_model[] = { ... };
 *   alignas(8) HELIA_RT_MODEL_REGION_SRAM unsigned char model_copy[] = { ... };
 */
#define HELIA_RT_MODEL_REGION_MRAM const
#define HELIA_RT_MODEL_REGION_SRAM Z_GENERIC_SECTION(.data)
#define HELIA_RT_MODEL_REGION_DTCM __dtcm_data_section

/*
 * Writable tensor arena placement options.
 *
 * Example:
 *   alignas(16) HELIA_RT_ARENA_REGION_SRAM uint8_t tensor_arena[kTensorArenaSize];
 *   alignas(16) HELIA_RT_ARENA_REGION_DTCM uint8_t tensor_arena[kTensorArenaSize];
 */
#define HELIA_RT_ARENA_REGION_SRAM
#define HELIA_RT_ARENA_REGION_DTCM __dtcm_bss_section

#endif /* HELIA_RT_ZEPHYR_MEMORY_REGIONS_H_ */
