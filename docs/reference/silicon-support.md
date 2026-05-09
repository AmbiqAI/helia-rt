# Silicon Support

<!-- TODO: Step 7 — Full content -->

!!! note "Work in progress"
    This page is part of the documentation refresh tracked in [#135](https://github.com/AmbiqAI/helia-rt/issues/135).

## Supported SoC Families

| SoC | Core | MVE / Helium | DSP | TCM Layout | Zephyr Board(s) |
|---|---|---|---|---|---|
| Apollo3 / Apollo3p | Cortex-M4F | — | ✓ | 64 KB unified | `apollo3p_evb` |
| Apollo4 / Apollo4p | Cortex-M4F | — | ✓ | TBD | `apollo4p_evb` |
| Apollo510 | Cortex-M55 | ✓ | ✓ | Split ITCM + DTCM | `apollo510_evb` |
| Atomiq | TBA | TBA | TBA | TBA | *(planned)* |

!!! info
    Verify TCM sizes against the Ambiq datasheet for your specific part number. The values above are approximate.

## Toolchain Compatibility per SoC

| SoC | GCC | armclang | ATfE |
|---|---|---|---|
| Apollo3 / Apollo3p | ✓ | ✓ | ✓ |
| Apollo4 / Apollo4p | ✓ | ✓ | ✓ |
| Apollo510 | ✓ | ✓ | ✓ (recommended) |

## Next Steps

- [Operator Coverage](operator-coverage.md)
- [Toolchains](../guides/toolchains.md)
