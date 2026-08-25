# CMSIS-Pack Integration

heliaRT can be packaged as a source-only CMSIS-Pack for teams that use Keil MDK, CMSIS-Toolbox, CMSIS-Pack Manager, or Arm ecosystem IDEs. The pack is generated from the same CMake source manifest used by the source, Zephyr, and NSX builds.

!!! note "Current scope"
    The repository builds and validates a local source pack. Checked-in `csolution` examples are still tracked by [#124](https://github.com/AmbiqAI/helia-rt/issues/124); pack-index publication is out of scope there — see below.

## Local Pack vs Published Distribution

heliaRT's CMSIS-Pack is **generated and validated, not distributed**. There is no
published pack you can `cpackget add` by name — you build it from a checkout of
this repository, as described below.

| | Status |
|---|---|
| Pack generated from the CMake source manifest | Yes — `tools/cmsis_pack/build_pack.py` |
| Pack built and validated on every relevant change | Yes — the `Build CMSIS-Pack from SSoT manifest` job in `.github/workflows/smoke_cmake.yml` |
| Pack available as a CI build artifact | Yes — uploaded as `helia-rt-cmsis-pack` |
| Pack attached to GitHub Releases | No |
| Pack published to a public pack index (`.pidx`) or the Keil index | No |

What CI checks on each run, in order: it builds the pack, runs
`packchk --disable-validation` against the staged PDSC, asserts the archive's
size and structure and that every `source` file referenced by the PDSC resolves
inside the archive, asserts the three component variants (`Reference`,
`CMSIS-NN`, `HELIA`) are present, and runs the repository PDSC contract check
(`tools/cmsis_pack/check_pdsc.py`) for pack identity and the `ns-cmsis-nn` pin.

Publishing to the public Keil pack index and hosting a `.pidx` pack index are
explicitly **out of scope** for [#124](https://github.com/AmbiqAI/helia-rt/issues/124),
which tracks the remaining consumer-validation work. Index publication is **not
currently tracked** — it has not been declined, but no issue or milestone plans
it. Build the pack locally and expect to keep doing so.

!!! tip "What this means for you"
    Pin the repository (tag or commit) rather than a pack version, build the
    pack as part of your own setup or CI step, and install it with
    `cpackget add` from the file you built. Treat the generated pack as
    reproducible output of a known revision, not as a released artifact with an
    independent lifecycle.

## Build A Local Pack

Build the pack from a checkout:

```bash
python3 tools/cmsis_pack/build_pack.py --output dist
```

The output follows CMSIS-Pack naming conventions:

```plaintext
dist/Ambiq.helia-rt.<version>.pack
```

The pack version comes from the release-please-managed `HELIA_RT_VERSION` macro in `tensorflow/lite/micro/helia_rt_version.h`.

## Validate The Pack

The CI pack job runs the repository's PDSC contract checker and `packchk` from CMSIS-Toolbox. To reproduce the same checks locally, [install CMSIS-Toolbox](https://open-cmsis-pack.github.io/cmsis-toolbox/installation/), then run:

```bash
python3 tools/cmsis_pack/build_pack.py --output dist --keep-stage
PACK=$(ls dist/Ambiq.helia-rt.*.pack | head -1)
STAGE="${PACK%.pack}.stage"

python3 tools/cmsis_pack/check_pdsc.py "${PACK}"
packchk --disable-validation "${STAGE}/Ambiq.helia-rt.pdsc"
```

`packchk --disable-validation` keeps the semantic pack checks enabled while avoiding a CMSIS-Toolbox 2.13.0 XSD-validation crash seen on this generated PDSC. The workflow still parses and validates the archive structure and PDSC contract separately.

## Install With cpackget

After building the pack, add it to a local CMSIS-Pack installation:

```bash
cpackget add dist/Ambiq.helia-rt.<version>.pack
```

Use the exact file name generated in `dist/`. A consumer project can then select one of the pack's component variants.

## Component Variants

The generated pack exposes these source-build variants:

| Variant | Backend | Notes |
|---|---|---|
| `Reference` | Portable TFLM kernels | No external NN library dependency. |
| `CMSIS-NN` | Arm CMSIS-NN kernels | Uses the open CMSIS-NN backend source set. |
| `HELIA` | Ambiq HELIA kernels | Requires the Ambiq `ns-cmsis-nn` / heliaCORE pack dependency. |

## Current Recommendation

Use CMSIS-Pack when your application is already built around CMSIS-Toolbox or Keil tooling. For Zephyr or neuralSPOT projects, the native integration paths remain more direct:

| Path | Best for |
|---|---|
| [Zephyr module](zephyr.md) | Product integration, Kconfig backend selection, source or prebuilt bundles. |
| [neuralSPOT](neuralspot.md) | Fast model profiling and deployment on Ambiq evaluation boards. |
| [Source build](source.md) | Custom build systems, direct static library linkage, and source-level debugging. |

## Compatibility Goals

The CMSIS-Pack flow should not change the application-facing runtime model. Teams should still use the familiar `.tflite` model format, `MicroInterpreter` lifecycle, resolver registration pattern, and static tensor arena strategy.

## Next Steps

- [Zephyr setup](zephyr.md) — current recommended product integration path
- [Toolchains](../guides/toolchains.md) — compare GCC, Arm Compiler 6, and ATfE
- [SPEED vs SIZE](../guides/speed-vs-size.md) — understand HELIA source-build kernel profiles
