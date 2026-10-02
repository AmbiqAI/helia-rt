---
title: CMSIS-Pack
description: Generate and validate a local source pack for a CMSIS-based application.
---

heliaRT provides a source-pack generator driven by the same source manifest as its CMake integrations. Use it with an existing CMSIS-based application when you want its project tools to manage runtime sources.

## Build the pack

Use a pinned runtime checkout with Python 3 and CMake available. From its root:

```bash
python3 tools/cmsis_pack/build_pack.py --output dist --keep-stage
```

For runtime v1.21.0, this produces:

```text
dist/Ambiq.helia-rt.1.21.0.pack
dist/Ambiq.helia-rt.1.21.0.stage/
```

The generator derives the version from the runtime source manifest. Use the filename printed by your checkout rather than assuming the example version. It includes the selected runtime sources, public headers, license and PDSC description.

## Validate and install

Install CMSIS-Toolbox for `packchk` and `cpackget`, then validate the generated pack:

```bash
python3 tools/cmsis_pack/check_pdsc.py dist/Ambiq.helia-rt.1.21.0.pack
packchk --disable-validation dist/Ambiq.helia-rt.1.21.0.stage/Ambiq.helia-rt.pdsc
cpackget add --agree-embedded-license dist/Ambiq.helia-rt.1.21.0.pack
```

The `packchk` invocation matches the repository's packaging workflow: semantic checks run with XSD validation disabled. The separate Python check verifies the pack's expected identity and dependency contract. Neither check builds an application; the example below does.

## Select a component

In your project's component selection, use the Ambiq `Machine Learning` / `TFLM Runtime` / `helia-rt` component and choose one backend variant:

| Variant | Dependency |
|---|---|
| `Reference` | Portable runtime kernels without an external NN library. |
| `CMSIS-NN` | Upstream Arm CMSIS-NN kernel dependency for your target. |
| `HELIA` | The Ambiq heliaCORE (`ns-cmsis-nn`) source component. The generated PDSC requires version 7.35.0 or newer. |

Keep runtime and kernel feature settings consistent. A HELIA component selection does not establish support for every operator, tensor type or model shape. Review [Model compatibility](/helia-rt/guide/model-compatibility/) before enabling float kernels.

## Build the example consumer

[`tools/cmsis_pack/examples/hello_world`](https://github.com/AmbiqAI/helia-rt/tree/main/tools/cmsis_pack/examples/hello_world) is a minimal CMSIS-Toolbox solution. It targets the generic `ARM::ARMCM55` device and selects the `Reference` variant. It links a `MicroInterpreter` against the installed pack. With CMSIS-Toolbox 2.13.0, the Arm GNU and ATfE compilers registered through `GCC_TOOLCHAIN_<version>` and `CLANG_TOOLCHAIN_<version>`, and the pack added with `cpackget`:

```bash
cd tools/cmsis_pack/examples/hello_world
cbuild hello_world.csolution.yml --toolchain GCC --update-rte --packs
cbuild hello_world.csolution.yml --toolchain CLANG --update-rte --packs
```

`--packs` installs the public `ARM::CMSIS` and `ARM::Cortex_DFP` packs on first use. The example links `--specs=nosys.specs` on GCC and `-lsemihost` on ATfE to satisfy the C library's I/O and exit hooks; a board application supplies its own retarget layer instead. The packaging workflow builds this example on both compilers. It proves the pack's sources compile and link together; it does not run a model.

Your consumer project still provides the device/board support, compiler settings, startup code, memory layout and application sources. Add the model and inference sequence from [First inference](/helia-rt/getting-started/first-inference/), build with your CMSIS toolchain, then verify known inputs and outputs on the board.

## Distribution scope

The repository workflow generates the pack as a CI artifact. This guide uses a pack built from your checkout; it does not depend on public pack-index availability. The [pack generator](https://github.com/AmbiqAI/helia-rt/blob/main/tools/cmsis_pack/build_pack.py), [contract checker](https://github.com/AmbiqAI/helia-rt/blob/main/tools/cmsis_pack/check_pdsc.py) and [packaging workflow](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/smoke_cmake.yml) define its contents and checks. Board-level validation of your own project is separate from package validation.
