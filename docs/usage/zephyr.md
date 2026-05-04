# :material-chip: Zephyr Setup

This page explains the supported Zephyr integration styles for heliaRT and how
to choose between them.

For the full step-by-step application flow, including exact `CMakeLists.txt`,
`prj.conf`, minimal bring-up code, build, flash, and UART logs, see
[Zephyr example](../examples/zephyr.md).

## Supported Integration Paths

heliaRT supports three Zephyr integration paths:

=== "Source Module + CMSIS-NN"

    Use this when you want:

    - source visibility
    - the public open-source optimized backend
    - the ability to debug or modify heliaRT internals

    This path uses:

    - local `helia-rt` source in `modules/helia-rt`
    - open `cmsis-nn` from the Zephyr workspace
    - `CONFIG_CMSIS_NN=y`
    - `CONFIG_HELIA_RT=y`
    - `CONFIG_HELIA_RT_BACKEND_CMSIS_NN=y`

=== "Source Module + HELIA"

    Use this when you want:

    - source visibility
    - Ambiq HELIA acceleration
    - the ability to debug or modify the raw backend integration

    This path uses:

    - local `helia-rt` source in `modules/helia-rt`
    - local `ns-cmsis-nn` in `modules/ns-cmsis-nn`
    - `CONFIG_HELIA_RT=y`
    - `CONFIG_NS_CMSIS_NN=y`
    - `CONFIG_HELIA_RT_BACKEND_HELIA=y`

=== "Prebuilt Release Module"

    Use this when you want:

    - the fastest path to a working integration
    - a published static library instead of building heliaRT from source
    - a smaller and leaner codebase

    This path uses:

    - one downloaded prebuilt module in `modules/helia-rt-m55-release`
    - no separate `ns-cmsis-nn` module
    - `CONFIG_HELIA_RT=y`
    - `CONFIG_FPU=y` for Cortex-M55 builds

## Module Discovery

Zephyr can discover heliaRT modules in two common ways:

### West-Managed Modules

Use this when you want:

- pinned workspace-level dependencies
- consistent CI behavior
- shared dependency management across apps

### `ZEPHYR_EXTRA_MODULES`

Use this when you want:

- app-local control over module paths
- easy switching between source and prebuilt variants
- simple local experimentation

The Zephyr example page shows the exact `ZEPHYR_EXTRA_MODULES` setup for each
integration path.

For model and tensor arena placement on Apollo memory regions, see
[Apollo memory placement](memory_placement.md).

## Notes

- Do not add both source and prebuilt heliaRT module variants to the same application.
- The prebuilt module selects its archive from board CPU, toolchain, and configured flavor.
- The prebuilt module also applies `TF_LITE_STATIC_MEMORY` so the application matches the prebuilt tensor layout.
- Open `cmsis-nn` and `ns-cmsis-nn` are different backends and should not be treated as interchangeable.

## Next Step

Continue with [Zephyr example](../examples/zephyr.md) for:

- exact workspace layout
- exact module placement
- exact app `CMakeLists.txt`
- exact `prj.conf`
- minimal bring-up code
- build, flash, and log commands
