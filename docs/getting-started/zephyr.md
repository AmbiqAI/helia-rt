# Zephyr setup

<div class="zephyr-page" markdown>

<section class="zephyr-hero" markdown>

<p class="section-eyebrow">Zephyr integration</p>

## Choose the module shape that matches your product stage

This page explains the supported Zephyr integration styles for heliaRT and how
to choose between them. Start with source modules when you need visibility and
debugging control, or use the prebuilt release module when you want the fastest
path to a repeatable application build.

For the full step-by-step application flow, including exact `CMakeLists.txt`,
`prj.conf`, minimal bring-up code, build, flash, and UART logs, see
[Zephyr example](../examples/zephyr.md).

</section>

## Supported Integration Paths

heliaRT supports three Zephyr integration paths. They all keep the application
surface familiar, but they make different tradeoffs around source visibility,
backend selection, and release packaging.

<div class="zephyr-path-grid" markdown>

<div class="zephyr-path-card" markdown>
<span class="logo-mark">SRC</span>
### Source + CMSIS-NN
Use this when you want the open Arm-optimized backend, source-level inspection,
and a conventional Zephyr module layout.
</div>

<div class="zephyr-path-card zephyr-path-card--primary" markdown>
<span class="logo-mark">HELIA</span>
### Source + HELIA
Use this when you want source visibility plus Ambiq HELIA acceleration through
the `ns-cmsis-nn` backend module.
</div>

<div class="zephyr-path-card" markdown>
<span class="logo-mark">LIB</span>
### Prebuilt release
Use this when you want a smaller workspace and a published static library that
matches the release build matrix.
</div>

</div>

=== "Source Module + CMSIS-NN"

    **Open optimized backend.**

    Use this path when you want source visibility and the public CMSIS-NN backend.
    It is a good default for teams that want a familiar Zephyr source module with
    open optimized kernels and easy local debugging.

    **Use this when you want:**

    - source visibility
    - the public open-source optimized backend
    - the ability to debug or modify heliaRT internals

    **This path uses:**

    - local `helia-rt` source in `modules/helia-rt`
    - open `cmsis-nn` from the Zephyr workspace
    - `CONFIG_CMSIS_NN=y`
    - `CONFIG_HELIA_RT=y`
    - `CONFIG_HELIA_RT_BACKEND_CMSIS_NN=y`

=== "Source Module + HELIA"

    **Ambiq accelerated source path.**

    Use this path when you want source-level integration and Ambiq HELIA kernel
    acceleration. It adds the `ns-cmsis-nn` module beside heliaRT so supported
    operators can take the Ambiq-tuned backend path.

    **Use this when you want:**

    - source visibility
    - Ambiq HELIA acceleration
    - the ability to debug or modify the raw backend integration

    **This path uses:**

    - local `helia-rt` source in `modules/helia-rt`
    - local `ns-cmsis-nn` in `modules/ns-cmsis-nn`
    - `CONFIG_HELIA_RT=y`
    - `CONFIG_NS_CMSIS_NN=y`
    - `CONFIG_HELIA_RT_BACKEND_HELIA=y`

=== "Prebuilt Release Module"

    **Release artifact path.**

    Use this path when you want to consume a published heliaRT release instead of
    building the runtime from source. The module selects the matching archive for
    board CPU, toolchain, and configured flavor.

    **Use this when you want:**

    - the fastest path to a working integration
    - a published static library instead of building heliaRT from source
    - a smaller and leaner codebase

    **This path uses:**

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

</div>
