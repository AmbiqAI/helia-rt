# Upgrading from Upstream LiteRT

heliaRT is a **drop-in replacement** for upstream LiteRT for Micro, formerly TensorFlow Lite for Microcontrollers / TFLM. The API surface is identical, so switching requires only changing where the library comes from.

## What Changes

| Aspect | Before (upstream LiteRT) | After (heliaRT) |
|---|---|---|
| Source / archive | `tensorflow/tflite-micro` | `AmbiqAI/helia-rt` |
| Additional backend | Reference + CMSIS-NN | Reference + CMSIS-NN + **HELIA** |
| Build variants | Single | **SPEED** and **SIZE** |
| Toolchain support | GCC, armclang | GCC, armclang, **ATfE** |
| Ambiq-specific tuning | None | heliaCORE kernel paths |

## What Stays the Same

- `.tflite` model format — no retraining or re-quantization needed
- `MicroInterpreter` lifecycle (`AllocateTensors` → `Invoke` → read output)
- `MicroMutableOpResolver` registration pattern
- Tensor arena sizing and static memory planning
- All upstream Reference and CMSIS-NN kernels remain available

## Step-by-Step: Zephyr

=== "Source module"

    Replace the upstream LiteRT module with heliaRT in your `west.yml`:

    ```yaml
    # west.yml — projects:
    - name: helia-rt
      url: https://github.com/AmbiqAI/helia-rt
      revision: main        # or pin to a release tag
      path: modules/lib/helia-rt
    ```

    Update `prj.conf`:

    ```cfg
    CONFIG_HELIA_RT=y
    CONFIG_HELIA_RT_BACKEND_HELIA=y   # or CMSIS_NN / REFERENCE
    ```

=== "Prebuilt bundle"

    Download the latest release archive from [GitHub Releases](https://github.com/AmbiqAI/helia-rt/releases) and point your `west.yml` at the extracted directory. See the [Zephyr getting-started guide](../getting-started/zephyr.md) for details.

## Step-by-Step: Makefile / Source Build

1. Clone heliaRT instead of upstream LiteRT:

    ```bash
    git clone https://github.com/AmbiqAI/helia-rt
    cd helia-rt
    ```

2. Build with the HELIA backend:

    ```bash
    make -f tensorflow/lite/micro/tools/make/Makefile \
        TARGET=cortex_m_generic \
        TARGET_ARCH=cortex-m55 \
        OPTIMIZED_KERNEL_DIR=helia \
        microlite
    ```

3. Link the resulting `libtensorflow-microlite.a` into your project exactly as before.

## Step-by-Step: neuralSPOT

neuralSPOT already bundles heliaRT. If you're using `ns_autodeploy`, you're already on heliaRT — no migration needed.

## Verifying the Upgrade

After building, confirm heliaRT is active:

```c
#include "tensorflow/lite/micro/heliart_version.h"
printf("heliaRT %s\n", HELIART_VERSION);
```

## Next Steps

- [Why heliaRT](../why-helia-rt.md) — the full pitch
- [Kernel Selection](kernel-selection.md) — how the backend is chosen at build time
- [Operator Coverage](../reference/operator-coverage.md) — what HELIA adds
