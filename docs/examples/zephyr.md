# :material-chip: Zephyr Examples

These examples are intended to help you bridge from setup to a working application.

## Example 1: Minimal Bring-Up

Goal:
confirm the heliaRT module is discovered correctly and that your Zephyr application links and boots.

Recommended structure:

```plaintext
<ws>/app/helia_rt_app/
├── CMakeLists.txt
├── prj.conf
└── src/
    └── main.cpp
```

Minimal `main.cpp`:

```cpp
#include <zephyr/sys/printk.h>

int main(void) {
  printk("heliaRT + Zephyr bring-up\n");
  return 0;
}
```

Use this first to verify:

- module discovery
- selected backend configuration
- board build and flash flow

## Example 2: Interpreter Skeleton

Goal:
set up the familiar TFLM-style execution path inside a Zephyr app.

Typical pieces:

- model data compiled into the app
- tensor arena buffer
- `MicroMutableOpResolver`
- `MicroInterpreter`
- input fill, invoke, and output readback

Suggested sequence:

1. Start with a small quantized model.
2. Add only the operators required by that model.
3. Bring up inference with a conservative tensor arena size.
4. Measure runtime and reduce memory once the path is stable.

## Example 3: Raw vs Prebuilt Module Comparison

Use the raw module when:

- you need to inspect or modify the runtime
- you are debugging kernel or integration issues
- your configuration is outside the published prebuilt matrix

Use the prebuilt bundle when:

- you want a faster initial setup
- you want a release-pinned module artifact
- your target fits the supported prebuilt matrix

## Example 4: Application Integration Checklist

Before debugging model behavior, verify:

- the correct module path is listed in `ZEPHYR_EXTRA_MODULES` or the west manifest
- `CONFIG_HELIA_RT=y` is enabled
- raw module users added `ns-cmsis-nn` when selecting the Ambiq backend
- the selected prebuilt flavor matches the target when using the prebuilt bundle
- the board console is enabled so initialization and inference logs are visible

For the full setup guide, see [Zephyr setup](../usage/zephyr.md).
