# Troubleshooting

Common build, link, and runtime issues — and how to fix them.

## Build Issues

### Missing HELIA backend module

```
FATAL_ERROR: CONFIG_HELIA_RT_BACKEND_HELIA requires the ns-cmsis-nn Zephyr
module (ZEPHYR_NS_CMSIS_NN_MODULE_DIR). Add it to your west manifest or
select REFERENCE / CMSIS-NN.
```

**Cause:** The HELIA backend builds against the `ns-cmsis-nn` module, and Zephyr
cannot find it in your workspace. `ns-cmsis-nn` is a public repository — no
credential or access request is required — but it is a separate module that
your west manifest has to pull in.

**Fix:** Add it to your `west.yml`:

```yaml
- name: ns-cmsis-nn
  url: https://github.com/AmbiqAI/ns-cmsis-nn
  revision: v7.31.0
  path: modules/ns-cmsis-nn
```

Then run `west update ns-cmsis-nn`. Alternatively, switch backends:

- Open CMSIS-NN backend: `CONFIG_HELIA_RT_BACKEND_CMSIS_NN=y`
- Reference kernels: `CONFIG_HELIA_RT_BACKEND_REFERENCE=y`

### Missing CMSIS-NN module (Zephyr)

```
FATAL_ERROR: CONFIG_HELIA_RT_BACKEND_CMSIS_NN requires an open CMSIS-NN
Zephyr module.
```

**Fix:** Add the CMSIS-NN module to your `west.yml`:

```yaml
- name: cmsis-nn
  url: https://github.com/zephyrproject-rtos/cmsis-nn
  revision: main
  path: modules/lib/cmsis-nn
```

### Third-party download failures

```
Something went wrong with the CMSIS download: ...
```

**Cause:** The Makefile build auto-downloads CMSIS and ns-cmsis-nn on first run. This can fail behind corporate proxies.

**Fix:**

1. Set `https_proxy` / `http_proxy` environment variables
2. Or manually download and set `CMSIS_PATH` / `NS_CMSIS_NN_PATH`:

    ```bash
    export CMSIS_PATH=/path/to/cmsis
    export NS_CMSIS_NN_PATH=/path/to/ns_cmsis_nn
    ```

## Link Issues

### Flash / ITCM overflow

```
region `FLASH' overflowed by 12345 bytes
```

**Fix options:**

1. For source HELIA builds, switch to the **SIZE** kernel profile
2. Reduce your operator resolver — only register operators your model actually uses
3. For source builds, lower compiler optimization for non-kernel code where appropriate
4. Place code in a larger memory region via linker script

### Duplicate symbol errors

```
multiple definition of `tflite::ops::micro::Register_CONV_2D()'
```

**Cause:** Usually from linking both a prebuilt heliaRT archive _and_ compiling kernels from source.

**Fix:** Use one or the other — never both. Remove either the `.a` or the source files from your build.

## Runtime Issues

### Arena too small

```
AllocateTensors() failed
```

or

```
Failed to allocate memory for tensor arena
```

**Fix:**

1. Increase the arena size in your application code
2. Use `interpreter.arena_used_bytes()` after `AllocateTensors()` to find the actual minimum
3. Align arena to 16 bytes: `alignas(16) uint8_t tensor_arena[ARENA_SIZE];`

### Incorrect output values

**Possible causes:**

- **Quantization mismatch:** model expects int8 input but you're feeding float (or vice versa). Check `input->type`.
- **Wrong input scaling:** the input must match the model's quantization parameters (`input->params.scale` and `zero_point`).
- **Model not compatible:** ensure the `.tflite` was quantized for int8/int16 LiteRT for Micro, not float-only LiteRT.

### Model loads but no inference output

**Check:** Did you call `interpreter.AllocateTensors()` before `interpreter.Invoke()`? Forgetting this is the most common beginner mistake.

## Getting More Help

- [:octicons-issue-opened-16: Open a GitHub issue](https://github.com/AmbiqAI/helia-rt/issues/new/choose)
- [:octicons-mail-16: Contact Ambiq AITG](mailto:support.aitg@ambiq.com)
