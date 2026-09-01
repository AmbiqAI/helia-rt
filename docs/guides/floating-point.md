# FP16 and FP32 HELIA Kernels

heliaRT can dispatch selected TensorFlow Lite Micro floating-point operators to
the optimized FP32 and FP16 kernels supplied by
[AmbiqAI/ns-cmsis-nn](https://github.com/AmbiqAI/ns-cmsis-nn). The heliaRT
operator wrappers and the linked ns-cmsis-nn library must agree on which float
APIs are available.

## Feature contract

ns-cmsis-nn v7.28.0 or later exports these definitions from its CMake target:

```text
ARM_NN_ENABLE_F32=0|1
ARM_NN_ENABLE_F16=0|1
```

They serve two purposes:

1. ns-cmsis-nn uses the corresponding CMake options to select the FP32 and FP16
   kernel sources compiled into its library.
2. heliaRT uses the exported definitions to compile only calls supported by
   that library.

When both projects are built from source through CMake, the resolved
ns-cmsis-nn target is authoritative. heliaRT adopts its exported values so the
operator wrappers and kernel archive cannot be compiled with conflicting float
definitions.

### Source of truth by build system

| Build system | Float-feature source of truth | How alignment works |
|---|---|---|
| Make | `tensorflow/lite/micro/tools/make/ext_libs/helia.inc` | Make compiles the heliaRT C++ wrappers and selected ns-cmsis-nn C sources directly into the same `libtensorflow-microlite.a`. `helia.inc` selects the FP sources and adds matching `ARM_NN_ENABLE_F32/F16` definitions to both C and C++ compilation. |
| Zephyr | Final Kconfig values | Kconfig resolves before either module's CMake file runs. Both heliaRT and ns-cmsis-nn consume the same `CONFIG_NS_CMSIS_NN_ENABLE_F32/F16` values while compiling their respective libraries. |
| NSX/CMake | The resolved ns-cmsis-nn target | ns-cmsis-nn selects its sources and exports `ARM_NN_ENABLE_F32/F16`. heliaRT reads and adopts those definitions, then links ns-cmsis-nn transitively into the application. |

An ordinary CMake static library does not physically absorb another static
library. In source NSX/CMake builds, `helia_rt::helia` carries
`nsx::cmsis_nn` in its public link interface, so the final application link
includes both archives. The Make release builder is different: it compiles both
projects' objects into one combined archive.

!!! important
    Resolve the CMake feature options during **configuration**, before
    `add_subdirectory()` processes ns-cmsis-nn — either as `-D` arguments on the
    `cmake` command line, or as a `set(... CACHE BOOL "" FORCE)` earlier in the
    parent `CMakeLists.txt`. Adding `-DARM_NN_ENABLE_F16=1` to compiler flags is
    not sufficient because source selection has already happened.

## Hardware support

The release archives support the following matrix:

| Target | Optimized FP32 | Optimized FP16 | Requirement |
|---|---:|---:|---|
| `cortex-m4+fp` | Yes | No | Cortex-M4 single-precision FPU |
| `cortex-m55` | Yes | Yes | Armv8.1-M with MVE floating point (MVEF/Helium) |

FP16 requires the compiler's `float16_t` support and an MVEF-capable target.
The Make build therefore enables FP16 only for `TARGET_ARCH=cortex-m55`.
Do not use the M55 FP16 archive on a Cortex-M55 configuration where floating
point MVE is disabled.

The published static-library matrix does not provide optimized floating-point
kernels for Cortex-M0 or Cortex-M4 builds without an FPU.

## Runtime behavior

The selected backend is fixed at build time, but each operator still validates
whether its tensor types, shapes, layout, padding, activation, and other
parameters are supported by the optimized kernel.

- FP32 operators generally fall back to the TFLM reference implementation when
  the optimized API is disabled or rejects the configuration.
- Many FP16 operators do not have a TFLM FP16 reference implementation. Where
  the limitation is knowable at graph preparation (FP16 disabled at build time,
  broadcast `ADD`/`MUL`, non-4D `PAD`, grouped `CONV_2D`, non-unit-beta
  `SOFTMAX`), the operator fails `AllocateTensors()` with a logged message;
  configurations the optimized kernel rejects at run time return
  `kTfLiteError` from `Invoke()` with a logged message.
- Grouped `CONV_2D` (input channels a multiple of filter channels) is outside
  the optimized float kernels' support: FP32 uses the reference kernel, FP16
  is rejected at `AllocateTensors()`.
- Pure data-movement operators (`TRANSPOSE`, `RESHAPE`) handle FP16 tensors
  bitwise through their 16-bit reference paths, so they work even without
  `ARM_NN_ENABLE_F16`.
- HELIA softmax supports only unit beta. FP32 softmax falls back to reference
  for non-unit beta; FP16 non-unit beta is unsupported.
- HELIA FP16/FP32 `UNIDIRECTIONAL_SEQUENCE_LSTM` preserves the TFLite hidden and
  cell state tensors across invocations when built with ns-cmsis-nn v7.29.0 or
  newer. With v7.28.0, optimized dispatch still requires zero initial state;
  FP32 falls back to reference for subsequent invocations, while FP16 stateful
  execution is unsupported.

See [Operator Coverage](../reference/operator-coverage.md) for the available
HELIA operator wrappers.

## Non-finite inputs (NaN and infinities)

The optimized floating-point kernels do not all treat NaN the way the TFLM
reference kernels do. The behavior differs by operator, and for `TANH` it also
differs by target, so it is stated here per case rather than as a single rule.

!!! warning "NaN is not a supported input to the optimized activation kernels"
    Feeding NaN to the optimized `TANH` or `LOGISTIC` kernels does not produce
    NaN. It produces a finite value at the activation's saturation bound. If
    your model can generate NaN and you rely on it propagating, do not use the
    optimized float activation path for that operator.

| Operator | Input | Optimized FP32/FP16 result | TFLM reference result |
|---|---|---|---|
| `TANH` | NaN, Armv8.1-M MVE targets (Cortex-M55) | Finite, negative, at the saturation bound | NaN |
| `TANH` | NaN, non-MVE targets (Cortex-M4, Cortex-M3) | NaN | NaN |
| `TANH` | ±Inf | ±1 | ±1 |
| `LOGISTIC` | NaN, all targets | Finite, at the upper saturation bound (1) | NaN |
| `LOGISTIC` | +Inf / −Inf | 1 / 0 | 1 / 0 |
| `ADD`, `MUL` | NaN | NaN, from the first ns-cmsis-nn release containing PR 380 (not yet cut); a finite activation bound before that | NaN |

Notes and version boundary:

- **`TANH` and `LOGISTIC` are by design.** ns-cmsis-nn documents NaN as
  unsupported input for these kernels: the vectorized `TANH` path uses
  `vminnmq`, which is IEEE `minNum` and returns the numeric operand against a
  quiet NaN, and the `LOGISTIC` path clamps its exponent input before
  evaluation. Restoring NaN would cost a compare and select in the vector loop
  body. This is not scheduled to change; ns-cmsis-nn issue 382 tracks NaN
  behavior for `RELU`/`RELU6`/`LEAKY_RELU`, a different function family, and
  does not cover `TANH` or `LOGISTIC`.
- **`ADD` and `MUL` are a fixed defect.** In the currently pinned v7.29.2 the
  output activation clamp discards NaN through compare-select ordering,
  returning an activation bound instead. ns-cmsis-nn PR 380 reclassifies NaN on
  the integer bit pattern, which holds at every optimization level. That fix is
  merged upstream but **not yet in any tagged release** -- the latest is v7.30.0
  (2026-08-30), which predates it. heliaRT gains the fixed behavior when its pin
  moves to the first release that contains PR 380.
- **The FP32 fallback softens this in practice.** Where an operator has a TFLM
  reference implementation, HELIA falls back to it when the optimized kernel
  declines the configuration, and the reference implementation propagates NaN
  normally. FP16 has no reference fallback.

## Make builds

The Make integration pins ns-cmsis-nn v7.29.2 and configures the float features
from `TARGET_ARCH`:

- FP32 is enabled for the HELIA backend.
- FP16 sources and declarations are enabled only for `cortex-m55`.

Build an M55 library with FP32 and FP16:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_generic \
  TARGET_ARCH=cortex-m55 \
  OPTIMIZED_KERNEL_DIR=helia \
  microlite
```

Build an M4+FP library with FP32 only:

```bash
make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_generic \
  TARGET_ARCH=cortex-m4+fp \
  OPTIMIZED_KERNEL_DIR=helia \
  microlite
```

## CMake source builds

### NSX entry points

An NSX application does not add the module subdirectories itself. It lists
modules in `nsx.yml`, and `nsx_bootstrap_app()` adds them in dependency order —
`nsx-cmsis-nn` before `nsx-helia-rt`. The `nsx` CLI drives CMake, and neither
`nsx configure` nor `nsx build` forwards `-D` options, so the float features
must be set in the application `CMakeLists.txt` **before** the bootstrap call:

```cmake
# ns-cmsis-nn reads these while its own CMakeLists is processed, and they drive
# both its source selection and the ARM_NN_ENABLE_F32/F16 definitions it
# exports. nsx-helia-rt is added afterwards, so it can only validate them.
set(NSX_CMSIS_NN_ENABLE_F32 ON CACHE BOOL "" FORCE)
set(NSX_CMSIS_NN_ENABLE_F16 ON CACHE BOOL "" FORCE)  # MVEF-capable targets only

include(${CMAKE_CURRENT_LIST_DIR}/cmake/nsx/modules.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/nsx/nsx_app_bootstrap.cmake)

nsx_bootstrap_app(
    APP_ROOT "${CMAKE_CURRENT_LIST_DIR}"
    BOARD "${NSX_BOARD}"
    MODULES ${NSX_APP_MODULES}
)

target_link_libraries(app PRIVATE nsx::helia_rt)
nsx_finalize_app(app)
```

The NSX HELIA backend **requires** FP32. NSX does not enable it for you, so
omitting the `set()` above is a configure-time `FATAL_ERROR` from
`nsx-helia-rt`, not a silent fallback. FP16 is optional; `nsx-helia-rt` emits a
warning when `CMAKE_SYSTEM_PROCESSOR` is `cortex-m55` and FP16 is left off,
because that target can use the optimized kernels.

!!! warning "ns-cmsis-nn revision"
    `NSX_CMSIS_NN_ENABLE_F32/F16` only exist, and only bridge to
    `ARM_NN_ENABLE_F32/F16`, in ns-cmsis-nn v7.28.0 or later. NSX registries
    that still pin an older `nsx-cmsis-nn` must be overridden in the app's
    `nsx.yml` (via a `module_registry` revision override, or `source.path` for
    a local working tree) before the float features can be enabled.

Because a static-archive build cannot reveal a missing kernel, verify the final
executable rather than the library:

```bash
arm-none-eabi-nm build/<board>/<app> | grep -c ' [tT] arm_.*_f16'
```

### Standalone CMake entry points

When adding the root ns-cmsis-nn project rather than its NSX module, use its
standalone option names:

```bash
cmake -S app -B build \
  -DARM_NN_ENABLE_F32=ON \
  -DARM_NN_ENABLE_F16=ON \
  -DHELIA_RT_ENABLE_HELIA=ON
cmake --build build
```

The parent project must add ns-cmsis-nn before heliaRT so
`helia_rt::helia` can resolve and inspect the dependency target:

```cmake
add_subdirectory(${NS_CMSIS_NN_DIR} ns-cmsis-nn)
add_subdirectory(${HELIA_RT_DIR} helia-rt)
target_link_libraries(app PRIVATE helia_rt::helia)
```

With an integer-only ns-cmsis-nn target, generic CMake still builds heliaRT:
FP32 uses reference fallbacks, while FP16 operators without reference support
remain unavailable.

## Source heliaRT with prebuilt ns-cmsis-nn

The ns-cmsis-nn NSX module can expose a prebuilt archive through
`NSX_CMSIS_NN_LIB`. As with the float features, these cache variables must be
set in the app `CMakeLists.txt` above `nsx_bootstrap_app()`, before the
nsx-cmsis-nn module is processed:

```cmake
set(NSX_CMSIS_NN_LIB      "/path/to/libns-cmsis-nn.a" CACHE FILEPATH "" FORCE)
set(NSX_CMSIS_NN_MANIFEST "/path/to/manifest.json"    CACHE FILEPATH "" FORCE)
set(NSX_CMSIS_NN_ENABLE_F32 ON CACHE BOOL "" FORCE)
set(NSX_CMSIS_NN_ENABLE_F16 ON CACHE BOOL "" FORCE)
```

The manifest records which float kernels were compiled into the archive.
ns-cmsis-nn validates the requested features against it and exports matching
`ARM_NN_ENABLE_F32/F16` definitions to heliaRT. A prebuilt archive without a
manifest cannot be verified by CMake; ns-cmsis-nn warns and trusts the requested
settings, so use the manifest distributed with the archive.

## Zephyr builds

Zephyr resolves Kconfig before module CMake processing, so both modules consume
the same final configuration:

```cfg
CONFIG_HELIA_RT=y
CONFIG_HELIA_RT_BACKEND_HELIA=y
CONFIG_NS_CMSIS_NN_ENABLE_F32=y
CONFIG_NS_CMSIS_NN_ENABLE_F16=y
```

The HELIA backend implies FP32 and implies FP16 when
`ARMV8_1_M_MVEF` is available. Explicit settings are useful when auditing a
product configuration. Kconfig prevents FP16 from being selected without MVEF.

Because Kconfig is resolved first, heliaRT and ns-cmsis-nn agree by
construction, and the Zephyr path performs no float-feature validation of its
own. `imply` is a weak default, though: an explicit
`CONFIG_NS_CMSIS_NN_ENABLE_F32=n` overrides it and silently drops the HELIA
float operators to reference implementations, with no warning and no error.
Check the generated `build/zephyr/.config` when auditing a configuration.

## Release static libraries

GitHub releases contain combined `libhelia-rt-*.a` archives. The Make build adds
both heliaRT objects and the selected ns-cmsis-nn kernel objects to the same
archive, so consumers do not link a separate ns-cmsis-nn library.

The release workflow builds:

| Archive target | Included float kernels |
|---|---|
| Cortex-M4+FP | FP32 |
| Cortex-M55 | FP32 and FP16 |

Each architecture is built for GCC, Arm Compiler 6, and ATfE in `debug`,
`release`, and `release_with_logs` variants. The release builder checks the
finished archive for representative FP32 kernel objects and, for M55, FP16
kernel objects. It also rejects FP16 objects in M4+FP archives.

Float support in a prebuilt heliaRT archive is fixed when that archive is
created. Application compiler definitions cannot add a kernel omitted from the
archive. Always choose the archive matching the target architecture, floating
point ABI, toolchain, and build variant.

## Migration notes for existing consumers

Enabling the float feature contract changes behavior for integrations built
against earlier heliaRT releases:

- **NSX apps**: the helia backend now fails configuration with a
  `FATAL_ERROR` unless `NSX_CMSIS_NN_ENABLE_F32` is set before
  `nsx_bootstrap_app()` **and** the resolved `nsx-cmsis-nn` module is
  v7.28.0 or newer. Every existing NSX helia application needs the two
  `set(... CACHE BOOL "" FORCE)` lines shown above plus a registry or
  `nsx.yml` revision bump.
- **Zephyr**: `CONFIG_HELIA_RT_BACKEND_HELIA` now `imply`s
  `NS_CMSIS_NN_ENABLE_F32/F16`. If your west workspace pins an ns-cmsis-nn
  module older than v7.28.0, those Kconfig symbols do not exist and the
  configuration step emits undefined-symbol warnings; update the module
  revision to silence them and to get the float kernels.
- **Library size**: Make-based helia builds now always compile the FP32
  kernels (and FP16 on `cortex-m55`) into the combined archive. Integer-only
  models still reference them transitively through the operator wrappers, so
  build applications with `-ffunction-sections -fdata-sections` and link with
  `--gc-sections` (the standard embedded configuration) to keep unused float
  code out of flash.

## Troubleshooting

`undefined reference to arm_*_f16`
: FP16 was enabled in heliaRT declarations but omitted from ns-cmsis-nn. Set
  the appropriate CMake option before adding ns-cmsis-nn, or select a prebuilt
  archive whose manifest reports FP16 support.

FP16 compiles but faults on target
: Confirm that the CPU and build flags enable Armv8.1-M MVE floating point.
  A Cortex-M55 core can be configured without MVEF.

FP32 unexpectedly uses reference code
: Confirm that the linked ns-cmsis-nn target exports
  `ARM_NN_ENABLE_F32=1` and that the operator configuration is supported by the
  optimized kernel.

Changing `CFLAGS` has no effect
: Source selection happens during CMake configuration. Set the CMake option
  (`NSX_CMSIS_NN_ENABLE_F16` for NSX, `ARM_NN_ENABLE_F16` standalone) before
  ns-cmsis-nn is added. For NSX apps this means a `set(... CACHE BOOL "" FORCE)`
  in the app `CMakeLists.txt` above `nsx_bootstrap_app()` — `nsx build` does not
  forward `-D` options.

`nsx-helia-rt: ... NSX_CMSIS_NN_ENABLE_F32 is not enabled`
: The NSX app did not set the float options before `nsx_bootstrap_app()`, or
  the resolved `nsx-cmsis-nn` predates v7.28.0 and does not define them.
