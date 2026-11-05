---
title: Migrate to heliaRT 2.5
description: Update source and build integrations for the LiteRT Micro baseline in heliaRT 2.5.
---

heliaRT 2.5.0 is planned and has not been released. This guide describes the upstream update already on `main`; a release date and transition notice are still pending. Continue using a supported published release for production.

The planned release updates the LiteRT Micro baseline and requires C++17. It includes source and build compatibility changes from 2.2.2. Update application includes and build settings, rebuild the runtime and application together, and validate your models before deployment.

The documented model, resolver and interpreter workflow remains available. Qualified names such as `tflite::MicroInterpreter` remain available through compatibility declarations; this does not preserve forward declarations or binary compatibility. See the [support policy](/helia-rt/guide/maintenance/support/) for supported releases and transition notices.

## Build with C++17

For source CMake builds, remove `HELIA_RT_CXX_STANDARD=14` or set `HELIA_RT_CXX_STANDARD=17`. Values below 17 fail configuration. The supplied runtime targets export the `cxx_std_17` compile requirement to consumers.

For Zephyr applications, use `CONFIG_STD_CPP17=y` as shown in the [Zephyr integration guide](/helia-rt/getting-started/zephyr/). Rebuild the selected runtime module with the application.

Independent source builds must define `TFLM_REMOVE_VIRTUAL_DELETE` consistently for the runtime and its consumers. Use the supplied build integration's definitions when linking its runtime target. Match archive, headers, compiler, target and kernel settings; do not combine old headers with a new archive.

## Update schema includes

Replace:

```cpp
#include "tensorflow/lite/schema/schema_generated.h"
```

with:

```cpp
#include "tensorflow/lite/micro/schema/schema_generated.h"
```

`Model` and `GetModel` retain their `tflite::` names. See [First inference](/helia-rt/getting-started/first-inference/) for the complete model-loading and invocation sequence.

The canonical header for `TfLiteTensor`, `TfLiteEvalTensor`, `TfLiteStatus` and `TfLiteType` is now `tensorflow/lite/micro/micro_common.h`. The previous `tensorflow/lite/micro/c/common.h` and `c/c_api_types.h` paths remain forwarding headers in this release.

## Replace old runtime forward declarations

Runtime classes now live in `tflite::micro`. Compatibility declarations preserve ordinary qualified uses such as `tflite::MicroInterpreter` and `tflite::MicroMutableOpResolver<N>`.

Replace declarations such as `namespace tflite { class MicroInterpreter; }` with an include of the corresponding public header:

```cpp
#include "tensorflow/lite/micro/micro_interpreter.h"
```

Apply the same check to forward declarations of allocators, planners, profilers and resource-variable classes. Consult the [API reference](/helia-rt/reference/) for their current headers and declarations.

## Review custom runtime integrations

`TfLiteContext` services are fixed method wrappers rather than assignable callback fields. Custom operators can call these services, but code that assigns `AllocatePersistentBuffer`, `RequestScratchBufferInArena`, `GetScratchBuffer`, `GetEvalTensor` or `ReportError` must be adapted. Do not replace a runtime-owned context or its `impl_` with an unrelated object.

Integrations using upstream internal headers need a separate review. For fixed-point helpers, the new header is `tensorflow/lite/micro/kernels/internal/fixedpoint.h`, with helpers in `tflite::micro`. Review direct dependencies on removed `micro_tensor_utils.h` and `tflite_bridge/*` sources against the new upstream implementation.

gemmlowp and ruy are no longer bundled dependencies. Remove assumptions that their include directories exist, including directory checks and imported-target include lists. Supply any dependency your own application still needs independently.

## Validate model behavior

Re-run representative input/output checks using the selected runtime, backend, kernel profile and toolchain. Changes to multiplier calculations can alter quantized results; do not assume bit-exact output compatibility with 2.2.2. Check stateful models across repeated invocations as well as individual inputs. Preserve application accuracy requirements when reviewing changed results.

Check dependent tools and generated firmware against the new headers and package layout. A passing runtime test does not establish compatibility for a profiler, estimator or firmware wrapper.
