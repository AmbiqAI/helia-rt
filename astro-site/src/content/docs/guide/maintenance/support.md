---
title: Support policy
description: Supported releases, critical fixes, deprecation notice and help channels.
---

Ambiq supports the current and previous heliaRT minor releases. Use the latest patch release of a supported minor when reproducing or reporting an issue.

| Track | Version | Status |
| --- | --- | --- |
| Current minor | `v1.21.x` | Supported |
| Previous minor | `v1.20.x` | Supported |
| Older minors | `v1.19.x` and earlier | Not supported |

`v1.18.0` is withdrawn because its heliaCORE pin is defective for the float32 and float16 kernels, including NaN handling and FP16 LSTM. It does not count as a supported minor. Upgrade to a supported release; its tag remains for reproducibility.

These commitments follow the recorded [support decision](https://github.com/AmbiqAI/helia-rt/issues/200) and [approved policy](https://github.com/AmbiqAI/helia-rt/pull/215). Release versions are maintained with the runtime's [changelog](https://github.com/AmbiqAI/helia-rt/blob/main/CHANGELOG.md).

## Critical and security fixes

Ambiq provides critical/security fixes for 12 months from the minor release date, within the current and previous minor window. Older minors are not eligible simply because fewer than 12 months have passed.

A critical defect is a security vulnerability or a wrong-inference/data-corruption defect with no workaround. If a later patch of the supported minor already fixes the issue, upgrade to that patch.

Report security issues to [support.aitg@ambiq.com](mailto:support.aitg@ambiq.com) rather than opening a public issue. Use [GitHub issues](https://github.com/AmbiqAI/helia-rt/issues/new/choose) for ordinary bug reports and integration questions.

## Deprecation notice

Ambiq gives at least 90 days notice before removing a supported public API, build option, toolchain or target. Notices appear in release notes, `CHANGELOG.md` and a documentation-site banner.

## Stateful quantized LSTM

The quantized HELIA `UNIDIRECTIONAL_SEQUENCE_LSTM` state-persistence fix first shipped in `v1.18.0` and is carried by later releases. Because `v1.18.0` is withdrawn, obtain the fix from a supported release. Treat quantized HELIA LSTM in `v1.17.0` and earlier as single-shot. See [the fix](https://github.com/AmbiqAI/helia-rt/pull/197).

That statement is quantized-only. FP32 and FP16 LSTM behavior has its own RT/kernel version and operator restrictions; check [Floating point](/helia-rt/guide/floating-point/) and model compatibility.

## License scope

Support and licensing are separate. Development and validation on a host or simulator do not establish support for production deployment on that CPU. See [Attribution and licensing](/helia-rt/guide/maintenance/attribution/) and the governing license.
