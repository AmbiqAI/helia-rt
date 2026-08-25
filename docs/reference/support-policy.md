<!--
  Wording approved by the portfolio authority (Dr. Adam Page) on 2026-08-25.
  The commitments here come from the agreed product decisions in
  https://github.com/AmbiqAI/helia-rt/issues/200 and the sign-off recorded on
  PR #215. Do not add support commitments that appear in neither source.
-->

# Support Policy

This page states which heliaRT releases Ambiq supports, how long critical fixes
stay available, and how much notice you get before something is removed.

## Supported Releases

Ambiq supports the **current and previous minor releases**.

Practically, that means the newest minor version and the one before it receive
bug fixes and are the versions Ambiq expects to reproduce reports against.

| Track | Version | Status |
|---|---|---|
| Current minor | `v1.17.x` | Supported |
| Previous minor | `v1.16.x` | Supported |
| Older minors | `v1.15.x` and earlier | Not supported |

Being supported means running the **latest patch release** of a supported minor.
A defect that is already fixed in a later patch of your minor is addressed by
upgrading to that patch.

<!-- Maintenance: this table is hand-maintained. Update it at each release, as
     part of the release checklist. -->

<!-- Note: how the supported window interacts with major versions is recorded as
     "revisit at 2.0". Nothing about it is stated on this page today. -->

<!-- Sequencing: the version column should publish after #214 ("align root CMake
     project version with release-please") lands, so the release header, root
     CMake, and pack manifest all report the same version. -->

!!! note "Release tag naming"
    Releases from `v1.16.0` onward are tagged `helia-rt-v<version>`.
    `v1.5.0`–`v1.15.0` use `heliaRT-v<version>` — `v1.7.0` carries both
    `HeliaRT-v1.7.0`, which holds the release, and `heliaRT-v1.7.0`.
    `v1.0.0`–`v1.3.0` predate the prefix and are tagged `v<version>`. All three
    schemes appear on the
    [Releases page](https://github.com/AmbiqAI/helia-rt/releases); legacy tags
    were not renamed.

## Critical And Security Fixes

Ambiq provides **critical/security fixes for 12 months**. The 12 months run from
the **release date of the minor**, not from the date it was superseded.

A fix is *critical* if it addresses either of:

- a security vulnerability, or
- a wrong-inference or data-corruption defect with no workaround.

This window applies within the current and previous minors only. Minors older
than that are not eligible for critical fixes, whatever their release date.

### Reporting A Security Issue

Report anything you believe is a security issue to
[support.aitg@ambiq.com](mailto:support.aitg@ambiq.com) rather than opening a
public issue. GitHub private vulnerability reporting is planned — see
[#219](https://github.com/AmbiqAI/helia-rt/issues/219) — but is not yet the
intake channel.

<!-- TODO(adam): SECURITY.md is still the inherited one-line redirect to
     Google's TensorFlow policy, so it is deliberately not linked above. Its
     rewrite is tracked in #219; drop this comment when that lands. -->

<!-- TODO(adam): no acknowledgement or triage time is committed here, because
     none was decided. Add one if that changes. -->

## Deprecation Notice

When a supported capability is scheduled for removal, Ambiq gives **at least 90
days deprecation notice**.

Notices are published in three places, so no single channel has to be watched:

- the release notes for the release that announces the deprecation,
- `CHANGELOG.md`, and
- a banner on this documentation site.

The surfaces covered by this commitment are the public API, build options, and
supported toolchains and targets.

## Known Unsupported Behavior

**Stateful/streaming quantized HELIA LSTM is unsupported until a released fix is
verified.** No release up to and including `v1.17.0` contains this fix: the
state-persistence change landed on `main` in
[#197](https://github.com/AmbiqAI/helia-rt/pull/197) after `v1.17.0` was cut.
Check the release notes of any release after `v1.17.0` before relying on
stateful behavior. Until then, treat quantized
`UNIDIRECTIONAL_SEQUENCE_LSTM` as single-shot.

This statement is quantized-only. It says nothing about the FP16 and FP32 LSTM
paths — see [FP16 and FP32](../guides/floating-point.md) for those.

<!-- TODO(adam): docs/reference/operator-coverage.md carries TWO state claims
     that a corrective pass must decide about explicitly:
       - :27 describes the HELIA int8/int16 LSTM kernels as stateful across
         invocations, which is stronger than #200 allows for released versions.
       - :115 says FP32/FP16 "carry across invocations requires ns-cmsis-nn
         v7.29.0+", which #200's quantized-only scope does not cover at all —
         so it is neither confirmed nor contradicted by the product decision.
     Deliberately not edited here; both are coverage claims. -->

## Where heliaRT May Run

Support scope and license scope are different things. Development, testing,
validation, benchmarking, CI, emulation, and simulation are permitted on
non-Ambiq CPUs; production or commercial deployment is licensed for
Ambiq-manufactured CPUs only. See
[Attribution and Trademarks](attribution.md#license-boundaries) for the license
boundary, and the repository
[LICENSE](https://github.com/AmbiqAI/helia-rt/blob/main/LICENSE) for the
governing terms.

## Getting Help

- [Submit an issue](https://github.com/AmbiqAI/helia-rt/issues/new/choose)
- [Contact Ambiq AITG](mailto:support.aitg@ambiq.com)

## Next Steps

- [Release Process](../contributing/release-process.md) — how versions are cut and tagged
- [Operator Coverage](operator-coverage.md) — per-backend kernel status
- [Attribution and Trademarks](attribution.md) — license scope and boundaries
