<!--
  DRAFT — awaiting approval. Every statement on this page is drawn from the
  agreed product decisions recorded in
  https://github.com/AmbiqAI/helia-rt/issues/200. Do not extend it with support
  commitments that are not in that issue.
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

<!-- TODO(adam): does the 12-month critical-fix window extend to minors outside
     current/previous, or only apply within them? #200 does not say. -->

<!-- TODO(adam): the version column publishes only after #214 ("align root CMake
     project version with release-please") lands, so the release header, root
     CMake, and pack manifest all report the same version. Sequence accordingly. -->

<!-- TODO(adam): confirm this table should be maintained by hand here, or
     generated from the release list at release time. As written it goes stale
     the moment a release lands. -->

<!-- TODO(adam): #200 says "current and previous minor releases" but does not
     say whether being supported requires running the latest patch of that
     minor, nor whether the window is scoped to the current major. Confirm the
     intent before publishing. -->

!!! note "Release tag naming"
    Releases from `v1.16.0` onward are tagged `helia-rt-v<version>`.
    `v1.5.0`–`v1.15.0` use `heliaRT-v<version>` — `v1.7.0` carries both
    `HeliaRT-v1.7.0`, which holds the release, and `heliaRT-v1.7.0`.
    `v1.0.0`–`v1.3.0` predate the prefix and are tagged `v<version>`. All three
    schemes appear on the
    [Releases page](https://github.com/AmbiqAI/helia-rt/releases); legacy tags
    were not renamed.

## Critical And Security Fixes

Ambiq provides **critical/security fixes for 12 months**.

<!-- TODO(adam): #200 does not define when the 12 months starts — release date
     of the affected version, or the date it was superseded — nor what
     qualifies a fix as critical/security. Both need to be settled before this
     is a policy anyone can rely on. -->

Report anything you believe is a security issue to
[support.aitg@ambiq.com](mailto:support.aitg@ambiq.com) rather than opening a
public issue.

<!-- TODO(adam): the repository's SECURITY.md is a one-line redirect to Google's
     TensorFlow security policy, inherited from upstream. Routing reporters
     there sends Ambiq's security intake to another company. Deliberately not
     linked above. Decide who rewrites SECURITY.md with a real Ambiq channel,
     and confirm support.aitg@ambiq.com (README.md:97, pyproject.toml) is the
     right intake address for security specifically. -->

<!-- TODO(adam): confirm whether Ambiq commits to an acknowledgement or triage
     time for security reports. #200 does not say, so none is stated here. -->

## Deprecation Notice

When a supported capability is scheduled for removal, Ambiq gives **at least 90
days deprecation notice**.

<!-- TODO(adam): #200 does not say where notice is published (release notes,
     CHANGELOG, docs banner, direct customer notice) or what counts as a
     deprecable surface — API, backend, toolchain, board, package format.
     Name the channel here so the commitment is checkable. -->

## Known Unsupported Behavior

**Stateful/streaming quantized HELIA LSTM is unsupported until a released fix is
verified.** No release up to and including `v1.17.0` contains this fix: the
state-persistence change landed on `main` in
[#197](https://github.com/AmbiqAI/helia-rt/pull/197) after `v1.17.0` was cut.
Check the release notes of any release after `v1.17.0` before relying on
stateful behavior. Until then, treat quantized
`UNIDIRECTIONAL_SEQUENCE_LSTM` as single-shot.

<!-- TODO(adam): docs/reference/operator-coverage.md carries TWO state claims
     that a corrective pass must decide about explicitly:
       - :27 describes the HELIA int8/int16 LSTM kernels as stateful across
         invocations, which is stronger than #200 allows for released versions.
       - :115 says FP32/FP16 "carry across invocations requires ns-cmsis-nn
         v7.29.0+", which #200's quantized-only scope does not cover at all —
         so it is neither confirmed nor contradicted by the product decision.
     Deliberately not edited here; both are coverage claims. -->

<!-- TODO(adam): confirm whether "stateful/streaming" in #200 also covers the
     FP16/FP32 paths documented in docs/guides/floating-point.md, or is
     quantized-only as its wording suggests. -->

## Where heliaRT May Run

Support scope and license scope are different things. Development, test, and CI
use is broad; production deployment is not. See
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
