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
ordinary bug fixes and are the versions Ambiq expects to reproduce reports
against. Older minors are out of ordinary support, though they may still be
eligible for critical fixes — see below.

| Track | Version | Status |
|---|---|---|
| Current minor | `v1.17.x` | Supported |
| Previous minor | `v1.16.x` | Supported |
| Older minors | `v1.15.x` and earlier | Out of ordinary support |

<!-- TODO(adam): confirm this table should be maintained by hand here, or
     generated from the release list at release time. As written it goes stale
     the moment a release lands. -->

<!-- TODO(adam): #200 says "current and previous minor releases" but does not
     say whether being supported requires running the latest patch of that
     minor, nor whether the window is scoped to the current major. Confirm the
     intent before publishing. -->

!!! note "Release tag naming"
    Releases from `v1.16.0` onward are tagged `helia-rt-v<version>`. Releases up
    to and including `v1.15.0` use the legacy `heliaRT-v<version>` form. Both
    naming schemes appear on the
    [Releases page](https://github.com/AmbiqAI/helia-rt/releases); the legacy
    tags were not renamed.

## Critical And Security Fixes

Ambiq provides **critical/security fixes for 12 months**.

<!-- TODO(adam): #200 does not define when the 12 months starts — release date
     of the affected version, or the date it was superseded — nor what
     qualifies a fix as critical/security. Both need to be settled before this
     is a policy anyone can rely on. -->

For anything you believe is a security issue, follow the process in
[`SECURITY.md`](https://github.com/AmbiqAI/helia-rt/blob/main/SECURITY.md)
rather than opening a public issue.

## Deprecation Notice

When a supported capability is scheduled for removal, Ambiq gives **at least 90
days deprecation notice**.

<!-- TODO(adam): #200 does not say where notice is published (release notes,
     CHANGELOG, docs banner, direct customer notice) or what counts as a
     deprecable surface — API, backend, toolchain, board, package format.
     Name the channel here so the commitment is checkable. -->

## Known Unsupported Behavior

**Stateful/streaming quantized HELIA LSTM is unsupported until a released fix is
verified.** As of the latest release, `v1.17.0`, no such fix has shipped: the
state-persistence change landed on `main` in
[#197](https://github.com/AmbiqAI/helia-rt/pull/197) after `v1.17.0` was cut.
Treat quantized `UNIDIRECTIONAL_SEQUENCE_LSTM` as single-shot until a release
containing that fix is published and verified.

<!-- TODO(adam): docs/reference/operator-coverage.md currently describes the
     HELIA int8/int16 LSTM kernels as stateful across invocations, which reads
     as a stronger claim than #200 allows for released versions. That page
     needs a matching correction — deliberately not edited here, since it is a
     coverage claim. -->

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
