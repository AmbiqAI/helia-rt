---
title: Releases
description: Version metadata, release artifacts and verification responsibilities.
---

heliaRT uses release-please to maintain its release PR, version metadata and changelog. The [release-please workflow](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/release-please.yml) runs on `main` and calls the artifact workflow with the created release tag.

## Version metadata

The [release configuration](https://github.com/AmbiqAI/helia-rt/blob/main/release-please-config.json) manages the public version header, NSX module metadata, root CMake version and CMake readme, along with the manifest and changelog. Review these together in a release PR.

The header macro identifies headers used to compile an application; retain archive provenance separately. See [Migrate from LiteRT](/helia-rt/getting-started/migrate-from-litert/#identify-the-runtime).

## Artifact generation

[helia_release.yml](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/helia_release.yml) checks out its supplied ref, builds the architecture/toolchain/flavor matrix and packages the outputs. The automatic release caller supplies the release tag, so release artifacts are built from that revision rather than whichever commit later becomes `main`.

The matrix includes `cortex-m4+fp` and `cortex-m55`, GCC/Arm Compiler 6/ATfE, and `debug`/`release_with_logs`/`release`. Packaging combines the outputs with headers and integration assets. Inspect the attached bundle and its contents rather than assuming a tag alone establishes successful delivery.

## Review a release

1. Review public capability changes, fixes, dependency pins and compatibility requirements in the changelog.
2. Ensure required checks refer to the release PR's exact head. If GitHub presents an approval gate, resolve it through the normal maintainer flow and wait for the checks.
3. Check version files and the [support policy](/helia-rt/guide/maintenance/support/), including the current/previous minor window and any withdrawn release.
4. After the release is created, verify artifact jobs, attached libraries, headers, bundle metadata and documentation delivery independently.
5. Record target execution evidence and remaining gaps. Do not describe a compiler's link probe as an executed model test.

The [release list](https://github.com/AmbiqAI/helia-rt/releases) is the source for published assets. A commit on `main`, a tag, a GitHub release and deployed documentation are separate states.

## Tag names

From `v1.16.0`, tags use `helia-rt-v<version>`. Older tags include `heliaRT-v<version>` and early unprefixed `v<version>` forms; `v1.7.0` also has a capitalized `HeliaRT-v1.7.0` release tag. Use the tag attached to the release you intend to reproduce rather than constructing an older tag name from a single convention.

## Documentation delivery

The [documentation workflow](https://github.com/AmbiqAI/helia-rt/blob/main/.github/workflows/docs.yml) builds the Astro site once, validates that artifact and deploys the tested output. Documentation changes on `main` do not require a product release.

The release caller invokes the reusable docs workflow with the release tag as `source_ref`. The workflow records the commit actually checked out as build provenance. Before publishing, it compares that source commit with the latest `main` commit so an older release build cannot replace newer documentation. There is one current public site rather than a site per release.

Review documentation build, validation and deployment results separately from firmware packaging. Use the [API reference](/helia-rt/reference/) for runtime contracts and links to authoritative declarations.
