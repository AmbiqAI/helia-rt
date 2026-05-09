# Release Process

<!-- TODO: Step 9 — Full content -->

!!! note "Work in progress"
    This page is part of the documentation refresh tracked in [#135](https://github.com/AmbiqAI/helia-rt/issues/135).

## Overview

heliaRT uses [release-please](https://github.com/googleapis/release-please) for automated version bumps and changelog generation.

## How It Works

1. Conventional commits (`feat:`, `fix:`, `chore:`) on `main` trigger release-please.
2. A bot PR is opened (or updated) with version bumps and CHANGELOG entries.
3. Merging the bot PR cuts a GitHub Release with a tag.
4. The `helia_release.yml` workflow builds and attaches prebuilt archives to the release.

## Version Files

- `.release-please-manifest.json`
- `CHANGELOG.md`
- `tensorflow/lite/micro/heliart_version.h`
- `nsx/nsx-module.yaml`

## Artifact Naming

<!-- TODO: Show the naming convention for release archives per arch/toolchain/variant -->

## Next Steps

- [Upstream Sync](upstream-sync.md)
- [Architecture](architecture.md)
