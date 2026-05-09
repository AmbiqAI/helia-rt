# Upstream Sync

<!-- TODO: Step 9 — Full content -->

!!! note "Work in progress"
    This page is part of the documentation refresh tracked in [#135](https://github.com/AmbiqAI/helia-rt/issues/135).

## Overview

heliaRT is periodically replanted on upstream [tflite-micro](https://github.com/tensorflow/tflite-micro) `main`. This page explains the process and the blast-radius reduction philosophy.

## Replant Workflow

<!-- TODO: Step-by-step replant process -->

1. Create a fresh branch from upstream `tflm/main`
2. Cherry-pick or merge Ambiq-specific commits
3. Resolve conflicts (intentionally minimal due to isolation strategy)
4. Run CI to validate
5. Squash-merge into `main`

## Blast-Radius Philosophy

- **Don't edit upstream files** when possible — use additive files instead.
- **Upstream workflows are API-disabled** (not YAML-edited) via `ci/disable_upstream_workflows.sh`.
- **Ambiq kernels** live in a dedicated directory (`kernels/helia/`) to avoid touching upstream kernel files.

## Post-Sync Checklist

- [ ] Run `./ci/disable_upstream_workflows.sh` to re-disable any upstream workflows
- [ ] Verify operator coverage matrix is still accurate
- [ ] Run full CI matrix

## Next Steps

- [Architecture](architecture.md)
- [Release Process](release-process.md)
