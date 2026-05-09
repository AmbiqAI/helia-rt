# Drift Reduction Opportunities

This file tracks helia-rt drift items where **upstream has moved on** and
helia could likely just re-adopt upstream's current version with minimal
validation. Each row in [`inline_drift.md`](inline_drift.md) that lists a
"reduction plan" should be cross-referenced here.

## Active queue

_(empty)_

## Recently resolved

- **2026-05** `ci/install_bazelisk.sh`, `ci/install_buildifier.sh`,
  `ci/sync_from_upstream_tf.sh`, `ci/tflite_files.txt`,
  `ci/Dockerfile.micro` — all re-adopted from `tflm/main` verbatim.
  `ci/Dockerfile.micro` is dead code in helia (the `helia-rt-ci` image is
  built from `.devcontainer/Dockerfile`), so re-adopting only reduces sync
  conflict surface. Bazelisk bumped 1.16.0 → 1.27.0; buildifier 4.2.3 →
  8.2.1; sync script switched to upstream's LiteRT-source allow-list flow;
  `ci/tflite_files.txt` had two latent bugs corrected (extra `ci/tflite_files.txt`
  line dropped, missing `tensorflow/compiler/mlir/lite/core/api/error_reporter.h`
  line added).

## Execution policy

- Each item is its own commit/PR (`chore: bump <tool> to upstream`).
- After landing, **delete the corresponding row** from this file's active
  queue (move to "Recently resolved") and from the corresponding section
  of `inline_drift.md`.
