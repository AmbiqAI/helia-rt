---
title: Upstream maintenance
description: Preserve LiteRT lineage while updating shared sources and integration manifests.
---

heliaRT is derived from [tensorflow/tflite-micro](https://github.com/tensorflow/tflite-micro). Keep the chosen upstream revision and downstream changes reviewable when updating the fork.

## Before a sync

Record the starting heliaRT and upstream revisions and inspect the [repository layout policy](https://github.com/AmbiqAI/helia-rt/blob/main/helia/docs/repository_layout.md), [patch instructions](https://github.com/AmbiqAI/helia-rt/blob/main/helia/patches/README.md) and [inline drift inventory](https://github.com/AmbiqAI/helia-rt/blob/main/helia/patches/inline_drift.md). Work on an isolated branch and preserve unrelated changes.

## Review the integration boundaries

- Resolver registrations and operator prepare/invoke signatures.
- Shared runtime, allocator and model-schema assumptions.
- Kernel basenames and common sources in the CMake manifest and Make lists.
- HELIA-specific tests and their registration.
- Third-party headers exported for CMake, Zephyr and packaging.
- Upstream file notices, patch applicability and any drift made unnecessary by the update.

Resolve conflicts by preserving the source contract rather than accepting one side wholesale. An upstream change can require adapter changes even when Git reports no textual conflict.

## Validate every affected path

Use the [testing matrix](/helia-rt/guide/maintenance/testing/) to distinguish reference tests from HELIA execution. Check source integrations and generated packages when manifests or headers change. Review model compatibility and authored examples against the resulting runtime API.

The repository includes [workflow management tooling](https://github.com/AmbiqAI/helia-rt/blob/main/ci/disable_upstream_workflows.sh). Inspect its scope and the repository's workflow policy before applying it: it changes GitHub workflow state. A source sync alone does not require blindly disabling every imported workflow.

## Record the result

Include the upstream revision, downstream adjustments, executed checks and untested target paths in the change description. Update the drift inventory and relevant documentation so the next sync starts from an accurate record. Follow [Releases](/helia-rt/guide/maintenance/releases/) for changes that alter public capability or compatibility.
