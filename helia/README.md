# helia/ — heliaRT-specific assets and conventions

This directory holds documentation, override patches, and other assets that are
**specific to helia-rt** and have no upstream counterpart in
`tensorflow/tflite-micro`.

heliaRT is a fork of [`tensorflow/tflite-micro`](https://github.com/tensorflow/tflite-micro)
that is periodically synchronized with upstream. To keep that synchronization
sustainable, helia-specific code is intentionally segregated from upstream
files wherever possible.

## Layout

| Path | Purpose |
| --- | --- |
| `helia/README.md` | This file |
| `helia/docs/repository_layout.md` | Contributor guide: where helia-specific files are allowed to live, and where they are **not** allowed to live |
| `helia/patches/` | Source patches that override upstream files. Used **only** when an in-tree extension hook (helia.inc, helia_tests.inc, kernels/helia/, etc.) is not available |
| `helia/patches/README.md` | Patch authoring conventions and how patches are applied |
| `helia/patches/applied/*.patch` | Patches applied to upstream files during the build |

## Long-term reference to pre-replant helia main

The state of helia-rt's `main` branch immediately before it was replanted on
top of upstream `tensorflow/tflite-micro` is preserved in the annotated git
tag **`pre-tflm-rebase-2026-05`** (commit `01b125d5`). Use it for archaeology
of any historical Apollo customization that may not have made it through the
replant:

```sh
git fetch --tags origin
git show pre-tflm-rebase-2026-05:<path>             # view a single file
git diff pre-tflm-rebase-2026-05..main -- <path>    # diff against current main
```

The tag is force-protected and must not be moved.

## Contributor entry point

If you are contributing helia-specific code or build glue, **start with**
[`docs/repository_layout.md`](docs/repository_layout.md) — it documents the
directories where helia files belong and the rules for modifying upstream
files.
