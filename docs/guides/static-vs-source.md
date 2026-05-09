# Static vs Source Builds

<!-- TODO: Step 5 — Full content -->

!!! note "Work in progress"
    This page is part of the documentation refresh tracked in [#135](https://github.com/AmbiqAI/helia-rt/issues/135).

## Overview

heliaRT ships in two distribution forms:

- **Static (prebuilt)** — pre-compiled `.a` archives per architecture/toolchain/variant. Fastest integration path.
- **Source** — full source tree. Maximum flexibility and debuggability.

## When to Use Each

| Criterion | Static | Source |
|---|---|---|
| Time to first build | Minutes | Longer (downloads + compile) |
| Debuggability | Limited (no source stepping) | Full |
| Custom kernel changes | Not possible | Yes |
| CI reproducibility | Pinned to release tag | Tracks HEAD or tag |

## File Layout

<!-- TODO: Show the release artifact naming convention and directory structure -->

## Next Steps

- [SPEED vs SIZE variants](speed-vs-size.md)
- [Toolchain selection](toolchains.md)
