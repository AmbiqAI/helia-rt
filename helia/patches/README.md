# helia/patches — upstream override patches

This directory holds unified-diff patches that the helia build applies to
upstream files at build time. Use it instead of editing upstream files
directly when:

- The change is too large for the [inline drift policy](../docs/repository_layout.md#inline-drift-policy)
  (more than ~5 lines, or non-trivial logic), AND
- No extension hook (`helia.inc`, `kernels/helia/`, `helia_tests.inc`,
  `*_helia.sh`, etc.) can express the change.

Building via `tools/ci_build/build_helia.sh` automatically invokes
`helia/patches/apply.sh` before the make build, so contributors normally
don't have to apply patches manually.

## Layout

```
helia/patches/
├── README.md              # this file
├── apply.sh               # idempotent patch applicator (called from build_helia.sh)
├── inline_drift.md        # inventory of unavoidable in-place edits to upstream files
└── applied/               # the patches themselves
    ├── 0001-<short-name>.patch
    └── ...
```

## Authoring a new patch

1. **First, prove there is no extension hook.** Check
   [`helia/docs/repository_layout.md`](../docs/repository_layout.md). If a
   hook fits, use it; do not write a patch.
2. **Make the change in your working tree** as if you were editing the
   upstream file directly.
3. **Generate the patch:**
   ```sh
   git diff --no-color -- tensorflow/lite/micro/<file> \
     > helia/patches/applied/NNNN-<short-name>.patch
   ```
   `NNNN` is a 4-digit ordinal. Patches are applied in lexical order.
4. **Revert your working-tree change.** The patch will be re-applied by
   `apply.sh` on the next build, which is the only place patches should be
   applied from.
5. **Verify idempotency:** running `helia/patches/apply.sh` twice in a row
   must succeed without error and produce no additional diff.
6. **Document it** with a one-line entry under
   [`inline_drift.md`](inline_drift.md) including:
   - file modified
   - reason the change cannot live in an extension hook
   - link to the upstream issue/PR (if any) that would let us drop the patch

## Removing a patch

Patches are *technical debt*. When upstream takes our change (or makes the
change unnecessary), delete the patch file and the matching entry in
`inline_drift.md`. The next sync from upstream will pick up the relevant
upstream code.

## Why patches and not forks of files?

A maintained patch set:

- Is small enough to read in one sitting.
- Makes upstream sync conflicts explicit (the patch fails to apply →
  someone has to look at it) instead of silent (a forked file silently
  diverges from upstream).
- Lets us cleanly drop changes once upstream accepts them.
