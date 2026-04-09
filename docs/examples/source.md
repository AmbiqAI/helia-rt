# :material-hammer-wrench: Source Integration Examples

Use source integration when you need direct control over how heliaRT is built and linked into your project.

## Typical Use Cases

- integrating into an existing embedded build system
- producing static libraries for a controlled toolchain and target
- experimenting with build flags or runtime internals
- preparing custom packaging beyond the published release bundles

## Example Flow

At a high level:

1. Clone the repository.
2. Choose target architecture, toolchain, and build type.
3. Build the static library.
4. Link it into your application.
5. Provide model data, tensor arena memory, and operator resolver setup in your app.

## When To Prefer Source Builds

Use this path instead of a prebuilt bundle when:

- you need a target or toolchain outside the supported prebuilt matrix
- you want to debug or modify runtime internals
- you want tighter control over archive generation and packaging

For the source-build guide, see [Source builds](../usage/source.md).
