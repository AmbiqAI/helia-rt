# :material-handshake: Contributing

Thank you for your interest in contributing to heliaRT. This section covers the project's design, workflows, and processes.

<div class="grid cards" markdown>

- :material-layers:{ .lg .middle } **Architecture**

    ---

    Source layout, kernel wiring, and design principles.

    [:octicons-arrow-right-24: Architecture](architecture.md)

- :material-sync:{ .lg .middle } **Upstream Sync**

    ---

    How the LiteRT for Micro replant works and blast-radius philosophy.

    [:octicons-arrow-right-24: Upstream Sync](upstream-sync.md)

- :material-tag:{ .lg .middle } **Release Process**

    ---

    release-please, version bumps, and artifact bundling.

    [:octicons-arrow-right-24: Release Process](release-process.md)

</div>

## Getting Started

1. [Open an issue](https://github.com/AmbiqAI/helia-rt/issues/new/choose) describing what you want to contribute and why.
2. Fork the repository and create a feature branch from `main`.
3. Read the [Architecture](architecture.md) page to understand the source layout.
4. Make changes following the guidelines below.
5. Open a pull request against `main`, linking the issue with `BUG=#nn`.

## PR Guidelines

- **One PR per concern** — keep PRs small and focused.
- **Conventional commits** — use `feat:`, `fix:`, `chore:`, `docs:` prefixes.
- **Pre-submit checks** before submitting:
    - `clang-format` for C/C++ style
    - `cpplint` for lint issues
    - Run `tensorflow/lite/micro/tools/ci_build/test_x86_default.sh` for basic tests
- **Merge, don't rebase** — do not force-push after review has started.

## License & CLA

heliaRT is released under the [Ambiq Apollo SDK License](https://github.com/AmbiqAI/helia-rt/blob/main/LICENSE). By contributing, you agree your changes are licensed under the same terms.

A [Google CLA](https://cla.developers.google.com/) is required for all contributions (inherited from upstream LiteRT for Micro).

## Code of Conduct

We follow [Google's Open Source Community Guidelines](https://opensource.google/conduct/). Be respectful and constructive.
