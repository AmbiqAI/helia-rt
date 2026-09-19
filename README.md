# heliaRT

**Drop-in LiteRT for Micro with Ambiq-tuned kernels.**

[![Tests](https://github.com/AmbiqAI/helia-rt/actions/workflows/tests_entry.yml/badge.svg)](https://github.com/AmbiqAI/helia-rt/actions/workflows/tests_entry.yml)
[![Release](https://img.shields.io/github/v/release/AmbiqAI/helia-rt?label=latest)](https://github.com/AmbiqAI/helia-rt/releases/latest)
[![License](https://img.shields.io/badge/license-Ambiq%20Apollo%20SDK-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-ambiqai.github.io%2Fhelia--rt-cyan)](https://ambiqai.github.io/helia-rt/)

heliaRT is Ambiq's optimized LiteRT for Micro runtime for Apollo platforms, compatible with TensorFlow Lite for Microcontrollers / TFLM. It adds heliaCORE, a set of Ambiq-tuned kernel implementations, on top of the standard LiteRT for Micro API so you get faster inference without changing your application code.

## Start building

Choose an integration path, then select source builds or a compatible prebuilt archive. Match the runtime headers, compiler ABI, architecture and float settings to your application.

| Path | Guide |
|---|---|
| Zephyr / west | [Zephyr integration](https://ambiqai.github.io/helia-rt/getting-started/zephyr/) |
| neuralSPOT-X | [NSX integration](https://ambiqai.github.io/helia-rt/getting-started/neuralspot-x/) |
| Make / CMake source builds | [Build from source](https://ambiqai.github.io/helia-rt/getting-started/source/) |
| Prebuilt static libraries | [Use a release archive](https://ambiqai.github.io/helia-rt/getting-started/cmake/) |
| CMSIS-Pack | [Create and consume a source pack](https://ambiqai.github.io/helia-rt/getting-started/cmsis-pack/) |

## Documentation

- [Getting started](https://ambiqai.github.io/helia-rt/getting-started/): integration choices and first inference.
- [User guide](https://ambiqai.github.io/helia-rt/guide/): backends, SPEED/SIZE profiles, build flags, memory, profiling and troubleshooting.
- [Operators and data types](https://ambiqai.github.io/helia-rt/guide/operators/): searchable adapter coverage and input-type constraints.
- [API reference](https://ambiqai.github.io/helia-rt/reference/): runtime contracts and source declarations.
- [Maintenance](https://ambiqai.github.io/helia-rt/guide/maintenance/architecture/): repository structure, testing, upstream sync and releases.

The documentation source and authoring instructions live in [astro-site](astro-site/README.md). Edit those pages instead of duplicating integration instructions in the repository readme. The site records the source commit used for each build.

## License

heliaRT is released under the [Ambiq Apollo SDK License](LICENSE). Free use, modification, and redistribution for Ambiq-manufactured CPUs; **production or commercial deployment on non-Ambiq CPUs is not licensed**, while development, testing, validation, benchmarking, CI, emulation, and simulation may run on any CPU. See [LICENSE](LICENSE) for details.

This repository also contains code derived from or compatible with upstream TensorFlow Lite for Microcontrollers / LiteRT for Micro and other third-party projects. Those files retain their original copyright and license notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and trademark notes.

## Trademarks and Non-affiliation

LiteRT, TensorFlow, Google, and related marks are trademarks of Google LLC. heliaRT is developed and maintained by Ambiq and is not affiliated with, endorsed by, sponsored by, or otherwise associated with Google LLC.

## Getting Help

- [Submit an issue](https://github.com/AmbiqAI/helia-rt/issues/new/choose)
- [Contact Ambiq AITG](mailto:support.aitg@ambiq.com)
