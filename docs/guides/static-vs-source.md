# Static vs Source Builds

heliaRT ships in two distribution forms. Choose based on how much control you need.

## Prebuilt Static Libraries

Every [GitHub release](https://github.com/AmbiqAI/helia-rt/releases) publishes a bundle containing pre-compiled `.a` archives covering the full matrix:

| Dimension | Values |
|---|---|
| Architecture | `cortex-m4+fp`, `cortex-m55` |
| Toolchain | `gcc`, `armclang`, `atfe` |
| Build variant | `debug`, `release`, `release_with_logs` |

Archive layout inside the bundle:

```
helia-rt-<tag>.zip
├── cortex-m4+fp/
│   ├── gcc/
│   │   ├── debug/
│   │   ├── release/
│   │   └── release_with_logs/
│   ├── armclang/
│   └── atfe/
├── cortex-m55/
│   └── ...
└── include/   # public headers
```

!!! tip "Fastest path"
    Unzip, point your build at the right `.a` + `include/`, and you're done.

## Source Builds

Clone the repository and build from source:

```bash
git clone https://github.com/AmbiqAI/helia-rt
cd helia-rt
make -f tensorflow/lite/micro/tools/make/Makefile \
    TARGET=cortex_m_generic \
    TARGET_ARCH=cortex-m55 \
    OPTIMIZED_KERNEL_DIR=helia \
    microlite
```

## When to Use Each

| Criterion | Static (prebuilt) | Source |
|---|---|---|
| Time to first build | Seconds (download + link) | Minutes (download deps + compile) |
| Debuggability | No source stepping | Full step-through debugging |
| Custom kernel changes | Not possible | Yes |
| CI reproducibility | Pinned to release tag | Tag or HEAD |
| Size | Smaller download | Full repo (~hundreds of MB) |
| Toolchain flexibility | Must match published toolchain | Any supported toolchain |

!!! info "Zephyr users"
    In a Zephyr workspace you can use _either_ approach:

    - **Source module:** add heliaRT as a west module and let CMake compile from source.
    - **Prebuilt bundle:** download and extract the release archive, then reference it as a library.

    See the [Zephyr getting-started guide](../getting-started/zephyr.md) for details.

## Next Steps

- [SPEED vs SIZE variants](speed-vs-size.md) — choose `release` or `release_with_logs`
- [Toolchain selection](toolchains.md) — pick `gcc`, `armclang`, or `atfe`
