# Troubleshooting

<!-- TODO: Step 5 — Full content -->

!!! note "Work in progress"
    This page is part of the documentation refresh tracked in [#135](https://github.com/AmbiqAI/helia-rt/issues/135).

## Build Issues

### Missing HELIA backend module

```
error: ns-cmsis-nn module not found
```

The HELIA backend requires the private `ns-cmsis-nn` module. If you don't have access, build with `OPTIMIZED_KERNEL_DIR=cmsis_nn` or leave it empty for Reference kernels. Contact [support.aitg@ambiq.com](mailto:support.aitg@ambiq.com) for access.

### Third-party download failures

<!-- TODO: Common curl/wget failures, proxy config, manual download fallback -->

## Link Issues

### ITCM overflow

<!-- TODO: Describe the linker error and how to resolve (custom .ld, reduce arena size, switch to SRAM) -->

### Duplicate symbol errors

<!-- TODO: Multiple OpResolver registrations, mixing static + source -->

## Runtime Issues

### Arena too small

<!-- TODO: How to diagnose and resize -->

### Incorrect output values

<!-- TODO: Quantisation mismatch, wrong model format -->

## Getting More Help

- [Submit a GitHub Issue](https://github.com/AmbiqAI/helia-rt/issues/new/choose)
- [Contact Ambiq AITG](mailto:support.aitg@ambiq.com)
