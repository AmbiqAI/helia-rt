# :material-book-open-page-variant: Examples

Examples are organized around practical bring-up and integration flows for heliaRT on Ambiq hardware.

## Example Tracks

- [Zephyr examples](zephyr.md): application skeletons and integration patterns for raw and prebuilt Zephyr module use.
- [neuralSPOT examples](neuralspot.md): model evaluation and profiling flows using `ns_autodeploy`.
- [Source integration examples](source.md): lower-level library build and embedding patterns.

## Common Example Themes

Examples focus on:

- Ambiq Apollo application bring-up
- model execution with a familiar TFLM-style interpreter flow
- operator resolver and tensor arena setup
- integration patterns aligned with supported Ambiq workflows

## Recommended Starting Points

- Start with [neuralSPOT setup](../usage/neuralspot.md) if you want the shortest path to profile a model.
- Start with [Zephyr setup](../usage/zephyr.md) if you are integrating heliaRT into an application workspace.
- Start with [source builds](../usage/source.md) if you need a custom build environment.
