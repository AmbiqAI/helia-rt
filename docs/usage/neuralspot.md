# :material-speedometer: neuralSPOT Setup

Use this path when you want the fastest way to evaluate a model with heliaRT on Ambiq hardware.

The main entry point is `ns_autodeploy`, which can build, flash, and profile a `.tflite` model with a much shorter setup cycle than a hand-built application.

## Best For

- first-pass model evaluation
- profiling a model on hardware
- checking whether a model is a good fit before deeper integration work
- quickly comparing model variants on supported Ambiq targets

## What You Need

- an Ambiq target board
- a `.tflite` model
- a working `neuralSPOT` environment with `ns_autodeploy`

## Quick Start

```bash
ns_autodeploy --tflite-filename=mymodel.tflite --model-name mymodel
```

This flow can:

- compile an application using heliaRT
- flash the target
- run the model on hardware
- collect profiling information for layer-level analysis

## When To Use This Path

Choose `ns_autodeploy` when your immediate question is:

- does this model run on my Ambiq target?
- how large is the tensor arena?
- which layers dominate runtime?
- how do two model variants compare on hardware?

Choose [Zephyr setup](zephyr.md) instead when you are integrating heliaRT into a Zephyr product or application workspace.

## Typical Workflow

1. Start with a `.tflite` model.
2. Run `ns_autodeploy` to build and flash a profiling run.
3. Inspect runtime and layer-level profiling output.
4. Iterate on model choice, quantization, or operator mix.
5. Move to Zephyr or another source-based integration once the model is validated.

## Expected Outcomes

From this path, users typically want to learn:

- whether the model fits and runs
- how fast the model executes on the target
- which operators are expensive
- whether the model is ready for deeper application integration

## Notes

- This is the quickest way to get useful hardware data from heliaRT.
- It is a deployment and profiling path, not the only integration path.
- Once the model is validated, many teams move to [Zephyr setup](zephyr.md) for application integration.

Learn more about `ns_autodeploy` in the Ambiq `neuralSPOT` documentation:

- [neuralSPOT tools documentation](https://ambiqai.github.io/neuralSPOT/tools/index.html)
