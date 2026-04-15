# :material-speedometer: neuralSPOT Examples

These examples focus on model evaluation and profiling with `ns_autodeploy`.

## Example 1: First Model Bring-Up

Goal:
run a `.tflite` model on hardware quickly and confirm it executes end to end.

```bash
ns_autodeploy --tflite-filename=mymodel.tflite --model-name mymodel
```

Use this first when you want a fast answer to:

- does the model run on the target?
- is the memory footprint reasonable?
- do I need to reduce model complexity before integration?

## Example 2: Profile and Compare Models

Goal:
compare two or more candidate models on the same Ambiq target.

Typical comparison points:

- total inference latency
- layer-level hotspots
- tensor arena size
- operator mix

This is often the fastest way to choose between model variants before application integration.

## Example 3: Hand-Off to Zephyr

Once profiling stabilizes, move to [Zephyr setup](../usage/zephyr.md) when you need:

- application-level control
- board services and RTOS integration
- a product-style build and packaging flow

For more on the tool itself, see the [neuralSPOT tools documentation](https://ambiqai.github.io/neuralSPOT/tools/index.html).
