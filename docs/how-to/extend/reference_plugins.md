# Reference Plugins

Three small plugins in `examples/plugins/` to copy from when you write your own. Each is a
folder you can install on its own, passes SensoryForge's contract check, and has a test
with a case worked by hand. They are in the source checkout, not in the installed package:
copy a folder out, or read it on GitHub.

| Folder | Adds | Registered as | Contract check |
|---|---|---|---|
| `sensoryforge-leaky-threshold/` | a neuron | `leaky_threshold` | `check_component("neuron", ...)` |
| `sensoryforge-exp-smoother/` | a temporal filter | `exp_smoother` | `check_component("filter", ...)` |
| `sensoryforge-ring-shape/` | a world shape | `ring` | `check_world_component("shape", "ring")` |

The formulas are textbook shapes chosen to show the contracts (a leaky integrate-and-fire
membrane, a single-pole low-pass, a Gaussian ring profile). They are not models of any real
cell or stimulus, and their parameter ranges are placeholders. Replace the formula, then the
test's hand-worked case, and keep the structure.

## What each one shows

- **Neuron.** Started with `sensoryforge new-component neuron leaky_threshold`. Shows the
  constructor that takes `noise_std` (the engine passes it to every neuron), `forward`
  returning the voltage trace and spikes, `to_dict`, `get_param_spec` with units, and a
  `register()` that adds it to `NEURON_REGISTRY`. See [Adding a Neuron](add_neuron.md).
- **Filter.** Started with `sensoryforge new-component filter exp_smoother`. Shows a filter
  that keeps state between calls, `reset_state`, `from_config`, and `FILTER_REGISTRY`. See
  [Adding a Filter](add_filter.md).
- **World shape.** Written by hand, since the scaffold has no world kind. A shape is a module
  level function of `(x, y, params)` in mm plus a list of parameter specs, and a `register()`
  calling `register_shape`. The engine multiplies by the contact's `amplitude`, so the
  function returns the profile only. See [Worlds](../worlds.md).

## Install and check one

```bash
cd examples/plugins/sensoryforge-ring-shape
pip install -e .
pip install pytest && pytest
sensoryforge list-components
```

Installing registers the entry point in the `sensoryforge.components` group, so every
SensoryForge process in that environment finds the component. See [Plugin Packages](plugins.md).

## Run all three in a data set

`ring_world.yml` draws ring-shaped contacts, `ring_dataset.yml` is a small data set on it,
and `sensor.yml` is an 8 by 8 sensor whose population uses the neuron and the filter. With the
three packages installed:

```bash
cd examples/plugins
sensoryforge batch sensor.yml --dataset ring_dataset.yml --output runs/plugins
```

This writes one bundle per entry. The same run is tested by
`tests/integration/test_reference_plugins.py`, which loads the plugins by their `register()`
functions instead of installing them.
