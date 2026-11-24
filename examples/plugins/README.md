# Reference plugins

Three small plugins to copy from. Each is a folder you can install on its own, passes
SensoryForge's contract check, and has a test with a case worked by hand.

| Folder | Adds | Name | Check |
|---|---|---|---|
| `sensoryforge-leaky-threshold/` | a neuron | `leaky_threshold` | `check_component("neuron", ...)` |
| `sensoryforge-exp-smoother/` | a temporal filter | `exp_smoother` | `check_component("filter", ...)` |
| `sensoryforge-ring-shape/` | a world shape | `ring` | `check_world_component("shape", "ring")` |

The formulas are textbook shapes chosen to show the contracts. They are not models of any
real cell or stimulus, and their parameter ranges are placeholders.

## Run all three in a data set

`ring_world.yml` draws ring-shaped contacts, `ring_dataset.yml` is a small data set on it,
and `sensor.yml` is an 8 by 8 sensor whose population uses the neuron and the filter.
After `pip install -e .` in each folder:

```bash
sensoryforge batch sensor.yml --dataset ring_dataset.yml --output runs/plugins
```

That writes one bundle per entry to `runs/plugins/`. The same run is tested in
`tests/integration/test_reference_plugins.py`, which loads the plugins by their `register()`
functions instead. How to write your own: `docs/how-to/extend/plugins.md` and
`docs/how-to/extend/reference_plugins.md`.
