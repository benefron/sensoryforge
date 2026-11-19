# Examples

This folder is in the repository, not in the installed package. What you can run from a
plain install ships in the package instead:

```bash
sensoryforge list-presets                 # whole sensors to start from
sensoryforge examples list                # configs a preset cannot give
sensoryforge examples copy amplitude_sweep
```

See the [quick starts](https://benefron.github.io/sensoryforge/) for where to begin.

## What is here

| Path | What it is | Checked by |
|---|---|---|
| `notebooks/01_quickstart.ipynb` | A notebook: the encoding workflow from grid to spikes | CI runs it (nbmake) |
| `scripts/example_pipeline.py` | A preset, one stimulus, two spike rasters | `tests/integration/test_example_scripts.py` |
| `scripts/generalized_pipeline_demo.py` | The YAML-driven pipeline with several stimulus types | `tests/integration/test_example_scripts.py` |
| `pressure_simulation_recipe.py` | The pressure-simulation recipe: four stimuli, one bundle each | `tests/integration/test_pressure_simulation_recipe.py` |
| `vision_rgb_onoff.py` | A three-channel (R, G, B) sensor with ON/OFF processing | `tests/integration/test_vision_rgb_onoff.py` |
| `plugins/` | Three reference plugins: a neuron, a filter, a world shape ([guide](https://benefron.github.io/sensoryforge/developer_guide/reference_plugins/)) | `tests/integration/test_reference_plugins.py` |
| `plugin_template/` | A plugin package with a processing layer and a receptive-field builder | its own `tests/` |
| `canonical_config.yml`, `canonical_batch_config.yml`, `canonical_template_config.yml` | Full configs written out field by field | `tests/integration/test_examples_smoke.py` |
| `example_config.yml`, `batch_config.yml` | The legacy config format, kept only so its reader stays tested | `tests/integration/test_examples_smoke.py` |

Run a script from a clone with the package installed (`pip install -e ".[dev]"`):

```bash
python examples/scripts/example_pipeline.py --save rasters.png
python examples/pressure_simulation_recipe.py --quick
```

## Adding an example

An example nobody runs stops working. Add a test that runs it (or add it to one above)
in the same change. A config a user should be able to get without a clone belongs in
`sensoryforge/examples/` instead, where `sensoryforge examples copy` finds it.
