# Your First Simulation

For anyone with SensoryForge installed ([Installation](installation.md)) who wants to run a
simulation from Python. By the end you will have run a shipped recipe with a stimulus of
your own, read its spikes, drawn a raster and counted spikes per neuron. The complete
script is at the end, is the file `docs/examples/first_simulation.py`, and is run by the
test suite, so it works as shown.

You need nothing but the install: no example file and no clone. The same run from the
terminal is [the command-line quick start](quickstart_cli.md).

## The idea

A run has four parts, and the script has one step for each:

1. a **config**: the sensor array, the populations that read it, the neuron models. A shipped
   recipe gives you a complete one;
2. a **stimulus**, rendered into frames;
3. the **engine**, which samples the frames with each population's receptive fields, filters
   them in time and runs the neurons;
4. the **results**: spikes per population.

Shapes are batch first: a stimulus is `[batch, time, height, width]` and a population's
spikes are `[batch, time, neurons]`. Units are ms, mm and mA.

## Step 1: A recipe, with your stimulus

```python
from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.presets import load_preset

config_dict = load_preset("tactile_sa1_ra1")
config_dict["stimulus"] = {"type": "gaussian", "amplitude": 1.0, "sigma": 0.5}
config = SensoryForgeConfig.from_dict(config_dict)
```

`tactile_sa1_ra1` is an 80 × 80 sensor array at 0.15 mm read by an SA and an RA population
(`sensoryforge list-presets` shows the others). The stimulus block is a Gaussian blob of
peak 1 and width 0.5 mm at the centre. Every stimulus type and its parameters:
`sensoryforge describe --section components`, or the
[component reference](../reference/components.md).

## Step 2: Render the stimulus

```python
from sensoryforge.stimuli.render import render_for_config

stimulus, _time_ms, _canvas, _dropped = render_for_config(
    config, duration_ms=300.0, dt_ms=config.simulation.dt_ms
)
print(tuple(stimulus.shape))      # (1, 300, 80, 80): [batch, time, height, width]
```

## Step 3: Run

```python
from sensoryforge.core.simulation_engine import SimulationEngine

engine = SimulationEngine(config)
results = engine.run(stimulus)
```

`results` maps each population's name to a dict of tensors. `results["SA Population"]["spikes"]`
is `[batch, time, neurons]`, the spike count in each time bin. Pass
`return_intermediates=True` to also get the `drive` and `filtered` currents (mA) and the
membrane `voltages` (mV), and `bundle_dir="my_run"` to write a
[data bundle](../reference/bundles.md).

## Step 4: Read the spikes

```python
for name, population in results.items:
    spikes = population["spikes"][0]          # batch 0: [time, neurons]
    print(name, int(spikes.sum), "spikes from", spikes.shape[1], "neurons")
```

A raster is the positions of the nonzero entries:

```python
times, neurons = (spikes > 0).nonzero(as_tuple=True)
time_ms = times * config.simulation.dt_ms
```

## Step 5: Change the stimulus

Edit the `stimulus` block and run again. A wider, stronger blob off the centre:

```python
config_dict["stimulus"] = {
    "type": "gaussian", "amplitude": 2.0, "sigma": 1.0,
    "center_x": 2.0, "center_y": -1.0,   # mm
}
```

Rebuild `config`, the stimulus and the engine as above (a config is read when the engine is
built).

## The complete script

```python
--8<-- "docs/examples/first_simulation.py"
```

```bash
python first_simulation.py
```

It prints the stimulus shape and one line per population, and saves `first_simulation.png`.

## Next

- [Batch processing](batch_processing_tutorial.md): many runs, a sweep over a parameter.
- [Custom neuron models](custom_neurons.md): write your own with the equation DSL.
- [Core concepts](../concepts/core_concepts.md): how the stages connect.
- [How-to guides](../how-to/yaml_configuration.md): configs, presets, receptive fields and more.
