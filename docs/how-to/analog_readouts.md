# Analog Readouts

## Overview

A population's neuron model normally produces spikes: `[batch, time, num_neurons]` boolean
events, reduced (Phase 2, F-008) to a sub-step spike **count** per record bin. Some models have no
spike condition at all — a leaky integrator meant to be read out as a continuous membrane
trajectory rather than a discrete event train. SensoryForge supports this as an **analog readout**:
the same pipeline (innervation → filter → gain → noise → neuron), but the last stage returns a
state trace instead of spikes.

```
Stimulus  [batch, time, H, W]
    ↓  ReceptiveFieldBank
    ↓  Filter (SA/RA, optional)
    ↓  DSL neuron model
Spikes [batch, time, N] (spiking)   or   State [batch, time, N] (analog)
```

This is the "or not spiking" half of the Equation DSL's purpose (see
[Equation DSL](equation_dsl.md)): a model is analog simply by omitting `threshold` (and `reset`,
which then has no spike event to trigger on and must also be omitted).

## Defining an analog model

```python
from sensoryforge.neurons.model_dsl import NeuronModel

model = NeuronModel(
    equations="dv/dt = (-(v - v_rest) + R*I) / tau_m",
    parameters={"v_rest": -65.0, "R": 1.0, "tau_m": 10.0},
    state_vars={"v": -65.0},
    # No threshold, no reset.
)

neuron = model.compile(dt=0.1, device="cpu")

import torch
current = torch.randn(2, 100, 5)  # [batch, steps, features] in mA
state_trace, spikes = neuron(current)
assert spikes is None
state_trace.shape  # (2, 101, 5) -- the state trace, every Euler step
```

With a `threshold`, the same class behaves exactly as before and `spikes` is a boolean tensor.

## Through `SimulationEngine`

The runnable example is `docs/examples/analog_dsl.py`: it builds one analog population and one
spiking population (same DSL neuron model, one without a `threshold` and one with), runs both
through `SimulationEngine`, and asserts the analog population's result carries `"state"` (not
`"spikes"`) while the spiking one carries `"spikes"` (not `"state"`). It is executed by
`tests/docs/test_docs_examples.py`.

A canonical population config with `neuron_model: dsl` and a thresholdless `dsl_config` runs like
any other population, but its result dict carries `"state"` instead of `"spikes"`:

```yaml
populations:
  - name: Analog Population
    neuron_model: dsl
    dsl_config:
      equations: "dv/dt = (-(v - v_rest) + R*I) / tau_m"
      parameters: {v_rest: -65.0, R: 1.0, tau_m: 10.0}
      state_vars: {v: -65.0}
    # readout: auto   # default -- infers analog from the absence of a threshold
```

```python
results = engine.run(stimulus)
state = results["Analog Population"]["state"]   # [batch, time, num_neurons]
assert "spikes" not in results["Analog Population"]
```

`state` holds the bin-end sample of each record step — the same reduction `"voltages"` already
uses for a spiking population, so an analog and a spiking population's outputs share one time axis
and shape convention: `[batch, time, num_neurons]`.

### `PopulationConfig.readout`

| Value | Effect |
|---|---|
| `"auto"` (default) | The readout follows `dsl_config`: analog when it has no `threshold`, spiking otherwise. |
| `"analog"` | Forces analog. Raises `ValueError` if `dsl_config` still defines a `threshold` (remove it, or use `"auto"`/`"spiking"`). |
| `"spiking"` | Forces spiking. Raises `ValueError` if `dsl_config` has no `threshold` (add one, or use `"auto"`/`"analog"`). |

`readout` is ignored for non-DSL neuron models (Izhikevich, AdEx, MQIF, ...), which always spike.

A population that declares `neuron_model: dsl` but omits `dsl_config` entirely raises a
`ValueError` naming the population, rather than failing later with a confusing `TypeError`.

## In the GUI

The Spiking tab's raster panel plots a spike scatter for a spiking population, and — for an analog
population — the state trace of every neuron instead, with the axis labelled with the state
variable's name (the first entry of the DSL model's `state_vars`, e.g. `v`). No raster points are
drawn for an analog population.

## See Also

- [Equation DSL](equation_dsl.md) — defining `equations`/`threshold`/`reset`/`parameters`/`state_vars`
- [Units and Gains](units_and_gains.md) — `input_gain`, filter units
