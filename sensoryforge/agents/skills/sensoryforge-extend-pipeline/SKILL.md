---
name: sensoryforge-extend-pipeline
description: Add a component SensoryForge does not have to the encoding pipeline, as an installable plugin that passes the contract check and is used from a config. Covers a neuron model and a temporal filter. Use when someone wants a new neuron, firing or spiking model, receptor dynamics, response filter, adaptation or temporal filter, or says SensoryForge cannot do what they need, in whatever words they put it.
---

# Extend the pipeline with a new component

A component is one part of the pipeline that a config names: a neuron model, a temporal
filter, and so on. You add one as its own small package, so nothing inside SensoryForge is
edited and the user's install stays as it is. You end with a package that passes the
contract check, shows up in `sensoryforge list-components`, and is used from a config.
This skill covers the neuron model and the temporal filter; ask what the user needs by
meaning, not by the word they chose ("firing model", "adaptation" and "receptor
dynamics" may each be either).

## 0. Read the working style

Run `sensoryforge agents brief` and follow it. If the command is not found, SensoryForge
is not installed here: say so and stop (it installs from source; it is not on PyPI yet).

## 1. Is it already there?

Run `sensoryforge list-components`, and `sensoryforge describe --json` for each name's
parameters. An existing component with other parameters may already do it: say so, and
stop here if the user is content. Two routes need no new package:

- **A neuron given by equations** needs no Python: the equation DSL compiles
  `dv/dt = ...` with a threshold and reset, or with none for an analog readout. The page
  is https://benefron.github.io/sensoryforge/user_guide/equation_dsl/ and a config uses
  it as `neuron_model: dsl` with a `dsl_config:` entry. Offer this first for a neuron.
- **A different value of an existing parameter** is a config change, not a component.

## 2. Get the model from the user

The dynamics, the parameters, their units and their ranges are the user's to give. Use
`sensoryforge-ground-assumptions` for anything missing and list every value with where it
came from (the user, a source they approved). Do not fill in an equation, a time constant
or a threshold from memory, and do not keep the scaffold's placeholder numbers as if they
were a model: they are there so the package runs. Write down before coding: what goes in
and what comes out (with units), each parameter with its unit and range, and one case the
user can check by hand (a step in, what comes out).

## 3. Scaffold the package

```
sensoryforge new-component neuron NAME --dest .
sensoryforge new-component filter NAME --dest .
```

This writes `sensoryforge-NAME/`: `pyproject.toml` with the entry point, a module with
the class and a `register()` function, a contract test in its own test folder and a README. It refuses
to write inside the installed package. Replace the placeholder body with the user's
model; keep the class's shape (below). Registered names are matched without regard to
case, and a different class under a taken name is refused; an identical repeat is
harmless, so `register()` may run twice.

## 4. A neuron model

Inherit `BaseNeuron`. The traps, each caught by the check or by the first real run:

- **The engine builds it as `Class(**model_params, dt=..., noise_std=...)`.** `__init__`
  must accept both: `dt=` is the integration step in ms, `noise_std=` is the membrane
  noise (use it, or accept it and say it does nothing). A neuron without them fails the
  first time a config uses it.
- **Shapes.** `forward()` takes the drive `[batch, steps, N]` in mA and returns
  `(v_trace, spikes)`, each `[batch, steps + 1, N]`, the first sample being the
  initial state. The second is `None` for an analog readout (no threshold). The check
  states the exact shapes; run it rather than trust this line.
- **Vectorise** over the last axis; loop only over time. Convert the step to seconds inside if
  the equations are per second; the API stays in ms.
- **`to_dict()` holds every `__init__` parameter, and `from_config()` of its result gives an
  equal `to_dict()`.** A parameter left out is silently lost when a run is saved and
  re-read. A constructor argument that is not a plain number (a tensor, an object) goes in
  the class attribute `_TO_DICT_EXCLUDE_PARAMS` and needs a default.
- **`get_param_spec()`** is a class method returning a list of `ParamSpec` with name,
  default, range and unit (the GUI and `describe` read it).
- **Noise** comes from `torch.randn_like`, which uses the global random number generator:
  the engine seeds and restores it per population. Never seed inside `forward()`.
- **State:** `reset_state()` returns the neuron to its start, and a second run after it
  gives the first run's result.

## 5. A temporal filter

Inherit `BaseFilter`. It turns the spatial drive into the current a neuron receives.

- **The engine builds it as `Class(**filter_params, dt=...)`**, with `dt=` the record step
  in ms. A config's `filter_params:` entry is passed through as written, so put the defaults in
  `__init__`; nothing else supplies them (the defaults of the built-in `sa` and `ra`
  filters are resolved from one table the engine reads, which a plugin does not join).
- **Shapes and units.** `forward()` takes `[batch, time, channels]` in mA and returns the
  same shape in mA.
- **`reset_state()`** (singular) clears whatever the filter carries between calls. Decide
  whether a call continues the last one or starts again, and say which in the docstring.
- **`to_dict()`, `from_config()` and `get_param_spec()`** follow the same rules as for a
  neuron. Time constants are in ms.
- **Names.** Do not reuse `sa`, `ra`, `none` or `identity`.

## 6. Check it

The package's own test calls `check_component("neuron", Class)` (or `"filter"`) from
`sensoryforge.testing.contracts`, which checks the parameter spec, one forward pass with
the canonical shapes, and the `to_dict()` round trip. Run the test with pytest if the user
has it (it is not installed with SensoryForge), or call `check_component()` from a few
lines of Python, as the example below does. Then add a test of the user's own case from step 2: the
check says the component is well formed, not that it is right. If the check fails, the
message names the class and the rule; fix the class, not the check.

## 7. Use it

Install the folder into the same environment (`python -m pip install ./sensoryforge-NAME`;
pip's editable mode while it is still changing), and `sensoryforge list-components` shows
the name. Or, with no install, list the module in the config so it loads first:

```yaml
plugins:
  - sensoryforge_NAME.component:register
populations:
  - name: demo
    neuron_type: SA
    neuron_model: <neuron name>
    filter_method: <filter name>
    model_params: {}      # the neuron's own parameters
    filter_params: {}     # the filter's own parameters
```

That is only the part of a population that names the components: write it into the user's
real config (`sensoryforge describe --schema config` lists the rest) and run
`sensoryforge validate --json CONFIG`.
A population has fields the user may not have given (its name, its neuron type, the
initial value of a state variable): leave them out and ask, or write them and say plainly
that they are placeholders for the user to settle. Report the package folder, the names it registers,
the checks that passed, and the values that came from the user.

## Verify

The scaffold's placeholder dynamics are gone or the user said to keep them, the contract
check passes for each class, the component's name appears in `sensoryforge
list-components` (after install, or after the `plugins:` entry), the user's own case from
step 2 is tested, and no number in the package came from you.

## Worked example

A neuron and a filter scaffolded side by side, checked, loaded twice (as an entry point
and a config's `plugins:` entry may both do), and driven with a step:

```python
# file: use_it.py
"""Check two scaffolded plugins and drive them, as a plain install can."""
import sys
from pathlib import Path

import torch

# a plugin folder is on the path here; an installed one is already
sys.path[:0] = [str(p.parent) for p in Path(".").glob("*/pyproject.toml")]

from sensoryforge.plugins import load_plugin_import_paths
from sensoryforge.registry import FILTER_REGISTRY, NEURON_REGISTRY
from sensoryforge.testing.contracts import check_component

paths = [
    "sensoryforge_demo_lif.component:register",
    "sensoryforge_demo_band.component:register",
]
print("loaded", len(load_plugin_import_paths(paths)))
print("loaded again", len(load_plugin_import_paths(paths)))

check_component("neuron", NEURON_REGISTRY.get_class("demo_lif"))
check_component("filter", FILTER_REGISTRY.get_class("demo_band"))
print("contract ok")

drive = torch.zeros(1, 40, 2)
drive[:, 10:30] = 60.0  # a step of drive, mA
current = FILTER_REGISTRY.create("demo_band")(drive)
v_trace, spikes = NEURON_REGISTRY.create("demo_lif")(current)
print("filtered", tuple(current.shape), "neuron", tuple(spikes.shape))
```

```bash
sensoryforge new-component neuron demo_lif --dest .
# prints: Scaffolded new neuron plugin package
sensoryforge new-component filter demo_band --dest .
# prints: Scaffolded new filter plugin package
python use_it.py
# prints: loaded again 2
# prints: contract ok
# prints: filtered (1, 40, 2) neuron (1, 41, 2)
```
