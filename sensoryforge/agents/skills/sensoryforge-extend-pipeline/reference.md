# Extend the pipeline: a solver, a processing layer, an event converter

This is the reference file of `sensoryforge-extend-pipeline`. Its steps 0-3 and 10-11 (the
brief, is it already there, get the model from the user, scaffold, check, use) apply to
everything here. Read the section for the kind the user needs.

## A solver

A solver is the numerical step an ODE-driven neuron takes. Inherit `BaseSolver`; the
scaffold (`sensoryforge new-component solver NAME --dest .`) writes a placeholder forward
Euler step, which is not the user's method: replace it.

- **`step(ode_func, state, t, dt)`** returns the next state, the same shape as the
  state (`[batch, ...]`). `ode_func(state, t)` returns the derivative, also that shape.
  Time and `dt=` are in ms. The neuron calls `step()` once per sub-step and feeds the result straight
  back, so never change the shape or the dtype.
- **`integrate(ode_func, state, t_span, dt)`** returns `[batch, steps + 1, ...]`, the first
  sample being the start state.
- **`to_dict()` holds a `type:` key, and for a scaffolded class it is the class name in
  lowercase with "solver" removed** (`demork2`), not the name the class is registered
  under (`demo_rk2`). Override `to_dict()` so that key is the registered name, and add every
  `__init__` parameter beyond `dt=`.
- **Randomness and state:** none between calls; a solver is a pure step.
- **Check** with `check_component("solver", Class)`. It tests the shape of one step and the
  round trip, not the order of accuracy: test that on a case with a known answer
  (`dx/dt = -x` from 1 has `exp(-t)`), and compare the error against forward Euler's at the
  same step. A higher order must show a smaller error, or it is not what the user asked
  for.
- **Reaching a run.** Today nothing in a run builds a solver from a config: a population's
  `solver_config:` is read by no stage, `get_solver()` knows only `euler` and `adaptive`,
  and the equation DSL's `compile()` accepts only `euler` and ignores a solver object. A
  plugin solver is therefore registered and checked, and used by Python code that calls it
  (`SOLVER_REGISTRY.create(name, dt=...)`), for instance inside a neuron model of the
  user's own. Say so to the user before they expect a config key to switch to it.

## A processing layer

A processing layer sits between receptor sampling and the receptive-field bank, per
population input. It takes the receptor responses `[..., M]` and may return more columns
(`[..., M2]` with `M2 >= M`). There is no scaffold: write one module with the class and a
`register()` function, and list it in the config's `plugins:` entry as
`module:register`.

- **Inherit `BaseProcessingLayer`.** `forward(receptor_responses, metadata=None)` is the
  signature; the last axis is the receptors, every leading axis passes through.
- **`expand_receptor_coords(receptor_coords)`** is a class method that must mirror
  `forward()` column for column. The bank is built on its result, so column k of the
  weights meets column k of `forward()`'s output. A layer that doubles the receptor axis
  and expands the coordinates in a different order gives a plausible, wrong result that no
  shape test catches. Leave it out only when the layer keeps `M`.
- **`REQUIRES_RECEPTOR_COORDS = True`** when the layer's own construction needs the
  positions (for distances); the pipeline then passes `receptor_coords=` to
  `from_config()`. Otherwise do not set it.
- **`to_dict()`** returns `{"method": <registered name>, "params": {...}}` with every
  `__init__` parameter; `from_config()` accepts that nested form and a flat one. Receptor
  coordinates stay out (put the name in `_TO_DICT_EXCLUDE_PARAMS`).
- **Use** it as an entry under an input's `processing:`, `- method: <layer name>` with
  `params:`. A population input with an empty list builds no pipeline at all.
- **Check** with `check_component("processing", Class)` (give it `instance=` when
  `__init__` needs the coordinates). Then write the test the check cannot: drive the
  layer with an input of one sign and assert which columns respond (for a positive and a
  negative plane, a positive input must light the positive plane only, and the negative
  input the other). Swapped planes pass every shape test.

## An event converter

A converter turns a neuron's drive into events. It is a neuron model: inherit `BaseNeuron`,
scaffold with `sensoryforge new-component neuron NAME --dest .` or write the module by
hand, and follow step 4 of the skill for the constructor, shapes, `to_dict()` and
`get_param_spec()`. What is specific to a converter:

- **Signed events.** Set the class attribute `SIGNED_EVENTS = True` when the counts carry a
  sign (+k ON, -k OFF). The run then returns them under the events key and a bundle stores them
  as signed integers, so nothing that counts spikes reads an OFF event as one. Without it
  the counts go under the spikes key.
- **How the engine runs it.** It builds the class with `dt=` the integration step in ms
  (0.05 by default) and `noise_std=` the population's `membrane_noise_std:`; `model_params:`
  holds the rest. It holds each record bin's drive over the sub-steps, calls `forward()`
  once on `[batch, steps, N]`, drops the sample at index 0, and sums each bin's sub-step
  counts. So `forward()` returns `(state_trace, events)`, both `[batch, steps + 1, N]`,
  the first sample being the start state.
- **The input floor.** A population of type SA, RA or SA2 has its neuron input clamped
  at 0 mA before the converter, whatever the model (only `level_crossing` and the DSL are
  exempt). A converter that reads negative drive then never sees it. Set the population's
  `input_floor:` to `-.inf` (or a value the user gives), and say which, because the bundle
  records the floor that was applied.
- **Names.** Use a new name. `level_crossing` and `sigma_delta` are pinned bit for bit by
  another project that emulates them, so a converter plugin never replaces or changes
  them, and a design that names a converter must be one SensoryForge can build at load.
- **Use** it as `neuron_model: <name>` with `model_params:` holding the constructor
  arguments except `dt=` and `noise_std=`. `filter_method: none` hands the drive straight
  to the converter.
- **Check** with `check_component("neuron", Class)`, then test the user's own case: the
  sign and the count for a drive they can compute by hand.

## Worked example

A processing layer and an event converter written by hand, a solver scaffolded, all three
checked and loaded, and the layer and the converter used in two runs of the engine:

```python
# file: demo_layer.py
"""A processing layer: each receptor's positive part, then its negative part."""
from typing import Any, Dict, List, Optional

import torch

from sensoryforge.core.processing import BaseProcessingLayer
from sensoryforge.stimuli.base import ParamSpec


class DemoSplitLayer(BaseProcessingLayer):
    def __init__(self, gain: float = 1.0) -> None:
        super().__init__()
        self.gain = float(gain)

    def forward(
        self, receptor_responses: torch.Tensor, metadata: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """[..., M] to [..., 2M]: columns 0..M-1 positive, M..2M-1 negative."""
        pos = torch.clamp(receptor_responses, min=0.0)
        neg = torch.clamp(-receptor_responses, min=0.0)
        return self.gain * torch.cat([pos, neg], dim=-1)

    @classmethod
    def expand_receptor_coords(cls, receptor_coords: torch.Tensor) -> torch.Tensor:
        """Mirror forward(): the same receptors twice, in the same order."""
        return torch.cat([receptor_coords, receptor_coords], dim=0)

    def to_dict(self) -> Dict[str, Any]:
        return {"method": "demo_split", "params": {"gain": self.gain}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DemoSplitLayer":
        params = dict(config.get("params") or {})
        return cls(gain=config.get("gain", params.get("gain", 1.0)))

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [ParamSpec("gain", dtype="float", default=1.0, min_val=0.0)]


def register() -> None:
    from sensoryforge.registry import PROCESSING_REGISTRY

    PROCESSING_REGISTRY.register("demo_split", DemoSplitLayer)
```

```python
# file: demo_events.py
"""An event converter: +1 above theta, -1 below minus theta, at every sub-step."""
from typing import List, Tuple

import torch

from sensoryforge.neurons.base import BaseNeuron
from sensoryforge.stimuli.base import ParamSpec


class DemoEventUnit(BaseNeuron):
    SIGNED_EVENTS = True  # the run returns the counts under "events"

    def __init__(self, theta: float = 0.5, dt: float = 0.05, noise_std: float = 0.0) -> None:
        super().__init__(dt=dt)
        self.theta = float(theta)
        self.noise_std = float(noise_std)

    def reset_state(self) -> None:
        """Nothing is carried between calls."""

    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = input_current  # [batch, steps, N], mA
        if self.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.noise_std
        events = (x > self.theta).to(x.dtype) - (x < -self.theta).to(x.dtype)
        start = torch.zeros_like(x[:, :1])  # index 0 is the start state
        return torch.cat([start, x], dim=1), torch.cat([start, events], dim=1)

    def to_dict(self) -> dict:
        return {"theta": self.theta, "dt": self.dt, "noise_std": self.noise_std}

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec("theta", dtype="float", default=0.5, min_val=0.0, unit="mA"),
            ParamSpec("noise_std", dtype="float", default=0.0, min_val=0.0, unit="mA"),
        ]


def register() -> None:
    from sensoryforge.registry import NEURON_REGISTRY

    NEURON_REGISTRY.register("demo_event", DemoEventUnit)
```

```python
# file: use_pipeline.py
"""Check a layer, a converter and a solver, then run the layer and the converter."""
import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(".").resolve())] + [
    str(p.parent) for p in Path(".").glob("*/pyproject.toml")
]

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    PopulationInput,
    RFBuilderConfig,
    SensoryForgeConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.plugins import load_plugin_import_paths
from sensoryforge.registry import NEURON_REGISTRY, PROCESSING_REGISTRY, SOLVER_REGISTRY
from sensoryforge.testing.contracts import check_component

paths = [
    "demo_layer:register",
    "demo_events:register",
    "sensoryforge_demo_rk2.component:register",
]
print("loaded", len(load_plugin_import_paths(paths)))
layer = PROCESSING_REGISTRY.get_class("demo_split")
check_component("processing", layer)
check_component("neuron", NEURON_REGISTRY.get_class("demo_event"))
check_component("solver", SOLVER_REGISTRY.get_class("demo_rk2"))
print("contract ok")

# what the check cannot see: a positive input lights the positive plane only
out = layer(gain=2.0)(torch.tensor([[[0.0, 3.0, 0.0]]]))
print("planes", out[..., :3].sum().item(), out[..., 3:].sum().item())

solver = SOLVER_REGISTRY.create("demo_rk2", dt=0.1)
path = solver.integrate(lambda state, t: -state, torch.ones(1, 1), (0.0, 1.0), 0.1)
print("solver path", tuple(path.shape))


def population(processing):
    return PopulationConfig(
        name="p",
        neuron_type="RA",
        neuron_model="demo_event",
        model_params={"theta": 0.5},
        filter_method="none",
        neurons_per_row=3,
        input_gain=1.0,
        input_floor=float("-inf"),  # RA would clamp the drive at 0 mA otherwise
        inputs=[
            PopulationInput(
                grid="G", rf=RFBuilderConfig(method="gaussian"), processing=processing
            )
        ],
    )


for label, processing in (("plain", []), ("split", [{"method": "demo_split"}])):
    config = SensoryForgeConfig(
        grids=[GridConfig(name="G", arrangement="grid", rows=6, cols=6, spacing=0.2)],
        populations=[population(processing)],
    )
    engine = SimulationEngine(config)
    bank = engine.populations[0]["inputs"][0]["bank"]
    events = engine.run(-torch.ones(1, 4, 6, 6))["p"]["events"]  # a negative field
    print(label, "receptors", bank.num_receptors, "net events", int(events.sum()))
```

```bash
sensoryforge new-component solver demo_rk2 --dest .
# prints: Scaffolded new solver plugin package
python use_pipeline.py
# prints: loaded 3
# prints: contract ok
# prints: planes 6.0 0.0
# prints: solver path (1, 11, 1)
# prints: plain receptors 36 net events -720
# prints: split receptors 72 net events 720
```
