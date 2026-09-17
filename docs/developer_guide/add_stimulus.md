# Adding a New Stimulus Type

This guide walks through every step required to add a new stimulus to SensoryForge.
After following it, your stimulus will be available in the CLI, the GUI Stimulus
Designer tab, and any code that uses the `STIMULUS_REGISTRY`.

There are two ways to ship a new stimulus — pick one before you start:

- **Plugin package** (recommended for most cases, including anyone outside
  the core project): `sensoryforge new-component stimulus MyStimulus --dest DIR`
  scaffolds an installable package that registers itself via an entry
  point — no edits to this checkout. See `plugins.md` for the full guide;
  this page's steps 2 and 4 (the class body and its test) still apply
  verbatim inside the scaffolded package.
- **In-repo** (for contributing to SensoryForge itself):
  `sensoryforge new-component stimulus MyStimulus --in-repo` writes
  directly into `sensoryforge/stimuli/`, `tests/`, `docs/`, and step 3
  below (manual registration in `register_components.py`) applies.

Component names are matched **case-insensitively** (H1, F-046): `"Ring"` and
`"ring"` are the same registration, so you don't need to worry about
exact-case collisions with a built-in stimulus's name — though a genuine
collision (same name, different class) still raises.

The complete runnable example is `docs/examples/plugin_stimulus.py` (a smaller ring stimulus,
`AnnulusStimulus`, than this page's `RingStimulus` walkthrough, but the same steps and shape): it
defines the class, registers it, runs the shared contract check, and generates one frame directly.
It is executed by `tests/docs/test_docs_examples.py`.

---

## 1. Understand the API contract

Every stimulus must satisfy the `BaseStimulus` interface
(`sensoryforge/stimuli/base.py`):

| Method | Required | Purpose |
|--------|----------|---------|
| `forward(xx, yy, **kwargs) → Tensor` | ✅ | Generate one spatial frame |
| `reset_state()` | ✅ | Clear internal state (noop if stateless) |
| `from_config(config) → cls` | ✅ | Construct from a YAML dict |
| `to_dict() → dict` | ✅ | Serialise parameters for round-trip YAML |
| `get_param_spec() → list[ParamSpec]` | ✅ (required on every component, G1) | GUI auto-discovery |

`to_dict()` should include every `__init__` parameter so
`from_config(instance.to_dict())` round-trips completely.
`sensoryforge.testing.contracts.check_component("stimulus", cls)` checks
the basic `from_config`/`to_dict` round-trip, `get_param_spec()`
presence, and a forward-pass shape check — but **not** the full
parameter-completeness check: the H3 completeness requirement (every
`__init__` argument present in `to_dict()`) is currently enforced for
neurons only (`_check_neuron` in `sensoryforge/testing/contracts.py`),
not `_check_stimulus`. `EdgeGrating` would fail it today (missing
`normalize`). Not yet enforced for other kinds (see ledger F-049).

Tensor shapes and units:

- `xx`, `yy`: `[H, W]` spatial coordinate grids in **mm**
- `forward()` return: **either** a single frame `[H, W]` (a static spatial
  pattern; `render_stimulus`, see step 5, expands it over time with a
  ramp/plateau/ramp envelope) **or** a whole sequence `[T, H, W]` (a
  stimulus with its own internal motion or dynamics, like `MovingStimulus`
  or the ported pressure-simulation stimuli in `sensoryforge/stimuli/
  tactile.py`) — both are valid (Phase 2, Wave K, Fact K-c). Units are
  **mA** (or dimensionless if used as a mask).

---

## 2. Create the module file

Add `sensoryforge/stimuli/my_stimulus.py`:

```python
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

from sensoryforge.stimuli.base import BaseStimulus, ParamSpec


class RingStimulus(BaseStimulus):
    """Annular pressure ring stimulus.

    Args:
        center_x: Ring centre x-coordinate in mm.
        center_y: Ring centre y-coordinate in mm.
        radius: Ring radius in mm.
        width: Ring wall half-width in mm.
        amplitude: Peak pressure amplitude in mA.
    """

    def __init__(
        self,
        center_x: float = 0.0,
        center_y: float = 0.0,
        radius: float = 1.0,
        width: float = 0.1,
        amplitude: float = 1.0,
    ) -> None:
        super().__init__()
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.width = width
        self.amplitude = amplitude

    def forward(self, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        dist = torch.sqrt((xx - self.center_x) ** 2 + (yy - self.center_y) ** 2)
        ring = torch.exp(-((dist - self.radius) ** 2) / (2 * self.width ** 2))
        return ring * self.amplitude

    def reset_state(self) -> None:
        pass  # Stateless — nothing to reset

    @classmethod
    def from_config(cls, config: dict) -> "RingStimulus":
        return cls(**config)

    def to_dict(self) -> dict:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "radius": self.radius,
            "width": self.width,
            "amplitude": self.amplitude,
        }

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec("center_x", label="Centre X", dtype="float",
                      default=0.0, min_val=-20.0, max_val=20.0, step=0.1, unit="mm"),
            ParamSpec("center_y", label="Centre Y", dtype="float",
                      default=0.0, min_val=-20.0, max_val=20.0, step=0.1, unit="mm"),
            ParamSpec("radius", label="Radius", dtype="float",
                      default=1.0, min_val=0.01, max_val=20.0, step=0.1, unit="mm"),
            ParamSpec("width", label="Width", dtype="float",
                      default=0.1, min_val=0.001, max_val=5.0, step=0.01, unit="mm"),
            ParamSpec("amplitude", label="Amplitude", dtype="float",
                      default=1.0, min_val=0.0, max_val=500.0, step=1.0, unit="mA"),
        ]
```

---

## 3. Register the stimulus (in-repo route only)

If you scaffolded with `--in-repo` (or are hand-writing a contribution to
this checkout), open `sensoryforge/register_components.py` and add two
lines inside `register_all()`. If you are shipping a plugin package
instead, skip this step — the scaffold's `register()` function plus your
package's entry point handles it; see `plugins.md`.

```python
from sensoryforge.stimuli.my_stimulus import RingStimulus  # add this import

def register_all():
    ...
    STIMULUS_REGISTRY.register("ring", RingStimulus)         # add this line
    ...
```

---

## 4. Write a unit test

Add `tests/unit/test_ring_stimulus.py`:

```python
import torch
import pytest
from sensoryforge.stimuli.my_stimulus import RingStimulus


@pytest.fixture
def grid():
    x = torch.linspace(-3, 3, 64)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    return xx, yy


def test_output_shape(grid):
    stim = RingStimulus()
    out = stim(*grid)
    assert out.shape == grid[0].shape


def test_peak_near_radius(grid):
    """Peak amplitude should occur at the ring radius."""
    stim = RingStimulus(radius=1.5, width=0.1, amplitude=5.0)
    out = stim(*grid)
    xx, yy = grid
    dist = torch.sqrt(xx ** 2 + yy ** 2)
    on_ring = (dist - 1.5).abs() < 0.15
    assert out[on_ring].mean() > out[~on_ring].mean()


def test_zero_outside_ring(grid):
    """Values far from the ring should be near zero."""
    stim = RingStimulus(radius=1.0, width=0.05, amplitude=1.0)
    out = stim(*grid)
    xx, yy = grid
    far = (xx ** 2 + yy ** 2).sqrt() > 2.5
    assert out[far].max() < 0.01


def test_from_config_roundtrip():
    stim = RingStimulus(center_x=0.5, radius=2.0)
    stim2 = RingStimulus.from_config(stim.to_dict())
    assert stim2.radius == stim.radius


def test_get_param_spec():
    specs = RingStimulus.get_param_spec()
    names = [s.name for s in specs]
    assert "radius" in names
    assert "amplitude" in names
```

Run: `pytest tests/unit/test_ring_stimulus.py -v`

---

## 5. Use from the CLI

Before Phase 2 Wave K (F-052), the CLI's canonical `run` path dispatched
stimulus types through a hard-coded chain in
`GeneralizedTactileEncodingPipeline.generate_stimulus` that only knew nine
names — a registered stimulus like `ring` was reachable through
`STIMULUS_REGISTRY.create()` in code, but **not** from a config file. Since
K1, `sensoryforge/stimuli/render.py`'s `render_stimulus()` is the CLI's and
`BatchExecutor`'s single dispatch point: it checks `STIMULUS_REGISTRY`
first, so *any* registered stimulus — built-in or a plugin's — is runnable
from a config file with no changes to the CLI. This is the concrete,
executed proof of that: a stimulus registered only inside
`tests/unit/test_render_stimulus.py::test_plugin_stimulus_registered_only_at_test_time_runs`
(a two-line `BaseStimulus` subclass) runs end to end through
`render_stimulus`, the same call the CLI makes.

After registering, the stimulus works with any canonical config:

```yaml
# my_config.yml
grids:
  - name: main_grid
    rows: 40
    cols: 40
    spacing: 0.15

populations:
  - name: SA Pop
    target_grid: main_grid
    neuron_type: SA
    neuron_model: Izhikevich
    neurons_per_row: 4
    innervation_method: gaussian
    connections_per_neuron: 4
    sigma_d_mm: 0.5
    filter_method: SA

simulation:
  dt_ms: 1.0
  device: cpu

stimuli:
  - type: ring          # ← your new stimulus key
    center_x: 0.0
    center_y: 0.0
    radius: 1.5
    width: 0.1
    amplitude: 8.0
```

```bash
sensoryforge run my_config.yml --duration 500
```

---

## 6. Verify GUI auto-discovery

If `get_param_spec()` is implemented, the Stimulus Designer tab will automatically
show spinboxes for all your parameters when the user selects your stimulus type.
No GUI code changes are needed.

Verify with:

```python
from sensoryforge.register_components import register_all
register_all()
from sensoryforge.registry import STIMULUS_REGISTRY
specs = STIMULUS_REGISTRY.get_param_spec("ring")
print([s.name for s in specs])
# ['center_x', 'center_y', 'radius', 'width', 'amplitude']
```

---

## Checklist

- [ ] Class inherits from `BaseStimulus`
- [ ] `forward()` returns `[H, W]` tensor in mA, on same device as inputs
- [ ] `reset_state()` implemented (noop if stateless)
- [ ] `to_dict()` includes every `__init__` parameter and is a round-trip fixed point with `from_config()` (H3)
- [ ] `get_param_spec()` provides UI-ready descriptors with units and ranges (required on every component, G1)
- [ ] Registered — either via a plugin package's entry point, or in `register_all()` with a lowercase snake_case key (in-repo route)
- [ ] Unit tests cover shape, physics, roundtrip, and `check_component("stimulus", cls)`
- [ ] Docstring includes tensor shapes and physical units
