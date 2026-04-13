# Adding a New Stimulus Type

This guide walks through every step required to add a new stimulus to SensoryForge.
After following it, your stimulus will be available in the CLI, the GUI Stimulus
Designer tab, and any code that uses the `STIMULUS_REGISTRY`.

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
| `get_param_spec() → list[ParamSpec]` | Recommended | GUI auto-discovery |

Tensor shapes and units:

- `xx`, `yy`: `[H, W]` spatial coordinate grids in **mm**
- `forward()` return: `[H, W]` pressure field in **mA** (or dimensionless if
  used as a mask)

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

## 3. Register the stimulus

Open `sensoryforge/register_components.py` and add two lines inside
`register_all()`:

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
  dt: 0.1
  device: cpu

stimulus:
  type: ring           # ← your new stimulus key
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
- [ ] `from_config()` + `to_dict()` are round-trip inverses
- [ ] `get_param_spec()` provides UI-ready descriptors with units and ranges
- [ ] Registered in `register_all()` with a lowercase snake_case key
- [ ] Unit tests cover shape, physics, and roundtrip
- [ ] Docstring includes tensor shapes and physical units
