# Adding a Grid Arrangement

A grid arrangement turns parameters into receptor coordinates: `[M, 2]` `(x, y)` positions in
mm. The five built-ins (`grid`, `poisson`, `hex`, `jittered_grid`, `blue_noise`,
`core/grid_arrangements.py`) are all `BaseGrid` subclasses registered in `GRID_REGISTRY`, and a
plugin adds one the same way. See [Sensor Arrays](../concepts/sensor_arrays.md) for what the
built-ins do and how composite/imported layouts compose with them.

The complete runnable example is `docs/examples/grid_arrangement_plugin.py`: it defines a
spiral (fovea-style radial) arrangement, registers it, checks the component contract, and then
drives a real `SimulationEngine` run through `GridConfig.coords_file` -- the same import path a
hand-digitised layout uses, so the engine never needs to know the arrangement plugin exists. It
is executed by `tests/docs/test_docs_examples.py`.

---

## 1. The contract

| Method | Purpose |
|---|---|
| `__init__(..., device="cpu")` | Compute coordinates once, store them, and call `super().__init__(xlim, ylim, device)` with the bounding box they span. |
| `get_all_coordinates()` | Return the stored `[M, 2]` tensor. No Python loops over receptors for anything that can be vectorised (a per-point relaxation pass, as `blue_noise` uses, is the one built-in exception). |
| `to_dict()` / `from_config()` | Every constructor parameter must round-trip (`_assert_to_dict_roundtrip_complete`, F-045); `from_config(instance.to_dict())` must be a fixed point. |
| `get_param_spec()` | A `ParamSpec` per user-facing parameter, for the GUI. |

Two things worth knowing about the built-ins, so you don't have to rediscover them:

- `ReceptorGrid` (`core/grid.py`) implements every built-in arrangement's coordinate math
  directly, dispatched on an `arrangement` string; `core/grid_arrangements.py`'s classes are thin
  subclasses that pin one `arrangement` value and add `get_param_spec()` so each one is
  independently registrable and contract-checkable. A plugin arrangement does not need to touch
  either file -- it is a standalone `BaseGrid` subclass, as the worked example is.
- A random arrangement should draw jitter from a **per-instance** `torch.Generator` seeded by a
  `seed` constructor parameter, not `torch.manual_seed` — building a grid must never read or
  advance the global RNG (F-050). See `core/grid.py`'s `_seeded_generator`/`_randn_seeded`/
  `_rand_seeded` helpers.

---

## 2. Write the class

```python
import math
import torch

from sensoryforge.core.grid_base import BaseGrid
from sensoryforge.stimuli.base import ParamSpec


class SpiralArrangement(BaseGrid):
    """Receptors on an Archimedean spiral: r_k = pitch_mm * sqrt(k), golden-angle spacing."""

    _GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

    def __init__(self, n_points=200, pitch_mm=0.1, center=(0.0, 0.0), device="cpu"):
        self.n_points = int(n_points)
        self.pitch_mm = float(pitch_mm)
        self.center = tuple(center)
        k = torch.arange(self.n_points, dtype=torch.float64)
        r = self.pitch_mm * torch.sqrt(k)
        theta = k * self._GOLDEN_ANGLE
        x = self.center[0] + r * torch.cos(theta)
        y = self.center[1] + r * torch.sin(theta)
        coords = torch.stack([x, y], dim=1).to(torch.float32)
        x_max = float(coords[:, 0].abs().max()) if self.n_points else 0.0
        y_max = float(coords[:, 1].abs().max()) if self.n_points else 0.0
        xlim = (self.center[0] - x_max, self.center[0] + x_max)
        ylim = (self.center[1] - y_max, self.center[1] + y_max)
        super().__init__(xlim, ylim, device)
        self._coords = coords.to(self.device)

    def get_all_coordinates(self):
        return self._coords

    def to_dict(self):
        result = super().to_dict()
        result.update({"n_points": self.n_points, "pitch_mm": self.pitch_mm,
                        "center": list(self.center)})
        return result

    @classmethod
    def from_config(cls, config):
        keys = {"n_points", "pitch_mm", "center", "device"}
        kwargs = {k: v for k, v in config.items() if k in keys}
        if isinstance(kwargs.get("center"), list):
            kwargs["center"] = tuple(kwargs["center"])
        return cls(**kwargs)

    @classmethod
    def get_param_spec(cls):
        return [
            ParamSpec("n_points", dtype="int", default=200, min_val=1, max_val=100000),
            ParamSpec("pitch_mm", dtype="float", default=0.1, min_val=1e-4, unit="mm"),
        ]
```

---

## 3. Register

```python
from sensoryforge.registry import GRID_REGISTRY

GRID_REGISTRY.register("spiral", SpiralArrangement)
```

Two routes, same as every other component kind (see
[Plugin Packages](../developer_guide/plugins.md)):

- **Plugin package (for third parties):** `sensoryforge new-component grid Spiral` scaffolds an
  installable package with a `pyproject.toml` entry point, discovered automatically.
- **In-repo (for contributing to SensoryForge):** `sensoryforge new-component grid Spiral
  --in-repo` and add one line to `register_components.py`'s `register_all()`.

---

## 4. Check it, then use it

```python
from sensoryforge.testing.contracts import check_component

check_component("grid", SpiralArrangement, SpiralArrangement(n_points=50))
```

A registered arrangement is not, by itself, something `GridConfig` can select directly --
`GridConfig.arrangement` only knows the five built-in names. To drive a `SimulationEngine`
run with your arrangement's coordinates, build an instance, save its coordinates, and point
`GridConfig.coords_file` at them (Phase 2, Wave L1):

```python
import tempfile
from pathlib import Path
import torch

from sensoryforge.config.schema import GridConfig, PopulationConfig, SensoryForgeConfig
from sensoryforge.core.simulation_engine import SimulationEngine

spiral = SpiralArrangement(n_points=64, pitch_mm=0.12)
with tempfile.TemporaryDirectory() as tmp:
    coords_path = Path(tmp) / "spiral.pt"
    torch.save(spiral.get_all_coordinates(), coords_path)

    config = SensoryForgeConfig(
        grids=[GridConfig(name="spiral_grid", coords_file=str(coords_path))],
        populations=[PopulationConfig(name="P", innervation_method="gaussian",
                                       neurons_per_row=3, filter_method="sa")],
    )
    engine = SimulationEngine(config)
    print(engine.populations[0]["bank"].num_receptors)  # 64
```

`SimulationEngine` samples the stimulus at each of those 64 coordinates -- real receptor
sampling, not a pixel-index assumption (Wave L3) -- so a raster stimulus at any resolution
drives the population correctly regardless of how the points were arranged.
