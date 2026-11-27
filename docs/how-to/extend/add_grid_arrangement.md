# Adding a Layout (Grid Arrangement)

A layout turns settings into positions: `[N, 2]` `(x, y)` in mm. You write it once and can
then name it in a config as a **sensor layout** (`grids[].arrangement`), as a **neuron
layout** (`populations[].neuron_arrangement`), or both. The five built-ins (`grid`, `poisson`,
`hex`, `jittered_grid`, `blue_noise`) have their own code; a plugin cannot take their names.
See [Sensor Arrays](../../concepts/sensor_arrays.md) for what the built-ins do.

The complete runnable example is `docs/examples/grid_arrangement_plugin.py`: a spiral that
is registered, checked, and run as both the sensor and the neuron layout. It is executed by
`tests/docs/test_docs_examples.py`.

An installed layout also appears in the app's two arrangement lists, with a form built from its
`get_param_spec()`. The app loads a plugin package through its entry point. For a config naming
a layout that is not installed, the app keeps the name and reports the problem.

---

## 1. The contract

| Method | Purpose |
|---|---|
| `__init__(self, xlim, ylim, device, ...)` | `xlim` and `ylim` are the area to fill, in mm. Then the layout's own settings, each with a default or required. Take `seed` only if the layout is random. Call `super().__init__(xlim, ylim, device)`. |
| `get_all_coordinates()` | Return `[N, 2]`, `N >= 1`. **The layout decides `N`.** Every point must lie in the area; SensoryForge raises on one outside and never moves or drops points. Row `k` is sensor or neuron `k` for every later stage. |
| `to_dict()` / `from_config()` | Every constructor parameter must round-trip; `from_config(instance.to_dict())` must be a fixed point. |
| `get_param_spec()` | A `ParamSpec` per setting. |

A random layout draws from its own `torch.Generator` seeded by `seed`, never from the global
random number generator. When the config sets no seed, the constructor's own default applies.

---

## 2. Write the class

```python
import math
import torch

from sensoryforge.core.grid_base import BaseGrid
from sensoryforge.stimuli.base import ParamSpec


class SpiralArrangement(BaseGrid):
    """A sunflower spiral filling the largest circle that fits in the area."""

    _GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

    def __init__(
        self,
        xlim: Tuple[float, float] = (-1.0, 1.0),
        ylim: Tuple[float, float] = (-1.0, 1.0),
        device: torch.device | str = "cpu",
        pitch_mm: float = 0.1,
    ) -> None:
        if not pitch_mm > 0:
            raise ValueError(f"pitch_mm must be > 0, got {pitch_mm}")
        super().__init__(tuple(xlim), tuple(ylim), device)
        self.pitch_mm = float(pitch_mm)
        radius = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2
        n_points = int((radius / self.pitch_mm) ** 2 + 1e-9) + 1
        k = torch.arange(n_points, dtype=torch.float64)
        r = self.pitch_mm * torch.sqrt(k)
        theta = k * self._GOLDEN_ANGLE
        x = (xlim[0] + xlim[1]) / 2 + r * torch.cos(theta)
        y = (ylim[0] + ylim[1]) / 2 + r * torch.sin(theta)
        self._coords = torch.stack([x, y], dim=1).to(self.device, torch.float32)

    def get_all_coordinates(self) -> torch.Tensor:
        return self._coords

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["pitch_mm"] = self.pitch_mm
        return result

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SpiralArrangement":
        keys = {"xlim", "ylim", "device", "pitch_mm"}
        return cls(**{k: v for k, v in config.items() if k in keys})

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        return [
            ParamSpec(
                "pitch_mm",
                label="Pitch",
                dtype="float",
                default=0.1,
                min_val=1e-4,
                unit="mm",
            ),
        ]
```

---

## 3. Register

```python
from sensoryforge.registry import GRID_REGISTRY

GRID_REGISTRY.register("spiral", SpiralArrangement)
```

Two routes, same as every other component kind (see
[Plugin Packages](plugins.md)):

- **Plugin package (for third parties):** `sensoryforge new-component grid Spiral` scaffolds an
  installable package with a `pyproject.toml` entry point, discovered automatically. Its
  placeholder draws uniform random points from a generator seeded by its `seed` parameter, so
  `from_config(to_dict())` gives the same points; keep randomness that way.
- **In-repo (for contributing to SensoryForge):** `sensoryforge new-component grid Spiral
  --in-repo` and add one line to `register_components.py`'s `register_all()`.

---

## 4. Check it

```python
from sensoryforge.testing.contracts import check_component

check_component("grid", SpiralArrangement, SpiralArrangement(pitch_mm=0.2))
```

For a layout that takes the area, the check also builds it on two other areas and fails if a
point falls outside, or if the same settings and seed give different points.

---

## 5. Use it in a config

Name the layout exactly as registered, and put its settings under `layout_params` (sensors) or
`neuron_layout_params` (neurons). A plugin package is loaded through its entry point or the
config's `plugins:` list.

```yaml
grids:
  - name: skin
    arrangement: spiral
    rows: 12          # rows, cols, spacing and the centre give the area, not the count
    cols: 8
    spacing: 0.2
    layout_params: {pitch_mm: 0.1}
populations:
  - name: SA
    neuron_arrangement: spiral
    neuron_layout_params: {pitch_mm: 0.25}
    innervation_method: gaussian
    filter_method: sa
    seed: 1
```

| | Sensor layout | Neuron layout |
|---|---|---|
| Area | `(rows - 1) * spacing` along x by `(cols - 1) * spacing` along y, around the centre; also the stimulus canvas | The grid's bounds, shrunk by `edge_offset` |
| Seed | The grid's `seed`; an error if the layout takes none | The population's `seed`, if the layout takes one |
| Ignored or refused | `density` and `coords_file` are errors | `neurons_per_row`, `neuron_rows`, `neuron_cols` and `neuron_jitter_factor` are ignored, with a warning if set |

A receptive-field builder that places its own neurons (`one_to_one`, `template`) ignores a
neuron layout and warns.

**Wiring differs from the built-in grids.** A plugin sensor grid is wired like a
position-file or composite grid: the `gaussian` builder only connects sensors within 3 sigma of
a neuron. The built-in arrangements apply no such cut-off. The same sensor positions and seed
therefore give the same sensor readings but different weights. To remove the cut-off, set
`innervation_params: {max_sigma_distance: 0.0}` on the population.

A saved run's config names the layout, so rebuilding an engine from it needs the plugin
installed. The error names the missing layout; no other layout is used in its place.

---

## 6. A grid plugin that does not take the area

A `BaseGrid` whose constructor lacks `xlim`, `ylim` or `device` cannot be named in a config.
Save its points (`torch.save(grid.get_all_coordinates(), "points.pt")`, or a CSV) and set the
grid's `coords_file:` to that file. This also suits measured or hand-drawn positions.
