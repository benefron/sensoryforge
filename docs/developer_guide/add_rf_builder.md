# Adding a Receptive-Field Builder

A receptive-field builder turns receptor coordinates (and, usually, neuron centres) into
a `ReceptiveFieldBank`: the `[N, M]` weights of one population plus its geometry and
provenance. The six built-in builders (`gaussian`, `distance_weighted`, `one_to_one`,
`uniform`, `template`, `imported`) are all `BaseInnervation` subclasses registered in
`INNERVATION_REGISTRY`, and a plugin adds one the same way. See
`../user_guide/receptive_fields.md` for what the built-ins do.

The complete runnable example is `docs/examples/rf_builder_plugin.py`; it defines,
registers and simulates a builder end to end and is executed by the test suite.

---

## 1. The contract

| Method | Purpose |
|---|---|
| `__init__(receptor_coords, neuron_centers, ..., device="cpu")` | Store the geometry and your parameters. Keep new parameters keyword-friendly and after the two tensors. |
| `compute_weights()` | Return `[N, M]` float weights; `weights[n, m]` is the gain from receptor `m` to neuron `n`. Vectorise with broadcasting; no Python loops over neurons. |
| `build(receptor_coords=None, neuron_centers=None, device=None)` | Inherited. Wraps `compute_weights()` in a bank with provenance. Override only to add provenance or to derive your own centres. |
| `to_dict()` / `from_config()` | Every constructor parameter except the two tensors must round-trip; `from_config(instance.to_dict())` must be a fixed point. |
| `get_param_spec()` | A `ParamSpec` per user-facing parameter, for the GUI. |

Two class attributes matter:

- `_TO_DICT_EXCLUDE_PARAMS = ("receptor_coords", "neuron_centers")` keeps the tensors out
  of the round-trip check (they are stored by the bank, not the config).
- `DERIVES_NEURON_CENTERS = True` tells the engine and the GUI that the builder lays out
  its own neurons (like `template`). They then pass `neuron_centers=None`, skip the
  population's `neurons_per_row` lattice, and warn that it is ignored.

`BaseInnervation.filter_params(params)` keeps only the keys your constructor takes from
the union of every method's parameters that a population config carries. That is how the
engine can hand the same `PopulationConfig` to any builder.

---

## 2. Write the class

```python
import torch

from sensoryforge.core.innervation import BaseInnervation
from sensoryforge.stimuli.base import ParamSpec


class RingRFBuilder(BaseInnervation):
    """Annular receptive fields: weight 1 on receptors within [r_in, r_out] of a neuron."""

    def __init__(self, receptor_coords, neuron_centers, r_in_mm=0.1, r_out_mm=0.4,
                 device="cpu"):
        super().__init__(receptor_coords, neuron_centers, device)
        if r_out_mm <= r_in_mm:
            raise ValueError(f"r_out_mm must exceed r_in_mm, got {r_in_mm} >= {r_out_mm}")
        self.r_in_mm = float(r_in_mm)
        self.r_out_mm = float(r_out_mm)

    def compute_weights(self, **kwargs):
        d = torch.cdist(self.neuron_centers, self.receptor_coords)  # [N, M] in mm
        return ((d >= self.r_in_mm) & (d <= self.r_out_mm)).to(torch.float32)

    def to_dict(self):
        result = super().to_dict()
        result.update({"r_in_mm": self.r_in_mm, "r_out_mm": self.r_out_mm})
        return result

    @classmethod
    def get_param_spec(cls):
        return [
            ParamSpec("r_in_mm", dtype="float", default=0.1, min_val=0.0, unit="mm"),
            ParamSpec("r_out_mm", dtype="float", default=0.4, min_val=0.0, unit="mm"),
        ]
```

`BaseInnervation.from_config` already pops the two tensors and the derived
`method`/`num_neurons`/`num_receptors` keys and forwards the rest to `__init__`, so this
class needs no `from_config` override. `to_dict()["method"]` is derived from the class
name; the bank's provenance uses the registry name instead.

---

## 3. Register it

**Plugin package (recommended):** `sensoryforge new-component innervation RingRF
--dest DIR` scaffolds an installable package whose entry point registers the class; see
`plugins.md`. **In-repo:** add one line to `register_all()` in
`sensoryforge/register_components.py`:

```python
INNERVATION_REGISTRY.register("ring", RingRFBuilder)
```

Names are matched case-insensitively (F-046).

---

## 4. Test it

```python
import torch
from sensoryforge.testing.contracts import check_component

coords = torch.rand(30, 2)
centers = torch.rand(4, 2)
check_component("innervation", RingRFBuilder, RingRFBuilder(coords, centers))
```

`check_component` asserts the `ParamSpec` list, one `compute_weights()` pass with the
right shape, the `to_dict()`/`from_config()` fixed point over every constructor
parameter, and, since Phase 2, that `build()` returns a `ReceptiveFieldBank` with
`[N, M]` weights, `[N, 2]`/`[M, 2]` coordinates and a provenance naming the builder.

Then use it from a config:

```yaml
populations:
  - name: ring population
    innervation_method: ring
    innervation_params: {r_in_mm: 0.1, r_out_mm: 0.4}
```

`innervation_params` is merged last into the parameters the engine passes to the
builder, so it can carry anything your constructor accepts.

---

## 5. Builders that derive their own centres

If your builder decides where the neurons are (a designed lattice, a file), set
`DERIVES_NEURON_CENTERS = True`, accept `neuron_centers=None` in `__init__` (warn if one
is given), compute the centres before calling `super().__init__`, and override
`from_config` to tolerate a missing `neuron_centers` key. `TemplateRFBuilder`
(`sensoryforge/core/rf_builders/template.py`) and `ImportedRFBuilder`
(`rf_builders/imported.py`) are the two shipped examples.

---

## Conventions to keep

- Coordinates are `(x, y)` in mm; convert at the boundary if a file uses `[y, x]`.
- Receptor `k` of a regular grid is `k = i * cols + j` with x as the slow index. Weight
  column `k` must refer to that receptor.
- Draw any randomness from a per-instance `torch.Generator` seeded by a `seed` parameter
  (see `_seeded_generator` in `innervation.py`); never call `torch.manual_seed`.
- Google-style docstrings with tensor shapes and units.
