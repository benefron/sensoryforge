# Receptive Fields

A population's receptive fields are the weights that map receptor responses onto its
neurons. Since Phase 2 they are one component: a **`ReceptiveFieldBank`** built by a
registered **builder**. The engine, the CLI, the batch runner and the GUI all build banks
through the same path, so a config exported from the GUI produces the same weights when
run from the command line.

```
Stimulus  [batch, time, rows, cols]
    ↓  flatten row-major → receptor responses [batch, time, M]
    ↓  ReceptiveFieldBank.forward  (weights [N, M])
Drive     [batch, time, N]
```

## Biological versus designed receptive fields

There are two ways to think about a receptive field:

- **Biological** builders sample connections the way an innervation experiment might:
  each neuron draws a random subset of nearby receptors (`gaussian`,
  `distance_weighted`), takes exactly the *k* nearest (`one_to_one`), or every
  receptor is owned by its nearest neuron (`uniform`). They take a `seed`, and their
  weights depend on it.
- **Designed** receptive fields start from a target of the *readout*, not from anatomy.
  The `template` builder takes one number, the resolvable distance `d`, and derives
  everything else from it. It is deterministic: two builds are bit-identical.
- **Imported** receptive fields come from a file: a folder the GUI exported, a saved
  bank, a population file from a companion project, or a designed matrix computed
  elsewhere.

All six builders live in `INNERVATION_REGISTRY` and are selected per population with
`innervation_method`.

## The design chain: d → σ, Δ, N

The `template` builder implements the design used by the pressure-simulation project:

| Quantity | Formula | With `d = 0.40 mm` |
|---|---|---|
| cutoff spatial frequency | `f_c = 1 / (2 d)` | 1.25 cycles/mm |
| Gaussian width | `σ = d / π` | 0.1273 mm |
| lattice pitch | `Δ = d` | 0.40 mm |
| number of neurons | `N = A / Δ²` | 36 |

`A` is the receptor area: the receptor bounding box extended by half a receptor spacing
on each side. A 16 × 16 grid at 0.15 mm spans 2.25 mm between its outermost receptors,
so the extended side is 2.4 mm and `2.4² / 0.4² = 36` neurons on a 6 × 6 lattice. The
lattice is inset from that box by `edge_offset_mm` (default `Δ / 2`) and centred, so the
margins are symmetric.

Each neuron then gets one Gaussian template, translated to its centre, truncated to the
`k` nearest receptors (default 28) with weights `exp(-r² / 2σ²)`, and normalised to unit
L2 norm (`normalize: l2`, the default; `sum` and `none` are also available). Interior
neurons therefore have identical templates up to translation.

Because the neuron count is derived, any `neurons_per_row`, `neuron_rows` or
`neuron_cols` on a `template` population is ignored, with a warning.

```yaml
populations:
  - name: SA designed
    neuron_type: SA
    innervation_method: template
    resolvable_distance_mm: 0.40
    innervation_params:      # optional
      k: 28
      normalize: l2
    filter_method: sa
```

You can instead give `sigma_mm` and `pitch_mm` explicitly in `innervation_params`;
giving both forms, or neither, raises.

## The six builders

| `innervation_method` | Neuron centres | Weights | Seeded |
|---|---|---|---|
| `gaussian` | `neurons_per_row` lattice | Poisson-distributed number of receptors per neuron, chosen with Gaussian probability, analytic Gaussian weights (`use_distance_weights: true`) or uniform random weights (the control arm) | yes |
| `distance_weighted` | lattice | as `gaussian`, with an exponential / linear / inverse-square decay and a hard `max_distance_mm` | yes |
| `one_to_one` | lattice | exactly `connections_per_neuron` receptors per neuron | yes |
| `uniform` | lattice | every receptor connects to its nearest neuron (Voronoi-like) | yes |
| `template` | derived from `d` | analytic Gaussian on the `k` nearest receptors, unit-L2 rows | no |
| `imported` | from the file | from the file | no |

Every builder exposes `get_param_spec()`, so the GUI shows its parameters, and
`build()` returns a bank whose `provenance` records the builder name, its full
configuration, the seed and the SensoryForge version.

## Importing receptive fields

`innervation_method: imported` with `innervation_params: {path: ...}` accepts three
sources:

1. **A CSV folder** exported by the GUI: `manifest.json`, `neuron_positions.csv`
   (`x_mm,y_mm`), `innervation_weights.csv` (N rows × M columns) and, since Phase 2,
   `bank.pt`.
2. **A `.pt` file**: a bank saved with `ReceptiveFieldBank.save`, or a
   pressure-simulation population file (`innervation_weights` of shape `[N, H, W]` or
   `[N, M]` plus `neuron_centers`).
3. **An `.npz` file** shaped like pressure-simulation's `ConstructedRF`: `H [N, N_grid]`,
   `centers [N, 2]` in `[y, x]` order, optional `sigma` and `pitch`.

The receptor count in the file must equal the target grid's; otherwise the import
raises an error naming both counts rather than padding with zeros. The bank's
provenance records the absolute source path and a SHA-256 of the file(s).

## Coordinates and ordering

- Inside SensoryForge every coordinate is `(x, y)` in millimetres. Files from
  pressure-simulation store centres as `[y, x]`; the `imported` builder swaps them at the
  boundary.
- A regular grid `ReceptorGrid(grid_size=(rows, cols))` builds its meshgrids with
  `indexing="ij"`, so the **first index is x** and the second is y. Receptor `k` of the
  flattened list is `k = i * cols + j`, the receptor at `(xx[i, j], yy[i, j])`.
- Stimulus frames `[time, rows, cols]` are flattened row-major the same way before they
  meet the bank, so weight column `k` always refers to receptor `k`.
- Pressure-simulation flattens its grids y-slow, x-fast. The `imported` builder
  re-orders the columns of an `.npz` `H` matrix to SensoryForge's ordering when the target
  grid is a full lattice, so a neuron's weights sit at its (converted) centre.
- Neuron centres from the `template` builder follow the same x-slow, y-fast ordering.

## Reproducibility

- Random receptor arrangements (`jittered_grid`, `blue_noise`, `poisson`) take a `seed`
  on `GridConfig`, `ReceptorGrid` and `CompositeReceptorGrid.add_layer`. The jitter is
  drawn from a per-instance generator, so building a grid never touches the global RNG and
  the same seed gives the same layout on every device.
- Biological builders take `seed` on the population. The same seed gives the same weights
  on CPU, MPS and CUDA.
- `template` and `imported` are deterministic.

## Working with banks in code

```python
from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.core.innervation import build_population_bank

coords = ReceptorGrid(grid_size=(16, 16), spacing=0.15).get_receptor_coordinates()
bank = build_population_bank(
    receptor_coords=coords,
    innervation_method="template",
    resolvable_distance_mm=0.40,
)
print(bank.num_neurons)            # 36
print(bank.provenance["derived"])  # sigma_mm, pitch_mm, lattice_shape, ...
bank.save("sa_designed.pt")
```

`bank(receptor_responses)` accepts `[batch, M]` or `[batch, time, M]` and returns the
neuron drive. `SimulationEngine` stores each population's bank under
`engine.populations[i]["bank"]`.

The older `InnervationModule` and `FlatInnervationModule` classes still work but emit a
`DeprecationWarning`; they build a bank internally and will be removed in Phase 4.
