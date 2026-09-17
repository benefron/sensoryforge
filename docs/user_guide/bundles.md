# Data Bundles

A **data bundle** is a self-contained directory holding one completed run: the config,
each population's receptive fields, the stimulus, and every population's drive/filtered/
spikes arrays. It is the contract SensoryForge, pressure-simulation, and downstream
learning pipelines share (Phase 2, Wave J; ledger `F-011`, `F-013`).

```
bundle_dir/
    config.json               # schema_version "2.0.0", kind "sensoryforge_bundle"
    population_01_<NAME>.pt   # ReceptiveFieldBank.save() output + grid_shape
    population_02_<NAME>.pt   # one file per population
    stimuli/
        stimulus.json         # tagged payload; pressure-simulation's schema where it maps
    neuron_modules/
        sensoryforge.json     # schema_version "1.0.0", kind "neuron_module"
    data.h5                   # frames, time axis, per-population drive/filtered/spikes
```

Write one with `sensoryforge run --bundle DIR` (see [CLI Reference](cli.md)), with
`sensoryforge batch` (every stimulus in a sweep gets its own bundle -- see
[Batch Processing](batch_processing.md)), or from Python with
[`SimulationEngine.run(bundle_dir=...)`][sensoryforge.core.simulation_engine.SimulationEngine.run].
Read one with `sensoryforge.io.bundle.load_bundle()`.

Writing a bundle is a fixed function (`sensoryforge.io.bundle`), not a registry-backed plugin
point like the seven component kinds in `docs/extending/` -- there is no `EXPORTER_REGISTRY` and
no supported way to add a second, alternative bundle format today. A different output format is a
separate concern from the bundle's own schema versioning described below.

## `config.json`

A superset of pressure-simulation's `1.0.0` "mechanoreceptor bundle" format, so its
viewer (`GUIs/ebkf_viewer.py`) opens a SensoryForge bundle unchanged -- it only reads
`grid` and `populations[*].tensors`/`name`.

| Field | Meaning |
|---|---|
| `schema_version` | `"2.0.0"`. `load_bundle` raises `ValueError` if the major version isn't `2`. |
| `kind` | `"sensoryforge_bundle"`. |
| `grid` | `{rows, cols, spacing_mm, center_mm, device}` of the run's primary grid. |
| `populations` | One entry per population: `name`, `neuron_type`, `color`, `parameters` (the builder parameters pressure-simulation's format expects: `neurons_per_row`, `connections_per_neuron`, `sigma_d_mm`, `weight_min`, `weight_max`, `seed`, `edge_offset`), `tensors` (the `.pt` filename), `visible`. |
| `config` | The full canonical config (`SensoryForgeConfig.to_dict()`). |
| `sensoryforge_version`, `created_at`, `bundle_created` | Provenance. |

## `population_NN_<NAME>.pt`

Exactly `ReceptiveFieldBank.save()` output, plus `grid_shape`:

| Key | Shape | Units |
|---|---|---|
| `innervation_weights` | `[N, M]` | dimensionless |
| `neuron_centers` | `[N, 2]` | mm, `(x, y)` |
| `receptor_coords` | `[M, 2]` | mm, `(x, y)` |
| `grid_shape` | `[rows, cols]` | -- lets a consumer reshape `[N, M]` to `[N, rows, cols]` |
| `provenance` | dict | builder name, its `to_dict()`, seed, source path (imported banks) |

`ReceptiveFieldBank.load(path)` reads this file directly; `load_bundle` uses it under the
hood for every population's `bank`.

## `stimuli/stimulus.json`

Always tagged with a `schema_version` and a `kind`, so a reader can tell which schema
it is holding rather than guessing. This matters more than it looks: pressure-simulation's
`generate_stimulus_from_json` reads every field with a `.get` default, so handed an empty
or foreign payload it does not raise. It silently returns a static Gaussian blob at the
origin, and its viewer will encode that and draw entirely plausible plots of a stimulus
you never ran.

Two kinds are written.

`kind: "stimulus"`, `schema_version: "1.0.0"` is pressure-simulation's own schema. It is
used only for the stimulus types that regenerate there exactly: `gaussian`, `point` and
`edge`. That claim is not asserted on trust. `tests/integration/test_bundle_stimulus_payload.py`
regenerates the frames from the written payload and compares them to the bundle's own
`/stimulus/frames` at zero tolerance.

`kind: "sensoryforge_stimulus"` is used for everything else, including a run whose caller
passed no stimulus config at all. It carries `reconstructible_by_pressure_simulation: false`,
so nothing downstream mistakes it for a payload that can be replayed there.

Both kinds keep the caller's original dict verbatim under a `sensoryforge` key, so no
information is lost in translation.

Two details of the translation are worth knowing if you write a payload by hand. The field
`total_ms` is the time of the **last sample**, `(n_frames - 1) * dt_ms`, not the duration,
because pressure-simulation builds its time axis as `arange(0, total_ms + dt/2, dt)`. And
its plateau mask is a strict `t < ramp_up + plateau`, so a plateau of exactly `total_ms`
leaves the final frame at zero. When a stimulus declares no ramps of its own, `plateau_ms`
is therefore written as `n_frames * dt_ms`.

## What pressure-simulation can and cannot do with a bundle

Its viewer loads any bundle this repository writes, and its encoder runs the
single-grid populations in one. Verified end to end, not assumed: a bundle written by
`sensoryforge run` was opened with that repository's own `_on_load_bundle` and
`run_encoding`, and produced real spiking from our receptive fields.

One boundary is worth knowing before you rely on it. A population built from several
inputs and combined with `concat` has a weight matrix whose receptor axis is the sum of
its inputs' receptor counts, because each input carries its own grid or channel.
pressure-simulation has a single receptor grid, so its encoder flattens one grid's frames
and the two widths disagree. The bundle still loads there and its weights are still
readable; it is the encoding step that cannot consume such a population. Single-input
populations, which is everything the pressure-simulation recipe builds, are unaffected.

## `neuron_modules/sensoryforge.json`

Without this file, pressure-simulation's viewer loads and displays a bundle but can
never run it: `_on_load_bundle` populates its "Neuron Module" combo box from
`sorted((bundle_dir / "neuron_modules").glob("*.json"))`, and its Run button is only
enabled once both that combo and the stimulus combo are non-empty. One file,
`sensoryforge.json`, carries every population's neuron/filter configuration:

```json
{
  "schema_version": "1.0.0",
  "kind": "neuron_module",
  "created_at": "2026-09-16T12:00:00+00:00",
  "stimulus": "stimulus.json",
  "device": "cpu",
  "population_configs": [
    {
      "name": "SA Population",
      "neuron_type": "SA",
      "model": "izhikevich",
      "filter_method": "sa",
      "enabled": true,
      "input_gain": 50.0,
      "noise_std": 0.0,
      "model_params": {},
      "filter_params": {},
      "selected_neuron": 0
    }
  ]
}
```

`_on_run` (the viewer's encode step) reads only `enabled`, `name`, `neuron_type`,
`filter_method`, `noise_std`, `model_params` and `filter_params` from each entry --
`model` is metadata (never read there) and `input_gain` is always overridden by the
viewer's own gain spinboxes. It matches an entry to a population by exact `name`
against `config.json`'s `populations[*].name` and silently drops any entry that
doesn't match, so `name` here is the *raw* population name, not the filesystem-safe
form `population_NN_<NAME>.pt` uses.

## `data.h5`

| Path | Shape | Dtype | Notes |
|---|---|---|---|
| `/stimulus/frames` | `[T, H, W]` or `[T, C, H, W]` | float32 | gzip-4 compressed |
| `/time_ms` | `[T]` | float32 | `t = i * dt_ms` |
| `/populations/<name>/drive` | `[T, N]` | float32 | mA, before the neuron model |
| `/populations/<name>/filtered` | `[T, N]` | float32 | mA, after filter/gain/noise |
| `/populations/<name>/spikes` | `[T, N]` | int16 | **counts**, see below |
| `/populations/<name>/state` | `[T, N]` | float32 | analog readouts (Wave N; not yet populated) |

Root attributes: `dt_ms`, `integrate_dt_ms`, `seed` (`-1` if none was given),
`sensoryforge_version`. The `/meta` group carries `config_yaml` (the full config as
YAML text) and `provenance_json` (every population's bank provenance, JSON-encoded).

### Spikes are per-bin counts, not a binary raster

`spikes[t, n]` is the number of sub-steps within record bin `t` that neuron `n` fired
in (F-008: the neuron integrates at `integrate_dt_ms`, finer than the record step
`dt_ms`, and each bin sums however many sub-step spikes landed in it). It can be
greater than 1. To recover a conventional binary raster:

```python
binary_raster = spikes > 0
```

## Reading a bundle

### With `load_bundle` (recommended)

```python
from sensoryforge.io.bundle import load_bundle

bundle = load_bundle("path/to/bundle")
bundle.config              # SensoryForgeConfig
bundle.banks["SA Population"].weights        # [N, M] torch.Tensor
bundle.stimulus                                # [T, H, W] or [T, C, H, W]
bundle.populations["SA Population"]["spikes"]  # [T, N] int16 counts
bundle.meta["dt_ms"], bundle.meta["seed"]
```

### With `h5py` directly (no SensoryForge import needed)

```python
import h5py

with h5py.File("path/to/bundle/data.h5", "r") as f:
    dt_ms = f.attrs["dt_ms"]
    spikes = f["populations"]["SA Population"]["spikes"][()]  # numpy array
```

### As a `pandas.DataFrame`

```python
import pandas as pd

df = pd.DataFrame(spikes, columns=[f"neuron_{i}" for i in range(spikes.shape[1])])
df.insert(0, "time_ms", time_ms)
```

### From pressure-simulation

Its viewer reads `config.json`'s `grid` and `populations[*].tensors` unchanged (it
predates the `.h5` payload and never looks for it), and rebuilds a population's drive
as `stimulus.view(T, H*W) @ W.T` using the receptor ordering documented in
[Receptive Fields](receptive_fields.md#coordinates-and-ordering). See
`tests/integration/test_bundle_pressure_sim_compat.py` for a re-implementation of that
loader used to verify this.

## The executed example

[`docs/examples/read_bundle.py`](https://github.com/benefron/sensoryforge/blob/main/docs/examples/read_bundle.py)
writes a small bundle and reads it back through every path above (`load_bundle`, raw
`h5py`, and `pandas`, when installed); `tests/docs/test_docs_examples.py` runs it as
part of the test suite.
