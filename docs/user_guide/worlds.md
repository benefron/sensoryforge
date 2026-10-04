# Worlds and data sets

The exact guarantees, for anyone writing code against this, are in the
[World engine contract](../reference/world_contract.md). This page teaches the pieces.

## What a world is

A **world** is a declared population of stimuli. It is a set of **classes**, each a
[layered](designing_stimuli.md) layer (a shape, a pattern, a motion, a modulation) whose
fields are not fixed numbers but **axes**: a range, a list of values or a named
distribution to draw from. Sampling a world gives **draws**, plain records that say which
class was drawn and every value it drew. Rendering turns a draw into frames on any
coordinates. Worlds are for building training and test data sets, for hypothesis tests that
need many stimuli with known factors, and for simulation-based inference.

## A first world

```yaml
world:
  name: first
  modality: tactile
  defaults:
    delay_ms:   {range: [0, 20]}
    touch_ms:   {range: [5, 15]}
    hold_ms:    {range: [20, 60]}
    release_ms: {range: [5, 15]}
    amplitude:  {range: [0.5, 1.0]}
    x_mm: {range: [-0.3, 0.3]}
    y_mm: {range: [-0.3, 0.3]}
  classes:
    dots:
      weight: 0.5
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.15, 0.45]}}
    edges:
      weight: 0.4
      layer: {shape: {kind: bar, profile: gaussian, length_mm: 0}}
      axes:
        width_mm: {range: [0.05, 0.15]}
        orientation_deg: {range: [0, 180], circular: true}
    quiet:
      kind: quiet
      weight: 0.1
      axes: {quiet_ms: {range: [20, 60]}}
```

```bash
sensoryforge world validate world.yml
sensoryforge world sample world.yml -n 3 --seed 7
```

`validate` prints the world's id, then each class with its kind, weight and the axes it
draws; `sample` prints one JSON record per draw. The world's id (`w-` and 12 hex digits)
changes whenever any value changes, but not when only the `description` does, and not when
the classes, axes or defaults are written in another order: classes are weighed in order
of their names, so the draws do not change either.

For a runnable world that uses every class kind, shape family, motion and modulation, see
`tests/fixtures/worlds/tactile_small.yml` (and `tests/fixtures/worlds/dataset_small.yml`
for a data set on it).

A world is checked when it loads, and each problem is a `ValueError` naming where it is
(`world.classes.twice.axes.contacts: ...`): `contacts` must be a whole number of at least 1;
every value an axis can take, and every fixed-draw value, must lie inside its field's valid
range (durations are never negative; a gaussian's `sigma_mm` lies within its `ParamSpec`'s
0.001 to 50 mm);
numeric fields need numbers, and PyYAML reads `3e-1` as text, so write `3.0e-1`; a key
written twice, and a class name that could not name a directory, are refused.

## Axes

An axis is one of five forms:

| Form | Meaning |
|---|---|
| `{value: v}` | a constant |
| `{range: [lo, hi]}` | uniform between the bounds; add `dist: log_uniform` for a log scale, `int: true` for integers, `circular: true` for angles, `probes: false` to keep it out of probe splits |
| `{values: [...], weights: [...]}` | a categorical choice, optionally weighted |
| `{dist: name}` | a registered distribution; `braille_cells` is the 63 non-empty braille cells, uniform |

An axis name binds to the first field it matches, in this order: an episode field
(`delay_ms`, `touch_ms`, `hold_ms`, `slide_ms`, `release_ms`, `contacts`, `pause_ms`,
`speed_mm_per_ms`, `direction_deg`), then `amplitude`, then `x_mm` and `y_mm`, then a shape,
pattern or modulation field by its bare name. A dotted path (`shape.width_mm`) names a
field exactly. A name that matches two fields, or none, fails when the world loads:

```yaml
edges:
  layer:
    shape:   {kind: bar, profile: gaussian}
    pattern: {kind: random, count: 3}      # random also has width_mm
  axes: {width_mm: {range: [0.05, 0.15]}}  # error: ambiguous, say shape.width_mm
```

The value of a field is the class's own axis if it has one; otherwise a value its layer
fixes; otherwise the world's `defaults`; otherwise a built-in (durations 0, `contacts` 1,
`amplitude` 1, `x_mm` and `y_mm` 0). `fixed_draws:` names draws by explicit values, and any
axis it leaves out takes its midpoint, for example
`braille_H: {class: braille, dots: "125", hold_ms: 40, delay_ms: 0}`.

## Time

Every draw is one episode, starting at time 0: quiet for `delay_ms`, then each of
`contacts` contacts, `pause_ms` apart. A contact ramps up over `touch_ms`, holds for
`hold_ms`, slides for `slide_ms` and ramps down over `release_ms`. The pattern moves only
while sliding, at `speed_mm_per_ms` towards `direction_deg`; with several contacts the
motion is spread over all of them, so each re-touch starts where the last one ended. Before
the start, in the lead-in, in the pauses and after the end, the stimulus is exactly zero.
From `tests/fixtures/worlds/tactile_small.yml`:

```yaml
sliders:
  weight: 0.15
  layer: {shape: {kind: gaussian, sigma_mm: 0.3}}
  axes: {slide_ms: {range: [20, 40]}, hold_ms: {range: [5, 20]}}
twice:
  weight: 0.05
  layer: {shape: {kind: gaussian, sigma_mm: 0.3}}
  axes: {contacts: {value: 2}, pause_ms: {range: [5, 15]}}
```

## Vibration and taps

A `modulation` multiplies the contact's envelope, counted from each touch. `sine` is a
vibration with `frequency_hz`, `depth` and `phase_deg`; `pulses` is repeated indentation
with `rate_hz`, `duty`, `edge_ms` and `depth`. Their parameters are fields like any other,
so an axis can draw them:

```yaml
taps:
  weight: 0.1
  layer:
    shape: {kind: disc, diameter_mm: 0.6}
    modulation: {kind: pulses, duty: 0.5, edge_ms: 2}
  axes: {rate_hz: {range: [20, 80], dist: log_uniform}}
vibes:
  weight: 0.1
  layer:
    shape: {kind: gaussian, sigma_mm: 0.4}
    modulation: {kind: sine, depth: 0.3}
  axes: {frequency_hz: {range: [50, 200]}}
```

## Sampling and rendering in Python

```python
import torch
from sensoryforge.world import Canvas, load_world, render, render_movie, sample

world = load_world("world.yml")
draws = sample(world, n=1000, seed=7)
canvas = Canvas.from_grid(40, 40, 0.15)             # centred, dim 0 = x
frames = render(draws[:100], canvas, torch.tensor([0.0, 10.0, 20.0]))   # [100, 3, 40, 40]
movie = render_movie(draws[0], canvas, dt_ms=1.0, duration_ms=200.0)    # [200, 40, 40]
```

Draw *i* depends only on the world, the seed and *i*, never on how many draws were asked
for: `sample(world, n=10, seed=7)[3]` equals `sample(world, n=1000, seed=7)[3]`, and
`sample(world, indices=[3], seed=7)` gives it alone. `classes=["dots"]` restricts the draw
to named classes, and a held-out class is drawn only when named.

`Canvas(xx, yy)` takes coordinate arrays of any shape, in mm, so you can render onto a
hexagonal array or another program's mesh; `Canvas.from_grid(rows, cols, spacing_mm)` is
SensoryForge's centred grid, and `Canvas.from_grid_config(grid_cfg)` follows a grid of a
config. `render(items, canvas, times_ms)` takes times as `[K]` (shared) or `[n, K]` (one
row per draw), measured from each item's start, and returns `[n, K, *canvas shape]`;
`render_movie` evaluates at `t = k · dt_ms`. Pass `dtype=torch.float64` for the precision the
contract compares in, and `device="cuda"` to render on a GPU.

Each internal evaluation is bounded by `max_elements` (default 2**23 elements for draws ×
times × canvas), but the output is allocated whole and sized by what you pass, so render
large jobs in slices. On an Apple M3 Pro with 6 threads, 4.1 M triples at 40×40 in float64
take about 4.6–4.7 min (`docs/reference/benchmarks.md`).

## Elements added in v1.2.0

Each of these is optional and changes nothing for a world that does not use it. The values
below are placeholders sized for small canvases. Exact formulas are in the
[contract](../reference/world_contract.md#10-what-v120-adds).

### A background under the contact

`background` is a level over the whole patch that rises and falls with the same contact
envelope as the feature (it does not move with the pattern). `clamp_min` floors the layer's
total where the envelope is positive, so zero-mean relief riding on the background never
presses below zero.

```yaml
relief:
  layer:
    shape: {kind: grating, signed: true, wavelength_mm: 0.4}
    clamp_min: 0.0
  axes:
    background: {range: [0.2, 0.5]}
    orientation_deg: {range: [0, 180], circular: true}
```

### Scans across the finger

`biased_direction` draws a scan direction whose lateral travel is `travel_ratio` times the
travel along the other axis; `axis_deg` is the lateral axis (0° = +x).

```yaml
axes:
  slide_ms: {range: [20, 40]}
  speed_mm_per_ms: {range: [0.001, 0.004]}
  direction_deg: {dist: biased_direction, travel_ratio: 2.5, axis_deg: 0}
```

### Linked axes, unstratified axes and groups

```yaml
groups:
  press:                                   # sugar: the class holds the resolved axes
    touch_ms: {range: [8, 14]}
    hold_ms: {range: [20, 40]}
    release_ms: {same_as: touch_ms}        # a press falls as it rose
classes:
  grouped_press:
    use: [press]
    layer: {shape: {kind: gaussian, sigma_mm: 0.15}}
  rough:
    layer: {shape: {kind: self_affine}}
    axes:
      seed: {range: [0, 16777215], int: true, stratify: false}  # i.i.d., never binned
```

### Textures and arrays

`self_affine` fills the patch with zero-mean random relief (Hurst exponent `hurst`, roll-off
wavelength `rolloff_mm`, nothing finer than `cutoff_mm`, its own `seed`; RMS =
`amplitude / √2`). `dot_array` is a lattice of Gaussian dots (`arrangement: square` or
`hexagonal`).

```yaml
rough:
  layer:
    shape: {kind: self_affine, hurst: 0.8, rolloff_mm: 0.6, cutoff_mm: 0.12, components: 64}
    clamp_min: 0.0
  axes:
    seed: {range: [0, 16777215], int: true, stratify: false}
    background: {range: [0.6, 1.0]}
dots:
  layer:
    shape: {kind: dot_array, sigma_mm: 0.12, spacing_mm: 0.7, arrangement: hexagonal}
  axes: {orientation_deg: {range: [0, 180], circular: true}}
```

### Indenters

`curved_contact` (a convex `sphere` or `cylinder` of `radius_mm`) and `step_edge` (a plate
ending in a rounded shoulder of `shoulder_radius_mm`) are pressed to the depth
`amplitude × envelope × modulation`, in units of `unit_mm` millimetres, so the contact patch
grows as the press rises and is exactly 0 at depth 0.

```yaml
defaults: {unit_mm: {value: 0.05}}
classes:
  sphere:
    layer: {shape: {kind: curved_contact, form: sphere, radius_mm: 0.8}}
    axes: {amplitude: {range: [0.6, 1.5]}}
  step:
    layer: {shape: {kind: step_edge, shoulder_radius_mm: 0.5}}
    axes:
      amplitude: {range: [0.6, 1.5]}
      orientation_deg: {range: [0, 360], circular: true}
```

### Braille text on several lines

`line_spacing_mm` sets the distance between lines, `/` starts a new line, and `letter_text`
draws letters by weight (`letters`, `weights`, `cells` per line, `lines`).

```yaml
braille_text:
  layer:
    shape: {kind: gaussian, sigma_mm: 0.05}
    pattern: {kind: braille, dot_spacing_mm: 0.1, cell_spacing_mm: 0.3, line_spacing_mm: 0.4}
  axes:
    text: {dist: letter_text, cells: 2, lines: 2, stratify: false}
```

### Sessions with a declared contact fraction

A `sessions:` section declares the length and contact fraction as axes, the mean gap, and
optional types with their own class mixes. A session is episodes from its type until their
contact time reaches the fraction, and the quiet time left is spent as gaps between
episodes (which render exactly 0). See [Sessions](#sessions).

```yaml
sessions:
  duration_ms: {range: [300, 600]}
  contact_fraction: {range: [0.2, 0.5]}
  gap_mean_ms: 40
  types:
    taps:     {weight: 3, classes: {grouped_press: 2, tied_press: 1}}
    textures: {weight: 1, classes: {rough: 1, dots: 1}}
```

A runnable world with one class per element is `tests/fixtures/worlds/elements_v1_2.yml`
(a data set on it: `tests/fixtures/worlds/dataset_v1_2.yml`).

## Sessions

`session(world, duration_ms=10_000, seed=7, index=0)` lays draws end to end, in the order the
world samples them, until the duration is full; the last draw is cut off and the session
records `truncated`. A world that declares `sessions:` draws sessions by its own model
(above). Its `quiet_fraction` is the share of the session with nothing touching (quiet draws,
lead-ins and pauses). A session renders like a draw: `render([s], canvas, times)`. Only the draws that
overlap a time are evaluated there, which keeps a ten-second session cheap.

## Data sets

A data set fixes a world, a seed and how many of what to draw:

```yaml
dataset:
  name: small
  world: tactile_small.yml
  seed: 20261002
  duration_ms: 120
  splits:
    train:      {n: 6, repeats: 2}
    validation: {n: 4, noise_repeats: 2}
    test:       {stratified: {bins: 2, per_bin: 2}}
    probes:     {per_bin: 1}
    held_out:   {stratified: {bins: 2, per_bin: 1}}
    sessions:   {n: 1, duration_ms: 300}
    fixed:      {draws: [braille_H]}
```

| Split | What it holds |
|---|---|
| `train` | `n` draws per repeat; `repeats` fresh draws, each with fresh noise |
| `validation` | like `train`; `noise_repeats` re-simulates the same draws with new noise |
| `test` | per class, `bins × per_bin` draws; each numeric axis cut into equal bins holding exactly `per_bin`, categorical axes balanced |
| `probes` | per class and numeric axis, `per_bin` draws just below and above the range |
| `held_out` | the held-out classes, stratified like `test` |
| `sessions` | `n` sessions of `duration_ms` |
| `fixed` | named draws of the world's `fixed_draws` |

```bash
sensoryforge dataset build dataset.yml --out data/small
```

writes `dataset.json` (the spec, the ids, the probes it had to skip) and `manifest.jsonl`,
one JSON row per entry: `entry` (for example `test/dots/0042`, `validation/r0/00003.n1`,
`probes/dots/sigma_mm-below/007`), `split`, `class`, the full `draw` record, its `bins`, its
`seeds` and the `world_id` and `dataset_id`. Reading it needs only JSON:

```python
from sensoryforge.world import load_manifest
rows = load_manifest("data/small/manifest.jsonl")
```

Building fails if any seed appears twice, so no draw is in two splits.

## Running a data set

```bash
sensoryforge batch --design designs/ra_v1 --dataset dataset.yml --output runs/small
```

simulates every entry through the sensor and writes one data bundle per entry to
`runs/small/<entry id>/` ([Data Bundles](bundles.md), schema 2.2.0). The sensor comes from
`--design`, `--preset` or a config file. Useful options:

- `--splits test,probes` runs only those splits; `--entries 100:200` runs a slice.
- `--tasks 50 --task-index 7` runs the eighth of 50 contiguous tasks, for a job array;
  `--tasks 50 --print-tasks` prints the 50 commands instead of running them.
- `--resume` skips every entry whose bundle is already complete. Bundles are written to a
  temporary folder and moved into place, so an interrupted run leaves no half bundle.

Each entry is simulated with noise seeded from the entry's own noise seed, so a rerun gives
the same spikes. `runs/small/batch.json` records the data set, the config and SensoryForge's
sha; each task appends to `runs/small/index/task_<i>.jsonl` one row per entry (`entry`,
`status`, `error`, `seconds`, ...). `read_batch_index("runs/small")` merges the rows, the
latest run of an entry winning. The command exits non-zero if any entry failed.

## Extending

Everything a world uses is registered, not special-cased. A new shape needs a function of
`(x, y, params)` and its parameter specs, `ParamSpec`s whose `min_val`/`max_val` are the
values a world may draw (the ring example from `tests/unit/test_world_kernel.py`):

```python
import torch
from sensoryforge.stimuli.base import ParamSpec
from sensoryforge.world import register_shape

def ring(x, y, p):
    r = torch.sqrt(x**2 + y**2)
    return torch.exp(-((r - p["radius_mm"]) ** 2) / (2.0 * p["width_mm"] ** 2))

register_shape("ring", ring, [
    ParamSpec("amplitude", dtype="float", default=1.0, min_val=0.0, max_val=1.0e4),
    ParamSpec("radius_mm", dtype="float", default=1.0, min_val=0.0, max_val=10.0,
              unit="mm"),
    ParamSpec("width_mm", dtype="float", default=0.1, min_val=0.001, max_val=10.0,
              unit="mm"),
])
```

Then `shape: {kind: ring}` works in a world and in a layered stimulus. In the same way
`register_pattern` and `register_modulation` add patterns and modulations,
`register_distribution(name, sample, support)` adds a distribution for `{dist: name}` axes
(give `support` when it is finite, so a test split can balance it), and
`register_class_kind(kind)` adds a class kind (a `ClassKind` instance) next to `layered` and
`quiet`. Each refuses a name that is already registered unless you pass `replace=True`.

A world can have several **channels** (`world: {channels: [a, b]}`); each class names the
channel it draws on with `channel:`, and renders have shape `[n, K, C, *S]`. A batch run maps
the world's channels onto the grid's `channels`.
