# The world engine — design

**Date:** 2026-10-02. **Status:** approved in brainstorming with Ben (sections 1–6), awaiting review
of this written form. **Branch:** `worktree-world-engine`. **Delivery:** tag `v1.1.0` on this branch;
Ben merges into `main` later.

**Source of the requirements:** pressure-simulation's brief
`docs_root/plans/briefs/sensoryforge_world_engine.md` (PS commit `3281cad`), requirements R1–R11 and
its eight contract tests. pressure-simulation ("PS") declares its stimulus world in SensoryForge's
("SF") schema and reads it through SF in-process; SF owns the schema and the API.

## 1. Purpose

SensoryForge becomes a place to **design sensory worlds, sample them, and generate data sets on
them**: a world is a declared distribution over stimuli (classes × parameter axes × distributions,
with a time structure); SF samples it deterministically, renders any draw at any time on any
coordinates, splits samples into data sets, and runs them through the encoder in bulk. One
interpreter of the world — SF's — so a stimulus PS designs on and a stimulus SF simulates can never
disagree.

SF does not know PS's design method, decoder or statistics; it stores PS's labels without
interpreting them.

## 2. Decisions taken in brainstorming

1. **A world class is a `layered` layer with random fields** — not a new stimulus vocabulary and not
   arbitrary registered types. A draw resolves to an ordinary layered stimulus (`Draw.to_layer()`)
   that the GUI and `sensoryforge run` can use. One stimulus language in SF.
2. **An episode is touch → hold → slide → release**, after a quiet lead-in, optionally repeated as
   several *contacts* separated by pauses. Motion happens only during slides.
3. **Temporal frequency is a layer *modulation***: `sine` (vibration) and `pulses` (repeated
   indentation), multiplying the contact envelope.
4. **Quiet appears three ways:** the lead-in and tail of every draw; a `quiet` class whose draws are
   zero throughout; pauses between contacts inside an episode.
5. **A session is draws laid end to end**; its quiet comes from lead-ins, tails and `quiet` draws (no
   separate gap mechanism, no quiet-fraction target).
6. **Approach A:** a new `sensoryforge.world` package with its own vectorised renderer, kept equal to
   `layered` by tests; `layered` gains only additive, default-off fields; the old sweep
   `BatchExecutor` is untouched; `batch --dataset` routes to a new runner. (Rejected: B, rewriting
   `layered` on the new kernel — risks ulp changes to every layered render PS's benchmarks use;
   C, extending `BatchExecutor` — inherits its timestamped roots, checkpoint races, `--device` bug
   and exit-0-on-failure.)
7. **Extensible by registries:** class kinds, shapes, patterns, modulations and distributions are
   registered; channels are declared; modality and units are metadata.
8. **Both repeat semantics:** `repeats` (fresh draws and fresh noise per replicate) and
   `noise_repeats` (same draws, *k* noise realisations).
9. **Signed texture:** `gabor` and `grating` shapes gain `signed` (default off), as the named `gabor`
   already has (D-0437899).
10. **Delivery:** version 1.1.0, tag `v1.1.0` on the branch; `main` and the `~/sensoryforge` checkout
    are untouched (PS's editable env and another session run from it).

## 3. The world schema

A world file is YAML with one top-level key, `world:`.

```yaml
world:
  name: tactile_example
  description: free text            # not part of the id
  modality: tactile                 # metadata
  units: {space: mm, time: ms}      # metadata; SF works in mm and ms
  channels: [pressure]              # sensor planes; default [value] (one plane)
  defaults:                         # shared axes; any class overrides any of them
    delay_ms:   {range: [0, 100]}
    touch_ms:   {range: [10, 50]}
    hold_ms:    {range: [100, 800]}
    slide_ms:   {value: 0}
    release_ms: {range: [10, 50]}
    contacts:   {value: 1}
    pause_ms:   {range: [50, 200]}
    speed_mm_per_ms: {range: [0.01, 0.1], dist: log_uniform}
    direction_deg:   {range: [0, 360], circular: true}
    amplitude:  {range: [0.2, 1.0]}
    x_mm: {range: [-2.5, 2.5]}
    y_mm: {range: [-2.5, 2.5]}
  classes:
    dots:
      weight: 0.3
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.15, 0.45]}}
    taps:
      weight: 0.1
      layer:
        shape: {kind: disc, diameter_mm: 1.0}
        modulation: {kind: pulses, duty: 0.5}
      axes: {rate_hz: {range: [2, 50], dist: log_uniform}}
    braille:
      weight: 0.2
      layer:
        shape: {kind: gaussian, sigma_mm: 0.15}
        pattern: {kind: braille, dot_spacing_mm: 0.35}
      axes: {dots: {dist: braille_cells}}
    static_hold:
      weight: 0.2
      layer: {shape: {kind: gaussian}}
      axes: {hold_ms: {range: [500, 2000]}, sigma_mm: {range: [0.3, 1.0]}}
    quiet:
      kind: quiet
      weight: 0.2
      axes: {quiet_ms: {range: [200, 2000]}}
  held_out:
    gratings:
      layer: {shape: {kind: grating}}
      axes: {wavelength_mm: {range: [0.3, 0.9]}}
  fixed_draws:
    braille_H: {class: braille, dots: "125", speed_mm_per_ms: 0.02, slide_ms: 900}
```

### 3.1 Classes

- `kind` (default `layered`) names a registered **class kind**. Built in: `layered` (a `layer:` dict,
  §3.4) and `quiet` (no layer; renders exactly zero; one axis `quiet_ms`, the draw's length).
- `weight` (≥ 0) is used by declared sampling; weights are normalised over `classes`. A class with
  weight 0 is never sampled by declared sampling but is still stratified in a test split.
- `layer` gives the fixed fields; `axes` gives the random ones. A field set in neither takes the
  `layered` default.
- `channel` (default the world's first channel) names the plane the class draws on.
- `held_out` classes have the same form, take no weight, and appear only in `held_out` splits.
- One layer per class in this release; multi-layer classes are a later extension.

### 3.2 Axes

| Form | Meaning |
|---|---|
| `{value: v}` | constant: recorded, never sampled or stratified |
| `{range: [lo, hi]}` | numeric, `dist: uniform` (default) or `log_uniform` (needs `lo > 0`) |
| `{range: [lo, hi], int: true}` | uniform over the integers `lo..hi` inclusive |
| `{values: [...], weights: [...]}` | categorical; weights default to equal |
| `{dist: <name>, ...}` | a registered distribution, e.g. `braille_cells` |

Flags: `circular: true` (angles: no out-of-range probes); `probes: false` (no probes for this axis).

`braille_cells` is uniform over the 63 non-empty six-dot cells, written as dot-number strings
(`"1"` … `"123456"`); this equals "each dot raised with p = 0.5, at least one".

### 3.3 Binding axis names to fields

An axis name resolves, in order, to:

1. an **episode field**: `delay_ms`, `touch_ms`, `hold_ms`, `slide_ms`, `release_ms`, `contacts`,
   `pause_ms`, `speed_mm_per_ms`, `direction_deg`; for `quiet`, `quiet_ms`;
2. `amplitude` (the shape's amplitude) and `x_mm`, `y_mm` (the pattern's placement);
3. a field of the class's shape, pattern or modulation, by bare name;
4. a dotted path (`shape.width_mm`, `pattern.width_mm`, `modulation.depth`).

A bare name that matches fields in more than one part is an error at load time naming the dotted
alternatives. An unknown name is an error. Defaults declared under `defaults` bind the same way, per
class; a default that names a field the class does not have (e.g. `rate_hz` for an unmodulated class)
is ignored for that class.

Built-in episode defaults (when neither the world nor the class declares them): `delay_ms` 0,
`touch_ms` 0, `hold_ms` 0, `slide_ms` 0, `release_ms` 0, `contacts` 1, `pause_ms` 0,
`speed_mm_per_ms` 0, `direction_deg` 0, `amplitude` 1, `x_mm` 0, `y_mm` 0. A non-quiet class whose
contact time (`touch + hold + slide + release`) can be zero for every draw is an error.

### 3.4 The layer and its new fields

A world class's `layer` is a `layered` layer (`sensoryforge/stimuli/layered.py`): `shape`, `pattern`,
`motion`, `timing`, plus the new `modulation`. The world maps an episode onto it:

| Episode | Layer field |
|---|---|
| `delay_ms` | `timing.onset_ms` |
| `touch_ms` | `timing.ramp_up_ms` |
| `hold_ms` | `timing.hold_ms` |
| `slide_ms` | `timing.slide_ms` (new) |
| `release_ms` | `timing.ramp_down_ms` |
| `contacts`, `pause_ms` | `timing.contacts`, `timing.pause_ms` (new) |
| `speed_mm_per_ms`, `direction_deg` | `motion: {kind: linear, start: [0, 0], end: v·(contacts·slide_ms)·(cos d, sin d), span: slide}` |
| `x_mm`, `y_mm` | `pattern.x_mm`, `pattern.y_mm` |
| `amplitude` | `shape.amplitude` |

`direction_deg` is measured from +x towards +y (0° moves along +x, 90° along +y), in SF's `(x, y)` mm
coordinates. A class may declare its own `motion` (`circular`, `path`) in its layer; the slide then follows that
motion with `span: slide`, and `speed_mm_per_ms`/`direction_deg` do not apply to it.

**New `layered` fields** (all additive; with their defaults every existing layered stimulus renders
bit-identically, pinned by the existing tests plus a new golden test):

- `timing.slide_ms` (default 0), `timing.contacts` (default 1), `timing.pause_ms` (default 0).
  One contact cycle is `ramp_up + hold + slide + ramp_down`, followed by `pause` before the next
  contact. The envelope rises over the ramp up, is 1 through hold and slide, and falls over the ramp
  down. `contacts > 1` requires an explicit `hold_ms`.
- `motion.span` gains `slide` (move during the slides only). **Motion progress is spread over all
  contacts:** with span window `[a, b)` inside each cycle, progress is
  `s = (k + clamp((τ − a)/(b − a), 0, 1)) / contacts` for contact `k` at cycle time `τ`, 0 before the
  first contact, 1 after the last. So a re-touch lands where the previous contact ended, for any
  motion kind. With `contacts = 1` this is today's formula.
- `modulation` (default `{kind: none}`), measured from each contact's start (`t_c`, ms):
  - `sine`: `frequency_hz`, `depth` ∈ [0, 1], `phase_deg`;
    `m = 1 − depth·(1 − cos(2π·frequency_hz·t_c/1000 + phase))/2` (1 at `t_c = 0`, phase 0).
  - `pulses`: `rate_hz`, `duty` ∈ (0, 1), `edge_ms` ≥ 0, `depth` ∈ [0, 1]; period
    `P = 1000/rate_hz`, `τ = t_c mod P`; pulse rises linearly over `edge` from 0, holds 1 until
    `duty·P`, falls linearly over `edge`, then 0; `edge` is clamped to `min(edge, duty·P, (1−duty)·P)`;
    `m = 1 − depth·(1 − pulse)`.
  - The modulation multiplies the whole contact envelope, ramps included. Values stay in [0, 1].
- `pattern` `braille` gains `dots` (default empty): cells by dot number separated by spaces
  (`"125 14"`); when non-empty it replaces `text`.
- `shape` `grating` and `gabor` gain `signed` (default false): the sine profile becomes `cos(phase)`
  instead of `(1 + cos(phase))/2`; the square profile becomes ±1.
- A shape, pattern or modulation kind that `layered` does not implement natively is looked up in
  the world kernel's registries (§5.4) and rendered with `n = 1`.

Every new field gets a `ParamSpec` (`MODULATIONS` joins `SHAPES`/`PATTERNS`/`MOTIONS`), so the
generated forms can show it.

### 3.5 Fixed draws

`fixed_draws: {name: {class: <class or held-out class>, <axis>: value, ...}}`. Unnamed axes take the
class's midpoint for numeric axes (geometric midpoint for `log_uniform`), the first value for
categorical ones, the constant for `{value:}`. A fixed draw may set values outside the declared
ranges; its record lists those axes in `out_of_range`.

### 3.6 Identity and validation

The world is normalised (defaults filled, per-class axes resolved, keys sorted) and hashed:
`world_id = "w-" + sha256(canonical JSON incl. the format tag "sensoryforge-world/1")[:12]`.
`description` is excluded. Loading fails with a `ValueError` naming the offending path for: unknown
kinds, unknown or ambiguous axis names, `lo > hi`, `log_uniform` with `lo ≤ 0`, negative weights or a
zero weight sum, a held-out name equal to a class name, a fixed draw naming an unknown class or axis,
a channel the world does not declare.

## 4. Sampling

### 4.1 Counter-based randomness

`H(a, b)` is a 64-bit integer mix (splitmix64 of `a` xor a splitmix64 of `b`), vectorised in numpy
over `uint64` arrays. Strings are mapped to integers by the first 8 bytes of their SHA-256.

- `draw_seed(i) = H(seed, i) mod 2⁵³` (JSON-safe in any language).
- Each random choice in draw *i* uses `u = (H(draw_seed, slot) >> 11) · 2⁻⁵³` ∈ [0, 1), with
  `slot = "class"` for the class choice and the axis name for an axis.

So draw *i* depends only on `(world, seed, i)`, never on `n` or on which other draws are requested,
and adding an axis to a class does not change its other axes' values.

### 4.2 From u to values

- class: by normalised weight (cumulative sum, first bin with `u < cum`).
- `uniform`: `lo + u·(hi − lo)`; `log_uniform`: `exp(log lo + u·(log hi − log lo))`.
- `int`: `lo + floor(u·(hi − lo + 1))`.
- categorical: by weight, as for the class.
- registered distributions: their own rule from `u` (and further slots if they need them).

### 4.3 API

```python
from sensoryforge.world import load_world, sample, session

world = load_world("world.yml")                        # or World.from_dict({...})
draws = sample(world, n=1000, seed=7)                  # list[Draw]
same  = sample(world, indices=range(500, 600), seed=7) # draw i is identical either way
dots  = sample(world, n=200, seed=7, classes=["dots"]) # weights renormalised over these
demo  = world.fixed_draw("braille_H")
layer = demo.to_layer()                                # an ordinary layered layer dict
s     = session(world, duration_ms=10_000, seed=7, index=0)
```

### 4.4 The draw record

`Draw` is a frozen dataclass with `to_dict()`/`from_dict()`; its dict is JSON-serialisable:

```json
{"world_id": "w-3fa2c81e09bd", "seed": 7, "index": 17, "draw_seed": 4471093307211,
 "class": "dots", "sampling": "declared",
 "values": {"sigma_mm": 0.31, "x_mm": -1.2, "y_mm": 0.4, "amplitude": 0.8,
            "delay_ms": 37.2, "touch_ms": 23.8, "hold_ms": 412.5, "slide_ms": 0.0,
            "release_ms": 18.1, "contacts": 1, "pause_ms": 0.0,
            "speed_mm_per_ms": 0.031, "direction_deg": 211.0},
 "timeline": [["quiet", 0.0, 37.2], ["touch", 37.2, 61.0], ["hold", 61.0, 473.5],
              ["release", 473.5, 491.6]],
 "end_ms": 491.6, "out_of_range": []}
```

- `values` holds every axis the class binds, constants included: the record alone re-renders the
  draw (with the world, for the class's fixed fields).
- `timeline` is derived from `values` (phases of zero length omitted); `end_ms` is the end of the last
  release (for `quiet`, `quiet_ms`).
- Time 0 is the start of the entry. Before 0 and after `end_ms` the draw renders exactly zero.
- `sampling` is `declared`, `stratified`, `probe`, `fixed` or `session`.
- Fixed draws have `seed`, `index`, `draw_seed` null.

### 4.5 Sessions

`session(world, duration_ms, seed, index)` lays draws end to end: session draw *k* uses
`sample(world, indices=[k], seed=H(seed, index))`, starts at the previous draw's `end_ms` (the first
at 0), and the last draw is cut at `duration_ms` and marked `truncated`. A `Session` record holds
`{"world_id", "seed", "index", "duration_ms", "items": [[start_ms, draw], ...], "quiet_fraction"}`,
where `quiet_fraction` is the realised share of time with no contact. `render` and `render_movie`
accept a session anywhere they accept a draw.

### 4.6 Determinism guarantees

- **Same machine:** the same `(world, seed, i)` gives the same record bit for bit in any process.
- **Across machines:** uniform, integer and categorical values are bit-identical; `log_uniform` and
  other values computed through `exp`/`log` may differ in the last bit (platform maths library). The
  record stored in a manifest is the canonical draw — the stance F-071 takes for golden fixtures.

## 5. Rendering

### 5.1 API

```python
from sensoryforge.world import Canvas, render, render_movie

canvas = Canvas.from_grid(rows=40, cols=40, spacing_mm=0.15, center_mm=(0.0, 0.0))
canvas = Canvas.from_grid_config(grid_cfg)    # same extent and layout as stimulus_canvas(grid_cfg)
canvas = Canvas.from_points(xy_mm)            # [M, 2]
canvas = Canvas(xx, yy)                       # any coordinate arrays of one shape S

frames = render(draws, canvas, times_ms, dtype=torch.float64, device="cpu")
# times_ms: [K] shared, or [n, K] per draw -> [n, K, *S], or [n, K, C, *S] if the world has C > 1
movie = render_movie(draw, canvas, dt_ms=1.0, duration_ms=500.0, dtype=..., device=...)
# [T, *S] (or [T, C, *S]), T = round(duration_ms / dt_ms), t_k = k * dt_ms in float64
```

- Canvas coordinates are float64 in mm, `(x, y)`; `from_grid` uses SF's centred layout with
  `indexing="ij"` (dim 0 is x), the formula of `create_grid_torch`, evaluated in float64.
- `render` accepts draws, sessions, or `Draw`/`Session` dicts.

### 5.2 What is computed

For a draw of a `layered` class, at point `x` and time `t`:

> value(x, t) = amplitude · envelope(t) · modulation(t) · Σₚ scaleₚ · shape(x − posₚ − offset(t))

with the envelope, motion progress and modulation of §3.4, and the pattern's positions `posₚ` and
scales. Unbounded shapes (`grating`) are drawn once, not per position, as in `layered`. A `quiet` draw
is zero. A multi-channel world writes each class into its channel's plane; other planes are zero.

### 5.3 Vectorisation

Draws are grouped by (class kind, shape kind, pattern kind, modulation kind). In a group every
parameter is a tensor; pattern positions are computed once per distinct pattern (minus placement,
which is a translation) and padded to the group's largest element count with a zero scale; times are
`[g, K]`. The group is one broadcast computation `[g, K, P, *S]`, summed over `P` slot by slot in a
fixed order, chunked over draws to a memory budget. No Python loop over draws or frames (patterns
whose positions depend on a per-draw seed, i.e. `random`, compute positions per draw).

### 5.4 Registries (extension points)

- `register_shape(name, fn, specs, unbounded=False)`: `fn(x, y, params) -> tensor`, broadcasting over
  leading dims, `params` a dict of tensors.
- `register_pattern(name, fn, specs)`: `fn(params) -> (positions [P, 2], scales [P])` at placement 0.
- `register_modulation(name, fn, specs)`: `fn(t_c, params) -> tensor in [0, 1]`.
- `register_distribution(name, fn, support=None)`: `fn(u, spec) -> values`; `support` lists the
  values of a finite distribution (used to stratify it).
- `register_class_kind(name, cls)`: a class kind binds axis names and renders a group
  (`render_group(draws, canvas, times, dtype, device) -> [g, K, *S]`). A kind that cannot vectorise may
  loop over its draws; it must still satisfy §5.5.

Plugins register on import, through SF's existing plugin mechanism (`plugins:` / the
`sensoryforge.components` entry-point group).

### 5.5 Guarantees (each pinned by a test)

1. **Batch invariance:** draw *i* rendered alone equals draw *i* rendered among *n*, and chunking does
   not change bits (elementwise kernel; fixed-order sums). Verified on this Mac: torch's `exp`, `sin`
   and `cos` give identical bits alone, in odd chunks and in million-element tensors (float32 and
   float64, 6 threads).
2. **Windows equal movies:** `render(draw, canvas, [t−τ, t, t+τ])` equals those frames of
   `render_movie` when the times are `k·dt`.
3. **One definition:** the batch runner records `render_movie(..., dtype=float64).to(float32)` on the
   design's canvas; PS's in-process render, cast the same way, equals `data.h5:/stimulus/frames` bit
   for bit.
4. **Equal to `layered`:** `render` of a draw equals `render_layers([draw.to_layer()], ...)` within
   float32 tolerance (abs 1e-5 on peak-1 stimuli; corrected from 1e-6 while planning:
   `layered` computes time in float32, whose step near 100 ms is ~1e-5 ms, and a 2 ms pulse
   edge turns that into ~5e-6 of value) for every built-in shape, pattern, motion and
   modulation. For hard-edged shapes (`disc` with `edge_mm: 0`, `flat` bar, `square` grating) the
   comparison excludes points within 1e-4 mm of an edge, where a last-bit difference in position
   flips a pixel between 0 and 1.
5. **Grid independence:** a draw rendered on 40×40 and on 80×80 grids at 0.15 mm (both centred)
   agrees on the shared points to 1e-12 (float64); not bit-equal, since the two grids compute their
   coordinates separately.
6. **Quiet is zero:** exactly 0.0 outside contacts, in lead-ins, tails, pauses, and `quiet` draws.

### 5.6 Device, dtype, speed

float32 or float64; CPU or CUDA. float64 on MPS raises a `ValueError` (MPS has no float64). The code
runs on Python ≥ 3.10 and torch ≥ 2.2 (PS's `bio-encoding`: Python 3.10.18, torch 2.2.2).

Speed target: the 4.1 M triples of PS's RA design at 40×40 (≈ 2·10¹⁰ pixel evaluations) render in
float64 on this laptop's CPU in at most 10 minutes, in chunks. `benchmarks/world_render.py` measures
it. If the generic path misses, a separable path for Gaussian-family shapes on regular canvases
(H + W exponentials per frame instead of H·W) is added; the path chosen is fixed per (shape, canvas
kind), so guarantees 1–2 still hold.

## 6. Data sets

### 6.1 The spec

A data-set file is YAML with one top-level key, `dataset:`.

```yaml
dataset:
  name: dev_set
  world: worlds/example_world.yml   # path relative to this file, or an inline world mapping
  world_id: w-3fa2c81e09bd          # optional pin: building fails if the world's id differs
  seed: 20261002
  duration_ms: 500                  # every entry's length (sessions set their own)
  splits:                           # any subset, in this order
    train:      {n: 400, repeats: 3}
    validation: {n: 200, noise_repeats: 2}
    test:       {stratified: {bins: 5, per_bin: 20}}
    probes:     {per_bin: 20}
    held_out:   {stratified: {bins: 5, per_bin: 20}}
    sessions:   {n: 5, duration_ms: 10000}
    fixed:      {draws: [braille_H]}
```

Every split accepts `repeats` (default 1: independent replicates, each with fresh draws and fresh
noise) and `noise_repeats` (default 1: each draw simulated *k* times with different noise seeds).

### 6.2 Splits

- **train, validation:** `n` draws by declared sampling (§4).
- **test:** per class, a Latin hypercube of `bins × per_bin` draws:
  - every non-constant axis of the class is stratified;
  - a numeric axis is cut into `bins` equal bins on its sampling scale (log bins for `log_uniform`);
    each bin gets exactly `per_bin` draws; a value in bin *b* is the inverse CDF of `(b + u)/bins`;
  - a categorical axis — values, a finite registered distribution (`braille_cells`), or an `int`
    axis — gets one bin per value, balanced to ±1 over the class's `bins × per_bin` draws (an `int`
    axis with more than 64 values is an error in a stratified split);
  - each axis's sequence of bins is shuffled independently by a seeded permutation, so the other
    axes vary jointly;
  - classes are stratified regardless of their weight (a weight-0 class still gets its draws).
- **probes:** per class, per numeric axis that is not `circular` and not `probes: false`: `per_bin`
  draws in the bin just below the range and `per_bin` just above (one bin width on the sampling
  scale), the other axes by declared sampling. Probe values stay within the field's valid domain
  (the `ParamSpec` `min_val`/`max_val`; episode durations ≥ 0); a side with no room left is skipped
  and listed in `dataset.json`.
- **held_out:** the world's held-out classes, stratified like test (requires `stratified`), or `n`
  declared draws over the held-out classes with equal weight.
- **sessions:** `n` sessions of `duration_ms` (§4.5).
- **fixed:** the named fixed draws, one entry each.

### 6.3 Seeds

- Split sampling seed: `H(seed, split, repeat)`. Draw seeds come from it as in §4.
- Stratified splits: the per-class, per-axis permutations come from `H(split seed, class, axis)`; each
  draw's within-bin `u` from its own `draw_seed`.
- Noise seed per entry: `H(seed, "noise", entry id)` mod 2⁵³.
- **Building fails if any draw seed or noise seed appears twice anywhere in the data set.**

### 6.4 Entries and ids

Entry ids are paths, deterministic, and fix the bundle directory:

| Split | Entry id |
|---|---|
| train, validation | `train/r0/00017` |
| test, held_out | `test/dots/0042`, `held_out/gratings/0042` |
| probes | `probes/dots/sigma_mm-below/007` |
| sessions | `sessions/003` |
| fixed | `fixed/braille_H` |

With `repeats > 1` the `r<k>` segment counts replicates (stratified and probe splits insert
`r<k>/` after the split name only when `repeats > 1`); with `noise_repeats > 1` the id gains `.n<k>`.
Entries are ordered by split (spec order), then by id; task slicing uses this order.

### 6.5 Output

`sensoryforge dataset build dataset.yml --out DIR` (no simulation) writes:

- `dataset.json`: the spec as given, the normalised world, `world_id`, `dataset_id`
  (`"d-" + sha256(normalised spec + world_id)[:12]`), SF's sha, counts per split and class, skipped
  probe sides.
- `manifest.jsonl`: one row per entry:

```json
{"entry": "test/dots/0042", "split": "test", "repeat": 0, "noise_repeat": 0, "class": "dots",
 "draw": {"...": "the full draw record"},
 "bins": {"sigma_mm": "[0.21, 0.27)", "x_mm": "[-1.5, -0.5)", "hold_ms": "[260, 420)"},
 "probe": null,
 "seeds": {"draw": 4471093307211, "noise": 8812300551027},
 "duration_ms": 500, "truncated": false,
 "world_id": "w-3fa2c81e09bd", "dataset_id": "d-91c04e2b7a10"}
```

Bin labels: `"[a, b)"`, the last bin `"[a, b]"`, a categorical value as itself, `"below"`/`"above"`
for probes (`probe: {"axis": "sigma_mm", "side": "below"}`). Bin edges are printed with 6 significant
digits. Declared-sampling entries have `bins: {}`. `truncated` marks a draw whose `end_ms` exceeds
`duration_ms`.

Python: `build_dataset(load_dataset(path)) -> list[Entry]` (the same rows); `Entry.to_dict()`.

## 7. Batch runs

### 7.1 Command

```bash
sensoryforge dataset build dataset.yml --out DS
sensoryforge batch --design DIR --dataset dataset.yml --output OUT
sensoryforge batch --design DIR --dataset dataset.yml --output OUT --tasks 50 --task-index 7
sensoryforge batch --design DIR --dataset dataset.yml --output OUT --tasks 50 --print-tasks
sensoryforge batch --preset tactile_sa1_ra1 --dataset dataset.yml --output OUT --splits test,probes
sensoryforge batch config.yml --dataset dataset.yml --output OUT --entries 0:100 --resume
```

- `batch` with `--dataset` runs the data-set runner; without it, the existing `BatchExecutor` runs
  exactly as today.
- The sensor comes from `--design DIR`, `--preset NAME` or a positional config file, resolved by the
  same code as `run`; `--device` sets `simulation.device` (for this path).
- `--splits` selects splits; `--tasks K --task-index i` runs the *i*-th of K contiguous slices of the
  (selected) entries; `--entries a:b` an explicit range; `--print-tasks` prints one shell command per
  task (the rows PS's `experiments/lsf/make_manifest.py` consumes); `--resume` skips entries whose
  bundle exists.
- `sensoryforge world validate world.yml` prints the id, classes and axes; `sensoryforge world sample
  world.yml -n 10 --seed 7` prints draw records as JSON lines.

### 7.2 Per process and per entry

The process builds the `SimulationEngine` once (RF banks loaded once), then for each entry:

1. renders `render_movie(draw or session, Canvas.from_grid_config(grids[0]), dt_ms, duration_ms,
   dtype=float64)` on the engine's device, casts to float32. A single-channel world drives a
   single-channel grid whatever the names; with more than one channel, the world's channel names
   must all be among the grid's `channels` (else a `ValueError` before any entry runs), and the
   frames are `[T, C, H, W]` in the grid's channel order, unnamed planes zero;
2. sets the entry's noise seed as `simulation.receptor_noise_seed` and the run seed, and replaces each
   population's `noise_seed`, when set, by `H(noise seed, population index)`;
3. runs `SimulationEngine.run` with the design manifest and the bundle written to
   `OUT/.partial/<entry>/`, then renames it to `OUT/<entry>/` (a bundle exists complete or not at all);
4. appends a row to `OUT/index/task_<i>.jsonl` (the single-process run is task 0 of 1):
   `{entry, bundle, status: ok|failed, error, seconds, design_id, sensoryforge_sha}`.

A failed entry is recorded and the run continues; the exit status is non-zero if any entry failed.
`OUT/batch.json` holds the data-set id, world id, design manifest, sensor config and SF's sha.
`read_batch_index(OUT)` merges the task files (one row per entry, latest wins).

### 7.3 What each bundle carries

- the design manifest, as `run --design` stamps it today;
- `stimuli/stimulus.json`: `{"schema_version": "2.2.0", "kind": "sensoryforge_world_entry",
  "entry": <the manifest row>, "layer": <draw.to_layer()> (a session: a list of `[start_ms, layer]`),
  "reconstructible_by_pressure_simulation": false}`;
- `config.json` gains `sensoryforge_sha` and `world: {world_id, dataset_id, entry}`;
- `data.h5` root attributes gain `sensoryforge_sha`.

Bundle schema 2.1.0 → **2.2.0** (additive: new optional fields, no field changed). Readers that accept
2.x keep working; PS's bundle reader does not check the version. Every bundle (not only world entries)
gains `sensoryforge_sha`.

### 7.4 SF's sha

`sensoryforge.provenance.source_info() -> {"sha", "dirty", "source"}`: `git rev-parse HEAD` and
`git status --porcelain` in the package's checkout when it is one (`source: "git"`); otherwise PEP 610
`direct_url.json`'s `vcs_info.commit_id` (`source: "direct_url"`; PS's pinned install has it, e.g.
`2dc83965…` for its current pin); otherwise `{"sha": "unknown"}`.

### 7.5 Known cost

Each bundle still carries every population's RF bank (~2 MB per population at 40×40), because PS's
reader loads banks from the bundle; a thousand entries take a few GB. Left as is.

## 8. Code layout

New package `sensoryforge/world/`, one job per module:

| Module | Job |
|---|---|
| `schema.py` | `World`, class and axis specs; load, validate, bind, normalise, `world_id` |
| `rng.py` | splitmix64 hashing, `u` streams, string slots |
| `distributions.py` | uniform, log_uniform, int, categorical, `braille_cells`; inverse CDFs; registry |
| `sampling.py` | `Draw`, `sample`, `session`, fixed draws, timelines |
| `kernel.py` | vectorised shapes, patterns, envelope, motion, modulation; their registries |
| `kinds.py` | class kinds (`layered`, `quiet`): binding, timelines, `to_layer`, group rendering |
| `render.py` | `Canvas`, `render`, `render_movie`, grouping and chunking |
| `dataset.py` | `DatasetSpec`, `Entry`, splits, strata, probes, ids, manifest |
| `runner.py` | the batch runner, index, `--print-tasks` |
| `__init__.py` | the public API |

Plus `sensoryforge/provenance.py`. Changes to existing modules: `stimuli/layered.py` (§3.4),
`io/bundle.py` (§7.3), `cli.py` (`dataset`, `world`, `batch --dataset`), `__init__.py` (version
1.1.0), `mkdocs.yml` (nav; `development/specs/` and `development/plans/` excluded).

## 9. Testing

- Unit tests per module (`tests/unit/test_world_*.py`).
- `tests/contract/test_world_contract.py` — PS's eight tests, inside SF:
  1. determinism across two processes (subprocess) and draw *i* alone vs in a batch;
  2. bundle `/stimulus/frames` equals the in-process render, bit for bit;
  3. windows equal movie frames;
  4. different noise seeds differ; the same seed gives identical spikes;
  5. no seed in two splits; every test bin holds `per_bin` (categorical ±1); probes labelled and out
     of range;
  6. 40×40 vs 80×80 agree on shared points;
  7. quiet stretches exactly zero;
  8. every bundle carries the design manifest, world id, entry record and SF's sha.
- World render vs `layered` for every built-in shape, pattern, motion and modulation (§5.5.4).
- A golden test that existing layered stimuli render bit-identically after the `layered` changes.
- The world and contract tests also run under PS's stack: `bio-encoding` (Python 3.10, torch 2.2.2)
  with `PYTHONPATH` set to this worktree — nothing installed (F-053).
- The full existing suite passes.
- `benchmarks/world_render.py` against §5.6's target.

## 10. Documentation

- `docs/user_guide/worlds.md`: designing a world, sampling, rendering, data sets, batch.
- `docs/reference/world_contract.md`: the contract PS's Phase 2b plan is written against — the schema,
  the sampling and rendering API, the data-set spec, the batch command, and each guarantee with the
  test that pins it; names the tag.
- `docs/user_guide/designing_stimuli.md`: the new `layered` fields.
- CLAUDE.md: a "Worlds and data sets" section; `DECISIONS.md` and ledger trailers for the decisions
  in §2.

## 11. Out of scope

A GUI world screen (a draw opens in the Stimulus screen as a layered stimulus); the replay class (R11),
which `register_class_kind` leaves room for; multi-layer classes; moving `layered` onto the new kernel
(approach B); batching several entries through one engine call (`write_bundle` requires B = 1).
