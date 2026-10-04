# World engine contract

This page is what pressure-simulation's Phase 2b is written against: what SensoryForge
**v1.2.0** guarantees about declared worlds, sampling, rendering, data sets and batch runs,
and the SensoryForge test that pins each guarantee. Where this page and the code disagree,
the tests decide. v1.2.0 adds the world elements of pressure-simulation's charter world
(its §174) and changes nothing for a world, data set or bundle that does not use them
(section 10).

## Getting it

```bash
conda run -n bio-encoding pip install --no-deps --force-reinstall \
  "sensoryforge @ git+file:///Users/benefron/sensoryforge@v1.2.0"
```

The in-process API needs Python ≥ 3.10, torch ≥ 2.2, numpy, PyYAML and h5py; it is tested
on Python 3.10.18 / torch 2.2.2 (`bio-encoding`) and Python 3.11 / torch 2.5.1. A bundle
records the sha of the SensoryForge that wrote it (`config.json["sensoryforge_sha"]`), read
from pip's `direct_url.json` in a pinned install.

## 1. A world

A world file has one key, `world:`. The user guide ([Worlds and data sets](../user_guide/worlds.md))
explains each part; the rules that matter to a caller:

- **Classes** (`classes:`) carry a `weight`; **held-out classes** (`held_out:`) carry none
  and are sampled only when named. A class has a `kind`: `layered` (default) or `quiet`
  (renders exactly 0 for `quiet_ms`); plugins may register more.
- A `layered` class is a layered **layer** (`shape`, `pattern`, `modulation`, optional
  `motion`) whose fields its **axes** draw. An axis is `{value: v}`, `{range: [lo, hi]}`
  (`dist: uniform | log_uniform`, `int: true`, `circular: true`, `probes: false`),
  `{values: [...], weights: [...]}`, or `{dist: <registered>}` (`braille_cells`: the 63
  non-empty cells as dot-number strings, uniform).
- An axis name binds, in order, to an episode field (`delay_ms`, `touch_ms`, `hold_ms`,
  `slide_ms`, `release_ms`, `contacts`, `pause_ms`, `speed_mm_per_ms`, `direction_deg`),
  `amplitude`, `x_mm`/`y_mm`, then a shape, pattern or modulation field by bare name, or a
  dotted path (`shape.width_mm`). Ambiguous or unknown names fail at load time. A layer's
  own keys `background` and `clamp_min` bind as axis names too (v1.2.0, section 10.1).
  An axis may also be `{same_as: <axis>}` (it copies another axis of the same draw) and
  any axis may say `stratify: false` (v1.2.0, section 10.4).
- Precedence: a class's own axes, then the fields its layer fixes, then the world's
  `defaults`, then built-ins (all durations 0, `contacts` 1, `amplitude` 1, `x_mm`/`y_mm` 0).
- `fixed_draws:` names draws by explicit values; unnamed axes take their midpoint
  (geometric for `log_uniform`); values outside a range are allowed and listed in the
  draw's `out_of_range`.
- **Identity:** `world.world_id` is `w-` + 12 hex digits of SHA-256 over the normalised
  world (everything except `description`). Draws depend on class **names**, never on the
  order a file writes its classes, axes or defaults in: classes are weighed in order of
  their names, so re-sorting a file (as `yaml.safe_dump` does) changes neither the id nor
  any draw.
- **Checked at load** (a `ValueError` naming the path, e.g.
  `world.classes.twice.axes.contacts`):
  - `contacts` is a whole number ≥ 1: an int constant, an `int: true` range with lo ≥ 1,
    or a list of ints ≥ 1;
  - every value an axis can take (a constant, both range bounds, each categorical value,
    a finite registered distribution's values) and every fixed-draw value lies inside its
    field's domain: durations, `quiet_ms` and `speed_mm_per_ms` ≥ 0; shape, pattern and
    modulation fields within their `ParamSpec`'s `min_val`/`max_val`, text fields among
    its `choices`. A fixed draw may leave the axis's range (it is then listed in
    `out_of_range`) but not the field's domain;
  - numeric fields take numbers, not text or booleans (PyYAML reads `3e-1` as the text
    `'3e-1'`: write `3.0e-1`); text fields take text, switches `true`/`false`; NaN and
    infinity are refused in ranges, constants, values and weights;
  - a class whose touch, hold, slide and release can only be 0 never touches and fails;
  - a key written twice in the file fails, as do two names for one field in one class's
    axes or in the defaults (`sigma_mm` and `shape.sigma_mm`), and class, held-out class
    and fixed-draw names that cannot name a directory (they must match
    `^[A-Za-z0-9_][A-Za-z0-9_.-]*$`).

## 2. One draw's time course

Time 0 is the start of the entry. A draw is quiet for `delay_ms`, then `contacts` contacts,
`pause_ms` apart; each is a linear ramp up over `touch_ms`, still for `hold_ms`, moving for
`slide_ms`, a linear ramp down over `release_ms`. Motion runs only during slides, at
`speed_mm_per_ms` towards `direction_deg` (0° = +x, 90° = +y), spread over all contacts so a
re-touch lands where the last contact ended. A modulation multiplies the whole contact
envelope, measured from each touch: `sine` (`frequency_hz`, `depth`, `phase_deg`;
1 − depth·(1 − cos(2πft + φ))/2) or `pulses` (`rate_hz`, `duty`, `edge_ms`, `depth`). Before
0, from the draw's `end_ms` on (`t ≥ end_ms`, a step release included), in lead-ins and in
pauses the stimulus is **exactly 0**.

## 3. Sampling

```python
from sensoryforge.world import load_world, sample, session
world = load_world("world.yml")
draws = sample(world, n=1000, seed=7)                    # draw i depends only on (world, 7, i)
same  = sample(world, indices=range(500, 600), seed=7)
some  = sample(world, n=200, seed=7, classes=["dots"])   # held-out classes may be named
demo  = world.fixed_draw("braille_H")
s     = session(world, duration_ms=10_000, seed=7, index=0)
```

`classes` names each class once (an empty list is an error); the order it lists them in
does not matter. A draw's record (`draw.to_dict()`, rebuilt by
`Draw.from_dict(record, world)`):
`world_id, seed, index, draw_seed, class, sampling, values, timeline, end_ms, out_of_range`.
`values` holds every axis, constants included. A session's record:
`world_id, sampling: "session", seed, index, session_seed, duration_ms, items: [[start_ms,
draw], ...], truncated, quiet_fraction, end_ms`; its draws are laid end to end.

## 4. Rendering

```python
from sensoryforge.world import Canvas, render, render_movie, movie_times
canvas = Canvas(xx, yy)                                   # any coordinates, mm
canvas = Canvas.from_grid(40, 40, 0.15)                   # SF's centred layout, dim 0 = x
frames = render(draws, canvas, times_ms, dtype=torch.float64, device="cpu")
movie  = render_movie(draw, canvas, dt_ms=1.0, duration_ms=500.0, dtype=torch.float64)
```

- `times_ms` is `[K]` (shared) or `[n, K]` (per draw), ms from each item's start.
- Output `[n, K, *S]` (`S` = the canvas shape), or `[n, K, C, *S]` for a world with `C > 1`
  channels. `render_movie` uses `t_k = k · dt_ms`.
- **Coordinates:** SensoryForge's canvases are centred with x on dim 0. To render on
  pressure-simulation's corner-origin `[y, x]` frames, pass that mesh: `Canvas(xx_ps, yy_ps)`
  where `xx_ps`, `yy_ps` are your `(H, W)` coordinate arrays — the world's positions are in
  its own mm, so choose the world's `x_mm`/`y_mm` ranges in the same frame as the mesh.
- **Memory.** `render` bounds each internal evaluation `[g, k, *S]` (draws × times × canvas)
  by `max_elements` (default 2**23 elements), so a large call does not allocate a large
  intermediate. A session's draws are each evaluated only at the times inside their own
  window. The output `[n, K, …]` itself is allocated by `render` and sized by what the caller
  passes: how many items and how many times. To render millions of triples, call `render`
  on slices (for example 8000 draws per call). Measured: 4.1 M triples at 40×40 in float64
  take about 4.6–4.7 min on an Apple M3 Pro with 6 threads (`docs/reference/benchmarks.md`,
  "World renderer").

## 5. Data sets

The YAML below sketches the shape of a data-set file;
`tests/fixtures/worlds/dataset_small.yml`, on the world
`tests/fixtures/worlds/tactile_small.yml`, is a runnable example.

```yaml
dataset:
  name: dev_set
  world: world.yml            # relative to this file, or an inline world
  world_id: w-…               # optional pin
  seed: 20261002
  duration_ms: 500
  splits:
    train:      {n: 400, repeats: 3}            # repeats: fresh draws and noise each
    validation: {n: 200, noise_repeats: 2}      # noise_repeats: same draws, new noise
    test:       {stratified: {bins: 5, per_bin: 20}}
    probes:     {per_bin: 20}                    # bins default to test's
    held_out:   {stratified: {bins: 5, per_bin: 20}}
    sessions:   {n: 5, duration_ms: 10000}
    fixed:      {draws: [braille_H]}
```

- **test**, per class: `bins × per_bin` draws; each numeric axis cut into `bins` equal bins
  on its sampling scale, each bin holding exactly `per_bin`; categorical axes (values, `int`,
  `braille_cells`) one bin per value, balanced to ±1; axes shuffled independently.
- **probes**, per class and numeric non-circular axis: `per_bin` draws one bin-width below
  and above the range, inside the field's valid domain; a side with no room is skipped and
  listed in `dataset.json["skipped_probes"]`.
- **Seeds:** split seed `H(seed, split, repeat)`; draw seed per draw; noise seed
  `H(seed, "noise", entry id)`. Building fails if any seed appears twice.
- **Entry ids:** `train/r0/00017`, `validation/r0/00003.n1`, `test/dots/0042`,
  `probes/dots/sigma_mm-below/007`, `held_out/gratings/0003`, `sessions/002`,
  `fixed/braille_H`.
- **Checked at load:** a key written twice fails; a split's keys must belong to its
  kind (train/validation: `n`; test: `stratified`; probes: `per_bin`, `bins`; held_out:
  `stratified` or `n`; sessions: `n`, `duration_ms`; fixed: `draws`; any split: `kind`,
  `repeats`, `noise_repeats`); split names must match `^[A-Za-z0-9_][A-Za-z0-9_.-]*$`.
- `sensoryforge dataset build dataset.yml --out DIR` writes `dataset.json` and
  `manifest.jsonl`. A row: `entry, split, repeat, noise_repeat, class, draw (the record),
  bins, probe, seeds {draw, noise}, duration_ms, truncated, world_id, dataset_id`. Bin labels
  are `"[a, b)"` (last bin `"[a, b]"`, 6 significant digits), a categorical value as itself,
  or `"below"`/`"above"`.

## 6. Batch runs

```bash
sensoryforge batch --design DIR --dataset dataset.yml --output OUT
sensoryforge batch --design DIR --dataset dataset.yml --output OUT --tasks 50 --task-index 7
sensoryforge batch --design DIR --dataset dataset.yml --output OUT --tasks 50 --print-tasks
sensoryforge batch sensor.yml --design DIR --dataset dataset.yml --output OUT --splits test,probes --resume
```

- The sensor is `--design`, `--preset` or a config file (a file given with either is merged
  over it). `--entries a:b` runs a slice; `--print-tasks` prints one command per task (rows
  for `experiments/lsf/make_manifest.py`).
- Each entry: rendered with `render_movie(..., dtype=torch.float64)` on
  `Canvas.from_grid_config(grids[0])` **on the CPU**, whatever `--device` says (CUDA's
  float64 `exp`/`sin`/`cos` differ from the CPU's in the last bits), cast to float32, moved
  to the engine's device and simulated. The entry's 53-bit
  noise seed sets `simulation.receptor_noise_seed`, and every population's `noise_seed` is
  set to `seed53(noise, "population", i)`, whether or not the design set one (the engine
  uses a population's seed only when that population has sensor or membrane noise). Both
  are recorded in the bundle's `config.json` (`config.simulation.receptor_noise_seed`,
  `config.populations[i].noise_seed`). The run seed passed to `SimulationEngine.run` is
  `noise & 0xFFFFFFFF`, the low 32 bits of the noise seed, because the engine seeds numpy's
  legacy global generator, which accepts only 32-bit seeds; with the receptors and every
  population seeded from 53 bits, no receptor, sensor or membrane noise depends on it (with
  32-bit seeds, the chance that two entries share one is about 1% at 10^4 entries and 69%
  at 10^5).
- Output: `OUT/<entry id>/` (a schema-2.2.0 bundle, written atomically), `OUT/batch.json`,
  `OUT/index/task_<i>.jsonl` (`entry, bundle, status, error, seconds, finished_at,
  design_id, sensoryforge_sha, task`); `read_batch_index(OUT)` merges them. The exit status
  is non-zero if any entry failed.
- Until v1.2.0 is merged into `~/sensoryforge`'s `main`, the `sensoryforge` env's CLI is
  older; run the batch from `bio-encoding` with `python -m sensoryforge.cli batch …`.
- An entry's movie is rendered in time chunks (at most 2**25 float64 elements each, cast
  to float32 chunk by chunk), so a long session never holds its whole float64 movie; the
  frames equal the unchunked render cast to float32, bit for bit (section 10.9). The
  simulation engine still holds the whole float32 stimulus: a 120 s session at 80×80 and
  1 ms is about 3.1 GB, and that stays open.

## 7. Bundles (schema 2.2.0)

Additive over 2.1.0: `config.json` gains `sensoryforge_sha` and, for an entry,
`world: {world_id, dataset_id, entry}`; `data.h5` attributes gain `sensoryforge_sha`;
`stimuli/stimulus.json` for an entry is `{schema_version, kind: "sensoryforge_world_entry",
entry (the manifest row), layer (the draw as a layered layer; a session: [[start, layer],
...]), dt_ms, total_ms, n_frames, grid, reconstructible_by_pressure_simulation: false}`.
`/stimulus/frames` holds the clean (noise-free) float32 frames.

## 8. Guarantees and the tests that pin them

| # | Guarantee | Test |
|---|---|---|
| 1 | Same world and seed give identical draws and frames in two processes; draw *i* alone equals draw *i* in a batch (bit for bit on one machine; verified on Apple Silicon and, single-threaded, on Linux x86-64 — see "Across machines") | `tests/contract/test_world_contract.py::test_1_determinism_across_processes_and_batch_sizes`, `tests/unit/test_world_render.py::test_draw_i_alone_equals_draw_i_in_a_batch_bit_for_bit` |
| 2 | A bundle's `/stimulus/frames` equals `render_movie(<the bundle's own record>, Canvas.from_grid_config(grid), dt, duration, dtype=float64).to(float32)` bit for bit, rendered on the CPU (the batch renders there on any `--device`) | `test_2_the_bundle_records_exactly_the_in_process_render` |
| 3 | Frames rendered at `[t−τ, t, t+τ]` equal those steps of the movie (times `k·dt`) | `test_3_windows_agree_with_movies` |
| 4 | Different noise seeds give different responses; the same seed gives identical spikes; the bundle records `simulation.receptor_noise_seed` equal to the entry's 53-bit noise seed | `test_4_noise_seeds` |
| 5 | No draw seed in two splits; every numeric test bin holds `per_bin` per class (categorical ±1); probes labelled and outside the range | `test_5_splits_strata_and_probes` |
| 6 | One draw on 40×40 and 80×80 at 0.15 mm agrees on the shared points to 1e-12 | `test_6_one_draw_on_40x40_and_80x80` |
| 7 | Quiet stretches of a session are exactly 0 | `test_7_session_quiet_stretches_are_exactly_zero` |
| 8 | Every bundle carries the design manifest, the world id, the entry's record and SensoryForge's sha | `test_8_every_bundle_carries_its_provenance` |
| — | A world render equals the `layered` render of `draw.to_layer()` to 1e-5 (layered keeps time in float32) for every built-in shape, pattern, motion and modulation; for hard-edged shapes (`disc` with `edge_mm: 0`, `flat` bar, `square` grating) points within 1e-4 mm of an edge are excluded | `tests/unit/test_world_render.py::test_world_render_equals_layered`, `tests/unit/test_world_render.py::test_every_shape_pattern_and_motion_equals_layered` |
| — | A draw is exactly 0 from its `end_ms` on, a step release included | `tests/unit/test_world_render.py::test_a_step_release_is_exactly_zero_from_end_ms` |
| — | The fixture world's draws are pinned: sha256 of `sample(tactile_small, n=50, seed=7)`'s records, floats rounded to 10 significant digits; a change means every world's draws changed | `tests/unit/test_world_sampling.py::test_the_fixture_worlds_draws_are_pinned` |
| 9 | A world or data-set file that uses no v1.2 key keeps its normalised form, `world_id`, `dataset_id`, draw records, session record, manifest and frames, and a v1.1.0 bundle's recorded stimulus rebuilds and re-renders, bit for bit (frames bit-exact on the machine, system and torch version of the recording, golden tolerance elsewhere) | `test_9_old_worlds_and_bundles_are_unchanged`, `tests/unit/test_world_v1_1_compat.py` |
| 10 | Every v1.2 element (one class each in `tests/fixtures/worlds/elements_v1_2.yml`) renders equal to `layered` to 1e-5, and one draw on 41×41 at 0.15 mm and 81×81 at 0.075 mm (the same 6 mm, nested points) agrees to 1e-12 | `test_10_every_v1_2_element_renders_equal_to_layered_and_on_both_grids`, `tests/unit/test_world_v1_2_render.py` |
| 11 | A session of a world that declares `sessions:` meets its declared contact fraction (up to its last episode's overshoot), and its gaps render exactly 0 | `test_11_model_sessions_meet_their_declared_fraction_and_gaps_are_zero`, `test_7_model_session_quiet_stretches_are_exactly_zero`, `tests/unit/test_world_session_model.py` |
| — | Draw *i* alone equals draw *i* in any batch or chunk, bit for bit, for every v1.2 element (masked loops in `dot_array` and `self_affine` included) | `tests/unit/test_world_v1_2_render.py::test_v1_2_draw_alone_equals_draw_in_a_batch` |
| — | Re-ordering a world's classes, axes or defaults changes neither `world_id`, nor any draw, nor a data set's entries | `tests/unit/test_world_sampling.py::test_class_order_changes_neither_the_id_nor_the_draws`, `tests/unit/test_world_dataset.py::test_class_order_does_not_change_the_entries` |

A test named without a path is in `tests/contract/test_world_contract.py`.

**Across machines:** uniform, integer and categorical values are bit-identical everywhere;
`log_uniform` values and rendered frames may differ in the last bit between platforms (the
platform maths library), so a manifest's stored record is the canonical draw (as for
SensoryForge's golden fixtures, F-071). Added in v1.2.0: `biased_direction` values use `atan2`,
`sin`, `cos` and a root solve, and `self_affine`, `dot_array`, `curved_contact` and `step_edge`
frames use `cos`, `exp` and `sqrt`, so they too may differ in the last bit between platforms
(libm); `letter_text` and every integer or categorical value stay bit-identical. A test that
needs equality across machines compares records, not floats.

Bit-for-bit batch invariance (guarantee 1: draw *i* alone equals draw *i* in a batch, and
chunking changes no bit) is verified on Apple Silicon (arm64), in SensoryForge's env and in
`bio-encoding`, and on Linux x86-64 (GitHub's `ubuntu-latest` runners, Python 3.10 and 3.11,
CPU torch 2.14.1), where every world and contract test passed. The test suite runs torch on
one thread (`OMP_NUM_THREADS=1`), so on x86-64 it is not yet shown with several threads or with
torch 2.2.2: there torch's SIMD and scalar paths for `exp`, `sin` and `cos` could still differ
in the last bit at thread-chunk boundaries, so draw *i* alone and in a batch might differ by
about 1 ulp. Guarantee 2 (bundle equals render) compares with the CPU render, which is what
the batch records on any machine and any `--device`.

## 9. Conventions pressure-simulation must map

- `bar` (an edge) uses `p = x·sinθ + y·cosθ`; `grating`/`gabor` stripes vary along
  `x·cosθ + y·sinθ`; both take degrees.
- Shapes peak at `amplitude` and are non-negative unless `signed: true` (`grating`, `gabor`:
  `cos` instead of `(1 + cos)/2`).
- Braille cells are dot numbers 1–6 (1–3 down the left column, 4–6 down the right), dot
  pitch `dot_spacing_mm`, centred on the cell.
- Batch frames are float32 (the engine's dtype); render in float64 and cast to compare.

## 10. What v1.2.0 adds

Every addition is **invisible until used** (guarantee 9): a key appears in a normalised world,
a draw record, a manifest row or a bundle only when the file uses it, and the format tags
(`sensoryforge-world/1`, `sensoryforge-dataset/1`) and the bundle schema (2.2.0) are unchanged.
The user guide ([Worlds and data sets](../user_guide/worlds.md)) has an example of each. None
of §174's values is written into SensoryForge; the fixtures use placeholders.

### 10.1 The background and the floor (`background`, `clamp_min`)

A layer may carry `background` (a uniform level over the whole patch, domain 0 to 1e4, an axis
name like any other) and `clamp_min` (a floor on the layer's total). The frame is
`envelope × (amplitude × shapes + background)`, so the background is its own draw, rises and
falls with the same contact envelope and modulation, and is **not** moved by the pattern or by
motion. The floor applies only where the envelope is positive; elsewhere the frame is exactly 0.
For an indenter (10.2) the frame is `envelope × background + indentation`. Zero-mean relief
(`self_affine`, a `signed` grating) on a background therefore never goes below `clamp_min`
(use `clamp_min: 0`: the skin leaves contact, there is no negative indentation). Layers without
either key run v1.1.0's expression unchanged.

### 10.2 Indenters: `curved_contact`, `step_edge`

A shape registered with `register_shape(..., indenter=True)` is **depth-driven**: it is
evaluated at the depth `d = amplitude × envelope × modulation` (the end mask included),
passed as `params["depth"]`, and returns the indentation itself; the amplitude is not applied a
second time. The footprint therefore grows during the rise, and depth 0 renders exactly 0.
Both implementations (`layered`, the world renderer) carry it. `unit_mm` is the millimetres of
indentation per unit of amplitude (a world `defaults:` axis for it is skipped in classes
whose shape lacks the field).

- `curved_contact`: `form: sphere | cylinder` (convex only; the concave surfaces are not
  built), `radius_mm` (0.1–1000), `orientation_deg` (the cylinder's axis, `bar` convention),
  `unit_mm`. With `r` the distance from the centre (sphere) or the axis (cylinder) and
  `sag = r² / (R + √(R² − r²))`: value `max(0, d − sag / unit_mm)` for `r ≤ R`, 0 beyond.
- `step_edge`: a flat plate with a rounded shoulder: `shoulder_radius_mm` (0 = a sharp step),
  `orientation_deg` (`bar` convention: `q = x sinθ + y cosθ`, the plate lies where `q ≤ 0`),
  `unit_mm`. Value `d` for `q ≤ 0`, `max(0, d − (ρ − √(ρ² − q²)) / unit_mm)` for `0 < q < ρ`,
  0 for `q ≥ ρ`.

### 10.3 Patch-filling shapes: `self_affine`, `dot_array`

Both are registered with `unbounded=False`, so `x_mm`/`y_mm` translate them.

- `self_affine`: a zero-mean random relief with **Persson's isotropic spectrum** (flat below
  `q0 = 2π / rolloff_mm`, ∝ `q^(−2(H+1))` above, zero above `q1 = 2π / cutoff_mm`),
  synthesised as a sum of `components` cosines (16–4096, default 256; radial wavenumbers by
  a stratified inverse CDF of `q·C(q)`, uniform directions and phases drawn from the shape's
  own `seed`, 0 to 2²⁴ − 1). Fields: `hurst` (0–1), `rolloff_mm`, `cutoff_mm`, `components`,
  `seed`, `amplitude`. **The RMS is `amplitude / √2`** (the RMS of a unit-peak signed
  sinusoid, so a self-affine and a periodic texture of equal amplitude carry equal power).
  It is a continuous function of position (the same on any canvas) and signed: use it with
  `background` and `clamp_min`. `seed` and `components` are exact as float32 integers (bounded
  by 2²⁴ − 1); draw `seed` with `stratify: false`.
- `dot_array`: a lattice of Gaussian bumps (each peaking at `amplitude`, overlaps summed):
  `sigma_mm`, `spacing_mm` (along a row), `row_spacing_mm` (0 = `spacing_mm`, or
  `spacing_mm·√3/2` for hexagonal), `arrangement: square | hexagonal` (hexagonal: every other
  row offset by half a spacing), `orientation_deg`. Terms beyond `ceil(7.5σ / min(a, b))`
  sites are dropped (below 1e-12 of the peak); more than 32 sites each way is a `ValueError`
  naming `sigma_mm` and `spacing_mm`.

Both loops run to the batch's maximum and mask each draw's surplus terms to exact zeros
(section 10.10).

### 10.4 Axis features: `stratify: false`, `same_as`

- `stratify: false` on an axis draws it i.i.d. in stratified splits: no bin, no label, no
  64-value limit on an `int` axis. The key reaches the world id only when used.
- `{same_as: <axis>}` copies another axis of the same draw (a press falls as it rose:
  `release_ms: {same_as: touch_ms}`). Links are validated at load (the target exists, is
  not itself a link and the copied value passes the link field's domain), are refused in
  `fixed_draws`, and are filled after all other axes in declared, stratified, probe and fixed
  sampling.

### 10.5 `biased_direction`

`{dist: biased_direction, travel_ratio: R, axis_deg: a}`: the direction of an anisotropic
Gaussian velocity (the angular central Gaussian), `θ = a + atan2(sin 2πu, s·cos 2πu)`, with the
stretch `s` solved from the declared travel ratio `R = E|cos θ| / E|sin θ|` by the closed form
`R(s) = s·atan(k) / artanh(k/s)`, `k = √(s² − 1)` (R = 2.5 gives s = 3.87625). One uniform per
draw, monotone in `u`; `R = 1` is uniform, `R < 1` stretches the perpendicular axis. Values
are degrees in [0, 360). `axis_deg` is the lateral axis (0° = +x). It is a quantile
distribution: stratified splits cut it into equal-probability bins.

### 10.6 Braille lines and `letter_text`

The `braille` pattern gains `line_spacing_mm` (0.1–1000, default 10, left out of a normalised
layer unless the file sets it) and `/` in `text` and in `dots` starts a new line: line `L` sits
at `y0 − L·spacing` and the cell index restarts on each line. `{dist: letter_text, letters,
weights, cells, lines}` draws `cells` letters on each of `lines` lines by the given weights
(default: a–z, uniform; a–z and space only), joined with `/`. The draw's 53 bits expand to
one sub-uniform per letter. One letter has a finite support (the letter list); several letters
per draw have none, so declare the axis with `stratify: false`.

### 10.7 Reusable axis groups: `groups:` and `use:`

`groups: {name: {axis: spec}}` at the world level and `use: [name]` in a class are sugar: the
class holds the resolved axes, `World.to_dict()` has no `groups` or `use` key, and draws and
ids equal those of the same world written out. Precedence: built-ins < `defaults` < used
groups < the class's own axes. A group axis naming a field the class lacks fails naming
`world.groups.<g>.<axis>`; two used groups setting one field fail; held-out classes may use
groups. A contact type is one class per (feature × contact type) with the group of axes that
type shares (§174's contact types; P1 option A).

### 10.8 The session model: `sessions:`

```yaml
sessions:
  duration_ms: {range: [300, 600]}       # an axis: a session's length, ms (> 0)
  contact_fraction: {range: [0.2, 0.5]}  # an axis: the share in contact, in [0, 1]
  gap_mean_ms: 40                        # the mean quiet gap, ms (> 0)
  types:                                 # optional
    taps:    {weight: 3, classes: {press: 2, tap: 1}}
    textures: {weight: 1, classes: {rough: 1, dots: 1}}
```

A session draws its type by weight, its length `D` and target fraction `f` from their axes,
then episodes (`sample(weights=` the type's classes`)`) until their contact time reaches
`f × D` or their length reaches `D`. The quiet budget `Q = max(0, D − length)` is spent as
`G = min(n + 1, max(1, round(Q / gap_mean_ms)))` **gaps at distinct episode boundaries**, their
lengths the spacings of `Q` cut at sorted uniforms (near-exponential, mean ≈ `gap_mean_ms`;
no exponential distribution is registered). The declared fraction is met in every session up to
the last episode's overshoot; a session whose episodes already fill `D` before reaching `f` is
cut at `D`. The session record gains `session_type` and `contact_fraction` (only for these
worlds); `quiet_fraction` is the realised quiet share. Gaps are holes between items and render
exactly 0. `session(world, duration_ms=None, ...)` draws the length; a `sessions` split
without `duration_ms` takes each session's own. **This supersedes the v1.1.0 spec's decision 5
("no gap mechanism, no quiet-fraction target") for worlds that declare `sessions:`;** all
others keep v1.1.0's sessions bit for bit.

### 10.9 Rendering long entries in chunks

The batch runner renders an entry's movie in float64 time chunks (`CHUNK_ELEMENTS = 2**25`
elements: `[k, C, H, W]`), casting each chunk to float32 straight into the frame buffer. Every
frame depends only on its own time, so the frames equal the whole render cast to float32, bit
for bit (the frozen v1.1.0 session bundle is reproduced). Open: the engine still holds the
whole float32 stimulus.

### 10.10 Batch invariance and the masked loops

Draw *i* alone equals draw *i* in any batch or chunk, bit for bit. A loop whose length depends
on a draw's parameters (`dot_array`'s neighbours, `self_affine`'s components) runs to the
group's maximum, masks each draw's surplus terms to exact zeros and accumulates in a fixed
order from a zero tensor (adding 0.0 leaves a float unchanged).

### 10.11 Provisional decisions

These answers were taken on Ben's behalf by the supervisor on 2026-10-04 and are
**provisional**; changing one changes only the named element. They are recorded in
`docs_root/DECISIONS.md`.

| # | Decision |
|---|---|
| P1 | One class per feature × contact type, with `groups:` / `use:` as shorthand (no class variants) |
| P2 | "Touches for" is the plateau `hold_ms` (no `contact_ms` field) |
| P3 | A press's fall equals its drawn rise through `same_as` |
| P4 | The lateral bias is the angular central Gaussian solved from the travel ratio |
| P5 | Indenters are depth-driven (the footprint grows in the rise); convex sphere and cylinder only |
| P6 | The background is its own draw sharing the contact envelope; the total is floored at 0 |
| P7 | Self-affine textures use the Persson spectrum from 256 cosines, RMS = amplitude/√2 |
| P8 | The declared contact fraction is met in every session; gaps close to exponential (budgeted layout) |
| P9 | Dot arrays on a square or hexagonal lattice of Gaussian bumps |

### 10.12 Differences from the other repo's list (its brief, section 8)

| Brief's row | This release |
|---|---|
| A direction distribution set by the travel ratio and the finger axis | `biased_direction` (10.5) |
| One axis bound to another | `same_as` (10.4); not allowed in `fixed_draws` |
| A background layer under every contact | `background` on a layer, plus `clamp_min`; not a second layer, and it ignores pattern and motion (10.1) |
| An exponential `dist`; a session declared by its in-contact fraction | No exponential distribution: `sessions:` declares the fraction and the gaps are the near-exponential budgeted spacings (10.8) |
| Session types with their own mixes and weights; a duration range | `sessions: types:` and a `duration_ms` axis (10.8) |
| A step edge | `step_edge`, depth-driven, with an orientation (10.2) |
| A curved contact, radius 5–40 mm | `curved_contact`: sphere or cylinder, convex only; the survey's concave 20–40 mm surfaces are not built (10.2) |
| A seeded self-affine surface, zero mean | `self_affine` with its own `seed`, RMS amplitude/√2 (10.3) |
| Braille line spacing | `line_spacing_mm` and `/` (10.6); letters by frequency with `letter_text` |
| Contact types per feature | One class per feature × contact type, `groups:` / `use:` as shorthand (10.7) |
| (not in the brief) | `dot_array` (10.3), `stratify: false` (10.4), chunked rendering (10.9) |
