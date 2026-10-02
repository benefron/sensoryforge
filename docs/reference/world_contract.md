# World engine contract

This page is what pressure-simulation's Phase 2b is written against: what SensoryForge
**v1.1.0** guarantees about declared worlds, sampling, rendering, data sets and batch runs,
and the SensoryForge test that pins each guarantee. Where this page and the code disagree,
the tests decide.

## Getting it

```bash
conda run -n bio-encoding pip install --no-deps --force-reinstall \
  "sensoryforge @ git+file:///Users/benefron/sensoryforge@v1.1.0"
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
  dotted path (`shape.width_mm`). Ambiguous or unknown names fail at load time.
- Precedence: a class's own axes, then the fields its layer fixes, then the world's
  `defaults`, then built-ins (all durations 0, `contacts` 1, `amplitude` 1, `x_mm`/`y_mm` 0).
- `fixed_draws:` names draws by explicit values; unnamed axes take their midpoint
  (geometric for `log_uniform`); values outside a range are allowed and listed in the
  draw's `out_of_range`.
- **Identity:** `world.world_id` is `w-` + 12 hex digits of SHA-256 over the normalised
  world (everything except `description`).

## 2. One draw's time course

Time 0 is the start of the entry. A draw is quiet for `delay_ms`, then `contacts` contacts,
`pause_ms` apart; each is a linear ramp up over `touch_ms`, still for `hold_ms`, moving for
`slide_ms`, a linear ramp down over `release_ms`. Motion runs only during slides, at
`speed_mm_per_ms` towards `direction_deg` (0° = +x, 90° = +y), spread over all contacts so a
re-touch lands where the last contact ended. A modulation multiplies the whole contact
envelope, measured from each touch: `sine` (`frequency_hz`, `depth`, `phase_deg`;
1 − depth·(1 − cos(2πft + φ))/2) or `pulses` (`rate_hz`, `duty`, `edge_ms`, `depth`). Before
0, after the draw's `end_ms`, in lead-ins and in pauses the stimulus is **exactly 0**.

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

A draw's record (`draw.to_dict()`, rebuilt by `Draw.from_dict(record, world)`):
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
  `Canvas.from_grid_config(grids[0])`, cast to float32 and simulated. The entry's 53-bit
  noise seed sets `simulation.receptor_noise_seed`; each population's `noise_seed`, if the
  design sets one, becomes `seed53(noise, "population", i)`. Both are recorded in the
  bundle's `config.json` (`config.simulation.receptor_noise_seed`,
  `config.populations[i].noise_seed`). The run seed passed to `SimulationEngine.run` is
  `noise & 0xFFFFFFFF`, the low 32 bits of the noise seed, because the engine seeds numpy's
  legacy global generator, which accepts only 32-bit seeds.
- Output: `OUT/<entry id>/` (a schema-2.2.0 bundle, written atomically), `OUT/batch.json`,
  `OUT/index/task_<i>.jsonl` (`entry, bundle, status, error, seconds, finished_at,
  design_id, sensoryforge_sha, task`); `read_batch_index(OUT)` merges them. The exit status
  is non-zero if any entry failed.
- Until v1.1.0 is merged into `~/sensoryforge`'s `main`, the `sensoryforge` env's CLI is
  older; run the batch from `bio-encoding` with `python -m sensoryforge.cli batch …`.

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
| 1 | Same world and seed give identical draws and frames in two processes; draw *i* alone equals draw *i* in a batch (bit for bit, same machine) | `tests/contract/test_world_contract.py::test_1_determinism_across_processes_and_batch_sizes`, `tests/unit/test_world_render.py::test_draw_i_alone_equals_draw_i_in_a_batch_bit_for_bit` |
| 2 | A bundle's `/stimulus/frames` equals `render_movie(<the bundle's own record>, Canvas.from_grid_config(grid), dt, duration, dtype=float64).to(float32)` bit for bit | `test_2_the_bundle_records_exactly_the_in_process_render` |
| 3 | Frames rendered at `[t−τ, t, t+τ]` equal those steps of the movie (times `k·dt`) | `test_3_windows_agree_with_movies` |
| 4 | Different noise seeds give different responses; the same seed gives identical spikes; the bundle records `simulation.receptor_noise_seed` equal to the entry's 53-bit noise seed | `test_4_noise_seeds` |
| 5 | No draw seed in two splits; every numeric test bin holds `per_bin` per class (categorical ±1); probes labelled and outside the range | `test_5_splits_strata_and_probes` |
| 6 | One draw on 40×40 and 80×80 at 0.15 mm agrees on the shared points to 1e-12 | `test_6_one_draw_on_40x40_and_80x80` |
| 7 | Quiet stretches of a session are exactly 0 | `test_7_session_quiet_stretches_are_exactly_zero` |
| 8 | Every bundle carries the design manifest, the world id, the entry's record and SensoryForge's sha | `test_8_every_bundle_carries_its_provenance` |
| — | A world render equals the `layered` render of `draw.to_layer()` to 1e-5 (layered keeps time in float32) | `tests/unit/test_world_render.py::test_world_render_equals_layered` |

A test named without a path is in `tests/contract/test_world_contract.py`.

**Across machines:** uniform, integer and categorical values are bit-identical everywhere;
`log_uniform` values and rendered frames may differ in the last bit between platforms (the
platform maths library), so a manifest's stored record is the canonical draw (as for
SensoryForge's golden fixtures, F-071).

## 9. Conventions pressure-simulation must map

- `bar` (an edge) uses `p = x·sinθ + y·cosθ`; `grating`/`gabor` stripes vary along
  `x·cosθ + y·sinθ`; both take degrees.
- Shapes peak at `amplitude` and are non-negative unless `signed: true` (`grating`, `gabor`:
  `cos` instead of `(1 + cos)/2`).
- Braille cells are dot numbers 1–6 (1–3 down the left column, 4–6 down the right), dot
  pitch `dot_spacing_mm`, centred on the cell.
- Batch frames are float32 (the engine's dtype); render in float64 and cast to compare.
