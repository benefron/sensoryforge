# World §174 addendum — implementation plan (v1.2.0)

> **For agentic workers:** use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to carry out this plan task by task. Each task is sized for one fresh
> implementer who has read this page's top sections (gap table, provisional decisions, global
> constraints, file map) and nothing else. Steps use checkbox (`- [ ]`) syntax.

**Date:** 2026-10-04. **Branch:** `feat/world-174-elements` (cut from `main` at `9c935de`, which is
tag `v1.1.0` plus lint and docs commits). **Worktree:** `/Users/benefron/sensoryforge/.claude/worktrees/world-174`.
**Delivery:** tag `v1.2.0`, created locally on this branch. Nothing is pushed.

**Goal.** pressure-simulation ("PS") declared its charter world on 2026-10-03 (PS decision record §174,
PS ledger D-7a8f9d8; the values come from PS's `docs_root/reports/finalization/world_survey.md`,
sections 2–11). PS writes that world as `worlds/charter_v1.yml` in SensoryForge's schema in its
Phase 2b. This plan builds the world elements §174 needs that v1.1.0 does not have: a background
contact under every contact, lateral-biased scan directions, patch-filling self-affine textures,
dot arrays, a step edge and a curved contact surface, multi-line braille text drawn by letter
frequency, a session model with a declared contact fraction and session types, and the small axis
features these need. Every addition is a declared, general element with parameters, in the
engine's existing style. None of the values in §174 is written into SensoryForge.

**Architecture.** No new subsystem. New shapes are registered in the world kernel (`register_shape`),
so `layered` uses them through its existing registry fallback. New distributions are registered
(`register_distribution`). Layer-level additions (`background`, `clamp_min`) and the indenter
composition are added to both implementations of the one stimulus language (`layered` and the world
renderer), which tests keep equal. The session model is an optional world section; worlds without it
keep v1.1.0's sessions. Old world files, old data sets and v1.1.0 bundles stay bit-equal (Task 1 pins
this before any code changes).

**Read with:** `docs/reference/world_contract.md` (v1.1.0 contract), `docs/user_guide/worlds.md`,
`docs/development/specs/2026-10-02-world-engine-design.md`, `.claude/rules/world-engine.md`.

---

## 1. Gap table: §174 against the v1.1.0 schema

Status: **today** = expressible in v1.1.0; **caveat** = expressible, with the stated limitation;
**missing** = needs this plan. "Task" names the task that builds or touches it.

| # | §174 element | §174 value | Status | How, in the schema's names (or what is missing) | Task |
|---|---|---|---|---|---|
| 1 | Scan speed | log-uniform 10–200 mm/s | today | `speed_mm_per_ms: {range: [0.01, 0.2], dist: log_uniform}` (a world default) | — |
| 2 | Braille scan speed | 10–60 mm/s | today | the braille class's own axis `speed_mm_per_ms: {range: [0.01, 0.06], dist: log_uniform}` overrides the default | — |
| 3 | Scan direction | mostly across the finger, lateral travel ≈ 2.5 × proximal–distal, declared finger axis on the grid | **missing** | `direction_deg` takes only uniform, categorical or a finite registered distribution. New: `direction_deg: {dist: biased_direction, travel_ratio: 2.5, axis_deg: <lateral axis on the canvas>, circular: true}` (P4). Its test-split strata need equal-probability bins for a continuous registered distribution (new `quantile` flag) | T3 |
| 4 | Feature orientation | uniform, independent of the scan | today | `orientation_deg: {range: [0, 180], circular: true}` (edges, gratings; `[0, 360]` for a step edge, whose two sides differ); every axis draws from its own counter-based slot, so it is independent of `direction_deg` | — |
| 5 | Contact types | tap, press, grasp separate | caveat | one class per (feature × contact type), each with its own `touch_ms`/`hold_ms`/`release_ms` axes; class weights multiplied by hand. YAML merge keys do not work with SF's duplicate-key loader, so each class spells its axes. New, optional: world `groups:` reused by a class's `use:` (P1) | T10 |
| 6 | Tap | rise 50–100 ms, touch 100–300 ms log-uniform, fall 50–150 ms | caveat | `touch_ms: {range: [50, 100]}`, `hold_ms: {range: [100, 300], dist: log_uniform}`, `release_ms: {range: [50, 150]}`; "touches for" read as the plateau `hold_ms` (P2); ramps are linear | — |
| 7 | Press | rise 300–600 ms, touch 0.5–2 s log-uniform, falls as it rose | caveat | `touch_ms: {range: [300, 600]}`, `hold_ms: {range: [500, 2000], dist: log_uniform}`. An independent fall from 300–600 is expressible today; a fall equal to the drawn rise is **missing**: new `release_ms: {same_as: touch_ms}` (P3) | T4 |
| 8 | Grasp | rise 150–600 ms | today | `touch_ms: {range: [150, 600]}` (its hold: row 13) | — |
| 9 | Repeated presses | 2–14 per bout, 0.2–1.0 s apart log-uniform | caveat | `contacts: {range: [2, 14], int: true}`, `pause_ms: {range: [200, 1000], dist: log_uniform}`. The presses of one draw are identical (one rise, hold, fall per draw); the int axis stratifies one bin per value (13 ≤ 64) | — |
| 10 | Amplitude unit | 1.0 = 2 mm of indentation | caveat | metadata: `units: {space: mm, time: ms, amplitude: "1.0 = 2 mm indentation"}` (free text, hashed into the id). The geometric shapes need the number: their field `unit_mm` (mm of indentation that amplitude 1.0 stands for), set once with a world default `unit_mm: {value: 2.0}` | T7, T8 |
| 11 | Feature amplitude | log-uniform 0.05–1.0 | today | `amplitude: {range: [0.05, 1.0], dist: log_uniform}` | — |
| 12 | Background contact | every contact presses the whole patch at a level from 0.05–1.0; features ride on it | **missing** | a layer is `amplitude × envelope × shape`, zero outside its features. New: layer field `background` (axis name `background`, bindable from `defaults` so every layered class gets it), added inside the contact envelope; layer constant `clamp_min` (P6) | T2 |
| 13 | Grasps and holds | 1–75 s log-uniform | today | `hold_ms: {range: [1000, 75000], dist: log_uniform}` | — |
| 14 | Static holds | 2–20 s log-uniform | today | a class with `hold_ms: {range: [2000, 20000], dist: log_uniform}` and no slide (answers PS C-073 on the world's side) | — |
| 15 | Session contact fraction | in contact 0.55–0.95 of the time, uniform, per session | **missing** | v1.1.0 sessions lay draws end to end; the quiet share is emergent. New world section `sessions: {contact_fraction: {range: [0.55, 0.95]}, ...}` (P8) | T11 |
| 16 | Quiet gaps | exponential, mean 6 s | **missing** | gaps come only from `quiet` draws, lead-ins and pauses. New `sessions.gap_mean_ms: 6000` (P8) | T11 |
| 17 | Session types | handling and exploration, 50/50 | **missing** | new `sessions.types: {handling: {weight: 0.5, classes: {...}}, exploration: {weight: 0.5, classes: {...}}}`; `sample(..., weights=)` | T11 |
| 18 | Session length | 30–120 s | **missing** | the data set fixes one `duration_ms` per sessions split. New `sessions.duration_ms: {range: [30000, 120000]}`, drawn per session. A 120 s session at 80×80 and 1 ms is 3.1 GB of float32 frames; the runner must render it in time chunks | T11, T12 |
| 19 | Dot diameter | log-uniform 0.25–2.5 mm as the Gaussian's FWHM | caveat | `gaussian` `sigma_mm: {range: [0.10616, 1.06165], dist: log_uniform}` (σ = FWHM / 2√(2 ln 2); log-uniform maps exactly). Records, bins and labels are in σ | — |
| 20 | Single dots | — | today | `gaussian` with pattern `single`, `x_mm`/`y_mm` axes | — |
| 21 | Dot arrays | 1.3–8.5 mm apart | **missing** | the `grid` pattern is a finite list of positions, evaluated one by one; a scan travels up to ~200 mm, so a patch-filling array cannot be a grid. New shape `dot_array` (P9) | T6 |
| 22 | Ridges | 0.3–1.0 mm wide, log-uniform (FWHM) | caveat | `bar`, `profile: gaussian`, `length_mm: 0`, `width_mm` is σ: `{range: [0.12740, 0.42466], dist: log_uniform}` | — |
| 23 | Step edge | shoulder radius 0.5–12.5 mm, log-uniform | **missing** | new indenter shape `step_edge` (`shoulder_radius_mm`, `orientation_deg`, `unit_mm`) (P5) | T7, T8 |
| 24 | Curved contact | radius 5–40 mm, log-uniform | **missing** | new indenter shape `curved_contact` (`radius_mm`, `form: sphere \| cylinder`, `orientation_deg`, `unit_mm`) (P5) | T7 |
| 25 | Self-affine textures | patch-filling, Hurst 0.6–1.0, roll-off 0.5–10 mm | **missing** | new shape `self_affine` (`hurst`, `rolloff_mm`, `cutoff_mm`, `components`, `seed`) (P7); its `seed` axis needs `stratify: false` | T4, T5 |
| 26 | Periodic textures | 0.3–6 mm | caveat | `grating` with `signed: true` (zero-mean sine), `wavelength_mm: {range: [0.3, 6.0], dist: log_uniform}`: patch-filling but one-dimensional (ridged surfaces, not a 2-D weave) (P7). Its "below" probes fall under 0.3 mm; set `probes: false` if unwanted | — |
| 27 | Textures as relief | zero-mean relief on the background contact | **missing** | needs `background` and `clamp_min: 0` (row 12) | T2 |
| 28 | Declared limit | structure finer than 0.3 mm left out | caveat | `self_affine` `cutoff_mm: 0.3`; periodic wavelength ≥ 0.3 by range | T5 |
| 29 | Gratings (held out) | period log-uniform 0.5–6 mm, equal bars and grooves | today | held-out class, `grating` `profile: square`, `duty: 0.5`, `wavelength_mm: {range: [0.5, 6.0], dist: log_uniform}` | — |
| 30 | Braille dot spacing, cell spacing | 2.3–2.5 mm, 6.0–6.22 mm | today | braille pattern `dot_spacing_mm`, `cell_spacing_mm` ranges | — |
| 31 | Braille line spacing | 10.0–10.16 mm | **missing** | the braille pattern lays one line. New `line_spacing_mm` and a `/` line break in `text` and `dots` | T9 |
| 32 | Braille dots | 1.0–1.6 mm (FWHM), 0.25–0.5 mm high | caveat | `shape.sigma_mm: {range: [0.42466, 0.67945]}`; height is amplitude: `{range: [0.125, 0.25]}` in 2 mm units | — |
| 33 | Braille letters | drawn by English letter frequency | caveat / **missing** | one cell: `text: {values: [a, …, z], weights: [<26 frequencies>]}` today. Several cells and lines (needed once a scan carries the text across the patch, or at 80×80): new distribution `letter_text` (`letters`, `weights`, `cells`, `lines`) | T9 |
| 34 | No signage variant | — | today | nothing to build | — |
| 35 | Ranges in mm; each field size reports which draws fit it | — | today (PS side) | every size is in the draw records; which draws fit a 16×16, 40×40 or 80×80 patch is a computation on records, done in PS. No SF work | — |
| 36 | Test split and probes over the new axes | — | **missing** | `stratify_class` refuses a registered distribution with no finite support. New: axis flag `stratify: false` (seeds, multi-cell text); continuous registered distributions marked `quantile` are stratified in equal-probability bins | T3, T4 |

---

## 2. Provisional decisions (pending the supervisor's or Ben's answer)

Neither §174, the contract nor the code settles these. The plan is written on the recommended
reading; each is marked **[P#]** where it is used. Changing an answer changes only the named task.

- **P1 · Contact types.** (A) one class per (feature × contact type), with optional world `groups:`
  of axes that a class pulls in with `use:` (T10, small, sugar only; draws are unchanged by it).
  (B) class "variants": one class whose categorical variant axis selects a group of axes per draw
  (larger: sampling, strata per variant, entry ids). **Recommended: A.** It keeps a class = one
  stratum, so PS reports per contact type per feature for free.
- **P2 · "touches for".** (a) the plateau `hold_ms` (§174's wording reads rise → touch → fall as three
  phases; expressible today). (b) the total contact time (the survey's sources measured keystroke
  duration), which needs a new episode field `contact_ms` with the hold derived as
  `max(0, contact − touch − slide − release)` (one extra small task; tap draws with rise + fall above
  the drawn contact would then lose their plateau). **Recommended: a.**
- **P3 · A press "falls as it rose".** (a) the fall equals the drawn rise: `release_ms: {same_as: touch_ms}`
  (T4). (b) an independent draw from the same 300–600 ms (expressible today). **Recommended: a**
  (the survey says "the fall mirrors the rise").
- **P4 · The lateral bias.** (a) `biased_direction`: the direction of an anisotropic Gaussian velocity
  (angular central Gaussian), θ = axis + atan2(sin 2πu, s·cos 2πu), whose stretch s is solved from
  the declared travel ratio by the closed form R(s) = s·atan(k) / artanh(k/s), k = √(s² − 1)
  (checked: R = 2.5 gives s = 3.87625; a 2·10⁶-point quadrature of E|cos θ| / E|sin θ| gives 2.5000000).
  One uniform per draw, monotone in u. (b) an axial von Mises with κ solved for the ratio (no closed
  inverse CDF). (c) a mixture: lateral sweeps with probability p, else uniform. **Recommended: a.**
  PS declares which canvas axis is lateral (`axis_deg`).
- **P5 · Indenter shapes (step edge, curved contact).** (a) depth-driven: the shape is a rigid
  indenter pressed to depth d(t) = amplitude × envelope(t) × modulation(t), so the footprint grows
  during the rise (value = max(0, d(t) − sag(x)/unit_mm)). (b) separable, like every other shape:
  a fixed footprint computed at peak depth, scaled by the envelope. **Recommended: a** (both elements
  are defined by physical radii; (b) would misstate the contact area during ramps). Curved contact:
  sphere and cylinder, convex only (the survey's concave 20–40 mm surfaces are not built).
- **P6 · Background and relief.** The background is its own draw (from the feature range, as §174
  says), rises and falls with the same contact envelope and modulation, is not moved by motion, and
  the layer total is floored at `clamp_min: 0` where zero-mean relief dips below it (the skin leaves
  contact; no negative indentation). Alternative: no floor, PS keeps relief amplitude below the
  background (not expressible: the schema has no joint constraints). **Recommended: the floor.**
- **P7 · Self-affine texture.** Persson's isotropic 2-D spectrum: flat below the roll-off wavenumber
  q0 = 2π/`rolloff_mm`, ∝ q^(−2(H+1)) above it, zero above q1 = 2π/`cutoff_mm`; synthesised as a sum of
  `components` (default 256) cosines with radial wavenumbers drawn by stratified inverse CDF of
  q·C(q), uniform directions and phases, from the shape's own `seed`; zero mean; RMS =
  amplitude/√2 (the RMS of a unit-peak signed sinusoid, so a self-affine and a periodic texture of
  equal amplitude carry equal power). Options for the scale: RMS = amplitude/√2 (recommended),
  RMS = amplitude, or 3·RMS = amplitude. Periodic textures stay the one-dimensional signed grating;
  a two-dimensional weave (`plaid`) is not built unless asked.
- **P8 · Session layout.** (a) budgeted: draw episodes from the session type's mix until their contact
  time reaches f × D; spend the rest of D as G = max(1, round(Q / gap_mean_ms)) gaps at distinct
  episode boundaries, lengths the uniform spacings of the quiet budget Q (near-exponential, mean
  ≈ gap_mean_ms). The declared fraction holds per session (up to the last episode's overshoot).
  (b) renewal: alternate bouts sized from f and gaps drawn from an exponential axis; gaps are exactly
  exponential, the fraction holds only on average, and an `exponential` distribution is added.
  **Recommended: a.** The realised quiet fraction is recorded either way. This supersedes the v1.1.0
  spec's decision 5 ("no gap mechanism, no quiet-fraction target") for worlds that declare
  `sessions:`; others keep it.
- **P9 · Dot arrays.** A lattice of Gaussian bumps (each peaking at `amplitude`, overlaps summed, as a
  `grid` of `gaussian`s sums), `arrangement: square | hexagonal`, spacing and row spacing, rotated by
  `orientation_deg`, translated by the pattern's `x_mm`/`y_mm`. Minor; no question unless the
  supervisor wants another bump profile.

---

## 3. Global constraints (every task)

- Work only in `/Users/benefron/sensoryforge/.claude/worktrees/world-174`, branch
  `feat/world-174-elements`. Never write on `main` or in another worktree. **Never push, never push a
  tag** (SensoryForge is a public repository). If a tool call is denied, stop and report it; do not
  work around it.
- **Never `pip install -e .`** from the worktree (F-053). Tests put the worktree on `sys.path`
  (`tests/conftest.py`); a subprocess gets `PYTHONPATH=<worktree>`.
- Run everything as `conda run -n sensoryforge python -m pytest …` from the worktree root. `conda run`
  passes no heredoc on stdin: write a script file instead. Code must also run on Python 3.10 +
  torch 2.2.2 + numpy 1.26 (PS's `bio-encoding`); no new dependencies.
- **Invisible until used (IU), the bit-equality rule:**
  1. A world or data-set file that uses no v1.2 key loads to the same normalised dict, `world_id`,
     `dataset_id`, draws, records, entries, session records and frames, bit for bit, as v1.1.0.
  2. A v1.2 key appears in `World.to_dict()`, `ClassSpec.to_dict()`, a normalised layer,
     `AxisSpec.to_dict()`, `Draw.to_dict()`, `Session.to_dict()`, a manifest row or a bundle payload
     only when the file uses it. No new built-in default is added to `LayeredKind.builtin_defaults`
     (it would put a new key in every record).
  3. Where a v1.2 feature changes a formula, the v1.1.0 expression stays verbatim on the path of draws
     that do not use it (no `+ 0` or `× 1` rewrites: they change float rounding).
  4. A field added to an existing shape or pattern kind (only braille `line_spacing_mm` in this plan)
     is left out of the normalised layer unless the file sets it, and read with its default.
  5. Format tags stay `sensoryforge-world/1` and `sensoryforge-dataset/1` (both are hashed into every
     id); bundle `SCHEMA_VERSION` stays `2.2.0` (every addition is optional and present only when used;
     bundles' `reconstructible_by_pressure_simulation` is already `false`).
- `layered` and the world renderer are two implementations of one language: any composition change
  goes into both, and `test_world_render_equals_layered`, `test_layered_golden.py` and the new
  element tests keep them equal (1e-5; layered keeps time in float32).
- New shapes, patterns and distributions are **registered**, not special-cased by kind name in the
  renderer. A capability several shapes share (the indenter composition) is a flag on the registered
  `ShapeKind`.
- Batch invariance: draw *i* alone equals draw *i* in any batch or chunk, bit for bit. A loop whose
  length depends on a draw's parameters runs to the group's maximum and masks each draw's surplus
  terms to exact zeros, accumulating in a fixed order from a zero tensor (adding 0.0 leaves a float
  unchanged).
- Golden comparisons against committed fixtures use `sensoryforge.testing.golden.assert_matches_golden`
  off the reference platform (F-071); exact sha256 comparisons only on macOS arm64, where the
  fixtures are made.
- Input validation raises `ValueError` naming the world path (`world.classes.<name>.axes.<axis>: …`),
  never `assert`. Google-style docstrings with shapes and units (mm, ms). `black` and `flake8` clean.
- SensoryForge's tests use placeholder values sized for its 8×8 fixture design and small canvases,
  never §174's values.
- Test file basenames are unique across `tests/` (no `__init__.py` files).
- Commits: Conventional Commits subject; a ledger trailer in the last paragraph (`Decision:`,
  `Finding:`, `Fixed:`, `Opens:`, `Refs:` … or `Ledger: none — <reason, 3+ words>`), one line each;
  end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The hooks in
  `~/sensoryforge/.git/hooks` check the trailer and sync the ledger; never run the sync by hand. A
  commit that adds, deletes or renames files says so in its body.

**The world gate** (run at the end of every task; expected: no failures, no errors):

```bash
cd /Users/benefron/sensoryforge/.claude/worktrees/world-174
conda run -n sensoryforge python -m pytest tests/unit/test_world_*.py tests/unit/test_layered_*.py \
  tests/unit/test_bundle_world_entry.py tests/contract/test_world_contract.py \
  tests/integration/test_world_batch.py tests/integration/test_cli_world_dataset.py -q
conda run -n sensoryforge black --check sensoryforge tests
conda run -n sensoryforge flake8 sensoryforge tests
```

**Baseline before any change (2026-10-04, this branch at `9c935de`, macOS arm64, Python 3.11.14,
torch 2.5.1):** `pytest -m "not gui"`: 1981 passed, 12 skipped, 554 deselected (3 min 41 s);
`pytest -m gui` (offscreen): 554 passed, 1993 deselected (4 min 20 s). No failures.

---

## 4. File map

| File | Status | Responsibility | Tasks |
|---|---|---|---|
| `tests/fixtures/make_world_v1_1_reference.py` | create | records v1.1.0's ids, records, manifests, sessions, frames and one bundle | T1 |
| `tests/fixtures/worlds/every_element_v1_1.yml` | create | v1.1.0's full element set as a file (the `EXTRA` world of `test_world_render.py`) | T1 |
| `tests/fixtures/worlds/v1_1_0_reference/` | create | `reference.json`, `frames.pt`, `bundle_test/`, `bundle_session/` | T1 |
| `tests/unit/test_world_v1_1_compat.py` | create | the IU guard | T1 |
| `tests/fixtures/worlds/elements_v1_2.yml` | create, then extend | a test world using every v1.2 element (one class per element) | T2–T11 |
| `tests/unit/test_world_v1_2_render.py` | create | parametrized over `elements_v1_2.yml`: equals layered, alone = batch, 40×40 vs 80×80 | T2 (grows with each class) |
| `sensoryforge/stimuli/layered.py` | modify | `background`, `clamp_min`, indenter branch, braille lines, `ADDED_FIELDS` | T2, T7, T9 |
| `sensoryforge/world/kinds.py` | modify | bind `background`; layer keys; indenter composition; `_fill` honours `ADDED_FIELDS` | T2, T7, T9 |
| `sensoryforge/world/kernel.py` | modify | `ShapeKind.indenter`; register the new shapes | T5–T8 |
| `sensoryforge/world/surfaces.py` | create | `self_affine`, `dot_array`, `curved_contact`, `step_edge` functions and specs | T5–T8 |
| `sensoryforge/world/distributions.py` | modify | `quantile` and `check` on registered distributions; `stratify`; `same_as`; `biased_direction`; `letter_text` | T3, T4, T9 |
| `sensoryforge/world/schema.py` | modify | link validation; `groups:`/`use:`; `sessions:` section | T4, T10, T11 |
| `sensoryforge/world/sampling.py` | modify | links; `sample(weights=)`; the session model | T4, T11 |
| `sensoryforge/world/dataset.py` | modify | `stratify: false`; quantile strata; links in strata and probes; sessions split | T3, T4, T11 |
| `sensoryforge/world/runner.py` | modify | time-chunked rendering into float32 | T12 |
| `sensoryforge/world/__init__.py` | modify | exports (`register_shape(..., indenter=)` already public via kernel) | T7, T11 |
| `tests/contract/test_world_contract.py` | modify | guarantees 9–11 | T1, T13 |
| `docs/reference/world_contract.md`, `docs/user_guide/worlds.md`, `CHANGELOG.md`, `CLAUDE.md`, `docs_root/DECISIONS.md`, `.claude/rules/world-engine.md`, `docs/development/specs/2026-10-02-world-engine-design.md` | modify | docs | T13 |
| `pyproject.toml`, `sensoryforge/__init__.py`, `CITATION.cff` | modify | version 1.2.0 | T14 |

## 5. Review focus

1. **IU everywhere:** an old world with a braille class must keep its `world_id` after braille gains
   `line_spacing_mm` (T9); an old session must keep its record after `Session` gains fields (T11).
   Pinned by T1's tests.
2. **Indenters at the envelope's zero:** depth 0 must render exactly 0 (quiet stretches and pauses,
   contract guarantee 7). Pinned in T7.
3. **Masked loops** (`dot_array` neighbours, `self_affine` components) in a batch with mixed parameters:
   bit-equal to each draw alone. Pinned in T5 and T6.
4. **Float32 rendering of integer fields** (`seed`, `components`): `_group_params` makes every number
   a tensor of the output dtype, so integer fields are bounded by 2²⁴ − 1 (exact in float32). Pinned in T5.
5. **Session gaps** render exactly 0, and a session longer than the data set's default duration is
   rendered whole, in chunks, equal to an unchunked render. Pinned in T11 and T12.

---

## 6. Tasks

Order: the guard first; then the elements most of `charter_v1.yml`'s classes use (background,
scan direction, the axis features); then the feature classes (textures, dot arrays, curved contact,
step edge, braille); then reuse sugar; then sessions; then docs and the release.

### Task 1 — Freeze v1.1.0's outputs and guard them (IU)

**Goal.** Before any code change, record what v1.1.0 produces for old worlds, data sets, sessions and
bundles, and add tests that fail if any later task changes it.

**Files.** Create `tests/fixtures/make_world_v1_1_reference.py`,
`tests/fixtures/worlds/every_element_v1_1.yml`, `tests/fixtures/worlds/v1_1_0_reference/reference.json`,
`tests/fixtures/worlds/v1_1_0_reference/frames.pt`,
`tests/fixtures/worlds/v1_1_0_reference/bundle_test/{stimulus.json,frames.npy}`,
`tests/fixtures/worlds/v1_1_0_reference/bundle_session/{stimulus.json,frames.npy}`,
`tests/unit/test_world_v1_1_compat.py`. Modify `tests/contract/test_world_contract.py` (add
`test_9_old_worlds_and_bundles_are_unchanged`, which calls the compat checks).

- [ ] Confirm the code is v1.1.0's: `git diff --quiet 9c935de -- sensoryforge tests` exits 0.
- [ ] Write `every_element_v1_1.yml` from `EXTRA` in `tests/unit/test_world_render.py` (same dict, as YAML).
- [ ] Write the generator. It records, for `tactile_small.yml` and `every_element_v1_1.yml`: `world_id`,
  `json.dumps(world.to_dict(), sort_keys=True)` sha256, the records digest of `sample(world, n=50, seed=7)`
  (floats rounded to 10 significant digits, as `DRAWS_DIGEST`), a session record digest
  (`session(tactile_small, 300.0, seed=7, index=0)`), the `dataset_small.yml` `dataset_id` and the sha256
  of its manifest rows (`build_dataset`, rows serialised with `sort_keys=True`, rounded as above), and
  float64 frames of 12 draws per world at 24 times on `Canvas.from_grid(12, 12, 0.1)` (`frames.pt`,
  plus their sha256 in `reference.json`, with `platform.machine()` and `sys.platform`).
- [ ] Run the generator twice: once on the worktree, once on a clean copy of tag v1.1.0
  (`git archive v1.1.0 | tar -x -C $SCRATCH/sf_v1_1_0`, run with `PYTHONPATH=$SCRATCH/sf_v1_1_0`).
  `cmp` the two `reference.json` files: identical. Commit the v1.1.0 copy's output.
- [ ] Record two v1.1.0 bundles with the v1.1.0 copy:
  `PYTHONPATH=$SCRATCH/sf_v1_1_0 conda run -n sensoryforge python -m sensoryforge.cli batch --design tests/fixtures/design_8x8 --dataset tests/fixtures/worlds/dataset_small.yml --output $SCRATCH/b110 --splits test --entries 0:1`
  and the same with `--splits sessions`. Keep each bundle's `stimuli/stimulus.json` and its
  `/stimulus/frames` (read with h5py) as `frames.npy`.
- [ ] Tests (written now; they pass now), in `test_world_v1_1_compat.py`:
  `test_old_world_ids_and_normal_forms_are_unchanged`, `test_old_draw_records_are_unchanged`,
  `test_old_session_record_is_unchanged`, `test_old_dataset_id_and_manifest_are_unchanged`,
  `test_old_frames_are_unchanged` (sha256 on darwin arm64, else `assert_matches_golden(atol=1e-12)`),
  `test_a_v1_1_bundle_record_rebuilds_and_rerenders_bit_equal` (for both bundles: `Draw.from_dict` /
  `Session.from_dict` with the world succeeds — the world id still matches — and
  `render_movie(..., Canvas.from_grid_config(<the bundle's grid>), dt, duration, dtype=float64).float()`
  equals `frames.npy`; exact on darwin arm64, golden tolerance elsewhere). The module docstring states
  the IU rule and says a failure means an old world changed.

**Verify.** `conda run -n sensoryforge python -m pytest tests/unit/test_world_v1_1_compat.py tests/contract/test_world_contract.py -q`
→ all passed (6 + 9 tests). The world gate passes.

**Acceptance.** The fixtures come from tag v1.1.0 and equal the worktree's; the six tests and contract
test 9 pass; nothing under `sensoryforge/` changed.

**Depends on:** nothing. **Size:** small–medium.

### Task 2 — The background contact and the floor (`background`, `clamp_min`) [P6]

**Goal.** A layer may carry a uniform `background` level, added inside the contact envelope, and a
`clamp_min` floor on its total, in both `layered` and the world renderer.

**Semantics.** For a layer with either key: `frame = envelope × modulation × (amplitude × Σ shape + background)`,
then `max(frame, clamp_min)` if `clamp_min` is set; exactly 0 wherever the envelope is 0 (so
`clamp_min` must be ≤ 0 or the floor applies only where the envelope is positive — implement the floor as
`where(env > 0, max(frame, clamp_min), 0)`). The background ignores pattern and motion. Without either
key, the v1.1.0 expression (`amplitude * env * total` in `kinds.py`, `frames * amplitude` in
`layered.py`) runs unchanged (IU 3).

**Files.** Modify `sensoryforge/stimuli/layered.py` (`render_layer`), `sensoryforge/world/kinds.py`
(`LayeredKind.normalise_layer` accepts `background` (number ≥ 0) and `clamp_min` (number) and keeps them
only if given; `resolve("background")` → `("layer", "background")`, after `amplitude`; `domain` →
`(0.0, 1.0e4)`; `field_info` → number; `fixed_in_layer` for it; `part_values` carries layer-level values;
`to_layer` emits `background` when bound or set and `clamp_min` when set; `render_group` new path).
Create `tests/fixtures/worlds/elements_v1_2.yml` (classes `bg_dots`: gaussian with a `background` axis;
`relief`: signed sine grating with a `background` axis and layer `clamp_min: 0.0`), `tests/unit/test_world_background.py`,
`tests/unit/test_world_v1_2_render.py`.

- [ ] Tests first (`test_world_background.py`):
  `test_background_alone_is_level_times_envelope` (amplitude 0: every canvas point equals
  `background × env(t)` exactly, from `contact_terms`); `test_background_adds_under_a_feature`
  (gaussian: centre = (A + B)·env, far field = B·env, to 1e-12); `test_the_floor_keeps_relief_nonnegative`
  (signed grating A = 1, B = 0.3, `clamp_min: 0`: equals `max(0, B + A cos(…))·env` and min ≥ 0);
  `test_quiet_stretches_stay_exactly_zero` (lead-in, pause, after `end_ms`: 0.0 exactly, with clamp);
  `test_a_world_default_background_binds_every_layered_class_and_skips_quiet`;
  `test_background_is_recorded_and_carried_into_to_layer`;
  `test_layers_without_the_keys_render_as_before` (relies on T1's compat tests and `test_layered_golden.py`).
- [ ] Tests first (`test_world_v1_2_render.py`, parametrized over every non-quiet class of
  `elements_v1_2.yml`): `test_v1_2_class_equals_layered` (as `test_world_render_equals_layered`, 1e-5),
  `test_v1_2_draw_alone_equals_draw_in_a_batch` (bit for bit, also with `max_elements=1`),
  `test_v1_2_draw_agrees_on_40x40_and_80x80` (shared points at 0.15 mm, 1e-12), `test_v1_2_determinism_by_seed`.
- [ ] Implement; run the tests; run the world gate.

**Verify.** `conda run -n sensoryforge python -m pytest tests/unit/test_world_background.py tests/unit/test_world_v1_2_render.py tests/unit/test_world_v1_1_compat.py -q` → all passed. World gate passes.

**Acceptance.** Both implementations agree; quiet is exact zero; old worlds unchanged (T1 green).

**Depends on:** T1. **Size:** medium.

### Task 3 — Continuous registered distributions in strata; `biased_direction` [P4]

**Goal.** Lateral-biased scan directions, and equal-probability test strata for any registered
distribution whose `sample(u)` is a quantile function.

**Files.** Modify `sensoryforge/world/distributions.py` (`Distribution` gains `quantile: bool = False`
and `check: Optional[Callable[[AxisSpec], None]] = None`; `register_distribution(..., quantile=False, check=None)`;
`AxisSpec.from_dict` calls `check` for a registered axis, so bad options fail at load naming the path;
register `biased_direction`), `sensoryforge/world/dataset.py` (`stratify_class`: a registered axis whose
distribution has `quantile=True` is stratified like a numeric one on the u scale: stratum `b` draws
`sample((b + u′)/bins)`, label `"[sample(b/bins), sample((b+1)/bins))"` at 6 significant digits; probes
unchanged — a circular axis has none). Extend `elements_v1_2.yml` (class `lateral_slide`). Create
`tests/unit/test_world_biased_direction.py`.

**`biased_direction`.** Options: `travel_ratio` (number > 0, required: expected |travel along the axis| /
expected |travel across it|), `axis_deg` (number, default 0: the biased axis on the canvas, 0° = +x).
For `travel_ratio` r > 1, solve s from R(s) = s·atan(k)/artanh(k/s), k = √(s² − 1), by 200 bisection
steps on [1, 10⁶] at load (cache by r); r < 1 uses 1/s on the swapped axis; r = 1 is uniform.
`sample(u) = (axis_deg + degrees(atan2(sin 2πu, s·cos 2πu))) mod 360`. `check` refuses unknown or missing
options and non-positive ratios. Values may differ in the last bit between platforms (libm), like
`log_uniform`; the contract says so (T13).

- [ ] Tests first: `test_travel_ratio_is_honoured` (u = (i + ½)/N, N = 2·10⁶: E|cos(θ − axis)| / E|sin(θ − axis)|
  = r to 1e-6 for r ∈ {0.5, 1, 2.5, 4}; known answer s(2.5) = 3.87625 to 1e-5);
  `test_ratio_one_is_uniform`; `test_axis_rotates_the_bias`; `test_values_lie_in_0_360`;
  `test_sampling_is_deterministic_and_draw_i_is_independent_of_n`;
  `test_quantile_strata_hold_per_bin_each` (a class with this axis in a test split: `per_bin` draws per
  u-bin, labels as specified); `test_bad_options_fail_at_load_naming_the_path`;
  `test_braille_cells_is_still_stratified_by_support` (no change for finite distributions).
- [ ] Implement; run tests; world gate.

**Verify.** `conda run -n sensoryforge python -m pytest tests/unit/test_world_biased_direction.py tests/unit/test_world_dataset.py tests/unit/test_world_v1_2_render.py -q` → all passed. World gate passes.

**Acceptance.** Ratio honoured, strata balanced, old worlds unchanged.

**Depends on:** T1 (T2 only for the shared test world file). **Size:** small–medium.

### Task 4 — Axis features: `stratify: false` and `same_as` [P3]

**Goal.** An axis may be drawn i.i.d. in the test split (`stratify: false`: texture seeds, multi-cell
text), and an axis may copy another axis's value in the same draw (`same_as`: a fall equal to the rise).

**Files.** Modify `sensoryforge/world/distributions.py` (`AxisSpec.stratify: bool = True`, key
`stratify` in `_AXIS_KEYS` and excluded from registered options, emitted by `to_dict` only when false;
new form `link` with attribute `link: str`, declared as exactly `{same_as: <axis name>}`, `to_dict` →
`{"same_as": name}`), `sensoryforge/world/schema.py` (after binding: a link's target is an axis of the
class, not itself a link, and every value the target can take passes the link's own field check
(`check_axis` of the target against the link's `FieldInfo`); `_only_zero` and `LayeredKind.check`
treat a link as its target; fixed draws may not set a link), `sensoryforge/world/sampling.py` (`sample`
and `fixed_draw` fill links after the other axes), `sensoryforge/world/dataset.py` (`stratify_class`:
`stratify: false` axes are sampled i.i.d. from their slot with no label; links copied, no label;
`_probe_draws` copies links after setting the probed axis). Extend `elements_v1_2.yml` (class
`tied_press`: `release_ms: {same_as: touch_ms}`, `contacts: {range: [2, 4], int: true}`). Create
`tests/unit/test_world_axis_links.py`.

- [ ] Tests first: `test_same_as_copies_in_declared_strata_probes_and_fixed_draws`;
  `test_a_link_to_a_missing_axis_or_to_a_link_fails_at_load`;
  `test_a_link_whose_target_leaves_its_field_domain_fails`;
  `test_a_fixed_draw_may_not_set_a_link`; `test_stratify_false_samples_iid_with_no_label`;
  `test_a_large_int_seed_axis_with_stratify_false_builds_a_test_split` (today it raises above 64 values);
  `test_the_new_keys_change_the_world_id_only_when_used`.
- [ ] Implement; run tests; world gate.

**Verify.** `conda run -n sensoryforge python -m pytest tests/unit/test_world_axis_links.py tests/unit/test_world_schema.py tests/unit/test_world_dataset.py -q` → all passed. World gate passes.

**Acceptance.** Links and the flag behave in every sampling mode; records list a link's value like
any axis; old worlds unchanged.

**Depends on:** T1. **Size:** small.

### Task 5 — `self_affine`: patch-filling self-affine relief [P7]

**Goal.** A zero-mean random rough surface with a declared Hurst exponent, roll-off and cutoff,
continuous in space (grid-independent), deterministic by its own seed.

**Files.** Create `sensoryforge/world/surfaces.py` (function and `ParamSpec`s), modify
`sensoryforge/world/kernel.py` (register it, `unbounded=False`, so the pattern's `x_mm`/`y_mm`
translate the surface). Extend `elements_v1_2.yml` (class `rough`: `seed: {range: [0, 16777215], int: true, stratify: false}`,
a `background` axis, `clamp_min: 0.0`). Create `tests/unit/test_world_self_affine.py`.

**Fields.** `amplitude`; `hurst` (0.0–1.0, default 0.8); `rolloff_mm` (0.01–1000, default 2.0);
`cutoff_mm` (0.001–100, default 0.3); `components` (int 16–4096, default 256); `seed` (int 0–16777215,
default 0: integers stay exact in float32). **Value.** h(x, y) = Σⱼ N^(−½) cos(qⱼ(x cos ψⱼ + y sin ψⱼ) + φⱼ),
RMS 1/√2 in expectation, then × amplitude by the usual composition. qⱼ by stratified inverse CDF of the
radial density ∝ q·C(q) on [0, q1] (C flat to q0, ∝ q^(−2(H+1)) to q1, closed-form inverse for both
pieces, log form at H = 0), stratum j at (j + vⱼ)/N; ψⱼ, φⱼ, vⱼ from `sensoryforge.world.rng` keyed by
(seed, j, slot), in float64, cached per distinct (seed, hurst, rolloff, cutoff, components) like
`pattern_batch`'s positions. The component loop runs to the group's largest `components` with surplus
terms masked to zero (batch invariance). Signed by construction (relief); documented as exempt from
the non-negative default, used with `background` and `clamp_min`.

- [ ] Tests first: `test_one_component_equals_its_cosine` (components 16 with a known table, or the
  table exposed by a helper: values equal the explicit cosine sum to 1e-12);
  `test_mean_and_rms` (64 seeds on a 128×128 canvas spanning 4 roll-off lengths: |mean| < 0.05·A,
  RMS = A/√2 within 10 %); `test_spectral_slope_matches_hurst` (radially averaged periodogram between
  2q0 and q1/2 on a fine canvas, averaged over seeds: slope −2(H+1) within ±0.2 for H ∈ {0.6, 1.0});
  `test_no_power_above_the_cutoff`; `test_same_seed_same_surface_other_seed_differs`;
  `test_a_batch_with_mixed_components_equals_each_alone_bit_for_bit`;
  `test_float32_and_float64_agree_to_1e-5` (seed 16777215 included); `test_x_mm_translates_the_surface`.
  The shared `test_world_v1_2_render.py` covers layered equality and 40×40 vs 80×80.
- [ ] Implement; time 1000 draws × 3 frames at 40×40 float64 and note it in the commit body.

**Verify.** `conda run -n sensoryforge python -m pytest tests/unit/test_world_self_affine.py tests/unit/test_world_v1_2_render.py -q` → all passed. World gate passes.

**Acceptance.** Statistics within tolerance; deterministic; grid-independent; batch-invariant.

**Depends on:** T4 (`stratify: false` for the seed axis in data sets); T2 for the test class's background. **Size:** large.

### Task 6 — `dot_array`: a patch-filling lattice of dots [P9]

**Goal.** An unbounded lattice of Gaussian bumps that a scan can cross for any distance.

**Files.** Modify `sensoryforge/world/surfaces.py`, `sensoryforge/world/kernel.py` (register,
`unbounded=False`). Extend `elements_v1_2.yml` (class `dot_array`). Create `tests/unit/test_world_dot_array.py`.

**Fields.** `amplitude`; `sigma_mm` (0.001–50); `spacing_mm` (0.01–1000, default 2.0); `row_spacing_mm`
(0–1000, default 0 = `spacing_mm` for square, `spacing_mm·√3/2` for hexagonal); `orientation_deg`;
`arrangement` (`square` | `hexagonal`, hexagonal rows offset by half a spacing). **Value.** In the
lattice frame, wrap each point to its nearest lattice site and sum bumps over the neighbours within
m = ⌈7.5 σ / min(a, b)⌉ sites each way (truncation below 1e-12 of the peak), loop to the group's largest
m with masked surplus; m > 32 raises `ValueError` naming σ and the spacing.

- [ ] Tests first: `test_equals_a_brute_force_sum_over_lattice_sites` (square and hexagonal, rotated,
  translated: equals an explicit sum over every site within 10σ of the canvas, 1e-12, float64);
  `test_peak_is_amplitude_when_dots_are_far_apart`; `test_equals_a_layered_grid_of_gaussians_on_the_canvas`
  (a `grid` pattern large enough to cover the canvas: 1e-12); `test_a_batch_with_mixed_sigma_and_spacing_equals_each_alone_bit_for_bit`;
  `test_x_mm_shifts_the_lattice`; `test_too_wide_a_dot_for_its_spacing_is_refused`.
- [ ] Implement; world gate.

**Verify.** `conda run -n sensoryforge python -m pytest tests/unit/test_world_dot_array.py tests/unit/test_world_v1_2_render.py -q` → all passed. World gate passes.

**Acceptance.** Known answers hold; batch-invariant; layered equal.

**Depends on:** T1. **Size:** medium.

### Task 7 — Indenter shapes and `curved_contact` [P5]

**Goal.** A registered shape may be an *indenter*: it is evaluated at depth d(t) = amplitude × envelope ×
modulation and returns the indentation itself. The first indenter is a curved contact surface.

**Files.** Modify `sensoryforge/world/kernel.py` (`ShapeKind.indenter: bool = False`;
`register_shape(..., indenter=False)`), `sensoryforge/world/kinds.py` (`render_group`: for an indenter,
`depth = (amplitude * env).reshape(lead)` after modulation and the end mask; `total = Σ scale × fn(…, {**params, "depth": depth × scale})`;
not multiplied again; then `background` and `clamp_min` as in T2), `sensoryforge/stimuli/layered.py`
(`_shape_kind` also returns the flag; `render_layer` evaluates an indenter with a `depth` tensor
`amplitude × envelope` broadcast over time, static or moving), `sensoryforge/world/surfaces.py`.
Extend `elements_v1_2.yml` (classes `sphere`, `cylinder`, with a world default `unit_mm`). Create
`tests/unit/test_world_indenters.py`.

**`curved_contact` fields.** `amplitude` (peak depth, in units); `radius_mm` (0.1–1000, default 20);
`form` (`sphere` | `cylinder`); `orientation_deg` (cylinder axis, the `bar` convention: across = x sinθ + y cosθ);
`unit_mm` (0.001–100, default 1.0: mm of indentation per unit amplitude). **Value.** r² = x² + y²
(sphere) or across² (cylinder); sag = R − √(R² − r²) for r ≤ R; value = max(0, d − sag/unit_mm) inside,
0 for r > R.

- [ ] Tests first: `test_centre_depth_equals_amplitude_times_envelope`; `test_contact_radius_follows_the_geometry`
  (in mm, δ = d·unit_mm: the value reaches 0 at a = √(2Rδ − δ²), sphere and cylinder, 1e-9);
  `test_the_footprint_grows_during_the_rise` (at half the rise the contact radius is that of δ/2);
  `test_a_cylinder_is_constant_along_its_axis`; `test_depth_zero_is_exactly_zero` (lead-in, pause,
  after `end_ms`); `test_a_registered_indenter_works_in_a_layered_stimulus`;
  `test_non_indenter_shapes_are_unchanged` (T1 green).
- [ ] Implement; world gate.

**Verify.** `conda run -n sensoryforge python -m pytest tests/unit/test_world_indenters.py tests/unit/test_world_v1_2_render.py tests/unit/test_world_kernel.py -q` → all passed. World gate passes.

**Acceptance.** Geometry known answers; exact zeros; both implementations equal.

**Depends on:** T2 (composition with `background`). **Size:** medium.

### Task 8 — `step_edge` [P5]

**Goal.** A flat surface that ends in a rounded shoulder, as an indenter.

**Files.** Modify `sensoryforge/world/surfaces.py`, `sensoryforge/world/kernel.py` (register,
`indenter=True`, `unbounded=False`). Extend `elements_v1_2.yml` (class `step`) and
`tests/unit/test_world_indenters.py`.

**Fields.** `amplitude`; `shoulder_radius_mm` (0–1000, default 2.0; 0 = a sharp step);
`orientation_deg` (`bar` convention; the plate lies where across ≤ 0, so 0–360 covers both sides);
`unit_mm`. **Value.** p = across: d for p ≤ 0; max(0, d − (ρ − √(ρ² − p²))/unit_mm) for 0 < p < ρ;
0 for p ≥ ρ.

- [ ] Tests first: `test_plate_side_is_the_depth`; `test_shoulder_profile_is_circular` (1e-12);
  `test_contact_ends_where_the_shoulder_rises_above_the_depth` (p = √(2ρδ − δ²) when δ < ρ; a drop
  from d − ρ/unit_mm to 0 at p = ρ when δ ≥ ρ); `test_a_sharp_step_is_a_hard_edge` (layered equality
  excludes points within 1e-4 mm of the edge, as for `disc`); `test_orientation_flips_the_plate_side`.
- [ ] Implement; world gate.

**Verify.** `conda run -n sensoryforge python -m pytest tests/unit/test_world_indenters.py tests/unit/test_world_v1_2_render.py -q` → all passed. World gate passes.

**Depends on:** T7. **Size:** small.

### Task 9 — Braille pages and letters: `line_spacing_mm`, `/`, `letter_text`

**Goal.** Several lines of braille, and text drawn letter by letter from declared weights.

**Files.** Modify `sensoryforge/stimuli/layered.py` (braille `line_spacing_mm`, 0.1–1000, default 10.0;
`/` starts a new line in `text` and in `dots`; line L sits at y0 − L·line_spacing_mm; the cell index
restarts on each line; a module-level `ADDED_FIELDS = {("pattern", "braille"): ("line_spacing_mm",)}`),
`sensoryforge/world/kinds.py` (`_fill` leaves `ADDED_FIELDS` out unless given — IU 4),
`sensoryforge/world/distributions.py` (register `letter_text`). Extend `elements_v1_2.yml` (class
`braille_text`). Create `tests/unit/test_world_braille_lines.py`.

**`letter_text`.** Options: `letters` (text, default `"abcdefghijklmnopqrstuvwxyz"`; every character a
letter the braille pattern knows, or a space), `weights` (numbers, one per letter, default equal),
`cells` (int ≥ 1, default 1, letters per line), `lines` (int ≥ 1, default 1). `sample(u)`: the draw's
53 bits are recovered exactly as `int(u · 2⁵³)`, expanded to `cells × lines` sub-uniforms with
`rng.uniforms(rng.draw_seeds(bits, range(cells·lines)), "letter")`, each mapped through the cumulative
weights; lines joined with `/`. `support` is the letter list when `cells × lines == 1` (stratified per
letter), else `None` (the axis needs `stratify: false`; the load error says so). `check` validates options.

- [ ] Tests first: `test_two_lines_put_dots_at_known_positions` ("ab/c": every dot's (x, y));
  `test_single_line_braille_renders_as_before` (`test_layered_golden.py` plus a world id check on
  `tactile_small`); `test_an_old_braille_layer_normalises_without_line_spacing`;
  `test_letter_text_is_deterministic_and_follows_its_weights` (20 000 letters: χ² p > 0.001);
  `test_one_cell_letter_text_is_stratified_by_letter`;
  `test_multi_cell_letter_text_without_stratify_false_fails_at_load_with_a_hint`;
  `test_unknown_letters_fail_at_load`.
- [ ] Implement; world gate; also run `pytest -m gui -q` once (the layered editor builds forms from
  `PATTERNS`).

**Verify.** `conda run -n sensoryforge python -m pytest tests/unit/test_world_braille_lines.py tests/unit/test_layered_golden.py tests/unit/test_world_v1_1_compat.py -q` → all passed. World gate passes; GUI suite passes.

**Depends on:** T4. **Size:** medium–small.

### Task 10 — Reusable axis groups: `groups:` and `use:` [P1]

**Goal.** Declare a contact type's axes once (`groups: {tap: {...}}`) and pull them into any class
(`use: [tap]`). Sugar only: the normalised class holds the resolved axes, so draws do not depend on it.

**Files.** Modify `sensoryforge/world/schema.py` (`groups` in `_TOP_KEYS`, `use` in `_CLASS_KEYS`;
precedence built-ins < `defaults` < used groups < class axes; a group axis naming a field the class
lacks fails (unlike `defaults`); two used groups setting one field fail; a field the class layer fixes
wins, as for defaults; `World.to_dict()` holds no `groups` key — the resolved class axes carry it).
Extend `elements_v1_2.yml`. Create `tests/unit/test_world_groups.py`.

- [ ] Tests first: `test_a_used_group_sets_the_class_axes`; `test_class_axes_override_a_group`;
  `test_two_groups_on_one_field_fail`; `test_an_unknown_group_or_field_fails_naming_the_path`;
  `test_a_world_written_with_groups_equals_the_same_world_written_out` (same `world_id`, same draws).
- [ ] Implement; world gate.

**Verify.** `conda run -n sensoryforge python -m pytest tests/unit/test_world_groups.py tests/unit/test_world_schema.py -q` → all passed. World gate passes.

**Depends on:** T1. **Size:** small.

### Task 11 — The session model: `sessions:` [P8]

**Goal.** Sessions of a declared length, with a declared contact fraction per session, quiet gaps of a
declared mean and session types with their own class mix. Worlds without `sessions:` keep v1.1.0's
sessions bit for bit.

**Schema.** World key `sessions:` with `duration_ms` (an axis; values > 0), `contact_fraction` (an axis;
values in [0, 1]), `gap_mean_ms` (number > 0), optional `types: {<name>: {weight: w ≥ 0, classes: {<class>: w ≥ 0}}}`
(classes of the world, not held out; weights summing > 0). Normalised into `World.to_dict()["sessions"]`
only when present.

**API.** `session(world, duration_ms=None, seed=0, index=0)` (positional calls keep working);
`sample(..., weights: Optional[Mapping[str, float]] = None)` — the classes are the mapping's keys,
weighed in name order; `None` runs today's code path.

**Layout (P8 a).** s = `seed53(seed, index)` (unchanged); type by its weights from
`uniforms([s], "session_type")`; D from `duration_ms` (slot `"duration_ms"`) unless the caller passes one;
f from `contact_fraction` (slot `"contact_fraction"`). Episodes k = 0, 1, … are
`sample(world, indices=[k], seed=s, weights=<type's classes>)`; stop when their contact time (touch,
hold, slide, release phases) reaches f·D or their total length reaches D. Quiet budget Q = max(0, D − E).
If Q > 0: G = min(n + 1, max(1, round(Q / gap_mean_ms))) gaps at boundaries
`sorted(rng.permutation(n + 1, s, "gap_boundaries")[:G])` (boundary 0 is before the first episode, n
after the last), lengths the spacings of Q·sorted(`uniforms(draw_seeds(s, range(G − 1)), "gap_cut")`).
Items keep `[start, draw]`; gaps are holes between items, which render exactly 0. The record adds
`session_type` (when types exist) and `contact_fraction` (the target); `quiet_fraction` is the realised
share, as today.

**Data sets.** A sessions split's `duration_ms` is optional when the world declares `sessions:` (an
explicit value overrides D); each entry's `duration_ms` is its session's D.

**Files.** Modify `sensoryforge/world/schema.py`, `sensoryforge/world/sampling.py` (`Session` gains
optional fields defaulting to `None`, kept out of `to_dict` when `None`), `sensoryforge/world/dataset.py`,
`sensoryforge/world/__init__.py`, `sensoryforge/cli.py` (`world validate` prints the session model).
Extend `elements_v1_2.yml` (`sessions:` with two types, short placeholder durations); create
`tests/fixtures/worlds/dataset_v1_2.yml` and `tests/unit/test_world_session_model.py`; extend
`tests/contract/test_world_contract.py::test_7_session_quiet_stretches_are_exactly_zero` to cover a
model session.

- [ ] Tests first: `test_a_world_without_sessions_keeps_v1_1_sessions` (T1 digest);
  `test_session_length_and_target_fraction_come_from_their_axes`;
  `test_the_realised_contact_fraction_reaches_the_target` (f ≤ realised ≤ f + longest episode/D, over
  50 sessions); `test_gaps_render_exactly_zero`; `test_gap_count_and_mean_follow_gap_mean_ms`;
  `test_type_frequencies_follow_their_weights` (2000 sessions, binomial 4σ);
  `test_episodes_come_only_from_their_types_classes`; `test_sessions_are_deterministic_and_differ_by_index`;
  `test_a_session_record_round_trips`; `test_a_sessions_split_takes_each_sessions_own_length`;
  `test_sample_weights_none_is_unchanged_and_weights_reweigh`; `test_bad_session_sections_fail_naming_the_path`.
- [ ] Implement; world gate.

**Verify.** `conda run -n sensoryforge python -m pytest tests/unit/test_world_session_model.py tests/unit/test_world_sampling.py tests/unit/test_world_dataset.py tests/contract/test_world_contract.py -q` → all passed. World gate passes.

**Acceptance.** Old sessions bit-equal; model sessions meet their declared statistics; gaps exact zero.

**Depends on:** T1. **Size:** large.

### Task 12 — Render long entries in time chunks (runner)

**Goal.** `sensoryforge batch --dataset` renders an entry whose float64 movie would exceed a memory
budget in time chunks, straight into a float32 frame buffer, bit-equal to rendering it whole. (The
simulation engine itself still holds the whole float32 stimulus: 3.1 GB for 120 s at 80×80 and 1 ms;
that is outside this plan and is reported as an open finding.)

**Files.** Modify `sensoryforge/world/runner.py` (`render_movie(...)` → a helper rendering
`movie_times(dt, duration)[k0:k1]` with `render([item], …, dtype=float64)` per chunk, cast each chunk to
float32; budget 2²⁵ float64 elements per chunk, a module constant overridable in tests). Create
`tests/unit/test_world_runner_chunks.py`.

- [ ] Tests first: `test_chunked_frames_equal_whole_frames_bit_for_bit` (a model session and a plain draw,
  budget forcing ≥ 5 chunks); `test_the_bundle_still_equals_the_in_process_render` (contract guarantee 2
  on a chunked entry).
- [ ] Implement; world gate.

**Verify.** `conda run -n sensoryforge python -m pytest tests/unit/test_world_runner_chunks.py tests/integration/test_world_batch.py tests/contract/test_world_contract.py -q` → all passed.

**Depends on:** T11. **Size:** small.

### Task 13 — Contract, guide, changelog, decisions, rules

**Goal.** The contract describes v1.2.0 exactly; every decision has its record.

**Files.** Modify `docs/reference/world_contract.md` (v1.2.0 throughout; "Getting it" pins `@v1.2.0`;
new subsections: the background and floor, indenters, the four shapes with their fields and value
formulas, `biased_direction`, `letter_text`, `stratify: false`, `same_as`, `groups`/`use`, braille
lines, `sessions:` with the layout; IU as a guarantee; guarantees 9–11 in the table with their tests;
"Across machines" adds `biased_direction` and the new shapes' libm last-bit caveat; the runner's
chunking; the open simulation-memory note), `docs/user_guide/worlds.md` (a section per element with a
short example), `CHANGELOG.md` (`[1.2.0] - <date>`, Added only; "nothing changes for existing worlds,
data sets or bundles"), `CLAUDE.md` ("Worlds and data sets (v1.1.0)" → v1.2.0, one paragraph),
`docs_root/DECISIONS.md` (one section per decision taken, in the file's template, with the answers to
P1–P9 as given by the supervisor/Ben), `.claude/rules/world-engine.md` (add the IU rule and the masked-loop
rule), `docs/development/specs/2026-10-02-world-engine-design.md` (an "Addendum, v1.2.0" paragraph:
decision 5 superseded for worlds with `sessions:`). Add to `tests/contract/test_world_contract.py`:
`test_10_every_v1_2_element_renders_equal_to_layered_and_on_both_grids`,
`test_11_model_sessions_meet_their_declared_fraction_and_gaps_are_zero`.

- [ ] Write the docs; `conda run -n sensoryforge mkdocs build --strict` passes; contract tests pass.

**Verify.** `conda run -n sensoryforge mkdocs build --strict` → exit 0; `conda run -n sensoryforge python -m pytest tests/contract -q` → all passed.

**Depends on:** T1–T12. **Size:** medium.

### Task 14 — Version 1.2.0, full verification, local tag

**Goal.** Release v1.2.0 locally.

**Files.** Modify `pyproject.toml` (`version = "1.2.0"`), `sensoryforge/__init__.py` (`__version__ = "1.2.0"`),
`CITATION.cff` (`version: 1.2.0`; it still says 1.0.0).

- [ ] `conda run -n sensoryforge python -m pytest -m "not gui" -q` → 0 failed (baseline 1981 passed, 12 skipped, plus the new tests).
- [ ] `QT_QPA_PLATFORM=offscreen conda run -n sensoryforge python -m pytest -m gui -q` → 0 failed (baseline 554 passed).
- [ ] `conda run -n sensoryforge black --check sensoryforge tests` → "All done!", no file would be reformatted.
- [ ] `conda run -n sensoryforge flake8 sensoryforge tests` → no output, exit 0.
- [ ] `conda run -n sensoryforge mkdocs build --strict` → exit 0.
- [ ] Commit, then `git tag -a v1.2.0 -m "SensoryForge 1.2.0: the world §174 addendum"` on this branch.
  **Do not push the branch or the tag.**

**Acceptance.** All suites pass, lint clean, tag `v1.2.0` exists locally and points at the release commit.

**Depends on:** T13. **Size:** small.
