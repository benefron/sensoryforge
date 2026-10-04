# Decisions record — the reasoning behind the ledger

**Level 2 of three.** `docs_root/LEDGER.md` holds the *fact* of each decision (one line, generated from
the commit trailer). This file holds the *reasoning*: why, on what evidence, against what
alternative. Level 3 — a deck script, a paper, the README's claims — is whatever audience surface
this repo has, if it has one; `.claude/ledger.conf` names it under `AUDIENCE_SURFACE=`, and it is
updated on request, never automatically.

A decision recorded here but not in the ledger is invisible to future sessions. A decision in the
ledger but not here is a verdict with no argument behind it. Write both.

**This file is append-only, and it keeps its history.** Every section is dated. A decision that is
later superseded is **never deleted and never edited**: it gains a `**Superseded by:**` line, and
the section that replaces it opens with `**Supersedes:**`. The corrections are the record — tidying
them away is the exact failure this system exists to prevent.

**How to add one.** `ledger-sync.sh` has already appended the dated fact to the **log** at the
bottom for you. Write the reasoning as a section here, newest first, using this template:

```markdown
## D-0NN · <short title> · <YYYY-MM-DD>

**Supersedes:** D-0MM · <date>          <!-- omit unless it replaces an earlier decision -->

**What was decided.** <quote the ledger entry verbatim — level 1 and level 2 must agree>

**Why.** <the evidence: numbers, measurements, the `F-`/`D-` ids that forced it. Not "it seemed
cleaner" — what was observed.>

**What was rejected.** <the alternative that was seriously considered, and the specific reason it
lost. This is the half that stops the question being re-opened in three months.>

**Where it lives.** <file:line, config key, module — where a reader verifies the decision is real>

**Ledger id + sha.** D-0NN · `<short sha>`

**Validation pending.** <what would still falsify or confirm this, or "none — settled">
```

And on the section it replaces, add one line — nothing else changes:

```markdown
**Superseded by:** D-0NN · <date>
```

---

# Sections

<!-- newest first; written by hand -->

## Open · Exact contact fraction or exponential gaps · 2026-10-04

**What is open.** A model session (`sessions:`) cannot hold both its declared contact fraction and exponential gaps exactly when it holds only a few gaps. Which to keep exact is the other project's owner's decision (Ben's); SensoryForge does not change the session construction until it is taken.

**Status.** Open. Recorded by the `Opens:` trailer of the commit that adds this section (the v1.2.1 documentation fixes).

**Why it is open.** The budgeted layout (D-6b0215f, P8 a) spends the quiet budget Q = max(0, D - E) as G = min(n + 1, max(1, round(Q / gap_mean_ms))) gaps, the spacings of Q at G - 1 sorted uniforms. Exponential gaps conditioned on summing to Q are exactly these spacings, so the fraction is met per session, but each gap is Q times a Beta(1, G - 1) share: CV sqrt((G - 1)/(G + 1)) = 0 / 0.58 / 0.71 / 0.82 for G = 1 / 2 / 3 / 5, never longer than Q. Measured on the reviewer's charter-like world (the other repo's values: contact fraction U[0.55, 0.95], gap_mean_ms 6000, sessions 30-120 s), seed 17, sessions 0-399: G is 1 in 26 % of exploration sessions and 0 in 42 % of handling sessions (whose episodes fill D first); the pooled gaps have mean 5.9 s, median 5.2 s (an exponential of mean 6 s: 4.2 s) and CV 0.72 (an exponential's: 1.0).

**The alternatives.** (a) Exact exponential gaps with the fraction met only on average: a renewal process alternating bouts sized from f with gaps drawn from a registered `exponential` distribution (not built; P8 b, rejected in D-6b0215f for exactly this reason). (b) Longer sessions: the layout unchanged, but more gaps per session make the conditioned gaps closer to exponential (CV 0.905 at G = 10), which the other repo sets through its `duration_ms` and `gap_mean_ms`. (c) Keep the layout as is and state its distribution (the v1.2.1 documentation does this; it is the default until the owner answers).

**Where it lives.** `sensoryforge/world/sampling.py` (`_model_session`); contract section 10.8 ("What the gaps are"); `tests/unit/test_world_session_model.py::test_gaps_are_uniform_spacings_of_the_quiet_budget`.

**Validation pending.** The owner's choice; then a change limited to `_model_session` and its tests (and an `exponential` distribution for (a)).

## D-3a10bce · A continuous registered distribution declares its bounds · 2026-10-04

**What was decided.** a registered distribution without a finite support binds a number field only when it declares bounds=, checked against the field's domain

**Why.** The whole-branch review of v1.2.0 found that the loader checked a registered distribution against its field only through its finite support, so a continuous one was never checked: `biased_direction` (values in [0, 360)) bound to a square grating's `duty` (domain 0.01-0.99) loaded and rendered nonsense duties. `register_distribution(..., bounds=)` declares the interval of the values; `check_axis` checks it like a range against the field's domain (`biased_direction` declares [0, 360]). A support-less distribution with no bounds cannot be checked against a number field, so it may not bind one; bounds on a text field are refused. The same bounds give the load-time shape checks the values such an axis can take.

**What was rejected.** Sampling the distribution at load to guess its range: not exact (a quantile function wrapped modulo 360, like `biased_direction`'s, has its extremes inside (0, 1)). Leaving undeclared distributions unchecked on number fields: it keeps the hole the review found. Built-in worlds are unaffected (v1.1.0's only registered distribution, `braille_cells`, has a finite support); a plugin distribution without a finite support that binds a number field needs `bounds=` from v1.2.1 on (CHANGELOG).

**Where it lives.** `sensoryforge/world/distributions.py` (`Distribution.bounds`, `register_distribution`, `AxisSpec.field_values`), `sensoryforge/world/schema.py` (`check_axis`); `tests/unit/test_world_biased_direction.py`; contract section 10.5.

**Ledger id + sha.** D-3a10bce · `7421092`

**Validation pending.** none, settled.

## D-096e081 · clamp_min's domain is [-1e4, 0] · 2026-10-04

**What was decided.** clamp_min's domain is [-1e4, 0], since a floor above 0 would lift every point in contact to it, a jump at contact onset

**Why.** `clamp_min` accepted any finite value (review of v1.2.0). The floor applies wherever the contact envelope is positive (D-32d7d70), so a positive floor lifts the first instant of every contact from 0 to the floor: the frame jumps at contact onset and offset, which the touch and release ramps exist to prevent. A negative floor lies between no floor (signed shapes may already go negative) and 0, so it is consistent with the layer language. The lower bound mirrors the 1e4 bound of `amplitude` and `background`; the layer's `background` is now checked against its own axis domain [0, 1e4] by the same table.

**What was rejected.** Only 0 (the use the other repo has): it would refuse floors that are harmless. Any finite value: the jump at onset.

**Where it lives.** `sensoryforge/world/kinds.py` (`_LAYER_DOMAIN`, `LayeredKind.normalise_layer`); `tests/unit/test_world_background.py`; contract section 10.1.

**Ledger id + sha.** D-096e081 · `7421092`

**Validation pending.** none, settled.

## D-0c2b6a3 · self_affine components on equal strata of xi · 2026-10-04

**Supersedes:** D-81193aa · 2026-10-04 (its synthesis only; the automatic back-link on D-81193aa also names D-096e081 and D-3a10bce, recorded in the same commit, which do not supersede it)

**What was decided.** self_affine puts its N components on equal strata of xi (q^2 below the roll-off, ln q above), each with its stratum's share of the Persson power; RMS amplitude/sqrt(2) kept (P7 still provisional)

**Why.** v1.2.0 gave each of the N cosines equal power and drew their radial wavenumbers by a stratified inverse CDF of q C(q). The ensemble spectrum was right, but the power of a Persson spectrum sits near the roll-off, so a single surface had almost no fine structure: with 256 cosines, cut-off 0.3 mm and roll-off 10 mm, 1.18 (H = 1) to 7.72 (H = 0.6) cosines per surface had wavelengths of 0.3-1 mm, the band the other repo reads its resolution from. The new layout maps q to xi = (q^2/q0^2 - 1)/2 below the roll-off q0 (equal steps are equal areas of the wavevector plane, the natural measure where the spectrum is flat) and xi = ln(q/q0) above it (equal ratios, the natural measure where it is self-affine); the two meet with equal slope at q0, and the declared radial power is d(xi) below q0 and exp(-2 H xi) d(xi) above. N strata of equal width each hold one cosine with amplitude sqrt(m_j / M) (its stratum's share of the power) at a wavenumber drawn from the declared power within the stratum. The expected spectrum is the declared one exactly, the powers sum to 1/2 in every surface, and every octave between roll-off and cut-off holds at least floor(N ln 2 / W) - 1 cosines in every surface (W = 1/2 + ln(q1/q0)). Derived, not fitted: nothing in it is tuned. Measured (200 seeds, v1.2.0 -> v1.2.1): cosines with wavelengths 0.3-1 mm per surface 1.18 -> 76.9 (H 1, roll-off 10 mm); CV of that band's power per surface 0.326 -> 0.009 from each surface's own table, 0.223 -> 0.127 from a rendered 6.4 mm patch; on the test's placeholder band (0.1-0.3 mm, roll-off 4 mm, cut-off 0.1 mm) 0.084 / 0.169 / 0.808 -> 0.010 / 0.012 / 0.014 at H = 0.6 / 0.8 / 1.0, within the derived bounds 0.065 / 0.074 / 0.084. Ensemble slopes on the reviewer's script: -3.190 / -3.590 / -3.990 against -3.2 / -3.6 / -4.0 (v1.2.0: -3.171 / -3.606 / -4.138).

**What was rejected.** Wavenumbers evenly spaced in log q from q0 with the flat part as its own block: the split of the N cosines between the two pieces would have been a free choice; the C1 scale xi fixes it (a half unit of xi below q0, ln(q1/q0) above). Per-component importance weights at the jittered wavenumber (a(q_j)^2 proportional to q^2 C(q)): unbiased, but the per-surface variance would then vary with the jitter; the stratum's own power keeps it exactly 1/2. More cosines with the old layout: the fine band's share grows only linearly in N.

**Where it lives.** `sensoryforge/world/surfaces.py` (`_components`, `self_affine_table`, `self_affine`); `tests/unit/test_world_self_affine.py`; contract section 10.3.

**Ledger id + sha.** D-0c2b6a3 · `7421092`

**Validation pending.** Ben's confirmation of P7 (the RMS convention), unchanged by this; the synthesis is settled by its known-answer tests.

## D-6b0215f · Sessions are budgeted · 2026-10-04

**What was decided.** sessions are budgeted (P8 a), gap count capped by the episode boundaries

**Status.** Provisional: taken on Ben's behalf by the supervisor on 2026-10-04 (the plan's open question P8, answered as the plan recommends); Ben has not yet confirmed it.

**Why.** The charter world declares a share of each session in contact. Drawing episodes until their contact time reaches f x D and spending the rest as G = min(n + 1, max(1, round(Q / gap_mean_ms))) gaps meets the declared fraction, up to the last episode's overshoot, in every session whose episodes' contact share is at least f (corrected 2026-10-04: when the episodes' own non-contact time, delay lead-ins and pauses between contacts, exceeds 1 - f of their length, the session fills D first and ends below f; with `contact_fraction` 0.97-0.99 on the fixture 199 of 200 sessions fell below target; whether to change the model is open), and keeps each session's record a pure function of (world, seed, index). Gap lengths are the spacings of Q cut at sorted uniforms, near-exponential with mean about gap_mean_ms. This supersedes the v1.1.0 spec's decision 5 (no gap mechanism, no quiet-fraction target) for worlds that declare `sessions:`; other worlds keep it.

**Corrected 2026-10-04 (v1.2.1, the whole-branch review).** Two statements above are wrong; the construction is unchanged. (1) The gaps are not near-exponential: given Q and G, each gap is Q times a Beta(1, G - 1) share (coefficient of variation sqrt((G - 1)/(G + 1)), none longer than Q), i.e. exponential gaps conditioned on summing to Q. With the other repo's values (contact fraction U[0.55, 0.95], gap_mean_ms 6000, sessions 30-120 s; 400 sessions) the gaps' CV is 0.72 (an exponential's: 1.0), their median 5.2 s, no gap is longer than its session's quiet budget, and 42 % of the long-episode sessions have no gap at all. (2) "episodes' contact share at least f" is not enough for the fraction: a session whose last episode runs past D is cut there (charter-like session 586: f 0.895, episodes' share 0.900, share within D 0.863). The guarantee holds in sessions whose episodes end before D (contract guarantee 11, reworded). See "Open · Exact contact fraction or exponential gaps" above.

**What was rejected.** (b) A renewal process (alternating bouts and exponential gaps): gaps exactly exponential but the fraction holds only on average, and it needs a new `exponential` distribution.

**Where it lives.** `sensoryforge/world/sampling.py` (`_model_session`), `sensoryforge/world/schema.py` (`SessionModel`); contract section 10.8; test 11.

**Ledger id + sha.** D-6b0215f · `771bd06`

**Validation pending.** Ben's confirmation of the provisional answer; changing it changes only the named element.

## D-80d1fcd · Groups are resolved into class axes at load · 2026-10-04

**What was decided.** groups are resolved into class axes at load (P1 option A), not kept as a runtime concept

**Status.** Provisional: taken on Ben's behalf by the supervisor on 2026-10-04 (the plan's open question P1, answered as the plan recommends); Ben has not yet confirmed it.

**Why.** One class per feature x contact type keeps a class a single stratum, so the other repo reports per contact type and per feature without extra machinery; `groups:` / `use:` only saves typing. Resolving at load means `World.to_dict()` has no `groups` or `use` key and draws and ids equal those of the same world written out.

**What was rejected.** (B) Class variants (a categorical variant axis choosing a group of axes per draw): larger, with sampling, strata per variant and entry ids to change.

**Where it lives.** `sensoryforge/world/schema.py`; `tests/unit/test_world_groups.py`; contract section 10.7.

**Ledger id + sha.** D-80d1fcd · `2487d37`

**Validation pending.** Ben's confirmation of the provisional answer; changing it changes only the named element.

## D-7d5e5bc · letter_text recovers the draw's bits · 2026-10-04

**What was decided.** letter_text recovers int(u * 2**53) as the draw's bits and maps cells x lines sub-uniforms through the cumulative weights

**Status.** Not one of the plan's open questions P1 to P9; an implementation choice made in the plan's Task 9.

**Why.** A registered distribution receives one uniform per draw. The draw's 53 random bits are exactly recoverable from that uniform, and expand to one sub-uniform per letter, so a multi-line text depends only on the draw. Several letters per draw have no finite support, so the axis needs `stratify: false`.

**What was rejected.** Giving distributions several uniforms per draw: it would change `rng` and every draw of every world.

**Where it lives.** `sensoryforge/world/distributions.py` (`_sample_letter_text`); `tests/unit/test_world_braille_lines.py`; contract section 10.6.

**Ledger id + sha.** D-7d5e5bc · `5d658d3`

**Validation pending.** none, settled.

## D-fda02f9 · step_edge is an indenter shape · 2026-10-04

**What was decided.** step_edge is an indenter shape, value d on the plate, d minus the circular sag on the shoulder, 0 past it

**Status.** Provisional: taken on Ben's behalf by the supervisor on 2026-10-04 (the plan's open question P5, answered as the plan recommends); Ben has not yet confirmed it.

**Why.** A step with a rounded shoulder is a rigid plate pressed to depth d, so it follows the same depth-driven rule as `curved_contact` (P5): the shoulder footprint grows during the rise and depth 0 renders exactly 0.

**What was rejected.** A separable shoulder shape scaled by the envelope: it would misstate the contact area during ramps.

**Where it lives.** `sensoryforge/world/surfaces.py` (`step_edge`); `tests/unit/test_world_indenters.py`; contract section 10.2.

**Ledger id + sha.** D-fda02f9 · `5cfab52`

**Validation pending.** Ben's confirmation of the provisional answer; changing it changes only the named element.

## D-84200e4 · Indenters are depth-driven · 2026-10-04

**What was decided.** indenters are depth-driven, the footprint growing during the rise (P5, provisional)

**Status.** Provisional: taken on Ben's behalf by the supervisor on 2026-10-04 (the plan's open question P5, answered as the plan recommends); Ben has not yet confirmed it.

**Why.** `ShapeKind.indenter` evaluates the shape at depth amplitude x envelope x modulation and takes its value as the indentation; the footprint is max(0, d - sag / unit_mm) and grows during a press. Both elements are defined by physical radii, and a fixed footprint scaled by the envelope would misstate the contact area during ramps.

**What was rejected.** (b) A separable shape with a fixed footprint at peak depth, like every other shape. Concave surfaces (the survey's 20 to 40 mm) are not built: convex sphere and cylinder only.

**Where it lives.** `sensoryforge/world/kernel.py` (`ShapeKind.indenter`), `sensoryforge/world/kinds.py`, `sensoryforge/stimuli/layered.py` (`_render_indenter`), `sensoryforge/world/surfaces.py`; contract section 10.2.

**Ledger id + sha.** D-84200e4 · `956af6c`

**Validation pending.** Ben's confirmation of the provisional answer; changing it changes only the named element.

## D-072fed4 · Dot arrays on a square or hexagonal lattice · 2026-10-04

**What was decided.** dot arrays on a square or hexagonal lattice of Gaussian bumps (P9, provisional)

**Status.** Provisional: taken on Ben's behalf by the supervisor on 2026-10-04 (the plan's open question P9, answered as the plan recommends); Ben has not yet confirmed it.

**Why.** Each point sums the Gaussian bumps of the lattice sites near it (as a `grid` of `gaussian`s would), so the array fills the patch and is translated by x_mm/y_mm. The neighbour loop runs to the batch's maximum with surplus terms masked to exact zeros, so a draw renders the same alone or in a batch.

**What was rejected.** Another bump profile: not asked for.

**Where it lives.** `sensoryforge/world/surfaces.py` (`dot_array`); `tests/unit/test_world_dot_array.py`; contract section 10.3.

**Ledger id + sha.** D-072fed4 · `7e16d00`

**Validation pending.** Ben's confirmation of the provisional answer; changing it changes only the named element.

## D-81193aa · Self-affine textures use the Persson spectrum · 2026-10-04

**What was decided.** self-affine textures use the Persson spectrum from cosines, RMS amplitude/sqrt(2) (P7, provisional)

**Status.** Provisional: taken on Ben's behalf by the supervisor on 2026-10-04 (the plan's open question P7, answered as the plan recommends); Ben has not yet confirmed it.

**Why.** Persson's isotropic spectrum (flat below the roll-off, q^-2(H+1) to the cut-off) as a sum of `components` cosines with radial wavenumbers by stratified inverse CDF, directions and phases from the shape's own seed: zero mean, a pure function of position, the same on any canvas. RMS = amplitude/sqrt(2), the RMS of a unit-peak signed sinusoid, so a self-affine and a periodic texture of equal amplitude carry equal power.

**What was rejected.** RMS = amplitude, or 3 x RMS = amplitude (the other scales offered). A two-dimensional weave (`plaid`) is not built.

**Where it lives.** `sensoryforge/world/surfaces.py` (`self_affine`); `tests/unit/test_world_self_affine.py`; contract section 10.3.

**Ledger id + sha.** D-81193aa · `f276323`

**Validation pending.** Ben's confirmation of the provisional answer; changing it changes only the named element.

**Superseded by:** D-0c2b6a3, D-096e081, D-3a10bce (2026-10-04, 7421092) — self_affine puts its N components on equal strata of xi (q^2 below the roll-off, ln q above), each with its stratum's s…

## D-b0be6a6 · A press falls as it rose through same_as · 2026-10-04

**What was decided.** a press falls as it rose through same_as, an axis link (P3a, provisional)

**Status.** Provisional: taken on Ben's behalf by the supervisor on 2026-10-04 (the plan's open question P3, answered as the plan recommends); Ben has not yet confirmed it.

**Why.** The survey says the fall mirrors the rise. `{same_as: touch_ms}` makes an axis copy another of the same draw; links are validated at load, refused in fixed draws and filled after the other axes in every sampling path.

**What was rejected.** (b) An independent draw of the fall from the same range: expressible today but not what the survey says.

**Where it lives.** `sensoryforge/world/sampling.py`, `sensoryforge/world/distributions.py` (`fill_links`), `sensoryforge/world/schema.py`; contract section 10.4.

**Ledger id + sha.** D-b0be6a6 · `62cb959`

**Validation pending.** Ben's confirmation of the provisional answer; changing it changes only the named element.

## D-1b2d1c1 · Lateral scan bias is an angular central Gaussian · 2026-10-04

**What was decided.** lateral scan bias is an angular central Gaussian solved from the travel ratio (P4a, provisional)

**Status.** Provisional: taken on Ben's behalf by the supervisor on 2026-10-04 (the plan's open question P4, answered as the plan recommends); Ben has not yet confirmed it.

**Why.** theta = axis + atan2(sin 2 pi u, s cos 2 pi u): one uniform per draw, monotone in u (so it stratifies into equal-probability bins), with the stretch s solved in closed form from the declared travel ratio. A 2e6-point quadrature of E|cos| / E|sin| at R = 2.5 gave 2.5000000 (s = 3.876253), run for this task.

**What was rejected.** (b) An axial von Mises with kappa solved for the ratio (no closed inverse CDF); (c) a mixture of lateral sweeps and uniform directions.

**Where it lives.** `sensoryforge/world/distributions.py` (`biased_direction`); `tests/unit/test_world_biased_direction.py`; contract section 10.5.

**Ledger id + sha.** D-1b2d1c1 · `19d7a32`

**Validation pending.** Ben's confirmation of the provisional answer; changing it changes only the named element.

## D-32d7d70 · The background is its own draw sharing the contact envelope · 2026-10-04

**What was decided.** the background is its own draw sharing the contact envelope, with the layer total floored at clamp_min where the envelope is positive (P6, provisional).

**Status.** Provisional: taken on Ben's behalf by the supervisor on 2026-10-04 (the plan's open question P6, answered as the plan recommends); Ben has not yet confirmed it.

**Why.** Every contact in the charter world rests on a background level drawn from the feature range, with features and zero-mean relief riding on it. The background rises and falls with the contact envelope and modulation, is not moved by motion, and the floor keeps relief from dipping below zero (the skin leaves contact; no negative indentation).

**What was rejected.** No floor, with the other repo keeping relief amplitude below the background: not expressible, the schema has no joint constraints.

**Where it lives.** `sensoryforge/stimuli/layered.py`, `sensoryforge/world/kinds.py`; `tests/unit/test_world_background.py`; contract section 10.1.

**Ledger id + sha.** D-32d7d70 · `6c0daf3`

**Validation pending.** Ben's confirmation of the provisional answer; changing it changes only the named element.

## P2 · "touches for" is the plateau hold_ms · 2026-10-04

**What was decided.** "Touches for" in the charter world is the plateau `hold_ms` of an episode (rise, touch, fall expressed as `touch_ms`, `hold_ms`, `release_ms`); no `contact_ms` field is added.

**Status.** Provisional: taken on Ben's behalf by the supervisor on 2026-10-04 (the plan's open question P2, answered as the plan recommends); Ben has not yet confirmed it.

**Why.** The survey's wording reads rise, touch and fall as three phases, which v1.1.0 already expresses; nothing new in SensoryForge is needed.

**What was rejected.** (b) A total contact time `contact_ms` with the hold derived as max(0, contact - touch - slide - release): one more episode field, and taps whose rise plus fall exceeds the drawn contact would lose their plateau.

**Where it lives.** Nothing in code: the other repo's world file declares `hold_ms`. Contract section 10.11.

**Ledger id + sha.** recorded by the `Decision:` trailer of the commit that adds this section (see the log below)

**Validation pending.** Ben's confirmation of the provisional answer.

<!-- SECTIONS_START -->

## D-0fbb84a · Every population gets its own noise seed in a batch run · 2026-10-02

**Supersedes:** 32-bit run seed, 53-bit noise seeds · 2026-10-02

**What was decided.** the batch runner sets every population's noise_seed to seed53(noise, "population", i), whether or not the design set one, so no population's noise hangs on the 32-bit run seed

**Why.** `SimulationEngine.run` seeds numpy's legacy global generator with its `seed` argument,
which rejects values at or above 2**32 (F-202cdd8), so the runner passes `noise & 0xFFFFFFFF`.
That run seed also seeds torch's global generator, which draws the sensor and membrane noise of
any population without a `noise_seed` of its own. 32-bit seeds collide: the chance that two
entries share one is about 1% at 10^4 entries and 69% at 10^5 (1 - exp(-n^2 / 2^33)), so two
entries of a large data set could get the same noise realisation. The superseded section set a
population's seed only when the design set one, which left every design without population seeds
(the test fixture's among them) on the 32-bit seed. The engine uses a population's `noise_seed`
only when that population has sensor or membrane noise, so setting it always changes nothing for
a noiseless population. The receptor noise already took the full 53-bit seed
(`simulation.receptor_noise_seed`).

**What was rejected.** Keeping "when the design sets one" (the collisions above); seeding the
engine's global generators from 53 bits (a change to the engine's own seeding, outside the world
engine, that would change every seeded run).

**Where it lives.** `sensoryforge/world/runner.py` (`run_dataset`, the per-entry seed block; the
run seed is still `noise & 0xFFFFFFFF`); spec section 7.2 step 2; contract section 6;
`tests/integration/test_world_batch.py::test_batch_writes_one_bundle_per_entry`.

**Ledger id + sha.** D-0fbb84a · `aa88bd3`. The finding it rests on is F-202cdd8 · `f80d6d4`; the
superseded section's "No ledger id" missed it.

**Validation pending.** whether the engine should one day seed its global generators from the full
53 bits (it would change every seeded run).

## D-bed3c30 · A world's classes are weighed in order of their names · 2026-10-02

**What was decided.** a world's classes are weighed and iterated in order of their names, so no draw, data-set entry or entry order depends on the order a world file writes its classes, axes or defaults in

**Why.** `world_id` hashes the normalised world with sorted keys, but `sample` built its cumulative
class weights in declared order. Reordering a world's classes, which `yaml.safe_dump` does by
sorting keys, changed every declared draw while `world_id`, and with it a data set's `world_id`
pin, stayed the same: the pin vouched for draws the world no longer gave. On the fixture world,
reversing every mapping's key order left `w-cba7a3063f16` unchanged and changed the class each
draw index picked. The data-set builder also looped over classes in declared order (stratified
test, held-out and probe splits), so entry order followed the file too.

**What was rejected.** Hashing the declared order into `world_id` (two files with the same content
would get different ids, and a `safe_dump` round trip would break every pin); keeping declared
order and documenting it (re-serialising a world is common, and the failure is silent).

**Where it lives.** `sensoryforge/world/sampling.py` (`class_pool`: sorted by name; an empty
`classes` list and a repeated name fail); `sensoryforge/world/schema.py` (classes, held-out
classes, fixed draws, axes and defaults parsed in name order; two names of one source that bind
one field fail); `sensoryforge/world/dataset.py` (`_by_name`);
`tests/unit/test_world_sampling.py::test_class_order_changes_neither_the_id_nor_the_draws`,
`tests/unit/test_world_dataset.py::test_class_order_does_not_change_the_entries`; the draws
themselves are pinned by `test_world_sampling.py::test_the_fixture_worlds_draws_are_pinned`.

**Ledger id + sha.** D-bed3c30 · `e6bf7f2`

**Validation pending.** none — settled. It changed every declared draw of every world before
v1.1.0 was tagged; nothing had pinned a draw yet, so the format tags stayed at `/1`.

## D-46a52e6 · The world engine ships as 1.1.0, tagged on its branch · 2026-10-02

**What was decided.** the world engine ships as version 1.1.0, tagged v1.1.0 on its branch, with main and the ~/sensoryforge checkout left untouched until Ben merges

**Why.** pressure-simulation's `bio-encoding` environment runs `~/sensoryforge` as an editable
install, and another session simulates from it; a merge into `main` there changes what they import
mid-work. A tag lets pressure-simulation pin a sha and run `tests/contract/test_world_contract.py`
against it without touching `main`. Ben merges once pressure-simulation's Phase 1b no longer
simulates from `~/sensoryforge`.

**What was rejected.** Merging to `main` first (changes a live editable install under a running
session); shipping unversioned (nothing for pressure-simulation to pin).

**Where it lives.** `pyproject.toml` and `sensoryforge/__init__.py` (`1.1.0`), `CHANGELOG.md`,
`docs/reference/world_contract.md` ("Getting it"); the tag itself is made by the controller after
review.

**Ledger id + sha.** D-46a52e6 · `e766b8a`

**Validation pending.** pressure-simulation's Phase 2b installing the tag and running the contract
tests; the merge to `main`, which is Ben's.

## D-d3605dd · Data sets offer both repeat semantics · 2026-10-02

**What was decided.** data sets offer both repeat semantics -- `repeats` (fresh draws and noise per replicate) and `noise_repeats` (same draws, k noise realisations)

**Why.** They answer different questions: `repeats` enlarges the sample, `noise_repeats` measures
how much of an error is noise on a fixed stimulus. Entry ids keep them apart (`train/r1/...`,
`validation/r0/00003.n1`).

**What was rejected.** Offering only one of them: with `repeats` alone the noise on a fixed stimulus
cannot be separated from draw variation; with `noise_repeats` alone a data set cannot grow.

**Where it lives.** `sensoryforge/world/dataset.py` (`SplitSpec.noise_repeats`, `repeats`,
`build_dataset`); `tests/unit/test_world_dataset.py`.

**Ledger id + sha.** D-d3605dd · `e766b8a`

**Validation pending.** none — settled (the entry ids and seeds are pinned by contract test 5).

## D-cccbd6a · A separate renderer, pinned by tests · 2026-10-02

**What was decided.** the world engine is a new sensoryforge.world package with its own vectorised renderer kept equal to `layered` by tests; `layered` gains only additive default-off fields and the old sweep BatchExecutor stays untouched (rewriting `layered` on the new kernel, or extending BatchExecutor, were rejected)

**Why.** The world renderer must evaluate thousands of draws at arbitrary times on arbitrary
coordinates in float64, which `layered`'s frame-by-frame float32 path does not do. The two
renderers are held together by `tests/unit/test_world_render.py::test_world_render_equals_layered`
and `tests/unit/test_layered_golden.py` (old layered stimuli render as before).

**What was rejected.** Rewriting `layered` on the new kernel: it risks last-bit changes to every
layered render pressure-simulation's benchmarks use. Extending `BatchExecutor`: it brings its
timestamped roots, checkpoint races, `--device` bug and exit-0-on-failure.

**Where it lives.** `sensoryforge/world/render.py`, `kernel.py`, `runner.py`;
`sensoryforge/stimuli/layered.py` and `episode.py`; `.claude/rules/world-engine.md`.

**Ledger id + sha.** D-cccbd6a · `e766b8a`

**Validation pending.** none — settled; the equality tests run in every CI pass.

## D-37c5247 · Quiet comes from the draws; a session is draws end to end · 2026-10-02

**What was decided.** quiet comes from every draw's lead-in and tail, a `quiet` class and pauses between contacts; a session is draws laid end to end with no separate gap mechanism or quiet-fraction target

**Why.** The quiet share then follows from the world's own declaration and its class weights, and a
session needs no parameters beyond a duration. `quiet_fraction` is reported, not set.

**What was rejected.** A gap mechanism with a quiet-fraction target: a second knob that fights the
class weights.

**Where it lives.** `sensoryforge/world/sampling.py` (`Session`, `session`, `quiet_fraction`),
`sensoryforge/world/kinds.py` (`QuietKind`); contract test 7.

**Ledger id + sha.** D-37c5247 · `e766b8a`

**Validation pending.** whether pressure-simulation's Phase 2b needs a controllable quiet share;
if so it is a new decision, not a change to this one.

## D-99d39b6 · An episode is touch, hold, slide, release, with contacts · 2026-10-02

**What was decided.** one episode is a quiet lead-in then touch, hold, slide, release, optionally repeated as several contacts separated by pauses; motion happens only during slides; temporal frequency is a layer modulation (sine vibration, pulses for repeated indentation)

**Why.** These are the phases a tactile stimulus has (an indentation, a held contact, a drag, a
lift), and repeated contacts with pauses give re-touch and tapping without a second mechanism.
Vibration and repeated indentation multiply the envelope instead of being new shapes, so they
combine with every shape. Motion only while sliding keeps a held contact still, which is what
"hold" means.

**What was rejected.** Vibration and taps as separate shapes (each shape would need its own copy);
motion over the whole episode (a held contact would drift).

**Where it lives.** `sensoryforge/stimuli/episode.py` (the timing math),
`sensoryforge/stimuli/layered.py` (`slide_ms`, `contacts`, `pause_ms`, `modulation`),
`sensoryforge/world/kernel.py`; `tests/unit/test_layered_episode.py`.

**Ledger id + sha.** D-99d39b6 · `e766b8a`

**Validation pending.** none — settled for 1.1.0.

## D-94c08c7 · A world class is a layered layer with random fields · 2026-10-02

**What was decided.** a world class is a `layered` layer with random fields (axes bind to its shape, pattern, modulation and episode fields), so a draw is an ordinary layered stimulus and SensoryForge keeps one stimulus language

**Why.** A draw resolves to `Draw.to_layer()`, which the GUI and `sensoryforge run` already take,
and every shape, pattern and motion added to layered is available to worlds.

**What was rejected.** A world-specific vocabulary (two languages to keep equal and document), and
arbitrary registered stimulus types (they have no common time course, so no common episode or
axes).

**Where it lives.** `sensoryforge/world/schema.py`, `kinds.py` (`LayeredKind`),
`sampling.py` (`Draw.to_layer`).

**Ledger id + sha.** D-94c08c7 · `e766b8a`

**Validation pending.** none — settled.

## F-963499b · Sessions render per draw, within the memory budget · 2026-10-02

**What was decided.** a session's draws are each evaluated only at the times inside their own
window, and every internal evaluation `[g, k, *S]` (draws x times x canvas) is bounded by
`max_elements`, blocked along time as well as over draws. The ledger records this as a finding:
"sessions evaluated every draw over every session frame (J*K*S work and memory) and long movies
ignored the element budget along time".

**Why.** Measured: a 10 s session went from about 11 s and 1.5 GB to 0.17 s. The output
`[n, K, ...]` is still allocated whole and sized by the caller.

**What was rejected.** Leaving sessions as they were with a smaller `max_elements`: it bounds the
chunk but not the work, which was still draws x frames.

**Where it lives.** `sensoryforge/world/render.py` (`render`, `time_block`/`per_chunk`);
`docs/reference/world_contract.md` section 4; `tests/unit/test_world_render.py`.

**Ledger id + sha.** F-963499b · `0c2854a`

**Validation pending.** none — settled (the benchmark in `docs/reference/benchmarks.md`, "World
renderer", records the speed).

## 32-bit run seed, 53-bit noise seeds · 2026-10-02

**Superseded by:** D-0fbb84a · 2026-10-02

No ledger id: recorded here only.

**What was decided.** The batch runner passes `seed = noise & 0xFFFFFFFF`, the low 32 bits of the
entry's noise seed, to `SimulationEngine.run`; the full 53-bit noise seed sets
`simulation.receptor_noise_seed`, and a design population's `noise_seed` becomes
`seed53(noise, "population", i)`.

**Why.** The engine seeds numpy's legacy global generator, which accepts only 32-bit seeds, so the
run seed has to be truncated; the receptor and population generators take their full seeds from
the config. Both are recorded in the bundle's `config.json`
(`config.simulation.receptor_noise_seed`, `config.populations[i].noise_seed`).

**What was rejected.** Shrinking every seed to 32 bits, which would break the 53-bit JSON-safe
seeds the manifest and the contract use.

**Where it lives.** `sensoryforge/world/runner.py` (`run_dataset`, the `seed=noise & 0xFFFFFFFF`
line); contract section 6; `test_4_noise_seeds` checks the recorded receptor seed equals the
entry's noise seed.

**Ledger id + sha.** none · `f80d6d4`

**Validation pending.** whether the engine should one day seed from the full 53 bits (a change to
its generators; it would change every seeded run).

## Layered equality is 1e-5, not 1e-6 · 2026-10-02

No ledger id: recorded here only.

**What was decided.** The world renderer equals the `layered` render of `draw.to_layer()` to 1e-5.
The spec first asked for 1e-6.

**Why.** `layered` keeps time in float32, so its frames differ from the float64 world render by
float32 rounding of the time axis; 1e-6 is below that.

**What was rejected.** Moving `layered` to float64 time to meet 1e-6: it risks last-bit changes to
every existing layered render (see D-cccbd6a).

**Where it lives.** `tests/unit/test_world_render.py::test_world_render_equals_layered`;
`docs/reference/world_contract.md` section 8; the implementation plan (commit `8572d5d`).

**Ledger id + sha.** none · `8572d5d`

**Validation pending.** none — settled.

## D-c5facd5 · RA is a signed level-crossing unit · 2026-09-30

**What was decided.** RA is a signed level-crossing unit, event-camera style: it emits an ON event when its input has risen by a threshold theta since its last event and an OFF event when it has fallen by theta, moving its reference by theta each time; the sign is carried by the event, not recovered later (SensoryForge neuron_model level_crossing, opt-in; AdEx and the rectified RA filter stay as the reference arm)

**Why.** Ben's decision (dated 2026-10-01 in the brief; committed 2026-09-30). The reference RA arm
(rectified derivative filter + AdEx) throws the sign of the change away and makes the decoder
recover it; D-4e669b4 already records that RA answers release as strongly as indentation. A
level-crossing unit keeps the sign in the event itself, and the running sum of events times theta
reconstructs the drive to within one theta, so the encoding is invertible up to a quantum.
Measured on this commit (`tests/unit/test_event_encoders.py`): a ramp of slope s gives ON events at
s/theta (0.019/ms vs 0.020 over a 1 s ramp: the last partial quantum) and none on a hold; a falling
ramp gives OFF events at the same rate; max |drive - sum*theta| = 0.9999 theta on a signed
multi-scale drive; white noise gives 0 / 1e-6 / 1.8e-3 events per sample at sigma/theta = 0.1 / 0.2
/ 0.3 (2.5e-2 at 0.4, 8.7e-2 at 0.5); a 2 ms refractory period gives exactly 0.5 events/ms.

**Hardware.** A reference register, a comparator and a sign bit: FPGA-trivial, one cycle per
sample; AER carries the polarity bit natively.

**What was rejected.** (1) Recovering the sign later from the rectified RA filter (the current arm):
the information is gone before the neuron. (2) Two unsigned ON/OFF populations: doubles the
population and the wiring for what one sign bit carries. (3) Feeding the level-crossing unit the
RA filter's output: that differentiates twice and discards the OFF half, so the unit runs with
`filter_method: none` and its input is not floored at 0 mA. (4) Storing the signed counts in the
bundle's `spikes` dataset: an existing reader would count -1 as a spike (or none); they go to a
separate `events` dataset (bundle schema 2.1.0) and the engine key `events`.

**Where it lives.** `sensoryforge/neurons/event_encoders.py` (`LevelCrossingNeuron`, registered
`level_crossing`); `config/defaults.py` (`UNFLOORED_NEURON_MODELS`); `core/simulation_engine.py`
(`SIGNED_EVENTS` -> key `events`); `io/bundle.py` (schema 2.1.0); `io/design.py`;
`docs/user_guide/event_encoders.md`, `bundles.md`.

**Ledger id + sha.** D-c5facd5 · `6124f87`

**Validation pending.** Decoding: pressure-simulation's SGA-KF must consume signed events (its
bundle reader currently requires `spikes` and fails loudly on an events population, by design).
Choosing theta against the RA drive's noise floor (the sigma/theta table above) for a real design.

## D-f0e2433 · SA gets a sigma-delta arm · 2026-09-30

**What was decided.** SA gets a sigma-delta arm: a non-leaky integrate-and-fire unit with subtractive reset whose spike rate is linear in its input level; what matters is the time-scale separation (RA = fast signed change, SA = slow absolute level), not the biological mechanism (SensoryForge neuron_model sigma_delta, opt-in)

**Why.** Ben's decision (dated 2026-10-01 in the brief; committed 2026-09-30). The design only
needs SA to report the slow absolute level linearly; a subtractive-reset integrator discards no
charge, so its count is the integral of the drive over theta and its rate is exactly drive/theta
with no rheobase (unlike the AdEx arm, whose rheobase D-8dde454/D-673e0ed had to calibrate gains
around). Measured on this commit: constant drives 0.2-20 mA at theta = 10 mA*ms give rate =
drive/theta to better than 1e-3, and 0.01 mA still fires at 0.001/ms; a boxcar low-pass of the
spikes recovers a 2 Hz, 1-9 mA sinusoid with quantisation RMS 0.85 / 0.39 / 0.20 / 0.083 / 0.041 mA
at 5 / 10 / 20 / 50 / 100 ms windows (about theta/window; best against the raw drive 0.095 mA at
50 ms); the quantisation error's power is 4.3e-3 / 12.6 / 4.4e3 / 4.8e3 in the 1-10 / 10-100 /
100-1000 / 1000-5000 Hz bands, i.e. first-order noise-shaped (high-pass).

**Hardware.** An accumulator, a comparator and a subtractor: FPGA-trivial, one cycle per sample.

**What was rejected.** (1) A leaky IF by default: the leak adds a rheobase theta/tau and breaks
linearity; kept as the opt-in `leak_tau_ms`. (2) Reset-to-zero: discards the residual charge each
spike, so the rate is no longer exactly linear and the error is no longer noise-shaped.
(3) Replacing the AdEx SA arm: it stays the default reference arm; nothing existing changes.

**Where it lives.** `sensoryforge/neurons/event_encoders.py` (`SigmaDeltaNeuron`, registered
`sigma_delta`); ordinary `spikes` in the engine and the bundle; `io/design.py`;
`docs/user_guide/event_encoders.md`.

**Ledger id + sha.** D-f0e2433 · `6124f87`

**Validation pending.** Choosing theta per design (rate budget vs. quantisation error, via the
window table above), and pressure-simulation's decoder reading sigma-delta rates.

## D-802e42b · Sensor noise lives in the sensors, before innervation · 2026-10-01

**What was decided.** Sensor noise lives in the sensors: one noise source per receptor, in pressure (stimulus) units, added to the stimulus on the receptor grid before innervation, shared by every population; pooling, filtering and gain then propagate it. The existing per-population current noise (sensor_noise_std, added after filter and gain) is a different thing -- neuron input noise -- and stays as its own option.

**Why.** Ben's decision, 2026-10-01. D-5cdc524 called `sensor_noise_std` "sensor (receptor-current)
noise", but it is added per population after the filter and the gain, so two populations reading
the same receptors got independent noise, the noise was not shaped by the receptive field, and its
units were mA rather than the pressure a receptor measures. SimulationEngine never applied the legacy
pipeline's `ReceptorNoiseTorch`. A physical sensor's noise is in the receptor, before any pooling:
SA and RA pooling the same receptors must see the same realisation, and a neuron's input noise must
scale with its weights. Measured on this commit (`tests/unit/test_receptor_noise.py`): with
one_to_one wiring and no filter the recorded current's noise is σ·gain (ratio 1.002); per neuron,
pooled input noise / (σ·‖w‖₂) = 1.000 for gaussian banks (K = 12 and 20, max deviation 1.6 %) and
1.003 / 1.000 for the imported design banks (sa / ra); the SA-RA input-noise cross-covariance matches
σ² W_sa W_raᵀ (correlation error < 0.05). For unequal weights ‖w‖₂ is below √K·mean(w)
(0.84 and 0.77 of it at K = 12 and 20), so √K is only the equal-weight special case.

**What was rejected.** (1) Renaming `sensor_noise_std`: every config, bundle and pressure-simulation
export (`noise_std` → `sensor_noise_std`) would change meaning; it keeps its name and is described as
the neuron input (current) noise in docstrings, docs and the GUI label. (2) Noise on the stimulus
pixel frame before sampling: on a non-grid arrangement bilinear sampling would smear one pixel's
noise over several receptors, which is not one source per receptor. (3) A per-population receptor
noise key: the noise belongs to the sensor sheet, so it is a single SimulationConfig field.

**Where it lives.** `sensoryforge/config/schema.py` (`SimulationConfig.receptor_noise_std`,
`receptor_noise_seed`, `effective_receptor_noise_std()`); `sensoryforge/core/simulation_engine.py`
(`run`, `_add_receptor_noise`: drawn once per (grid, channel), cached, seed
`receptor_noise_seed + 7919·grid_index + channel_index`); `sensoryforge/io/design.py` (`load_design`
reads the design's top-level keys); `docs/user_guide/bundles.md`, `configuration_schema.md`,
`units_and_gains.md`; `tests/unit/test_receptor_noise.py`. Unset or 0 is bit-identical to before
(`tests/integration/test_noise_split_golden.py` still passes).

**Ledger id + sha.** D-802e42b · `9a3f550`

**Validation pending.** pressure-simulation must export a top-level `receptor_noise_std` (in its
stimulus units) in `design.json` for designs to run with receptor noise; until then a design runs
with none. Whether pressure-simulation's declared per-population `noise_std` (the current noise it
derived from its receptor floor) should then drop to 0 is its decision.

## D-5cdc524 · Sensor noise and membrane noise are separate; a design's declared noise is simulated · 2026-09-30

**Superseded in part by:** D-802e42b · 2026-10-01 (the naming only: sensor noise is the receptor noise before innervation; `sensor_noise_std` is the neuron input (current) noise)

**What was decided.** Sensor (receptor-current) noise and membrane noise are separate SensoryForge parameters; a design directory's declared noise_std is simulated as sensor noise and membrane_noise_std as membrane noise

**Why.** pressure-simulation now declares its sensor noise as a specification (its DECISIONS §110):
each population record in `design.json` carries `noise_std` (mA, the declared design-unit floor
converted to the injected current at the design's gains; 3.59 mA SA and 37.9 mA RA on the default
40x40 design), `noise_seed`, and a separate `membrane_noise_std` (default 0). Two things stopped
SensoryForge from running that (pressure-simulation's C-130): `load_design` ignored the noise keys, so
every `run --design` was noise-free; and `PopulationConfig.noise_std` was the only noise key, used
both as the post-gain current noise (`_run_pop_from_drive`) and as the neuron's Langevin noise
(`build_neuron`), so "sensor noise on, membrane noise off" could not be expressed. The two are
different physical things: one is the receptor's noise, which the design's information numbers
assume and which shows up in the recorded `filtered` current; the other belongs to the spiking
stage and never appears in `filtered`.

**What was rejected.** (1) Keeping one key and accepting the coupling for noisy runs: a declared
membrane noise of 0 would then be silently replaced by 3.59 mA-equivalent Langevin noise on SA.
(2) Renaming `noise_std` outright: every existing config, bundle and preset override would change
meaning or break. It stays as a deprecated alias that sets both (a `FutureWarning` when non-zero;
an explicit new key overrides it), and `tests/integration/test_noise_split_golden.py` pins every
preset and two alias configs to spikes recorded with the pre-change code.

**Where it lives.** `sensoryforge/config/schema.py` (`PopulationConfig.sensor_noise_std`,
`membrane_noise_std`, `effective_sensor_noise_std()`, `effective_membrane_noise_std()`);
`sensoryforge/core/simulation_engine.py` (`build_neuron`, `run`); `sensoryforge/io/design.py`
(`load_design`); `docs/user_guide/bundles.md` § "The design's declared noise is simulated";
`tests/unit/test_sensor_noise_split.py`.

**Ledger id + sha.** D-5cdc524 · `7f7c6a3`

**Validation pending.** pressure-simulation's own sensor-noise parity test
(`tests/design/test_sensor_noise.py`) pins the old coupling and must be updated when its pin moves
to this commit; its `design.export.sensoryforge_config` workaround becomes redundant.

## D-673e0ed · The non-adapting AdEx RA also gets the low-threshold gain · 2026-09-28

**What was decided.** the AdEx recipe's RA gain is set like its SA gain -- RA fires from about 10% of the benchmark stimulus's rate of pressure change (10x the lowest gain at which the held benchmark's ramp makes RA fire) -- instead of matching TouchSim's RA sensitivity, so RA's rate stays proportional to the rate of change over the benchmark range; the Izhikevich recipe keeps the TouchSim-matched RA gain

**Why.** SA's low-threshold gain (D-8dde454) made SA sensitive, so the amplitude per mm fitted against TouchSim shrank to 0.21. Matching TouchSim's RA sensitivity relative to SA then pushed AdEx RA's gain to 740. There its onset rate over the benchmark ramp speeds (5-40 units/s) is compressed to 149-320 Hz near the refractory ceiling (R^2 0.81 against speed). At gains 100-200 it rises proportionally (R^2 0.94). pressure-simulation needs RA proportional to the rate of pressure change (D-6bae4df), and firing from 10% of the benchmark's rate still detects small movements.

**What was rejected.** The TouchSim-matched gain (740), for the compression above.

**Where it lives.** `scripts/calibrate_recipe_gains.py` (`RA_RULES`, `choose_ra_gain_low_threshold`); `sensoryforge/presets/tactile_sa1_ra1_adex.yml`.

**Ledger id + sha.** D-673e0ed · `df3320e` (decision)

**Validation pending.** How well pressure-simulation's Kalman filter decodes with these gains.

## D-8dde454 · Izhikevich keeps its adaptation; the non-adapting AdEx SA gets a low threshold · 2026-09-28

**What was decided.** the Izhikevich recipe keeps its TouchSim-fitted spike-frequency adaptation (d 15 for SA, 24 for RA), because without it SA is too steep to be graded; the AdEx recipe, pressure-simulation's model, stays non-adapting, with SA's gain set so SA fires from about 10% of the benchmark pressure (10x the lowest gain at which a held benchmark stimulus fires), accepting rates above P5's band for a held stimulus

**Why.** Measured with a 1 mm probe after removing adaptation. AdEx at the P5 gain was silent below about 40% of the benchmark pressure (0, 0, 0, 18, 35, 55 ... 107 Hz from 0.1 to 2.0). At about 3-4x the gain it fires from about 10% and rises roughly linearly. Izhikevich without adaptation is too steep: at its P5 gain it is silent below about 60%, and lowering the threshold pushes it to 470-800 Hz. Both choices are yours (2026-09-28). The drive is linear in both gain and pressure, so a held benchmark at 10% of its amplitude with gain g drives the neuron exactly as the full benchmark does with gain g/10. The rule therefore finds the lowest firing gain g0 by bisection and takes 10 g0.

**What was rejected.** Keeping P5's 45 Hz held-stimulus gain for AdEx (a dead zone below 40% of the benchmark pressure). A non-adapting Izhikevich recipe (no gain gives it both a low threshold and physiological rates).

**Where it lives.** `scripts/calibrate_recipe_gains.py` (`SA_RULES`, `choose_sa_gain_low_threshold`); `sensoryforge/presets/tactile_sa1_ra1.yml` (`d`), `tactile_sa1_ra1_adex.yml` (gains); `tests/integration/test_recipe_calibration.py`.

**Ledger id + sha.** D-8dde454 · `67c45b5` (decision)

**Validation pending.** A held benchmark stimulus now drives AdEx SA above P5's 20-100 Hz band (about 125 Hz). The Kalman filter's behaviour with these rates is untested.

## D-ce22df3 · Simple neurons by default; adaptation and the clamp opt-in · 2026-09-28

**What was decided.** the tactile recipes' neurons have no spike-frequency adaptation and no voltage clamp by default -- AdEx SA1_tonic and RA1_phasic get a = b = 0, the Izhikevich recipe sets d = 0, and v_floor defaults to none on Izhikevich and AdEx; adaptation (the TouchSim-fitted values, kept as AdEx presets SA1_adapting and RA1_adapting and as documented Izhikevich d values) and the clamp are opt-in options

**Superseded in part by:** D-8dde454 · 2026-09-28 (the Izhikevich recipe keeps d 15 / 24)

**Why.** pressure-simulation is the priority (D-6bae4df). Its models should stay simple enough for hardware, and adaptation adds a history-dependent term between rate and pressure that its Kalman-filter inference would have to model. The clamp was a guard against Euler blow-up at coarse steps (D-007). The neurons now integrate at 0.05 ms by default (F-008), and the input floor (D-43dc520) keeps negative drive out of tactile neurons, so the recipes' voltage stays near rest without it. With the clamp off, SensoryForge's Izhikevich also matches pressure-simulation's unclamped neurons (F-037).

**What was rejected.** Keeping the TouchSim-fitted adaptation by default (the Kalman concern above). Keeping the clamp by default: it was never reached in a way that changed spikes, and it hid the AdEx adaptation problem (F-f59aa11) rather than solving it.

**Where it lives.** `sensoryforge/neurons/adex.py` (`ADEX_PRESETS`: `SA1_tonic`/`RA1_phasic` non-adapting, `SA1_adapting`/`RA1_adapting` opt-in; `v_floor` default `None`); `sensoryforge/neurons/izhikevich.py` (`v_floor` default `None`). MQIF keeps its -120 mV clamp; you named only Izhikevich and AdEx.

**Ledger id + sha.** D-ce22df3 · `5251e45` (decision)

**Validation pending.** A user integrating strongly negative input at a coarse step should set `v_floor`; nothing warns them.

## D-4e669b4 · RA answers release as strongly as indentation, by design · 2026-09-28

**What was decided.** RA's response to release as strong as to indentation (the RA filter is symmetric in the rate of change) is kept by design and documented as a known difference from TouchSim's RA

**Why.** It suits the pressure-simulation design: RA reports the magnitude of the rate of pressure change in either direction. TouchSim's RA releases at 0.67-0.8 of its onset and is silent at 0.2 mm (F-bad9126).

**Where it lives.** `sensoryforge/filters/sa_ra.py::RAFilterTorch` (eq. 8, `|dI/dt|`); documented in `docs/user_guide/units_and_gains.md`.

**Ledger id + sha.** D-4e669b4 · `5251e45` (decision)

**Validation pending.** None -- settled.

## D-407c639 · A larger gabor default · 2026-09-28

**What was decided.** the named gabor stimulus's default sigma and wavelength become 1.0 mm (those of the layered gabor shape), so a default gabor spans many receptors of a 0.15 mm grid

**Why.** At sigma 0.3 mm and wavelength 0.5 mm, a default gabor covered about two receptors of the recipes' 0.15 mm grid. It drove tactile_sa1_ra1 to 54 SA and 0 RA spikes in 500 ms, against 1214 and 82 for a default gaussian (F-d33d335). At 1.0 mm, 73 receptors of a 41 x 41 grid sit above half its peak.

**What was rejected.** Changing `texture`'s defaults too. It uses the same class but its own render-level defaults (2.0 mm), and it was not the problem.

**Where it lives.** `sensoryforge/stimuli/texture.py::GaborTexture` (constructor and ParamSpec defaults).

**Ledger id + sha.** D-407c639 · `5251e45` (decision), `63ec27a` (change)

**Validation pending.** None -- settled.

## D-43dc520 · A tactile afferent's neuron input is floored at zero · 2026-09-24

**What was decided.** the current a tactile afferent population's neuron receives is floored at 0 mA (PopulationConfig.input_floor, which resolves to 0 for SA, RA and SA2 populations and to no floor otherwise); the recorded filtered drive stays signed, so bundles and pressure-simulation's decoder see the same signal

**Why.** With SA fitted to SA1's ramp response (k2 = 8, D-f4d0967) and a gain that puts a held stimulus at 45 Hz, the SA current on a moving stimulus's trailing edge reached about -100 to -136 mA (F-96ff772). Izhikevich SA hit its -120 mV floor there. AdEx, with only a linear leak, would have fallen to about -1000 mV without its clamp. A mechanoreceptor's transduction current cannot reverse, so negative mechanical drive should silence the afferent, not hyperpolarize it.

The floor sits where the current enters the neuron (`SimulationEngine._run_pop_from_drive`, after gain and noise). The recorded `filtered` keeps its sign, and pressure-simulation's decoder reads velocity sign from it (F-001). It resolves by population: 0 mA for SA, RA and SA2 on a built-in neuron model. A DSL model, which may be an analog readout of a signed signal, gets none, and so does any other type. The vision preset sets `-.inf` explicitly, because it reuses the SA type label for non-tactile populations.

**What was rejected.** Rectifying the SA filter's output (`clip_to_positive`). The recorded, decoded signal must stay signed (F-001).

**Where it lives.** `sensoryforge/config/schema.py::PopulationConfig.input_floor`; `sensoryforge/config/defaults.py::resolve_input_floor`; `sensoryforge/core/simulation_engine.py::_run_pop_from_drive`; `tests/unit/test_input_floor.py`.

**Ledger id + sha.** D-43dc520 · `9e2b70a` (decision)

**Validation pending.** The legacy pipelines (`GeneralizedTactileEncodingPipeline`, `TactileEncodingPipelineTorch`) do not apply the floor. The GUI does not show the field yet.

## D-f5853a4 · AdEx gets an absolute refractory period · 2026-09-24

**What was decided.** AdEx neurons get an absolute refractory period t_ref (default 0 ms, so existing configs are unchanged); the SA1_tonic and RA1_phasic presets use 2 ms, capping their rates near 500 Hz

**Why.** At the TouchSim-fitted sensitivity, AdEx RA reached 1200 Hz per afferent on `moving_edge` (six spikes in 5 ms), because AdEx has no refractory period. After a spike, `t_ref` holds the voltage at v_reset while the adaptation current keeps evolving. With 2 ms, a constant 400 mA drive gives 395 Hz instead of 880 Hz, and `moving_edge` peaks at 400 Hz. With `t_ref` = 0 the update is bit-identical to the code before the change.

**What was rejected.** Capping rates in the analysis (a scoring ceiling), which would have left the simulated spike trains unphysiological.

**Where it lives.** `sensoryforge/neurons/adex.py` (`t_ref`, `ADEX_PRESETS`); `tests/unit/test_adex_refractory.py`.

**Ledger id + sha.** D-f5853a4 · `9e2b70a` (decision)

**Validation pending.** Izhikevich has no refractory period and reaches 600-800 Hz on `moving_edge`. Its f-I curve is steep, and no decision covers it yet.

## D-f4d0967 · SA matches TouchSim's SA1 · 2026-09-24

**Update 2026-09-28 (D-ce22df3, D-8dde454):** the fitted adaptation stays in the Izhikevich recipe. For AdEx it moved to the opt-in presets `SA1_adapting`/`RA1_adapting`, and the AdEx recipe is non-adapting. The SA filter's `k2` = 8.0 is unchanged.

**Update 2026-09-24 (D-43dc520, D-f5853a4):** with the neuron-input floor and AdEx's refractory period, the AdEx recipe's gains became SA 370 / RA 490. AdEx now agrees with SA1 at every compared depth except 0.2 mm, where its hold is 2.9 Hz against 8.6 (one spike against three). Its hold ISI CV is about 0.5-0.6, against P5's 0.5. Izhikevich is unchanged.

**What was decided.** SensoryForge's SA matches TouchSim's SA1 in its ramp (dynamic) response and in a graded rise of rate with indentation, for both the Izhikevich and the AdEx recipe

**Why.** The TouchSim comparison (`a2e8c0b`, F-20e111e) found two problems. SA's ramp response was 2-3x weaker than SA1's (hold/onset about 0.55 against 0.22). AdEx SA had a hard threshold: silent up to the 0.7 mm level, then 40 Hz. `scripts/validation/fit_afferents.py` searched grids scored by the RMS log error against SA1's onset and hold rates over 0.1-1.25 mm. The amplitude per mm was refitted at every point (the model is linear up to the neuron, so this absorbs SA's gain). The search found two levers:
- **The ramp response comes from the SA filter's `k2`,** the gain on the input's rate of change, the only velocity-sensitive term. Parvizi-Fard et al. (2021) used 3.0. Izhikevich's best is 10 and AdEx's is 5. **8.0** costs each about one spike at one depth, and it keeps one SA filter for both recipes and for pressure-simulation: its `design/drive.py` builds filters from the same resolver defaults, and its design directories cannot carry filter parameters (F-094).
- **The graded rise comes from spike-frequency adaptation.** Izhikevich `d` goes 8 -> 15, set in the recipe, because RS is Izhikevich's published preset. AdEx `SA1_tonic` gets b 0 -> 28, tau_w 200 -> 110 ms and v_reset -58 -> -70 mV.

Result (`benchmarks/results/touchsim_comparison/`): both recipes agree with SA1 on the hold rate, the onset rate and hold/onset at every compared depth. Izhikevich reaches 40/60/100/160 Hz onset against 40/60/100/180, and 8.6/17/26/40 Hz hold against 8.6/11/26/43.

Two consequences followed:
- **SA's gain rule changed.** With SA1-like dynamics, SA answers motion at several times its hold rate. On the AdEx recipe the static and moving benchmark rates (13 against 61-74 Hz) spread wider than P5's 20-100 Hz band, so the old rule could not be met: it wanted all four stimuli in the band. SA's gain now puts the one held stimulus (`ramp_gaussian`'s hold) at 44.7 Hz, which is how P5 states its SA band. The moving stimuli's rates are reported (Izhikevich 83-86 Hz, AdEx 167-196 Hz); TouchSim's SA1 itself reaches 180 Hz during a ramp.
- **The release current reaches the voltage floor.** The larger `k2` makes the negative SA current on a trailing edge larger. Izhikevich SA reaches its -120 mV floor there. Its spikes are identical with and without the clamp, which is what F-037 is about, and the recipe test now asserts exactly that.

**What was rejected.**
- **Tuning `k2` per recipe.** pressure-simulation runs AdEx on the default filters, so a recipe-only value would not reach its designs.
- **Adaptation alone, without `k2`.** It flattened the rate-intensity curve, but left the ramp response at about half of SA1's, because the drive is still rising during the ramp.
- **The first AdEx fit (b = 48-56).** It looked best, but in this AdEx form `w` enters dv/dt in mV, so it drove the voltage into the -130 mV clamp after every spike, and the clamp shaped the dynamics. Fits now reject any point whose hold voltage comes within 20 mV of the floor.

**Where it lives.** `sensoryforge/config/defaults.py::FILTER_DEFAULTS` (and `SAFilterTorch`'s defaults); `sensoryforge/neurons/adex.py::ADEX_PRESETS`; `sensoryforge/presets/tactile_sa1_ra1.yml` and `tactile_stochastic_control.yml` (`model_params.d`); `scripts/validation/fit_afferents.py`, `benchmarks/results/afferent_fit/`; `scripts/calibrate_recipe_gains.py`.

**Ledger id + sha.** D-f4d0967 · `ce166e0` (decision)

**Validation pending.** TouchSim is one model of SA1, not recordings. Real afferent data would be the stronger test. At its calibrated gains the AdEx recipe leaves the physiological voltage range (open entry), so AdEx SA's fit holds on the probe's range, not on the benchmark stimuli.

## D-d9bd411 · RA matches TouchSim's RA sensitivity · 2026-09-24

**Update 2026-09-28 (D-673e0ed):** the Izhikevich recipe keeps the TouchSim-matched RA gain. The AdEx recipe's RA gain now follows the low-threshold rule instead.

**Update 2026-09-24 (D-43dc520, D-f5853a4):** RA's peak on `moving_edge` is now 400 Hz for AdEx (refractory period) and 600 Hz for Izhikevich. AdEx's RA gain is 490 against SA 370.

**What was decided.** SensoryForge's RA matches TouchSim's RA in sensitivity relative to SA, firing at the small indentations where TouchSim's RA fires, so small movements are detected; TouchSim's RA, not P5 alone, sets RA's calibration

**Why.** At intensities matched on SA's hold rate, RA stayed silent at 0.2-0.7 mm, where TouchSim's RA fires 20-60 Hz at onset (F-47cc697). RA's job is to report small movements. Because the amplitude per mm absorbs SA's gain, TouchSim constrains RA's gain relative to SA's.
- **RA's gain:** `scripts/calibrate_recipe_gains.py` sweeps it on a 15% grid at the calibrated SA gain, and takes the geometric centre of the gains whose onset rates best match TouchSim's RA at 0.1-1.25 mm. The 0.1 mm depth, where TouchSim's RA is silent, penalises an RA that is too sensitive.
- **RA's adaptation:** it makes RA's rate grow gradually with ramp speed instead of saturating a few spikes above threshold. Izhikevich `d` goes 2 -> 24; AdEx `RA1_phasic` gets b 20 -> 40 and v_reset -58 -> -70 mV.

Result: RA's onset rates match TouchSim's at every depth: Izhikevich 0/20/40/60/100 Hz, identical to TouchSim; AdEx 0/20/40/60/120 Hz. Gains: Izhikevich 410 against SA 380; AdEx 660 against SA 500.

**What was rejected.** Keeping P5's RA rule, the centre of the gains whose bursts stay within 150-400 Hz, which set RA's sensitivity from the benchmark stimuli alone and left RA blind to small indentations. P5's RA criteria are now reported as a check. The silent hold still holds. The burst band does not: on the fast `moving_edge`, RA reaches 600 Hz (Izhikevich) and 1200 Hz (AdEx).

**Where it lives.** `scripts/calibrate_recipe_gains.py::choose_ra_gain_touchsim`; `scripts/validation/compare_with_touchsim.py::ra_onset_error`; the three tactile presets.

**Ledger id + sha.** D-d9bd411 · `ce166e0` (decision)

**Validation pending.** Two points stay open:
- **RA's release.** It is as strong as its onset, because the RA filter responds to the absolute rate of change. TouchSim's RA releases more weakly and is silent at 0.2 mm (open entry).
- **Peak rates on fast stimuli** exceed what afferents reach, AdEx's especially, which has no refractory period (open entry).

## D-ea0f017 · Tactile recipes get per-population gains, calibrated against P5 · 2026-09-24

**Update 2026-09-28 (D-8dde454, D-673e0ed):** the AdEx recipe's SA and RA gains follow the low-threshold rule, not P5. The Izhikevich recipe keeps P5 for SA and TouchSim for RA.

**Superseded in part by:** D-d9bd411 · 2026-09-24 (RA's gain rule; SA's is refined in D-f4d0967's section)

**What was decided.** SensoryForge's tactile recipes give SA and RA their own input gains, calibrated on the responsive-set rate against the P5 bands over the four benchmark stimuli, with the drive scale reconciled with pressure-simulation's design-time model (its C-032)

**Why.** At the shared gain of 50, the Izhikevich SA baseline reached 8.75 Hz on the recipe's one genuine hold, against P5's 20-100 Hz (F-093). AdEx RA fired nothing on `drifting_grating`, whose onset drive peaks at 3.08 mA against a 7.57 mA rheobase (F-092). A gain sweep (`scripts/calibrate_recipe_gains.py`, 10% steps from 30 to about 400) showed that no single gain meets both populations' targets. Two rules were fixed before choosing, so the gains follow from P5 rather than from a preference:
- **SA:** the gain that puts the geometric mean of the four stimuli's responsive-set rates at the band's log centre, 44.7 Hz.
- **RA:** the geometric centre of the gains at which every onset burst reaches 150-400 Hz per afferent and a held stimulus is silent.

Results: `tactile_sa1_ra1` SA 220 / RA 61; `tactile_sa1_ra1_adex` SA 55 / RA 86. `tactile_stochastic_control` shares `tactile_sa1_ra1`'s gains, so the control arm still differs only in its receptive fields. Two methodological corrections came out of the sweep:
1. **The static hold was scored from the instant the ramp ended.** That counted RA spikes 1-6 ms later, the tail of the ramp response through the 8 ms RA filter, as hold firing. P5 allows spikes for the first ~30 ms of a static stimulus, so the hold is now scored from 30 ms after the ramp.
2. **F-092's objection to raising RA sensitivity does not hold.** It treated `moving_edge`'s steady interval as a hold, but the edge never stops moving there (F-089).

The drive-scale reconciliation needed no change on this side. Pressure-simulation's 150x (its C-032) was its decoder multiplying `input_gain` into SensoryForge's `filtered` array a second time. That array already includes the gain, and pressure-simulation fixed its decoder in `d448fd8`.

**What was rejected.** Leaving input gain entirely to pressure-simulation's design directories. The recipes must run sensibly without a design, and one shared gain cannot put SA and RA in their bands at once. Also rejected: tuning AdEx's membrane resistance R instead. R and `input_gain` scale the same current, and the gain is the knob every neuron model shares.

**Where it lives.** `sensoryforge/presets/tactile_sa1_ra1*.yml`, `tactile_stochastic_control.yml`; `scripts/calibrate_recipe_gains.py`; `benchmarks/results/recipe_calibration/`; `tests/integration/test_recipe_calibration.py`, which fails if either recipe leaves P5 or comes within 20 mV of `v_floor` (F-037).

**Ledger id + sha.** D-ea0f017 · `b4a853b` (decision), `63d37c3` (calibration)

**Validation pending.** P5 is the calibration target, so meeting it validates nothing by itself. The TouchSim comparison (`benchmarks/results/touchsim_comparison/`, `a2e8c0b`) is the independent check. It agrees on the Izhikevich SA rate-intensity curve and on RA's silent hold, but finds RA far less sensitive relative to SA than TouchSim's RA. That points at the RA gain, or at the RA filter, as the next thing to revisit.

## D-030 · The canonical stimulus block is what `sensoryforge run` renders · 2026-09-21

**What was decided.** the canonical stimulus: block is what sensoryforge run renders (via stimuli.render.render_for_config, shared with the GUI); a legacy stimuli: list still wins with a deprecation notice

**Why.** `sensoryforge run` chose its stimulus from the legacy top-level `stimuli:` list and never read `SensoryForgeConfig.stimulus`, so a canonical YAML exported from the GUI ran a default trapezoid instead of the stimulus it declared (the finding on `e79ff48`). Rendering through the same `render_for_config` the GUI uses makes GUI and CLI runs of one config identical, which `tests/integration/test_gui_engine_equality.py` pins.

**What was rejected.** Dropping the legacy `stimuli:` list outright: existing legacy configs would change behaviour silently. It keeps precedence when present, with a deprecation notice, until the legacy path is removed.

**Where it lives.** `sensoryforge/cli.py` (the stimulus selection, around the deprecation comment at line 321); `sensoryforge/stimuli/render.py::render_for_config`.

**Ledger id + sha.** D-030 · `e79ff48`

**Validation pending.** none — settled.

## D-029 · One receptor/receptive-field preview widget · 2026-09-21

**What was decided.** the receptor/receptive-field preview is one reusable widget (sensoryforge/gui/widgets/grid_preview.py) built from GridConfig via core.simulation_engine.build_grid, the same code path SimulationEngine uses

**Why.** The GUI audit (`docs_root/gui_audit/10_plan.md`) found the old GUI drew grids through private code that differed from the engine: the Spiking tab sampled receptors with a flat reshape the engine does not use (F-076), and `poisson`/`hex` grids crashed config paths. A preview built by `build_grid` shows exactly the receptors a run uses, for every arrangement.

**What was rejected.** Keeping the preview inside the Mechanoreceptor tab's drawing code: that tab was deleted in Phase 3, and its preview was bound to the tab's private dataclasses rather than `GridConfig`. The drawing code was lifted, not rewritten.

**Where it lives.** `sensoryforge/gui/widgets/grid_preview.py`; `sensoryforge/core/simulation_engine.py::build_grid`.

**Ledger id + sha.** D-029 · `c4d6317`

**Validation pending.** none — settled.

## D-027 · Every pyqtgraph widget is built by plot_factory · 2026-09-17

**What was decided.** every pyqtgraph widget in GUI v2 is built by sensoryforge/gui/widgets/plot_factory.py; pyqtgraph signals are connected only through plot_factory.connect (functools.partial, no bound methods or widget-closing lambdas) and released by teardown()

**Why.** F-035: with the cyclic GC on, the old GUI segfaulted (3 of 3 runs) in `ScatterPlotItem.renderSymbol` through a ViewBox lambda left by a destroyed tab, and the test suite had to turn the GC off, hiding the whole crash class. Signal wiring that holds no reference back to a widget removes that failure mode; putting all construction in one module gives it one place to be enforced. A 50-times build/destroy test under `gc.collect()` proves it (`tests/gui_v2/test_plot_factory.py`), and the suite now runs with the GC on.

**What was rejected.** Fixing each offending lambda in place: the audit also found three separate plotting paths (F-063), so the hazard would have come back with the next plot.

**Where it lives.** `sensoryforge/gui/widgets/plot_factory.py` (`connect`, `teardown`).

**Ledger id + sha.** D-027 · `0bfc3ea`

**Validation pending.** F-085 remains open: collecting pyqtgraph objects left in cycles by a destroyed window can still destroy a live ViewBox, mitigated by the test harness but not explained.

## D-026 · Forms are generated from get_param_spec() · 2026-09-17

**What was decided.** GUI v2 forms are generated from get_param_spec() by sensoryforge/gui/widgets/param_form.py and bind to the Session by dotted path; no per-component GUI code

**Why.** The audit found three large form tabs with private dataclasses that File > Load/Save barely reached, and parameters editable in two places. `get_param_spec()` is required on every component since G1, so a form generated from it covers built-ins and plugins alike, and a new plugin appears in the GUI with no GUI code. Binding by dotted path (`populations.1.filter_params.tau_r`) writes straight into the one `SensoryForgeConfig`, so there is nothing to translate on save.

**What was rejected.** Hand-written forms per component: each plugin would need GUI code, and each form would drift from the component's real parameters. The Circuit tab's `build_param_form` was the model and was lifted.

**Where it lives.** `sensoryforge/gui/widgets/param_form.py`.

**Ledger id + sha.** D-026 · `8657bea`

**Validation pending.** none — settled.

## D-025 · Stimuli are rendered on a regular canvas and sampled at receptors · 2026-09-17

**What was decided.** stimuli are rendered on a regular canvas of the array's extent (sensoryforge/stimuli/canvas.py) for every arrangement and sampled at receptor coordinates by SimulationEngine; the "grid" canvas is bit-identical to ReceptorGrid.get_coordinates()

**Why.** `poisson` and `hex` arrangements crashed every config-driven path (CLI, BatchExecutor, Circuit tab) at stimulus render, because they built a `ReceptorGrid` only to call `get_coordinates()`, which raises when an arrangement has no lattice (`04ab68a`). The engine already samples frames at each receptor's real position (`grid_sample`, Wave L), so a regular canvas over the array's extent serves every arrangement. Keeping the `grid` canvas bit-identical preserves the golden fixtures.

**What was rejected.** Giving each irregular arrangement a lattice of its own: `poisson` and `hex` have none that matches their receptors, and any invented one would be sampled anyway.

**Where it lives.** `sensoryforge/stimuli/canvas.py`; `SimulationEngine`'s receptor sampling.

**Ledger id + sha.** D-025 · `04ab68a`

**Validation pending.** none — settled.

## D-022 · One light theme · 2026-09-17

**What was decided.** GUI v2 uses one light theme (sensoryforge/gui/theme.py): app #F4F5F7, panels #FFFFFF, accent #2563EB, population colours SA #2563EB / RA #E8630A; no dark mode

**Why.** The audit's visual findings included a dark DockArea beside white heatmaps. One palette removes that mismatch. Light was chosen because the existing light tab was the best-received one, and figures exported from a light GUI match print. SA and RA get fixed, distinguishable colours so a population keeps its colour across every screen.

**What was rejected.** A dark theme, or both themes: listed as an explicit non-goal in the GUI v2 plan, since two themes double the styling that has to be checked.

**Where it lives.** `sensoryforge/gui/theme.py`.

**Ledger id + sha.** D-022 · `5369bc9`

**Validation pending.** none — settled.

---

# Log — every recorded decision, in order

Appended automatically by `.claude/hooks/ledger-sync.sh` from `Decision:` and `Retires:` trailers,
so level 2 always holds at least the dated fact even before anyone writes the reasoning above.
Do not hand-edit between the markers.

| Date | Id | One line | Commit |
|---|---|---|---|
<!-- DECISIONS_LOG_START -->
| 2026-09-22 | D-039 | upgrade the living-ledger template from v1 to v3 — enforced commit trailers, automatic post-commit ledger sync, Refs: backlinks, the level-2 decisions record, stale-rule detection, automatic cross-repo index push | `36d2e3f` |
| 2026-09-24 | D-0437899 | every stimulus defaults to peak amplitude 1.0, pressure-simulation's convention: the named gaussian, texture and moving types and their layered presets change from 30 to 1.0, and no stimulus renders negative values by default (gabor, texture) | `b4a853b` |
| 2026-09-24 | D-ea0f017 | SensoryForge's tactile recipes give SA and RA their own input gains, calibrated on the responsive-set rate against the P5 bands over the four benchmark stimuli, with the drive scale reconciled with pressure-simulation's design-time model (its C-032) | `b4a853b` |
| 2026-09-24 | D-4aafcdc | the quantitative afferent comparison uses touchsim output generated once in a throwaway environment and committed as fixture data; touchsim never becomes a dependency | `b4a853b` |
| 2026-09-24 | D-88b4b41 | GridConfig.density sets the receptor count of poisson, hex and blue_noise layouts (density times the rows x cols x spacing extent); setting it on grid or jittered_grid, where spacing fixes the count, is an error | `b4a853b` |
| 2026-09-24 | D-d9bd411 | SensoryForge's RA matches TouchSim's RA in sensitivity relative to SA, firing at the small indentations where TouchSim's RA fires, so small movements are detected; TouchSim's RA, not P5 alone, sets RA's calibration | `ce166e0` |
| 2026-09-24 | D-f4d0967 | SensoryForge's SA matches TouchSim's SA1 in its ramp (dynamic) response and in a graded rise of rate with indentation, for both the Izhikevich and the AdEx recipe | `ce166e0` |
| 2026-09-24 | D-7e71f68 | SA's calibrated gain puts the one held benchmark stimulus (ramp_gaussian's static hold) at 44.7 Hz, the centre of P5's band, with ISI CV below 0.5; the moving stimuli's SA rates are reported, not required to lie in the band, because SA now answers motion as TouchSim's SA1 does | `3b30da8` |
| 2026-09-24 | D-43dc520 | the current a tactile afferent population's neuron receives is floored at 0 mA (PopulationConfig.input_floor, which resolves to 0 for SA, RA and SA2 populations and to no floor otherwise); the recorded filtered drive stays signed, so bundles and pressure-simulation's decoder see the same signal | `9e2b70a` |
| 2026-09-24 | D-f5853a4 | AdEx neurons get an absolute refractory period t_ref (default 0 ms, so existing configs are unchanged); the SA1_tonic and RA1_phasic presets use 2 ms, capping their rates near 500 Hz | `9e2b70a` |
| 2026-09-24 | D-6bae4df | for now SensoryForge's tactile models serve the pressure-simulation project and stay simple enough to implement in hardware (Izhikevich or AdEx, no new model terms); the requirement is that SA and RA both fire, SA's rate proportional to pressure and RA's to its rate of change; fixing AdEx's adaptation voltage range and other SensoryForge-specific extensions are deferred | `c368ac7` |
| 2026-09-28 | D-ce22df3 | the tactile recipes' neurons have no spike-frequency adaptation and no voltage clamp by default -- AdEx SA1_tonic and RA1_phasic get a = b = 0, the Izhikevich recipe sets d = 0, and v_floor defaults to none on Izhikevich and AdEx; adaptation (the TouchSim-fitted values, kept as AdEx presets SA1_adapting and RA1_adapting and as documented Izhikevich d values) and the clamp are opt-in options | `5251e45` |
| 2026-09-28 | D-4e669b4 | RA's response to release as strong as to indentation (the RA filter is symmetric in the rate of change) is kept by design and documented as a known difference from TouchSim's RA | `5251e45` |
| 2026-09-28 | D-407c639 | the named gabor stimulus's default sigma and wavelength become 1.0 mm (those of the layered gabor shape), so a default gabor spans many receptors of a 0.15 mm grid | `5251e45` |
| 2026-09-28 | D-8dde454 | the Izhikevich recipe keeps its TouchSim-fitted spike-frequency adaptation (d 15 for SA, 24 for RA), because without it SA is too steep to be graded; the AdEx recipe, pressure-simulation's model, stays non-adapting, with SA's gain set so SA fires from about 10% of the benchmark pressure (10x the lowest gain at which a held benchmark stimulus fires), accepting rates above P5's band for a held stimulus | `67c45b5` |
| 2026-09-28 | D-673e0ed | the AdEx recipe's RA gain is set like its SA gain -- RA fires from about 10% of the benchmark stimulus's rate of pressure change (10x the lowest gain at which the held benchmark's ramp makes RA fire) -- instead of matching TouchSim's RA sensitivity, so RA's rate stays proportional to the rate of change over the benchmark range; the Izhikevich recipe keeps the TouchSim-matched RA gain | `df3320e` |
| 2026-09-30 | D-5cdc524 | Sensor (receptor-current) noise and membrane noise are separate SensoryForge parameters; a design directory's declared noise_std is simulated as sensor noise and membrane_noise_std as membrane noise | `7f7c6a3` |
| 2026-09-30 | D-802e42b | Sensor noise lives in the sensors: one noise source per receptor, in pressure (stimulus) units, added to the stimulus on the receptor grid before innervation, shared by every population; pooling, filtering and gain then propagate it. The existing per-population current noise (sensor_noise_std, added after filter and gain) is a different thing -- neuron input noise -- and stays as its own option. | `9a3f550` |
| 2026-09-30 | D-c5facd5 | RA is a signed level-crossing unit, event-camera style: it emits an ON event when its input has risen by a threshold theta since its last event and an OFF event when it has fallen by theta, moving its reference by theta each time; the sign is carried by the event, not recovered later (SensoryForge neuron_model level_crossing, opt-in; AdEx and the rectified RA filter stay as the reference arm) | `6124f87` |
| 2026-09-30 | D-f0e2433 | SA gets a sigma-delta arm: a non-leaky integrate-and-fire unit with subtractive reset whose spike rate is linear in its input level; what matters is the time-scale separation (RA = fast signed change, SA = slow absolute level), not the biological mechanism (SensoryForge neuron_model sigma_delta, opt-in) | `6124f87` |
| 2026-10-02 | D-94c08c7 | a world class is a `layered` layer with random fields (axes bind to its shape, pattern, modulation and episode fields), so a draw is an ordinary layered stimulus and SensoryForge keeps one stimulus language | `e766b8a` |
| 2026-10-02 | D-99d39b6 | one episode is a quiet lead-in then touch, hold, slide, release, optionally repeated as several contacts separated by pauses; motion happens only during slides; temporal frequency is a layer modulation (sine vibration, pulses for repeated indentation) | `e766b8a` |
| 2026-10-02 | D-37c5247 | quiet comes from every draw's lead-in and tail, a `quiet` class and pauses between contacts; a session is draws laid end to end with no separate gap mechanism or quiet-fraction target | `e766b8a` |
| 2026-10-02 | D-cccbd6a | the world engine is a new sensoryforge.world package with its own vectorised renderer kept equal to `layered` by tests; `layered` gains only additive default-off fields and the old sweep BatchExecutor stays untouched (rewriting `layered` on the new kernel, or extending BatchExecutor, were rejected) | `e766b8a` |
| 2026-10-02 | D-d3605dd | data sets offer both repeat semantics -- `repeats` (fresh draws and noise per replicate) and `noise_repeats` (same draws, k noise realisations) | `e766b8a` |
| 2026-10-02 | D-46a52e6 | the world engine ships as version 1.1.0, tagged v1.1.0 on its branch, with main and the ~/sensoryforge checkout left untouched until Ben merges | `e766b8a` |
| 2026-10-02 | D-bed3c30 | a world's classes are weighed and iterated in order of their names, so no draw, data-set entry or entry order depends on the order a world file writes its classes, axes or defaults in | `e6bf7f2` |
| 2026-10-02 | D-0fbb84a | the batch runner sets every population's noise_seed to seed53(noise, "population", i), whether or not the design set one, so no population's noise hangs on the 32-bit run seed | `aa88bd3` |
| 2026-10-04 | D-32d7d70 | the background is its own draw sharing the contact envelope, with the layer total floored at clamp_min where the envelope is positive (P6, provisional). | `6c0daf3` |
| 2026-10-04 | D-1b2d1c1 | lateral scan bias is an angular central Gaussian solved from the travel ratio (P4a, provisional) | `19d7a32` |
| 2026-10-04 | D-b0be6a6 | a press falls as it rose through same_as, an axis link (P3a, provisional) | `62cb959` |
| 2026-10-04 | D-81193aa | self-affine textures use the Persson spectrum from cosines, RMS amplitude/sqrt(2) (P7, provisional) | `f276323` |
| 2026-10-04 | D-072fed4 | dot arrays on a square or hexagonal lattice of Gaussian bumps (P9, provisional) | `7e16d00` |
| 2026-10-04 | D-84200e4 | indenters are depth-driven, the footprint growing during the rise (P5, provisional) | `956af6c` |
| 2026-10-04 | D-fda02f9 | step_edge is an indenter shape, value d on the plate, d minus the circular sag on the shoulder, 0 past it | `5cfab52` |
| 2026-10-04 | D-7d5e5bc | letter_text recovers int(u * 2**53) as the draw's bits and maps cells x lines sub-uniforms through the cumulative weights | `5d658d3` |
| 2026-10-04 | D-80d1fcd | groups are resolved into class axes at load (P1 option A), not kept as a runtime concept | `2487d37` |
| 2026-10-04 | D-6b0215f | sessions are budgeted (P8 a), gap count capped by the episode boundaries | `771bd06` |
| 2026-10-04 | D-822574c | "touches for" is the plateau hold_ms, no contact_ms field (P2 a, provisional) | `9082178` |
| 2026-10-04 | D-47d4dac | release the world 174 addendum as v1.2.0, tagged locally only and not pushed | `85a8a32` |
| 2026-10-04 | D-0c2b6a3 | self_affine puts its N components on equal strata of xi (q^2 below the roll-off, ln q above), each with its stratum's share of the Persson power; RMS amplitude/sqrt(2) kept (P7 still provisional) | `7421092` |
| 2026-10-04 | D-096e081 | clamp_min's domain is [-1e4, 0], since a floor above 0 would lift every point in contact to it, a jump at contact onset | `7421092` |
| 2026-10-04 | D-3a10bce | a registered distribution without a finite support binds a number field only when it declares bounds=, checked against the field's domain | `7421092` |
| 2026-10-04 | D-7d51fbe | release the whole-branch review's fixes as v1.2.1, tagged locally only and not pushed | `8757ca9` |
<!-- DECISIONS_LOG_END -->
