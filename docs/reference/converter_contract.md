# Event converter contract

This page is what pressure-simulation's Phase 3 is written against: what SensoryForge
**v1.3.0** guarantees about its two event converters, the signed level-crossing unit (RA) and
the sigma-delta unit (SA), from a population's drive to the events a bundle stores, and the
SensoryForge test that pins each guarantee. Pressure-simulation emulates both units bit for
bit and decodes their events, so the updates below are stated exactly. Where this page and
the code disagree, the tests decide.

v1.3.0 adds one optional parameter, the level-crossing unit's leaky reference
(`reference_leak_tau_ms`), and records in every bundle the input floor each population's
converter received (bundle schema 2.3.0). With the leak off both converters, and everything
the engine does before them, run as in v1.2.1, bit for bit (guarantee 1). Nothing changes for
worlds and data sets ([World engine contract](world_contract.md)).

## Getting it

```bash
conda run -n bio-encoding pip install --no-deps --force-reinstall \
  "sensoryforge @ git+file:///Users/benefron/sensoryforge@v1.3.0"
```

Units: time ms, currents mA, charge mA·ms. `δ` is the converter's sub-step
(`simulation.integrate_dt_ms`, default 0.05 ms), `dt_ms` the record step (one bin), and
`n_sub = round(dt_ms / δ)` the sub-steps per bin (`dt_ms` must be a whole multiple of `δ`).

## 1. From drive to events

`SimulationEngine.run` and `_run_pop_from_drive` (unchanged in v1.3.0), for each population in
this order:

1. **Receptor noise** is added to the stimulus at every receptor and record step, in stimulus
   units, before innervation (`simulation.receptor_noise_std`, seeded by
   `simulation.receptor_noise_seed`; a data-set entry carries its own seed).
2. **Innervation:** `drive = W · (stimulus + noise)`, one value per neuron and record bin,
   recorded as `/populations/<name>/drive`.
3. **Front-end filter** at the record step. A level-crossing population runs with
   `filter_method: none`; a sigma-delta population keeps whatever front end it declares.
4. **Input gain:** times `input_gain`, giving mA.
5. **Sensor (current) noise** per record bin, `sensor_noise_std` in mA.
6. **Cast to float32.** Steps 3–6 are recorded as `/populations/<name>/filtered` (signed,
   before the floor).
7. **Input floor:** `clamp(min=floor)` when `resolve_input_floor(neuron_type, neuron_model,
   input_floor)` returns a floor: 0 mA for a tactile population (SA, RA, SA2) on a built-in
   model such as `sigma_delta` or AdEx, none for `level_crossing` and DSL models, an explicit
   `PopulationConfig.input_floor` as given (`-inf` = none). Since v1.3.0 the bundle records the
   value applied (`encoder.input_floor_ma`, section 4).
8. **Hold:** each bin's value is repeated over its `n_sub` sub-steps.
9. **The converter** (sections 2 and 3), built with `dt = integrate_dt_ms` and
   `noise_std = membrane_noise_std`. It adds that comparator noise to its input at every
   sub-step, after the floor (so the noise is not floored), from the global RNG (reseeded from
   the population's `noise_seed` when it has one and some noise is set).
10. **Counting:** the converter's index-0 sample (its initial state) is dropped and each bin's
    `n_sub` sub-step counts are summed. For the level-crossing unit that is a net signed count
    (a bin with both ON and OFF events nets them, which needs a swing of 2θ inside one bin).
    Stored as `int16` in `/populations/<name>/events` (signed; attributes `signed`, `polarity`,
    `theta`) or `/populations/<name>/spikes`.

## 2. The level-crossing unit: `LevelCrossingNeuron`, `level_crossing`

| Parameter | Units | Domain | Default | Meaning |
|---|---|---|---|---|
| `theta` (θ) | mA (the gained drive's units) | > 0 | 1.0 | the input change per event |
| `refractory_ms` | ms | ≥ 0 | 0 | the dead time: least interval between two events of one neuron |
| `initial_reference` | — | `"zero"`, `"first"` | `"zero"` | the reference before the first sub-step |
| `reference_leak_tau_ms` (τ) | ms | `None`, or finite and ≥ `dt` | `None` | since v1.3.0: the reference's relaxation time constant; `None` = no leak |
| `dt` (δ) | ms | > 0 | 0.05 (set by the engine) | the sub-step |
| `noise_std` | mA | ≥ 0 | 0 (set by the engine) | comparator noise added to the input at each sub-step |

Any other value of `reference_leak_tau_ms` (zero, negative, below `dt`, `inf`, `nan`, a bool, a
string) raises `ValueError` naming the parameter, at construction and therefore at
`load_design`, so "no leak" has one spelling: `None` (`null` in JSON).

**State per neuron:** the reference `r` and a dead-time counter `b`. Before the first
sub-step `r = 0` (`"zero"`) or `r = x₁`, the first sub-step's input (`"first"`), and `b = 0`.
The dead time in sub-steps is `R = 0` when `refractory_ms = 0`, else
`R = max(1, round(refractory_ms / δ))` (Python's `round`, half to even). The leak per sub-step
is `λ = δ / τ`, computed once as a Python float.

**The update at sub-step n = 1, 2, …**, with `x` the held input plus comparator noise:

```
1. d = x - r
2. m = floor(|d| / theta + 1e-5)                 # whole quanta in d (1e-5: crossing tolerance, in thetas)
3. if R == 0:  k = sign(d) * m                    # several events in one sub-step, as one signed count
   else:       k = sign(d) * [m >= 1 and b == 0]  # at most one event
               b = R - 1 if k != 0 else max(b - 1, 0)
4. r = r + k * theta
5. if tau is set:  r = r + lam * (x - r)          # v1.3.0
6. record r (the state trace) and k (int16)
```

The torch expressions, which a bit-for-bit emulator copies:
`magnitude = torch.floor(d.abs() / theta + 1e-5)`, `ref = ref + k * theta`, then
`ref = ref + leak * (x - ref)` with `leak = dt / reference_leak_tau_ms`. With `None`, line 5 is
skipped entirely (no multiplication by zero, which could flip the sign of a zero).

**What it means.** Write `u = x − r` for the unit's difference.

- **One quantum (no leak, no dead time).** After line 4, `|x − r| < θ` up to the tolerance, so
  `r₀` plus the running sum of `k·θ` tracks the input within one θ.
- **A step.** A jump of `Δx` that then holds emits `⌊|Δx|/θ + 1e-5⌋` events of its sign on that
  sub-step, and then the unit is silent, with or without the leak (the residual only shrinks
  under line 5).
- **The leak forgets a held level.** With `x` held, `u` shrinks by `1 − λ` per sub-step: time
  constant about τ. The unit reports change only.
- **The difference as a high-pass.** Line 1 sees `u′ₙ = (1 − λ)·uₙ₋₁ + (xₙ − xₙ₋₁)`: leak the
  old state, then add the new input, the sigma-delta leak's own structure.
- **A steady slope.** For `xₙ = s·n·δ`, `r₀ = 0`, no dead time and `τ > δ`, the unit fires
  **iff `s > θ(1 − 1e-5)/τ`**. Before its first event `uₙ = (1 − λ)·sτ·(1 − (1 − λ)ⁿ)`; the first
  event is at sub-step `n* = ⌈ln(1 − θ(1 − 1e-5)/(sτ)) / ln(1 − λ)⌉`, and
  `T′(1 − λ) ≤ n*·δ < T′ + δ` with `T′ = −τ ln(1 − θ(1 − 1e-5)/(sτ))`, the continuous interval.
- **Why this form.** Of the simple discretisations (leak before or after the comparison;
  factor `δ/τ` or `1 − e^(−δ/τ)`), only this one, factor `δ/τ` with the leak after the reset,
  both fires a steady slope iff `s > θ/τ` and fires a step of `Δx` as `⌊Δx/θ⌋` events at once.
- **Held bins.** The engine holds the input over a bin (section 1, step 8), so with no dead
  time events fall only on a bin's first sub-step. Over K bins without an event,
  `u′ = ρᴷ·u₀ + Σₘ ρᴷ⁻ᵐ·Δxₘ` with `ρ = (1 − λ)^n_sub` and `Δxₘ` the drive's jump at bin m; a
  slope `s` fires iff `s·dt_ms/(1 − ρ) ≥ θ(1 − 1e-5)`, slightly below `θ/τ` (by a factor of about
  `1 − dt_ms/(2τ)`).
- **The dead time pays a fast change out late.** With `R ≥ 1` the unit fires at most once per
  R sub-steps and the reference moves one θ per event, so a change faster than `θ/(Rδ)` is paid
  out after it ends, one event every R sub-steps, until `|x − r| < θ`. Without a leak nothing
  is lost: the total count is `⌊total change/θ + 1e-5⌋`. **With a leak, line 5 still runs while
  the unit is blocked**, so part of a pending change leaks away: after a step `Δx` at sub-step
  1, event j (at sub-step `1 + jR`) fires iff `cⱼ ≥ θ(1 − 1e-5)`, where `c₀ = Δx` and
  `cⱼ₊₁ = (1 − λ)ᴿ(cⱼ − θ)`.
- **Initial reference.** With `"zero"`, a drive already at X on the first sub-step emits
  `⌊X/θ + 1e-5⌋` ON events there; with `"first"` the first sub-step emits nothing.
- **Sign.** `+k` is k ON events (the input rose by k·θ), `−k` k OFF events; the bundle's
  `event_encoding.polarity` says the same.

## 3. The sigma-delta unit: `SigmaDeltaNeuron`, `sigma_delta` (unchanged in v1.3.0)

| Parameter | Units | Domain | Default | Meaning |
|---|---|---|---|---|
| `theta` (θ) | mA·ms | > 0 | 100.0 | the charge per spike |
| `leak_tau_ms` (τ) | ms | `None`, or > 0 | `None` | the accumulator's leak; `None` = none |
| `refractory_ms` | ms | ≥ 0 | 0 | the dead time |
| `dt` (δ) | ms | > 0 | 0.05 (set by the engine) | the sub-step |
| `noise_std` | mA | ≥ 0 | 0 (set by the engine) | comparator noise added to the input at each sub-step |

**State per neuron:** the accumulator `u`, starting at 0, and the counter `b`, with R as in
section 2. **The update at sub-step n**, with `x` the held, floored input plus comparator noise:

```
1. u = u + dt * x                         # no leak
   u = u + dt * x - decay * u             # leak: decay = dt / leak_tau_ms; the u on the right is the old value
2. m = max(floor(u / theta + 1e-5), 0)    # never a negative count
3. if R == 0:  n = m
   else:       n = [m >= 1 and b == 0]
               b = R - 1 if n > 0 else max(b - 1, 0)
4. u = u - n * theta                      # subtract, never reset to zero
5. if R >= 1:  u = min(u, theta)          # on EVERY sub-step, blocked or not (anti-windup)
6. record u and n (int16)
```

- **No leak, no dead time:** the count up to time t is `⌊∫x dt/θ⌋` within one spike, a rate of
  `x/θ` per ms with no rheobase.
- **The leak adds a rheobase θ/τ:** a constant x fires only if `xτ ≥ θ(1 − 1e-5)`.
- **The dead time loses charge, it does not pay it out late:** at most one spike per R
  sub-steps, and line 5 keeps the accumulator at or below θ, so a dropped drive gives at most
  one more spike, never a burst. Before v1.3.0 the docstring and user guide said the clip
  applies "while the neuron is refractory"; the code has always clipped on every sub-step
  whenever `refractory_ms > 0`, and the words now say so (no behaviour change).
- **Floor and sign:** the engine floors a tactile SA population's input at 0 mA before the unit;
  comparator noise is added after the floor; `u` may go below 0, the count never does.

## 4. The keys

`model_params` holds exactly the converter's constructor arguments: SensoryForge builds the
converter from it, `load_design` checks it at load by constructing the class (a key the class
does not take refuses the design), and the engine sets `dt` and `noise_std` itself. A design may
declare the sub-step and the floor it assumes as keys **beside** `model_params` in the
population record (`sub_step_ms`, `input_floor_ma`); SensoryForge does not read them (it ignores
population keys it does not know) and the reader checks them against what the bundle records.
The sign convention and the dead-time semantics are not keys: this page fixes them for
SensoryForge ≥ 1.3.0, and a bundle's `sensoryforge_version` and `sensoryforge_sha` say which
SensoryForge wrote it.

| Item | `design.json` (population record unless named) | Bundle | Units |
|---|---|---|---|
| converter | `neuron_model` | `config.json` `populations[i].encoder.model` | — |
| θ | `model_params.theta` | `encoder.params.theta`; RA also the `events` dataset's attribute `theta` | RA mA, SA mA·ms |
| SA leak τ | `model_params.leak_tau_ms` (absent or `null` = none) | `encoder.params.leak_tau_ms` | ms |
| RA reference leak τ (v1.3.0) | `model_params.reference_leak_tau_ms` (absent or `null` = none) | `encoder.params.reference_leak_tau_ms` (`null` = none; absent in bundles before 2.3.0) | ms |
| dead time | `model_params.refractory_ms` | `encoder.params.refractory_ms` | ms |
| RA initial reference | `model_params.initial_reference` | `encoder.params.initial_reference` | `"zero"` / `"first"` |
| sub-step δ | not read (the engine uses `simulation.integrate_dt_ms`); a design may declare `sub_step_ms` | `encoder.params.dt`; `config.simulation.integrate_dt_ms`; `data.h5` attribute `integrate_dt_ms` | ms |
| record step | `decisions.dt_ms` | `config.simulation.dt_ms`; `data.h5` attribute `dt_ms` | ms |
| input gain | `input_gain` | `config.populations[i].input_gain` (inside `config.json`'s `config`) | mA per stimulus unit |
| input floor (2.3.0) | not read (resolved, section 1 step 7); a design may declare `input_floor_ma` | `encoder.input_floor_ma`: the floor the converter received, mA, or `null` = none | mA |
| comparator noise | `membrane_noise_std` | `encoder.params.noise_std` | mA per sub-step |
| sensor (current) noise | `noise_std` | `neuron_modules/sensoryforge.json` `sensor_noise_std` | mA per bin |
| receptor noise | top level `receptor_noise_std`, `receptor_noise_seed` | `config.simulation.receptor_noise_std`, `receptor_noise_seed` | stimulus units |

## 5. Guarantees and the tests that pin them

| # | Guarantee | Test |
|---|---|---|
| 1 | With the leak off (`None` or absent) the level-crossing unit, and the sigma-delta unit, give state traces and events equal (`torch.equal`) to v1.2.1's forward loops, kept frozen in the test, on seeded ramps, steps, a signed multi-scale drive and white noise at σ/θ = 0.3, dead time 0 and 2 ms, both initial references, float32 and float64, and with comparator noise from a seeded RNG | `test_1_leak_off_is_v1_2_1_bit_for_bit_unit`, `test_1_sigma_delta_is_v1_2_1_bit_for_bit_unit`, `test_1_comparator_noise_is_drawn_as_in_v1_2_1` |
| 1 | `sensoryforge run --design` on a design with one level-crossing and one sigma-delta population, with receptor noise and a fixed seed, writes `drive`, `filtered`, `events` and `spikes` equal to a golden recorded at v1.2.1 (bit for bit on the recording machine, macOS arm64 with torch 2.5.1; SensoryForge's golden tolerance elsewhere, F-071), with the leak key absent and with it `null` | `test_1_leak_off_is_v1_2_1_bit_for_bit_engine` (golden: `tests/fixtures/converter_v1_2_1/`, recorded on `git archive v1.2.1` by `tests/fixtures/make_converter_v1_2_1_golden.py`) |
| 2 | A steady slope fires iff `s > θ(1 − 1e-5)/τ` (no event at `(θ/τ)(1 − 1e-3)` over 50τ; an event at `(θ/τ)(1 + 1e-3)`); for s from 1.3 to 20 times θ/τ the first event is at exactly `n*`; before it `uₙ` equals `(1 − λ)·sτ·(−expm1(n·log1p(−λ)))` to a relative 1e-9; `T′(1 − λ) ≤ n*δ < T′ + δ`; over a grid of θ and τ/δ from 2.5 to 400 | `test_2_a_steady_slope_fires_iff_faster_than_theta_over_tau` |
| 2 | With the drive held over each bin, a slope fires iff `s·dt_ms/(1 − ρ) ≥ θ(1 − 1e-5)`, and events fall on a bin's first sub-step | `test_2_with_held_bins_the_threshold_is_theta_one_minus_rho_over_dt` |
| 3 | A step of `Δx/θ` in {3, 3.5, 0.99, 1, −2, −4.25} fires `sign(Δx)·⌊|Δx|/θ + 1e-5⌋` on its sub-step and nothing over the next 50τ, leak off and on, float32; two steps of 0.6θ separated by 20τ make one ON event without the leak and none with it | `test_3_a_step_fires_its_quanta_at_once_then_falls_silent`, `test_3_the_leak_forgets_a_held_level` |
| 4 | Level-crossing, no leak, a ramp of `3θ/(Rδ)` then a hold: events of magnitude 1 exactly R sub-steps apart, continuing into the hold until `|x − r| < θ`, `⌊rise/θ + 1e-5⌋` in total; with a leak, after a step: event j at sub-step `1 + jR` exactly while `cⱼ ≥ θ(1 − 1e-5)`, and no other; sigma-delta, `δx = 3θ`: one spike every R sub-steps, accumulator ≤ θ after every sub-step, at most one more spike after the drive drops | `test_4_level_crossing_pays_a_fast_ramp_out_late_without_a_leak`, `test_4_level_crossing_with_a_leak_loses_part_of_a_pending_step`, `test_4_sigma_delta_dead_time_loses_charge_and_never_bursts` |
| 5 | Every bundle records `encoder.input_floor_ma`: 0.0 for a sigma-delta SA, `null` for a level-crossing RA, 0.0 for an AdEx SA (through `run --design`), 0.5 for an explicit 0.5, `null` for `-inf`, always the value the engine applied; `encoder.params` equals the unit's `to_dict()` (with `dt = integrate_dt_ms` and `reference_leak_tau_ms`); `schema_version` is 2.3.0 | `test_5_every_bundle_records_the_floor_each_converter_received`, `test_5_an_explicit_floor_is_recorded_as_applied` |
| 6 | A design whose RA `model_params` carries `reference_leak_tau_ms` loads, builds the unit with it and records it in the bundle; 0, negative, below `dt`, `inf`, `nan` and a string are refused at load naming the population (`populations[i]`); keys beside `model_params` (`sub_step_ms`, `input_floor_ma`, `encoder`) change nothing | `test_6_the_design_key_reaches_the_unit_and_the_bundle`, `test_6_a_bad_leak_is_refused_at_load_naming_the_population`, `test_6_keys_beside_model_params_are_ignored` |
| 7 | The same input and seed give identical events in two calls and in two processes; without comparator noise one neuron alone equals that neuron inside a population and a batch, bit for bit, leak on and off, dead time 0 and 2 sub-steps; with comparator noise and a fixed `noise_seed`, two engine runs are identical | `test_7_same_input_and_seed_give_the_same_events_in_two_calls_and_processes`, `test_7_one_neuron_alone_equals_it_in_a_population_and_a_batch`, `test_7_comparator_noise_with_a_noise_seed_is_reproducible` |
| 8 | The bad values of guarantee 6, and a bool, raise at construction; `None` and any finite τ ≥ `dt` are accepted; `to_dict()` carries τ and `from_config(to_dict())` is a fixed point; the neuron contract check passes with and without a leak; the parameter has a `ParamSpec` (ms, advanced) | `test_8_bad_leak_values_raise_at_construction`, `test_8_the_leak_is_accepted_from_dt_up_and_round_trips`, `test_8_the_leak_has_a_param_spec` |
| — | Without a leak the signed event sum times θ tracks a signed multi-scale drive within θ; a refractory period caps the rate; the sigma-delta rate is linear in a constant drive with no rheobase | `tests/unit/test_event_encoders.py` |
| — | Through the engine, level-crossing events are signed and stored as `events` (never `spikes`), its input is not floored, and `load_design` accepts both converters | `tests/integration/test_event_encoders_engine.py` |

A test named without a path is in `tests/contract/test_converter_contract.py`.

## 6. What v1.3.0 changes

- `LevelCrossingNeuron(reference_leak_tau_ms=None)`: line 5 of section 2. Default off; with
  `None` the unit is v1.2.1's, bit for bit.
- Bundle schema 2.3.0, additive: `encoder.input_floor_ma` on every population and
  `encoder.params.reference_leak_tau_ms` on level-crossing ones. `load_bundle` reads any 2.x
  bundle; a bundle from before 2.3.0 has neither key (read the leak's absence as `null`).
- The sigma-delta clip's words (section 3).

Unchanged: the sigma-delta unit; `resolve_input_floor` and every default it resolves (the floor
is only recorded); `load_design`, which reads no new key; the world engine's draws, frames,
seeds and data-set entries ([World engine contract](world_contract.md), guarantees 1–11); the
bundle beyond the two keys above.
