# Units and Gains in SensoryForge

This guide explains how physical units flow through the simulation pipeline
and why the default `input_gain` is set to **50** rather than 1.

---

## The Unit Chain

Every simulation follows this pipeline:

```
GaussianStimulus (or other)
    amplitude = 1.0 mA (default)          [batch, time, H, W]
        ↓  Innervation module
        ↓  weights ∈ [0.1, 1.0]  (dimensionless)
        ↓  average weight ≈ 0.3 for neurons near the stimulus center
    drive ≈ 0.30 mA                        [batch, time, N_neurons]
        ↓  SAFilterTorch  (k1 = 0.05)
        ↓  steady-state ≈ k1 × drive = 0.015 mA
    filtered ≈ 0.015 mA                    [batch, time, N_neurons]
        ↓  input_gain (default = 50)
    effective input ≈ 0.75 mA             [batch, time, N_neurons]
        ↓  IzhikevichNeuronTorch
        ↓  threshold ≈ 3–5 effective current units
    spikes                                  [batch, time, N_neurons]  bool
```

**Time unit:** ms at all user-facing APIs; seconds in ODE integration.  
**Spatial unit:** mm throughout.  
**Current unit:** "mA" is a labelling convention — the Izhikevich equations
use dimensionless current that happens to be calibrated so that ~3–10 units
produce firing.

---

## Why the SA/RA Filters Scale Down the Signal

The SA and RA filter parameters (`k1`, `tau_r`, `tau_d`, etc.) were adapted
from Parvizi-Fard *et al.* (2021), where the input was skin-indentation force
in **N/mm²**.  A physiological indentation of 10 N/mm² through the SA filter
gives:

```
SA output ≈ k1 × 10 N/mm² = 0.05 × 10 = 0.5 mA
```

That 0.5 mA is roughly at the Izhikevich firing threshold — appropriate.

In SensoryForge, `amplitude` is measured in **mA** (a modelling convenience,
not a literal charge unit).  A default `amplitude = 1.0 mA` stimulus goes
through the innervation module (reducing it to ~0.3 mA) and then the SA
filter (reducing by another 20×), delivering only ~0.015 mA to the neuron.
That is 200× below the firing threshold.

**This is not a bug in the filter equations** — they are physiologically
correct for the scale they were designed for.  The mismatch is purely a
difference in input-axis convention between the original calibration and
SensoryForge's mA convention.

---

## The `input_gain` Parameter

`input_gain` is a multiplicative scalar applied to the filter output before
the neuron model:

```python
neuron_input = filter_output * input_gain + noise
```

It compensates for the N/mm² → mA convention gap.

### Default value: 50

A gain of 50 places the effective neuron input in the ~0.75–5 mA range for
default stimulus amplitudes (1–10 mA), which sits near the Izhikevich
regular-spiking threshold.

| Stimulus amplitude | SA filter output | × gain=50 | Izhikevich fires? |
|---|---|---|---|
| 0.5 mA | 0.008 mA | 0.40 mA | No |
| 1.0 mA | 0.015 mA | 0.75 mA | Borderline |
| 2.0 mA | 0.030 mA | 1.50 mA | Yes |
| 3.0 mA | 0.045 mA | 2.25 mA | Yes |
| 10.0 mA | 0.15 mA | 7.5 mA | Yes (strong) |

**Typical range:** 20–200.  If you are not seeing spikes, increase
`input_gain` first.  If you are seeing runaway high-frequency firing,
decrease it.

---

## Calibrated gains in the tactile recipes

The shipped tactile recipes do not use the default. Their populations are
fitted to TouchSim's SA1 and RA afferent models (Saal et al. 2017; see
`benchmarks/results/touchsim_comparison/`), and each population has its own
gain from `scripts/calibrate_recipe_gains.py`:

| Recipe | SA gain | RA gain |
|---|---|---|
| `tactile_sa1_ra1` (Izhikevich) | 380 | 410 |
| `tactile_sa1_ra1_adex` (AdEx) | 370 | 490 |
| `tactile_stochastic_control` | 380 | 410 (the same as `tactile_sa1_ra1`, so the control arm differs only in its receptive fields) |

- **SA:** the gain that puts SA's rate during a held stimulus
  (`ramp_gaussian`'s hold, over its responsive neurons) at the centre of P5's
  20–100 Hz band on a log scale (44.7 Hz), with an ISI CV below 0.5. SA now
  answers motion as SA1 does, at several times its hold rate, so the moving
  stimuli drive it harder (Izhikevich 83–86 Hz, AdEx 86–106 Hz); P5 states
  its band for a held stimulus. AdEx SA's hold is less regular than P5 asks
  (ISI CV about 0.5–0.6 against < 0.5): its ramp response builds adaptation
  that decays through the hold, so its intervals shorten as the hold goes on.
- **RA:** the gain whose onset rates best match TouchSim's RA at the same
  indentations, so RA fires at the small movements TouchSim's RA detects. RA
  must stay silent during a held stimulus. P5's 150–400 Hz burst band is
  reported, not required: on the fast `moving_edge` RA reaches 600 Hz
  (Izhikevich) and 400 Hz (AdEx, whose 2 ms refractory period caps it).

The shapes of the responses come from a separate fit
(`scripts/validation/fit_afferents.py`, results in
`benchmarks/results/afferent_fit/`): the SA filter's `k2` (8.0, a shared
default), spike-frequency adaptation (Izhikevich `d`: SA 15, RA 24, set in the
recipe; AdEx `SA1_tonic`/`RA1_phasic` presets). The gains, and the full
sweeps, are in `benchmarks/results/recipe_calibration/recipe_calibration.md`.

**The neuron's input is floored at 0 mA for tactile afferents**
(`PopulationConfig.input_floor`; D-43dc520). A negative mechanical drive,
such as the SA filter's response to a trailing edge, silences the afferent
instead of hyperpolarizing it. The recorded `filtered` drive stays signed,
which is what bundles store and pressure-simulation's decoder reads. Other
populations (a DSL analog readout, the vision preset) get no floor. Set
`input_floor: -.inf` to disable it.

**AdEx has a 2 ms refractory period** in the `SA1_tonic`/`RA1_phasic`
presets (`t_ref`; D-f5853a4), which caps its rate near 500 Hz.

**Known limit (AdEx).** After strong stimulation the AdEx recipe's
adaptation variable, which acts in mV, builds up to hundreds of mV, and the
voltage falls far below rest once the drive stops: to about −380 mV without
the −130 mV clamp. See the ledger's open entry before trusting AdEx dynamics
after strong stimulation. Izhikevich reaches its floor only briefly, with
spikes identical with and without the clamp.

---

## Tuning `input_gain` for Different Neuron Models

Different models have different effective thresholds:

| Model | Approx. threshold (effective current) | Suggested starting gain |
|---|---|---|
| Izhikevich (RS) | ~3 | 50 |
| AdEx | ~0.5 nA × R_m | 50–100 |
| MQIF | ~1–2 | 50 |
| LIF | depends on tau_m, R_m | 50–200 |

To calibrate for your setup:
1. Set `input_gain = 1`.  Confirm no spikes (this validates the filter is
   working and the drive is sub-threshold as expected).
2. Increase gain by 10× increments until you see reliable spiking.
3. Back off slightly to avoid saturation.

---

## Worked Example

```yaml
populations:
  - name: SA Pop
    filter_method: sa
    input_gain: 50       # compensates for Parvizi-Fard N/mm² calibration
    model: Izhikevich
    model_params:
      a: 0.02
      b: 0.2
      c: -65.0
      d: 8.0
```

With a Gaussian stimulus at `amplitude = 3.0 mA` and `sigma = 0.5 mm`:

1. Peak drive at the nearest neuron ≈ 3.0 × 0.8 (innervation weight) = 2.4 mA
2. SA filter steady-state ≈ 2.4 × 0.05 = 0.12 mA
3. After gain=50: 0.12 × 50 = 6 mA — well above threshold
4. Izhikevich RS neuron fires at ~25–40 Hz (typical for sustained input)

---

## Summary

| Parameter | Location | Default | Why |
|---|---|---|---|
| `input_gain` | `PopulationConfig`, GUI Populations → Readout & noise | 50 | Compensates for Parvizi-Fard N/mm² filter calibration vs SensoryForge mA convention |
| SA filter `k1` | `SAFilterTorch.DEFAULT_CONFIG` | 0.05 | Parvizi-Fard et al. (2021) value — do not change |
| Stimulus `amplitude` | `StimulusConfig`, GUI Stimulus screen | 1.0 for every stimulus type: a unit peak, pressure-simulation's convention. `repeated_pattern`'s six overlapping copies each peak at 1.0 and sum to about 3.9 | At gain 50 a unit peak is borderline (table above): raise `amplitude` or `input_gain` if a population is silent (ledger F-083) |

---

## Reproducibility (seeds) {: #reproducibility-seeds }

SensoryForge has three independent seeds, each covering a different source of
randomness (F-075):

| Seed | Location | Controls |
|---|---|---|
| `SimulationConfig.seed` (run seed) | `simulation.seed` in the canonical config, or `sensoryforge run --seed` | Seeds `torch`/`numpy`/`random` once, at the start of `SimulationEngine.run()`, before stimulus sampling and the population loop. The broadest knob — reproduces anything in the run that draws from the *global* RNG and isn't independently seeded below. |
| `PopulationConfig.noise_seed` | Per population, in `populations[].noise_seed` | Reproduces that one population's membrane noise (`noise_std > 0`) independently of the run seed and of every other population: `SimulationEngine.run()` builds a `torch.Generator` from it and passes it into `_run_pop_from_drive`, which both draws the post-filter additive noise from that generator and reseeds the global RNG from the same value immediately before the neuron call (the built-in neuron models' own Langevin noise, driven by the same `noise_std`, has no generator parameter of its own). `None` (default) leaves that population's noise on the ambient global RNG, uncontrolled by either seed. |
| `PopulationConfig.seed` | Per population, in `populations[].seed` | Seeds that population's innervation (receptive-field) wiring at build time. **F-006 is open**: this reseeds the *global* RNG and consumes it in a different order from pressure-simulation, so the same seed gives different wiring across the two repos; it does not affect noise or the neuron. |

None of the three imply the others. A fully reproducible noisy run needs the
run seed for anything that isn't independently seeded, plus a `noise_seed`
per population whose noise must be reproducible on its own (e.g. when
sweeping other settings while holding one population's noise fixed). With no
seed set anywhere, two runs of the same config are not expected to match —
the ambient RNG state is whatever the process happened to be in.
