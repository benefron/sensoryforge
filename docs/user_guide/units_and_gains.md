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

The shipped tactile recipes do not use the default. Each population has its
own gain from `scripts/calibrate_recipe_gains.py`, and the two recipes follow
different aims:

| Recipe | Neurons | SA gain | RA gain |
|---|---|---|---|
| `tactile_sa1_ra1_adex` (AdEx) | no adaptation, 2 ms refractory period | 160 | 270 |
| `tactile_sa1_ra1` (Izhikevich) | TouchSim-fitted adaptation (`d` SA 15, RA 24) | 380 | 410 |
| `tactile_stochastic_control` | as `tactile_sa1_ra1` | 380 | 410 (so the control arm differs only in its receptive fields) |

**`tactile_sa1_ra1_adex` is pressure-simulation's model** (D-ce22df3,
D-8dde454, D-673e0ed). Its neurons are kept simple, with no spike-frequency
adaptation and no voltage clamp, so each rate follows its present drive, which
the Kalman-filter inference assumes. Both gains put the firing threshold at
10% of the benchmark stimulus:
- SA fires from 10% of the benchmark pressure;
- RA fires from 10% of its rate of change.

Above threshold the rates are roughly proportional. Measured with a 1 mm probe:
- SA's hold rate against pressure: 0, 25, 45, 76, 103, 137, 171, 214 Hz from 0.1 to 2.0 of the benchmark amplitude (R² 0.97).
- RA's onset rate against ramp speed: 63 to 208 Hz over 5 to 40 amplitude units per second (R² 0.86, flattening at the fastest ramps).

A held benchmark stimulus therefore fires SA at about 125 Hz, above P5's 20–100 Hz band. That is accepted, so that weak pressures are not invisible to SA.

**`tactile_sa1_ra1` is fitted to TouchSim's SA1 and RA afferent models**
(Saal et al. 2017; `benchmarks/results/touchsim_comparison/`, D-f4d0967,
D-d9bd411):
- SA's gain puts a held stimulus at 45 Hz, the centre of P5's band.
- RA's gain matches TouchSim's RA sensitivity.
- Its adaptation makes SA's rate rise gradually from near zero. Without it, Izhikevich is too steep: SA would be silent below about 60% of the benchmark pressure.

The shapes of the responses come from a separate fit
(`scripts/validation/fit_afferents.py`, results in
`benchmarks/results/afferent_fit/`), including the SA filter's `k2` (8.0, a
shared default). The TouchSim-fitted AdEx adaptation is available as the
opt-in presets `SA1_adapting` / `RA1_adapting` (`model_params: {preset:
SA1_adapting}`).

**RA answers release as strongly as indentation** (D-4e669b4). The RA filter
responds to the size of the rate of change (`|dI/dt|`), so lifting a probe
drives RA as much as pressing it. That suits the pressure-simulation design.
TouchSim's RA releases more weakly (0.67–0.8 of its onset) and is silent at
0.2 mm.

**A tactile afferent's neuron input is floored at 0 mA**
(`PopulationConfig.input_floor`; D-43dc520). A negative mechanical drive,
such as the SA filter's response to a trailing edge, silences the afferent
instead of hyperpolarizing it. The recorded `filtered` drive stays signed:
bundles store it, and pressure-simulation's decoder reads it. Other
populations get no floor (a DSL analog readout, the vision preset). Set
`input_floor: -.inf` to disable it.

**The voltage clamp is opt-in.** Izhikevich and AdEx default to
`v_floor: null` (no clamp). Set it (e.g. −120 mV) when integrating strongly
negative input at a coarse step, where the Euler update can blow up.

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
| `PopulationConfig.noise_seed` | Per population, in `populations[].noise_seed` | Reproduces that one population's noise (`sensor_noise_std` and/or `membrane_noise_std` > 0; the deprecated `noise_std` sets both) independently of the run seed and of every other population: `SimulationEngine.run()` builds a `torch.Generator` from it and passes it into `_run_pop_from_drive`, which both draws the sensor noise (post-filter, post-gain, additive on the current) from that generator and reseeds the global RNG from the same value immediately before the neuron call (the built-in neuron models' own Langevin noise, driven by `membrane_noise_std`, has no generator parameter of its own). `None` (default) leaves that population's noise on the ambient global RNG, uncontrolled by either seed. |
| `PopulationConfig.seed` | Per population, in `populations[].seed` | Seeds that population's innervation (receptive-field) wiring at build time. **F-006 is open**: this reseeds the *global* RNG and consumes it in a different order from pressure-simulation, so the same seed gives different wiring across the two repos; it does not affect noise or the neuron. |

None of the three imply the others. A fully reproducible noisy run needs the
run seed for anything that isn't independently seeded, plus a `noise_seed`
per population whose noise must be reproducible on its own (e.g. when
sweeping other settings while holding one population's noise fixed). With no
seed set anywhere, two runs of the same config are not expected to match —
the ambient RNG state is whatever the process happened to be in.
