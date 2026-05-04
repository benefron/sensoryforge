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
from Pierzowski *et al.* (1995), where the input was skin-indentation force
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
    input_gain: 50       # compensates for Pierzowski N/mm² calibration
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
| `input_gain` | `PopulationConfig`, SpikingNeuronTab spinbox | 50 | Compensates for Pierzowski N/mm² filter calibration vs SensoryForge mA convention |
| SA filter `k1` | `SAFilterTorch.DEFAULT_CONFIG` | 0.05 | Pierzowski (1995) value — do not change |
| Stimulus `amplitude` | GUI Stimulus Designer | 1.0 mA | User-facing; increase to 3–10 for robust spiking at default gain |
