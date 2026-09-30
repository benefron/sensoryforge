# Event Encoders: Level-Crossing (RA) and Sigma-Delta (SA)

Two population models that are not membrane models. They turn a population's drive into
events with the two simplest analog-to-event circuits, so the RA/SA split can be studied as
a split of **time scales** -- RA reports fast signed change, SA reports slow absolute level --
rather than of biological mechanisms. They are opt-in arms next to the reference arms (AdEx /
Izhikevich neurons and the rectified RA filter), which are unchanged and remain the defaults.

| `neuron_model` | Class | Arm | Output key | Output |
|---|---|---|---|---|
| `level_crossing` | `LevelCrossingNeuron` | RA | `events` | signed counts: `+k` ON, `-k` OFF |
| `sigma_delta` | `SigmaDeltaNeuron` | SA | `spikes` | ordinary counts (>= 0) |

Both live in `sensoryforge/neurons/event_encoders.py`, are registered in
`register_components.py`, and take the engine's integration step `dt` (ms,
`integrate_dt_ms`) and a `noise_std` (Gaussian noise added to the input at every step, in the
drive's units; 0 by default) like every other neuron model.

## Level-crossing (`level_crossing`, RA)

An event-camera-style unit. Each neuron holds a reference level. It emits an **ON** event
when its input has risen by `theta` since its last event and an **OFF** event when it has
fallen by `theta`, and moves its reference by `theta` each time. The sign is carried by the
event, not recovered later.

```
d   = x - ref
k   = sign(d) * floor(|d| / theta)     # signed event count this step
ref = ref + k * theta
```

A step that moves the input by several `theta` emits several events at once, as one signed
count. The running sum of events times `theta` tracks the input to within `theta`, so the
encoding is invertible up to one quantum.

| Parameter | Default | Unit | Meaning |
|---|---|---|---|
| `theta` | 1.0 | mA (the drive's units, after gain) | input change per event |
| `refractory_ms` | 0 | ms | minimum interval between two events of one neuron; 0 = none (several events per step allowed) |
| `initial_reference` | `"zero"` | -- | `"zero"`: reference starts at 0; `"first"`: at the first input sample (report changes only) |
| `noise_std` | 0 | mA | comparator noise added to the input each step |

With a refractory period the unit emits at most one event per `refractory_ms` and the
reference still moves by one `theta` per event, so a change faster than
`theta / refractory_ms` is paid out late rather than lost (slew-rate limited).

**Input: the drive before any derivative filter.** Use `filter_method: none`. The unit
differentiates by construction; feeding it the rectified RA filter's output would
differentiate twice and throw away the OFF half. Its input is also **not floored** at 0 mA
(`resolve_input_floor` returns none for `level_crossing`), since OFF events live in the
falling half of the drive. `input_gain` still applies, so `theta` is in the gained units.

```yaml
populations:
  - name: RA events
    neuron_type: RA
    neuron_model: level_crossing
    filter_method: none
    input_gain: 5.0
    model_params: {theta: 0.2, refractory_ms: 1.0}
```

**Results are `events`, not `spikes`.** `SimulationEngine.run()` returns the per-bin signed
count under `"events"` (the net count of the bin's integration sub-steps; a bin holding both
an ON and an OFF event would net them, which needs the drive to swing by `2 * theta` within
one record step). Nothing that counts spikes will see an OFF event as a spike: the CLI prints
ON and OFF totals separately, and bundles store them in their own dataset (see
[Data Bundles](bundles.md)).

Measured (`tests/unit/test_event_encoders.py`, `pytest -s` prints them): a ramp of slope `s`
gives ON events at `s / theta` (0.019/ms for 0.020 expected over a 1 s ramp: one event short,
the last partial quantum) and none on the hold; a falling ramp gives OFF events at the same
rate; the signed sum times `theta` stays within 0.9999 `theta` of a signed multi-scale
drive; white noise of std `sigma` gives 0, 1e-6 and 1.8e-3 events per sample at
`sigma / theta` = 0.1, 0.2 and 0.3 (2.5e-2 at 0.4, 8.7e-2 at 0.5); a 2 ms refractory period
caps the rate at exactly 0.5/ms with a minimum interval of 2 ms.

## Sigma-delta (`sigma_delta`, SA)

A non-leaky integrate-and-fire unit with subtractive reset -- a first-order sigma-delta
modulator:

```
u = u + dt * x
n = floor(u / theta)          # spikes this step
u = u - n * theta             # subtract, never reset to zero
```

No charge is ever discarded, so the spike count up to time `t` is the integral of the drive
divided by `theta`, to within one spike. For a constant drive the rate is exactly
`x / theta` spikes per ms (`1000 * x / theta` Hz) with **no rheobase**, and the quantisation
error is first-order noise-shaped: pushed to high frequencies, so a low-pass of the spike
train recovers a slow drive.

| Parameter | Default | Unit | Meaning |
|---|---|---|---|
| `theta` | 100.0 | mA*ms | charge per spike; rate = drive / theta per ms (default: 10 Hz per mA) |
| `leak_tau_ms` | none | ms | optional accumulator leak; adds a rheobase `theta / leak_tau_ms` and breaks exact linearity |
| `refractory_ms` | 0 | ms | minimum inter-spike interval (caps the rate at `1 / refractory_ms`); while refractory the accumulator is clipped at `theta` (anti-windup) |
| `noise_std` | 0 | mA | noise added to the input each step |

A sigma-delta SA population also runs with `filter_method: none` (it integrates the level
itself); a tactile SA population's input keeps the default 0 mA floor.

Measured (`tests/unit/test_event_encoders.py`): constant drives of 0.2 to 20 mA at
`theta` = 10 mA*ms give rates equal to `drive / theta` to better than 1e-3 (0.01 mA still
fires at 0.001/ms: no rheobase). A 2 Hz sinusoid (1 to 9 mA, 100 to 900 Hz) low-passed with a
boxcar has RMS quantisation error 0.85, 0.39, 0.20, 0.083 and 0.041 mA at 5, 10, 20, 50 and
100 ms windows (about `theta / window`); against the raw drive the error is smallest near 50 ms
(0.095 mA, 2.4 % of the amplitude), after which the boxcar's own attenuation of the sinusoid
dominates. The spectrum of the quantisation error (spikes minus expected count) has mean power
4.3e-3, 12.6, 4.4e3 and 4.8e3 in the 1-10, 10-100, 100-1000 and 1000-5000 Hz bands: it rises
by two to three orders of magnitude per decade below the firing rate and is flat above it.

## Hardware view

Both are one-cycle-per-sample logic on an FPGA or ASIC: level-crossing is a reference
register, a comparator and a sign bit; sigma-delta is an accumulator, a comparator and a
subtractor. The address-event representation (AER) carries the polarity bit natively, so a
level-crossing event is one address plus one bit.

## In a design directory

`load_design` accepts `neuron_model: "level_crossing"` or `"sigma_delta"` with their
parameters in `model_params` (checked at load time). Because both run with
`filter_method: "none"`, the population's `neuron_type` is taken from the model
(`level_crossing` -> RA, `sigma_delta` -> SA) unless the record names one itself.
