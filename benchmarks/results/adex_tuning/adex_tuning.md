# AdEx population tuning -- Phase 2b T2

Grid: 80x80 @ 0.15 mm | seed=0 | quick=False | input_gain used=50.0 | wall-clock=26.9s

## Measured filtered-drive current range (mA)

| stimulus | population | min | median | p95 | max |
|---|---|---|---|---|---|
| ramp_gaussian | SA Population | 0.00 | 0.00 | 2.36 | 10.25 |
| ramp_gaussian | RA Population | 0.00 | 0.00 | 0.00 | 5.79 |
| moving_edge | SA Population | -2.49 | 0.00 | 9.70 | 11.94 |
| moving_edge | RA Population | 0.00 | 0.12 | 5.97 | 22.71 |
| braille | SA Population | -1.95 | -0.00 | 0.15 | 9.97 |
| braille | RA Population | 0.00 | 0.00 | 0.68 | 8.55 |
| drifting_grating | SA Population | -1.73 | 2.94 | 8.88 | 9.81 |
| drifting_grating | RA Population | 0.00 | 2.26 | 3.41 | 4.95 |

20 constant-current f-I levels chosen: linspace(0, 27.26, 20) mA (1.2x the largest measured max across every stimulus x population).

## f-I curve summary (steady-state, last 500 ms of a 1000 ms run)

| model | min rate (Hz) | max rate (Hz) | rate at max current (Hz) |
|---|---|---|---|
| AdEx SA1_tonic | 0.00 | 40.00 | 40.00 |
| AdEx RA1_phasic (steady) | 0.00 | 0.00 | 0.00 |
| AdEx RA1_phasic (first 30 ms) | 0.00 | 0.00 | 0.00 |
| Izhikevich RS | 0.00 | 58.00 | 58.00 |
| Izhikevich FS | 0.00 | 408.00 | 408.00 |

## Per-stimulus tables

### ramp_gaussian

Windows (ms): onset=(0.0, 30.0), hold/steady-drive=(50.0, 250.0) (scored), offset=None

| population | model | total spikes | onset count | hold count | offset count | peak inst. rate (Hz, 5ms bins) | SA mean rate (Hz) | SA ISI CV |
|---|---|---|---|---|---|---|---|---|
| SA Population | Izhikevich | 352 | 12 | 76 | N/A | 3200.00 | 0.42 | 0.124 |
| SA Population | AdEx | 0 | 0 | 0 | N/A | 0.00 | 0.00 | N/A |
| RA Population | Izhikevich | 32 | 16 | 4 | N/A | 1600.00 | N/A | N/A |
| RA Population | AdEx | 0 | 0 | 0 | N/A | 0.00 | N/A | N/A |

### moving_edge

Windows (ms): onset=(0.0, 30.0), hold/steady-drive=(70.0, 270.0) (N/A -- informational only), offset=(320.0, 330.0)

| population | model | total spikes | onset count | hold count | offset count | peak inst. rate (Hz, 5ms bins) | SA mean rate (Hz) | SA ISI CV |
|---|---|---|---|---|---|---|---|---|
| SA Population | Izhikevich | 1680 | 121 | 1190 | 14 | 14400.00 | 6.61 | 0.018 |
| SA Population | AdEx | 0 | 0 | 0 | 0 | 0.00 | 0.00 | N/A |
| RA Population | Izhikevich | 3698 | 354 | 2342 | 257 | 38000.00 | N/A | N/A |
| RA Population | AdEx | 0 | 0 | 0 | 0 | 0.00 | N/A | N/A |

### braille

Windows (ms): onset=(0.0, 30.0), hold/steady-drive=(350.0, 550.0) (N/A -- informational only), offset=(870.0, 900.0)

| population | model | total spikes | onset count | hold count | offset count | peak inst. rate (Hz, 5ms bins) | SA mean rate (Hz) | SA ISI CV |
|---|---|---|---|---|---|---|---|---|
| SA Population | Izhikevich | 349 | 0 | 120 | 0 | 1000.00 | 0.67 | 0.955 |
| SA Population | AdEx | 0 | 0 | 0 | 0 | 0.00 | 0.00 | N/A |
| RA Population | Izhikevich | 604 | 0 | 210 | 0 | 1200.00 | N/A | N/A |
| RA Population | AdEx | 0 | 0 | 0 | 0 | 0.00 | N/A | N/A |

### drifting_grating

Windows (ms): onset=(0.0, 30.0), hold/steady-drive=(400.0, 600.0) (N/A -- informational only), offset=(970.0, 1000.0)

| population | model | total spikes | onset count | hold count | offset count | peak inst. rate (Hz, 5ms bins) | SA mean rate (Hz) | SA ISI CV |
|---|---|---|---|---|---|---|---|---|
| SA Population | Izhikevich | 8818 | 0 | 2034 | 0 | 36000.00 | 11.30 | 0.609 |
| SA Population | AdEx | 0 | 0 | 0 | 0 | 0.00 | 0.00 | N/A |
| RA Population | Izhikevich | 680 | 0 | 0 | 0 | 28000.00 | N/A | N/A |
| RA Population | AdEx | 0 | 0 | 0 | 0 | 0.00 | N/A | N/A |

## P5 criteria: PASS / FAIL / N-A

- SA-I (ramp_gaussian): **FAIL** -- mean rate 0.00 Hz (target 20-100 Hz), ISI CV N/A (target < 0.5).
- SA-I (moving_edge): **N/A** -- no static hold in this stimulus (moving/drifting); reported as an informational steady-drive interval (mean rate 0.00 Hz, ISI CV N/A).
- SA-I (braille): **N/A** -- no static hold in this stimulus (moving/drifting); reported as an informational steady-drive interval (mean rate 0.00 Hz, ISI CV N/A).
- SA-I (drifting_grating): **N/A** -- no static hold in this stimulus (moving/drifting); reported as an informational steady-drive interval (mean rate 0.00 Hz, ISI CV N/A).
- RA-I silent-hold (ramp_gaussian): **PASS** -- 0 spikes in the (50.0, 250.0) ms hold window (target: 0).
- RA-I transient burst (ramp_gaussian): peak instantaneous rate 0.00 Hz (target up to ~300 Hz) -- FAIL.
- RA-I silent-hold (moving_edge): **N/A** -- no static hold in this stimulus; steady-drive-interval spike count = 0.
- RA-I transient burst (moving_edge): peak instantaneous rate 0.00 Hz (target up to ~300 Hz) -- FAIL.
- RA-I silent-hold (braille): **N/A** -- no static hold in this stimulus; steady-drive-interval spike count = 0.
- RA-I transient burst (braille): peak instantaneous rate 0.00 Hz (target up to ~300 Hz) -- FAIL.
- RA-I silent-hold (drifting_grating): **N/A** -- no static hold in this stimulus; steady-drive-interval spike count = 0.
- RA-I transient burst (drifting_grating): peak instantaneous rate 0.00 Hz (target up to ~300 Hz) -- FAIL.
