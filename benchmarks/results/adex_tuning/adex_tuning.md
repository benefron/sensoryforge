# AdEx population tuning -- Phase 2b T2

Grid: 80x80 @ 0.15 mm | seed=0 | quick=False | input_gain used=50.0 | wall-clock=27.2s

## Responsive-set tuning targets (drive-derived, frac=0.5)

Responsive set defined per (stimulus, population, window) from the MEASURED drive alone (see module docstring); these are the numbers the AdEx `R` retune (Phase 2b T2b problem 2) is tuned against, not fitted after the fact.

| stimulus | population | window | n responsive | mean (mA) | p10 (mA) | p50 (mA) | p90 (mA) | peak (mA) |
|---|---|---|---|---|---|---|---|---|
| ramp_gaussian | SA Population | hold | 32 | 5.53 | 3.82 | 5.25 | 7.30 | 10.25 |
| ramp_gaussian | RA Population | onset | 32 | 3.08 | 1.16 | 3.04 | 4.78 | 5.67 |
| ramp_gaussian | RA Population | hold | 32 | 0.15 | 0.00 | 0.00 | 0.26 | 5.06 |
| moving_edge | SA Population | hold | 575 | 2.67 | -1.41 | 0.52 | 9.56 | 10.41 |
| moving_edge | RA Population | onset | 101 | 7.50 | 3.08 | 7.54 | 11.79 | 13.87 |
| moving_edge | RA Population | hold | 101 | 0.72 | 0.00 | 0.00 | 3.20 | 6.06 |
| braille | SA Population | hold | 24 | 2.21 | -1.32 | 0.35 | 7.91 | 9.76 |
| braille | RA Population | onset | 7 | 1.72 | 0.30 | 1.74 | 3.12 | 4.11 |
| braille | RA Population | hold | 7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| drifting_grating | SA Population | hold | 686 | 4.18 | -1.15 | 4.68 | 8.81 | 8.99 |
| drifting_grating | RA Population | onset | 450 | 1.80 | 0.68 | 1.93 | 2.78 | 3.08 |
| drifting_grating | RA Population | hold | 450 | 2.20 | 0.67 | 2.43 | 3.37 | 3.42 |

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
| AdEx SA1_tonic | 0.00 | 400.00 | 400.00 |
| AdEx RA1_phasic (steady) | 0.00 | 124.00 | 124.00 |
| AdEx RA1_phasic (first 30 ms) | 0.00 | 300.00 | 300.00 |
| Izhikevich RS | 0.00 | 58.00 | 58.00 |
| Izhikevich FS | 0.00 | 408.00 | 408.00 |

## Per-stimulus tables

### ramp_gaussian

Windows (ms): onset=(0.0, 30.0), hold/steady-drive=(50.0, 250.0) (scored), offset=None

| population | model | resp. n | total spikes | onset count | hold count | offset count | peak per-neuron Hz (5ms/2ms) | peak mean-per-neuron Hz (5ms/2ms) | pop. spike flux Hz (5ms) | SA mean rate resp. / whole-pop (Hz) | SA ISI CV (resp.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SA Population | Izhikevich | 32 | 352 | 12 | 76 | N/A | 200.00 / 500.00 | 100.00 / 125.00 | 3200.00 | 8.75 / 0.42 | 0.124 |
| SA Population | AdEx | 32 | 1820 | 0 | 396 | N/A | 200.00 / 500.00 | 175.00 / 312.50 | 5600.00 | 60.00 / 2.20 | 0.470 |
| RA Population | Izhikevich | 32 | 32 | 16 | 4 | N/A | 200.00 / 500.00 | 50.00 / 125.00 | 1600.00 | N/A / N/A | N/A |
| RA Population | AdEx | 32 | 12 | 4 | 0 | N/A | 200.00 / 500.00 | 50.00 / 125.00 | 1600.00 | N/A / N/A | N/A |

### moving_edge

Windows (ms): onset=(0.0, 30.0), hold/steady-drive=(70.0, 270.0) (N/A -- informational only), offset=(320.0, 330.0)

| population | model | resp. n | total spikes | onset count | hold count | offset count | peak per-neuron Hz (5ms/2ms) | peak mean-per-neuron Hz (5ms/2ms) | pop. spike flux Hz (5ms) | SA mean rate resp. / whole-pop (Hz) | SA ISI CV (resp.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SA Population | Izhikevich | 575 | 1680 | 121 | 1190 | 14 | 200.00 / 500.00 | 13.57 / 13.91 | 7800.00 | 9.60 / 6.61 | 0.018 |
| SA Population | AdEx | 575 | 5488 | 104 | 4132 | 65 | 200.00 / 500.00 | 42.78 / 45.22 | 24600.00 | 33.09 / 22.96 | 0.202 |
| RA Population | Izhikevich | 101 | 3698 | 354 | 2342 | 257 | 200.00 / 500.00 | 180.20 / 287.13 | 18200.00 | N/A / N/A | N/A |
| RA Population | AdEx | 101 | 1082 | 188 | 594 | 124 | 200.00 / 500.00 | 132.67 / 232.67 | 13400.00 | N/A / N/A | N/A |

### braille

Windows (ms): onset=(0.0, 30.0), hold/steady-drive=(350.0, 550.0) (N/A -- informational only), offset=(870.0, 900.0)

| population | model | resp. n | total spikes | onset count | hold count | offset count | peak per-neuron Hz (5ms/2ms) | peak mean-per-neuron Hz (5ms/2ms) | pop. spike flux Hz (5ms) | SA mean rate resp. / whole-pop (Hz) | SA ISI CV (resp.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SA Population | Izhikevich | 24 | 349 | 0 | 120 | 0 | 200.00 / 500.00 | 33.33 / 62.50 | 800.00 | 10.00 / 0.67 | 0.932 |
| SA Population | AdEx | 24 | 739 | 0 | 260 | 0 | 200.00 / 500.00 | 58.33 / 62.50 | 1400.00 | 24.58 / 1.44 | 1.644 |
| RA Population | Izhikevich | 7 | 604 | 0 | 210 | 0 | 200.00 / 500.00 | 57.14 / 71.43 | 400.00 | N/A / N/A | N/A |
| RA Population | AdEx | 7 | 173 | 0 | 60 | 0 | 200.00 / 500.00 | 28.57 / 71.43 | 200.00 | N/A / N/A | N/A |

### drifting_grating

Windows (ms): onset=(0.0, 30.0), hold/steady-drive=(400.0, 600.0) (N/A -- informational only), offset=(970.0, 1000.0)

| population | model | resp. n | total spikes | onset count | hold count | offset count | peak per-neuron Hz (5ms/2ms) | peak mean-per-neuron Hz (5ms/2ms) | pop. spike flux Hz (5ms) | SA mean rate resp. / whole-pop (Hz) | SA ISI CV (resp.) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SA Population | Izhikevich | 686 | 8818 | 0 | 2034 | 0 | 200.00 / 500.00 | 50.73 / 100.58 | 34800.00 | 12.00 / 11.30 | 0.126 |
| SA Population | AdEx | 686 | 34844 | 0 | 7996 | 0 | 200.00 / 500.00 | 125.95 / 249.27 | 86400.00 | 51.14 / 44.42 | 0.352 |
| RA Population | Izhikevich | 450 | 680 | 0 | 0 | 0 | 200.00 / 500.00 | 42.67 / 100.00 | 19200.00 | N/A / N/A | N/A |
| RA Population | AdEx | 450 | 0 | 0 | 0 | 0 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 | N/A / N/A | N/A |

## Sanity check: Izhikevich SA baseline under the corrected metric

Where the Izhikevich (not AdEx) SA population lands on the responsive-set metric -- if it still lands far below 20 Hz on a genuine hold (ramp_gaussian), that is a finding about the recipe's gain, not about AdEx.

- Izhikevich SA (ramp_gaussian, hold (scored)): responsive-set mean rate 8.75 Hz (n=32), whole-pop 0.42 Hz, ISI CV 0.124 -- BELOW the 20-100 Hz band.
- Izhikevich SA (moving_edge, steady-drive (N/A for P5)): responsive-set mean rate 9.60 Hz (n=575), whole-pop 6.61 Hz, ISI CV 0.018 -- BELOW the 20-100 Hz band.
- Izhikevich SA (braille, steady-drive (N/A for P5)): responsive-set mean rate 10.00 Hz (n=24), whole-pop 0.67 Hz, ISI CV 0.932 -- BELOW the 20-100 Hz band.
- Izhikevich SA (drifting_grating, steady-drive (N/A for P5)): responsive-set mean rate 12.00 Hz (n=686), whole-pop 11.30 Hz, ISI CV 0.126 -- BELOW the 20-100 Hz band.

## P5 criteria: PASS / FAIL / N-A

- SA-I (ramp_gaussian): **PASS** -- responsive-set (n=32) mean rate 60.00 Hz (target 20-100 Hz), ISI CV 0.470 (target < 0.5).
- SA-I (moving_edge): **N/A** -- no static hold in this stimulus (moving/drifting); reported as an informational steady-drive interval (responsive-set mean rate 33.09 Hz, ISI CV 0.202).
- SA-I (braille): **N/A** -- no static hold in this stimulus (moving/drifting); reported as an informational steady-drive interval (responsive-set mean rate 24.58 Hz, ISI CV 1.644).
- SA-I (drifting_grating): **N/A** -- no static hold in this stimulus (moving/drifting); reported as an informational steady-drive interval (responsive-set mean rate 51.14 Hz, ISI CV 0.352).
- RA-I silent-hold (ramp_gaussian): **PASS** -- 0 spikes in the (50.0, 250.0) ms hold window (target: 0).
- RA-I transient burst (ramp_gaussian): peak PER-NEURON instantaneous rate 200.00 Hz (best responsive afferent, 5 ms bin; mean-across-responsive-set peak 50.00 Hz) (target up to ~300 Hz, per afferent) -- PASS.
- RA-I silent-hold (moving_edge): **N/A** -- no static hold in this stimulus; steady-drive-interval spike count = 594.
- RA-I transient burst (moving_edge): peak PER-NEURON instantaneous rate 200.00 Hz (best responsive afferent, 5 ms bin; mean-across-responsive-set peak 132.67 Hz) (target up to ~300 Hz, per afferent) -- PASS.
- RA-I silent-hold (braille): **N/A** -- no static hold in this stimulus; steady-drive-interval spike count = 60.
- RA-I transient burst (braille): peak PER-NEURON instantaneous rate 200.00 Hz (best responsive afferent, 5 ms bin; mean-across-responsive-set peak 28.57 Hz) (target up to ~300 Hz, per afferent) -- PASS.
- RA-I silent-hold (drifting_grating): **N/A** -- no static hold in this stimulus; steady-drive-interval spike count = 0.
- RA-I transient burst (drifting_grating): peak PER-NEURON instantaneous rate 0.00 Hz (best responsive afferent, 5 ms bin; mean-across-responsive-set peak 0.00 Hz) (target up to ~300 Hz, per afferent) -- FAIL.

Pass bars used above: SA-I mean rate in 20-100 Hz with ISI CV < 0.5; RA-I silent hold = exactly 0 spikes; RA-I transient burst = a peak per-afferent rate of at least 150 Hz, i.e. half of P5's "up to ~300 Hz" (which is a ceiling, not a floor). Note that a peak of exactly 200 Hz is the 5 ms bin's cap for a neuron firing a SINGLE spike in that bin -- the RA "burst" here is one spike per responsive afferent, synchronized across the set, not a multi-spike burst within one afferent.

Note: RA's silence during a genuine hold is produced mainly by the RA filter differentiating the drive to ~0 during a static stimulus, not by AdEx adaptation alone -- evidenced by moving_edge's steady-drive interval, where the edge keeps moving (never static) and RA still fires 594 spikes in that window.

Note on SA1_tonic's R: R=6.0 was selected by scanning R and taking the smallest value whose responsive-set pooled ISI CV stayed under 0.5 on ramp_gaussian's hold window -- the reported ISI CV = 0.470 is therefore a fitted outcome of that scan, not an independent confirmation of the CV criterion. The purely principled placement (rheobase at the p10 of the measured hold drive, 3.82 mA) gives R ~= 4.8; R = 6.0 is within about 25% of that value, so most but not all of the choice is the scan rather than the drive-percentile principle alone.
