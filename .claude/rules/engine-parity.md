---
paths:
  - "sensoryforge/filters/sa_ra.py"
  - "sensoryforge/core/innervation.py"
  - "sensoryforge/core/generalized_pipeline.py"
  - "sensoryforge/core/simulation_engine.py"
  - "sensoryforge/config/default_config.yml"
  - "sensoryforge/config/schema.py"
  - "sensoryforge/neurons/izhikevich.py"
  - "sensoryforge/gui/validation.py"
  - "sensoryforge/gui/execution/*.py"
  - "sensoryforge/gui/screens/populations_cards.py"
---

# Engine parity — read before editing these files

SensoryForge is the released forward model that `~/Documents/pressure simulation` (Paper B) cites.
Two parity contracts apply, and both must hold:

1. **SensoryForge ↔ pressure-simulation:** the same config + seed produces the same filtered
   response and spikes in both encoders.
2. **GUI ↔ CLI/engine:** a config exported from the GUI produces the same model when run through
   `SimulationEngine`, including partial neuron overrides.

## Rule: defaults come from one resolver

Filter (SA `tau_r`/`tau_d`/`k1`/`k2`, RA `tau_RA`/`k3`), neuron-preset and neuron-input-floor
defaults are owned by `sensoryforge/config/defaults.py` (`FILTER_DEFAULTS`,
`NEURON_PRESET_BY_TYPE`, `resolve_filter_params`, `resolve_neuron_params`, `resolve_input_floor`).
`SimulationEngine`, the canonical adapter in `core/generalized_pipeline.py`,
`TactileEncodingPipelineTorch` (`core/pipeline.py`) and `CombinedSARAFilter` (`filters/sa_ra.py`)
all call it; pressure-simulation's `design/drive.py` builds its filters from the same defaults, so a
change here changes its design-time drive model too. Never add a new hard-coded copy.
`resolve_neuron_params` always starts from a preset (the explicit `preset` override, else the
neuron type's preset) and applies single-parameter overrides on top -- never drop the others when
only one is overridden. The GUI v2 forms display these resolved values without copying them;
`tests/gui_v2` re-runs a config with every displayed value written explicitly and requires
identical output.

## Settled -- do not re-propose

- `SAFilterTorch.clip_to_positive` defaults to `False`: the recorded `filtered` drive stays signed,
  because pressure-simulation's decoder reads velocity sign from it. Negative drive is kept from the
  neuron by the neuron-input floor instead (0 mA for SA/RA/SA2 populations, D-43dc520).
- τ_RA = 8 ms and RA gain k3 = 2.0 everywhere (D-015, D-Q1). SA `k2` = 8.0, not Parvizi-Fard's 3.0,
  fitted to TouchSim's SA1 ramp response (D-f4d0967).
- **Two recipes, two aims.** `tactile_sa1_ra1_adex` is pressure-simulation's model: non-adapting
  AdEx (`SA1_tonic`/`RA1_phasic`, a = b = 0), so each rate follows its present drive, which the
  Kalman-filter inference assumes, with gains that put each threshold at 10% of the benchmark
  (SA 160, RA 270; D-ce22df3, D-8dde454, D-673e0ed). `tactile_sa1_ra1` keeps the TouchSim-fitted
  Izhikevich adaptation (`d` 15/24; SA 380 at P5's 45 Hz held rate, RA 410 at TouchSim's RA
  sensitivity). The AdEx adaptation is opt-in (`SA1_adapting`/`RA1_adapting`). Re-run
  `scripts/calibrate_recipe_gains.py` if a filter or neuron preset changes; do not reset gains to a
  shared 50.
- **No voltage clamp by default** on Izhikevich and AdEx (`v_floor` none, D-ce22df3), so
  SensoryForge's Izhikevich matches pressure-simulation's unclamped neurons.
- RA answers release as strongly as indentation (the RA filter is symmetric in the rate of change);
  kept by design (D-4e669b4).
- Connection weights default to the analytic Gaussian of distance (`use_distance_weights=True`);
  the uniform-random weights are the stochastic control arm (`use_distance_weights=False`).
- Innervation draws from a per-instance `torch.Generator` on CPU and never touches the global RNG.

Filter citation is Parvizi-Fard et al. (2021) plus Kandel Ch.21 for τ values (D-013). Never write
"Pierzowski".

Never value-search-and-replace `0.3` — six different σ's share that number (pressure-simulation
`NAMING.md`). Edit by parameter name only.
