---
paths:
  - "sensoryforge/filters/sa_ra.py"
  - "sensoryforge/core/innervation.py"
  - "sensoryforge/core/generalized_pipeline.py"
  - "sensoryforge/core/simulation_engine.py"
  - "sensoryforge/config/default_config.yml"
  - "sensoryforge/config/schema.py"
  - "sensoryforge/neurons/izhikevich.py"
  - "sensoryforge/gui/default_params.json"
  - "sensoryforge/gui/tabs/spiking_tab.py"
  - "sensoryforge/gui/neuron_explorer.py"
---

# Engine parity — read before editing these files

SensoryForge is the released forward model that `~/Documents/pressure simulation` (Paper B) cites.
Two parity contracts apply, and both must hold:

1. **SensoryForge ↔ pressure-simulation:** the same config + seed produces the same filtered
   response and spikes in both encoders.
2. **GUI ↔ CLI/engine:** a config exported from the GUI produces the same model when run through
   `SimulationEngine`, including partial neuron overrides (F-031, closed).

## Rule: defaults come from one resolver

Filter (SA `tau_r`/`tau_d`/`k1`/`k2`, RA `tau_RA`/`k3`) and Izhikevich defaults are owned by
`sensoryforge/config/defaults.py` (`FILTER_DEFAULTS`, `NEURON_PRESET_BY_TYPE`,
`resolve_filter_params`, `resolve_neuron_params`). `SimulationEngine`, `SpikingNeuronTab`, the
canonical adapter in `core/generalized_pipeline.py`, `TactileEncodingPipelineTorch`
(`core/pipeline.py`) and `CombinedSARAFilter` (`filters/sa_ra.py`) all call it. Never add a new
hard-coded copy. `resolve_neuron_params` always starts from a preset (the explicit `preset`
override, else `NEURON_PRESET_BY_TYPE` for the neuron type) and applies `a`/`b`/`c`/`d` overrides
on top -- never drop the other three when only one is overridden (F-031).
`gui/default_params.json`'s copy is drift-tested in `tests/unit/test_config_defaults.py`.

## State as of 2026-09-14, after Wave A + A8 (see `docs_root/LEDGER.md`)

- **F-001 closed:** `SAFilterTorch.clip_to_positive` defaults to `False`. Do not re-enable by default.
- **τ_RA = 8 ms everywhere (D-015, F-026 closed).**
- **D-Q1 decided: RA filter gain k3 = 2.0 everywhere (F-030 closed).** Resolver-owned. Matches both
  repos' `RAFilterTorch` class default, pressure-simulation's `config/pipeline_config.yml` and its
  decoder gain. pressure-simulation's `encode_runner.py` default of 1.0 and any RA input-gain preset
  retuning are follow-up work, not yet done.
- **F-031 closed:** `resolve_neuron_params` always expands the neuron-type (or explicit) preset
  first, then applies `a`/`b`/`c`/`d` overrides on top -- GUI, engine and the legacy adapter now
  agree for partial overrides too, and the adapter no longer raises `KeyError`.
- **F-032 closed:** `TactileEncodingPipelineTorch`, the legacy `DEFAULT_CONFIG`, and
  `CombinedSARAFilter` all route through the resolver instead of keeping their own RS/tau_RA=30 copies.
- **F-003 open:** the default innervation builder is the stochastic one (uniform-random weights,
  Gaussian only in the selection probability), which pressure-simulation retired to a control arm.
  `use_distance_weights=True` is the analytic-weight path.
- **F-006 open:** innervation reseeds the *global* RNG and consumes it in a different order from
  pressure-simulation, so the same seed gives different wiring across repos.

Filter citation is Parvizi-Fard et al. (2021) plus Kandel Ch.21 for τ values (D-013). Never write
"Pierzowski".

Never value-search-and-replace `0.3` — six different σ's share that number (pressure-simulation
`NAMING.md`). Edit by parameter name only.
