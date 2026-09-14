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
   `SimulationEngine`. It holds for default parameters; partial neuron overrides still diverge (F-031).

## Rule: defaults come from one resolver

Filter (SA `tau_r`/`tau_d`/`k1`/`k2`, RA `tau_RA`) and Izhikevich defaults are owned by
`sensoryforge/config/defaults.py` (`FILTER_DEFAULTS`, `NEURON_PRESET_BY_TYPE`,
`resolve_filter_params`, `resolve_neuron_params`). `SimulationEngine`, `SpikingNeuronTab` and the
canonical adapter in `core/generalized_pipeline.py` call it. Never add a new hard-coded copy.
Copies that still exist and must be kept equal until they are routed through the resolver:
`gui/default_params.json` (drift-tested in `tests/unit/test_config_defaults.py`),
`CombinedSARAFilter`'s default dict, the legacy `DEFAULT_CONFIG` neuron and filter keys, and
`TactileEncodingPipelineTorch` in `core/pipeline.py` (F-032).

## State as of 2026-09-14, after Wave A (see `docs_root/LEDGER.md`)

- **F-001 closed:** `SAFilterTorch.clip_to_positive` defaults to `False`. Do not re-enable by default.
- **τ_RA = 8 ms everywhere (D-015, F-026 closed).**
- **RA k3 is undecided (F-030, D-Q1):** the resolver does not own k3. GUI 100, engine 2.0,
  pressure-simulation class/decoder 2.0 but its runner 1.0. Do not pick a value without the user.
- **F-031 open:** overriding any one of `a`/`b`/`c`/`d` makes `resolve_neuron_params` drop the
  neuron-type preset, so GUI and engine diverge for RA populations and the legacy adapter raises
  `KeyError`. The preset must stay the base; overrides replace individual values.
- **F-032 open:** RS defaults for RA remain in `TactileEncodingPipelineTorch` and hand-written legacy
  configs.
- **F-003 open:** the default innervation builder is the stochastic one (uniform-random weights,
  Gaussian only in the selection probability), which pressure-simulation retired to a control arm.
  `use_distance_weights=True` is the analytic-weight path.
- **F-006 open:** innervation reseeds the *global* RNG and consumes it in a different order from
  pressure-simulation, so the same seed gives different wiring across repos.

Filter citation is Parvizi-Fard et al. (2021) plus Kandel Ch.21 for τ values (D-013). Never write
"Pierzowski".

Never value-search-and-replace `0.3` — six different σ's share that number (pressure-simulation
`NAMING.md`). Edit by parameter name only.
