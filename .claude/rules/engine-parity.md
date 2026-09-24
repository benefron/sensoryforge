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
The GUI v2 forms display these resolved values without copying them; `tests/gui_v2`
re-runs a config with every displayed value written explicitly and requires identical output.

## State as of 2026-09-14, after Wave A + A8 (see `docs_root/LEDGER.md`)

- **F-001 closed:** `SAFilterTorch.clip_to_positive` defaults to `False`. Do not re-enable by default.
- **τ_RA = 8 ms everywhere (D-015, F-026 closed).**
- **D-Q1 decided: RA filter gain k3 = 2.0 everywhere (F-030, F-034 closed).** Resolver-owned here.
  pressure-simulation matches since its commit `f7784f9` (runner, viewer and decoder fallbacks).
  Its RA input gains were tuned at k3 = 1.0 and are tracked in its own ledger, not here.
- **F-037 open:** SensoryForge's Izhikevich/AdEx/MQIF clamp voltage at `v_floor` (D-007);
  pressure-simulation's neurons do not, so spikes can differ for strongly negative drive.
- **F-031 closed:** `resolve_neuron_params` always expands the neuron-type (or explicit) preset
  first, then applies `a`/`b`/`c`/`d` overrides on top -- GUI, engine and the legacy adapter now
  agree for partial overrides too, and the adapter no longer raises `KeyError`.
- **F-032 closed:** `TactileEncodingPipelineTorch`, the legacy `DEFAULT_CONFIG`, and
  `CombinedSARAFilter` all route through the resolver instead of keeping their own RS/tau_RA=30 copies.
- **F-003 closed:** connection weights default to the analytic Gaussian of distance
  (`use_distance_weights=True`, `config/schema.py`). The uniform-random weights pressure-simulation
  retired are still reachable with `use_distance_weights=False`, as the stochastic control arm.
- **F-006 closed:** innervation draws from a per-instance `torch.Generator`
  (`core/innervation.py::_seeded_generator`, on CPU per F-038) and never touches the global RNG.

Filter citation is Parvizi-Fard et al. (2021) plus Kandel Ch.21 for τ values (D-013). Never write
"Pierzowski".

Never value-search-and-replace `0.3` — six different σ's share that number (pressure-simulation
`NAMING.md`). Edit by parameter name only.
