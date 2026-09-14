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
   `SimulationEngine`. Today it does not (F-026).

## Rule: a default is not changed until every copy is changed

Filter and neuron defaults currently live in all of these places. Change them together, or better,
route them through one resolver (Phase 1 task A4):

- class signatures: `filters/sa_ra.py` (`SAFilterTorch`, `RAFilterTorch`, `CombinedSARAFilter`),
  `neurons/izhikevich.py`
- `core/generalized_pipeline.py` `DEFAULT_CONFIG` and the canonical adapter's fallbacks
- `config/default_config.yml`
- `gui/default_params.json` (drives the Spiking tab widgets **and** which values get exported)
- `gui/neuron_explorer.py`
- `examples/*.yml`, `tests/fixtures/*.yml`, `README.md`, `docs/user_guide/*.md`

## State as of 2026-09-14 (see `docs_root/LEDGER.md` for the full records)

- **F-001 closed:** `SAFilterTorch.clip_to_positive` defaults to `False` (sign-preserving SA, as in
  pressure-simulation). Rectification is opt-in. Do not re-enable it by default.
- **τ_RA = 8 ms is the decision (D-015)** but only the filter class, `CombinedSARAFilter` and
  `config/default_config.yml` carry it. The copies listed under F-026 still say 30 or 15.
- **RA k3 is undecided (F-030):** GUI 100, engine 2.0, pressure-simulation class/decoder 2.0 but its
  runner uses 1.0 for fast-spiking RA. Do not pick a value without the user.
- **F-004 open:** Izhikevich presets exist (`preset=` argument). `SimulationEngine` gives RA
  populations `preset="FS"`; the GUI still builds regular-spiking neurons for RA.
- **F-003 open:** the default innervation builder is the stochastic one (uniform-random weights,
  Gaussian only in the selection probability), which pressure-simulation retired to a control arm.
  The "Gaussian falloff" docstring in `innervation.py` is wrong. `use_distance_weights=True` is the
  analytic-weight path.
- **F-006 open:** innervation reseeds the *global* RNG and consumes it in a different order from
  pressure-simulation, so the same seed gives different wiring across repos.
- **F-025 open:** the canonical→legacy adapter squares neuron counts.

Filter citation is Parvizi-Fard et al. (2021) plus Kandel Ch.21 for τ values (D-013). Never write
"Pierzowski".

Never value-search-and-replace `0.3` — six different σ's share that number (pressure-simulation
`NAMING.md`). Edit by parameter name only.
