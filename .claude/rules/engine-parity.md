---
paths:
  - "sensoryforge/filters/sa_ra.py"
  - "sensoryforge/core/innervation.py"
  - "sensoryforge/config/default_config.yml"
  - "sensoryforge/config/schema.py"
  - "sensoryforge/neurons/izhikevich.py"
---

# Engine parity with pressure-simulation — read before editing these files

SensoryForge is the released forward model that `~/Documents/pressure simulation` (Paper B) cites.
The two encoders **must produce the same spikes for the same config + seed**. Open findings
(see `docs_root/LEDGER.md`):

- **F-001** `SAFilterTorch(clip_to_positive=True)` rectifies SA here; pressure-simulation does not and
  its decoder recovers velocity sign from SA. Do not add more rectification; decision pending.
- **F-002** τ_RA: here 30 ms (class) / 15 ms (YAML); pressure-simulation locked **8 ms** (Kandel
  Ch.21, its commit `0ee0653`). Do not "fix" one copy without the other two and the docstring.
- **F-003** The default innervation is the *stochastic* builder (uniform-random weights, Gaussian
  only in the selection probability). Pressure-simulation has retired it to a control arm; the
  docstring "Gaussian falloff" at `innervation.py:880` is wrong. `use_distance_weights=True` is the
  analytic-weight path.
- **F-005** Filter equations are **Parvizi-Fard et al. 2021** (as `sa_ra.py` says). "Pierzowski
  (1995)" in CLAUDE.md / docs is unverifiable — do not propagate it.
- **F-006** Seeding: innervation reseeds the *global* RNG and consumes it in a different order from
  pressure-simulation; same seed ⇒ different wiring across repos.

Never value-search-and-replace `0.3` — six different σ's share that number (pressure-simulation
`NAMING.md`). Edit by parameter name only.
