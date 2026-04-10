# Units & Gains Audit — SensoryForge Pipeline
**Date:** 2026-04-10
**Scope:** `core/generalized_pipeline.py`, `filters/sa_ra.py`, `neurons/`, `core/innervation.py`

---

## Data-Flow Diagram with Annotated Units

```
Stimulus array [batch, T, H, W]
  dtype: float32
  range: [0, amplitude]  ← user sets amplitude (e.g. 30.0)
  unit: DIMENSIONLESS — no physical unit enforced at this stage
    │
    │  Processing layers (identity by default)
    ▼
Mechanoreceptor responses [batch, T, H, W]
  unit: dimensionless
    │
    │  InnervationModule / FlatInnervationModule
    │  Gaussian receptive field → weighted sum
    │  weight_range: SA=[0.05, 1.0], SA2=[0.4, 0.75]
    ▼
Neural drive [batch, T, N_neurons]
  unit: DIMENSIONLESS × stimulus_amplitude
  typical range: 0–30 (for amplitude=30, weight≈1.0)
    │
    ├── SA path: SAFilterTorch(tau_r, tau_d, k1=0.05, k2=3.0)
    │   Filter docstring claims input "in mA" — but unit is actually
    │   the above dimensionless product.
    │   Output: [batch, T, N_neurons], same dimensionless unit
    │
    ├── RA path: RAFilterTorch(tau_ra, k3=2.0)
    │   Same discrepancy: docstring says "mA", input is dimensionless.
    │
    └── SA2 path: sa2_scale × neural_drive   (scale = 0.005)
        Output ≈ 0.005 × 30 = 0.15 "units"
          │
          ▼
    Neuron input [batch, T, N_neurons]
      unit: DIMENSIONLESS (treated as mA by neuron docstrings)
      Izhikevich: dv = 0.04v² + 5v + 140 − u + I_t
                  (I_t treated as mA in the published equation)
```

---

## Key Findings

### F1 — Stimulus amplitude is dimensionless (no unit conversion)
**Severity: MEDIUM**

The stimulus generator produces values in `[0, amplitude]` with no physical unit.
`SAFilterTorch` and `RAFilterTorch` docstrings claim their input is "in mA", but they
receive dimensionless values. The filters are linear in their input, so this is a
gain rescaling issue, not a correctness bug per se — but it means the `k1`, `k2`, `k3`
gain parameters are calibrated to absorb the dimensional mismatch implicitly.

**Current behaviour:** Works because k1=0.05 attenuates the 0–30 range down to 0–1.5,
which is in a biologically plausible mA range for the neuron models.

**Recommended fix:** Document that "mA" in filter docstrings means "normalised units
proportional to mA". No code change required until absolute calibration is needed.

### F2 — SA2 scale factor (0.005) is unexplained magic number
**Severity: MEDIUM**

`sa2_filtered = 0.005 × sa2_inputs` at `generalized_pipeline.py:1416`.

SA2 neurons receive 60× less drive than SA/RA neurons. The SA/RA filters have gain
k1=0.05 applied over `tau_d=30 ms`, producing ~1.5× effective gain at steady state.
SA2 receives direct innervation × 0.005, so effective drive is about 6× lower than
SA steady state. This suppresses SA2 firing relative to SA/RA, matching the known
biology (SA2 = low-density, slowly adapting, weaker response).

**Recommended fix:** Add a comment in the code explaining the biological rationale.
Expose `sa2_scale` as a user-configurable param (already in config dict — just needs
documentation).

### F3 — Innervation weights provide implicit unit conversion
**Severity: LOW**

`weight_range=(0.05, 1.0)` for SA/RA and `(0.4, 0.75)` for SA2. These ranges are the
only "unit conversion" between dimensionless stimulus and neuron drive. They are not
documented as such.

**Recommended fix:** Add docstring to `InnervationModule` clarifying that
`weight_range` controls the effective gain of the stimulus-to-neuron mapping.

### F4 — dt mismatch between pipeline and filters
**Severity: HIGH — FIXED in this session (item 8)**

`SAFilterTorch` and `RAFilterTorch` both default to `dt=0.1 ms`. The pipeline
`DEFAULT_DT_MS` was 1.0 ms (GUI). The filter internal `dt` is not passed from
the pipeline config — each filter object uses its own `__init__` default.

**Status:** GUI `DEFAULT_DT_MS` lowered to 0.1 ms (committed). Filter dt defaults
already 0.1 ms — they are now consistent. However, `generalized_pipeline.py` does
not pass `dt` from `config["neurons"]["dt"]` to the filter constructors.
**Remaining action:** Verify filter construction in `_create_filters()` propagates
the pipeline dt to filter instances.

### F5 — Filter dt not propagated from pipeline config
**Severity: MEDIUM**
**File:** `generalized_pipeline.py:705-740`

```python
self.sa_filter = SAFilterTorch(
    tau_r=filter_cfg["sa_tau_r"],
    tau_d=filter_cfg["sa_tau_d"],
    k1=filter_cfg["sa_k1"],
    k2=filter_cfg["sa_k2"],
    # ← dt NOT passed — filter uses its own default (0.1 ms)
)
```

If a user sets `neurons.dt = 0.5 ms`, the filter still integrates at 0.1 ms.
This inconsistency is currently benign (filter default matches corrected GUI default)
but is architecturally fragile.

**Recommended fix:** Pass `dt=config["neurons"]["dt"]` to all filter constructors
in `_create_filters()`.

---

## Summary Table

| Finding | Severity | Status | Action Required |
|---------|----------|--------|----------------|
| F1: Stimulus dimensionless vs filter "mA" | MEDIUM | Open | Add documentation; no code fix needed |
| F2: SA2 scale=0.005 unexplained | MEDIUM | Open | Add comment + docs |
| F3: Innervation weights as implicit unit conversion | LOW | Open | Add docstring |
| F4: dt mismatch GUI vs neurons | HIGH | **FIXED** | Monitor; re-test |
| F5: Filter dt not propagated from pipeline config | MEDIUM | Open | Pass dt to filter constructors |

---

## Recommended Next Steps

1. **Fix F5** (filter dt propagation) — one-line fix per filter constructor call in `_create_filters()`
2. **Document F1/F2/F3** in filter and innervation docstrings
3. **Add integration test** that asserts filter.dt == pipeline.dt after construction
