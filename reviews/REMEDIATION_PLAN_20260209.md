# Remediation Plan

**Date:** 2026-02-09  
**Source:** reviews/REVIEW_AGENT_FINDINGS_20260209.md  
**Last Updated:** 2026-02-09

## Progress Summary

| Status | Count |
|--------|-------|
| ✅ Completed | 8 |
| 🔲 Remaining | 12 |

---

## Completed

### Critical (C1–C2) — All Done ✅
1. **C1** ✅ — Vectorize Python loop in `innervation.py` innervation map construction  
   *Commit:* `e3987f8` — Replaced per-neuron Python loop with batched `torch.multinomial` + `scatter_`
2. **C2** ✅ — Convert noise modules to `nn.Module`, fix global RNG pollution  
   *Commit:* `6c13b98` — Both noise classes now inherit `nn.Module`, use per-instance `torch.Generator`

### High (H1–H5) — Done ✅
3. **H1** ✅ — Create `BaseFilter`, `BaseNeuron`, `BaseStimulus` abstract base classes  
   *Commit:* `4d99409` — Three ABCs created, exported from package `__init__.py`, 13 tests
4. **H2** ✅ — Vectorize `get_weights_per_neuron()` in innervation  
   *Commit:* `e3987f8` — (combined with C1)
5. **H3** ✅ — Fix `get_grid_spacing()` axis computation  
   *Commit:* `2a4e5ac` — Corrected dim-0/dim-1 access for ij-indexed meshgrids, 4 tests
6. **H4** ✅ — Add `plotly` to dependencies  
   *Commit:* (build dep commit) — Added `plotly>=5.0` to `setup.py` and `requirements.txt`
7. **H5** ✅ — Fix global RNG pollution in noise modules  
   *Commit:* `6c13b98` — (combined with C2)

### Medium — Partial
17. **M9** ✅ — Remove unused `torchvision`/`torchaudio` from dependencies  
    *Commit:* (build dep commit) — Removed from both `setup.py` and `requirements.txt`

---

## Remaining

### High (H6–H8)
8. **H6** 🔲 — Normalize neuron `forward()` signatures  
   *Details:* Rename `x` → `input_current` in `FANeuronTorch`, add `reset_state()` to all models
9. **H7** 🔲 — Vectorize stimulus generation loops in `generalized_pipeline.py`  
   *Details:* Replace `for t_idx in range(n_timesteps)` with broadcasting
10. **H8** 🔲 — Fix DSL I-symbol consistency in `model_dsl.py`  
    *Details:* Ensure `Symbol('I', real=True)` used consistently in `_create_lambdas()`

### Medium (M1–M8, M10)
11. **M1** 🔲 — Fix Izhikevich `u_init` handling for tuple `b`  
    *Details:* `torch.full` fails when fill value is a tensor; use `.expand()` instead
12. **M3** 🔲 — Fix missing `gabor_texture` import in generalized pipeline  
    *Details:* `NameError` at runtime for `stimulus_type="texture"` with `pattern="gabor"`
13. **M5** 🔲 — Rename Poisson disk → jittered Poisson in `CompositeGrid`  
    *Details:* Docstring says "Poisson disk" but implementation is jittered grid
14. **M6** 🔲 — Fix `reset_filter_states` private attribute access in `pipeline.py`  
    *Details:* Directly manipulates child module internals instead of calling reset methods
15. **M7** 🔲 — Fix `MechanoreceptorModule.update_parameters` buffer shape mismatch  
    *Details:* Assigns 2D tensor to 4D buffer, breaking subsequent `F.conv2d`
16. **M8** 🔲 — Fix `TactileSpikingNetwork` key overwrite  
    *Details:* `results.update(pipeline_results)` overwrites adapter spike outputs
18. **M10** 🔲 — Fix CLI `forward()` kwargs mismatch  
    *Details:* `duration` passed as kwarg but `forward()` doesn't accept it

### Low (L1–L4)
19. **L1–L4** 🔲 — Grid spacing type consistency, docstring completeness, redundant reset, top-level exports

### Final
20. **Report** 🔲 — Write `reviews/REMEDIATION_REPORT_20260209.md`
