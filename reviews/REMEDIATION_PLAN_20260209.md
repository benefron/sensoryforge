# Remediation Plan

**Date:** 2026-02-09  
**Source:** reviews/REVIEW_AGENT_FINDINGS_20260209.md  
**Last Updated:** 2026-02-09  
**Status:** ✅ Complete — All 20 items resolved

## Progress Summary

| Status | Count |
|--------|-------|
| ✅ Completed | 20 |
| 🔲 Remaining | 0 |

---

## Completed

### Critical (C1–C2) — All Done ✅
1. **C1** ✅ — Vectorize Python loop in `innervation.py` innervation map construction  
   *Commit:* `e3987f8` — Replaced per-neuron Python loop with batched `torch.multinomial` + `scatter_`
2. **C2** ✅ — Convert noise modules to `nn.Module`, fix global RNG pollution  
   *Commit:* `6c13b98` — Both noise classes now inherit `nn.Module`, use per-instance `torch.Generator`

### High (H1–H8) — All Done ✅
3. **H1** ✅ — Create `BaseFilter`, `BaseNeuron`, `BaseStimulus` abstract base classes  
   *Commit:* `4d99409` — Three ABCs created, exported from package `__init__.py`, 13 tests
4. **H2** ✅ — Vectorize `get_weights_per_neuron()` in innervation  
   *Commit:* `e3987f8` — (combined with C1)
5. **H3** ✅ — Fix `get_grid_spacing()` axis computation  
   *Commit:* `2a4e5ac` — Corrected dim-0/dim-1 access for ij-indexed meshgrids, 4 tests
6. **H4** ✅ — Add `plotly` to dependencies  
   *Commit:* `bfd9237` — Added `plotly>=5.0` to `setup.py` and `requirements.txt`
7. **H5** ✅ — Fix global RNG pollution in noise modules  
   *Commit:* `6c13b98` — (combined with C2)
8. **H6** ✅ — Normalize neuron `forward()` signatures  
   *Commit:* `e0d84b6` — Renamed `x` → `input_current` in FA, added `reset_state()` to all models
9. **H7** ✅ — Vectorize stimulus generation loops  
   *Commit:* `bf1578e` — 5 stimulus types vectorized via tensor broadcasting
10. **H8** ✅ — Fix DSL `compile()` solver compatibility  
    *Commit:* `abdec16` — Accepts `str | BaseSolver`, verified I-symbol consistency

### Medium (M1–M10) — All Actionable Items Done ✅
11. **M1** ✅ — Fix Izhikevich `u_init` handling for tuple `b`  
    *Commit:* `fad1a1b` — Uses `.expand().clone()` instead of `torch.full()`
12. **M3** ✅ — Fix missing `gabor_texture` import  
    *Commit:* `d71d80c` — Added import to generalized pipeline
13. **M5** ✅ — Rename Poisson disk → jittered Poisson  
    *Commit:* `c26e82b` — Updated docstring to clarify approximation
14. **M6** ✅ — Fix `reset_filter_states` private attribute access  
    *Commit:* `d040645` — Added `clear_state()` methods, called from pipeline
15. **M7** ✅ — Fix kernel buffer shape mismatch  
    *Commit:* `f86f54f` — Keeps 4D shape in `update_parameters()`
16. **M8** ✅ — Fix adapter spike key overwrite  
    *Commit:* `e4cbb5c` — Prefixed keys with `adapter_`
17. **M9** ✅ — Remove unused `torchvision`/`torchaudio` dependencies  
    *Commit:* `bfd9237` — (combined with H4)
18. **M10** ✅ — Fix CLI `forward()` kwargs mismatch  
    *Commit:* `6d43cce` — Properly builds `stimulus_params` and passes `duration`

### Low (L1–L4) — All Done ✅
19. **L1–L4** ✅ — Grid type consistency, docstrings, redundant reset, top-level exports  
    *Commit:* `fd21708` — All four low-priority fixes in a single commit

### Report ✅
20. **Report** ✅ — `reviews/REMEDIATION_REPORT_20260209.md`
