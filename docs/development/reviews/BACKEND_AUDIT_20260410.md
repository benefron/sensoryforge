# Backend Feature Audit: protocol_backend.py vs generalized_pipeline.py
**Date:** 2026-04-10
**Scope:** `gui/protocol_backend.py` (GUI source of truth) vs `core/generalized_pipeline.py` (shared CLI/Batch backend)
**Purpose:** C3-Step1 — identify features to port from GUI into the shared backend

---

## Executive Summary

The GUI backend (`protocol_backend.py`) is significantly more feature-complete than the
shared backend (`generalized_pipeline.py`). Key gaps exist in stimulus generation,
population handling, and filter/neuron configuration. Severity: High for unification.

---

## Feature Gaps: GUI has, Shared backend lacks

### 1. Multiple Population Support (protocol-level)
- **Feature**: Multi-population execution with independent configurations per population
- **Location**: `protocol_backend.py:261-425` (`_execute_packets` loop over `self._populations`)
- **Generalized Status**: Partial — hardcoded SA/RA/SA2 only, no dynamic pop count
- **Severity**: HIGH — blocks true N-population parity

### 2. Protocol-Based Stimulus Packing (10+ Stimulus Protocols)
- **Feature**: `_pack_slow_gaussians`, `_pack_fast_gaussians`, `_pack_shear_sweeps`, `_pack_rotating_edges`, `_pack_center_surround`, `_pack_moving_gaussians`, `_pack_macro_gaussians`, `_pack_edge_steps`, `_pack_point_pulses`, `_pack_large_disk_pulses`, `_pack_texture_sequences`
- **Location**: `protocol_backend.py:811–1298`
- **Generalized Status**: NO — pipeline only has 8 generic generators (trapezoidal, gaussian, step, ramp, texture, moving, timeline, repeated_pattern)
- **Severity**: MEDIUM — GUI richness, not required for core simulation

### 3. Per-Population Configuration Dictionaries
- **Feature**: `_population_configs` dict mapping population name to `{model, filter_method, input_gain, noise_std, model_params, filter_params}`
- **Location**: `protocol_backend.py:426-438` (`_prepare_population_config`)
- **Generalized Status**: NO — pipeline neurons/filters hardcoded as SA/RA/SA2
- **Severity**: HIGH — essential for flexible population variation

### 4. Dynamic dt and Solver Injection
- **Feature**: Per-packet dt resolution with solver override; supports `model_params['dt']` + fallback to `packet_dt_ms`
- **Location**: `protocol_backend.py:440-446` (`_resolve_dt_ms`); `689-724` (`_create_neuron_model` with registry + signature inspection)
- **Generalized Status**: Partial — pipeline accepts dt in config but no per-population solver objects
- **Severity**: MEDIUM — dt is configured but solver type selection incomplete

### 5. Tau Extraction & Neuron Model Inspection
- **Feature**: Exhaustive `tau_ms` extraction from multiple attribute aliases (`tau_m`, `tau_mem`, `tau_leak`, `tau_s`); fallback to override
- **Location**: `protocol_backend.py:448-480` (`_extract_tau_ms`)
- **Generalized Status**: NO — no equivalent
- **Severity**: LOW — instrumentation only

### 6. Input Gain Per Population
- **Feature**: Per-population `input_gain` scaling applied post-filter: `drive = filtered * gain`
- **Location**: `protocol_backend.py:547-548`
- **Generalized Status**: NO — fixed scale only; no input_gain knob
- **Severity**: MEDIUM — affects encoding sensitivity

### 7. Filter Instantiation via Registry at Runtime
- **Feature**: Dynamic filter creation with `FILTER_REGISTRY.get_class(method)` + fallback; signature-based kwarg filtering
- **Location**: `protocol_backend.py:636-673` (`_apply_filter`)
- **Generalized Status**: Partial — filters created at init time, not per-packet; registry lookups present but less flexible
- **Severity**: LOW — filters are created; timing differs

### 8. Noise Std Per Population
- **Feature**: Per-packet `noise_std` from config, applied as Gaussian perturbation: `drive = drive + torch.randn_like(drive) * noise_std`
- **Location**: `protocol_backend.py:549-554`
- **Generalized Status**: Partial — membrane noise modules exist but no per-packet `noise_std` override
- **Severity**: MEDIUM — noise control important for robustness studies

### 9. RunResult Data Container
- **Feature**: Structured output `RunResult` with `spike_counts, drive_tensor, filtered_tensor, unfiltered_tensor, spike_tensor, stimulus_tensor` per population
- **Location**: `protocol_backend.py:80-93` (`RunResult`); `394-406`
- **Generalized Status**: Different — flat dict output, no wrapper
- **Severity**: LOW — semantic difference; functionally equivalent

### 10. Fine-Grained Debug Instrumentation
- **Feature**: 20+ debug calls with statistics (shape, min/max/mean, gain, dt resolution, filter state)
- **Location**: Throughout `protocol_backend.py`
- **Generalized Status**: Minimal — basic logging only
- **Severity**: LOW — dev tooling

---

## Porting Priority Order

| Priority | Item | Why |
|----------|------|-----|
| 1 | Item 3: Per-population config dict | Foundation for all other N-pop features |
| 2 | Item 1: N-population loop | Core parity with GUI |
| 3 | Item 6: Input gain per population | Affects all encoding results |
| 4 | Item 8: Per-population noise_std | Robustness and variability |
| 5 | Item 4: Dynamic solver injection | dt/solver consistency |
| 6 | Item 2: Stimulus protocols | Port 10+ protocols after core parity |

---

## Batch Executor (batch_executor.py) Summary

`BatchExecutor` delegates entirely to `GeneralizedTactileEncodingPipeline.from_config()`.
It inherits all the same gaps as the generalized pipeline. No additional features found.
No dt override logic — uses whatever `config["neurons"]["dt"]` is set to.

---

## Next Steps (C3-Step2)

Write integration parity test that runs the same config through GUI protocol_backend
and generalized_pipeline, and asserts spike counts are within tolerance. This test
will fail initially (due to population handling differences) and will pass once the
backend is unified.
