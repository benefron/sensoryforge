# Bayesian Implementation Agent Instructions

**Version:** 2.0 – Event-Based Kalman Filter  
**Updated:** January 2026

You are an expert AI software engineer specializing in Bayesian inference, State-Space Models (SSM), and PyTorch. You are tasked with implementing the Event-Based Kalman Filter framework for the `pressure-simulation` project.

## Core Directives

1.  **Follow the Plan**: Your primary sources of truth are:
    *   `plans/event-based-kalman-filter-design-review.md` (Resolved Design Decisions)
    *   `docs_root/BAYESIAN_AGENT_PLAN.md` (Implementation Roadmap)
    *   `docs_root/BAYESIAN_SSM_DECODER_DESIGN.md` §8.4 (Event-Based KF Specification)
    
    Do not deviate from the phases defined there without explicit user approval.

2.  **Context Awareness**: Before writing any code, you MUST read the following files to ground your understanding:
    *   `plans/event-based-kalman-filter-design-review.md` (Design decisions and resolutions)
    *   `docs_root/BAYESIAN_SSM_DECODER_DESIGN.md` (Formal definitions: H, F, Q, R, Event-Based KF)
    *   `docs_root/SA_RA_DECODER_ANALYSIS.md` (SA/RA population roles and complementary information)
    *   `docs_root/SCIENTIFIC_HYPOTHESIS.md` (Biological principles and central hypothesis)
    *   `config/pipeline_config.yml` (The Configuration)
    *   `encoding/innervation_torch.py` (The Data Source)

3.  **Memory Management**: If you feel you are losing context or "wandering", stop and re-read:
    *   `plans/event-based-kalman-filter-design-review.md` for resolved design decisions
    *   `docs_root/BAYESIAN_SSM_DECODER_DESIGN.md` §8.4 for Event-Based KF specification

## Event-Based Kalman Filter Core Concepts

### Architecture Overview

The Event-Based Kalman Filter uses RA activity to gate spatially-localized Q/R modulation:

1. **RA as gating mechanism**: RA spiking determines WHERE and WHEN to use fast-dynamics parameters
2. **Spatially localized updates**: Only active regions receive fast-dynamics treatment
3. **Sign inference from SA**: RA is unsigned; sign comes from SA temporal trend (with RA phase as fallback)
4. **Dual observation matrices**: H_SA → amplitude, H_RA → velocity (after sign correction)
5. **SA anchoring**: SA continues updating even when RA inactive (slow mode)

### Architectural Invariants

**CRITICAL: These must always hold:**

1. **F is FIXED**: The constant velocity dynamics matrix never changes. Adaptation happens via Q/R only.
2. **RA never sets state directly**: RA modulates Q and R, defines spatial masks, but never overwrites amplitude or velocity values.
3. **SA-only fallback**: Setting R_RA → ∞ must yield a stable SA-only Kalman filter that correctly reconstructs amplitude.
4. **No hard reset**: Inactive regions are NOT reset to zero. State evolves via prediction (or freezes); P grows.
5. **Block-diagonal P**: Acknowledged as efficiency approximation (cells treated as independent).

### Key Design Decisions (Resolved)

| Decision | Resolution |
|----------|------------|
| Dynamics F | **FIXED** constant velocity: [I, Δt·I; 0, I]. Never changes. |
| RA activation | Per-neuron spiking (threshold built in) + RF projection to grid |
| Mode transition | Exponential decay with τ ≈ 50ms |
| Sign inference | LOCAL per grid cell: SA_local = W_SA.T @ z_SA; d(SA_local)/dt provides sign; RA phase fallback when SA ambiguous |
| Mask smoothing | Gaussian blur with σ = innervation RF sigma (preserves spatial prior) |
| Q/R values | Linear interpolation based on mask value |
| Q interpretation | Absorbs unmodeled acceleration |
| R interpretation | Reflects measurement trust |
| Prediction scope | Full grid (global) |
| Update scope | Localized to active regions only |
| Inactive regions | State frozen or evolves via prediction. P grows. No hard reset to zero. |
| SA in slow mode | SA continues updating even when RA inactive |

### Critical Constraints

- **F is FIXED** – never adapt the dynamics matrix
- **RA never sets amplitude/velocity directly** – only modulates Q and R
- **Spatial prior (σ) must be preserved** – use innervation σ for mask smoothing
- **Sign inference before update (LOCAL, per grid cell)** – SA neurons (Izhikevich) fire rate ≥ 0; stop firing when stimulus decreases. Derivative gives direction:
  ```python
  # Project SA to grid (LOCAL, not global!)
  SA_local = W_SA.T @ z_SA  # [N_grid]
  SA_derivative = (SA_local - SA_local_prev) / dt
  SA_trend = alpha * SA_derivative + (1 - alpha) * SA_trend_prev  # [N_grid]
  
  # Per-grid-cell sign
  for j in range(N_grid):
      if abs(SA_trend[j]) >= threshold:
          sign[j] = np.sign(SA_trend[j])
      else:
          sign[j] = +1 if RA_is_onset[j] else -1  # fallback
  v_signed = sign * abs(v_unsigned)  # [N_grid]
  ```
- **Spiking regime** – observations are binned spike rates, not continuous currents
- **SA continues in slow mode** – RA absence does NOT disable SA updates
- **No hard reset** – inactive regions evolve via prediction only, P grows

## Execution Workflow

Execute the following phases in order. Do not proceed to the next phase until the current one is complete and verified.

### Phase 1: Documentation & Theory (COMPLETED)
*   ✅ Updated `docs_root/BAYESIAN_SSM_DECODER_DESIGN.md` with Event-Based KF spec
*   ✅ Updated `docs_root/SA_RA_DECODER_ANALYSIS.md` with new architecture
*   ✅ Updated `docs_root/BAYESIAN_AGENT_PLAN.md` with implementation phases
*   ✅ Created `plans/event-based-kalman-filter-design-review.md` with resolutions

### Phase 2: Core Implementation
*   **Task**: Implement the Event-Based Kalman Filter engine.
*   **Files to Create/Edit**:
    *   `decoding/event_based_kalman.py`: Main `EventBasedKalmanFilter` class
    *   `decoding/ssm_models.py`: Add `EventBasedTactileSSM` if needed
*   **Key Methods**:
    *   `compute_activity_mask(z_RA)`: Project RA activity to grid
    *   `update_mask_decay(mask, dt)`: Exponential temporal decay
    *   `compute_sa_trend(z_SA)`: EMA derivative for sign inference
    *   `get_signed_ra(z_RA, sa_trend)`: Apply sign correction
    *   `interpolate_noise(mask)`: Compute local Q and R
    *   `predict()`: Global prediction step
    *   `update(z_k, mask)`: Localized update step
*   **Critical Checks**:
    *   Does mask computation use W_RA.T @ relu(z_RA)?
    *   Does mask smoothing use the spatial prior σ?
    *   Is sign inference applied before Kalman update?
    *   Are Q/R interpolated based on mask value?

### Phase 3: Integration
*   **Task**: Wire into pipeline.
*   **Files to Edit**:
    *   `config/pipeline_config.yml`: Add `event_based_kalman` section
    *   `decoding/pipeline.py`: Add factory method
    *   `decoding/modules/__init__.py`: Export new class
*   **Critical Check**: Can pipeline instantiate `EventBasedKalmanFilter` from config?

### Phase 4: Testing
*   **Task**: Verify correctness.
*   **Files to Create**:
    *   `tests/test_event_based_kalman.py`: Unit tests
    *   `examples/test_event_based_kalman.py`: Demo script
*   **Test Cases**:
    *   Mask computation from RA spikes
    *   Mask temporal decay
    *   SA trend computation
    *   Sign inference (SA trend and RA phase fallback)
    *   Q/R interpolation
    *   Localized vs global updates
    *   **Architectural invariant**: Verify SA-only mode (R_RA=∞) reconstructs correctly
    *   **No hard reset**: Verify inactive regions maintain state estimate

### Phase 5: Sparse Event-Driven Implementation (Future)
*   **Task**: Enable scalability for large grids and hardware deployment.
*   **Files to Create**:
    *   `decoding/sparse_event_kalman.py`: `SparseEventBasedKalman` class
*   **Key Concepts**:
    *   Track only active cells (set of indices)
    *   Per-cell state/covariance (dictionary storage)
    *   Cell activation from RA spikes, retirement when mask decays
    *   O(N_active) complexity instead of O(N_grid²)
*   **Assumptions**:
    *   Block-diagonal P (cells are independent)
    *   Dormant cells return to prior
*   **Reference**: `written_outcomes/Event_Based_Kalman_Filter_Technical_Document.md` §14

## Technical Constraints

*   **Imports**: Use absolute imports (e.g., `from decoding.modules.base import AnalyticalModule`).
*   **Type Hinting**: Use strict type hints (`torch.Tensor`, `Dict`, `Optional`).
*   **Style**: Follow existing coding style (docstrings, variable naming).
*   **Error Handling**: Fail gracefully if config is missing.
*   **Tensor Broadcasting**: Use PyTorch vectorized ops, no loops over neurons.
*   **Sparse Compatibility**: Dense and sparse implementations must produce identical outputs.

## Default Parameter Values

| Parameter | Value | Description |
|-----------|-------|-------------|
| bin_size_ms | 10-20 | Spike binning window |
| τ_decay | 50ms | Mask exponential decay |
| α (SA trend) | 0.2 | EMA smoothing factor |
| Q_amplitude_slow | 1e-6 | Slow dynamics process noise |
| Q_amplitude_fast | 1e-3 | Fast dynamics process noise |
| Q_velocity_slow | 1e-4 | Slow velocity noise |
| Q_velocity_fast | 1e-1 | Fast velocity noise |
| R_SA_slow | 0.01 | SA measurement noise (slow) |
| R_SA_fast | 0.1 | SA measurement noise (fast) |
| R_RA_slow | 10.0 | RA measurement noise (slow) |
| R_RA_fast | 0.1 | RA measurement noise (fast) |

## Recovery Instructions

If you encounter an error or get stuck:
1.  **Stop**.
2.  **Read** the error message carefully.
3.  **Check** the relevant file content using `read_file`.
4.  **Refer** back to these documents:
    *   `plans/event-based-kalman-filter-design-review.md` for design decisions
    *   `docs_root/BAYESIAN_SSM_DECODER_DESIGN.md` §8.4 for specification
5.  **Ask** the user for clarification if the plan is ambiguous.

## Document Reference Quick Guide

| Document | Purpose | When to Reference |
|----------|---------|-------------------|
| `event-based-kalman-filter-design-review.md` | Resolved design decisions | Before implementing any component |
| `BAYESIAN_SSM_DECODER_DESIGN.md` §8.4 | Formal Event-Based KF specification | Writing math, implementing matrices |
| `SA_RA_DECODER_ANALYSIS.md` | SA/RA roles, sign inference | Designing observation model |
| `BAYESIAN_AGENT_PLAN.md` | Implementation phases, checklist | Tracking progress, next steps |
| `Event_Based_Kalman_Filter_Technical_Document.md` | Full technical spec with pseudocode | Implementation details, sparse algorithm |
