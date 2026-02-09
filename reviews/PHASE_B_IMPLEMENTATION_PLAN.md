# Phase B Implementation Plan — YAML ↔ GUI Bidirectional Sync

**Date:** 2026-02-09  
**Prereq:** Phase A agents merged (ProtocolSuiteTab, CompositeGrid, Extended Stimuli, DSL/Solver)

---

## B.1 — Tab Config API (`get_config()` / `set_config()`)

### Approach

Add two public methods to each tab:

| Tab | `get_config() → dict` | `set_config(config: dict) → None` |
|-----|----------------------|-----------------------------------|
| MechanoreceptorTab | Grid type + params, population list | Set grid widgets, rebuild populations |
| StimulusDesignerTab | Stimulus type/params, texture/moving params | Set all stimulus widgets and visibility |
| SpikingNeuronTab | Per-population configs, solver, DSL, device | Set population configs, solver, DSL editor |

### YAML Structure (round-trip)

```yaml
metadata:
  version: "0.3.0"
  created: "2026-02-09T12:00:00"

grid:
  type: standard  # or composite
  rows: 40
  cols: 40
  spacing_mm: 0.15
  center: [0.0, 0.0]
  # composite-only:
  xlim: [-5.0, 5.0]
  ylim: [-5.0, 5.0]
  composite_populations:
    - name: sa1
      density: 100.0
      arrangement: grid
      filter: SA

populations:
  - name: "SA #1"
    neuron_type: SA
    neurons_per_row: 10
    connections_per_neuron: 28.0
    sigma_d_mm: 0.3
    weight_range: [0.1, 1.0]
    edge_offset: 0.0
    seed: 42
    color: [66, 135, 245, 255]

stimulus:
  name: "Default Stimulus"
  type: gaussian
  motion: static
  start: [0.0, 0.0]
  end: [0.0, 0.0]
  spread: 0.3
  orientation_deg: 0.0
  amplitude: 1.0
  speed_mm_s: 0.0
  ramp_up_ms: 50.0
  plateau_ms: 200.0
  ramp_down_ms: 50.0
  total_ms: 300.0
  dt_ms: 1.0
  texture:
    subtype: gabor
    wavelength: 0.5
    orientation_deg: 0.0
    sigma: 0.3
    phase: 0.0
    edge_count: 5
    edge_width: 0.05
    noise_scale: 1.0
    noise_kernel_size: 5
  moving:
    subtype: linear
    linear: {start: [0,0], end: [2,0], num_steps: 100, sigma: 0.3}
    circular: {center: [0,0], radius: 1.0, num_steps: 100, start_angle: 0, end_angle: 6.28, sigma: 0.3}
    slide: {start: [0,0], end: [2,0], num_steps: 100, sigma: 0.3}

simulation:
  device: cpu
  solver:
    type: euler
    method: dopri5
    rtol: 1.0e-05
    atol: 1.0e-07
  population_configs:
    "SA #1":
      model: Izhikevich
      filter_method: sa
      enabled: true
      input_gain: 1.0
      noise_std: 0.0
      model_params: {a: 0.02, b: 0.2, c: -65.0, d: 8.0}
      filter_params: {tau_r: 5, tau_d: 30, k1: 0.05, k2: 3.0}
      dsl_equations: ""
      dsl_threshold: ""
      dsl_reset: ""
      dsl_parameters: {}
```

### Implementation Order

1. MechanoreceptorTab.get_config() / set_config()
2. StimulusDesignerTab.get_config() / set_config()
3. SpikingNeuronTab.get_config() / set_config()
4. Rewrite main.py _save_config() / _load_config()

### Important Constraints

- set_config() must be called in order: mechano → stimulus → spiking (spiking depends on population names from mechano)
- All widgets must be updated with blockSignals(True) to avoid cascading signal side effects
- Missing keys should fall back to defaults (never crash on partial configs)
- Round-trip fidelity: save → load → save must produce identical YAML

---

## B.2 — Integration Tests

New file: `tests/integration/test_gui_phase2.py`

Test cases:
1. YAML save → load → re-save round-trip produces identical config
2. Each Phase 2 dropdown/control exists in the correct tab
3. Config loading populates all tab widgets correctly
4. Error handling for invalid/partial configs
5. ProtocolSuiteTab is wired as 4th tab
