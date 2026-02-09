# GUI Phase 2 Integration Plan

**Date:** 2026-02-09  
**Scope:** Integrate Phase 2 features (CompositeGrid, Equation DSL, Extended Stimuli, Adaptive Solvers) into the SensoryForge GUI  
**Source:** phase2_agent_tasks/TASK_GUI_CLI_PIPELINE_INTEGRATION.md

---

## Executive Summary

The GUI currently has **zero Phase 2 integration** — all 8,000+ lines use only
Phase 1 APIs. The work splits cleanly across 4 independent files (one per tab),
enabling parallel execution with a sequential merge phase for cross-cutting YAML
sync.

**Strategy: Parallel (Phase A) → Sequential (Phase B)**

---

## Current State

| Component | Phase 2 Status | Key Gap |
|-----------|---------------|---------|
| MechanoreceptorTab | None | No CompositeGrid support |
| StimulusDesignerTab | None | No texture/moving stimuli |
| SpikingNeuronTab | None | No DSL models, no solver selection |
| ProtocolSuiteTab | Dead code | Not wired into main window |
| NeuronExplorer | None | No DSL, no solver selection |
| main.py YAML I/O | Stub only | Load shows dialog, save writes template |
| default_params.json | Has Phase 2 section | Not read by any GUI code |

---

## Phase A — Parallel Tasks (4 agents)

### Agent A: Wire ProtocolSuiteTab + CompositeGrid in MechanoreceptorTab

**Files:** `sensoryforge/gui/main.py`, `sensoryforge/gui/tabs/__init__.py`, `sensoryforge/gui/tabs/mechanoreceptor_tab.py`

#### A.1 Wire ProtocolSuiteTab (small)
- Import `ProtocolSuiteTab` in `tabs/__init__.py` (add to `__all__`)
- Import in `main.py`, instantiate with refs to other 3 tabs
- Add as 4th tab: `tabs.addTab(self.protocol_tab, "Protocol Suite")`
- Instantiate `ProtocolExecutionController`, connect `run_requested` signal

#### A.2 CompositeGrid in MechanoreceptorTab (large)
- Add grid type radio/combobox: **Standard** (default) | **Composite**
- When Standard: existing UI unchanged
- When Composite:
  - Show shape + arrangement selectors per population
  - Population list with density (float), arrangement (poisson/hex/jittered_grid)
  - Use `CompositeGrid` from `sensoryforge.core.composite_grid`
  - Render multi-population scatter plot (color-coded by population)
- Update `_save_configuration()` and `_load_configuration()` to persist grid type + composite params
- Emit `grid_changed` signal with composite grid data
- Read `phase2.composite_grid` defaults from `default_params.json`

**Dependencies:** None  
**Risk:** Low — touches only its own tab + small main.py change

---

### Agent B: Extended Stimuli in StimulusDesignerTab

**Files:** `sensoryforge/gui/tabs/stimulus_tab.py`

#### B.1 Add texture stimulus type
- Add "Texture" button to stimulus type selector (alongside Gaussian/Point/Edge)
- When selected, show sub-type: gabor / grating / checkerboard / noise
- Parameter controls per sub-type:
  - Gabor: wavelength, orientation, sigma, phase, contrast
  - Grating: frequency, orientation, contrast
  - Checkerboard: check_size, contrast
  - Noise: noise_type (perlin/white/pink), scale
- Wire to `sensoryforge.stimuli.texture` module functions
- Generate preview using existing canvas infrastructure

#### B.2 Add moving stimulus type
- Add "Moving" button to stimulus type selector
- Sub-types: tap / slide / trajectory
- Parameter controls:
  - Tap: position, duration, amplitude
  - Slide: start/end position, speed, width
  - Trajectory: waypoints, speed
- Wire to `sensoryforge.stimuli.moving` module functions
- Animate preview using existing animation infrastructure

#### B.3 Wire new stimulus API
- Replace direct `sensoryforge.stimuli.stimulus` usage with new modular API
  where applicable (`sensoryforge.stimuli.gaussian`, etc.)
- Ensure backward compatibility with existing Gaussian/Point/Edge

**Dependencies:** None  
**Risk:** Low — isolated to stimulus_tab.py

---

### Agent C: DSL + Solver in SpikingNeuronTab

**Files:** `sensoryforge/gui/tabs/spiking_tab.py`

#### C.1 Equation DSL integration
- Add "DSL (Custom Equations)" to model combobox
- When selected, hide standard parameter table, show:
  - Equations: multi-line QPlainTextEdit
  - Threshold: single-line QLineEdit
  - Reset: single-line QLineEdit
  - Parameters: QTableWidget (name → value pairs)
- Pre-populate with Izhikevich equations as example
- "Compile" button → `NeuronModel(equations=...).compile(solver=...)`
- Show compilation errors in a status label
- Store compiled module for use in `_run_simulation()`

#### C.2 Solver selection
- Add solver dropdown in simulation config section: **Euler** (default) | **Adaptive**
- When Adaptive:
  - Method dropdown: dopri5 / bosh3 / adams
  - rtol: QDoubleSpinBox (default 1e-5)
  - atol: QDoubleSpinBox (default 1e-7)
- Instantiate via `sensoryforge.solvers.euler.EulerSolver` or
  `sensoryforge.solvers.adaptive.AdaptiveODESolver`
- Pass solver to neuron model creation / DSL compile

#### C.3 Wire solver into simulation
- Modify `_create_neuron_model()` to accept solver parameter
- Modify `_run_simulation()` to pass solver to forward()
- Update simulation status messages

**Dependencies:** None  
**Risk:** Medium — both features touch the same file, so must be one agent

---

### Agent D: Neuron Explorer Phase 2 Update

**Files:** `sensoryforge/gui/neuron_explorer.py`

#### D.1 DSL model in explorer
- Add "DSL (Custom)" to model selector
- Show equations editor panel when selected
- Compile and use for parameter sweeps

#### D.2 Solver selection in explorer
- Add solver dropdown (Euler/Adaptive)
- Pass to simulation loop

**Dependencies:** None (shares concepts with Agent C but separate file)  
**Risk:** Low — fully isolated

---

## Phase B — Sequential Tasks (after Phase A merges)

### B.1 Tab Config API + YAML ↔ GUI Bidirectional Sync

**Files:** ALL tab files + `main.py`

- Add `get_config() → dict` method to each tab
- Add `set_config(config: dict) → None` method to each tab
- Overhaul `_load_config()` in main.py:
  - Parse YAML → call each tab's `set_config()` with relevant section
  - Show summary after load
- Overhaul `_save_config()` in main.py:
  - Call each tab's `get_config()` → merge → write YAML
  - Generate valid config compatible with `GeneralizedTactileEncodingPipeline.from_config()`
- Ensure round-trip fidelity: save → load → save produces identical YAML

### B.2 Integration Tests

**Files:** New `tests/integration/test_gui_phase2.py`

- Test YAML load → verify tab state populated
- Test tab state → YAML save → reload → verify identical
- Test each Phase 2 dropdown/control exists and is functional
- Test ProtocolSuiteTab is wired and accepts run requests
- Test error handling for invalid configs

---

## Agent Task Specs (for remote assignment)

Each agent should receive:
1. This plan document (their section only)
2. The `SHARED_CONTEXT.md` from `phase2_agent_tasks/`
3. The `copilot-instructions.md` for coding conventions
4. The current file(s) they'll modify
5. The Phase 2 modules they'll integrate with (read-only)

### Merge Order
```
1. Agent A (main.py + mechanoreceptor_tab.py)
2. Agent D (neuron_explorer.py)           ← no conflicts
3. Agent B (stimulus_tab.py)              ← no conflicts
4. Agent C (spiking_tab.py)               ← no conflicts
5. Phase B.1 (all files — YAML sync)      ← after all Phase A
6. Phase B.2 (integration tests)          ← after B.1
```

---

## Acceptance Criteria

- [ ] All 4 Phase 2 features accessible from GUI dropdowns/controls
- [ ] ProtocolSuiteTab visible as 4th tab
- [ ] CompositeGrid creates multi-population grids with visualization
- [ ] Texture/moving stimuli generate preview and feed into simulation
- [ ] DSL equations compile and run in SpikingNeuronTab
- [ ] Solver selection (Euler/Adaptive) affects simulation
- [ ] YAML load populates all GUI controls
- [ ] YAML save captures all GUI state
- [ ] Round-trip: save → load → save produces identical config
- [ ] All existing tests still pass (307 passed, 3 skipped)
- [ ] New integration tests for GUI Phase 2 features
