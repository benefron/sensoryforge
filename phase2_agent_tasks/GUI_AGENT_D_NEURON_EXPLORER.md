# Agent D: Neuron Explorer Phase 2 Update

**Priority:** Phase A (parallel)  
**Files to modify:** `sensoryforge/gui/neuron_explorer.py`  
**Estimated complexity:** Medium  
**Dependencies:** None — fully independent of Agents A, B, C

---

## Pre-requisites

Read these documents first:
1. `phase2_agent_tasks/SHARED_CONTEXT.md` — coding standards, project overview
2. `.github/copilot-instructions.md` — full coding conventions, docstring style, commit format
3. `reviews/GUI_PHASE2_INTEGRATION_PLAN.md` — overall plan (your section is "Agent D")

---

## Context

The `NeuronExplorer` is a **standalone window** (not a tab) that provides an 
interactive neuron model explorer with matplotlib plots. It supports:
- All 5 neuron models (Izhikevich, AdEx, MQIF, FA, SA)
- Stimulus types: step, sine, ramp, trapezoid
- Parameter sweeps and single-neuron response visualization

Phase 2 added the **Equation DSL** and **Solver infrastructure** which are not 
exposed in the explorer.

---

## Phase 2 APIs (read-only reference)

### Equation DSL (`sensoryforge.neurons.model_dsl`)

```python
from sensoryforge.neurons.model_dsl import NeuronModel

model = NeuronModel(
    equations="dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms\ndu/dt = (a * (b*v - u)) / ms",
    threshold="v >= 30 * mV",
    reset="v = c\nu = u + d",
    parameters={"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
)

# Compile to nn.Module
module = model.compile(solver="euler", dt=0.5, num_neurons=1, device="cpu")

# forward(input_current) → (spikes, state_dict)
spikes, state = module(current_tensor)
```

### Solvers (`sensoryforge.solvers`)

```python
from sensoryforge.solvers.euler import EulerSolver
from sensoryforge.solvers.adaptive import AdaptiveSolver

# Euler
solver = EulerSolver(dt=0.5)

# Adaptive (requires torchdiffeq)
solver = AdaptiveSolver(method="dopri5", dt=0.5, rtol=1e-5, atol=1e-7)
```

---

## default_params.json Phase 2 Section

```json
{
  "phase2": {
    "dsl_neuron": {
      "template": {
        "equations": "dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms\ndu/dt = (a * (b*v - u)) / ms",
        "threshold": "v >= 30 * mV",
        "reset": "v = c\nu = u + d",
        "parameters": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0}
      }
    },
    "solvers": {
      "euler": {"dt": 0.001},
      "adaptive": {"method": "dopri5", "rtol": 1e-5, "atol": 1e-7}
    }
  }
}
```

---

## Current NeuronExplorer Architecture

Read `sensoryforge/gui/neuron_explorer.py` (936 lines) for full details.

```
NeuronExplorer(QMainWindow)
├── Model selector: QComboBox ["Izhikevich", "AdEx", "MQIF", "FA", "SA"]
├── Parameter panel: dynamic param controls per model
├── Stimulus panel: type (step/sine/ramp/trapezoid), amplitude, frequency, duration
├── Simulation controls: dt, duration, device, run button
├── Plot area: matplotlib canvas with voltage trace, spike markers
├── Status bar: simulation time, spike count
└── Key methods:
    ├── _create_model() → nn.Module from selected model + params
    ├── _generate_stimulus() → current trace Tensor
    ├── _run_simulation() → run model over stimulus, collect traces
    ├── _update_plots() → render voltage/spike plots
    └── _on_model_changed() → swap parameter controls
```

---

## What To Do

### D.1: Add DSL Model to Explorer

1. **Add "DSL (Custom Equations)" to model selector combobox**.

2. **Create DSL editor panel** (shown when DSL selected, replacing standard params):
   - **Equations**: `QPlainTextEdit`, monospaced, 4-5 lines
   - **Threshold**: `QLineEdit`
   - **Reset**: `QLineEdit`
   - **Parameters**: `QTableWidget` (Name, Value columns)
     - "Add Parameter" / "Remove Parameter" buttons
   - **Compile button** + **status label**
   - Pre-populate with Izhikevich template from `default_params.json`

3. **Toggle in `_on_model_changed()`**:
   - Standard models: show normal parameter controls, hide DSL editor
   - DSL: hide standard controls, show DSL editor

4. **Implement `_compile_dsl()`**:
   ```python
   def _compile_dsl(self) -> None:
       try:
           from sensoryforge.neurons.model_dsl import NeuronModel
           equations = self.dsl_equations.toPlainText()
           threshold = self.dsl_threshold.text()
           reset = self.dsl_reset.text()
           params = self._get_dsl_params()
           self._dsl_model = NeuronModel(
               equations=equations, threshold=threshold,
               reset=reset, parameters=params,
           )
           self.dsl_status.setText("✓ Ready")
           self.dsl_status.setStyleSheet("color: green;")
       except Exception as e:
           self._dsl_model = None
           self.dsl_status.setText(f"✗ {e}")
           self.dsl_status.setStyleSheet("color: red;")
   ```

5. **Update `_create_model()`** to handle DSL:
   ```python
   if model_name == "DSL (Custom Equations)":
       if self._dsl_model is None:
           raise ValueError("Compile the DSL model first")
       return self._dsl_model.compile(
           solver=self._get_solver(),
           dt=self.dt,
           num_neurons=1,
           device=self.device,
       )
   ```

### D.2: Add Solver Selection

1. **Add solver dropdown** in simulation controls:
   - `QComboBox`: ["Euler", "Adaptive (RK45)"]
   - Default: "Euler"

2. **Adaptive config controls** (visible when Adaptive selected):
   - Method: `QComboBox` ["dopri5", "bosh3", "adaptive_heun"]
   - rtol: `QDoubleSpinBox` (1e-5)
   - atol: `QDoubleSpinBox` (1e-7)

3. **Implement `_get_solver()`**:
   ```python
   def _get_solver(self) -> str:
       if self.cmb_solver.currentText().startswith("Euler"):
           return "euler"
       return "adaptive"
   ```

4. **Note for built-in models**: Solver selection affects DSL models directly.
   For built-in models, show info tooltip: "Built-in models use native integration."

### D.3: DSL Preset Templates

Add a "Load Template" button with preset DSL templates:
- Izhikevich (Regular Spiking)
- Izhikevich (Chattering)
- Izhikevich (Fast Spiking)
- Simple LIF (Leaky Integrate-and-Fire)

Each template pre-fills the equations, threshold, reset, and parameters fields.

---

## Key Constraints

- **DO NOT change existing model behavior** — all 5 models must work as before.
- **Handle DSL compilation errors gracefully** — display error, don't crash.
- **Handle missing `torchdiffeq`** — show install hint if adaptive solver selected 
  but package unavailable.
- **This is a standalone window** — no signal connections to other tabs needed.
- **Read defaults from `default_params.json`**.

---

## Verification

- Launch explorer → model dropdown shows 6 options
- Select DSL → equations editor appears with Izhikevich template
- Click Compile → shows success
- Run simulation → voltage trace renders correctly
- Load Template > Simple LIF → equations swap to LIF model
- Select Adaptive solver → method/rtol/atol appear
- Built-in models still work unchanged

---

## Commit Format

```
feat(gui): add DSL model and solver selection to NeuronExplorer
```

## Tests

Add a minimal test in `tests/unit/test_gui_agent_d.py` that:
- Imports `NeuronExplorer` without error (or handles Qt unavailability gracefully)
- Verifies DSL template from default_params.json compiles correctly
- Verifies compiled module produces expected output shape

(Full GUI integration tests will be added in Phase B.)
