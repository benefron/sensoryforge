# Agent C: DSL Neuron Model + Solver Selection in SpikingNeuronTab

**Priority:** Phase A (parallel)  
**Files to modify:** `sensoryforge/gui/tabs/spiking_tab.py`  
**Estimated complexity:** Medium  
**Dependencies:** None — fully independent of Agents A, B, D

---

## Pre-requisites

Read these documents first:
1. `phase2_agent_tasks/SHARED_CONTEXT.md` — coding standards, project overview
2. `.github/copilot-instructions.md` — full coding conventions, docstring style, commit format
3. `reviews/GUI_PHASE2_INTEGRATION_PLAN.md` — overall plan (your section is "Agent C")

---

## Context

The `SpikingNeuronTab` currently supports 5 hard-coded neuron models (Izhikevich, 
AdEx, MQIF, FA, SA) and uses implicit forward Euler integration. Phase 2 added:

- **Equation DSL** (`sensoryforge.neurons.model_dsl`) — define neuron models via equations
- **Solver infrastructure** (`sensoryforge.solvers`) — pluggable Euler and adaptive (Dormand-Prince) solvers

These are fully implemented but **not exposed in the GUI**.

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
module = model.compile(solver="euler", dt=0.5, num_neurons=100, device="cpu")

# Use: forward(input_current) → (spikes, state_dict)
spikes, state = module(current_tensor)
# state contains: {"voltage": Tensor, "spikes": Tensor, ...}
```

#### NeuronModel.compile() signature:
```python
def compile(
    self,
    solver: str | BaseSolver = "euler",
    dt: float = 0.5,
    num_neurons: int = 1,
    device: str | torch.device = "cpu",
) -> nn.Module:
```

#### NeuronModel.from_config() for YAML loading:
```python
config = {
    "equations": "dv/dt = ...\ndu/dt = ...",
    "threshold": "v >= 30 * mV",
    "reset": "v = c\nu = u + d",
    "parameters": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
}
model = NeuronModel.from_config(config)
```

### Solvers (`sensoryforge.solvers`)

```python
from sensoryforge.solvers.euler import EulerSolver
from sensoryforge.solvers.adaptive import AdaptiveSolver

# Euler (default)
solver = EulerSolver(dt=0.5)

# Adaptive (requires torchdiffeq)
solver = AdaptiveSolver(method="dopri5", dt=0.5, rtol=1e-5, atol=1e-7)

# From config:
solver = EulerSolver.from_config({"dt": 0.5})
solver = AdaptiveSolver.from_config({"method": "dopri5", "rtol": 1e-5, "atol": 1e-7})
```

---

## default_params.json Phase 2 Section (already exists, unused)

```json
{
  "phase2": {
    "neuron_types": ["izhikevich", "adex", "mqif", "dsl"],
    "solver_types": ["euler", "adaptive"],
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
  },
  "gui": {
    "default_neuron_type": "izhikevich",
    "default_solver": "euler"
  }
}
```

---

## Current SpikingNeuronTab Architecture

```
SpikingNeuronTab(QWidget)
├── Constructor: __init__(self, mechanoreceptor_tab, stimulus_tab)
├── Receives: grid data, population list, stimulus library
├── Per-population config panel:
│   ├── cmb_model: QComboBox ["Izhikevich", "AdEx", "MQIF", "FA", "SA"]
│   ├── cmb_filter: QComboBox ["SA", "RA", "None"]
│   ├── Parameter table: model-specific params (from default_params.json)
│   ├── dbl_input_gain, dbl_noise_std
│   └── chk_population_enabled
├── Simulation controls:
│   ├── Stimulus selector (from library)
│   ├── Duration, dt, device
│   └── Run button
├── Results display:
│   ├── Voltage trace (matplotlib)
│   ├── Spike raster (matplotlib)
│   └── Neuron selector / neuron map
├── Save/Load module bundles (JSON)
└── Key methods:
    ├── _create_neuron_model(config) → nn.Module
    ├── _run_simulation() → SimulationResult
    ├── _on_model_changed() → swaps param table
    ├── _update_population_config() → stores PopulationConfig
    └── _save_neuron_module() / _load_neuron_module()
```

### Key existing code patterns:

The model combobox triggers `_on_model_changed()` which swaps the parameter 
controls. The `_create_neuron_model()` method instantiates the appropriate 
`nn.Module` based on the model name string:

```python
# Current pattern (simplified):
if model_name == "Izhikevich":
    from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
    return IzhikevichNeuronTorch(config)
elif model_name == "AdEx":
    ...
```

---

## What To Do

### C.1: Equation DSL Integration

1. **Add "DSL (Custom Equations)" to model combobox**:
   - Append to the existing model list: `["Izhikevich", "AdEx", "MQIF", "FA", "SA", "DSL (Custom)"]`

2. **Create DSL editor panel** (shown when "DSL (Custom)" is selected):
   - **Equations**: `QPlainTextEdit`, 5-6 lines tall, monospaced font
     - Placeholder: `"dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms\ndu/dt = (a * (b*v - u)) / ms"`
   - **Threshold**: `QLineEdit` 
     - Placeholder: `"v >= 30 * mV"`
   - **Reset**: `QLineEdit`
     - Placeholder: `"v = c\nu = u + d"`
   - **Parameters**: `QTableWidget` with 2 columns (Name, Value)
     - Pre-populate with Izhikevich defaults: a=0.02, b=0.2, c=-65.0, d=8.0
     - "Add Parameter" / "Remove Parameter" buttons
   - **Compile button**: `QPushButton("Compile Model")`
   - **Status label**: shows "✓ Compiled successfully" or "✗ Error: ..."
   - Load defaults from `default_params.json > phase2 > dsl_neuron > template`

3. **Hide standard param table when DSL selected**:
   - In `_on_model_changed()`: if model is "DSL (Custom)", hide standard parameter 
     controls and show DSL editor panel. Otherwise, show standard controls and 
     hide DSL panel.

4. **Implement DSL compilation** in a `_compile_dsl_model()` method:
   ```python
   def _compile_dsl_model(self) -> None:
       try:
           from sensoryforge.neurons.model_dsl import NeuronModel
           
           equations = self.dsl_equations_edit.toPlainText()
           threshold = self.dsl_threshold_edit.text()
           reset = self.dsl_reset_edit.text()
           parameters = self._get_dsl_parameters()  # dict from table
           
           model = NeuronModel(
               equations=equations,
               threshold=threshold,
               reset=reset,
               parameters=parameters,
           )
           
           self._compiled_dsl_model = model
           self.dsl_status_label.setText("✓ Compiled successfully")
           self.dsl_status_label.setStyleSheet("color: green;")
       except Exception as e:
           self._compiled_dsl_model = None
           self.dsl_status_label.setText(f"✗ Error: {str(e)}")
           self.dsl_status_label.setStyleSheet("color: red;")
   ```

5. **Update `_create_neuron_model()`** to handle DSL:
   ```python
   if model_name == "DSL (Custom)":
       if self._compiled_dsl_model is None:
           raise ValueError("DSL model not compiled yet")
       return self._compiled_dsl_model.compile(
           solver=self._get_selected_solver(),
           dt=self.dt,
           num_neurons=num_neurons,
           device=device,
       )
   ```

### C.2: Solver Selection

1. **Add solver section** in the simulation controls area (near dt/device):
   - `QComboBox` labeled "Solver": ["Euler (default)", "Adaptive (RK45)"]
   - Default: "Euler" (from `default_params.json > gui > default_solver`)

2. **Create adaptive solver config panel** (visible when Adaptive selected):
   - Method: `QComboBox` ["dopri5", "bosh3", "adaptive_heun"]
   - rtol: `QDoubleSpinBox` (default 1e-5, range 1e-10 to 1e-1, scientific notation)
   - atol: `QDoubleSpinBox` (default 1e-7, range 1e-12 to 1e-1, scientific notation)
   - Load defaults from `default_params.json > phase2 > solvers > adaptive`

3. **Implement `_get_selected_solver()`**:
   ```python
   def _get_selected_solver(self) -> str:
       """Return solver name string for neuron model creation."""
       if self.cmb_solver.currentText().startswith("Euler"):
           return "euler"
       return "adaptive"
   ```

4. **Thread solver into simulation**:
   - For hand-written models (Izhikevich, AdEx, etc.): these currently use their 
     built-in forward Euler. The solver selection should be noted in the UI but 
     may not change their behavior (they don't accept a solver parameter).
     Show a tooltip: "Solver selection applies to DSL models. Built-in models use 
     their native integration method."
   - For DSL models: pass solver to `compile()` as shown in C.1 step 5.

### C.3: Update Save/Load

- Extend the neuron module save/load to persist:
  - DSL equations, threshold, reset, parameters (when DSL model is selected)
  - Solver selection and adaptive solver params

---

## Key Constraints

- **DO NOT change existing neuron model behavior** — Izhikevich/AdEx/MQIF/FA/SA 
  must work exactly as before.
- **DSL compilation errors must be handled gracefully** — show error message, 
  don't crash the GUI.
- **AdaptiveSolver requires `torchdiffeq`** — if not installed, show a clear 
  message: "Install torchdiffeq: pip install torchdiffeq"
- **Model combobox items for existing models stay unchanged** — DSL is appended.
- **Read defaults from `default_params.json`** `phase2` section.

---

## Verification

- Launch GUI → Spiking tab model dropdown shows 6 options (5 existing + DSL)
- Select "DSL (Custom)" → standard params hide, equations editor shows
- Default equations are Izhikevich template from default_params.json
- Click "Compile" → status shows "✓ Compiled successfully"
- Enter invalid equations → status shows error message
- Run simulation with DSL model → voltage trace and spike raster display
- Solver dropdown shows Euler (default) and Adaptive
- Select Adaptive → method/rtol/atol controls appear
- Select Euler → adaptive controls hide
- All 5 existing models still work unchanged
- Save/load preserves DSL configuration

---

## Commit Format

```
feat(gui): add DSL neuron model editor and solver selection to SpikingNeuronTab
```

## Tests

Add a minimal test in `tests/unit/test_gui_agent_c.py` that:
- Imports `NeuronModel` and compiles with default Izhikevich equations
- Imports `EulerSolver` and `AdaptiveSolver` without error
- Verifies compiled DSL module has `forward()` method
- Verifies DSL module produces output with expected shape

(Full GUI integration tests will be added in Phase B.)
