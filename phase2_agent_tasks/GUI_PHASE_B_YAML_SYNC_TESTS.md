# Phase B: YAML ↔ GUI Bidirectional Sync + Integration Tests

**Priority:** Phase B (sequential — after all Phase A agents merge)  
**Files to modify:** ALL tab files + `main.py` + new test file  
**Estimated complexity:** Large  
**Dependencies:** Requires Agents A, B, C, D to be completed and merged first

---

## Pre-requisites

Read these documents first:
1. `phase2_agent_tasks/SHARED_CONTEXT.md` — coding standards, project overview
2. `.github/copilot-instructions.md` — full coding conventions, docstring style, commit format
3. `reviews/GUI_PHASE2_INTEGRATION_PLAN.md` — overall plan (your section is "Phase B")

**CRITICAL:** This task must be done AFTER all Phase A agents have merged.
You will be working with the updated files from Agents A, B, C, and D.

---

## Task B.1: Tab Config API

### Goal
Add `get_config() → dict` and `set_config(config: dict) → None` methods to each 
tab, enabling programmatic serialization and deserialization of GUI state.

### MechanoreceptorTab (`mechanoreceptor_tab.py`)

```python
def get_config(self) -> dict:
    """Extract current tab state as a config dictionary.
    
    Returns:
        dict with keys:
            - grid_type: "standard" | "composite"
            - grid_size: int (standard only)
            - spacing: float (standard only)
            - center: [float, float] (standard only)
            - xlim: [float, float] (composite only)
            - ylim: [float, float] (composite only)
            - populations: list of dicts with name, count/density, arrangement, etc.
    """

def set_config(self, config: dict) -> None:
    """Populate tab controls from a config dictionary.
    
    Triggers grid rebuild and signal emissions.
    """
```

### StimulusDesignerTab (`stimulus_tab.py`)

```python
def get_config(self) -> dict:
    """Extract current stimulus config.
    
    Returns:
        dict with keys:
            - type: "gaussian" | "point" | "edge" | "texture" | "moving"
            - subtype: texture/motion sub-type (if applicable)
            - params: dict of stimulus-specific parameters
            - temporal: dict with ramp_up, plateau, ramp_down
            - motion: dict with static/moving settings
    """

def set_config(self, config: dict) -> None:
    """Set stimulus from config dict. Triggers preview update."""
```

### SpikingNeuronTab (`spiking_tab.py`)

```python
def get_config(self) -> dict:
    """Extract neuron/simulation config.
    
    Returns:
        dict with keys:
            - populations: list of PopulationConfig dicts
            - solver: "euler" | "adaptive"
            - solver_config: dict (method, rtol, atol for adaptive)
            - simulation: dict (dt, duration, device)
            - dsl_model: dict (equations, threshold, reset, parameters) if DSL
    """

def set_config(self, config: dict) -> None:
    """Set tab state from config. Triggers UI updates."""
```

---

## Task B.2: YAML Load (main.py `_load_config()`)

### Current Behavior
Shows an informational dialog — does NOT populate GUI controls.

### New Behavior
1. Parse YAML file
2. Map YAML sections to tab config dicts
3. Call each tab's `set_config()` with the appropriate section
4. Show summary of what was loaded

### YAML Schema Mapping

```yaml
# YAML section       → Tab.set_config() key mapping
pipeline:             → MechanoreceptorTab (grid_size, spacing, center, device)
grid:                 → MechanoreceptorTab (grid_type, composite populations)
neurons:              → SpikingNeuronTab (populations, model type, params)
  type: dsl           → SpikingNeuronTab (dsl_model section)
stimuli:              → StimulusDesignerTab (first stimulus entry)
solver:               → SpikingNeuronTab (solver, solver_config)
```

### Implementation Pattern

```python
def _load_config(self) -> None:
    filename, _ = QFileDialog.getOpenFileName(...)
    if not filename:
        return
    
    import yaml
    with open(filename) as f:
        config = yaml.safe_load(f)
    
    # Map to tab configs
    if "pipeline" in config or "grid" in config:
        grid_config = self._map_yaml_to_grid_config(config)
        self.mechanoreceptor_tab.set_config(grid_config)
    
    if "stimuli" in config:
        stim_config = self._map_yaml_to_stimulus_config(config)
        self.stimulus_tab.set_config(stim_config)
    
    if "neurons" in config or "solver" in config:
        neuron_config = self._map_yaml_to_neuron_config(config)
        self.spiking_tab.set_config(neuron_config)
    
    QMessageBox.information(self, "Loaded", f"Configuration loaded from {filename}")
```

---

## Task B.3: YAML Save (main.py `_save_config()`)

### Current Behavior
Writes a static template — does NOT capture GUI state.

### New Behavior
1. Call each tab's `get_config()`
2. Merge into unified YAML structure
3. Write to file
4. Ensure output is compatible with `GeneralizedTactileEncodingPipeline.from_config()`

### Implementation Pattern

```python
def _save_config(self) -> None:
    filename, _ = QFileDialog.getSaveFileName(...)
    if not filename:
        return
    
    config = {
        "metadata": {
            "name": "SensoryForge Configuration",
            "version": "0.2.0",
            "created": datetime.now().isoformat(),
        },
    }
    
    # Gather from tabs
    grid_config = self.mechanoreceptor_tab.get_config()
    stim_config = self.stimulus_tab.get_config()
    neuron_config = self.spiking_tab.get_config()
    
    # Map to YAML schema
    config.update(self._map_grid_config_to_yaml(grid_config))
    config.update(self._map_stimulus_config_to_yaml(stim_config))
    config.update(self._map_neuron_config_to_yaml(neuron_config))
    
    import yaml
    with open(filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
```

---

## Task B.4: Round-Trip Fidelity

Ensure: `save → load → save` produces identical YAML content.

Test this explicitly:
1. Set up GUI state programmatically via `set_config()`
2. Call `get_config()` → save YAML
3. Create fresh tabs → call `set_config()` with saved config
4. Call `get_config()` again → compare dicts

---

## Task B.5: Integration Tests

Create `tests/integration/test_gui_phase2.py`:

```python
"""Integration tests for GUI Phase 2 features.

These tests verify that Phase 2 components (CompositeGrid, DSL, extended
stimuli, solvers) are properly integrated into the GUI tabs.

Note: Tests create widget instances programmatically without showing them.
PyQt5 requires a QApplication instance — use the qapp fixture.
"""
import pytest

@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for test session."""
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app

class TestMechanoreceptorTabPhase2:
    """Test CompositeGrid integration in MechanoreceptorTab."""
    
    def test_grid_type_selector_exists(self, qapp):
        """Tab should have a grid type combobox."""
        ...
    
    def test_composite_grid_creation(self, qapp):
        """Switching to composite should create CompositeGrid."""
        ...
    
    def test_get_set_config_round_trip(self, qapp):
        """get_config → set_config should preserve state."""
        ...

class TestStimulusTabPhase2:
    """Test extended stimuli in StimulusDesignerTab."""
    
    def test_texture_type_available(self, qapp):
        """Texture stimulus type should be selectable."""
        ...
    
    def test_moving_type_available(self, qapp):
        """Moving stimulus type should be selectable."""
        ...

class TestSpikingTabPhase2:
    """Test DSL and solver in SpikingNeuronTab."""
    
    def test_dsl_model_in_dropdown(self, qapp):
        """DSL option should appear in model combobox."""
        ...
    
    def test_solver_dropdown_exists(self, qapp):
        """Solver selection combobox should exist."""
        ...

class TestYamlRoundTrip:
    """Test YAML save/load round-trip fidelity."""
    
    def test_standard_config_round_trip(self, qapp, tmp_path):
        """Save → load → save should produce identical config."""
        ...
    
    def test_composite_config_round_trip(self, qapp, tmp_path):
        """Composite grid config survives round-trip."""
        ...
    
    def test_dsl_config_round_trip(self, qapp, tmp_path):
        """DSL model config survives round-trip."""
        ...

class TestProtocolSuiteWired:
    """Test ProtocolSuiteTab is connected."""
    
    def test_protocol_tab_exists_in_main(self, qapp):
        """Main window should have 4 tabs including Protocol Suite."""
        ...
```

---

## Key Constraints

- **Each tab's `get_config()`/`set_config()` must be fully self-contained** — 
  no cross-tab dependencies in serialization.
- **YAML output must be compatible with `GeneralizedTactileEncodingPipeline.from_config()`**
  where applicable (pipeline, neurons, stimuli sections).
- **Existing save/load (JSON bundles)** should continue working alongside YAML.
- **All 307+ existing tests must still pass.**

---

## Verification

- Load YAML → all tab controls populated correctly
- Save YAML → file contains current GUI state
- Round-trip: save → load → save produces identical YAML
- Load → modify → save captures changes
- Invalid YAML shows clear error message
- All existing tests pass + new integration tests pass

---

## Commit Format

```
feat(gui): add YAML ↔ GUI bidirectional config sync and integration tests
```
