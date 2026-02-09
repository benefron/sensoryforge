# Agent A: Wire ProtocolSuiteTab + CompositeGrid in MechanoreceptorTab

**Priority:** Phase A (parallel)  
**Files to modify:** `sensoryforge/gui/main.py`, `sensoryforge/gui/tabs/__init__.py`, `sensoryforge/gui/tabs/mechanoreceptor_tab.py`  
**Estimated complexity:** Large  
**Dependencies:** None — fully independent of Agents B, C, D

---

## Pre-requisites

Read these documents first:
1. `phase2_agent_tasks/SHARED_CONTEXT.md` — coding standards, project overview
2. `.github/copilot-instructions.md` — full coding conventions, docstring style, commit format
3. `reviews/GUI_PHASE2_INTEGRATION_PLAN.md` — overall plan (your section is "Agent A")

---

## Task A.1: Wire ProtocolSuiteTab into Main Window (small)

### Context

`ProtocolSuiteTab` and `ProtocolExecutionController` exist as fully implemented
classes but are **not connected to the main window**. The tab is not imported in
`tabs/__init__.py` and not instantiated in `main.py`.

### Current State

**`sensoryforge/gui/tabs/__init__.py`** exports only 3 tabs:
```python
from sensoryforge.gui.tabs.mechanoreceptor_tab import MechanoreceptorTab
from sensoryforge.gui.tabs.stimulus_tab import StimulusDesignerTab
from sensoryforge.gui.tabs.spiking_tab import SpikingNeuronTab
```

**`sensoryforge/gui/main.py`** creates 3 tabs in `SensoryForgeWindow.__init__()`.

### What To Do

1. **`sensoryforge/gui/tabs/__init__.py`**: Add import and export for `ProtocolSuiteTab`:
   ```python
   from sensoryforge.gui.tabs.protocol_suite_tab import ProtocolSuiteTab
   ```

2. **`sensoryforge/gui/main.py`**: 
   - Import `ProtocolSuiteTab` (already available via `tabs/__init__`)
   - Import `ProtocolExecutionController` from `sensoryforge.gui.protocol_execution_controller`
   - Import `ProjectRegistry` from `sensoryforge.utils.project_registry`
   - After creating the 3 existing tabs, add:
     ```python
     self.protocol_tab = ProtocolSuiteTab(
         self.mechanoreceptor_tab,
         self.stimulus_tab,
         self.spiking_tab,
     )
     tabs.addTab(self.protocol_tab, "Protocol Suite")
     ```
   - Create the execution controller:
     ```python
     self._registry = ProjectRegistry()
     self._protocol_controller = ProtocolExecutionController(
         mechanoreceptor_tab=self.mechanoreceptor_tab,
         spiking_tab=self.spiking_tab,
         protocol_tab=self.protocol_tab,
         registry=self._registry,
     )
     ```
   - Connect signals:
     ```python
     self.protocol_tab.run_requested.connect(
         self._protocol_controller.execute_batch
     )
     self._protocol_controller.run_completed.connect(
         self.protocol_tab.on_run_completed
     )
     self._protocol_controller.run_failed.connect(
         self.protocol_tab.on_run_failed
     )
     self._protocol_controller.batch_finished.connect(
         self.protocol_tab.on_batch_finished
     )
     ```

### Verification

- The app should launch with 4 tabs instead of 3.
- The Protocol Suite tab should display its library and queue UI.
- No errors on startup.

---

## Task A.2: CompositeGrid Support in MechanoreceptorTab (large)

### Context

The `MechanoreceptorTab` currently uses only `GridManager` (standard grid). 
Phase 2 added `CompositeGrid` in `sensoryforge/core/composite_grid.py` which 
supports multi-population spatial substrates with different arrangements 
(grid, poisson, hex, jittered_grid).

### CompositeGrid API (read-only reference)

```python
from sensoryforge.core.composite_grid import CompositeGrid

# Create grid with spatial bounds
grid = CompositeGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0), device="cpu")

# Add populations
grid.add_population(name="SA1", density=100.0, arrangement="grid")
grid.add_population(name="RA1", density=50.0, arrangement="hex")

# Query
coords = grid.get_population_coordinates("SA1")  # → Tensor [N, 2]
config = grid.get_population_config("SA1")  # → dict
count = grid.get_population_count("SA1")  # → int
names = grid.list_populations()  # → ["SA1", "RA1"]

# Arrangement types: "grid", "poisson", "hex", "jittered_grid"
```

### default_params.json Phase 2 Section (already exists, unused)

```json
{
  "phase2": {
    "grid_types": ["standard", "composite"],
    "composite_grid": {
      "populations": {
        "sa1": {"density": 100.0, "arrangement": "grid", "filter": "SA"},
        "ra1": {"density": 70.0, "arrangement": "hex", "filter": "RA"},
        "sa2": {"density": 30.0, "arrangement": "poisson", "filter": "SA"}
      }
    }
  },
  "gui": {
    "default_grid_type": "standard"
  }
}
```

### Current MechanoreceptorTab Architecture

```
MechanoreceptorTab(QWidget)
├── Signals: grid_changed, configuration_directory_changed, populations_changed
├── Grid controls: grid_size (int spin), spacing (double spin), center_x/y
├── Population list: QListWidget with NeuronPopulation items
├── Population editor: count, RF radius, min/max weight, seed
├── Visualization: matplotlib canvas showing grid + innervation heatmap
├── Save/Load: JSON bundle with .pt tensor files
└── Key methods:
    ├── _rebuild_grid() → creates GridManager, emits grid_changed
    ├── _add_population() → adds NeuronPopulation to list
    ├── _rebuild_innervation() → creates innervation maps
    └── _save_configuration() / _load_configuration()
```

### What To Do

1. **Add grid type selector** at the top of the grid configuration section:
   - `QComboBox` with items: "Standard Grid", "Composite Grid"
   - Default: "Standard Grid" (from `default_params.json > gui > default_grid_type`)
   - Connect `currentIndexChanged` to `_on_grid_type_changed()`

2. **Create composite grid config panel** (hidden by default):
   - `xlim_min`, `xlim_max` double spinboxes (default: -5.0, 5.0)
   - `ylim_min`, `ylim_max` double spinboxes (default: -5.0, 5.0)
   - Population table with columns: Name, Density, Arrangement, Filter
     - Arrangement: QComboBox per row (grid/poisson/hex/jittered_grid)
     - Filter: QComboBox per row (SA/RA/None)
   - "Add Population" / "Remove Population" buttons
   - Read defaults from `default_params.json > phase2 > composite_grid > populations`

3. **Toggle visibility** in `_on_grid_type_changed()`:
   - Standard: show existing grid_size/spacing/center controls, hide composite panel
   - Composite: hide standard controls, show composite panel

4. **Update `_rebuild_grid()`**:
   - If grid type is "Standard": existing behavior (use `GridManager`)
   - If grid type is "Composite":
     ```python
     from sensoryforge.core.composite_grid import CompositeGrid
     cg = CompositeGrid(xlim=(xmin, xmax), ylim=(ymin, ymax), device=device)
     for pop in composite_populations:
         cg.add_population(name=pop.name, density=pop.density,
                          arrangement=pop.arrangement)
     ```
   - Store `self._composite_grid = cg` (or `None` if standard)
   - Emit `grid_changed` with the composite grid data

5. **Update visualization** for composite mode:
   - Color-code scatter plot by population
   - Show legend with population names and counts
   - Use `grid.get_population_coordinates(name)` for each population

6. **Update save/load** to persist grid type and composite parameters.

7. **Update `populations_changed` signal** to include composite grid population info
   when in composite mode.

### Key Constraints

- **DO NOT change the Standard Grid behavior** — it must work exactly as before.
- **DO NOT modify the signal interface** — `grid_changed`, `populations_changed`,
  `configuration_directory_changed` signatures stay the same. The `object` payload
  can include additional keys.
- **Read `default_params.json`** for initial defaults — the `phase2` section has
  composite grid config.

### Verification

- Launch GUI → MechanoreceptorTab shows grid type dropdown defaulting to "Standard"
- Switch to "Composite Grid" → standard controls hide, composite panel shows
- Add populations → visualization shows color-coded scatter plot
- Switch back to "Standard" → original controls restored
- Save/load preserves grid type and composite settings
- `grid_changed` signal fires with correct data in both modes

---

## Commit Format

```
feat(gui): wire ProtocolSuiteTab and add CompositeGrid support to MechanoreceptorTab
```

## Tests

Add a minimal test in `tests/unit/test_gui_agent_a.py` that:
- Imports `MechanoreceptorTab`, `ProtocolSuiteTab` without error
- Verifies `ProtocolSuiteTab` is in `sensoryforge.gui.tabs.__all__` (or importable)
- Verifies `CompositeGrid` can be imported from expected location

(Full GUI integration tests will be added in Phase B.)
