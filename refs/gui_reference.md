# GUI Reference

Quick lookup for widget names, signals, and cross-tab wiring.  
Module → class map: **[module_map.md](module_map.md)**  
GUI architecture overview: **[CLAUDE.md § GUI Structure](../CLAUDE.md#gui-structure)**

---

## Main Window (`sensoryforge/gui/main.py` → `SensoryForgeWindow`)

### Tab instances
```python
self.mechanoreceptor_tab   # MechanoreceptorTab
self.stimulus_tab          # StimulusDesignerTab
self.spiking_tab           # SpikingNeuronTab
self.visualization_tab     # VisualizationTab
self.batch_tab             # BatchTab
```

### Signal wiring (in `__init__`)
| Signal | Source | Receiver |
|--------|--------|----------|
| `grid_changed` | `mechanoreceptor_tab` | `spiking_tab._on_grid_changed` |
| `populations_changed` | `mechanoreceptor_tab` | `spiking_tab._on_populations_changed` |
| `configuration_directory_changed` | `mechanoreceptor_tab` | `spiking_tab._on_config_dir_changed` |
| `simulation_finished` | `spiking_tab` | `visualization_tab` (receives results) |

### Key methods
- `_load_config()` — loads YAML → calls `set_config()` on mechanoreceptor → stimulus → spiking tabs
- `_save_config()` — collects config from tabs → `_gui_config_to_canonical()` → writes YAML
- `_gui_config_to_canonical(config) → SensoryForgeConfig`
- `_canonical_to_gui_config(config) → dict`
- `_new_project()`, `_open_project()`, `_push_experiment_manager()`

---

## MechanoreceptorTab (`tabs/mechanoreceptor_tab.py`)

### Signals emitted
```python
grid_changed = pyqtSignal(object)                   # emits GridManager or composite dict
configuration_directory_changed = pyqtSignal(object) # emits Path | None
populations_changed = pyqtSignal(object)             # emits List[NeuronPopulation]
```

### Internal data
```python
self.grids: List[GridEntry]          # receptor grid layers
self.populations: List[NeuronPopulation]
self.grid_manager: GridManager       # built by _generate_grids()
```

### Key widgets
| Attribute | Type | Role |
|-----------|------|------|
| `chk_expert_mode` | `QCheckBox` | Basic/Expert toggle (QSettings: `"gui/mechanoreceptor_tab/expert_mode"`) |
| `_expert_only_widgets` | `List[QWidget]` | Widgets hidden in Basic mode |
| `_adv_seeds_group` | `CollapsibleGroupBox` | Advanced seeds section (expert only) |
| `_pos_group` | `CollapsibleGroupBox` | Position/offset section (expert only) |
| `_weights_group` | `CollapsibleGroupBox` | Weight min/max section (expert only) |
| `_adv_neurons_group` | `CollapsibleGroupBox` | Far connections / jitter (expert only) |
| `_csv_group` | `CollapsibleGroupBox` | Custom CSV import/export (expert only) |
| `chk_square_grid` | `QCheckBox` | Force rows == cols for receptor grid |
| `chk_square_neurons` | `QCheckBox` | Force neuron_rows == neuron_cols |
| `spin_neurons_per_col` | `QDoubleSpinBox` | Neuron cols (visible only when `chk_square_neurons` unchecked) |

### Key methods
| Method | Notes |
|--------|-------|
| `set_config(config: dict)` | Loads `{"grids": [...], "populations": [...]}` |
| `get_config() → dict` | Returns same structure |
| `_on_expert_mode_toggled(checked)` | Shows/hides `_expert_only_widgets`; saves to QSettings |
| `_on_square_neurons_toggled(checked)` | Shows/hides `spin_neurons_per_col` |
| `_generate_grids()` | Builds `GridManager`, emits `grid_changed` |
| `_generate_populations()` | Instantiates non-CSV populations; preserves CSV stubs |
| `_on_population_selected(row: int)` | Loads population into form; calls `_load_population_into_form()` |
| `_load_population_into_form(pop)` | Populates all spinboxes from `NeuronPopulation` |
| `_on_population_editor_changed()` | Writes form values back to `populations[selected]` |
| `_on_add_population()` | Adds new `NeuronPopulation`, emits `populations_changed` |

### `GridEntry` dataclass
```python
name: str
rows: int; cols: int
spacing: float  # mm
arrangement: str  # "grid" | "hex" | "poisson"
center_x: float; center_y: float  # mm
color: QColor
# plus offset, seed, grid_manager reference after generation
```

### `NeuronPopulation` dataclass
```python
name: str
neuron_type: str      # "SA" | "RA" | "SA2"
neuron_rows: int; neuron_cols: int
target_grid: str      # GridEntry.name
innervation_method: str  # "gaussian" | "point"
sigma_d_mm: float
connections_per_neuron: int
csv_folder: Optional[str]  # set → prevents regeneration
flat_module: Optional[FlatInnervationModule | _CSVPopulationModule]
```

### `_CSVPopulationModule` dataclass (defined in mechanoreceptor_tab.py)
Duck-types `FlatInnervationModule` so CSV populations work with `_update_innervation_graphics_flat`:
```python
neuron_centers: torch.Tensor      # [N, 2]
innervation_weights: torch.Tensor # [N, M]
receptor_coords: torch.Tensor     # [M, 2]
num_neurons: int
```

---

## SpikingNeuronTab (`tabs/spiking_tab.py`)

### Signals emitted
```python
simulation_finished = pyqtSignal(object, object, object, object, object, object)
# args: (populations, spikes_list, drive_list, filtered_list, voltages_list, times)
```

### Constructor dependencies
```python
SpikingNeuronTab(mechanoreceptor_tab: MechanoreceptorTab,
                 stimulus_tab: StimulusDesignerTab,
                 parent=None)
```

### Key widgets
| Attribute | Type | Role |
|-----------|------|------|
| `chk_expert_mode` | `QCheckBox` | Basic/Expert toggle (QSettings: `"gui/spiking_tab/expert_mode"`) |
| `_expert_only_widgets_spiking` | `List[QWidget]` | Widgets hidden in Basic mode |
| `dbl_input_gain` | `QDoubleSpinBox` | Input gain — default `50.0`; tooltip explains Pierzowski calibration |
| `cmb_model` | `QComboBox` | Neuron model selector |
| `cmb_filter` | `QComboBox` | Filter method (`none` / `SA` / `RA`) |
| `dbl_dt` | `QDoubleSpinBox` | Integration timestep (ms) |
| `cmb_device` | `QComboBox` | `cpu` / `cuda` |
| `cmb_solver` | `QComboBox` | ODE solver |
| `model_params_section` | `CollapsibleGroupBox` | Auto-generated model param spinboxes (expert only) |
| `filter_params_section` | `CollapsibleGroupBox` | Auto-generated filter param spinboxes (expert only) |
| `dsl_section` | `CollapsibleGroupBox` | DSL equation editor (expert only) |

### Key methods
| Method | Notes |
|--------|-------|
| `set_config(config: dict)` | Loads population configs |
| `get_config() → dict` | Returns population configs |
| `_on_expert_mode_toggled(checked)` | Shows/hides `_expert_only_widgets_spiking` |
| `_on_model_changed(model_name)` | Rebuilds model param form |
| `_on_filter_changed(index)` | Rebuilds filter param form |
| `_build_population_section()` | Creates `dbl_input_gain` (setValue 50.0) and population controls |
| `_on_populations_changed(populations)` | Receives `populations_changed` from MechanoreceptorTab |
| `_on_grid_changed(grid_manager)` | Receives `grid_changed` from MechanoreceptorTab |

### `PopulationConfig` (defined at top of spiking_tab.py)
```python
name: str
model: str            # "Izhikevich" | "AdEx" | "MQIF" | "FA" | "SA" | "DSL"
filter_method: str    # "none" | "sa" | "ra"
input_gain: float     # default 50.0
model_params: dict
filter_params: dict
dsl_equations: str
dsl_threshold: str
dsl_reset: str
dsl_params: dict
neuron_type: str      # for display
```

---

## StimulusDesignerTab (`tabs/stimulus_tab.py`)

### Constructor dependencies
```python
StimulusDesignerTab(mechanoreceptor_tab: MechanoreceptorTab, parent=None)
```
Uses `mechanoreceptor_tab.grid_manager` for stimulus generation.

### Key methods
- `set_config(config: dict)` — loads `{"stimuli": [...]}`
- `get_config() → dict`
- `get_selected_stimulus() → Optional[StimulusConfig]`
- `_update_preview_canvas()` — regenerates preview using `StimulusGenerator`

---

## VisualizationTab (`tabs/visualization_tab.py`)

Passive display; no outbound signals.

### Receives
- `simulation_finished` from `SpikingNeuronTab` via main window wiring
- `populations_changed` from `MechanoreceptorTab` via main window wiring

### Key methods
- `set_simulation_results(results)` — called after simulation completes
- `set_populations(populations)` — called when populations change

### DockArea panels
Preset layouts rendered via pyqtgraph `DockArea`. Panel types: `StimulusPanel`, `RasterPanel`, `NeuronHeatmapPanel`, `FiringRatePanel`, `VoltagePanel`, `ReceptorPanel`.

---

## BatchTab (`tabs/batch_tab.py`)

No cross-tab signal dependencies. Reads saved YAML configs from disk.  
Uses `BatchExecutor` for local runs and SLURM script export.

---

## Expert Mode Pattern (both MechanoreceptorTab + SpikingNeuronTab)

```python
# At top of _setup_ui():
self._expert_only_widgets = []                 # or _expert_only_widgets_spiking
self.chk_expert_mode = QCheckBox("Expert mode")
self.chk_expert_mode.setChecked(False)
settings = QSettings()
key = "gui/mechanoreceptor_tab/expert_mode"    # or spiking_tab
self.chk_expert_mode.setChecked(settings.value(key, False, type=bool))
self.chk_expert_mode.toggled.connect(self._on_expert_mode_toggled)

# When building advanced widgets:
self._adv_group = CollapsibleGroupBox("Advanced")
self._expert_only_widgets.append(self._adv_group)

# At end of _setup_ui():
self._on_expert_mode_toggled(self.chk_expert_mode.isChecked())

# Method:
def _on_expert_mode_toggled(self, checked: bool):
    for w in self._expert_only_widgets:
        w.setVisible(checked)
    QSettings().setValue(key, checked)
```

---

## Testing Qt Widgets (headless)

- Use `isHidden()` not `isVisible()` — `isVisible()` requires parent chain shown on screen
- Hold `QApplication` at module level to prevent GC: `_APP = None; _ensure_app()`
- Use `scope="module"` fixtures when multiple tabs must coexist (avoids pyqtgraph GC crash)
- See `tests/unit/test_expert_mode.py` for the canonical pattern
