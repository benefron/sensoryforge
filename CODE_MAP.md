# SensoryForge Code Map

Quick-navigation index. Each section answers "where do I go when I need to do X."  
Full architectural context: **[CLAUDE.md](CLAUDE.md)**

---

## Run the app / tests

```bash
conda activate sensoryforge
python sensoryforge/gui/main.py           # GUI
sensoryforge run examples/example_config.yml --duration 1000  # CLI
pytest tests/unit/ -q                     # unit tests
pytest tests/unit/test_<module>.py -v     # single file
```

---

## I need to...

### Add a neuron model
Guide: [docs/developer_guide/add_neuron.md](docs/developer_guide/add_neuron.md)  
Base class: `sensoryforge/neurons/base.py` → `BaseNeuron`  
Existing models: `neurons/{izhikevich,adex,mqif,fa,sa}.py`  
Register: `sensoryforge/register_components.py` → `register_all()`

### Add a filter
Guide: [docs/developer_guide/add_filter.md](docs/developer_guide/add_filter.md)  
Base class: `sensoryforge/filters/base.py` → `BaseFilter`  
Existing: `filters/sa_ra.py` → `SAFilterTorch`, `RAFilterTorch`

### Add a stimulus type
Guide: [docs/developer_guide/add_stimulus.md](docs/developer_guide/add_stimulus.md)  
Base class + `ParamSpec`: `sensoryforge/stimuli/base.py`  
Existing: `stimuli/{gaussian,texture,moving}.py`

### Understand the data flow / tensor shapes / units
[CLAUDE.md § Architecture](CLAUDE.md#architecture) — canonical pipeline diagram  
[docs/user_guide/units_and_gains.md](docs/user_guide/units_and_gains.md) — SA filter calibration, `input_gain=50` rationale

### Work on the GUI
Tab files, signals, widget names, cross-tab wiring: **[refs/gui_reference.md](refs/gui_reference.md)**  
Collapsible widget: `sensoryforge/gui/widgets/collapsible.py`  
Visualization panels: `sensoryforge/gui/visualization/` (one file per panel type)

### Work on configuration / YAML schema
Canonical dataclasses: `sensoryforge/config/schema.py` → `SensoryForgeConfig`, `GridConfig`, `PopulationConfig`  
Key defaults (`dt`, `input_gain`, filter params): `sensoryforge/config/default_config.yml`  
[docs/user_guide/yaml_configuration.md](docs/user_guide/yaml_configuration.md)

### Find or write tests
Full test inventory: **[refs/test_map.md](refs/test_map.md)**  
Unit tests live in `tests/unit/`, integration in `tests/integration/`

### Find a module, class, or function
Complete module → class/function map: **[refs/module_map.md](refs/module_map.md)**

### Understand the simulation engine vs legacy pipeline
Two engines — canonical (`SimulationEngine`) vs legacy (`GeneralizedTactileEncodingPipeline`):  
[CLAUDE.md § Execution Engines](CLAUDE.md#execution-engines)  
GUI backend entry point: `SimulationEngine._run_pop_from_drive()` (static)

### Use the registry system
`sensoryforge/registry.py` → `ComponentRegistry`  
`sensoryforge/register_components.py` → `register_all()` — all built-in keys listed here

### Run a batch / SLURM sweep
`sensoryforge/core/batch_executor.py` → `BatchExecutor`  
Example config: `examples/batch_config.yml`  
[docs/user_guide/batch_processing.md](docs/user_guide/batch_processing.md)

---

## Key file locations (one-liners)

| What | Where |
|------|-------|
| Pipeline data flow | `sensoryforge/core/simulation_engine.py` |
| Grid + receptor coords | `sensoryforge/core/grid.py`, `composite_grid.py` |
| Innervation weights | `sensoryforge/core/innervation.py` |
| CLI entry point | `sensoryforge/cli.py` |
| GUI main window | `sensoryforge/gui/main.py` |
| All tab files | `sensoryforge/gui/tabs/` |
| Config schema | `sensoryforge/config/schema.py` |
| Known tech debt | [CLAUDE.md § Known Technical Debt](CLAUDE.md#known-technical-debt) |
| Dev report / audit | `devo_reports/development_9_april_2026`, `reviews/CODE_REVIEW_20260408.md` |
