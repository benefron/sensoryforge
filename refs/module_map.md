# Module Map

All Python modules → primary classes and key public functions.  
For data flow and pipeline diagrams: **[CLAUDE.md § Architecture](../CLAUDE.md#architecture)**

---

## Neurons (`sensoryforge/neurons/`)

| File | Class | Key methods |
|------|-------|-------------|
| `base.py` | `BaseNeuron(nn.Module)` | `forward(input_current) → (v_trace, spikes)`, `reset_state()`, `from_config()`, `to_dict()` |
| `izhikevich.py` | `IzhikevichNeuronTorch` | `forward(input_current, a, b, c, d, threshold, solver)`, `reset_state()` |
| `adex.py` | `AdExNeuronTorch` | `forward(input_current, solver)`, `reset_state()`, `_dynamics(v, w, input_t)` |
| `mqif.py` | `MQIFNeuronTorch` | `forward(input_current, solver)`, `reset_state()`, `_dynamics(v, u, input_t)` |
| `sa.py` | `SANeuronTorch` | `forward(input_current, ..., reset_states=True)`, `reset_state()` |
| `fa.py` | `FANeuronTorch` | `forward(input_current, ..., reset_states=True)`, `reset_state()` |
| `model_dsl.py` | `NeuronModel` | `compile(solver, dt, device, noise_std)`, `to_dict()`, `from_config()` |
| `model_dsl.py` | `_CompiledNeuronModule` | `forward(input_current) → (v_trace, spikes)` — returned by `NeuronModel.compile()` |

Input shape: `[batch, steps, neurons]` → output `(v_trace, spikes)` both `[batch, steps+1, neurons]`

---

## Filters (`sensoryforge/filters/`)

| File | Class | Key methods |
|------|-------|-------------|
| `base.py` | `BaseFilter(nn.Module)` | `forward(x, dt=None)`, `reset_state()`, `from_config()`, `to_dict()` |
| `sa_ra.py` | `SAFilterTorch` | `forward(I_in, dI_in_dt, reset_states)`, `reset_state()`, `reset_states(batch, neurons, device)`, `forward_steady_state()`, `forward_multi_step()` |
| `sa_ra.py` | `RAFilterTorch` | `forward(I_in, dI_in_dt, reset_states)`, `reset_state()`, `forward_edge_response()`, `forward_steady_state()` |
| `sa_ra.py` | `CombinedSARAFilter` | `forward(sa_inputs, ra_inputs, reset_states)`, `forward_enhanced(method)`, `reset_all_states()` |
| `noise.py` | `MembraneNoiseTorch` | `forward(current) → noisy_current` |
| `noise.py` | `ReceptorNoiseTorch` | `forward(responses) → noisy_responses` |

Note: `SAFilterTorch` uses `reset_states()` (plural) — breaks `BaseFilter` polymorphism. New filters must use `reset_state()` (singular).

---

## Stimuli (`sensoryforge/stimuli/`)

| File | Class / Function | Notes |
|------|-----------------|-------|
| `base.py` | `BaseStimulus(nn.Module)` | `forward(xx, yy, **kwargs)`, `get_param_spec() → List[ParamSpec]` |
| `base.py` | `ParamSpec` | `name, label, dtype, default, min_val, max_val, step, unit, tooltip` |
| `gaussian.py` | `GaussianStimulus` | `forward(xx, yy)`, `get_param_spec()` |
| `texture.py` | `GaborTexture`, `EdgeGrating` | `forward(xx, yy)`, `get_param_spec()` |
| `texture.py` | functions | `gabor_texture()`, `edge_grating()`, `noise_texture()` |
| `moving.py` | `MovingStimulus` | `forward(xx, yy) → [num_steps, H, W]` |
| `moving.py` | functions | `linear_motion()`, `circular_motion()`, `custom_path_motion()`, `tap_sequence()`, `slide_trajectory()` |
| `stimulus.py` | `StimulusGenerator` | `generate_batch_stimuli(configs, time_steps)`, `_generate_single_stimulus()`, `to_device()` |
| `stimulus.py` | functions | `point_pressure_torch()`, `gaussian_pressure_torch()`, `edge_stimulus_torch()`, `gabor_texture_torch()` |
| `builder.py` | `StimulusBuilder` | Chainable stimulus builder API |

---

## Core (`sensoryforge/core/`)

| File | Class / Function | Notes |
|------|-----------------|-------|
| `grid.py` | `ReceptorGrid(BaseGrid)` | `get_coordinates()`, `get_receptor_coordinates()`, `get_grid_properties()`, `from_config()`, `to_dict()` |
| `grid.py` | `create_grid_torch()`, `get_grid_spacing()` | Factory functions |
| `grid_base.py` | `BaseGrid` | Abstract base for all grid types |
| `composite_grid.py` | `CompositeReceptorGrid(BaseGrid)` | `add_layer()`, `get_layer_coordinates()`, `get_all_coordinates()`, `list_layers()` — multi-layer receptor grid |
| `innervation.py` | `BaseInnervation` | `compute_weights()`, `from_config()`, `to_dict()` |
| `innervation.py` | `GaussianInnervation`, `UniformInnervation`, `OneToOneInnervation`, `DistanceWeightedInnervation` | Concrete strategies |
| `innervation.py` | `InnervationModule(nn.Module)` | `forward(mechanoreceptor_responses)`, `get_connection_density()`, `visualize_neuron_connections()` |
| `innervation.py` | `FlatInnervationModule(nn.Module)` | `forward(receptor_responses)`, `get_weights_per_neuron()` — used by MechanoreceptorTab |
| `innervation.py` | `create_innervation()`, `create_neuron_centers()`, `create_sa_innervation()`, `create_ra_innervation()` | Factory functions |
| `simulation_engine.py` | `SimulationEngine` | **Canonical engine** — `run(...)`, `_run_pop_from_drive(drive, filter_module, neuron_model, ...)` (static), `_build_grids()`, `_build_populations()`, `_stimulus_to_receptors()` |
| `generalized_pipeline.py` | `GeneralizedTactileEncodingPipeline` | Legacy multi-population pipeline; `from_config()`, `run()` |
| `pipeline.py` | `TactileEncodingPipelineTorch` | Original SA/RA pipeline; `forward(stimulus)`, `generate_stimulus()` |
| `batch_executor.py` | `BatchExecutor` | `execute_batch(stimulus_configs, ...)` |
| `experiment_manager.py` | `ExperimentManager` | Owns project dir (`stimuli/`, `results/`, `figures/`); passed to all tabs |
| `compression.py` | `CompressionOperator` | Population-to-population signal compression |
| `mechanoreceptors.py` | legacy helpers | Older SA/RA pipeline helpers |

---

## Config (`sensoryforge/config/`)

| File | Class / Function | Notes |
|------|-----------------|-------|
| `schema.py` | `SensoryForgeConfig` | Top-level dataclass; `from_dict()`, `to_dict()`, `from_yaml()`, `to_yaml()` |
| `schema.py` | `GridConfig` | Grid layer config; `from_dict()`, `to_dict()` |
| `schema.py` | `PopulationConfig` | Population config; `input_gain=50.0` default; `from_dict()`, `to_dict()` |
| `yaml_utils.py` | `load_yaml()`, `save_yaml()` | YAML I/O with merge-key support |

---

## Registry & Registration

| File | Class / Function | Notes |
|------|-----------------|-------|
| `registry.py` | `ComponentRegistry` | `register(name, cls)`, `create(name, **kwargs)`, `list_components()`, `get_param_spec(name)` |
| `registry.py` | `NEURON_REGISTRY`, `FILTER_REGISTRY`, `INNERVATION_REGISTRY`, `STIMULUS_REGISTRY`, `SOLVER_REGISTRY`, `GRID_REGISTRY`, `PROCESSING_REGISTRY` | Module-level singletons |
| `register_components.py` | `register_all()` | Registers all built-in components; called at pipeline import time |

---

## Solvers (`sensoryforge/solvers/`)

| File | Class | Notes |
|------|-------|-------|
| `base.py` | `BaseSolver` | Abstract ODE solver interface |
| `euler.py` | `EulerSolver` | Fixed-step Euler integration |
| `adaptive.py` | `AdaptiveSolver` | Wrapper for `torchdiffeq` / `torchode` (optional deps) |

---

## CLI (`sensoryforge/cli.py`)

Entry points: `run`, `validate`, `list-components`, `visualize`  
Key functions: `load_config_file()`, `validate_config()`, `run_simulation()`, `list_components()`  
Routing: canonical configs (`grids` + `populations` keys, no `pipeline` key) → `SimulationEngine`; legacy → `GeneralizedTactileEncodingPipeline`

---

## GUI (`sensoryforge/gui/`)

See **[gui_reference.md](gui_reference.md)** for widget names, signals, and cross-tab wiring.

| File | Class | Role |
|------|-------|------|
| `main.py` | `SensoryForgeWindow` | Main window; wires all tabs |
| `tabs/mechanoreceptor_tab.py` | `MechanoreceptorTab` | Grid + population config |
| `tabs/spiking_tab.py` | `SpikingNeuronTab` | Neuron model config + simulation run |
| `tabs/stimulus_tab.py` | `StimulusDesignerTab` | Stimulus creation + preview |
| `tabs/visualization_tab.py` | `VisualizationTab` | Post-simulation display |
| `tabs/batch_tab.py` | `BatchTab` | Batch sweep + SLURM export |
| `widgets/collapsible.py` | `CollapsibleGroupBox` | Expandable/collapsible panel widget |
| `visualization/base_panel.py` | `BasePanel` | Abstract visualization panel |
| `visualization/*.py` | `NeuronPanel`, `RasterPanel`, `FiringRatePanel`, `NeuronHeatmapPanel`, `ReceptorPanel`, `StimulusPanel`, `VoltagePanel` | Concrete panel types |
| `visualization/playback_bar.py` | `PlaybackBar` | Time scrubber; signals `time_changed(float)`, `play_toggled(bool)` |
| `protocol_backend.py` | `ProtocolBackend` | Low-level stimulus delivery for protocol suite |
| `figure_builder.py` | `FigureBuilder` | Export publication figures |
| `filter_utils.py` | helpers | Shared filter utilities for GUI |
