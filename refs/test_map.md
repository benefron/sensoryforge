# Test Map

Inventory of all test files and what they cover.  
Run tests: `pytest tests/unit/ -q` or `pytest tests/ -q --tb=short`  
Full test inventory format: file → feature → key test names → special setup.

---

## Unit Tests (`tests/unit/`)

### Neurons
| File | Covers | Key tests | Notes |
|------|--------|-----------|-------|
| `test_pytorch_neurons.py` | All neuron forward passes | Shape consistency for Izhikevich, AdEx, MQIF | Parametrized over models |
| `test_neuron_api_h6.py` | Neuron API contract | Output shapes, dtype, spike bool type | H6 milestone |
| `test_izhikevich_consistency.py` | Izhikevich parameter sensitivity | Determinism, stochastic variants | |
| `test_izhikevich_rebuilt.py` | Rebuilt Izhikevich integration | Single-neuron correctness | |
| `test_izhikevich_rebuilt_batch.py` | Batched Izhikevich | Batch dimension consistency | |
| `test_izhikevich_tuple_m1.py` | Tuple parameter handling | `(mean, std)` parameter format | M1 milestone |
| `test_oscillation_regression.py` | Oscillation suppression (item 8) | `v_floor` clamp prevents drift in Izhikevich/AdEx/MQIF; `clip_to_positive` in SA filter | Item 8 regression guard |
| `test_sa2_fix.py` | SA2 neuron fix | SA2 correctness after refactor | |
| `test_neuron_visualization.py` | Neuron output plotting | No crash on `plot_spike_train()`, `plot_voltage()` | |
| `test_model_dsl.py` | Equation DSL compiler | Parse, compile, forward pass | Requires `sympy` |
| `test_dsl_isymbol_h8.py` | DSL iSymbol handling | Symbol parsing edge cases | H8 milestone |

### Filters
| File | Covers | Key tests | Notes |
|------|--------|-----------|-------|
| `test_filters.py` | SA/RA filter dynamics | Shape preservation, temporal axis, steady-state | |
| `test_filters_vs_theory.py` | Filter vs Pierzowski theory | Numerical comparison vs analytical solution | |
| `test_filter_reset_state.py` | Filter state reset | State cleared between calls | |
| `test_noise.py` | Noise modules | `MembraneNoiseTorch`, `ReceptorNoiseTorch` output shape and statistics | |
| `test_gain_defaults.py` | Input gain default (item 9) | `test_gain_50_produces_spikes` (amplitude=3.0), `test_gain_1_produces_silence` (amplitude=5.0), `test_population_config_default_input_gain` (==50.0), `test_spiking_tab_code_sets_gain_spinbox_to_50` (source inspection) | Item 9 guard |

### Stimuli
| File | Covers | Key tests | Notes |
|------|--------|-----------|-------|
| `test_gaussian_stimulus.py` | Gaussian stimulus | Shape, amplitude, sigma scaling | |
| `test_moving_stimulus.py` | Moving stimulus | Linear, circular, custom path trajectories | |
| `test_texture_stimulus.py` | Gabor / EdgeGrating | Texture field shapes | |
| `test_extended_stimuli.py` | Extended stimulus types | New stimulus types via ParamSpec | |
| `test_stimulus_builder.py` | `StimulusBuilder` | Chainable builder API | |
| `test_stimulus_vectorize_h7.py` | Vectorized stimulus generation | Batch generation performance | H7 milestone |
| `test_gabor_import_m3.py` | Gabor import | `GaborTexture` importable after refactor | M3 milestone |
| `test_stimulus_grid_inmemory.py` | In-memory workflow (item 7) | Full pipeline without filesystem | Item 7 guard; **skip in full suite** — slow |

### Grid & Innervation
| File | Covers | Key tests | Notes |
|------|--------|-----------|-------|
| `test_innervation_locality.py` | Local connectivity | Gaussian innervation spatial falloff | |
| `test_innervation_square_grid.py` | Square grid innervation | Weight matrix for square receptor array | |
| `test_innervation_spatial_coverage.py` | Full coverage | All receptors have at least one connection | |
| `test_innervation_center_alignment.py` | Center alignment | Neuron centers align with receptor grid | |
| `test_innervation_methods.py` | All innervation strategies | Gaussian, Uniform, OneToOne, DistanceWeighted | |
| `test_composite_grid.py` | Multi-layer grids (item 2) | `TestCoordinateConsistency` — shared axis across grids | Item 2 guard |
| `test_grid_spacing.py` | Grid spacing edge cases | 1×1 degenerate grid, `get_grid_spacing()` guard | Item 12 |
| `test_edge_cases.py` | Full-pipeline edge cases (item 12) | Min/max grid sizes, empty populations, zero drive | Item 12 guard |
| `test_mechanoreceptor_m7.py` | Mechanoreceptor module | M7 milestone tests | |
| `test_compression_operator.py` | Population compression | `CompressionOperator` shapes | |

### GUI — Mechanoreceptor Tab
| File | Covers | Key tests | Notes |
|------|--------|-----------|-------|
| `test_grid_population_ux.py` | Grid/population UX (items 1-4) | Add/remove grid, auto-default `neurons_per_row`, tab rename, per-col toggle, non-square population | Qt required; uses `isHidden()` |
| `test_population_csv.py` | CSV import/export (item 5) | `_CSVPopulationModule` duck type, export 3-file structure, manifest content, round-trip allclose, csv_folder persistence, receptor mismatch zero-fill | Qt for preservation test |
| `test_expert_mode.py` | Basic/Expert mode toggle (item 6) | `test_basic_mode_hides_mech_widgets`, `test_expert_mode_shows_mech_widgets`, `test_basic_mode_hides_spiking_widgets`, `test_expert_mode_shows_spiking_widgets` + QSettings persistence | Module-scoped fixtures; `_APP` module-level guard against QApplication GC |

### GUI — Other Tabs
| File | Covers | Key tests | Notes |
|------|--------|-----------|-------|
| `test_stimulus_tab_gui.py` | StimulusDesignerTab | Stimulus type switching, parameter change | Qt required |
| `test_gui_agent_a.py` | GUI agent A milestone | GUI interaction tests | Qt required |
| `test_gui_agent_b.py` | GUI agent B milestone | GUI interaction tests | Qt required |
| `test_gui_agent_c.py` | GUI agent C milestone | GUI interaction tests | Qt required |
| `test_gui_agent_d.py` | GUI agent D milestone | GUI interaction tests | Qt required |
| `test_protocol_backend.py` | ProtocolBackend | Low-level protocol execution | Qt required |

### Config & Registry
| File | Covers | Key tests | Notes |
|------|--------|-----------|-------|
| `test_config_schema.py` | SensoryForgeConfig schema | `from_dict`, `to_dict`, `from_yaml`, `to_yaml` round-trips | |
| `test_component_registry.py` | ComponentRegistry | `register`, `create`, `get_param_spec`, unregistered error | |
| `test_param_spec.py` | ParamSpec + registry (item 13/14) | Auto-discovery, `get_param_spec()` on all stimulus types | Items 13-14 guard |
| `test_extensibility.py` | Plugin extension points | Custom neuron/filter/stimulus registration | |
| `test_project_registry.py` | ProjectRegistry | Project listing and metadata | |
| `test_base_classes.py` | BaseNeuron, BaseFilter, BaseStimulus | Interface contract tests | |

### Pipeline & Simulation
| File | Covers | Key tests | Notes |
|------|--------|-----------|-------|
| `test_generalized_pipeline_yaml.py` | YAML → GeneralizedPipeline | Round-trip fidelity | |
| `test_yaml_utils.py` | YAML merge keys | `load_yaml()` with anchors | |
| `test_solvers.py` | ODE solver integration | Euler, adaptive; shape + convergence | |
| `test_solver_refactor.py` | Solver refactoring | No regression after solver module split | |
| `test_dt_mismatch.py` | dt consistency | Warns when stimulus dt ≠ neuron dt | |
| `test_simulation_engine_features.py` | SimulationEngine features | Multi-population, canonical routing | |
| `test_run_pop_from_drive.py` | `_run_pop_from_drive` (item 15/16) | Static method output shape/dtype, filter + gain + noise path | Items 15-16 guard |
| `test_cli_canonical_repoint.py` | CLI → SimulationEngine routing (item 15) | Canonical config detected and routed correctly | Item 15 guard |
| `test_batch_executor.py` | BatchExecutor (legacy) | Local batch execution, output format | |
| `test_batch_executor_canonical.py` | BatchExecutor (canonical) | Canonical config batch path | |
| `test_experiment_manager.py` | ExperimentManager | Project dir creation, path resolution | |
| `test_phase3_features.py` | Phase 3 features | Timeline, batch, CLI improvements | |

---

## Integration Tests (`tests/integration/`)

| File | Covers | Key tests | Notes |
|------|--------|-----------|-------|
| `test_pipeline.py` | Full SA/RA pipeline | `test_forward_static_gaussian_produces_spikes`, override configs | `TactileEncodingPipelineTorch` |
| `test_yaml_pipeline.py` | YAML → pipeline round-trip | Load, run, compare output | |
| `test_simulation_engine.py` | SimulationEngine end-to-end | Multi-population canonical run | |
| `test_engine_parity.py` | Engine vs pipeline parity (item 16) | 21 tests comparing `SimulationEngine` vs `GeneralizedTactileEncodingPipeline` output | Item 16 guard |
| `test_registry_integration.py` | Registry integration | `register_all()` + create all built-in components | |
| `test_gui_phase2.py` | Phase 2 GUI features | DSL neuron in GUI, solver selection | Qt required |
| `test_gui_cli_parity.py` | GUI ↔ YAML round-trip | Save config from GUI → reload → check equivalence | Qt required |
| `test_phase3_pipeline.py` | Phase 3 pipeline | Tab rename (item 1), timeline, batch | Qt required; item 1 guard |
| `test_regression_refactoring.py` | Refactoring regressions | Output consistency after architectural changes | |

---

## Common Fixtures & Guards

**`conftest.py`** provides:
- `sample_pressure_grid: np.ndarray [80, 80]`
- `sample_spike_train: torch.Tensor [1000, 100]`
- `test_parameters: dict` — `{grid_size, sa_count, ra_count, dt, timesteps}`
- Thread limits: `OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, torch.set_num_threads(1)`

**Qt app guard** (used in all GUI tests):
```python
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
```

**Module-scoped fixture pattern** (used in `test_expert_mode.py` to avoid pyqtgraph GC crash):
```python
_APP = None  # module-level prevents GC

def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    return _APP

@pytest.fixture(scope="module")
def mech_tab():
    _ensure_app()
    return MechanoreceptorTab()
```

**Key caveat**: Use `isHidden()` not `isVisible()` for headless widget visibility checks. `isVisible()` requires the parent chain to be shown on screen.
