# Phase 2 Integration Plan (GUI/CLI/YAML/Pipeline)

## Goals
- Expose Phase 2 components (CompositeGrid, DSL neurons, extended stimuli, adaptive solvers) without overwhelming users.
- Maintain backward compatibility with existing pipelines and GUI flows.
- Provide a stable YAML schema and a CLI workflow for reproducible runs.

## Guiding UX Principles
- **Core controls in main UI**: grid type, solver type, stimulus type, device, duration.
- **Advanced options gated**: DSL editor, solver tolerances, template overrides in an “Advanced” modal/panel.
- **YAML-first power path**: allow GUI to load/save YAML templates rather than making every field editable in-place.

## Pipeline Integration (Core)
1. Add a `PipelineFactory.from_config()` entry point in `sensoryforge/core/generalized_pipeline.py`.
2. Implement component registries for:
   - Grid (`GridManager`, `CompositeGrid`)
   - Stimuli (`gaussian`, `texture`, `moving`)
   - Filters (`SA`, `RA`, `custom`)
   - Neurons (`izhikevich`, `adex`, `dsl`)
   - Solvers (`euler`, `adaptive`)
3. Provide a YAML schema with validation (duplicate keys, required fields, type checks).
4. Keep existing `TactileEncodingPipelineTorch` intact for backward compatibility.

## CLI Integration
- Add `sensoryforge/cli.py` with commands:
  - `run <config.yml>`
  - `validate <config.yml>`
  - `list-components`
  - `visualize <config.yml>` (graph or textual summary)
- CLI should default to YAML validation before running.

## GUI Integration
- Add YAML load/save actions (menu + hotkeys).
- Add a compact “Component Selector” panel:
  - Grid type: Standard vs Composite
  - Solver type: Euler vs Adaptive
  - Neuron type: Hand-written vs DSL (advanced only)
  - Stimulus type: Gaussian, Texture, Moving
- Advanced modal:
  - DSL equation editor
  - Solver tolerances (rtol/atol)
  - Template overrides

## Testing Plan
- Unit: YAML validation, registry resolution, component instantiation.
- Integration: YAML → Pipeline → run, CompositeGrid population creation, DSL compile + run, adaptive solver selection.
- CLI: `run`, `validate`, `list-components` smoke tests.

## Documentation Plan
- Add user guide sections for each Phase 2 component.
- Provide YAML examples and CLI usage guide.
- Include a “GUI + YAML workflow” tutorial with screenshots.
