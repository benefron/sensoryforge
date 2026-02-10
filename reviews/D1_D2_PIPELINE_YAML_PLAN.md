# D.1-D.2 Pipeline & YAML Update Plan

## Scope

Update `generalized_pipeline.py` and YAML config files to support Phase 3 features:
composite grids with offset/color, `FlatInnervationModule`, processing layers,
timeline stimuli, and repeated-pattern stimuli.

## D.1: YAML Schema Updates

### 1. `DEFAULT_CONFIG` in `generalized_pipeline.py`
- Add `grid` section with `type`, `populations` including `offset`, `color`
- Add `processing_layers` list (default empty → IdentityLayer)
- No changes to `stimuli` in DEFAULT_CONFIG (handled via `generate_stimulus()`)

### 2. `examples/example_config.yml`
- Add commented Phase 3 composite grid example with offset/color
- Add commented processing_layers section
- Add timeline and repeated_pattern stimulus examples

### 3. `sensoryforge/config/default_config.yml`
- Update Phase 2 composite_grid section to include offset/color fields
- Add processing_layers section

## D.2: Pipeline Code Updates

### 1. New imports (line ~13)
- `FlatInnervationModule` from `.innervation`
- `ProcessingPipeline` from `.processing`
- `TimelineStimulus`, `RepeatedPatternStimulus` from `sensoryforge.stimuli.builder`

### 2. `__init__` — composite grid creation (lines ~154-190)
- When `grid.type == "composite"`, use `add_layer()` with `offset` and `color`
  kwargs instead of deprecated `add_population(filter=...)`
- Store `composite_grid` for later use in innervation

### 3. New `_create_processing_layers()` method
- Read `processing_layers` from config (default `[]`)
- Build `ProcessingPipeline.from_config(configs)`
- Store as `self.processing_pipeline`

### 4. `_create_innervation()` update (lines ~250+)
- If `self.composite_grid` is not None and has layers, offer
  `FlatInnervationModule` path using flat coordinates from
  `get_all_coordinates()` or per-layer `get_layer_coordinates()`
- Backward-compatible: original `InnervationModule` path remains default

### 5. `generate_stimulus()` update (lines ~455-470)
- Add `"timeline"` branch → delegates to `TimelineStimulus`
- Add `"repeated_pattern"` branch → delegates to `RepeatedPatternStimulus`

### 6. `forward()` update (lines ~870+)
- After `mechanoreceptor_responses` assignment, apply
  `self.processing_pipeline(mechanoreceptor_responses)` before innervation
- Return `processing_applied` in intermediates dict

## Backward Compatibility

All changes are additive. Existing configs and API calls continue to work:
- `CompositeGrid` alias still works
- `add_population()` still calls `add_layer()`
- `InnervationModule` remains default when no composite grid configured
- `IdentityLayer` is used when processing_layers list is empty/absent
