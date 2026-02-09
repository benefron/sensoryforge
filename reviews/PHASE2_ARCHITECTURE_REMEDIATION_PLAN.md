# Phase 2 Architecture Remediation Plan

**Date:** 2026-02-09  
**Status:** 🔴 CRITICAL — Architecture misalignment identified  
**Scope:** Fixing fundamental design issues in Grid, Innervation, Stimulus, and GUI

---

## Executive Summary

The current Phase 2 implementation has several critical architectural misalignments:

1. **CompositeGrid confusion** — Conflates **receptor grids** with **sensory neuron filters** (SA/RA)
2. **Missing innervation layer** — No way to connect receptor grid → sensory neurons for composite grids
3. **Stimulus functionality loss**  — Phase 2 broke the composability and flexibility of stimuli
4. **Protocol tab still wired** — Was supposed to be removed
5. **Missing innervation methods** — Docs don't document one-to-one, distance-based weighting, etc.

This plan restructures the architecture to align with the correct biological pipeline.

---

## The Correct Pipeline (Biological Ground Truth)

```
┌──────────────────────────────┐
│ 1. RECEPTOR GRID             │  ← Mechanoreceptors in skin (spatial points)
│    - Position in space       │
│    - Distribution pattern    │
│    - (NO filter/neuron type) │
└──────────────────────────────┘
              ↓
┌──────────────────────────────┐
│ 2. INNERVATION               │  ← Connections: receptors → sensory neurons
│    - One-to-one              │
│    - Gaussian (current)      │
│    - Distance-weighted       │
│    - User-extensible         │
└──────────────────────────────┘
              ↓
┌──────────────────────────────┐
│ 3. SENSORY NEURON LAYER      │  ← SA/RA neurons (already implemented)
│    - SA (Slowly Adapting)    │
│    - RA (Rapidly Adapting)   │
│    - Filters + Spiking       │
└──────────────────────────────┘
              ↑
┌──────────────────────────────┐
│ 4. STIMULUS                  │  ← Pressure patterns applied to receptor grid
│    - Type (Gaussian, edge...) │
│    - Optional movement       │
│    - Stackable/composable    │
│    - Additive superposition  │
└──────────────────────────────┘
```

### Key Insight: **Layers ≠ Filters**

- **Receptor layer** = spatial points (grid, poisson, hex arrangement)
- **Innervation** = how those points connect to neurons
- **Sensory neuron layer** = SA/RA filter types + spiking models

**CompositeGrid should stack receptor layers, NOT filter types.**

---

## Problem 1: Composite Grid Architecture

### Current (WRONG)

```python
grid = CompositeGrid(xlim=(-5, 5), ylim=(-5, 5))
grid.add_population("SA1", density=100, arrangement="grid", filter="SA")  # ← WRONG
grid.add_population("RA",  density=50,  arrangement="hex",  filter="RA")  # ← WRONG
```

**Issues:**
- Conflates receptor density with neuron filter type
- `filter="SA"` has no meaning at the receptor level
- Can't innervate because receptors ARE being treated as neurons
- GUI shows "SA/RA" dropdowns for receptor populations (biologically nonsensical)

### Correct (NEW)

```python
# Option A: Single grid with one arrangement
grid = ReceptorGrid(
    rows=40, cols=40,
    spacing=0.15,
    arrangement="grid"  # or "poisson", "hex", "jittered_grid"
)

# Option B: Composite grid with MULTIPLE RECEPTOR LAYERS
grid = CompositeReceptorGrid()
grid.add_layer("layer1", rows=40, cols=40, arrangement="grid")
grid.add_layer("layer2", rows=60, cols=60, arrangement="poisson")
grid.add_layer("layer3", rows=30, cols=30, arrangement="hex")

# Innervation: Connect receptor grid → sensory neurons
innervation_sa = create_innervation(
    receptor_coords=grid.get_layer_coordinates("layer1"),
    neuron_centers=sa_neuron_centers,
    method="gaussian",  # or "one_to_one", "distance_weighted"
    **params
)

innervation_ra = create_innervation(
    receptor_coords=grid.get_layer_coordinates("layer1"),
    neuron_centers=ra_neuron_centers,
    method="gaussian",
    **params
)
```

**Key Differences:**
1. Grid layers are **receptor arrangements only** (no filter types)
2. Innervation is **separate step** (connects receptors → neurons)
3. Filter type (SA/RA) assigned **at neuron population level**

---

## Problem 2: Missing Innervation Methods

### Currently Available
- ✅ Gaussian-weighted random sampling (only method implemented)

### Missing (Need to Implement)
- ❌ **One-to-one** — Each receptor connects to exactly one neuron
- ❌ **Distance-based weighting** — Connection strength = f(distance)
- ❌ **User-extensible** — Easy way to add custom innervation strategies

### Proposed API

```python
from sensoryforge.core.innervation import create_innervation

# Method 1: Gaussian (current)
W = create_innervation(
    receptor_coords, neuron_centers,
    method="gaussian",
    connections_per_neuron=28.0,
    sigma_d_mm=0.3,
    weight_range=(0.1, 1.0),
    seed=42
)

# Method 2: One-to-one
W = create_innervation(
    receptor_coords, neuron_centers,
    method="one_to_one",
    # Each receptor maps to nearest neuron
)

# Method 3: Distance-weighted
W = create_innervation(
    receptor_coords, neuron_centers,
    method="distance_weighted",
    max_distance_mm=1.0,     # Cutoff
    decay_function="exponential",  # or "linear", "inverse_square"
    decay_rate=2.0
)

# Method 4: Custom (user-defined)
from sensoryforge.core.innervation import BaseInnervation

class MyInnervation(BaseInnervation):
    def compute_weights(self, receptor_coords, neuron_centers, **kwargs):
        # User logic here
        return weight_tensor
```

### GUI Exposure

Each innervation method exposes different params:

| Method | GUI Controls |
|--------|--------------|
| Gaussian | connections_per_neuron, sigma_d_mm, weight_range, seed |
| One-to-one | (no params) |
| Distance-weighted | max_distance_mm, decay_function, decay_rate |
| Custom | (dynamically populated from class) |

---

## Problem 3: Stimulus Architecture

### Current (BROKEN)

In the GUI Phase 2 implementation:
- Stimulus type and movement are **orthogonal choices** (good intent, bad design)
- Texture/Moving are **separate tabs** (user can only pick one)
- Can't do "Gaussian + circular motion" or "Gabor + linear slide"
- Can't stack mult

iple stimuli (e.g., two Gaussian points)
- Lost additive superposition

### Correct (NEW)

```
Stimulus = Base Type + Optional Movement + Optional Composition
```

#### Base Types (choose one)
- Gaussian
- Point
- Edge
- Gabor (texture)
- EdgeGrating (texture)
- Noise (texture)

#### Movement (optional, applies to ANY base type)
- Static (default)
- Linear
- Circular  
- Custom path

#### Composition (optional, additive superposition)
- Single stimulus (default)
- Multiple stimuli (stack/add them)

### API Design

```python
from sensoryforge.stimuli import Stimulus

# Example 1: Static Gaussian
stim = Stimulus.gaussian(center=(0, 0), amplitude=1.0, sigma=0.3)

# Example 2: Moving Gaussian (circular motion)
stim = Stimulus.gaussian(center=(0, 0), amplitude=1.0, sigma=0.3) \
    .with_motion("circular", center=(0,0), radius=1.0, num_steps=100)

# Example 3: Static Gabor
stim = Stimulus.gabor(center=(0, 0), wavelength=0.5, orientation=45)

# Example 4: Moving Gabor (linear slide)
stim = Stimulus.gabor(center=(0, 0), wavelength=0.5, orientation=45) \
    .with_motion("linear", start=(0,0), end=(2,0), num_steps=50)

# Example 5: Multiple Gaussians (Braille simulation)
stim = Stimulus.compose([
    Stimulus.gaussian(center=(0.0, 0.0), amplitude=1.0, sigma=0.2),
    Stimulus.gaussian(center=(0.5, 0.0), amplitude=1.0, sigma=0.2),
    Stimulus.gaussian(center=(1.0, 0.0), amplitude=1.0, sigma=0.2),
], mode="add")  # or "max", "mean"

# Example 6: Multiple Gaussians with DIFFERENT motion
stim = Stimulus.compose([
    Stimulus.gaussian((0,0), 1.0, 0.2).with_motion("linear", (0,0), (1,0), 50),
    Stimulus.gaussian((1,0), 1.0, 0.2).with_motion("static"),
])
```

### GUI Design

```
┌─────────────────────────────────────────┐
│ Stimulus Designer                       │
├─────────────────────────────────────────┤
│ Stimulus List (stackable):             │
│   [+] Add Stimulus                      │
│   ┌───────────────────────────────────┐ │
│   │ Stimulus #1                       │ │
│   │ Type: [Gaussian ▼]                │ │
│   │ Movement: [Static ▼]              │ │  ← Dropdown: Static/Linear/Circular/Custom
│   │ ┌─ Gaussian Params ─────────────┐ │ │
│   │ │ Center X: [0.0]               │ │ │
│   │ │ Center Y: [0.0]               │ │ │
│   │ │ Amplitude: [1.0]              │ │ │
│   │ │ Sigma: [0.3]                  │ │ │
│   │ └───────────────────────────────┘ │ │
│   │ ┌─ Motion Params (hidden if Static) ─┐
│   │ │ (shows params based on motion type) │
│   │ └─────────────────────────────────┘ │ │
│   │ [Remove] [Duplicate]              │ │
│   └───────────────────────────────────┘ │
│   ┌───────────────────────────────────┐ │
│   │ Stimulus #2                       │ │
│   │ ... (another stimulus)            │ │
│   └───────────────────────────────────┘ │
│                                         │
│ Composition Mode: [Add ▼]              │  ← Add/Max/Mean
│ Target Grid Layer: [Layer1 ▼]          │  ← Which receptor layer
│                                         │
│ [Preview Canvas] ← Shows superposition  │
└─────────────────────────────────────────┘
```

---

## Problem 4: ProtocolSuiteTab Still Wired

### Current
- ProtocolSuiteTab is the 4th tab in main.py
- User said it should be deleted
- Takes up screen space and adds complexity

### Fix
- Remove from `sensoryforge/gui/tabs/__init__.py`
- Remove from `sensoryforge/gui/main.py`
- Remove `ProtocolSuiteTab` class file (or move to `archive/`)
- Update tests to expect 3 tabs instead of 4

---

## Implementation Roadmap

### Phase 1: Core Architecture (Blocking)

#### Task 1.1: Rename and Refactor Grid
- [ ] Rename `GridManager` → `ReceptorGrid` (clarity)
- [ ] Add `arrangement` parameter to `ReceptorGrid.__init__()`
  - Options: `"grid"`, `"poisson"`, `"hex"`, `"jittered_grid"`
  - Default: `"grid"` (current behavior)
- [ ] Move arrangement generation logic from `CompositeGrid` to `ReceptorGrid`
- [ ] Update tests

#### Task 1.2: Redefine CompositeGrid
- [ ] Rename `CompositeGrid` → `CompositeReceptorGrid`
- [ ] Remove `filter` parameter from `add_population()` → `add_layer()`
- [ ] Each layer is just a `ReceptorGrid` with a name
- [ ] Layers share xlim/ylim but have independent arrangement/density
- [ ] Update YAML schema:
  ```yaml
  grid:
    type: composite
    xlim: [-5.0, 5.0]
    ylim: [-5.0, 5.0]
    layers:
      - name: layer1
        rows: 40
        cols: 40
        arrangement: grid
      - name: layer2
        rows: 60
        cols: 60
        arrangement: poisson
  ```
- [ ] Update tests

#### Task 1.3: Extend Innervation Methods
- [ ] Create `BaseInnervation` abstract class in `innervation.py`
- [ ] Implement `GaussianInnervation` (refactor existing code)
- [ ] Implement `OneToOneInnervation`
  - Each receptor → nearest neuron (Voronoi-like)
- [ ] Implement `DistanceWeightedInnervation`
  - Weight = f(distance), user-configurable decay
- [ ] Add `create_innervation()` factory function
- [ ] Document extensibility pattern for user-defined methods
- [ ] Update tests

#### Task 1.4: Refactor Stimulus API
- [ ] Create `Stimulus` builder class in `stimuli/__init__.py`
- [ ] Static factories: `.gaussian()`, `.point()`, `.edge()`, `.gabor()`, `.edge_grating()`, `.noise()`
- [ ] Add `.with_motion(type, **params)` method
- [ ] Add `.compose([stim1, stim2, ...], mode="add")` static factory
- [ ] Backward compatibility: keep old functions, mark deprecated
- [ ] Update tests

---

### Phase 2: GUI Updates

#### Task 2.1: MechanoreceptorTab — Grid Refactor
- [ ] **Standard Grid Section**
  - Add "Arrangement" dropdown: Grid, Poisson, Hex, Jittered Grid
  - Show arrangement-specific params (if any)
  
- [ ] **Composite Grid Section**
  - Remove "Filter" dropdown from population table
  - Rename "Population" → "Layer"
  - Table columns: Name, Rows, Cols, Arrangement, Density (calculated), [Remove]
  - Add layer button → shows dialog for layer config
  - Visual: Stackable list (like neuron populations, can show/hide each)
  
- [ ] **Innervation Section** (NEW)
  - For **each neuron population**:
    - Dropdown: "Innervation Method" (Gaussian, One-to-One, Distance-Weighted, Custom)
    - Dynamic param widgets based on selected method
    - "Target Receptor Layer" dropdown (if composite grid)
  - This replaces current hardcoded Gaussian params

#### Task 2.2: StimulusDesignerTab — Stackable Stimuli
- [ ] Replace current UI with **stimulus list** (QListWidget with custom items)
- [ ] Each stimulus item has:
  - Type dropdown (Gaussian, Point, Edge, Gabor, EdgeGrating, Noise)
  - Movement dropdown (Static, Linear, Circular, Custom)
  - Param section (type-specific)
  - Motion param section (motion-specific, hidden if Static)
  - Remove/Duplicate buttons
  
- [ ] Add "Composition Mode" dropdown (Add, Max, Mean)
- [ ] Add "Target Grid Layer" dropdown (which receptor layer)
- [ ] Preview canvas shows **superposition** of all stimuli
- [ ] "Add Stimulus" button to stack more

#### Task 2.3: Remove ProtocolSuiteTab
- [ ] Remove import from `main.py`
- [ ] Remove tab instantiation and `addTab()` call
- [ ] Remove from `tabs/__init__.py`
- [ ] Update integration tests (expect 3 tabs)
- [ ] Move `ProtocolSuiteTab` class to `archive/` folder

---

### Phase 3: CLI & YAML

#### Task 3.1: Update YAML Schema
- [ ] Grid section supports `arrangement` for standard grids
- [ ] Composite grid uses `layers` (no filter)
- [ ] Innervation section per neuron population
  ```yaml
  populations:
    - name: "SA #1"
      neuron_type: SA
      innervation:
        method: gaussian  # or one_to_one, distance_weighted
        target_layer: layer1  # (if composite)
        gaussian_params:
          connections_per_neuron: 28.0
          sigma_d_mm: 0.3
          weight_range: [0.1, 1.0]
          seed: 42
  ```
- [ ] Stimulus section supports list of stimuli
  ```yaml
  stimulus:
    composition_mode: add
    target_layer: layer1
    stimuli:
      - type: gaussian
        movement: linear
        gaussian_params: {...}
        motion_params: {...}
      - type: gabor
        movement: static
        gabor_params: {...}
  ```

#### Task 3.2: Update Pipeline
- [ ] `GeneralizedTactileEncodingPipeline.from_config()` parses new schema
- [ ] Grid instantiation uses new `ReceptorGrid` / `CompositeReceptorGrid`
- [ ] Innervation uses `create_innervation()` factory
- [ ] Stimulus uses `Stimulus.compose()`
- [ ] Backward compatibility for old YAML (best effort)

---

### Phase 4: Documentation

#### Task 4.1: Architecture Docs
- [ ] Update `docs/user_guide/overview.md` with correct pipeline diagram
- [ ] Create `docs/user_guide/receptor_grids.md`
  - Explain receptor vs. neuron distinction
  - Show arrangement types
  - Explain when to use composite
- [ ] Create `docs/user_guide/innervation.md`
  - Document all innervation methods
  - Show extensibility pattern
  - Examples for each method
- [ ] Update `docs/user_guide/stimuli.md`
  - New builder API
  - Stackable stimuli examples
  - Movement compositions

#### Task 4.2: Update Copilot Instructions
- [ ] Rewrite architecture section in `.github/copilot-instructions.md`
- [ ] Emphasize receptor/neuron distinction
- [ ] Document new APIs and conventions

#### Task 4.3: Migration Guide
- [ ] Create `docs/MIGRATION_v0.2_to_v0.3.md`
- [ ] Document breaking changes
- [ ] Provide code examples (old → new)
- [ ] YAML migration examples

---

## Testing Strategy

### Unit Tests
- [ ] `test_receptor_grid.py` — arrangement types, density, bounds
- [ ] `test_composite_receptor_grid.py` — multi-layer stacking
- [ ] `test_innervation.py` — all innervation methods
- [ ] `test_stimulus_builder.py` — new Stimulus API
- [ ] `test_stimulus_composition.py` — additive superposition

### Integration Tests
- [ ] `test_pipeline_with_composite_grid.py` — end-to-end
- [ ] `test_yaml_round_trip_v3.py` — new schema round-trip
- [ ] `test_gui_grid_and_innervation.py` — GUI config → backend
- [ ] `test_gui_stimulus_stacking.py` — GUI multi-stimulus → backend

### Regression Tests
- [ ] Ensure old single-grid configs still work (backward compat)
- [ ] Ensure old Gaussian stimulus still works (deprecated but functional)

---

## Migration Path (Avoiding Breakage)

### Phase 1: Additive Changes (Non-Breaking)
1. Add `arrangement` to `ReceptorGrid` (default `"grid"` preserves behavior)
2. Add new innervation methods (old `create_innervation_map_tensor` still works)
3. Add new `Stimulus` builder API (old functions still work)
4. Add tests for new features

### Phase 2: Deprecation Warnings
1. Mark `CompositeGrid.add_population(filter=...)` as deprecated
2. Mark old stimulus functions as deprecated
3. Add migration warnings in logs

### Phase 3: Breaking Changes (v0.3.0)
1. Remove `filter` parameter from `CompositeGrid`
2. Rename `CompositeGrid` → `CompositeReceptorGrid`
3. Update YAML schema (with version detection)
4. Remove deprecated stimulus functions

---

## Success Criteria

### Architecture
- [x] Clear separation: receptors (grid) → innervation → neurons (SA/RA)
- [x] Composite grids stack **receptor layers**, not filter types
- [x] Multiple innervation methods available and user-extensible
- [x] Stimuli are stackable and composable with optional movement

### GUI
- [x] Arrangement selection available for standard grids
- [x] Composite grid shows layers without filter confusion
- [x] Innervation method selection per neuron population
- [x] Stimulus designer supports stacking multiple stimuli
- [x] ProtocolSuiteTab removed

### CLI/YAML
- [x] YAML schema aligns with new architecture
- [x] Pipeline parses and executes new configs
- [x] Backward compatibility maintained (or migration guide provided)

### Documentation
- [x] Clear architecture diagrams  
- [x] Complete API documentation for all new features
- [x] User guide examples
- [x] Migration guide from v0.2 to v0.3

### Testing
- [x] 100% test coverage for new components
- [x] Integration tests pass
- [x] No regression in existing tests

---

## Timeline Estimate

| Phase | Tasks | Estimated Effort | Blocking Dependencies |
|-------|-------|------------------|----------------------|
| **1. Core Architecture** | 4 tasks | 16-20 hours | None |
| **2. GUI Updates** | 3 tasks | 12-16 hours | Phase 1 complete |
| **3. CLI & YAML** | 2 tasks | 6-8 hours | Phase 1 complete |
| **4. Documentation** | 3 tasks | 10-12 hours | Phases 1-3 complete |
| **Total** | 12 tasks | **44-56 hours** | Sequential dependencies |

---

## Open Questions

1. **Composite grid density vs. explicit counts?**
   - Current: Density (receptors/mm²) → calculate count
   - Alternative: User specifies rows/cols directly?
   - **Recommendation:** Support both (density OR rows/cols, not both)

2. **Stimulus composition modes beyond "add"?**
   - Add (superposition)
   - Max (take maximum at each point)
   - Mean (average)
   - Other?
   - **Recommendation:** Start with Add, Max, Mean (cover 90% of use cases)

3. **Should we support per-stimulus target layers?**
   - Current plan: All stimuli in a composition target the same layer
   - Future: Each stimulus could target a different layer (for multi-modal)
   - **Recommendation:** Start with single target, add multi-target in v0.4

4. **Grid arrangement extensibility?**
   - Should users be able to define custom arrangements?
   - **Recommendation:** Not in v0.3 (add in future if requested)

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Prioritize tasks** (can parallelize Core + Docs)
3. **Create feature branch** (`feature/architecture-remediation-v0.3`)
4. **Implement Phase 1** (core architecture)
5. **Test + Review** before proceeding to GUI
6. **Iterate** through Phases 2-4

---

## Conclusion

The current Phase 2 implementation conflates **receptor grids** (spatial points) with **sensory neurons** (SA/RA filters), breaking the biological pipeline. This remediation plan:

1. **Separates concerns** — receptors → innervation → neurons
2. **Adds missing features** — innervation methods, stimulus stacking
3. **Fixes GUI confusion** — removes filter from grid, adds stimulus list
4. **Maintains compatibility** — migration path with deprecation warnings
5. **Documents properly** — clear architecture, examples, migration guide

**Estimated effort: 44-56 hours of focused implementation.**

Once complete, SensoryForge will have a **biologically accurate**, **user-friendly**, and **extensible** architecture suitable for publication and community adoption.
