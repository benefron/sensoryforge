# Phase 3 Architecture Remediation Plan

**Date:** 2026-02-10  
**Status:** 🟡 READY FOR REVIEW  
**Scope:** Fixing Composite Grid UX, Innervation Wiring, Stimulus Timeline, Extensibility Architecture  

---

## Executive Summary

After a thorough investigation of the entire codebase — backend (`core/`, `stimuli/`, `filters/`, `neurons/`, `solvers/`), GUI (`gui/tabs/`), pipeline (`generalized_pipeline.py`), and all planning/review docs — this plan addresses the following concrete issues:

### What Works ✅
- **Backend `ReceptorGrid`** — supports all 4 arrangements (grid, poisson, hex, jittered_grid)
- **Backend `CompositeReceptorGrid`** — correctly uses `add_layer()` without filter conflation
- **Backend innervation classes** — `GaussianInnervation`, `OneToOneInnervation`, `DistanceWeightedInnervation`, factory function all exist
- **Backend `Stimulus` builder** — `StaticStimulus`, `MovingStimulus`, `CompositeStimulus`, fluent API all exist
- **3 tabs only** — Protocol tab already removed from `__init__.py` and `main.py`

### What's Broken/Missing 🔴
1. **Composite Grid GUI** — Uses a small `QTableWidget` with just 3 columns (Name/Density/Arrangement) and a dropdown switcher. No colors, no offsets, no way to select which grid a neuron population wires into. Basically unusable.
2. **Neuron Population ↔ Composite Grid wiring** — `_on_add_population()` always calls `population.instantiate(self.grid_manager)`, but when in composite mode `self.grid_manager` is `None` → you can't add any neuron populations to a composite grid.
3. **Innervation distance weighting** — The `GaussianInnervation` backend does all-to-all probability sampling (multinomial over ALL receptors). It should respect sigma as a spatial locality constraint, not just a probability weight. Distance-weighted innervation exists but GUI doesn't properly expose threshold/locality.
4. **Stimulus GUI** — Has 5 type buttons (Gaussian/Point/Edge/Texture/Moving) where Texture and Moving are treated as separate, mutually exclusive *types* rather than orthogonal properties. No timeline concept. Stack exists but doesn't properly compose or handle timing. Can't create repeating patterns (textures via copy+shift). No individual vs. global timing control.
5. **Architecture extensibility** — No clear hook points for intermediate processing layers (ON/OFF fields, lateral inhibition, cross-grid fusion).

---

## The Correct Architecture (User's Vision)

### Composite Grid ≈ List of Named Grids

```
┌─────────────────────────────────────────────┐
│ Grid Workspace (shared coordinate space)     │
│                                              │
│  ┌─ Grid "SA1 skin" ─────────────────────┐  │
│  │ Arrangement: hex  │ Color: 🔵 Blue    │  │
│  │ Rows: 40 Cols: 40 │ Offset: (0,0)     │  │
│  └────────────────────────────────────────┘  │
│  ┌─ Grid "RA1 skin" ─────────────────────┐  │
│  │ Arrangement: grid │ Color: 🟠 Orange  │  │
│  │ Rows: 60 Cols: 60 │ Offset: (0.05, 0) │  │
│  └────────────────────────────────────────┘  │
│  ┌─ Grid "SA2 deep" ─────────────────────┐  │
│  │ Arrangement: poisson │ Color: 🟢 Green│  │
│  │ Density: 30 │ Offset: (0, 0.05)       │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  [+ Add Grid]                                │
└─────────────────────────────────────────────┘
```

- There is NO "Standard vs. Composite" dropdown. There is ONE grid creation flow.
- You start with one grid. You can add more grids → that makes it composite.
- Each grid has: name, arrangement, size params, color, spatial offset.
- Each grid is independently editable and removable.
- All grids share the coordinate space (they overlay).

### Neuron Populations Wired to Specific Grids

For each neuron population, you choose:
- **Which grid** it is wired to (dropdown listing all grids)
- **Innervation method** (gaussian, one-to-one, distance-weighted)
- **Innervation parameters** (sigma, connections, decay, etc.)

The innervation respects spatial locality — distances are computed from neuron centers to receptors, and the sigma/mean-innervation determines how many and how strongly nearby receptors connect. NOT all-to-all.

### Stimulus Timeline

```
┌─────────────────────────────────────────────────────┐
│ Timeline (total duration: 500 ms, dt: 0.5 ms)      │
│                                                      │
│ ┌─ Sub-Stimulus "Gaussian 1" ────────────────────┐  │
│ │ Shape: Gaussian  │ Motion: Static               │  │
│ │ Center: (0,0)  Sigma: 0.3  Amplitude: 1.0      │  │
│ │ Onset: 0 ms  Duration: 500 ms                   │  │
│ │ Timing: Ramp 50ms → Plateau → Ramp-off 50ms    │  │
│ └─────────────────────────────────────────────────┘  │
│ ┌─ Sub-Stimulus "Moving Bar" ────────────────────┐  │
│ │ Shape: Edge(bar) │ Motion: Linear               │  │
│ │ Start: (-2,0) End: (2,0)  Width: 0.05          │  │
│ │ Onset: 100 ms  Duration: 300 ms                 │  │
│ └─────────────────────────────────────────────────┘  │
│ ┌─ Sub-Stimulus "Braille dots" ──────────────────┐  │
│ │ Shape: Gaussian (repeated pattern)              │  │
│ │ Pattern: 3×2 grid, spacing 0.5mm               │  │
│ │ Onset: 0 ms  Duration: 500 ms                   │  │
│ │ Motion: Static                                   │  │
│ └─────────────────────────────────────────────────┘  │
│                                                      │
│ Composition: Add  │  [+ Add Sub-Stimulus]           │
│                                                      │
│ [====█=========================================]     │  ← playback scrubber
│ 0 ms                                       500 ms   │
└─────────────────────────────────────────────────────┘
```

Key ideas:
- Each sub-stimulus has its own **onset time** and **duration** within the global timeline
- Shape and Motion are **orthogonal**: any shape can have any motion
- **Repeating patterns**: a shape can be copied/shifted (N×M grid with spacing) to create textures
- **Global controls**: total time, dt, playback
- **Individual controls**: per sub-stimulus onset, duration, timing envelope
- Can save timeline as a whole, or individual sub-stimuli separately

### Extensibility Layer Hooks

The architecture must be ready to insert processing layers between grid → neurons:

```
Grid Layer(s)
    ↓
[Future: Cross-grid fusion layer]
    ↓
[Future: ON/OFF center-surround filter]
    ↓
[Future: Lateral inhibition]
    ↓
Innervation → Neuron Populations
    ↓
SA/RA Filters → Spiking Neurons
```

This means the data flow must be a **composable pipeline** of `nn.Module` layers, not hardcoded grid→innervation→neuron wiring.

---

## Detailed Problem Analysis

### Problem 1: Composite Grid GUI

**Current state** (in `mechanoreceptor_tab.py`):
- Line 249: `QComboBox` with "Standard Grid" / "Composite Grid" ← forces a binary choice
- Line 317-325: Composite grid uses a `QTableWidget` with 3 columns (Name, Density, Arrangement)
- Line 600: `_generate_composite_grid()` creates `CompositeReceptorGrid` but sets `self.grid_manager = None`
- Line 760: `_on_add_population()` checks `if self.grid_manager is None` → shows warning → **BLOCKS population creation for composite grids**
- No color assignment per grid layer
- No offset support
- No way to select which grid a neuron population connects to

**Fix**: Unify the grid creation into a single list-based workflow. Remove the Standard/Composite dropdown. Always present a list of grids. If there's 1 grid, it's standard. If there are multiple, it's composite. Each grid entry gets its own controls for arrangement, size, color, and offset.

### Problem 2: Neuron Population → Grid Wiring

**Current state**:
- `NeuronPopulation.instantiate()` takes a `GridManager` → creates `InnervationModule` which needs meshgrids (xx, yy)
- `InnervationModule.__init__()` calls `grid_manager.get_coordinates()` for xx, yy → only works for grid/jittered_grid arrangements
- For composite grids, there's NO `grid_manager` (it's set to None)
- There's no UI to choose which grid a population connects to

**Fix**: 
- Add a `target_grid` field to `NeuronPopulation`
- When creating populations for composite grids, create a temporary `ReceptorGrid` from the selected layer's coordinates
- Or better: refactor `InnervationModule` to work with flat `[N,2]` receptor coordinates instead of requiring meshgrids

### Problem 3: Innervation Locality

**Current state** (in `innervation.py`):
- `GaussianInnervation.compute_weights()`: computes distances to ALL receptors, uses gaussian weights as probabilities for multinomial sampling → connections CAN be far away (just unlikely)
- `DistanceWeightedInnervation.compute_weights()`: has `max_distance_mm` cutoff → correct locality behavior

**Fix**: The Gaussian innervation should also have a max-distance cutoff. Beyond `k * sigma_d_mm` (e.g., 3σ), connection probability should be zero. This ensures spatial locality while preserving the Gaussian weighting within range. This is a backend change to `GaussianInnervation`.

### Problem 4: Stimulus GUI Architecture

**Current state** (in `stimulus_tab.py`):
- 5 type buttons: Gaussian, Point, Edge, **Texture**, **Moving** — but Texture and Moving are types, not modifiers
- Texture group shows sub-types (Gabor, Edge Grating, Noise) with separate params
- Moving group shows sub-types (Linear, Circular, Slide) with separate params
- Motion Mode radio buttons (Static/Moving) exist but only affect the basic types
- Stack section exists but each entry is a flat `StimulusConfig` — no timeline onset/duration
- No repeating pattern feature
- No individual timing per sub-stimulus

**Fix**: Redesign the stimulus tab:
1. Remove "Texture" and "Moving" as *types* — they become properties
2. Shape types: Gaussian, Point, Edge/Bar, Gabor, Edge Grating, Noise
3. Each sub-stimulus has: Shape + Motion (orthogonal) + Timing (onset, duration, envelope)
4. Add "Repeat Pattern" feature: copy a shape on an N×M grid with configurable spacing
5. Timeline scrubber showing all sub-stimuli on a temporal axis

### Problem 5: Extensibility Architecture

**Current state**:
- Pipeline is `Grid → InnervationModule → Filters → Neurons` — hardwired
- No interface for inserting intermediate processing layers
- `CompositeReceptorGrid` doesn't support cross-layer operations

**Fix**: Define a `ProcessingLayer` base class that sits between grid and neurons. For now, implement it as a pass-through. Future layers (ON/OFF, lateral inhibition, cross-grid fusion) will subclass it. The pipeline should accept a list of processing layers.

---

## Implementation Plan

### Phase A: Backend Fixes (No GUI changes)

#### A.1: Add Spatial Offset to CompositeReceptorGrid
- Add `offset: Tuple[float, float] = (0.0, 0.0)` parameter to `add_layer()`
- Apply offset to generated coordinates: `coords[:, 0] += offset[0]; coords[:, 1] += offset[1]`
- Store offset in layer config
- Update tests

#### A.2: Add Color Metadata to CompositeReceptorGrid
- Add `color: Optional[Tuple[int,int,int,int]]` parameter to `add_layer()`
- Store in layer config metadata — purely for GUI use

#### A.3: Fix GaussianInnervation Spatial Locality
- Add `max_sigma_distance: float = 3.0` parameter (defaults to 3σ cutoff)
- After computing Gaussian weights, zero out weights beyond `max_sigma_distance * sigma_d_mm`
- This makes the innervation respect spatial locality while preserving stochastic sampling within range

#### A.4: Add ProcessingLayer Base Class
- Create `sensoryforge/core/processing.py` with `BaseProcessingLayer(nn.Module)`
- Abstract method: `forward(receptor_responses: torch.Tensor, metadata: dict) -> torch.Tensor`
- Implement `IdentityLayer` (pass-through) as default
- Update pipeline to accept a list of processing layers between grid and innervation

#### A.5: Add Onset/Duration to Stimulus Builder
- Extend `StimulusConfig` (and `StaticStimulus`/`MovingStimulus`) with `onset_ms: float` and `duration_ms: float`
- `CompositeStimulus.forward()` should handle per-stimulus timing within a global timeline
- Add `repeat_pattern(nx, ny, spacing_x, spacing_y)` feature to `Stimulus` builder — auto-generates grid of copies

#### A.6: Make InnervationModule Work with Flat Coordinates
- Refactor `InnervationModule` to accept `receptor_coords: torch.Tensor [N, 2]` as an alternative to `grid_manager`
- This allows wiring neuron populations to any receptor layer, including non-grid arrangements from composite grids

---

### Phase B: GUI Redesign — Grid Workspace

#### B.1: Remove Standard/Composite Dropdown
- Remove `cmb_grid_type`, `_on_grid_type_changed`, `standard_grid_widget`, `composite_grid_widget`
- Replace with unified "Grid Workspace" that always shows a list of grids

#### B.2: Grid List Widget
- Replace `composite_pop_table` with a custom `QListWidget` where each entry is a collapsible card:
  ```
  ┌───────────────────────────────────────┐
  │ 🔵 "SA1 receptors"    [⚙] [🗑]      │
  │ Arrangement: [hex ▼]                  │
  │ Rows: [40] Cols: [40] Spacing: [0.15] │
  │ Offset X: [0.0] Offset Y: [0.0]      │
  │ Density: 100/mm² (for poisson/hex)    │
  │ Receptors: 1600                        │
  └───────────────────────────────────────┘
  ```
- Each grid has:
  - Name (editable text)
  - Color (pick button + swatch)
  - Arrangement dropdown (grid/poisson/hex/jittered_grid)
  - Size controls (rows/cols for grid types, density for poisson/hex)
  - Spacing (for grid/jittered_grid)
  - Coordinate offset (dx, dy) — small shift to avoid exact overlap
  - Count display (calculated)
- Shared coordinate bounds (xlim, ylim) or per-grid center+spacing
- "Add Grid" button at bottom
- Generate button applies all grids at once

#### B.3: Neuron Population → Grid Wiring
- Add "Target Grid" dropdown to population creation panel
  - Lists all grids by name
  - Default: first grid
- When adding a population:
  - Get receptor coordinates from the selected grid layer
  - Create neuron centers within that grid's bounds
  - Compute innervation weights using receptor_coords and neuron_centers directly
- This works regardless of composite/single grid
- Store `target_grid_name` in `NeuronPopulation`

#### B.4: Visualization Update
- Plot all grids with their assigned colors (scatter plots)
- When a neuron population is selected, show its connections to the target grid
- Color-code by grid layer in the plot

---

### Phase C: GUI Redesign — Stimulus Timeline

#### C.1: Reshape Type Selection
- Remove "Texture" and "Moving" as type buttons
- New type buttons: **Gaussian, Point, Bar, Gabor, Grating, Noise**
- Each type shows its own parameter panel

#### C.2: Add Orthogonal Motion Controls
- For ANY shape type, add a "Motion" section:
  - Motion type: Static, Linear, Circular, Custom Path
  - Motion parameters shown/hidden based on selection
- This replaces the old "Motion Mode" radio + "Moving" type button

#### C.3: Sub-Stimulus List with Timeline Properties
- Each sub-stimulus in the stack gets:
  - Name (editable)
  - Shape type + params
  - Motion type + params
  - **Onset (ms)**: when this sub-stimulus starts in the global timeline
  - **Duration (ms)**: how long it lasts
  - **Temporal envelope**: ramp-up, plateau, ramp-down (per sub-stimulus, or use global)
  - **Use global timing**: checkbox that syncs to the global envelope

#### C.4: Repeating Pattern Feature
- For any shape, add a "Repeat Pattern" toggle
- When enabled, shows: N copies X, N copies Y, Spacing X, Spacing Y
- The shape is auto-copied in an N×M grid pattern
- Preview shows the repeated pattern

#### C.5: Timeline Scrubber
- Horizontal bar showing the global time range
- Each sub-stimulus shown as a colored bar segment at its onset+duration
- Playback cursor for animation
- Click to jump to any time point

#### C.6: Stimulus Saving
- Save entire timeline as one stimulus file
- Save individual sub-stimuli as separate files
- Load and compose

---

### Phase D: Pipeline & YAML Updates

#### D.1: Update YAML Schema
```yaml
# New grid format
grids:
  - name: "SA1 receptors"
    arrangement: hex
    rows: 40
    cols: 40
    spacing_mm: 0.15
    offset: [0.0, 0.0]
    color: [66, 135, 245, 255]
  - name: "RA1 receptors"
    arrangement: grid
    rows: 60
    cols: 60
    spacing_mm: 0.12
    offset: [0.05, 0.0]
    color: [245, 135, 66, 255]

# Populations reference grids by name
populations:
  - name: "SA #1"
    neuron_type: SA
    target_grid: "SA1 receptors"
    neurons_per_row: 10
    innervation:
      method: gaussian
      connections_per_neuron: 28.0
      sigma_d_mm: 0.3
      max_sigma_distance: 3.0
      weight_range: [0.1, 1.0]
      seed: 42

# Stimulus timeline
stimulus:
  total_time_ms: 500.0
  dt_ms: 0.5
  composition_mode: add
  sub_stimuli:
    - name: "Center gaussian"
      shape: gaussian
      shape_params:
        center: [0.0, 0.0]
        amplitude: 1.0
        sigma: 0.3
      motion: static
      onset_ms: 0.0
      duration_ms: 500.0
      envelope:
        ramp_up_ms: 50.0
        plateau_ms: 400.0
        ramp_down_ms: 50.0
    - name: "Sliding bar"
      shape: edge
      shape_params:
        orientation_deg: 90.0
        width: 0.05
        amplitude: 0.8
      motion: linear
      motion_params:
        start: [-2.0, 0.0]
        end: [2.0, 0.0]
      onset_ms: 100.0
      duration_ms: 300.0
      envelope:
        ramp_up_ms: 20.0
        plateau_ms: 260.0
        ramp_down_ms: 20.0

# Processing layers (future extensibility)
processing_layers: []
```

#### D.2: Update `GeneralizedTactileEncodingPipeline`
- Parse new `grids` list format (backward-compat with old `grid` dict)
- Parse new `stimulus.sub_stimuli` with onset/duration
- Support `processing_layers` (empty list = pass-through for now)

---

### Phase E: Tests & Documentation

#### E.1: Tests
- `test_grid_workspace.py` — single grid, multi-grid, offsets, colors
- `test_innervation_locality.py` — verify Gaussian cutoff works
- `test_processing_layers.py` — identity layer, future extensibility
- `test_stimulus_timeline.py` — onset, duration, composition
- `test_stimulus_repeat_pattern.py` — N×M copy+shift
- `test_gui_grid_list.py` — GUI grid creation workflow
- `test_gui_stimulus_timeline.py` — GUI stimulus workflow
- `test_yaml_new_schema.py` — round-trip with new format

#### E.2: Documentation
- Update [docs/user_guide/](docs/user_guide/) with new grid workflow
- Update [docs/tutorials/](docs/tutorials/) with composite grid + stimulus timeline examples
- Update copilot instructions with new architecture

---

## Implementation Order (Priority)

### Sprint 1: Backend Foundations (est. 8-10 hours)
1. **A.1** — Grid offset support
2. **A.2** — Grid color metadata
3. **A.3** — GaussianInnervation spatial cutoff
4. **A.6** — InnervationModule flat-coordinate support
5. **A.5** — Stimulus onset/duration + repeat pattern

### Sprint 2: Grid GUI Overhaul (est. 12-16 hours)
1. **B.1** — Remove Standard/Composite dropdown
2. **B.2** — New grid list widget
3. **B.3** — Neuron population → grid wiring
4. **B.4** — Visualization update

### Sprint 3: Stimulus GUI Overhaul (est. 12-16 hours)
1. **C.1** — Reshape type selection (remove Texture/Moving types)
2. **C.2** — Orthogonal motion controls
3. **C.3** — Sub-stimulus with timeline properties
4. **C.4** — Repeating pattern feature
5. **C.5** — Timeline scrubber
6. **C.6** — Stimulus saving

### Sprint 4: Pipeline, YAML, Tests (est. 8-10 hours)
1. **A.4** — ProcessingLayer base class
2. **D.1** — YAML schema update
3. **D.2** — Pipeline update
4. **E.1** — Tests
5. **E.2** — Docs

**Total estimated effort: 40-52 hours**

---

## Key Design Decisions

### 1. No More Standard vs. Composite Dropdown
**Rationale**: The distinction is confusing and limits composability. A single grid IS a composite grid with one layer. The UI should let you add more grids freely.

### 2. Shape and Motion Are Orthogonal
**Rationale**: Any shape (Gaussian, bar, Gabor) can have any motion (static, linear, circular). Coupling them (as in the current "Texture" and "Moving" types) limits combinatorics.

### 3. Stimulus Timeline with Onset/Duration
**Rationale**: Real experiments require precise timing control. Each sub-stimulus needs its own onset and duration within a global timeline. This also enables creating complex stimulus sequences (e.g., tap-hold-release).

### 4. Repeating Patterns via Copy+Shift
**Rationale**: Textures (gratings, dot arrays, braille) are often regular patterns. Rather than special-casing them, we allow any shape to be replicated on an N×M grid. This is simpler and more general.

### 5. Processing Layer Hooks
**Rationale**: SensoryForge needs to grow toward supporting ON/OFF center-surround fields, lateral inhibition, and cross-modal fusion. Defining the interface NOW (even as pass-through) ensures the architecture can grow without breaking changes.

### 6. Innervation Spatial Locality
**Rationale**: The current Gaussian innervation can assign connections to distant receptors (just with low probability). This is biologically unrealistic. A hard cutoff at 3σ ensures locality while preserving the Gaussian receptive field shape.

---

## Migration / Backward Compatibility

- **Old YAML configs** with `grid: {type: standard, ...}` will still parse (converted internally to single-grid list)
- **Old YAML configs** with `grid: {type: composite, ...}` will still parse (converted to multi-grid list without offsets)
- **CompositeReceptorGrid** Python API unchanged (offset and color are optional params with defaults)
- **Old stimulus configs** without onset/duration default to onset=0, duration=total_time
- **`GaussianInnervation`** gets a new optional param (`max_sigma_distance`) with a large default that preserves old behavior if not explicitly set

---

## Open Questions for User

1. **Grid coordinate space**: Should all grids share a single (xlim, ylim) bounding box, or should each grid define its own center+spacing and the shared space be computed as the union?
   - **Recommendation**: Each grid has its own center+spacing. The plot auto-ranges to show all. Composite mode uses shared xlim/ylim as before.

2. **Neuron center placement**: Currently neurons are placed on a square lattice within the grid bounds. Should we also support hex/poisson arrangement for neuron centers?
   - **Recommendation**: Keep square lattice for now. Add as future enhancement.

3. **Stimulus preview for non-grid arrangements**: Currently preview requires meshgrids (grid/jittered_grid). How to handle poisson/hex?
   - **Recommendation**: For preview, interpolate poisson/hex coordinates onto a regular grid. For actual simulation, use the native coordinates.

---

## Files That Will Be Modified

### Backend
- [sensoryforge/core/composite_grid.py](sensoryforge/core/composite_grid.py) — add offset, color params
- [sensoryforge/core/innervation.py](sensoryforge/core/innervation.py) — add spatial cutoff to Gaussian, flat-coord support
- [sensoryforge/core/processing.py](sensoryforge/core/processing.py) — **NEW** — ProcessingLayer base class
- [sensoryforge/stimuli/builder.py](sensoryforge/stimuli/builder.py) — add onset, duration, repeat_pattern
- [sensoryforge/stimuli/stimulus.py](sensoryforge/stimuli/stimulus.py) — may need minor updates for timeline support
- [sensoryforge/core/generalized_pipeline.py](sensoryforge/core/generalized_pipeline.py) — new YAML parsing, processing layers

### GUI
- [sensoryforge/gui/tabs/mechanoreceptor_tab.py](sensoryforge/gui/tabs/mechanoreceptor_tab.py) — **MAJOR REWRITE** — grid list, population wiring
- [sensoryforge/gui/tabs/stimulus_tab.py](sensoryforge/gui/tabs/stimulus_tab.py) — **MAJOR REWRITE** — timeline, orthogonal motion, repeat patterns
- [sensoryforge/gui/main.py](sensoryforge/gui/main.py) — minor updates for new config format

### Tests
- [tests/unit/test_composite_grid.py](tests/unit/test_composite_grid.py) — update for offsets, colors
- [tests/unit/test_innervation.py](tests/unit/test_innervation.py) — add locality tests
- New test files for processing layers, stimulus timeline, GUI workflows

### Config
- [sensoryforge/gui/default_params.json](sensoryforge/gui/default_params.json) — update defaults
- [examples/example_config.yml](examples/example_config.yml) — new schema example

---

## Conclusion

This plan addresses every issue raised:
1. ✅ **Composite grid** → becomes a simple list of grids with colors, offsets, and arrangement options
2. ✅ **Neuron population wiring** → each population targets a specific grid, innervation respects spatial locality
3. ✅ **Stimulus** → shape + motion are orthogonal, timeline with onset/duration, repeating patterns, composition
4. ✅ **Architecture extensibility** → processing layer hooks for future ON/OFF, lateral inhibition, cross-grid fusion
5. ✅ **Modular and extensible** → all built on `nn.Module` base classes with YAML serialization

Awaiting user review before implementation begins.
