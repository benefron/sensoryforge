# Phase 3 B: GUI Grid Workspace Redesign Plan

## Problem
The mechanoreceptor tab still shows the old Phase 2 UI with:
- "Standard Grid" / "Composite Grid" dropdown (`cmb_grid_type`)
- Separate `standard_grid_widget` (rows/cols/spacing/center/arrangement)
- Separate `composite_grid_widget` (bounds + 3-column table: Name/Density/Arrangement)
- Target grid dropdown disabled unless composite mode

Phase 3 plan (B.1–B.4) calls for replacing this with a **unified grid list** where every grid entry has its own parameters, color, and offset.

## Design

### Unified Grid List
Replace `cmb_grid_type`, `standard_grid_widget`, and `composite_grid_widget` with:

1. **Grid list** (`QListWidget`) — each entry is a grid layer
2. **Per-grid editor panel** — shown below the list, edits the selected grid entry
3. **"Add Grid" / "Remove Grid" buttons** — manage the list
4. **"Generate All Grids" button** — builds the grid(s)

Each grid entry stores:
- `name` (str) — editable
- `arrangement` (str) — "grid", "poisson", "hex", "jittered_grid"
- `rows` / `cols` (int) — for grid/jittered_grid
- `density` (float) — for poisson/hex
- `spacing` (float) — mm
- `center_x`, `center_y` (float) — mm
- `offset_x`, `offset_y` (float) — shift to avoid overlap
- `color` (QColor) — for visualization

### Single-Grid Behavior
When only one grid exists, behavior is identical to old "Standard Grid" — generates a `GridManager`.

### Multi-Grid Behavior
When 2+ grids exist, generates a `CompositeReceptorGrid` with `add_layer()` for each. The bounds are computed from the union of all grids.

### Target Grid Dropdown
- Always enabled (even with one grid)
- Always populated with grid names + "(all receptors)"
- Default: "(all receptors)" for single grid; first grid for multi

### Visualization
- Each grid layer plotted with its user-defined color
- Hardcoded color palette removed

### Config Round-Trip
- `get_config()` exports a `grids` list (not `grid.type`)
- `set_config()` accepts both new `grids` list and legacy `grid` dict
- Legacy configs with `grid.type: standard` → single-entry grids list
- Legacy configs with `grid.type: composite` → multi-entry grids list

## Scope
- File: `sensoryforge/gui/tabs/mechanoreceptor_tab.py`
- No backend changes needed (CompositeReceptorGrid already supports offset + color)
- No test changes needed for backend (GUI tests may need update)

## Implementation Sequence
1. Add `_grid_entries` data structure and helper class
2. Replace grid section UI (remove dropdown, add list + editor)
3. Implement unified `_generate_grids()` method
4. Wire target grid dropdown (always active)
5. Update visualization to use per-grid colors
6. Update `get_config()` / `set_config()` with backward compat
7. Run tests, verify GUI launches
