# Sensor arrays

A sensor array is a `GridConfig`: a set of receptor coordinates, optionally split into named
channels and/or named layers. This page covers the geometry side; see
[Units and Shapes](units_and_shapes.md) for the channel *tensor axis* and
[Receptive Fields](../user_guide/receptive_fields.md) for how a population wires onto an array.

## Why receptor index is not pixel index

A stimulus frame is a raster: `[H, W]` values on a regular pixel lattice. A receptor array is,
in general, a *scattered set of points* -- and before Phase 2 Wave L, `SimulationEngine` assumed
the two were the same thing: receptor `k` got pixel `k` of the flattened frame. That is only
true for a `"grid"` arrangement whose resolution matches the frame exactly.

`SimulationEngine` now samples the stimulus at each receptor's own `(x, y)` position with
bilinear interpolation (`_sample_stimulus_at_receptors`, `core/simulation_engine.py`), and takes
the old reshape only when it is providing exactly the same answer (a `"grid"` arrangement whose
row/column count equals the frame's `H`/`W`) — kept as a fast path, not a special case of
correctness. Every other arrangement is sampled the same way: hex, Poisson, jittered, blue-noise,
imported coordinates and composite layers all go through the same code, so none of them needs its
own reshape logic and none of them can silently disagree with what the frame actually shows at
that point in space.

## The four built-in arrangements

| Arrangement | Coordinates | Reproducible with a seed? |
|---|---|---|
| `"grid"` | Regular rectangular lattice, `rows` x `cols` at `spacing` mm | N/A (deterministic) |
| `"hex"` | Hexagonally packed, density-derived from `rows`/`cols`/`spacing` | N/A (deterministic) |
| `"jittered_grid"` | Regular lattice plus Gaussian jitter | Yes, via `GridConfig.seed` (F-050) |
| `"blue_noise"` | Jitter + Lloyd-style relaxation (blue-noise-like spacing) | Yes, via `GridConfig.seed` (F-050) |
| `"poisson"` | Jittered grid at a target density (an approximation, not true Poisson-disk sampling) | Yes, via `GridConfig.seed` (F-050) |

All five are registered grid-arrangement classes (`core/grid_arrangements.py`) discoverable via
`GRID_REGISTRY`, so `sensoryforge list-components` shows them and a plugin can add a sixth (see
[Adding a Grid Arrangement](../extending/add_grid_arrangement.md)).

## Imported coordinates

`GridConfig.coords_file` (Phase 2, Wave L1) points at an `[M, 2]` CSV (`x,y`, optional header) or
`.pt` file of receptor coordinates in mm, instead of `rows`/`cols`/`spacing`/`arrangement`. The
engine wraps the file's coordinates as a single-layer `CompositeReceptorGrid`, whose bounds are
the coordinates' own bounding box. This is the path a hand-digitised layout, a fabrication file,
or a plugin arrangement (built once, then exported) all take to reach a simulation without any
engine code needing to know how the points were generated -- see
`docs/examples/grid_arrangement_plugin.py` for a worked example that builds a plugin arrangement
and imports it this way.

## Composite grids

`GridConfig(arrangement="composite", layers=[...])` (Phase 2, Wave L4) builds a
`CompositeReceptorGrid` from an ordered list of layer specs. Each entry needs a `name` and
exactly one of:

- `density` (+ optional `arrangement`, `offset`, `seed`, `color`) -- generated the same way a
  single-layer `GridConfig` would be, via `CompositeReceptorGrid.add_layer`;
- `coordinates` -- an inline `[n, 2]` list;
- `coords_file` -- a CSV/`.pt` file, loaded the same way `GridConfig.coords_file` is.

```yaml
grids:
  - name: skin_patch
    arrangement: composite
    rows: 20        # only used to size the shared bounding box
    cols: 20
    spacing: 0.15
    layers:
      - name: sa_fine
        density: 60.0
        arrangement: grid
      - name: ra_coarse
        density: 20.0
        arrangement: hex
```

**Layer order is the receptor-index contract.** `CompositeReceptorGrid.get_all_coordinates()`
concatenates layers in insertion order, so receptor index `k` depends on which layers were added
before the one `k` falls in -- exactly like row-major pixel flattening is the contract for a
`"grid"` arrangement. The grid's `provenance["layers"]` records each layer's name and receptor
count in that same order, so a bundle reader can slice a flat `[M, 2]`/`[N, M]` array back into
per-layer pieces without re-deriving how they were built.

A population innervates every layer of its target grid by default. `PopulationConfig.target_layers`
(a list of layer names) restricts it to a named subset; building on layer A's coordinates never
depends on layer B's density, arrangement, or receptor count -- changing B rebuilds only B.

## Channels

See [Units and Shapes](units_and_shapes.md#the-channel-axis-phase-2-wave-l) for the tensor-level
rule. Geometrically, `GridConfig.channels` just names the sensor planes an array carries (for
example `["pressure", "temperature"]` on one physical patch); the receptor *coordinates* are
shared across channels -- a channel is a value at each point, not a different point set. Wave L
carries the axis through configuration and stimulus composition; which channel(s) a population
reads is a Wave M concern (multi-input populations).
