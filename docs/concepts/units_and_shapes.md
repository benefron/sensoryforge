# Units and shapes

The canonical reference for tensor shapes, units and coordinate conventions used throughout
SensoryForge. Base classes and docstrings link here instead of repeating the rules.

## The pipeline

```
Stimulus  [batch, time, H, W]              (or [batch, time, C, H, W] with C > 1 -- see below)
    ↓  sampled at receptor coordinates → receptor responses [batch, time, M]
    ↓  ReceptiveFieldBank (weights [N, M])
    ↓  [batch, time, N_neurons]
    ↓  Filter (SAFilterTorch / RAFilterTorch)
    ↓  [batch, time, N_neurons]  in mA
    ↓  Neuron (Izhikevich / AdEx / MQIF / DSL-compiled)
Spikes    [batch, time, N_neurons]  bool  -- or State [batch, time, N_neurons] float for an
                                              analog (thresholdless DSL) readout
```

- **Time:** milliseconds (`ms`) at every user-facing API; seconds only inside ODE integration.
- **Space:** millimetres (`mm`) throughout.
- **Batch dimension is always first:** `[batch, ...]`.
- **Coordinates are `(x, y)` in mm everywhere inside SensoryForge.** For a regular
  `ReceptorGrid(grid_size=(rows, cols))`, the meshgrid is built with `indexing="ij"`, so the
  **first** frame axis is x (`rows`) and the **second** is y (`cols`): frame element `[i, j]`
  sits at `(x[i], y[j])`, and receptor `k = i * cols + j`. Convert at the boundary when
  importing coordinates that use a different order (for example pressure-simulation's
  `[y, x]` centres).

## The channel axis (Phase 2, Wave L)

A `GridConfig` names the sensor planes it carries in `channels` (default `["value"]`, a single
unnamed channel). A stimulus tensor is `[batch, time, H, W]` when the target grid has one
channel — the shape used everywhere before Wave L, unchanged — and becomes
`[batch, time, C, H, W]` when `C = len(grid.channels) > 1`, with the channel axis immediately
after time and before the two spatial axes (the same position `torch.nn.functional.grid_sample`
expects for a batched image tensor, once time is folded into the batch axis: see
`SimulationEngine._sample_stimulus_at_receptors`).

`StimulusConfig.channel` names which plane of the target grid one stimulus config drives.
Several stimulus configs with different `channel` values compose into one multi-channel
tensor built by `render_stimulus` (`sensoryforge/stimuli/render.py`, Wave K): each stimulus
fills its own plane along axis 2; planes with no stimulus are left zero. `channel=None` (the
default) targets the single/first channel, so existing single-channel configs are unaffected.

A population's receptive-field bank projects receptor responses `[batch, time, M]` regardless
of how many channels the stimulus had — which channel(s) a population reads is a Wave M
concern (multi-input populations); Wave L only carries the axis through the stimulus and
sensor-array configuration.

## Receptor sampling (Phase 2, Wave L3)

`SimulationEngine` no longer assumes receptor index equals stimulus pixel index. It samples
the stimulus frame at each receptor's `(x, y)` coordinate with bilinear interpolation
(`torch.nn.functional.grid_sample`), except on the fast path — a `"grid"`-arrangement receptor
lattice whose resolution matches the stimulus frame exactly — where it reshapes as before,
bit-identical to earlier releases. See `SimulationEngine._sample_stimulus_at_receptors` for the
index algebra between SensoryForge's `(x, y)` convention and `grid_sample`'s `(width, height)`
convention, which do not agree naively.

## Rendering a stimulus for a non-lattice arrangement (Phase 0, F-076)

A `"poisson"` or `"hex"` receptor arrangement has no regular lattice, so it cannot itself supply a
render canvas the way a `"grid"` arrangement's `ReceptorGrid.get_coordinates()` does.
`sensoryforge.stimuli.canvas.stimulus_canvas(grid_cfg)` is the one helper every config-driven entry
point (`sensoryforge run`, `BatchExecutor`, the GUI's run path) uses to build a
stimulus's render canvas: for `"grid"` it reproduces `ReceptorGrid.get_coordinates()` bit-for-bit,
and for every other arrangement — `"poisson"`, `"hex"`, `"jittered_grid"`, `"blue_noise"`,
`"composite"`, and an imported `coords_file` — it renders on a regular canvas spanning the
receptor array's actual extent (the same `rows`/`cols`/`spacing`/`center` bounds
`SimulationEngine._build_grids` uses to build that arrangement's own grid, or the imported
coordinates' bounding box when `coords_file` is set). The stimulus is then sampled at each
receptor's real `(x, y)` position as described above — the canvas's resolution only needs to cover
the array's extent, not match the receptor count or layout.
