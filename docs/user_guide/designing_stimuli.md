# Designing stimuli

A **layered** stimulus is a stack of layers. Each layer is four independent choices,
and the layers add up (or take the maximum) frame by frame.

| Part | Kinds | Main parameters |
|---|---|---|
| **Shape** — one element | `gaussian`, `disc`, `bar`, `grating`, `gabor` | amplitude; sigma, diameter and soft edge, width and length, wavelength, orientation, phase |
| **Pattern** — where copies go | `single`, `grid`, `list`, `random`, `braille` | position; rows, columns, spacing and an on/off mask; a list of points; count, region, minimum distance, amplitude jitter and seed; braille text |
| **Motion** — how the pattern moves | `none`, `linear`, `circular`, `path` | start and end, radius and turns, waypoints; during the hold or the whole layer |
| **Timing** | — | `onset_ms`, `ramp_up_ms`, `hold_ms`, `ramp_down_ms` (unset hold: until the ramp down ends the run) |

Every shape is non-negative: a pressure between 0 and its amplitude. The grating and
the Gabor use a raised cosine. Positions are `(x, y)` in mm and times in ms.

## A braille word over a bumpy texture

```yaml
stimulus:
  type: layered
  combine: sum
  layers:
    - shape:   {kind: disc, amplitude: 1.0, diameter_mm: 0.8, edge_mm: 0.2}
      pattern: {kind: braille, text: "hello", dot_spacing_mm: 2.5, cell_spacing_mm: 6.0}
      motion:  {kind: linear, start: [5, 0], end: [-30, 0], span: hold}
      timing:  {onset_ms: 0, ramp_up_ms: 50, hold_ms: 800, ramp_down_ms: 50}
    - shape:   {kind: gaussian, amplitude: 0.5, sigma_mm: 0.3}
      pattern: {kind: random, count: 120, width_mm: 30, height_mm: 12,
                min_distance_mm: 0.8, amplitude_jitter: 0.3, seed: 0}
      motion:  {kind: linear, start: [8, 0], end: [-8, 0]}
      timing:  {onset_ms: 200, ramp_up_ms: 100, ramp_down_ms: 100}
```

A braille cell can also be drawn by hand with a `grid` pattern of 3 rows × 2 columns and
a `mask` such as `"10 10 01"` (row by row, top to bottom).

## In the GUI

On the **Stimulus** screen choose the type **layered**. The layer list has Add,
Duplicate, Remove and Move up/down; untick a layer to leave it out. For the selected
layer, the Shape, Pattern and Motion sections each start with a kind; changing the kind
resets that part to the new kind's defaults. The Timing section sets the ramps and hold.
**Start from preset…** loads an editable starting point.

## Presets

`sensoryforge.stimuli.presets.preset(name, duration_ms)` returns a layered stimulus:

- `moving_edge`, `braille`, `drifting_grating`, `ramp_gaussian` — pressure-simulation's
  four stimuli. They match the named stimulus types to within 2% of peak (the named types
  sample their ramps one step differently; the moving edge matches exactly). The named
  types themselves are unchanged.
- `gaussian`, `moving`, `repeated_pattern`, `gabor`, `edge_grating` — the other named types
  as layers.
- `braille_word`, `bumpy_texture`, `probe_sequence` — examples of stacking.
