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

Every shape's `amplitude` defaults to 1.0, the peak of one element. This is the
convention of every stimulus in SensoryForge: each named type also peaks at 1.0 when
its amplitude is not set.

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

## Slides, repeated contacts and vibration

A layer's timing and a layer's `modulation` can describe more than one press. Every field
below defaults off, and a stimulus that does not use them renders exactly as before
(`tests/unit/test_layered_golden.py` pins this). The examples follow
`tests/unit/test_layered_episode.py`.

**A slide.** `timing.slide_ms` adds a stretch after the hold during which the pattern moves,
when the motion has `span: slide`. The contact then ramps down.

```yaml
- shape:   {kind: gaussian, sigma_mm: 0.3}
  motion:  {kind: linear, start: [0, 0], end: [2, 0], span: slide}
  timing:  {onset_ms: 0, ramp_up_ms: 0, hold_ms: 20, slide_ms: 20, ramp_down_ms: 0}
```

**Repeated contacts.** `timing.contacts` repeats the touch, hold, slide and release,
`pause_ms` apart (the pause is exactly zero). Motion is spread over all the contacts, so each
re-touch starts where the last one ended. `contacts` above 1 needs an explicit `hold_ms`.

```yaml
- shape:   {kind: gaussian, sigma_mm: 0.3}
  motion:  {kind: linear, start: [0, 0], end: [2, 0], span: slide}
  timing:  {onset_ms: 0, ramp_up_ms: 0, hold_ms: 10, slide_ms: 10, ramp_down_ms: 0,
           contacts: 2, pause_ms: 10}
```

**Vibration and taps.** `modulation` multiplies the envelope, counted from each touch:
`{kind: sine, frequency_hz, depth, phase_deg}` swings the amplitude between 1 and 1 − depth,
and `{kind: pulses, rate_hz, duty, edge_ms, depth}` repeats indentation.

```yaml
- shape:      {kind: gaussian, sigma_mm: 0.5}
  modulation: {kind: sine, frequency_hz: 50, depth: 0.5}
  timing:     {onset_ms: 0, ramp_up_ms: 0, hold_ms: 100, ramp_down_ms: 0}
- shape:      {kind: disc, diameter_mm: 0.6}
  modulation: {kind: pulses, rate_hz: 50, duty: 0.5, edge_ms: 2}
  timing:     {onset_ms: 0, ramp_up_ms: 0, hold_ms: 100, ramp_down_ms: 0}
```

**Braille by dots.** A `braille` pattern takes `dots: "125 14"`, cells separated by spaces,
each a string of dot numbers 1 to 6 (1 to 3 down the left column, 4 to 6 down the right), as
an alternative to `text`.

**Signed carriers.** `signed: true` on a `grating` or `gabor` shape uses `cos` in place of
the raised cosine `(1 + cos) / 2`, so the lobes between the stripes are negative.

Worlds draw these fields from ranges, see [Worlds and data sets](worlds.md).

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
  as layers. Each element peaks at 1.0, like the named types; `repeated_pattern`'s six
  Gaussians overlap, so their sum peaks at about 3.9.
- `braille_word`, `bumpy_texture`, `probe_sequence` — examples of stacking.

## Named types: amplitude and sign

Every named stimulus type peaks at 1.0 when its `amplitude` is not set, and none renders
a negative value by default. `gabor` and `texture` (the same Gabor patch, `texture` with
a wider window and longer wavelength) draw amplitude × Gaussian window ×
(1 + cos(carrier)) / 2, the same raised cosine as the layered `gabor` shape. Set
`signed: true` in the stimulus `params` (an advanced parameter on the Stimulus screen)
for the zero-mean form, amplitude × window × cos(carrier), whose lobes between the
stripes are negative, for example to model signed contrast in vision. A negative
pressure drives a negative current through the filters.

```yaml
stimulus:
  type: gabor
  params:
    signed: true
```
