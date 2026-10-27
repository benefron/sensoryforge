---
name: sensoryforge-design-world
description: Turn a plain-language description of a set of stimuli into a validated SensoryForge world file, and show a preview of what it draws. Use when someone describes the stimuli they want to generate or study (shapes, motions, vibrations, textures, kinds of input and how each varies), or wants to change an existing world, in whatever words they put it.
---

# Design a world

A world is a declared population of stimuli: classes of stimulus whose fields are drawn
from ranges (axes). Today the declaration is a world file (YAML); the steps below are the
same for any declaration kind that `sensoryforge describe` lists. You end with a file that
validates, sample draws the user has looked at and a preview picture they have approved.

## 0. Read the working style

Run `sensoryforge agents brief` and follow it. If the command is not found, SensoryForge
is not installed here: say so and stop (it installs from source; it is not on PyPI yet).

## 1. Find out what they mean

Ask, one short question at a time, for what you cannot read off their words:

- The kinds of stimulus, one class each, and how often each should occur (weights).
- For each class, what varies and over what range. Ranges are the user's.
- Timing: how long a draw lasts, delays, holds, repeated contacts, vibration, motion.
- Kinds that must be held out of training, and the space unit if it is not millimetres.
- Whether a world file already exists. If it does, read it and change it; do not rewrite
  it. Any changed value gives the world a new id.

## 2. Map their words to the world's kinds

`sensoryforge describe --section world` lists the shapes, patterns, modulations,
distributions and class kinds that are registered; `sensoryforge describe --json --section
world` gives each one's parameters with their ranges and units; `sensoryforge describe
--schema world` is the file's JSON Schema. Use those names, not names from memory. The
page that teaches the format, with examples, is
https://benefron.github.io/sensoryforge/user_guide/worlds/.

Choose by meaning and tell the user what you chose and why ("a ridge is a bar shape with a
gaussian profile"). If no registered kind fits, say so and follow the plan skill's "When a
component is missing" (`sensoryforge-design-and-plan`).

## 3. Ranges: never invent them

A value comes from the user, from a source they approve, or from reasoning they approve.
If they do not know a range, use `sensoryforge-ground-assumptions`: they describe it, or you
find sources for them to approve. Or propose one, write it in the file with a comment
`# assumption: needs your OK`, and list it when you report back.
A range outside what the shape allows is refused when the world loads, so the validator
also keeps you honest, but it cannot tell you what is meant.

## 4. Write the file

Start small: the classes the user named, the axes they gave, defaults for the rest of the
timing. Add elements only when they ask. Never overwrite an existing file without saying
so first.

## 5. Validate

```
sensoryforge world validate world.yml
sensoryforge validate --json --kind world world.yml
```

The first prints the world's id and each class with the axes it draws. The second lists
problems as "path", "message" and "fix", and exits 1 when there are any. Fix them and
run it again until it is clean. If a fix would change what the user meant (a range, a
class), ask first; do not bend the world to the validator.

## 6. Sample, then preview

Print a few draws: `sensoryforge world sample world.yml -n 5 --seed 0` (one JSON record
each: class, every drawn value, a timeline). Show the user a short table of class and the
values that matter and ask whether they look right.

Then render a picture. The canvas is the sensor's geometry: ask for its rows, columns and
spacing, or read them from a recipe with `sensoryforge describe --json --section recipes`.
Write the script below, set `ROWS, COLS, SPACING` to the sensor's, and run
`python preview.py world.yml preview.png DURATION_MS N`. It draws one row per draw and six
evenly spaced times across the columns, with one colour scale so rows are comparable. Show
the image (give its full path if you cannot display it) and ask: is this what you meant?
Change the world and render again until they say yes.

## 7. Change it in words

The user answers in their own words: "slower", "softer", "more quiet between touches".
Take each phrase in turn.

1. **Map it to fields by meaning**, with the glossary and `sensoryforge describe --json
   --section world`, and say the mapping in a line ("slower: longer touch and hold times;
   softer: a lower amplitude; quieter between touches: a longer pause"). If two readings
   fit, ask which.
2. **The size of the change is theirs.** A number they gave is used as it is (origin: you).
   With only a direction, ask by how much, or propose a value marked
   `# assumption: needs your OK` and show it as a proposal. Never use a "typical" value from
   memory; a value that needs a source goes to `sensoryforge-ground-assumptions`.
3. **Change only the fields named** and keep the rest. Save the new version as its own file
   (`world_v2.yml`) unless they say to overwrite.
4. **Show what changed** as a table of field, old, new and origin. Validate, sample with
   the same seed as before so the draws can be compared, render the preview again and ask:
   is this what you meant?

Repeat until they say it is right.

## 8. Stop and report

Give the file path, the world id, the classes with their weights, every value marked as an
assumption, and the picture. A value still marked as an assumption is a decision still
open: ask about it, whatever the pace. Then offer the next step, `sensoryforge-build-dataset`,
at the pace the brief sets (ask first, or say what comes next and carry on). Do not build or
run anything the user has not approved.

## Verify

The world validates with no problems; the sampled draws lie inside the ranges the user
gave; every value you changed has an origin; the user has seen a preview of the final file.

## Worked example

The user wants dots and edges of different sizes, one touch each. A first world:

```yaml
# file: world.yml
world:
  name: demo
  defaults:
    delay_ms:   {range: [0, 10]}
    touch_ms:   {range: [5, 15]}
    hold_ms:    {range: [20, 40]}
    release_ms: {range: [5, 15]}
    amplitude:  {range: [0.5, 1.0]}
    x_mm: {range: [-0.3, 0.3]}
    y_mm: {range: [-0.3, 0.3]}
  classes:
    dots:
      weight: 0.6
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.15, 0.45]}}
    edges:
      weight: 0.4
      layer: {shape: {kind: bar, profile: gaussian, length_mm: 0}}
      axes:
        width_mm: {range: [0.05, 0.15]}
        orientation_deg: {range: [0, 180], circular: true}
```

```bash
sensoryforge world validate world.yml
# prints: class dots
sensoryforge validate --json --kind world world.yml
# prints: "valid": true
sensoryforge world sample world.yml -n 2 --seed 7
# prints: "class": "dots"
```

A range that the shape refuses is reported with its path, and the command exits 1:

```yaml
# file: broken.yml
world:
  name: broken
  classes:
    dots:
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.0, 0.45]}}
```

```bash
sensoryforge validate --json --kind world broken.yml
# exit: 1
# prints: world.classes.dots.axes.sigma_mm
```

The preview script:

```python
# file: preview.py
"""A contact sheet of a world: one row per draw, six times across the columns."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sensoryforge.world import Canvas, load_world, render_movie, sample

# The canvas is the sensor's: grid rows, columns and spacing in the world's space unit.
ROWS, COLS, SPACING = 40, 40, 0.15
world_file, out = sys.argv[1], sys.argv[2]
duration_ms, n = float(sys.argv[3]), int(sys.argv[4])

canvas = Canvas.from_grid(ROWS, COLS, SPACING)
draws = sample(load_world(world_file), n=n, seed=0)
movies = [render_movie(d, canvas, dt_ms=1.0, duration_ms=duration_ms) for d in draws]
top = max(float(m.max()) for m in movies) or 1.0  # one colour scale for every panel
times = [round(k * (int(duration_ms) - 1) / 5) for k in range(6)]

fig, axes = plt.subplots(n, 6, figsize=(12, 2 * n), squeeze=False)
for r, (draw, movie) in enumerate(zip(draws, movies)):
    for c, t in enumerate(times):
        ax = axes[r][c]
        ax.imshow(movie[t].T, origin="lower", vmin=0.0, vmax=top)
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title(f"t = {t} ms")
        if c == 0:
            ax.set_ylabel(draw.class_name)
fig.savefig(out, dpi=80)
print(f"{n} draws, written to {out}")
```

```bash
python preview.py world.yml preview.png 80 4
# prints: written to preview.png
```

The user then says "make the dots slower". Slower reads as longer touch and hold times; no
number was given, so the agent asks how much. The answer, "hold 40 to 80 ms", is theirs:

```yaml
# file: world_v2.yml
world:
  name: demo
  defaults:
    delay_ms:   {range: [0, 10]}
    touch_ms:   {range: [5, 15]}
    hold_ms:    {range: [40, 80]}   # you: "hold 40 to 80 ms"
    release_ms: {range: [5, 15]}
    amplitude:  {range: [0.5, 1.0]}
    x_mm: {range: [-0.3, 0.3]}
    y_mm: {range: [-0.3, 0.3]}
  classes:
    dots:
      weight: 0.6
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.15, 0.45]}}
    edges:
      weight: 0.4
      layer: {shape: {kind: bar, profile: gaussian, length_mm: 0}}
      axes:
        width_mm: {range: [0.05, 0.15]}
        orientation_deg: {range: [0, 180], circular: true}
```

```bash
sensoryforge validate --json --kind world world_v2.yml
# prints: "valid": true
sensoryforge world sample world_v2.yml -n 2 --seed 7
# prints: "class": "dots"
```
