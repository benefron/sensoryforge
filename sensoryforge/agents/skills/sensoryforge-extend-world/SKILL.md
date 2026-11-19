---
name: sensoryforge-extend-world
description: Add a stimulus shape, pattern, modulation, distribution or class kind to the world engine as a plugin that passes the world contract check; a class kind is how a world for another modality is added. Use when someone asks whether the worlds have a stimulus shape, texture or vibration, says they cannot draw what they need, wants a new shape, texture, vibration or sampling rule for generated stimuli, or wants worlds and data sets for a sense SensoryForge has no world for (light, sound, anything with its own frames), in whatever words they put it.
---

# Extend the world engine

A world draws stimuli: each draw is one class with a value for every axis, rendered as
frames. Everything it uses is looked up by name in a registry, so you add a piece by
registering it from a small plugin; nothing inside SensoryForge is edited. You end with
a plugin that passes the world contract check, a world file or dict that uses it, and
frames the user has looked at. A new component of the sensing pipeline (a neuron, a
filter, a grid) is `sensoryforge-extend-pipeline`, not this skill.

## 0. Read the working style

Run `sensoryforge agents brief` and follow it. If the command is not found, SensoryForge
is not installed here: say so and stop (it installs from source; it is not on PyPI yet).

## 1. Which piece, and is it already there?

Run `sensoryforge describe --section world` first (the world's registries are not the
pipeline's `list-components`). Ask what the stimulus does, then choose by meaning:

- **A shape** is the spatial profile of one contact, a function of position.
- **A pattern** says where copies of a shape sit (an array, a texture, a row of dots).
- **A modulation** changes a contact's strength over time (a vibration, a pulse train).
- **A distribution** is how an axis draws its values from a range (log-uniform, a table).
- **A class kind** is a whole kind of stimulus with its own axes, timeline and renderer. A
  world for another modality is one: the tactile contact kinds are two of them.

`sensoryforge describe --section world` lists what is registered and `sensoryforge describe
--json --section world` gives each one's parameters with units and ranges. A built-in with
other parameters, or a combination of two (a pattern of a shape, a modulation on it), may
already do it: say so and stop if the user is content. A change of range is a world file
edit, not a plugin (`sensoryforge-design-world`).

## 2. Get the physics from the user

The formula, the parameters, their units and their valid ranges are the user's to give.
Use `sensoryforge-ground-assumptions` for anything missing and list each value with where
it came from. A parameter's range is the range a world may draw: it is a claim about the
stimulus, so ask. Write down before coding: the inputs with units, the output (a value in
the stimulus's units, or a field of frames), and one case the user can check by hand.

## 3. Write the plugin

There is no scaffold for these kinds: write one module with module-level functions and a
`register()` function (below). Define every function at module level: a function built
anew on each call is a different object, and a second registration of it is refused. The
same registration run again is harmless, so an entry point and a config may both load it.
Registered names are matched without regard to case; a different thing under a taken
name is refused.

### A shape

`register_shape(name, fn, specs)`. `fn(x, y, params)` takes two coordinate tensors in mm
and returns a tensor of their shape and dtype. the specs are a list of `ParamSpec`, and one
of them must be called amplitude. Each spec's lowest and highest allowed values are the
values a world may draw, and its default lies between them. Vectorise over the batch: a draw alone must
equal the same draw inside a batch, bit for bit, so no reduction over the batch and no
loop whose length depends on a draw. `unbounded=True` marks a shape that fills the canvas;
`indenter=True` one that takes the contact depth from the renderer (then it has no
parameter called depth).

### A pattern and a modulation

`register_pattern(name, fn, specs)`: `fn(params)` returns `(positions, scales)`, two lists
of equal length, positions `(x, y)` in mm. It is a pure function of its parameters (the
renderer caches by them). `register_modulation(name, fn, specs)`: `fn(tc, params)` takes
the time since the touch in ms and returns values in `[0, 1]` of the same shape.

### A distribution

`register_distribution(name, sample, support)`: `sample(u, axis)` takes the uniform
numbers in `[0, 1)` (including 0 and the largest float below 1) and returns one plain
number per input, each depending on its own input only and the same on every call. Give
`support(axis)` when the values are a finite list, so a test split can balance them, and
a bounds function when you know the range. Options of the axis come from the axis entry.

### A class kind

Subclass `ClassKind`, give it a name (a class attribute), and fill in what it leaves open. The smallest
readable model is the built-in quiet kind; the methods are:

- **`normalise_layer(layer, where)`** returns the class's layer with defaults filled, or
  `None` when the kind has no layer; a kind that takes none refuses one with `ValueError`.
- **`builtin_defaults(layer)`** is `{axis: constant}` for every axis the kind binds: the value
  a world gets when it sets none. Each is a value the user must give or approve; if they
  have not, use the neutral one (zero) and say it is your placeholder.
  **`resolve(name, layer)`** maps an axis name to a field `(group, name)` and raises
  `UnknownField` for any other. **`domain()`** and **`field_info()`** give each field's
  valid range, which a world is checked against when it loads.
- **`end_ms(values)`** and **`timeline(values)`**: when a draw ends, and its phases as
  `[name, start_ms, end_ms]`, in order, inside that end, zero-length phases left out.
- **`render_group(spec, draws, X, Y, times)`** returns frames `[draws, K, *S]` for a group of draws at
  times `[draws, K]` ms since each draw's start. Read each draw's value from `draw.values`. The
  frames are exactly 0 before time 0 and from the end on, and one draw alone equals the same draw
  in a group, so vectorise over the draws.
- **The modality is the world's** `modality:` word, which a data set records. The core
  stays neutral: units and axis names are yours, and the frame's meaning is yours to state
  in the plugin's README. The tactile words belong to the tactile kinds.

Call `register_class_kind()` with an instance of it: an instance. A kind that changes how existing
worlds or data sets draw is a different matter: the tactile draws are pinned by another
project, so never register under, or `replace=` over, a name that exists.

## 4. Check it

Register, then check each piece by name with `check_world_component(kind, name)` from
`sensoryforge.testing.world_contracts`, where the kind is shape, pattern, modulation,
distribution or class_kind. A distribution that needs options, and a class kind, take
`example=`: a world mapping, as `World.from_dict` takes it, with a class of this kind. The
check calls your function as the renderer does and raises `AssertionError` naming the
piece and the rule; report that `check_world_component()` passed, by that name; fix the piece, not the check. Then test the user's own case from
step 2, which the check cannot know. Pytest is not installed with SensoryForge: call the
check from a few lines of Python, as the example does.

## 5. Use it

Package the module (a folder with `pyproject.toml` and a `sensoryforge.components` entry
point to the register function, as `sensoryforge new-component` writes for the other
kinds) and install it into the same environment; the world commands then see the new
names. Or load it in Python: `load_plugin_import_paths(["module:register"])`. Write a
world that uses it, run `sensoryforge world validate` on the file, and `sensoryforge world
sample FILE -n 3 --seed 7` to read the draws. Render a few and show the user the frames:
they decide whether it is the stimulus they meant.

## Verify

The contract check passes for every piece you registered, the world loads and draws, the
user has seen frames and agreed, and no formula, range or unit in the plugin came from you
unconfirmed.

## Worked example

A plugin with a ring shape and a class kind for a light flash (a uniform field at a level
for a time), checked, loaded twice and drawn from:

```python
# file: demo_world.py
"""A world plugin: a ring shape and a class kind for another modality (a light flash)."""
import torch

from sensoryforge.stimuli.base import ParamSpec
from sensoryforge.world import register_class_kind, register_shape
from sensoryforge.world.kinds import ClassKind, FieldInfo, UnknownField

_FIELDS = {"onset_ms": 0.0, "flash_ms": 0.0, "level": 1.0}


def ring(x, y, p):
    r = torch.sqrt(x**2 + y**2)
    return torch.exp(-((r - p["radius_mm"]) ** 2) / (2.0 * p["width_mm"] ** 2))


class FlashKind(ClassKind):
    """A uniform field at the level from the onset for the flash time, then nothing."""

    name = "flash"

    def normalise_layer(self, layer, where):
        if layer:
            raise ValueError(f"{where}: a flash class has no layer")
        return None

    def builtin_defaults(self, layer):
        return dict(_FIELDS)

    def resolve(self, name, layer):
        if name in _FIELDS:
            return ("flash", name)
        raise UnknownField(f"{name!r}: a flash class has the axes {sorted(_FIELDS)}")

    def domain(self, ref, layer):
        return (0.0, None)

    def field_info(self, ref, layer):
        return FieldInfo("number", 0.0, None)

    def end_ms(self, values):
        return float(values["onset_ms"]) + float(values["flash_ms"])

    def timeline(self, values):
        start, end = float(values["onset_ms"]), self.end_ms(values)
        phases = [["quiet", 0.0, start], ["flash", start, end]]
        return [p for p in phases if p[2] > p[1]]

    def render_group(self, spec, draws, X, Y, times):
        def column(name):
            vals = [float(d.values[name]) for d in draws]
            return torch.tensor(vals, dtype=X.dtype, device=X.device).view(-1, 1)

        start = column("onset_ms")
        on = (times >= start) & (times < start + column("flash_ms"))
        frames = (column("level") * on).to(X.dtype)  # [g, K]
        shape = frames.shape + (1,) * X.ndim
        return frames.reshape(shape).expand(frames.shape + tuple(X.shape)).contiguous()


def register():
    amp = ParamSpec("amplitude", dtype="float", default=1.0, min_val=0.0, max_val=1.0e4)
    radius = ParamSpec(
        "radius_mm", dtype="float", default=1.0, min_val=0.0, max_val=10.0, unit="mm"
    )
    width = ParamSpec(
        "width_mm", dtype="float", default=0.1, min_val=0.001, max_val=10.0, unit="mm"
    )
    register_shape("ring", ring, [amp, radius, width])
    register_class_kind(FlashKind())
```

```python
# file: use_world.py
"""Register a world plugin, check it, and draw from a world that uses it."""
import torch

from sensoryforge.plugins import load_plugin_import_paths
from sensoryforge.testing.world_contracts import check_world_component
from sensoryforge.world import Canvas, World, render, sample

print("loaded", len(load_plugin_import_paths(["demo_world:register"])))
print("loaded again", len(load_plugin_import_paths(["demo_world:register"])))

spec = {
    "world": {
        "name": "flashes",
        "modality": "light",
        "defaults": {"onset_ms": {"range": [0, 10]}, "level": {"range": [0.5, 1.0]}},
        "classes": {
            "flash": {"kind": "flash", "weight": 1, "axes": {"flash_ms": {"range": [5, 20]}}}
        },
    }
}
check_world_component("shape", "ring")
check_world_component("class_kind", "flash", example=spec)
print("contract ok")

draws = sample(World.from_dict(spec), n=6, seed=7)
frames = render(draws, Canvas.from_grid(8, 8, 0.2), torch.tensor([0.0, 10.0, 40.0]))
print("frames", tuple(frames.shape), "lit", bool(frames.amax() > 0))
```

```bash
python use_world.py
# prints: loaded again 1
# prints: contract ok
# prints: frames (6, 3, 8, 8) lit True
```
