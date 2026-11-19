# sensoryforge-ring-shape

A reference SensoryForge world plugin: a small, tested, installable package to copy from.
It registers the `ring` shape with the world engine: the profile
`exp(-(r - radius_mm)^2 / (2 * width_mm^2))` around a contact's centre, 1 on the circle
of radius `radius_mm`. The engine multiplies by the contact's `amplitude` itself, so the
function does not read it. A teaching formula, not a model of any real stimulus.

There is no scaffold for world pieces, so it is written by hand, as
`sensoryforge-extend-world` describes: module-level `ring` and `RING_SPECS`, and a
`register()` that calls `register_shape` (a second run is harmless).

## Install and check

```bash
pip install -e .
sensoryforge world validate ../ring_world.yml
pip install pytest && pytest      # check_world_component("shape", "ring") and a hand-worked case
```

A world uses it with `layer: {shape: {kind: ring}}` and the axes `radius_mm` and
`width_mm`: see `../ring_world.yml`.

Code: `sensoryforge_ring_shape/shape.py`. Contract:
`sensoryforge.testing.world_contracts.check_world_component`.
