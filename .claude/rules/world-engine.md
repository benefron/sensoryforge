---
paths:
  - "sensoryforge/world/**"
  - "sensoryforge/stimuli/layered.py"
  - "sensoryforge/stimuli/episode.py"
  - "sensoryforge/io/bundle.py"
---

# The world engine is a contract with pressure-simulation — read before editing

- **pressure-simulation pins this code by sha** and runs `tests/contract/test_world_contract.py`
  against it (`docs/reference/world_contract.md`). A change that alters any draw, frame,
  seed, entry id or bundle field breaks its reproducibility: say so in the commit, bump the
  format tag (`sensoryforge-world/1`, `sensoryforge-dataset/1`, bundle `SCHEMA_VERSION`), and
  update the contract page.
- **Never change `world/rng.py`'s constants or hashing**: every draw of every world changes.
- **`layered` and the world renderer are two implementations of one language.** A change to
  one needs the same change in the other; `tests/unit/test_world_render.py::
  test_world_render_equals_layered` and `tests/unit/test_layered_golden.py` must keep passing
  (the golden test means old layered stimuli render as before, R10).
- New shapes, patterns, modulations, distributions and class kinds are registered
  (`register_*`), not added as special cases in the renderer.
