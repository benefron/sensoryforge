# World Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Declared stimulus worlds in SensoryForge — a schema, a deterministic counter-based sampler, a vectorised renderer at arbitrary times, data sets with seeded splits, strata and probes, and `sensoryforge batch --dataset` — delivered as tag `v1.1.0` with a contract document for pressure-simulation.

**Architecture:** A new `sensoryforge/world/` package (one job per module: `rng`, `distributions`, `kernel`, `kinds`, `schema`, `sampling`, `render`, `dataset`, `runner`). A world class is a `layered` layer whose fields are drawn from axes; the world renderer re-implements the layer semantics as one broadcast tensor computation and is pinned equal to `layered` by tests. `layered` gains additive, default-off fields (slide, contacts, pause, modulation, braille `dots`, `signed`); contact timing and modulation math live in a shared `stimuli/episode.py`. The old sweep `BatchExecutor` is untouched; `batch --dataset` routes to the new runner.

**Tech Stack:** Python ≥ 3.10, torch ≥ 2.2, numpy, PyYAML, h5py (all already dependencies). pytest.

**Spec:** `docs/development/specs/2026-10-02-world-engine-design.md` (read it with this plan; section numbers below, e.g. "spec §5.5", refer to it).

## Global Constraints

- Work only in the worktree `/Users/benefron/sensoryforge/.claude/worktrees/world-engine` on branch `worktree-world-engine`. Never switch the branch of, or commit in, `/Users/benefron/sensoryforge` (pressure-simulation runs it as an editable install and another session simulates with it).
- **Never `pip install -e .` from the worktree** (F-053). Tests put the worktree on `sys.path` through `tests/conftest.py`; subprocesses get `PYTHONPATH=<worktree>`.
- Run tests with `conda run -n sensoryforge python -m pytest …` from the worktree root.
- Code must run on Python 3.10 + torch 2.2.2 + numpy 1.26 (pressure-simulation's `bio-encoding` env) and on SF's env (Python 3.11, torch 2.5.1). No new dependencies.
- Units: space mm, time ms; coordinates `(x, y)`; meshgrids `indexing="ij"` (dim 0 is x); batch dimension first.
- Input validation raises `ValueError` (never `assert`); catch specific exceptions, never bare `except Exception`.
- Google-style docstrings with tensor shapes and units on public functions; `black` formatting; `flake8` clean.
- Every existing behaviour stays bit-identical by default (R10): registered stimuli, `layered` without the new fields, `run --design`, the old `batch` sweep path.
- Golden comparisons use `sensoryforge.testing.golden.assert_matches_golden` (F-071), never `torch.equal` against a fixture.
- Every commit: Conventional Commits subject; a ledger trailer in its last paragraph (`Decision:`/`Finding:`/`Fixed:`/`Refs:` … or `Ledger: none — <reason, 3+ words>`); single-line trailers; ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The post-commit hook syncs the ledger by itself.
- Test file basenames must be unique across `tests/` (there are no `__init__.py` files).

## Review Focus

1. **A class whose `random` pattern seed is an axis** — positions differ per draw, so the per-pattern position cache must key on the seed; expected: renders correctly and equals `layered`. Pinned in Task 6 (`test_random_pattern_seed_axis_matches_layered`).
2. **A draw longer than its data-set entry** (`twice` in the test world reaches 215 ms in a 120 ms entry) — expected: the manifest says `truncated: true` and the bundle simply stops at `duration_ms`. Pinned in Task 9 (`test_long_draws_are_marked_truncated`).
3. **float64 on an MPS device** — expected: a clear `ValueError` from `render`, and the batch runner renders on CPU before moving to MPS. Pinned in Task 6 (`test_float64_on_mps_is_refused`).
4. **Re-running a batch into the same output** — without `--resume` bundles are replaced; with `--resume` finished ones are skipped; a `.partial` left by a crash is cleaned. Pinned in Task 10 (`test_rerun_replaces_resume_skips_and_partials_are_cleaned`).
5. **A world file edited after a data set pinned its `world_id`** — expected: building fails, naming both ids. Pinned in Task 9 (`test_world_pin_mismatch_fails`).

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `sensoryforge/stimuli/episode.py` | create | contact envelope, slide progress, sine/pulse modulation (pure torch, shared) |
| `sensoryforge/stimuli/layered.py` | modify | new fields (`slide_ms`, `contacts`, `pause_ms`, span `slide`, `MODULATIONS`, braille `dots`, `signed`), fallback to world registries |
| `sensoryforge/world/__init__.py` | create | public API |
| `sensoryforge/world/rng.py` | create | splitmix64 counter hashing |
| `sensoryforge/world/distributions.py` | create | `AxisSpec`, distributions registry, strata, probes |
| `sensoryforge/world/kernel.py` | create | vectorised shapes, pattern batches, motion offsets; registries |
| `sensoryforge/world/kinds.py` | create | class kinds (`layered`, `quiet`): binding, timeline, `to_layer`, `render_group` |
| `sensoryforge/world/schema.py` | create | `World`, `ClassSpec`, `load_world`, world id |
| `sensoryforge/world/sampling.py` | create | `Draw`, `sample`, `fixed_draw`, `Session`, `session` |
| `sensoryforge/world/render.py` | create | `Canvas`, `render`, `render_movie` |
| `sensoryforge/world/dataset.py` | create | `DatasetSpec`, `Entry`, `build_dataset`, `write_dataset` |
| `sensoryforge/world/runner.py` | create | `run_dataset`, `read_batch_index`, task slicing |
| `sensoryforge/provenance.py` | create | `source_info()` (SF's sha) |
| `sensoryforge/io/bundle.py` | modify | schema 2.2.0, sha, world-entry payload |
| `sensoryforge/cli.py` | modify | `dataset build`, `world validate|sample`, `batch --dataset` |
| `tests/fixtures/worlds/tactile_small.yml` | create | the test world |
| `tests/fixtures/worlds/dataset_small.yml` | create | the test data set |
| `tests/fixtures/make_layered_golden.py`, `tests/fixtures/layered_golden.pt` | create | pre-change `layered` renders |
| `benchmarks/world_render.py` | create | the 4.1 M-triple timing |
| docs (Task 12) | create/modify | user guide, contract, CLAUDE.md, DECISIONS.md, changelog |

---

### Task 1: `layered` extensions — episode math, slide, contacts, modulation, braille cells, signed

**Files:**
- Create: `tests/fixtures/make_layered_golden.py`, `tests/fixtures/layered_golden.pt`, `sensoryforge/stimuli/episode.py`, `tests/unit/test_layered_golden.py`, `tests/unit/test_layered_episode.py`
- Modify: `sensoryforge/stimuli/layered.py`

**Interfaces:**
- Produces (`sensoryforge/stimuli/episode.py`):
  - `contact_terms(t, onset, up, hold, slide, down, contacts, pause) -> (env, tau, k, local)` — all tensors broadcasting against `t` (ms); `env` in [0, 1]; `tau` ms since the current contact's touch; `k` contact index (float); `local = t - onset`.
  - `span_progress(tau, k, local, contacts, start, length) -> progress` in [0, 1], spread over all contacts.
  - `sine_modulation(tc, frequency_hz, depth, phase_deg)`, `pulse_modulation(tc, rate_hz, duty, edge_ms, depth)` — tensors in [0, 1].
- Produces (`sensoryforge/stimuli/layered.py`): `MODULATIONS: Dict[str, List[ParamSpec]]` (`none`, `sine`, `pulses`); `modulate_sine(tc, params)`, `modulate_pulses(tc, params)` (params: dict of tensors); `layer_modulation(modulation, timing, time_ms, run_ms) -> Optional[Tensor[T]]`; timing fields `slide_ms`, `contacts`, `pause_ms`; motion span `slide`; braille `dots`; grating/gabor `signed`.

- [ ] **Step 1: Record the pre-change renders (golden)**

Create `tests/fixtures/make_layered_golden.py`:

```python
"""Write tests/fixtures/layered_golden.pt: layered renders from before the world engine.

Run once, before changing sensoryforge/stimuli/layered.py:

    conda run -n sensoryforge python tests/fixtures/make_layered_golden.py

tests/unit/test_layered_golden.py re-renders STACKS and requires the same frames,
so every layered stimulus written before the new fields existed renders as it did.
"""

from __future__ import annotations

from pathlib import Path

import torch

from sensoryforge.stimuli.layered import default_layer, render_layers
from sensoryforge.stimuli.presets import PRESETS, preset

N = 17
XS = torch.linspace(-2.0, 2.0, N)
XX, YY = torch.meshgrid(XS, XS, indexing="ij")
TOTAL_MS = 60.0
EVERY = 3  # keep every third frame: the fixture stays small


def _layer(shape, pattern=None, motion=None, timing=None):
    layer = default_layer(shape["kind"])
    layer["shape"].update(shape)
    if pattern is not None:
        layer["pattern"] = pattern
    if motion is not None:
        layer["motion"] = motion
    if timing is not None:
        layer["timing"] = timing
    return layer


_T = {"onset_ms": 5.0, "ramp_up_ms": 10.0, "hold_ms": 20.0, "ramp_down_ms": 10.0}
_OPEN = {"onset_ms": 0.0, "ramp_up_ms": 8.0, "hold_ms": None, "ramp_down_ms": 8.0}

STACKS = {
    "gaussian": ([_layer({"kind": "gaussian", "sigma_mm": 0.4}, timing=_T)], "sum"),
    "disc_soft": ([_layer({"kind": "disc", "diameter_mm": 1.2}, timing=_T)], "sum"),
    "disc_hard": (
        [_layer({"kind": "disc", "diameter_mm": 1.2, "edge_mm": 0.0}, timing=_T)],
        "sum",
    ),
    "bar_finite": (
        [_layer({"kind": "bar", "length_mm": 1.5, "orientation_deg": 30.0}, timing=_T)],
        "sum",
    ),
    "bar_flat": ([_layer({"kind": "bar", "profile": "flat"}, timing=_T)], "sum"),
    "grating_sine": (
        [_layer({"kind": "grating", "wavelength_mm": 0.8, "phase_deg": 40.0}, timing=_T)],
        "sum",
    ),
    "grating_square": (
        [_layer({"kind": "grating", "profile": "square", "duty": 0.3}, timing=_T)],
        "sum",
    ),
    "gabor": ([_layer({"kind": "gabor", "orientation_deg": 60.0}, timing=_T)], "sum"),
    "grid_mask": (
        [
            _layer(
                {"kind": "gaussian", "sigma_mm": 0.2},
                pattern={"kind": "grid", "rows": 2, "cols": 3, "spacing_mm": 0.8,
                         "mask": "101 011"},
                timing=_T,
            )
        ],
        "sum",
    ),
    "list": (
        [
            _layer(
                {"kind": "gaussian", "sigma_mm": 0.2},
                pattern={"kind": "list", "positions": [[0.5, 0.5], [-0.5, 0.2]],
                         "amplitudes": [1.0, 0.5]},
                timing=_T,
            )
        ],
        "sum",
    ),
    "random": (
        [
            _layer(
                {"kind": "gaussian", "sigma_mm": 0.15},
                pattern={"kind": "random", "count": 6, "width_mm": 3.0,
                         "height_mm": 3.0, "seed": 3, "amplitude_jitter": 0.3},
                timing=_T,
            )
        ],
        "sum",
    ),
    "braille_text": (
        [
            _layer(
                {"kind": "gaussian", "sigma_mm": 0.15},
                pattern={"kind": "braille", "text": "hi", "dot_spacing_mm": 0.4,
                         "cell_spacing_mm": 1.2, "x_mm": -0.6},
                timing=_T,
            )
        ],
        "sum",
    ),
    "linear_hold": (
        [
            _layer(
                {"kind": "gaussian", "sigma_mm": 0.3},
                motion={"kind": "linear", "start": [-1.0, 0.0], "end": [1.0, 0.5]},
                timing=_T,
            )
        ],
        "sum",
    ),
    "linear_all": (
        [
            _layer(
                {"kind": "gaussian", "sigma_mm": 0.3},
                motion={"kind": "linear", "start": [0.0, -1.0], "end": [0.0, 1.0],
                        "span": "all"},
                timing=_T,
            )
        ],
        "sum",
    ),
    "circular": (
        [
            _layer(
                {"kind": "gaussian", "sigma_mm": 0.3},
                motion={"kind": "circular", "radius_mm": 0.8, "revolutions": 0.75},
                timing=_T,
            )
        ],
        "sum",
    ),
    "path": (
        [
            _layer(
                {"kind": "gaussian", "sigma_mm": 0.3},
                motion={"kind": "path", "waypoints": [[-1, -1], [1, -1], [1, 1]]},
                timing=_T,
            )
        ],
        "sum",
    ),
    "open_hold": ([_layer({"kind": "gaussian", "sigma_mm": 0.5}, timing=_OPEN)], "sum"),
    "stack_max": (
        [
            _layer({"kind": "gaussian", "sigma_mm": 0.5}, timing=_T),
            _layer({"kind": "disc", "diameter_mm": 0.8}, timing=_OPEN),
        ],
        "max",
    ),
}


def render_all():
    """``{name: frames [T/EVERY, N, N]}`` for every stack and every preset."""
    out = {}
    for name, (layers, combine) in STACKS.items():
        frames = render_layers(
            layers, XX, YY, dt_ms=1.0, total_ms=TOTAL_MS, combine=combine
        )
        out[name] = frames[::EVERY].clone()
    for name in sorted(PRESETS):
        chosen = preset(name, TOTAL_MS)
        frames = render_layers(
            chosen["layers"], XX, YY, dt_ms=1.0, total_ms=TOTAL_MS,
            combine=chosen["combine"],
        )
        out[f"preset_{name}"] = frames[::EVERY].clone()
    return out


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "layered_golden.pt"
    torch.save(render_all(), target)
    print(f"wrote {target}")
```

Run (before any change to `layered.py`):

```bash
conda run -n sensoryforge python tests/fixtures/make_layered_golden.py
```

Expected: `wrote …/tests/fixtures/layered_golden.pt` (a few hundred kB).

- [ ] **Step 2: Write the golden test**

Create `tests/unit/test_layered_golden.py`:

```python
"""Every layered stimulus written before the world engine renders as it did (R10)."""

import importlib.util
from pathlib import Path

import pytest
import torch

from sensoryforge.testing.golden import assert_matches_golden

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_layered_golden", FIXTURES / "make_layered_golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GOLDEN = torch.load(FIXTURES / "layered_golden.pt")
FRESH = _generator().render_all()


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_layered_renders_as_before(name):
    assert_matches_golden(FRESH[name], GOLDEN[name], what=f"layered {name}")
```

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_layered_golden.py -q`
Expected: all pass (nothing has changed yet). Commit:

```bash
git add tests/fixtures/make_layered_golden.py tests/fixtures/layered_golden.pt tests/unit/test_layered_golden.py
git commit -m "test(layered): golden renders from before the world engine

Ledger: none — test fixture recording current behaviour

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Write the failing tests for the new fields**

Create `tests/unit/test_layered_episode.py`:

```python
"""Layered stimuli: slides, several contacts, modulation, braille cells, signed carriers."""

import pytest
import torch

from sensoryforge.stimuli.episode import contact_terms, pulse_modulation, span_progress
from sensoryforge.stimuli.layered import (
    MODULATIONS,
    TIMING_SPECS,
    default_layer,
    pattern_positions,
    render_layers,
)

H = 81
XS = torch.linspace(-8.0, 8.0, H)
XX, YY = torch.meshgrid(XS, XS, indexing="ij")


def _layer(shape, timing, motion=None, modulation=None, pattern=None):
    layer = default_layer(shape["kind"])
    layer["shape"].update(shape)
    layer["timing"] = timing
    if motion is not None:
        layer["motion"] = motion
    if modulation is not None:
        layer["modulation"] = modulation
    if pattern is not None:
        layer["pattern"] = pattern
    return layer


def _centroid_x(frame):
    return float((frame * XX).sum() / frame.sum())


def _centre(frame):
    return float(frame[40, 40])


def test_new_timing_fields_default_to_the_old_behaviour():
    names = {s.name: s.default for s in TIMING_SPECS}
    assert names["slide_ms"] == 0.0
    assert names["contacts"] == 1
    assert names["pause_ms"] == 0.0
    assert set(MODULATIONS) == {"none", "sine", "pulses"}


def test_slide_moves_only_after_the_hold():
    layer = _layer(
        {"kind": "gaussian", "sigma_mm": 0.3},
        {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": 20, "slide_ms": 20,
         "ramp_down_ms": 0},
        motion={"kind": "linear", "start": [0, 0], "end": [2, 0], "span": "slide"},
    )
    frames = render_layers([layer], XX, YY, dt_ms=1.0, total_ms=50.0)
    assert _centroid_x(frames[10]) == pytest.approx(0.0, abs=1e-4)
    assert _centroid_x(frames[30]) == pytest.approx(1.0, abs=1e-3)
    assert _centroid_x(frames[39]) == pytest.approx(1.9, abs=1e-3)
    assert float(frames[40].abs().max()) == 0.0  # contact over


def test_contacts_pause_and_retouch_where_the_last_ended():
    layer = _layer(
        {"kind": "gaussian", "sigma_mm": 0.3},
        {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": 10, "slide_ms": 10,
         "ramp_down_ms": 0, "contacts": 2, "pause_ms": 10},
        motion={"kind": "linear", "start": [0, 0], "end": [2, 0], "span": "slide"},
    )
    frames = render_layers([layer], XX, YY, dt_ms=1.0, total_ms=60.0)
    assert float(frames[20:30].abs().max()) == 0.0  # the pause is exactly zero
    assert _centroid_x(frames[19]) == pytest.approx(0.95, abs=1e-3)
    assert _centroid_x(frames[30]) == pytest.approx(1.0, abs=1e-3)  # re-touch
    assert _centroid_x(frames[49]) == pytest.approx(1.95, abs=1e-3)
    assert float(frames[50:].abs().max()) == 0.0


def test_contacts_need_an_explicit_hold():
    layer = _layer(
        {"kind": "gaussian"},
        {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": None, "ramp_down_ms": 0,
         "contacts": 2},
    )
    with pytest.raises(ValueError, match="contacts > 1 needs an explicit hold_ms"):
        render_layers([layer], XX, YY, dt_ms=1.0, total_ms=50.0)


def test_sine_modulation_swings_by_its_depth():
    layer = _layer(
        {"kind": "gaussian", "sigma_mm": 0.5},
        {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": 100, "ramp_down_ms": 0},
        modulation={"kind": "sine", "frequency_hz": 50.0, "depth": 0.5},
    )
    frames = render_layers([layer], XX, YY, dt_ms=1.0, total_ms=100.0)
    assert _centre(frames[0]) == pytest.approx(1.0, abs=1e-6)
    assert _centre(frames[10]) == pytest.approx(0.5, abs=1e-6)  # half a period
    assert _centre(frames[20]) == pytest.approx(1.0, abs=1e-5)


def test_pulses_tap_at_their_rate():
    layer = _layer(
        {"kind": "gaussian", "sigma_mm": 0.5},
        {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": 100, "ramp_down_ms": 0},
        modulation={"kind": "pulses", "rate_hz": 50.0, "duty": 0.5},
    )
    frames = render_layers([layer], XX, YY, dt_ms=1.0, total_ms=100.0)
    centre = torch.tensor([_centre(f) for f in frames])
    expected = ((torch.arange(100) % 20) < 10).float()
    assert torch.equal(centre, expected)


def test_pulse_edges_ramp_and_are_clamped_to_the_period():
    tc = torch.arange(0.0, 20.0, 1.0, dtype=torch.float64)
    one = torch.tensor(1.0, dtype=torch.float64)
    m = pulse_modulation(tc, 50.0 * one, 0.5 * one, 4.0 * one, one)
    assert m[0] == 0.0 and m[2] == pytest.approx(0.5) and m[4] == 1.0
    assert m[10] == 1.0 and m[12] == pytest.approx(0.5) and m[14] == 0.0
    wide = pulse_modulation(tc, 50.0 * one, 0.5 * one, 100.0 * one, one)
    assert float(wide.max()) <= 1.0 and float(wide.min()) >= 0.0


def test_contact_terms_and_progress_for_a_batch_of_parameters():
    t = torch.arange(0.0, 60.0, dtype=torch.float64).unsqueeze(0).expand(2, -1)
    col = lambda a, b: torch.tensor([[a], [b]], dtype=torch.float64)  # noqa: E731
    env, tau, k, local = contact_terms(
        t, col(0, 5), col(0, 0), col(10, 10), col(10, 0), col(0, 0), col(2, 1),
        col(10, 0),
    )
    assert float(env[0, 25]) == 0.0 and float(env[0, 30]) == 1.0
    assert float(env[1, 4]) == 0.0 and float(env[1, 5]) == 1.0
    progress = span_progress(tau, k, local, col(2, 1), col(10, 10), col(10, 0))
    assert float(progress[0, 15]) == pytest.approx(0.25)
    assert float(progress[0, 59]) == 1.0
    assert float(progress[1].abs().max()) == 0.0  # no slide, no motion


def test_braille_dots_give_the_same_cell_as_its_letter():
    by_text = pattern_positions({"kind": "braille", "text": "h"})
    by_dots = pattern_positions({"kind": "braille", "text": "z", "dots": "125"})
    assert by_dots == by_text


def test_braille_dots_reject_invalid_cells():
    with pytest.raises(ValueError, match="dot numbers 1-6"):
        pattern_positions({"kind": "braille", "dots": "127"})


def test_signed_grating_has_negative_lobes():
    timing = {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": None, "ramp_down_ms": 0}
    raised = render_layers(
        [_layer({"kind": "grating", "wavelength_mm": 1.0}, timing)],
        XX, YY, dt_ms=1.0, total_ms=2.0,
    )
    signed = render_layers(
        [_layer({"kind": "grating", "wavelength_mm": 1.0, "signed": True}, timing)],
        XX, YY, dt_ms=1.0, total_ms=2.0,
    )
    assert float(raised.min()) >= 0.0
    assert float(signed.min()) < -0.99 and float(signed.max()) > 0.99
    assert torch.allclose(signed, 2.0 * raised - 1.0, atol=1e-6)
```

- [ ] **Step 4: Run them to see them fail**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_layered_episode.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sensoryforge.stimuli.episode'`.

- [ ] **Step 5: Create `sensoryforge/stimuli/episode.py`**

```python
"""Contact-episode timing and modulation, shared by layered stimuli and worlds.

One *contact* is a ramp up (``up``), a still ``hold``, a moving ``slide`` and
a ramp down (``down``). An episode is ``contacts`` of them after ``onset``,
``pause`` apart. Every function is pure torch arithmetic on tensors that
broadcast against the time tensor ``t`` (ms), so the same code serves one
layer (0-d parameters) and a batch of world draws (``[g, 1]`` parameters).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def _safe(x: torch.Tensor) -> torch.Tensor:
    """``x`` where positive, else 1: a divisor that never divides by zero."""
    return torch.where(x > 0, x, torch.ones_like(x))


def contact_terms(
    t: torch.Tensor,
    onset: torch.Tensor,
    up: torch.Tensor,
    hold: torch.Tensor,
    slide: torch.Tensor,
    down: torch.Tensor,
    contacts: torch.Tensor,
    pause: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The contact envelope and clock at times ``t``.

    Args:
        t: Times in ms, any shape.
        onset: Quiet lead-in before the first touch, ms.
        up: Ramp up (touch), ms.
        hold: Still contact at full amplitude, ms.
        slide: Moving contact at full amplitude, ms.
        down: Ramp down (release), ms.
        contacts: Number of contacts (a float tensor holding an integer).
        pause: Lift between contacts, ms.

    Returns:
        ``(envelope, tau, k, local)``, each broadcast to ``t``: the envelope
        in ``[0, 1]`` (exactly 0 before ``onset``, in pauses and after the
        last contact); ``tau``, ms since the current contact's touch began;
        ``k``, the contact's index; ``local = t - onset``.
    """
    local = t - onset
    cycle = up + hold + slide + down
    period = cycle + pause
    k = torch.floor(local / _safe(period))
    tau = local - k * period
    plateau_end = up + hold + slide
    env = torch.zeros_like(tau)
    env = torch.where((tau >= 0) & (tau < up), tau / _safe(up), env)
    env = torch.where((tau >= up) & (tau < plateau_end), torch.ones_like(env), env)
    env = torch.where(
        (tau >= plateau_end) & (tau < cycle),
        1.0 - (tau - plateau_end) / _safe(down),
        env,
    )
    active = (local >= 0) & (k < contacts)
    env = torch.where(active, env, torch.zeros_like(env))
    return env.clamp(0.0, 1.0), tau, k, local


def span_progress(
    tau: torch.Tensor,
    k: torch.Tensor,
    local: torch.Tensor,
    contacts: torch.Tensor,
    start: torch.Tensor,
    length: torch.Tensor,
) -> torch.Tensor:
    """Motion progress in ``[0, 1]``, spread over all contacts.

    Within each contact the motion runs over ``[start, start + length)`` of
    the contact's clock; contact ``k`` covers ``[k, k + 1] / contacts`` of the
    path, so a re-touch lands where the previous contact ended.

    Args:
        tau: Ms since the current contact's touch (from :func:`contact_terms`).
        k: Contact index (from :func:`contact_terms`).
        local: ``t - onset`` (from :func:`contact_terms`).
        contacts: Number of contacts.
        start: Start of the moving span within a contact, ms.
        length: Length of the moving span, ms; 0 means no motion.

    Returns:
        Progress, broadcast to ``tau``: 0 before the first contact, 1 after the
        last, 0 everywhere when ``length`` is 0.
    """
    within = ((tau - start) / _safe(length)).clamp(0.0, 1.0)
    progress = ((k + within) / contacts).clamp(0.0, 1.0)
    progress = torch.where(local < 0, torch.zeros_like(progress), progress)
    return torch.where(length > 0, progress, torch.zeros_like(progress))


def sine_modulation(
    tc: torch.Tensor,
    frequency_hz: torch.Tensor,
    depth: torch.Tensor,
    phase_deg: torch.Tensor,
) -> torch.Tensor:
    """Vibration: ``1 - depth * (1 - cos(2 pi f tc + phase)) / 2``, in ``[0, 1]``.

    Args:
        tc: Ms since the contact's touch.
        frequency_hz: Vibration frequency, Hz.
        depth: 0 (none) to 1 (from zero to peak).
        phase_deg: Phase at the touch, degrees (0: at peak).

    Returns:
        The modulation factor, broadcast to ``tc``.
    """
    angle = 2.0 * math.pi * frequency_hz * tc / 1000.0 + phase_deg * (math.pi / 180.0)
    return 1.0 - depth * (1.0 - torch.cos(angle)) / 2.0


def pulse_modulation(
    tc: torch.Tensor,
    rate_hz: torch.Tensor,
    duty: torch.Tensor,
    edge_ms: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    """Repeated indentation: a train of taps, ``1 - depth * (1 - pulse)``.

    Each period ``P = 1000 / rate_hz`` ms the pulse rises linearly over
    ``edge`` from 0, holds 1 until ``duty * P``, falls linearly over ``edge``,
    then stays 0. ``edge`` is clamped to ``min(edge_ms, duty P, (1 - duty) P)``.

    Args:
        tc: Ms since the contact's touch.
        rate_hz: Taps per second.
        duty: Fraction of each period pressed, in ``(0, 1)``.
        edge_ms: Rise and fall time of each tap, ms (0: a step).
        depth: 0 (no taps) to 1 (lift fully between taps).

    Returns:
        The modulation factor in ``[0, 1]``, broadcast to ``tc``.
    """
    period = 1000.0 / rate_hz
    on = duty * period
    edge = torch.minimum(torch.minimum(edge_ms, on), period - on)
    phase = torch.remainder(tc, period)
    rise = torch.where(edge > 0, phase / _safe(edge), torch.ones_like(phase))
    fall = torch.where(edge > 0, 1.0 - (phase - on) / _safe(edge), torch.zeros_like(phase))
    pulse = torch.where(
        phase < on,
        rise.clamp(max=1.0),
        torch.where(phase < on + edge, fall, torch.zeros_like(phase)),
    )
    return 1.0 - depth * (1.0 - pulse.clamp(0.0, 1.0))
```

- [ ] **Step 6: Extend `sensoryforge/stimuli/layered.py`**

Make these edits (keep every existing line not mentioned):

1. Imports: add `from sensoryforge.stimuli.episode import contact_terms, pulse_modulation, sine_modulation, span_progress`.
2. `_label`: add `"_hz"` to the stripped suffixes: `for suffix in ("_mm", "_ms", "_deg", "_hz"):`.
3. `_i` gains `advanced=False` (pass `advanced=advanced` to `ParamSpec`). Add a bool helper after `_v`:

```python
def _b(name, default, help_="", advanced=False):
    """A boolean switch."""
    return ParamSpec(
        name,
        label=_label(name),
        dtype="bool",
        default=default,
        help=help_,
        tooltip=help_,
        advanced=advanced,
    )
```

4. After `_ORIENTATION`, add:

```python
_SIGNED = _b(
    "signed",
    False,
    "Zero-mean carrier with negative lobes (cos, or +/-1 for square) "
    "instead of the non-negative raised cosine.",
    advanced=True,
)
```

and append `_SIGNED` to the spec lists of `SHAPES["grating"]` and `SHAPES["gabor"]`.

5. `PATTERNS["braille"]`: append
   `_s("dots", "", "Cells by dot number, e.g. '125 14'; when set it replaces text.")`.
6. `_SPAN` becomes `_c("span", "hold", ["hold", "all", "slide"], "Move during the hold, from onset to the end of the ramp down, or during the slide only.")`.
7. `TIMING_SPECS`: insert after `hold_ms`
   `_f("slide_ms", 0.0, 0.0, 1.0e7, "ms", "Moving time after the hold (motion span 'slide').", advanced=True),`
   and append after `ramp_down_ms`
   `_i("contacts", 1, 1, 10000, "Touches: ramp up, hold, slide, ramp down repeat pause_ms apart.", advanced=True),`
   `_f("pause_ms", 0.0, 0.0, 1.0e7, "ms", "Lift between contacts.", advanced=True),`.
8. After `TIMING_SPECS`, add the modulations:

```python
#: Modulation kind -> its parameters (temporal frequency on the contact).
MODULATIONS: Dict[str, List[ParamSpec]] = {
    "none": [],
    "sine": [
        _f("frequency_hz", 10.0, 0.001, 1.0e4, "Hz", "Vibration frequency."),
        _f("depth", 1.0, 0.0, 1.0, "", "0 = none, 1 = from zero to peak."),
        _f("phase_deg", 0.0, -360.0, 360.0, "deg", "Phase at each touch (0 = peak)."),
    ],
    "pulses": [
        _f("rate_hz", 5.0, 0.001, 1.0e4, "Hz", "Taps per second."),
        _f("duty", 0.5, 0.01, 0.99, "", "Fraction of each period pressed."),
        _f("edge_ms", 0.0, 0.0, 1.0e4, "ms", "Rise and fall of each tap (0 = step)."),
        _f("depth", 1.0, 0.0, 1.0, "", "0 = no taps, 1 = lift fully between taps."),
    ],
}


def modulate_sine(tc: torch.Tensor, p: Dict[str, Any]) -> torch.Tensor:
    """``sine`` modulation; ``p`` holds tensors (see :data:`MODULATIONS`)."""
    return sine_modulation(tc, p["frequency_hz"], p["depth"], p["phase_deg"])


def modulate_pulses(tc: torch.Tensor, p: Dict[str, Any]) -> torch.Tensor:
    """``pulses`` modulation; ``p`` holds tensors (see :data:`MODULATIONS`)."""
    return pulse_modulation(tc, p["rate_hz"], p["duty"], p["edge_ms"], p["depth"])


_MODULATION_FUNCTIONS: Dict[str, Callable] = {
    "sine": modulate_sine,
    "pulses": modulate_pulses,
}
```

9. Replace `_stripes` with the signed-aware version (the unsigned branches are the old code):

```python
def _stripes(across, p):
    wavelength = float(p["wavelength_mm"])
    phase = torch.remainder(
        2.0 * math.pi * across / wavelength
        + math.radians(float(p.get("phase_deg", 0.0))),
        2.0 * math.pi,
    )
    signed = bool(p.get("signed", False))
    if p.get("profile", "sine") == "square":
        duty = float(p.get("duty", 0.5))
        # On for the part of each period centred on phase 0.
        centred = torch.minimum(phase, 2.0 * math.pi - phase)
        on = (centred <= math.pi * duty).to(across.dtype)
        return 2.0 * on - 1.0 if signed else on
    return torch.cos(phase) if signed else raised_cosine(phase)
```

10. Replace `_braille_positions` with:

```python
def _braille_cells(p) -> List[Tuple[int, str]]:
    """``(cell index, dot numbers)`` per non-blank cell: ``dots`` if set, else ``text``."""
    dots = str(p.get("dots") or "").strip()
    if dots:
        cells = []
        for index, cell in enumerate(dots.split()):
            if any(ch not in "123456" for ch in cell) or len(set(cell)) != len(cell):
                raise ValueError(
                    f"braille pattern: cell {cell!r} must be distinct dot numbers "
                    "1-6, e.g. '125'"
                )
            cells.append((index, cell))
        return cells
    cells = []
    for index, letter in enumerate(str(p.get("text", "")).lower()):
        if letter == " ":
            continue
        if letter not in _BRAILLE:
            raise ValueError(f"braille pattern: no cell for {letter!r} (a-z only)")
        cells.append((index, _BRAILLE[letter]))
    return cells


def _braille_positions(p) -> Tuple[List[Tuple[float, float]], List[float]]:
    pitch = float(p["dot_spacing_mm"])
    step = float(p["cell_spacing_mm"])
    x0, y0 = float(p.get("x_mm", 0.0)), float(p.get("y_mm", 0.0))
    positions = []
    for index, cell in _braille_cells(p):
        cell_x = x0 + index * step
        for dot in cell:
            n = int(dot) - 1
            col, row = divmod(n, 3)  # 1-3 left column, 4-6 right; top to bottom
            positions.append((cell_x + (col - 0.5) * pitch, y0 + (1 - row) * pitch))
    return positions, [1.0] * len(positions)
```

11. Add the general timing helpers above `layer_envelope`:

```python
def _timing_values(timing, run_ms: float):
    """``(onset, up, hold, slide, down, contacts, pause)`` with defaults filled."""
    t = {**defaults(TIMING_SPECS), **(timing or {})}
    onset = float(t["onset_ms"] or 0.0)
    up = float(t["ramp_up_ms"] or 0.0)
    down = float(t["ramp_down_ms"] or 0.0)
    slide = float(t.get("slide_ms") or 0.0)
    contacts = int(t.get("contacts") or 1)
    pause = float(t.get("pause_ms") or 0.0)
    if contacts < 1:
        raise ValueError(f"timing: contacts must be >= 1, got {contacts}")
    hold = t["hold_ms"]
    if hold is None:
        if contacts > 1:
            raise ValueError("timing: contacts > 1 needs an explicit hold_ms")
        hold = max(run_ms - onset - up - slide - down, 0.0)
    return onset, up, float(hold), slide, down, contacts, pause


def _is_single_contact(timing) -> bool:
    """True for timing the pre-world code handled (no slide, one contact)."""
    t = timing or {}
    return float(t.get("slide_ms") or 0.0) == 0.0 and int(t.get("contacts") or 1) == 1


def _contact_clock(timing, time_ms: torch.Tensor, run_ms: float):
    """:func:`~sensoryforge.stimuli.episode.contact_terms` for one layer's timing."""
    values = _timing_values(timing, run_ms)
    as_t = [torch.tensor(float(v), dtype=time_ms.dtype, device=time_ms.device) for v in values]
    onset, up, hold, slide, down, contacts, pause = as_t
    return contact_terms(time_ms, onset, up, hold, slide, down, contacts, pause), values
```

12. `layer_envelope`: insert at the top of the function body (before `t = {...}`):

```python
    if not _is_single_contact(timing):
        (env, _, _, _), _ = _contact_clock(timing, time_ms, run_ms)
        return env
```

13. `_motion_fraction`: insert at the top of the function body:

```python
    if span not in ("hold", "all", "slide"):
        raise ValueError(f"motion span must be hold, all or slide, got {span!r}")
    if span == "slide" or not _is_single_contact(timing):
        (env, tau, k, local), values = _contact_clock(timing, time_ms, run_ms)
        onset, up, hold, slide, down, contacts, pause = values
        start, length = {
            "hold": (up, hold),
            "slide": (up + hold, slide),
            "all": (0.0, up + hold + slide + down),
        }[span]

        def as_t(v):
            return torch.tensor(float(v), dtype=time_ms.dtype, device=time_ms.device)

        return span_progress(tau, k, local, as_t(contacts), as_t(start), as_t(length))
```

14. Add `layer_modulation` after `motion_offsets`:

```python
def layer_modulation(
    modulation, timing, time_ms: torch.Tensor, run_ms: float
) -> Optional[torch.Tensor]:
    """The layer's modulation over time ``[T]`` in ``[0, 1]``, or ``None``.

    Measured from each contact's touch (see :mod:`sensoryforge.stimuli.episode`).
    """
    modulation = modulation or {"kind": "none"}
    kind = modulation.get("kind", "none")
    if kind == "none":
        return None
    if kind in _MODULATION_FUNCTIONS:
        specs, fn = MODULATIONS[kind], _MODULATION_FUNCTIONS[kind]
    else:
        kernel = _world_kernel()
        if kind not in kernel.MODULATION_KINDS or kind in MODULATIONS:
            raise ValueError(
                f"unknown modulation kind {kind!r}; known: "
                f"{sorted(set(MODULATIONS) | set(kernel.MODULATION_KINDS))}"
            )
        specs, fn = kernel.MODULATION_KINDS[kind].specs, kernel.MODULATION_KINDS[kind].fn
    params = {**defaults(specs), **modulation}
    (_, tau, _, _), _ = _contact_clock(timing, time_ms, run_ms)
    tensors = {
        name: torch.tensor(float(value), dtype=time_ms.dtype, device=time_ms.device)
        for name, value in params.items()
        if name != "kind" and _is_number(value)
    }
    return fn(tau, tensors)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _world_kernel():
    """The world kernel's registries (imported lazily: the kernel imports this module)."""
    from sensoryforge.world import kernel

    return kernel
```

(`_world_kernel` is used again in Task 3; define it here once. `layer_modulation`'s
fallback branch only runs once Task 3 registers kinds; until then it raises the
"unknown modulation kind" error for anything but the built-ins.)

15. `render_layer`: after `envelope = layer_envelope(...)`, add

```python
    modulation = layer_modulation(
        layer.get("modulation"), layer.get("timing"), time_ms, run_ms
    )
    if modulation is not None:
        envelope = envelope * modulation
```

16. Module docstring: add a paragraph after the four choices:

```
A layer may also carry a **modulation** -- ``none``, ``sine`` (vibration) or
``pulses`` (repeated indentation) -- that multiplies its envelope, and its
timing may add a ``slide_ms`` (motion span ``slide``), several ``contacts``
and the ``pause_ms`` between them. All default off.
```

- [ ] **Step 7: Run the new tests and the golden test**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_layered_episode.py tests/unit/test_layered_golden.py tests/unit/test_layered_stimuli.py -q`
Expected: all pass. The golden test proves old layers render unchanged.

- [ ] **Step 8: Run the GUI and stimulus suites (new timing fields reach the generated forms)**

Run: `conda run -n sensoryforge python -m pytest tests/gui_v2 tests/unit -q -x -k "layer or stimulus or preset"`
Expected: all pass. If a GUI test pins the exact list of timing fields, update it to include `slide_ms`, `contacts`, `pause_ms` (they are `advanced=True`, so the Basic view is unchanged) and say so in the commit body.

- [ ] **Step 9: Commit**

```bash
git add sensoryforge/stimuli/episode.py sensoryforge/stimuli/layered.py tests/unit/test_layered_episode.py
git commit -m "feat(stimuli): layered slides, contacts, modulation, braille cells and signed carriers

Additive layered fields, all default off (golden test: old layers render
unchanged): timing slide_ms with motion span slide, contacts and pause_ms
(motion progress spread over all contacts, so a re-touch lands where the
last contact ended); a modulation part (sine vibration, pulses for
repeated indentation) measured from each touch; braille dots by number;
signed grating/gabor carriers. Contact timing and modulation math live in
stimuli/episode.py so the world renderer uses the same code.

Refs: D-99d39b6

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Counter-based randomness and axes

**Files:**
- Create: `sensoryforge/world/__init__.py` (empty docstring module for now), `sensoryforge/world/rng.py`, `sensoryforge/world/distributions.py`, `tests/unit/test_world_rng.py`, `tests/unit/test_world_axes.py`

**Interfaces:**
- Produces (`rng`): `SEED_LIMIT = 2**53`; `key(part) -> int`; `mix(a, b) -> np.ndarray[uint64]`; `hash_parts(*parts) -> int`; `seed53(*parts) -> int`; `draw_seeds(seed: int, indices) -> np.ndarray[uint64]` (values < 2**53); `uniforms(seeds, slot) -> np.ndarray[float64]` in [0, 1); `permutation(n, *parts) -> np.ndarray[int64]`; `_splitmix64(x)`.
- Produces (`distributions`): `AxisSpec` (frozen dataclass: `name, form, value, lo, hi, dist, values, weights, circular, probes, options, domain`) with `from_dict(name, spec)`, `to_dict()`, `is_random`, `sample(u) -> list`, `support() -> Optional[list]`, `bin_edges(bins) -> list[float]`, `stratum_values(labels, bins, u) -> list`, `bin_label(b, bins) -> str`, `probe_values(side, bins, u) -> Optional[list]`, `midpoint()`, `contains(v) -> bool`, `with_domain(lo, hi)`; `register_distribution(name, sample, support=None)`; `DISTRIBUTIONS`; `BRAILLE_CELLS` (63 strings); `plain(value)`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_world_rng.py`:

```python
"""Counter-based hashing: reference values, independence from n, ranges."""

import hashlib

import numpy as np

from sensoryforge.world import rng


def test_splitmix64_matches_the_reference_sequence():
    # SplitMix64 seeded with 0: its first two outputs.
    first = rng._splitmix64(np.array([0], dtype=np.uint64))[0]
    second = rng._splitmix64(np.array([0x9E3779B97F4A7C15], dtype=np.uint64))[0]
    assert int(first) == 0xE220A8397B1DCDAF
    assert int(second) == 0x6E789E6AA1B965F4


def test_hashes_are_pinned():
    # Changing any of these changes every draw of every world, and with it
    # pressure-simulation's data sets (.claude/rules/world-engine.md). Values
    # computed with SF's env and bio-encoding before implementation: identical.
    assert rng.hash_parts(7, "class") == 18279675696851888426
    assert rng.seed53(20261002, "train", 0) == 7454758737064703
    seeds = rng.draw_seeds(7, [0, 1, 2])
    assert seeds.tolist() == [7125699848674262, 6996912338094675, 4294997795970717]
    assert rng.uniforms(seeds, "class").tolist() == [
        0.14982635123855137,
        0.23457615854478897,
        0.2137307826344046,
    ]


def test_string_keys_are_sha256_prefixes():
    digest = hashlib.sha256(b"sigma_mm").digest()[:8]
    assert rng.key("sigma_mm") == int.from_bytes(digest, "little")
    assert rng.key(-1) == 2**64 - 1


def test_draw_seeds_do_not_depend_on_how_many_are_asked_for():
    many = rng.draw_seeds(7, np.arange(1000))
    some = rng.draw_seeds(7, [3, 999])
    assert many[3] == some[0] and many[999] == some[1]
    assert int(many.max()) < rng.SEED_LIMIT
    assert len(set(many.tolist())) == 1000


def test_uniforms_are_in_the_unit_interval_and_differ_by_slot():
    seeds = rng.draw_seeds(1, np.arange(20000))
    a = rng.uniforms(seeds, "a")
    b = rng.uniforms(seeds, "b")
    assert a.dtype == np.float64 and 0.0 <= a.min() and a.max() < 1.0
    assert abs(a.mean() - 0.5) < 0.01
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.03
    assert np.array_equal(a, rng.uniforms(seeds, "a"))


def test_seed53_and_permutation():
    assert rng.seed53(1, "train", 0) != rng.seed53(1, "train", 1)
    assert rng.seed53(1, "train", 0) == rng.seed53(1, "train", 0)
    perm = rng.permutation(50, 9, "x")
    assert sorted(perm.tolist()) == list(range(50))
    assert perm.tolist() != list(range(50))
    assert np.array_equal(perm, rng.permutation(50, 9, "x"))
```

`tests/unit/test_world_axes.py`:

```python
"""Axes: forms, sampling, strata, probes, errors."""

import numpy as np
import pytest

from sensoryforge.world import rng
from sensoryforge.world.distributions import BRAILLE_CELLS, AxisSpec

U = rng.uniforms(rng.draw_seeds(3, np.arange(20000)), "u")


def test_constant_axis():
    axis = AxisSpec.from_dict("contacts", {"value": 2})
    assert not axis.is_random and axis.sample(U[:3]) == [2, 2, 2]
    assert axis.to_dict() == {"value": 2}


def test_uniform_and_log_uniform_stay_in_range():
    lin = AxisSpec.from_dict("a", {"range": [0.15, 0.45]})
    log = AxisSpec.from_dict("b", {"range": [0.01, 0.1], "dist": "log_uniform"})
    v = np.array(lin.sample(U))
    w = np.array(log.sample(U))
    assert v.min() >= 0.15 and v.max() < 0.45
    assert w.min() >= 0.01 and w.max() <= 0.1
    assert abs(np.median(np.log(w)) - np.log(np.sqrt(0.001))) < 0.05


def test_int_axis_reaches_both_ends():
    axis = AxisSpec.from_dict("contacts", {"range": [1, 3], "int": True})
    values = axis.sample(U)
    assert set(values) == {1, 2, 3} and all(isinstance(v, int) for v in values)
    assert axis.support() == [1, 2, 3]


def test_categorical_follows_its_weights():
    axis = AxisSpec.from_dict("letter", {"values": ["a", "b"], "weights": [3, 1]})
    values = axis.sample(U)
    assert abs(values.count("a") / len(values) - 0.75) < 0.02


def test_braille_cells_are_the_63_non_empty_cells():
    assert len(BRAILLE_CELLS) == 63 == len(set(BRAILLE_CELLS))
    axis = AxisSpec.from_dict("dots", {"dist": "braille_cells"})
    values = axis.sample(U)
    assert set(values) == set(BRAILLE_CELLS)
    assert axis.support() == BRAILLE_CELLS


def test_strata_cover_equal_bins_on_the_sampling_scale():
    log = AxisSpec.from_dict("s", {"range": [0.01, 1.0], "dist": "log_uniform"})
    edges = log.bin_edges(2)
    assert edges == pytest.approx([0.01, 0.1, 1.0])
    values = log.stratum_values(np.array([0, 1]), 2, np.array([0.5, 0.5]))
    assert values[0] < 0.1 < values[1]
    assert log.bin_label(0, 2) == "[0.01, 0.1)" and log.bin_label(1, 2) == "[0.1, 1]"


def test_probes_lie_strictly_outside_and_inside_the_domain():
    axis = AxisSpec.from_dict("sigma_mm", {"range": [0.15, 0.45]}).with_domain(0.001, 50.0)
    below = np.array(axis.probe_values("below", 5, U))
    above = np.array(axis.probe_values("above", 5, U))
    assert below.max() < 0.15 and below.min() >= 0.15 - 0.06 - 1e-12
    assert above.min() > 0.45 and above.max() <= 0.45 + 0.06 + 1e-12
    worst = axis.probe_values("below", 5, np.array([1.0 - 2.0**-53]))
    assert worst[0] < 0.15


def test_probes_skip_a_side_with_no_room():
    axis = AxisSpec.from_dict("hold_ms", {"range": [0, 100]}).with_domain(0.0, None)
    assert axis.probe_values("below", 5, U[:3]) is None
    assert axis.probe_values("above", 5, U[:3]) is not None
    circular = AxisSpec.from_dict("d", {"range": [0, 360], "circular": True})
    assert circular.probe_values("above", 5, U[:3]) is None


def test_midpoint_and_contains():
    log = AxisSpec.from_dict("s", {"range": [0.01, 1.0], "dist": "log_uniform"})
    assert log.midpoint() == pytest.approx(0.1)
    assert log.contains(0.5) and not log.contains(2.0)
    cat = AxisSpec.from_dict("c", {"values": ["x", "y"]})
    assert cat.midpoint() == "x" and not cat.contains("z")


@pytest.mark.parametrize(
    "spec, message",
    [
        ({"range": [2, 1]}, "lo > hi"),
        ({"range": [0, 1], "dist": "log_uniform"}, "needs lo > 0"),
        ({"range": [0, 1], "dist": "normal"}, "must be one of"),
        ({"values": []}, "is empty"),
        ({"dist": "nope"}, "unknown distribution"),
        ({"rang": [0, 1]}, "unknown keys"),
        ({"value": 1, "range": [0, 1]}, "takes only 'value'"),
        ({"range": [0.5, 2], "int": True}, "integer bounds"),
    ],
)
def test_invalid_axes_are_named(spec, message):
    with pytest.raises(ValueError, match=message):
        AxisSpec.from_dict("x", spec)
```

- [ ] **Step 2: Run to see them fail**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_world_rng.py tests/unit/test_world_axes.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sensoryforge.world'`.

- [ ] **Step 3: Implement**

`sensoryforge/world/__init__.py` (Task 6 fills in the exports):

```python
"""Declared stimulus worlds: sample them, render them, build data sets on them."""
```

`sensoryforge/world/rng.py`:

```python
"""Counter-based randomness for worlds (spec §4.1).

Every random choice in a world is a pure function of integers: a 64-bit
splitmix64 mix of a seed, a draw's index and a *slot* naming the choice.
Nothing is drawn from a stream, so draw ``i`` depends only on ``(seed, i)``
-- never on how many draws were asked for -- and the bits are the same on
every machine (integer arithmetic plus one exact scaling to ``[0, 1)``).
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Union

import numpy as np

#: Seeds are kept below 2**53 so they survive JSON in any language.
SEED_LIMIT = 2**53

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_SECOND = np.uint64(0xD1B54A32D192ED03)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_MASK64 = (1 << 64) - 1

Part = Union[int, str, np.integer]


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """One splitmix64 output per element of ``x`` (uint64, wrapping)."""
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))


def key(part: Part) -> int:
    """A 64-bit integer for one part of a key: ints mod 2**64, strings by SHA-256."""
    if isinstance(part, str):
        return int.from_bytes(hashlib.sha256(part.encode("utf-8")).digest()[:8], "little")
    if isinstance(part, bool):
        raise TypeError("a hash key part must be an int or a str, not a bool")
    return int(part) & _MASK64


def mix(a, b) -> np.ndarray:
    """Hash two uint64 values (or arrays of them, broadcasting) into one."""
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return _splitmix64(_splitmix64(a) ^ _splitmix64(b + _SECOND))


def hash_parts(*parts: Part) -> int:
    """A 64-bit hash of a sequence of ints and strings."""
    if not parts:
        raise ValueError("hash_parts needs at least one part")
    h = np.asarray(key(parts[0]), dtype=np.uint64)
    for part in parts[1:]:
        h = mix(h, key(part))
    return int(h)


def seed53(*parts: Part) -> int:
    """A seed below 2**53 derived from ``parts`` (the hash's top 53 bits)."""
    return hash_parts(*parts) >> 11


def draw_seeds(seed: int, indices: Iterable[int]) -> np.ndarray:
    """Per-draw seeds ``H(seed, i)`` (top 53 bits) for each index ``i``."""
    idx = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices)
    idx = idx.astype(np.int64, copy=False)
    if idx.size and int(idx.min()) < 0:
        raise ValueError("draw indices must be >= 0")
    return mix(np.uint64(key(seed)), idx.astype(np.uint64)) >> np.uint64(11)


def uniforms(seeds, slot: Part) -> np.ndarray:
    """``u = H(seed, slot)`` mapped exactly to ``[0, 1)``, one per seed (float64)."""
    bits = mix(np.asarray(seeds, dtype=np.uint64), key(slot)) >> np.uint64(11)
    return bits.astype(np.float64) * (1.0 / SEED_LIMIT)


def permutation(n: int, *parts: Part) -> np.ndarray:
    """A permutation of ``range(n)`` determined by ``parts`` (stable argsort)."""
    u = uniforms(draw_seeds(seed53(*parts), np.arange(n)), "permutation")
    return np.argsort(u, kind="stable")
```

`sensoryforge/world/distributions.py`:

```python
"""Axes of a world: how one parameter is sampled, stratified and probed.

See spec §3.2 (forms), §4.2 (from ``u`` to values) and §6.2 (strata, probes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

FORMS = ("constant", "numeric", "int", "categorical", "registered")
NUMERIC_DISTS = ("uniform", "log_uniform")
_AXIS_KEYS = {"value", "range", "dist", "int", "values", "weights", "circular", "probes"}


def plain(value: Any) -> Any:
    """A JSON-plain scalar (str, int, float, bool or None), else ``ValueError``."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    raise ValueError(f"axis values must be numbers or strings, got {value!r}")


def _braille_cells() -> List[str]:
    return [
        "".join(combo)
        for size in range(1, 7)
        for combo in combinations("123456", size)
    ]


#: The 63 non-empty six-dot braille cells as dot-number strings ("1" ... "123456").
BRAILLE_CELLS: List[str] = _braille_cells()


@dataclass(frozen=True)
class Distribution:
    """A registered distribution: values from ``u``, and its finite support if any."""

    sample: Callable[[np.ndarray, "AxisSpec"], List[Any]]
    support: Optional[Callable[["AxisSpec"], List[Any]]] = None


DISTRIBUTIONS: Dict[str, Distribution] = {}


def register_distribution(
    name: str,
    sample: Callable[[np.ndarray, "AxisSpec"], List[Any]],
    support: Optional[Callable[["AxisSpec"], List[Any]]] = None,
) -> None:
    """Register a distribution usable as ``{dist: <name>}`` on an axis.

    Args:
        name: The name axes use.
        sample: ``sample(u, axis) -> values``, ``u`` a float64 array in ``[0, 1)``.
        support: ``support(axis) -> values`` for a finite distribution (needed
            to stratify it), else ``None``.
    """
    if name in NUMERIC_DISTS:
        raise ValueError(f"{name!r} is a built-in numeric distribution")
    DISTRIBUTIONS[name] = Distribution(sample=sample, support=support)


def _sample_braille_cells(u: np.ndarray, axis: "AxisSpec") -> List[str]:
    idx = np.minimum((u * len(BRAILLE_CELLS)).astype(np.int64), len(BRAILLE_CELLS) - 1)
    return [BRAILLE_CELLS[i] for i in idx]


register_distribution("braille_cells", _sample_braille_cells, lambda axis: list(BRAILLE_CELLS))


@dataclass(frozen=True)
class AxisSpec:
    """One axis of a world class (spec §3.2).

    Attributes:
        name: The axis name as declared (bare, or dotted like ``shape.width_mm``).
        form: ``constant``, ``numeric``, ``int``, ``categorical`` or ``registered``.
        value: The constant (``constant`` only).
        lo, hi: The range (``numeric``, ``int``).
        dist: ``uniform``/``log_uniform`` (numeric) or a registered name.
        values, weights: The categories and their weights (``categorical``).
        circular: An angle: no out-of-range probes.
        probes: False to switch probes off for this axis.
        options: Extra keys passed to a registered distribution.
        domain: ``(lo, hi)`` valid values of the bound field; probes stay inside.
    """

    name: str
    form: str
    value: Any = None
    lo: Optional[float] = None
    hi: Optional[float] = None
    dist: str = "uniform"
    values: Tuple[Any, ...] = ()
    weights: Tuple[float, ...] = ()
    circular: bool = False
    probes: bool = True
    options: Tuple[Tuple[str, Any], ...] = ()
    domain: Tuple[Optional[float], Optional[float]] = (None, None)

    # ------------------------------------------------------------ parsing

    @classmethod
    def from_dict(cls, name: str, spec: Any) -> "AxisSpec":
        """Parse ``{value: v}``, ``{range: [lo, hi], ...}``, ``{values: [...]}`` or ``{dist: ...}``.

        Raises:
            ValueError: Naming the axis and what is wrong.
        """
        where = f"axis {name!r}"
        if not isinstance(spec, dict):
            raise ValueError(f"{where}: expected a mapping such as {{range: [lo, hi]}}, got {spec!r}")
        flags = {
            "circular": bool(spec.get("circular", False)),
            "probes": bool(spec.get("probes", True)),
        }
        if "value" in spec:
            if set(spec) != {"value"}:
                raise ValueError(f"{where}: a constant takes only 'value', got {sorted(spec)}")
            return cls(name=name, form="constant", value=plain(spec["value"]))
        if "dist" in spec and spec["dist"] not in NUMERIC_DISTS and "range" not in spec:
            dist = spec["dist"]
            if dist not in DISTRIBUTIONS:
                raise ValueError(f"{where}: unknown distribution {dist!r}; known: {sorted(DISTRIBUTIONS)}")
            options = tuple(
                sorted((k, plain(v)) for k, v in spec.items() if k not in {"dist", "circular", "probes"})
            )
            return cls(name=name, form="registered", dist=dist, options=options, **flags)
        unknown = set(spec) - _AXIS_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(_AXIS_KEYS)}")
        if "values" in spec:
            values = tuple(plain(v) for v in spec["values"] or [])
            if not values:
                raise ValueError(f"{where}: 'values' is empty")
            weights = tuple(float(w) for w in spec.get("weights", [1.0] * len(values)))
            if len(weights) != len(values):
                raise ValueError(f"{where}: {len(weights)} weights for {len(values)} values")
            if any(w < 0 for w in weights) or sum(weights) <= 0:
                raise ValueError(f"{where}: weights must be >= 0 with a positive sum")
            return cls(name=name, form="categorical", values=values, weights=weights, **flags)
        if "range" in spec:
            bounds = spec["range"]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"{where}: range must be [lo, hi], got {bounds!r}")
            lo, hi = float(bounds[0]), float(bounds[1])
            if lo > hi:
                raise ValueError(f"{where}: lo > hi in range {bounds!r}")
            if spec.get("int"):
                if lo != math.floor(lo) or hi != math.floor(hi):
                    raise ValueError(f"{where}: an int axis needs integer bounds, got {bounds!r}")
                return cls(name=name, form="int", lo=lo, hi=hi, **flags)
            dist = spec.get("dist", "uniform")
            if dist not in NUMERIC_DISTS:
                raise ValueError(f"{where}: dist {dist!r} for a range must be one of {NUMERIC_DISTS}")
            if dist == "log_uniform" and lo <= 0:
                raise ValueError(f"{where}: log_uniform needs lo > 0, got {lo}")
            return cls(name=name, form="numeric", lo=lo, hi=hi, dist=dist, **flags)
        raise ValueError(f"{where}: needs one of value, range, values or dist; got {sorted(spec)}")

    def to_dict(self) -> Dict[str, Any]:
        """The normalised declaration (what the world id hashes)."""
        if self.form == "constant":
            return {"value": self.value}
        if self.form == "numeric":
            out: Dict[str, Any] = {"range": [self.lo, self.hi], "dist": self.dist}
        elif self.form == "int":
            out = {"range": [int(self.lo), int(self.hi)], "int": True}
        elif self.form == "categorical":
            out = {"values": list(self.values), "weights": list(self.weights)}
        else:
            out = {"dist": self.dist, **dict(self.options)}
        if self.circular:
            out["circular"] = True
        if not self.probes:
            out["probes"] = False
        return out

    def with_domain(self, lo: Optional[float], hi: Optional[float]) -> "AxisSpec":
        """This axis with the bound field's valid range attached."""
        return replace(self, domain=(lo, hi))

    # ----------------------------------------------------------- sampling

    @property
    def is_random(self) -> bool:
        """False for a constant."""
        return self.form != "constant"

    @property
    def _log(self) -> bool:
        return self.form == "numeric" and self.dist == "log_uniform"

    def _to_scale(self, v: float) -> float:
        return math.log(v) if self._log else float(v)

    def _from_scale(self, t: np.ndarray) -> np.ndarray:
        return np.exp(t) if self._log else np.asarray(t, dtype=np.float64)

    def sample(self, u: np.ndarray) -> List[Any]:
        """One value per ``u`` (declared sampling, spec §4.2)."""
        u = np.asarray(u, dtype=np.float64)
        if self.form == "constant":
            return [self.value] * u.size
        if self.form == "numeric":
            if self._log:
                t_lo, t_hi = math.log(self.lo), math.log(self.hi)
                return np.exp(t_lo + u * (t_hi - t_lo)).tolist()
            return (self.lo + u * (self.hi - self.lo)).tolist()
        if self.form == "int":
            span = int(self.hi) - int(self.lo) + 1
            steps = np.minimum(np.floor(u * span).astype(np.int64), span - 1)
            return [int(self.lo) + int(s) for s in steps]
        if self.form == "categorical":
            w = np.asarray(self.weights, dtype=np.float64)
            cum = np.cumsum(w) / w.sum()
            idx = np.minimum(np.searchsorted(cum, u, side="right"), len(self.values) - 1)
            return [self.values[i] for i in idx]
        return DISTRIBUTIONS[self.dist].sample(u, self)

    def support(self) -> Optional[List[Any]]:
        """The finite set of values, or ``None`` for a continuous axis."""
        if self.form == "categorical":
            return list(self.values)
        if self.form == "int":
            return list(range(int(self.lo), int(self.hi) + 1))
        if self.form == "registered":
            dist = DISTRIBUTIONS[self.dist]
            return None if dist.support is None else list(dist.support(self))
        return None

    # --------------------------------------------------- strata and probes

    def _require_numeric(self) -> None:
        if self.form != "numeric":
            raise ValueError(f"axis {self.name!r} is {self.form}, not numeric")

    def bin_edges(self, bins: int) -> List[float]:
        """``bins + 1`` edges, equal on the sampling scale (log for log_uniform)."""
        self._require_numeric()
        t_lo, t_hi = self._to_scale(self.lo), self._to_scale(self.hi)
        inner = [float(self._from_scale(t_lo + k / bins * (t_hi - t_lo))) for k in range(1, bins)]
        return [self.lo, *inner, self.hi]

    def stratum_values(self, labels: np.ndarray, bins: int, u: np.ndarray) -> List[float]:
        """A value inside bin ``labels[j]`` for each ``u[j]`` (inverse CDF of ``(b + u) / bins``)."""
        self._require_numeric()
        t_lo, t_hi = self._to_scale(self.lo), self._to_scale(self.hi)
        t = t_lo + (np.asarray(labels, dtype=np.float64) + np.asarray(u)) / bins * (t_hi - t_lo)
        return np.clip(self._from_scale(t), self.lo, self.hi).tolist()

    def bin_label(self, b: int, bins: int) -> str:
        """``"[a, b)"`` (``"[a, b]"`` for the last bin), 6 significant digits."""
        edges = self.bin_edges(bins)
        close = "]" if b == bins - 1 else ")"
        return f"[{edges[b]:.6g}, {edges[b + 1]:.6g}{close}"

    def probe_values(self, side: str, bins: int, u: np.ndarray) -> Optional[List[float]]:
        """Values one bin-width below or above the range, or ``None`` if there is no room.

        Probes stay inside :attr:`domain`; circular and ``probes: false`` axes,
        and non-numeric ones, have none.
        """
        if self.form != "numeric" or self.circular or not self.probes:
            return None
        if side not in ("below", "above"):
            raise ValueError(f"probe side must be 'below' or 'above', got {side!r}")
        t_lo, t_hi = self._to_scale(self.lo), self._to_scale(self.hi)
        width = (t_hi - t_lo) / bins
        if width <= 0:
            return None
        d_lo, d_hi = self.domain
        u = np.asarray(u, dtype=np.float64)
        if side == "below":
            a, b = t_lo - width, t_lo
            if d_lo is not None and not (self._log and d_lo <= 0):
                a = max(a, self._to_scale(d_lo))
            if a >= b:
                return None
            v = self._from_scale(a + u * (b - a))
            v = np.minimum(v, np.nextafter(self.lo, -np.inf))
            if d_lo is not None:
                v = np.maximum(v, d_lo)
        else:
            a, b = t_hi, t_hi + width
            if d_hi is not None:
                b = min(b, self._to_scale(d_hi))
            if b <= a:
                return None
            v = self._from_scale(b - u * (b - a))
            v = np.maximum(v, np.nextafter(self.hi, np.inf))
            if d_hi is not None:
                v = np.minimum(v, d_hi)
        return v.tolist()

    # -------------------------------------------------------- fixed draws

    def midpoint(self) -> Any:
        """The value a fixed draw takes when it does not set this axis."""
        if self.form == "constant":
            return self.value
        if self.form == "numeric":
            t = (self._to_scale(self.lo) + self._to_scale(self.hi)) / 2.0
            return float(self._from_scale(t))
        if self.form == "int":
            return int(math.floor((self.lo + self.hi) / 2.0))
        if self.form == "categorical":
            return self.values[0]
        support = self.support()
        if support:
            return support[0]
        return DISTRIBUTIONS[self.dist].sample(np.zeros(1), self)[0]

    def contains(self, value: Any) -> bool:
        """Whether ``value`` lies in the declared range or set."""
        if self.form == "constant":
            return value == self.value
        if self.form in ("numeric", "int"):
            return isinstance(value, (int, float)) and self.lo <= value <= self.hi
        support = self.support()
        return True if support is None else value in support
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_world_rng.py tests/unit/test_world_axes.py -q`
Expected: all pass. If `test_splitmix64_matches_the_reference_sequence` fails, the hash is wrong — fix `_splitmix64`, never the expected constants.

- [ ] **Step 5: Commit**

```bash
git add sensoryforge/world/__init__.py sensoryforge/world/rng.py sensoryforge/world/distributions.py tests/unit/test_world_rng.py tests/unit/test_world_axes.py
git commit -m "feat(world): counter-based randomness and axis distributions

splitmix64 hashing of (seed, index, slot) gives every draw its values
without a stream, so draw i never depends on n. Axes: constant, uniform,
log_uniform, int, categorical and registered distributions (braille_cells);
strata on the sampling scale; probes one bin outside the range, inside the
field's domain.

Refs: D-94c08c7

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The world kernel — vectorised shapes, pattern batches, motion, registries

**Files:**
- Create: `sensoryforge/world/kernel.py`, `tests/unit/test_world_kernel.py`
- Modify: `sensoryforge/stimuli/layered.py` (fall back to the kernel registries for unknown shape and pattern kinds)

**Interfaces:**
- Consumes: `layered.SHAPES`, `layered.PATTERNS`, `layered.MODULATIONS`, `layered.pattern_positions`, `layered.modulate_sine`, `layered.modulate_pulses`, `layered._world_kernel`, `layered._is_number` (Task 1).
- Produces: `ShapeKind(fn, specs, unbounded)`, `PatternKind(fn, specs)` (+ `.placement`), `ModulationKind(fn, specs)`; `SHAPE_KINDS`, `PATTERN_KINDS`, `MODULATION_KINDS`; `register_shape(name, fn, specs, unbounded=False)`, `register_pattern(name, fn, specs)`, `register_modulation(name, fn, specs)`; `shape_specs(kind)`, `pattern_specs(kind)`, `modulation_specs(kind)` (raise `ValueError("unknown <what> kind …; known: …")`); `pattern_batch(kind, patterns, dtype, device) -> (positions [g, P, 2], scales [g, P])` (placement included); `motion_offsets(motion, progress [g, K]) -> [g, K, 2]`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_world_kernel.py`:

```python
"""The world kernel: vectorised shapes equal layered's; pattern batches; motion; registries."""

import pytest
import torch

from sensoryforge.stimuli import layered
from sensoryforge.world import kernel

N = 33
XS = torch.linspace(-2.0, 2.0, N, dtype=torch.float64)
X, Y = torch.meshgrid(XS, XS, indexing="ij")

PARAMS = {
    "gaussian": [{"sigma_mm": 0.2}, {"sigma_mm": 0.7}],
    "disc": [{"diameter_mm": 1.0, "edge_mm": 0.2}, {"diameter_mm": 0.6, "edge_mm": 0.0}],
    "bar": [
        {"width_mm": 0.1, "length_mm": 0.0, "orientation_deg": 30.0},
        {"width_mm": 0.3, "length_mm": 1.0, "orientation_deg": 100.0, "profile": "flat"},
    ],
    "grating": [
        {"wavelength_mm": 0.5, "orientation_deg": 20.0, "phase_deg": 45.0},
        {"wavelength_mm": 0.9, "orientation_deg": 70.0, "signed": True},
        {"wavelength_mm": 0.7, "profile": "square", "duty": 0.3},
    ],
    "gabor": [
        {"sigma_mm": 0.5, "wavelength_mm": 0.4, "orientation_deg": 10.0, "phase_deg": 90.0},
        {"sigma_mm": 0.8, "wavelength_mm": 0.7, "orientation_deg": 135.0, "signed": True},
    ],
}


def _tensors(params):
    return {
        k: torch.tensor([float(v)], dtype=torch.float64).view(1, 1, 1)
        if isinstance(v, (int, float)) and not isinstance(v, bool)
        else v
        for k, v in params.items()
    }


@pytest.mark.parametrize("kind", sorted(PARAMS))
def test_vectorised_shapes_equal_layered(kind):
    for given in PARAMS[kind]:
        params = {**layered.defaults(layered.SHAPES[kind]), **given}
        want = layered._SHAPE_FUNCTIONS[kind](X, Y, params)
        got = kernel.SHAPE_KINDS[kind].fn(X, Y, _tensors(params))[0]
        torch.testing.assert_close(got, want, atol=1e-12, rtol=0)


def test_a_batch_of_parameters_equals_each_alone_bit_for_bit():
    sigmas = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64).view(3, 1, 1)
    gaussian = kernel.SHAPE_KINDS["gaussian"].fn
    batch = gaussian(X, Y, {"sigma_mm": sigmas})
    for i in range(3):
        alone = gaussian(X, Y, {"sigma_mm": sigmas[i : i + 1]})
        assert torch.equal(batch[i], alone[0])


def test_pattern_batch_pads_with_zero_scales_and_adds_placement():
    base = {"kind": "braille", **layered.defaults(layered.PATTERNS["braille"])}
    patterns = [
        {**base, "dots": "1", "x_mm": 0.5, "y_mm": 0.0},
        {**base, "dots": "123456", "x_mm": 0.0, "y_mm": -0.5},
    ]
    pos, scales = kernel.pattern_batch("braille", patterns, torch.float64, "cpu")
    assert pos.shape == (2, 6, 2) and scales.shape == (2, 6)
    assert scales[0].tolist() == [1.0, 0, 0, 0, 0, 0]
    assert scales[1].tolist() == [1.0] * 6
    for i, pattern in enumerate(patterns):
        want, _ = layered.pattern_positions(pattern)
        assert pos[i, : len(want)].tolist() == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize(
    "motion",
    [
        {"kind": "linear", "start": [0.0, 0.0], "end": [2.0, 1.0]},
        {"kind": "circular", "radius_mm": 1.0, "revolutions": 0.5, "start_deg": 10.0},
        {"kind": "path", "waypoints": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]},
    ],
)
def test_motion_offsets_follow_layered(motion):
    time_ms = torch.arange(11, dtype=torch.float64)
    timing = {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": 10, "ramp_down_ms": 0}
    want = layered.motion_offsets({**motion, "span": "hold"}, timing, time_ms, 10.0)
    progress = (time_ms / 10.0).view(1, 11)
    got = kernel.motion_offsets(motion, progress)[0]
    torch.testing.assert_close(got, want, atol=1e-12, rtol=0)


def test_a_registered_shape_also_works_in_a_layered_stimulus():
    def ring(x, y, p):
        r = torch.sqrt(x**2 + y**2)
        return torch.exp(-((r - p["radius_mm"]) ** 2) / (2.0 * p["width_mm"] ** 2))

    specs = [
        layered._f("amplitude", 1.0, 0.0, 1.0e4),
        layered._f("radius_mm", 1.0, 0.0, 10.0, "mm"),
        layered._f("width_mm", 0.1, 0.001, 10.0, "mm"),
    ]
    kernel.register_shape("test_ring", ring, specs)
    try:
        layer = layered.default_layer()
        layer["shape"] = {"kind": "test_ring", "radius_mm": 1.0}
        layer["timing"] = {"onset_ms": 0, "ramp_up_ms": 0, "hold_ms": None, "ramp_down_ms": 0}
        xx, yy = X.float(), Y.float()
        frames = layered.render_layers([layer], xx, yy, dt_ms=1.0, total_ms=2.0)
        want = ring(xx, yy, {"radius_mm": torch.tensor(1.0), "width_mm": torch.tensor(0.1)})
        torch.testing.assert_close(frames[0], want, atol=1e-6, rtol=0)
        assert kernel.shape_specs("test_ring") == specs
    finally:
        kernel.SHAPE_KINDS.pop("test_ring")


def test_unknown_kinds_name_the_known_ones():
    with pytest.raises(ValueError, match="unknown shape kind 'blob'"):
        kernel.shape_specs("blob")
    with pytest.raises(ValueError, match="unknown pattern kind 'spiral'"):
        kernel.pattern_specs("spiral")
    with pytest.raises(ValueError, match="unknown modulation kind 'wobble'"):
        kernel.modulation_specs("wobble")
    assert set(kernel.MODULATION_KINDS) >= {"none", "sine", "pulses"}
```

- [ ] **Step 2: Run to see them fail**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_world_kernel.py -q`
Expected: FAIL — `ImportError: cannot import name 'kernel'`.

- [ ] **Step 3: Implement `sensoryforge/world/kernel.py`**

```python
"""Vectorised shapes, pattern batches and motion for world rendering (spec §5.2-§5.4).

The five built-in shapes are :mod:`sensoryforge.stimuli.layered`'s, rewritten
so every numeric parameter may be a tensor broadcasting against the
coordinates: one call evaluates a whole group of draws. Patterns and
modulations reuse layered's own code. Each part is a registry: a plugin adds a
shape, pattern or modulation once, and worlds and layered stimuli (which fall
back to these registries) can both use it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sensoryforge.stimuli import layered
from sensoryforge.stimuli.base import ParamSpec
from sensoryforge.stimuli.texture import raised_cosine

Positions = Tuple[List[Tuple[float, float]], List[float]]
ShapeFn = Callable[[torch.Tensor, torch.Tensor, Dict[str, Any]], torch.Tensor]
PatternFn = Callable[[Dict[str, Any]], Positions]
ModulationFn = Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]


@dataclass(frozen=True)
class ShapeKind:
    """A registered shape: ``fn(x, y, params) -> values``."""

    fn: ShapeFn
    specs: List[ParamSpec]
    unbounded: bool = False


@dataclass(frozen=True)
class PatternKind:
    """A registered pattern: ``fn(params) -> (positions, scales)`` at placement 0."""

    fn: PatternFn
    specs: List[ParamSpec]

    @property
    def placement(self) -> bool:
        """True when ``x_mm``/``y_mm`` place the pattern (a translation)."""
        return any(s.name == "x_mm" for s in self.specs)


@dataclass(frozen=True)
class ModulationKind:
    """A registered modulation: ``fn(tc, params) -> factor in [0, 1]``; ``None``: none."""

    fn: Optional[ModulationFn]
    specs: List[ParamSpec]


SHAPE_KINDS: Dict[str, ShapeKind] = {}
PATTERN_KINDS: Dict[str, PatternKind] = {}
MODULATION_KINDS: Dict[str, ModulationKind] = {}


def register_shape(
    name: str, fn: ShapeFn, specs: Sequence[ParamSpec], unbounded: bool = False
) -> None:
    """Register a shape kind for worlds and layered stimuli.

    Args:
        name: The ``shape.kind`` that selects it.
        fn: ``fn(x, y, params) -> values``. ``x``, ``y`` are mm offsets from the
            element's centre, shape ``[..., *S]``; numeric ``params`` are
            tensors broadcasting against them; strings and bools are plain.
        specs: Its parameters; must include ``amplitude``.
        unbounded: Drawn once over the plane, not at each pattern position.
    """
    if not any(s.name == "amplitude" for s in specs):
        raise ValueError(f"shape {name!r}: its specs must include 'amplitude'")
    SHAPE_KINDS[name] = ShapeKind(fn=fn, specs=list(specs), unbounded=bool(unbounded))


def register_pattern(name: str, fn: PatternFn, specs: Sequence[ParamSpec]) -> None:
    """Register a pattern kind: ``fn(params) -> (positions, scales)`` at placement 0, mm."""
    PATTERN_KINDS[name] = PatternKind(fn=fn, specs=list(specs))


def register_modulation(
    name: str, fn: Optional[ModulationFn], specs: Sequence[ParamSpec]
) -> None:
    """Register a modulation kind: ``fn(tc, params) -> [0, 1]``, ``tc`` ms since the touch."""
    MODULATION_KINDS[name] = ModulationKind(fn=fn, specs=list(specs))


def _specs(table: Dict[str, Any], kind: str, what: str) -> List[ParamSpec]:
    if kind not in table:
        raise ValueError(f"unknown {what} kind {kind!r}; known: {sorted(table)}")
    return table[kind].specs


def shape_specs(kind: str) -> List[ParamSpec]:
    """A shape kind's parameters."""
    return _specs(SHAPE_KINDS, kind, "shape")


def pattern_specs(kind: str) -> List[ParamSpec]:
    """A pattern kind's parameters."""
    return _specs(PATTERN_KINDS, kind, "pattern")


def modulation_specs(kind: str) -> List[ParamSpec]:
    """A modulation kind's parameters."""
    return _specs(MODULATION_KINDS, kind, "modulation")


# ------------------------------------------------------------- shapes


def _gaussian(x, y, p):
    s = p["sigma_mm"]
    return torch.exp(-(x**2 + y**2) / (2.0 * s**2))


def _disc(x, y, p):
    radius = p["diameter_mm"] / 2.0
    edge = p["edge_mm"]
    r = torch.sqrt(x**2 + y**2)
    hard = (r <= radius).to(x.dtype)
    safe = torch.where(edge > 0, edge, torch.ones_like(edge))
    soft = ((radius - r) / safe + 0.5).clamp(0.0, 1.0)
    return torch.where(edge > 0, soft, hard)


def _rotated(x, y, orientation_deg):
    theta = torch.deg2rad(orientation_deg)
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    # The moving-edge convention of pressure-simulation: p = x sin + y cos.
    return x * sin_t + y * cos_t, x * cos_t - y * sin_t


def _bar(x, y, p):
    across, along = _rotated(x, y, p["orientation_deg"])
    width = p["width_mm"]
    if p.get("profile", "gaussian") == "flat":
        value = (across.abs() <= width / 2.0).to(x.dtype)
    else:
        value = torch.exp(-(across**2) / (2.0 * width**2))
    length = p["length_mm"]
    finite = value * (along.abs() <= length / 2.0).to(x.dtype)
    return torch.where(length > 0, finite, value)


def _stripes(across, p):
    phase = torch.remainder(
        2.0 * math.pi * across / p["wavelength_mm"] + torch.deg2rad(p["phase_deg"]),
        2.0 * math.pi,
    )
    signed = bool(p.get("signed", False))
    if p.get("profile", "sine") == "square":
        centred = torch.minimum(phase, 2.0 * math.pi - phase)
        on = (centred <= math.pi * p["duty"]).to(across.dtype)
        return 2.0 * on - 1.0 if signed else on
    return torch.cos(phase) if signed else raised_cosine(phase)


def _grating(x, y, p):
    theta = torch.deg2rad(p["orientation_deg"])
    return _stripes(x * torch.cos(theta) + y * torch.sin(theta), p)


def _gabor(x, y, p):
    return _gaussian(x, y, p) * _grating(x, y, p)


for _kind, _fn in (
    ("gaussian", _gaussian),
    ("disc", _disc),
    ("bar", _bar),
    ("grating", _grating),
    ("gabor", _gabor),
):
    register_shape(_kind, _fn, layered.SHAPES[_kind], unbounded=_kind == "grating")


# --------------------------------------------- patterns and modulations


def _layered_pattern(kind: str) -> PatternFn:
    def positions(params: Dict[str, Any]) -> Positions:
        return layered.pattern_positions({**params, "kind": kind, "x_mm": 0.0, "y_mm": 0.0})

    return positions


for _kind in layered.PATTERNS:
    register_pattern(_kind, _layered_pattern(_kind), layered.PATTERNS[_kind])

register_modulation("none", None, layered.MODULATIONS["none"])
register_modulation("sine", layered.modulate_sine, layered.MODULATIONS["sine"])
register_modulation("pulses", layered.modulate_pulses, layered.MODULATIONS["pulses"])

_POSITION_CACHE: Dict[str, Positions] = {}
_CACHE_LIMIT = 100_000


def pattern_batch(
    kind: str,
    patterns: Sequence[Dict[str, Any]],
    dtype: torch.dtype,
    device: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Element positions and scales for a group of draws' patterns.

    Positions are computed once per distinct pattern at placement 0 (cached),
    padded to the group's largest element count with zero scales, then
    translated by each pattern's ``x_mm``/``y_mm``.

    Args:
        kind: The pattern kind shared by the group.
        patterns: One full pattern dict per draw.
        dtype: Output dtype.
        device: Output device.

    Returns:
        ``(positions [g, P, 2] mm, scales [g, P])``.
    """
    registered = PATTERN_KINDS[kind]
    lists = []
    for pattern in patterns:
        params = {k: v for k, v in pattern.items() if k not in ("kind", "x_mm", "y_mm")}
        cache_key = kind + json.dumps(params, sort_keys=True, default=str)
        if cache_key not in _POSITION_CACHE:
            if len(_POSITION_CACHE) >= _CACHE_LIMIT:
                _POSITION_CACHE.clear()
            _POSITION_CACHE[cache_key] = registered.fn(params)
        lists.append(_POSITION_CACHE[cache_key])
    g = len(patterns)
    size = max([1] + [len(points) for points, _ in lists])
    positions = torch.zeros(g, size, 2, dtype=torch.float64)
    scales = torch.zeros(g, size, dtype=torch.float64)
    for i, (points, weights) in enumerate(lists):
        if points:
            positions[i, : len(points)] = torch.tensor(points, dtype=torch.float64)
            scales[i, : len(weights)] = torch.tensor(weights, dtype=torch.float64)
    if registered.placement:
        place = torch.tensor(
            [[float(p.get("x_mm", 0.0)), float(p.get("y_mm", 0.0))] for p in patterns],
            dtype=torch.float64,
        )
        positions = positions + place[:, None, :]
    return positions.to(device=device, dtype=dtype), scales.to(device=device, dtype=dtype)


# ------------------------------------------------------------- motion


def motion_offsets(motion: Dict[str, Any], progress: torch.Tensor) -> torch.Tensor:
    """Translation ``[..., 2]`` in mm at motion progress ``progress`` (in ``[0, 1]``).

    The formulas of :func:`sensoryforge.stimuli.layered.motion_offsets`,
    evaluated on any progress tensor.
    """
    kind = motion.get("kind", "none")
    if kind == "none":
        zeros = torch.zeros_like(progress)
        return torch.stack([zeros, zeros], dim=-1)
    if kind == "linear":
        (x0, y0), (x1, y1) = motion["start"], motion["end"]
        return torch.stack(
            [x0 + progress * (x1 - x0), y0 + progress * (y1 - y0)], dim=-1
        )
    if kind == "circular":
        angle = (
            math.radians(float(motion.get("start_deg", 0.0)))
            + 2.0 * math.pi * float(motion["revolutions"]) * progress
        )
        r = float(motion["radius_mm"])
        return torch.stack([r * torch.cos(angle), r * torch.sin(angle)], dim=-1)
    if kind == "path":
        points = torch.tensor(motion["waypoints"], dtype=progress.dtype, device=progress.device)
        if points.shape[0] < 2:
            return points[0].expand(*progress.shape, 2).clone()
        seg = (points[1:] - points[:-1]).norm(dim=1)
        cum = torch.cat([torch.zeros(1, dtype=seg.dtype, device=seg.device), seg.cumsum(0)])
        total = float(cum[-1]) or 1.0
        d = progress * total
        idx = torch.searchsorted(cum, d.clamp(max=total - 1e-9).contiguous(), right=True) - 1
        idx = idx.clamp(0, len(seg) - 1)
        frac = ((d - cum[idx]) / seg[idx].clamp(min=1e-12)).unsqueeze(-1)
        return points[idx] + frac * (points[idx + 1] - points[idx])
    raise ValueError(f"unknown motion kind {kind!r}; known: none, linear, circular, path")
```

- [ ] **Step 4: Make `layered` fall back to the registries**

In `sensoryforge/stimuli/layered.py`:

1. Add, next to `_world_kernel`:

```python
def _shape_kind(kind: str, like: torch.Tensor):
    """``(fn, specs, unbounded)``: built in, else from the world kernel's registry."""
    if kind in SHAPES:
        return _SHAPE_FUNCTIONS[kind], SHAPES[kind], kind in _UNBOUNDED
    kernel = _world_kernel()
    if kind not in kernel.SHAPE_KINDS:
        raise ValueError(
            f"unknown shape kind {kind!r}; known: "
            f"{sorted(set(SHAPES) | set(kernel.SHAPE_KINDS))}"
        )
    registered = kernel.SHAPE_KINDS[kind]

    def fn(x, y, p):
        tensors = {
            k: torch.tensor(float(v), dtype=like.dtype, device=like.device)
            if _is_number(v)
            else v
            for k, v in p.items()
        }
        return registered.fn(x, y, tensors)

    return fn, registered.specs, registered.unbounded
```

2. In `render_layer`, replace

```python
    if kind not in SHAPES:
        raise ValueError(f"unknown shape kind {kind!r}; known: {sorted(SHAPES)}")
    params = {**defaults(SHAPES[kind]), **shape}
    fn = _SHAPE_FUNCTIONS[kind]
```

with

```python
    fn, specs, unbounded = _shape_kind(kind, xx)
    params = {**defaults(specs), **shape}
```

and in its inner `draw`, replace `if kind in _UNBOUNDED:` with `if unbounded:`.

3. In `pattern_positions`, replace

```python
    if kind not in PATTERNS:
        raise ValueError(f"unknown pattern kind {kind!r}; known: {sorted(PATTERNS)}")
```

with

```python
    if kind not in PATTERNS:
        kernel = _world_kernel()
        if kind not in kernel.PATTERN_KINDS:
            raise ValueError(
                f"unknown pattern kind {kind!r}; known: "
                f"{sorted(set(PATTERNS) | set(kernel.PATTERN_KINDS))}"
            )
        registered = kernel.PATTERN_KINDS[kind]
        p = {**defaults(registered.specs), **pattern}
        positions, scales = registered.fn(
            {k: v for k, v in p.items() if k not in ("kind", "x_mm", "y_mm")}
        )
        if registered.placement:
            dx, dy = float(p.get("x_mm", 0.0)), float(p.get("y_mm", 0.0))
            positions = [(x + dx, y + dy) for x, y in positions]
        return list(positions), list(scales)
```

- [ ] **Step 5: Run the kernel tests, the layered tests and the golden test**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_world_kernel.py tests/unit/test_layered_stimuli.py tests/unit/test_layered_episode.py tests/unit/test_layered_golden.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add sensoryforge/world/kernel.py sensoryforge/stimuli/layered.py tests/unit/test_world_kernel.py
git commit -m "feat(world): vectorised kernel with shape, pattern and modulation registries

The five layered shapes rewritten to take tensor parameters (equal to
layered's to 1e-12), pattern positions batched per distinct pattern with
zero-scale padding, motion offsets for any progress tensor. A shape or
pattern registered here also works in a layered stimulus, which falls
back to these registries for kinds it does not know.

Refs: D-cccbd6a

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Class kinds, the world schema and the test world

**Files:**
- Create: `sensoryforge/world/kinds.py`, `sensoryforge/world/schema.py`, `tests/fixtures/worlds/tactile_small.yml`, `tests/unit/test_world_schema.py`, `tests/unit/test_world_kinds.py`

**Interfaces:**
- Consumes: `AxisSpec`, `plain` (Task 2); `kernel.shape_specs/pattern_specs/modulation_specs` (Task 3); `layered.defaults`, `layered.MOTIONS`.
- Produces (`kinds`): `Ref = Tuple[str, str]` (`part`, `field`; parts `episode`, `shape`, `pattern`, `modulation`, `quiet`); `UnknownField`, `AmbiguousField` (both `ValueError`); `EPISODE_FIELDS`; `ClassKind` with `name`, `normalise_layer(layer, where)`, `builtin_defaults(layer) -> Dict[str, Any]`, `resolve(name, layer) -> Ref`, `domain(ref, layer) -> (lo, hi)`, `fixed_in_layer(ref, raw_layer) -> bool`, `check(spec)`, `end_ms(values) -> float`, `timeline(values) -> List[[phase, start, end]]`, `to_layer(spec, values) -> Optional[dict]`, `render_group(spec, draws, X, Y, times) -> Tensor`; `LayeredKind` (+ `part_values(spec, values)`, `known(layer)`; Task 6 adds its `render_group`), `QuietKind`; `CLASS_KINDS`; `register_class_kind(kind, replace=False)`.
- Produces (`schema`): `FORMAT = "sensoryforge-world/1"`; `ClassSpec(name, kind, weight, channel, layer, axes, bindings, held_out)` with `kind_obj`, `random_axes`, `to_dict()`; `World(name, description, modality, units, channels, classes, held_out, fixed, world_id)` with `class_spec(name)`, `to_dict()`, `fixed_draw(name)` (works from Task 5), `from_dict(data)`; `load_world(source) -> World` (path, dict or `World`).

- [ ] **Step 1: Create the test world** `tests/fixtures/worlds/tactile_small.yml`

```yaml
# A small world for SensoryForge's own tests: every class kind, shape family,
# motion, modulation and contact pattern the world engine supports, at sizes
# that fit an 8x8 grid at 0.15 mm (tests/fixtures/design_8x8) and durations
# short enough to simulate quickly. The values are placeholders, not a
# survey of natural touch.
world:
  name: tactile_small
  description: a small world for SensoryForge's own tests
  modality: tactile
  units: {space: mm, time: ms}
  defaults:
    delay_ms:   {range: [0, 20]}
    touch_ms:   {range: [5, 15]}
    hold_ms:    {range: [20, 60]}
    release_ms: {range: [5, 15]}
    speed_mm_per_ms: {range: [0.005, 0.02], dist: log_uniform}
    direction_deg:   {range: [0, 360], circular: true}
    amplitude:  {range: [0.5, 1.0]}
    x_mm: {range: [-0.3, 0.3]}
    y_mm: {range: [-0.3, 0.3]}
  classes:
    dots:
      weight: 0.2
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.15, 0.45]}}
    edges:
      weight: 0.15
      layer: {shape: {kind: bar, profile: gaussian, length_mm: 0}}
      axes:
        width_mm: {range: [0.05, 0.15]}
        orientation_deg: {range: [0, 180], circular: true}
    braille:
      weight: 0.15
      layer:
        shape: {kind: gaussian, sigma_mm: 0.15}
        pattern: {kind: braille, dot_spacing_mm: 0.35}
      axes: {dots: {dist: braille_cells}}
    sliders:
      weight: 0.15
      layer: {shape: {kind: gaussian, sigma_mm: 0.3}}
      axes: {slide_ms: {range: [20, 40]}, hold_ms: {range: [5, 20]}}
    orbit:
      weight: 0.05
      layer:
        shape: {kind: gaussian, sigma_mm: 0.25}
        motion: {kind: circular, radius_mm: 0.2, revolutions: 1}
      axes: {slide_ms: {range: [20, 40]}}
    taps:
      weight: 0.1
      layer:
        shape: {kind: disc, diameter_mm: 0.6}
        modulation: {kind: pulses, duty: 0.5, edge_ms: 2}
      axes: {rate_hz: {range: [20, 80], dist: log_uniform}}
    vibes:
      weight: 0.1
      layer:
        shape: {kind: gaussian, sigma_mm: 0.4}
        modulation: {kind: sine, depth: 0.3}
      axes: {frequency_hz: {range: [50, 200]}}
    twice:
      weight: 0.05
      layer: {shape: {kind: gaussian, sigma_mm: 0.3}}
      axes: {contacts: {value: 2}, pause_ms: {range: [5, 15]}}
    quiet:
      kind: quiet
      weight: 0.05
      axes: {quiet_ms: {range: [20, 60]}}
  held_out:
    gratings:
      layer: {shape: {kind: grating}}
      axes:
        wavelength_mm: {range: [0.3, 0.9]}
        orientation_deg: {range: [0, 180], circular: true}
  fixed_draws:
    braille_H: {class: braille, dots: "125", hold_ms: 40, delay_ms: 0}
    wide_dot: {class: dots, sigma_mm: 1.0}
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_world_schema.py`:

```python
"""World declarations: binding, defaults, precedence, identity, validation."""

import copy
import re
from pathlib import Path

import pytest
import yaml

from sensoryforge.world.schema import World, load_world

WORLDS = Path(__file__).resolve().parents[1] / "fixtures" / "worlds"
RAW = yaml.safe_load((WORLDS / "tactile_small.yml").read_text())


def _world(mutate=None):
    raw = copy.deepcopy(RAW)
    if mutate is not None:
        mutate(raw["world"])
    return World.from_dict(raw)


def test_the_fixture_world_loads():
    world = load_world(WORLDS / "tactile_small.yml")
    assert set(world.classes) == {
        "dots", "edges", "braille", "sliders", "orbit", "taps", "vibes", "twice", "quiet",
    }
    assert set(world.held_out) == {"gratings"}
    assert set(world.fixed) == {"braille_H", "wide_dot"}
    assert re.fullmatch(r"w-[0-9a-f]{12}", world.world_id)
    assert world.channels == ["value"]
    assert load_world(world) is world


def test_axes_bind_to_fields_and_constants_are_recorded():
    world = _world()
    dots = world.classes["dots"]
    assert dots.bindings["sigma_mm"] == ("shape", "sigma_mm")
    assert dots.bindings["x_mm"] == ("pattern", "x_mm")
    assert dots.bindings["hold_ms"] == ("episode", "hold_ms")
    assert dots.axes["slide_ms"].to_dict() == {"value": 0.0}
    assert dots.axes["contacts"].to_dict() == {"value": 1}
    assert world.classes["braille"].bindings["dots"] == ("pattern", "dots")
    assert world.classes["taps"].bindings["rate_hz"] == ("modulation", "rate_hz")
    assert set(world.classes["quiet"].axes) == {"quiet_ms"}


def test_world_defaults_skip_fields_a_class_lacks_or_fixes():
    def add(w):
        w["defaults"]["rate_hz"] = {"range": [1, 2]}
        w["defaults"]["sigma_mm"] = {"range": [0.2, 0.3]}

    world = _world(add)
    assert "rate_hz" not in world.classes["dots"].axes  # not modulated
    assert world.classes["taps"].axes["rate_hz"].hi == 80  # its own axis wins
    assert world.classes["dots"].axes["sigma_mm"].hi == 0.45  # its own axis wins
    assert "sigma_mm" not in world.classes["braille"].axes  # fixed in its layer
    assert "sigma_mm" not in world.classes["edges"].axes  # a bar has no sigma


def test_a_layer_fixed_amplitude_beats_the_world_default():
    def fix(w):
        w["classes"]["dots"]["layer"]["shape"]["amplitude"] = 2.0

    assert _world(fix).classes["dots"].axes["amplitude"].to_dict() == {"value": 2.0}


def test_ambiguous_names_need_a_dotted_path():
    def ambiguous(w):
        w["classes"]["edges"]["layer"]["pattern"] = {"kind": "random", "count": 3}

    with pytest.raises(ValueError, match="ambiguous.*shape.width_mm"):
        _world(ambiguous)

    def dotted(w):
        ambiguous(w)
        axes = w["classes"]["edges"]["axes"]
        axes["shape.width_mm"] = axes.pop("width_mm")

    edges = _world(dotted).classes["edges"]
    assert edges.bindings["shape.width_mm"] == ("shape", "width_mm")


def test_identity_ignores_the_description_but_not_values():
    base = _world().world_id
    assert _world(lambda w: w.update(description="other")).world_id == base
    widened = _world(
        lambda w: w["classes"]["dots"]["axes"]["sigma_mm"].update(range=[0.15, 0.46])
    )
    assert widened.world_id != base


def _never_touch(w):
    w["defaults"].update(touch_ms={"value": 0}, release_ms={"value": 0})
    w["classes"]["sliders"]["axes"].update(hold_ms={"value": 0}, slide_ms={"value": 0})


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda w: w["held_out"]["gratings"].update(weight=1.0), "take no weight"),
        (lambda w: w["classes"]["dots"].update(kind="hologram"), "unknown class kind"),
        (lambda w: w["classes"]["dots"]["layer"].update(timing={"hold_ms": 5}), "episode axes"),
        (lambda w: w["classes"]["dots"].update(channel="heat"), "not one of the world's channels"),
        (_never_touch, "never touches"),
        (lambda w: w["fixed_draws"].update(bad={"class": "nope"}), "unknown class"),
        (lambda w: w["fixed_draws"].update(bad={"class": "dots", "radius_mm": 1}), "radius_mm"),
        (
            lambda w: w["classes"].update(gratings={"layer": {"shape": {"kind": "grating"}}}),
            "also classes",
        ),
        (lambda w: w["classes"]["dots"]["layer"]["shape"].update(colour=1), "has no fields"),
        (
            lambda w: w["classes"]["dots"]["axes"].update(radius_mm={"range": [0, 1]}),
            "radius_mm.*known",
        ),
        (lambda w: w.update(colour="blue"), "unknown keys"),
        (lambda w: w["classes"]["orbit"]["layer"]["motion"].update(span="hold"), "drop 'span'"),
    ],
)
def test_invalid_worlds_are_named(mutate, message):
    with pytest.raises(ValueError, match=message):
        _world(mutate)
```

`tests/unit/test_world_kinds.py`:

```python
"""Class kinds: timelines, end times, the layered form of a draw, the registry."""

from pathlib import Path

import pytest

from sensoryforge.world.kinds import LayeredKind, register_class_kind
from sensoryforge.world.schema import load_world

WORLD = load_world(
    Path(__file__).resolve().parents[1] / "fixtures" / "worlds" / "tactile_small.yml"
)


def _values(class_name, **overrides):
    spec = WORLD.class_spec(class_name)
    values = {name: axis.midpoint() for name, axis in spec.axes.items()}
    values.update(overrides)
    return spec, values


def test_timeline_and_end_of_a_two_contact_episode():
    spec, v = _values(
        "twice", delay_ms=5.0, touch_ms=10.0, hold_ms=20.0, release_ms=10.0, pause_ms=7.0
    )
    kind = spec.kind_obj
    assert kind.end_ms(v) == 5.0 + 2 * 40.0 + 7.0
    phases = kind.timeline(v)
    assert [p[0] for p in phases] == [
        "quiet", "touch", "hold", "release", "pause", "touch", "hold", "release",
    ]
    assert phases[-1][2] == pytest.approx(92.0)


def test_to_layer_maps_the_episode_onto_layered_fields():
    spec, v = _values(
        "sliders", delay_ms=3.0, touch_ms=4.0, hold_ms=5.0, slide_ms=20.0,
        release_ms=6.0, speed_mm_per_ms=0.01, direction_deg=90.0,
        amplitude=0.7, x_mm=0.1, y_mm=-0.2,
    )
    layer = spec.kind_obj.to_layer(spec, v)
    assert layer["timing"] == {
        "onset_ms": 3.0, "ramp_up_ms": 4.0, "hold_ms": 5.0, "slide_ms": 20.0,
        "ramp_down_ms": 6.0, "contacts": 1, "pause_ms": 0.0,
    }
    assert layer["motion"]["kind"] == "linear" and layer["motion"]["span"] == "slide"
    assert layer["motion"]["end"] == pytest.approx([0.0, 0.2], abs=1e-12)
    assert layer["shape"]["amplitude"] == 0.7
    assert (layer["pattern"]["x_mm"], layer["pattern"]["y_mm"]) == (0.1, -0.2)
    assert layer["modulation"] == {"kind": "none"}


def test_a_declared_motion_is_kept_and_moves_during_the_slide():
    spec, v = _values("orbit")
    layer = spec.kind_obj.to_layer(spec, v)
    assert layer["motion"]["kind"] == "circular" and layer["motion"]["span"] == "slide"


def test_the_quiet_kind():
    spec = WORLD.class_spec("quiet")
    v = {"quiet_ms": 30.0}
    assert spec.kind_obj.end_ms(v) == 30.0
    assert spec.kind_obj.timeline(v) == [["quiet", 0.0, 30.0]]
    assert spec.kind_obj.to_layer(spec, v) is None


def test_class_kinds_cannot_be_registered_twice():
    with pytest.raises(ValueError, match="already registered"):
        register_class_kind(LayeredKind())
```

- [ ] **Step 3: Run to see them fail**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_world_schema.py tests/unit/test_world_kinds.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sensoryforge.world.schema'`.

- [ ] **Step 4: Implement `sensoryforge/world/kinds.py`**

```python
"""Class kinds: how a world class binds axis names, times its draws and renders them.

Spec §3.1, §3.3, §3.4, §5.4. A class kind is registered by name; ``layered``
(a layered layer with random fields) and ``quiet`` (exactly zero) are built
in. A plugin adds a kind with :func:`register_class_kind`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from sensoryforge.stimuli.layered import MOTIONS, defaults
from sensoryforge.world import kernel

Ref = Tuple[str, str]


class UnknownField(ValueError):
    """An axis name that is not a field of the class."""


class AmbiguousField(ValueError):
    """A bare axis name that matches fields in more than one part."""


#: Episode fields every layered class has (spec §3.3).
EPISODE_FIELDS: Tuple[str, ...] = (
    "delay_ms",
    "touch_ms",
    "hold_ms",
    "slide_ms",
    "release_ms",
    "contacts",
    "pause_ms",
    "speed_mm_per_ms",
    "direction_deg",
)
_EPISODE_DEFAULTS: Dict[str, Any] = {
    "delay_ms": 0.0,
    "touch_ms": 0.0,
    "hold_ms": 0.0,
    "slide_ms": 0.0,
    "release_ms": 0.0,
    "contacts": 1,
    "pause_ms": 0.0,
    "speed_mm_per_ms": 0.0,
    "direction_deg": 0.0,
}
_EPISODE_DOMAIN: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "contacts": (1.0, None),
    "direction_deg": (None, None),
}
_CONTACT_PHASES = (
    ("touch", "touch_ms"),
    ("hold", "hold_ms"),
    ("slide", "slide_ms"),
    ("release", "release_ms"),
)
_PARTS = ("shape", "pattern", "modulation")
_MOTION_KINDS = ("none", "linear", "circular", "path")


class ClassKind:
    """Base class of class kinds; :class:`LayeredKind` documents the contract by example."""

    name = ""

    def normalise_layer(self, layer: Any, where: str) -> Optional[Dict[str, Any]]:
        """The class's ``layer`` with defaults filled (``None`` if the kind has none)."""
        raise NotImplementedError

    def builtin_defaults(self, layer: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """``{axis name: constant}`` every class of this kind binds."""
        raise NotImplementedError

    def resolve(self, name: str, layer: Optional[Dict[str, Any]]) -> Ref:
        """The field an axis name binds to; raises UnknownField / AmbiguousField."""
        raise NotImplementedError

    def domain(
        self, ref: Ref, layer: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[float]]:
        """The valid range of a field (probes stay inside it)."""
        return (None, None)

    def fixed_in_layer(self, ref: Ref, raw_layer: Any) -> bool:
        """True when the class's own layer sets this field (a world default yields)."""
        return False

    def check(self, spec: Any) -> None:
        """Kind-specific validation of a parsed class."""

    def end_ms(self, values: Dict[str, Any]) -> float:
        """When a draw with these values ends, ms."""
        raise NotImplementedError

    def timeline(self, values: Dict[str, Any]) -> List[List[Any]]:
        """``[[phase, start_ms, end_ms], ...]``, zero-length phases omitted."""
        raise NotImplementedError

    def to_layer(self, spec: Any, values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The draw as an ordinary layered layer dict (``None`` if it has none)."""
        return None

    def render_group(
        self,
        spec: Any,
        draws: List[Any],
        X: torch.Tensor,
        Y: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Frames ``[g, K, *S]`` for a group of draws at times ``[g, K]`` (ms)."""
        raise NotImplementedError


def _fill(part: Any, default_kind: str, specs_fn, where: str) -> Dict[str, Any]:
    part = dict(part or {"kind": default_kind})
    kind = part.get("kind", default_kind)
    try:
        specs = specs_fn(kind)
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None
    names = {s.name for s in specs}
    extra = set(part) - names - {"kind"}
    if extra:
        raise ValueError(
            f"{where}: {kind!r} has no fields {sorted(extra)}; it has {sorted(names)}"
        )
    return {
        "kind": kind,
        **defaults(specs),
        **{k: v for k, v in part.items() if k != "kind"},
    }


class LayeredKind(ClassKind):
    """A class drawn as a layered layer: shape x pattern x modulation, timed by an episode."""

    name = "layered"

    def normalise_layer(self, layer, where):
        layer = dict(layer or {})
        if "timing" in layer:
            raise ValueError(
                f"{where}: set timing with the episode axes (delay_ms, touch_ms, "
                "hold_ms, slide_ms, release_ms, contacts, pause_ms), not layer.timing"
            )
        unknown = set(layer) - {"shape", "pattern", "motion", "modulation"}
        if unknown:
            raise ValueError(
                f"{where}: unknown parts {sorted(unknown)}; "
                "a layer has shape, pattern, motion, modulation"
            )
        out: Dict[str, Any] = {
            "shape": _fill(layer.get("shape"), "gaussian", kernel.shape_specs, f"{where}.shape"),
            "pattern": _fill(
                layer.get("pattern"), "single", kernel.pattern_specs, f"{where}.pattern"
            ),
            "modulation": _fill(
                layer.get("modulation"), "none", kernel.modulation_specs, f"{where}.modulation"
            ),
            "motion": None,
        }
        motion = layer.get("motion")
        if motion is not None:
            motion = dict(motion)
            kind = motion.get("kind", "none")
            if kind not in _MOTION_KINDS:
                raise ValueError(
                    f"{where}.motion: unknown kind {kind!r}; known: {list(_MOTION_KINDS)}"
                )
            if "span" in motion:
                raise ValueError(
                    f"{where}.motion: a world class moves during its slides; drop 'span'"
                )
            specs = [s for s in MOTIONS[kind] if s.name != "span"]
            extra = set(motion) - {"kind"} - {s.name for s in specs}
            if extra:
                raise ValueError(f"{where}.motion: {kind!r} has no fields {sorted(extra)}")
            out["motion"] = {"kind": kind, **defaults(specs), **motion}
        return out

    def _names(self, layer) -> Dict[str, set]:
        return {
            "shape": {s.name for s in kernel.shape_specs(layer["shape"]["kind"])},
            "pattern": {s.name for s in kernel.pattern_specs(layer["pattern"]["kind"])},
            "modulation": {
                s.name for s in kernel.modulation_specs(layer["modulation"]["kind"])
            },
        }

    def known(self, layer) -> List[str]:
        """Every name an axis of this class can bind."""
        names = self._names(layer)
        bare = set().union(*names.values())
        dotted = {f"{p}.{n}" for p in _PARTS for n in names[p]}
        return sorted(set(EPISODE_FIELDS) | bare | dotted)

    def builtin_defaults(self, layer):
        out = dict(_EPISODE_DEFAULTS)
        out["amplitude"] = layer["shape"]["amplitude"]
        if "x_mm" in self._names(layer)["pattern"]:
            out["x_mm"] = layer["pattern"]["x_mm"]
            out["y_mm"] = layer["pattern"]["y_mm"]
        return out

    def resolve(self, name, layer):
        if name in EPISODE_FIELDS:
            return ("episode", name)
        names = self._names(layer)
        if name == "amplitude":
            return ("shape", "amplitude")
        if name in ("x_mm", "y_mm"):
            if name in names["pattern"]:
                return ("pattern", name)
            raise UnknownField(
                f"{name!r}: pattern {layer['pattern']['kind']!r} has no placement"
            )
        if "." in name:
            part, field = name.split(".", 1)
            if part in names and field in names[part]:
                return (part, field)
            raise UnknownField(
                f"{name!r} is not a field of this class; known: {self.known(layer)}"
            )
        hits = [part for part in _PARTS if name in names[part]]
        if not hits:
            raise UnknownField(
                f"{name!r} is not a field of this class; known: {self.known(layer)}"
            )
        if len(hits) > 1:
            raise AmbiguousField(
                f"{name!r} is ambiguous; write one of {[h + '.' + name for h in hits]}"
            )
        return (hits[0], name)

    def domain(self, ref, layer):
        part, field = ref
        if part == "episode":
            return _EPISODE_DOMAIN.get(field, (0.0, None))
        specs_fn = {
            "shape": kernel.shape_specs,
            "pattern": kernel.pattern_specs,
            "modulation": kernel.modulation_specs,
        }[part]
        spec = next(s for s in specs_fn(layer[part]["kind"]) if s.name == field)
        return (spec.min_val, spec.max_val)

    def fixed_in_layer(self, ref, raw_layer):
        part, field = ref
        return part in _PARTS and field in ((raw_layer or {}).get(part) or {})

    def check(self, spec):
        phases = [spec.axes[field] for _, field in _CONTACT_PHASES]
        if all(not a.is_random and float(a.value) == 0.0 for a in phases):
            raise ValueError(
                f"class {spec.name!r}: touch_ms + hold_ms + slide_ms + release_ms is "
                "always 0, so it never touches; give it a hold_ms"
            )

    def end_ms(self, values):
        contacts = int(values["contacts"])
        contact = sum(float(values[field]) for _, field in _CONTACT_PHASES)
        return (
            float(values["delay_ms"])
            + contacts * contact
            + (contacts - 1) * float(values["pause_ms"])
        )

    def timeline(self, values):
        out: List[List[Any]] = []
        t = 0.0

        def add(phase: str, length: float) -> None:
            nonlocal t
            if length > 0:
                out.append([phase, t, t + length])
                t += length

        add("quiet", float(values["delay_ms"]))
        for k in range(int(values["contacts"])):
            if k:
                add("pause", float(values["pause_ms"]))
            for phase, field in _CONTACT_PHASES:
                add(phase, float(values[field]))
        return out

    def part_values(self, spec, values) -> Dict[str, Dict[str, Any]]:
        """The class's shape, pattern and modulation dicts with these axis values set."""
        parts = {part: dict(spec.layer[part]) for part in _PARTS}
        for name, value in values.items():
            part, field = spec.bindings[name]
            if part in parts:
                parts[part][field] = value
        return parts

    def to_layer(self, spec, values):
        parts = self.part_values(spec, values)
        v = values
        timing = {
            "onset_ms": float(v["delay_ms"]),
            "ramp_up_ms": float(v["touch_ms"]),
            "hold_ms": float(v["hold_ms"]),
            "slide_ms": float(v["slide_ms"]),
            "ramp_down_ms": float(v["release_ms"]),
            "contacts": int(v["contacts"]),
            "pause_ms": float(v["pause_ms"]),
        }
        declared = spec.layer.get("motion")
        if declared is not None:
            motion = {**declared, "span": "slide"}
        else:
            travel = float(v["speed_mm_per_ms"]) * int(v["contacts"]) * float(v["slide_ms"])
            theta = math.radians(float(v["direction_deg"]))
            motion = {
                "kind": "linear",
                "start": [0.0, 0.0],
                "end": [travel * math.cos(theta), travel * math.sin(theta)],
                "span": "slide",
            }
        return {
            "shape": parts["shape"],
            "pattern": parts["pattern"],
            "motion": motion,
            "timing": timing,
            "modulation": parts["modulation"],
        }


class QuietKind(ClassKind):
    """A class whose draws are exactly zero for ``quiet_ms``."""

    name = "quiet"

    def normalise_layer(self, layer, where):
        if layer:
            raise ValueError(f"{where}: a quiet class has no layer")
        return None

    def builtin_defaults(self, layer):
        return {"quiet_ms": 0.0}

    def resolve(self, name, layer):
        if name == "quiet_ms":
            return ("quiet", "quiet_ms")
        raise UnknownField(f"{name!r}: a quiet class has one axis, quiet_ms")

    def domain(self, ref, layer):
        return (0.0, None)

    def end_ms(self, values):
        return float(values["quiet_ms"])

    def timeline(self, values):
        q = float(values["quiet_ms"])
        return [["quiet", 0.0, q]] if q > 0 else []

    def render_group(self, spec, draws, X, Y, times):
        shape = (len(draws), times.shape[1]) + tuple(X.shape)
        return torch.zeros(shape, dtype=X.dtype, device=X.device)


CLASS_KINDS: Dict[str, ClassKind] = {}


def register_class_kind(kind: ClassKind, *, replace: bool = False) -> None:
    """Register a class kind under ``kind.name``.

    Raises:
        ValueError: If the name is empty, or taken and ``replace`` is false.
    """
    if not kind.name:
        raise ValueError("a class kind needs a name")
    if kind.name in CLASS_KINDS and not replace:
        raise ValueError(f"class kind {kind.name!r} is already registered")
    CLASS_KINDS[kind.name] = kind


register_class_kind(LayeredKind())
register_class_kind(QuietKind())
```

- [ ] **Step 5: Implement `sensoryforge/world/schema.py`**

```python
"""World declarations: load, validate, bind, normalise and identify (spec §3)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from sensoryforge.world.distributions import AxisSpec, plain
from sensoryforge.world.kinds import (
    CLASS_KINDS,
    AmbiguousField,
    ClassKind,
    Ref,
    UnknownField,
)

FORMAT = "sensoryforge-world/1"
_TOP_KEYS = {
    "name",
    "description",
    "modality",
    "units",
    "channels",
    "defaults",
    "classes",
    "held_out",
    "fixed_draws",
}
_CLASS_KEYS = {"kind", "weight", "layer", "axes", "channel"}


@dataclass
class ClassSpec:
    """One class of a world, with every axis resolved (spec §3.1).

    Attributes:
        name: The class name.
        kind: The class kind's registered name.
        weight: Its weight in declared sampling (0 for a held-out class).
        channel: The world channel it draws on.
        layer: Its normalised layer (``None`` for ``quiet``).
        axes: Every bound name -> its axis, constants included.
        bindings: Every bound name -> ``(part, field)``.
        held_out: True for a held-out class.
    """

    name: str
    kind: str
    weight: float
    channel: str
    layer: Optional[Dict[str, Any]]
    axes: Dict[str, AxisSpec]
    bindings: Dict[str, Ref]
    held_out: bool = False

    @property
    def kind_obj(self) -> ClassKind:
        """The registered class kind."""
        return CLASS_KINDS[self.kind]

    @property
    def random_axes(self) -> List[AxisSpec]:
        """The axes that are sampled (not constants)."""
        return [a for a in self.axes.values() if a.is_random]

    def to_dict(self) -> Dict[str, Any]:
        """The normalised declaration."""
        out: Dict[str, Any] = {
            "kind": self.kind,
            "channel": self.channel,
            "layer": self.layer,
            "axes": {name: axis.to_dict() for name, axis in sorted(self.axes.items())},
        }
        if not self.held_out:
            out["weight"] = self.weight
        return out


@dataclass
class World:
    """A declared stimulus world (spec §3). Build it with :func:`load_world`."""

    name: str
    description: str
    modality: str
    units: Dict[str, str]
    channels: List[str]
    classes: Dict[str, ClassSpec]
    held_out: Dict[str, ClassSpec]
    fixed: Dict[str, Dict[str, Any]]
    world_id: str = ""

    def class_spec(self, name: str) -> ClassSpec:
        """A class or held-out class by name."""
        if name in self.classes:
            return self.classes[name]
        if name in self.held_out:
            return self.held_out[name]
        raise ValueError(
            f"no class {name!r} in world {self.name!r}; classes: "
            f"{sorted(self.classes)}, held out: {sorted(self.held_out)}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """The normalised world; its canonical JSON is what :attr:`world_id` hashes."""
        return {
            "format": FORMAT,
            "name": self.name,
            "modality": self.modality,
            "units": dict(self.units),
            "channels": list(self.channels),
            "classes": {n: c.to_dict() for n, c in self.classes.items()},
            "held_out": {n: c.to_dict() for n, c in self.held_out.items()},
            "fixed_draws": {
                n: {"class": f["class"], **f["values"]} for n, f in self.fixed.items()
            },
        }

    def fixed_draw(self, name: str):
        """The named fixed draw (see :func:`sensoryforge.world.sampling.fixed_draw`)."""
        from sensoryforge.world.sampling import fixed_draw

        return fixed_draw(self, name)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "World":
        """Parse a world mapping (optionally under a top-level ``world:`` key)."""
        return _parse_world(data)


def load_world(source: Union[str, Path, Dict[str, Any], World]) -> World:
    """Load a world from a YAML file or a mapping; a :class:`World` passes through."""
    if isinstance(source, World):
        return source
    if isinstance(source, dict):
        return World.from_dict(source)
    path = Path(source)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping with a 'world:' key")
    return World.from_dict(data)


def _parse_world(data: Any) -> World:
    raw = data.get("world", data) if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        raise ValueError("a world is a mapping (optionally under a top-level 'world:' key)")
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        raise ValueError(f"world: unknown keys {sorted(unknown)}; allowed: {sorted(_TOP_KEYS)}")
    channels = [str(c) for c in (raw.get("channels") or ["value"])]
    if len(set(channels)) != len(channels) or not all(channels):
        raise ValueError(f"world.channels: give distinct non-empty names, got {channels}")
    defaults = {
        str(name): AxisSpec.from_dict(str(name), spec)
        for name, spec in (raw.get("defaults") or {}).items()
    }
    if not raw.get("classes"):
        raise ValueError("world.classes: declare at least one class")
    classes = {
        str(n): _parse_class(str(n), c, defaults, channels, held_out=False)
        for n, c in raw["classes"].items()
    }
    held = {
        str(n): _parse_class(str(n), c, defaults, channels, held_out=True)
        for n, c in (raw.get("held_out") or {}).items()
    }
    clash = set(classes) & set(held)
    if clash:
        raise ValueError(f"world.held_out: {sorted(clash)} are also classes")
    if sum(c.weight for c in classes.values()) <= 0:
        raise ValueError("world.classes: the weights sum to 0")
    fixed = {
        str(n): _parse_fixed(str(n), f, classes, held)
        for n, f in (raw.get("fixed_draws") or {}).items()
    }
    units = raw.get("units") or {"space": "mm", "time": "ms"}
    world = World(
        name=str(raw.get("name", "world")),
        description=str(raw.get("description", "")),
        modality=str(raw.get("modality", "")),
        units={str(k): str(v) for k, v in units.items()},
        channels=channels,
        classes=classes,
        held_out=held,
        fixed=fixed,
    )
    canonical = json.dumps(
        world.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    world.world_id = "w-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return world


def _parse_class(
    name: str,
    raw: Any,
    defaults: Dict[str, AxisSpec],
    channels: List[str],
    held_out: bool,
) -> ClassSpec:
    where = f"world.{'held_out' if held_out else 'classes'}.{name}"
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a mapping")
    unknown = set(raw) - _CLASS_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(_CLASS_KEYS)}")
    kind_name = str(raw.get("kind", "layered"))
    if kind_name not in CLASS_KINDS:
        raise ValueError(
            f"{where}: unknown class kind {kind_name!r}; known: {sorted(CLASS_KINDS)}"
        )
    kind = CLASS_KINDS[kind_name]
    raw_layer = raw.get("layer")
    layer = kind.normalise_layer(raw_layer, f"{where}.layer")
    if held_out:
        if "weight" in raw:
            raise ValueError(f"{where}: held-out classes take no weight")
        weight = 0.0
    else:
        weight = float(raw.get("weight", 1.0))
        if weight < 0:
            raise ValueError(f"{where}.weight: must be >= 0, got {weight}")
    channel = str(raw.get("channel", channels[0]))
    if channel not in channels:
        raise ValueError(
            f"{where}.channel: {channel!r} is not one of the world's channels {channels}"
        )

    axes: Dict[str, AxisSpec] = {}
    bindings: Dict[str, Ref] = {}

    def bind(axis_name: str, ref: Ref, axis: AxisSpec) -> None:
        for other, other_ref in list(bindings.items()):
            if other_ref == ref and other != axis_name:
                del bindings[other]
                del axes[other]
        lo, hi = kind.domain(ref, layer)
        axes[axis_name] = axis.with_domain(lo, hi)
        bindings[axis_name] = ref

    for field_name, value in kind.builtin_defaults(layer).items():
        constant = AxisSpec(name=field_name, form="constant", value=value)
        bind(field_name, kind.resolve(field_name, layer), constant)
    for axis_name, axis in defaults.items():
        try:
            ref = kind.resolve(axis_name, layer)
        except UnknownField:
            continue
        except AmbiguousField as exc:
            raise ValueError(f"{where}: world default {exc}") from None
        if kind.fixed_in_layer(ref, raw_layer):
            continue
        bind(axis_name, ref, axis)
    for axis_name, spec in (raw.get("axes") or {}).items():
        axis_name = str(axis_name)
        try:
            ref = kind.resolve(axis_name, layer)
        except (UnknownField, AmbiguousField) as exc:
            raise ValueError(f"{where}.axes: {exc}") from None
        bind(axis_name, ref, AxisSpec.from_dict(axis_name, spec))

    cls = ClassSpec(
        name=name,
        kind=kind_name,
        weight=weight,
        channel=channel,
        layer=layer,
        axes=dict(sorted(axes.items())),
        bindings=dict(sorted(bindings.items())),
        held_out=held_out,
    )
    kind.check(cls)
    return cls


def _parse_fixed(
    name: str, raw: Any, classes: Dict[str, ClassSpec], held: Dict[str, ClassSpec]
) -> Dict[str, Any]:
    where = f"world.fixed_draws.{name}"
    if not isinstance(raw, dict) or "class" not in raw:
        raise ValueError(f"{where}: needs a 'class'")
    class_name = str(raw["class"])
    spec = classes.get(class_name) or held.get(class_name)
    if spec is None:
        raise ValueError(f"{where}: unknown class {class_name!r}")
    values: Dict[str, Any] = {}
    for key, value in raw.items():
        if key == "class":
            continue
        try:
            ref = spec.kind_obj.resolve(str(key), spec.layer)
        except (UnknownField, AmbiguousField) as exc:
            raise ValueError(f"{where}: {exc}") from None
        bound = next((n for n, r in spec.bindings.items() if r == ref), None)
        if bound is None:
            raise ValueError(
                f"{where}: {key!r} is not an axis of class {class_name!r} "
                f"(its axes: {sorted(spec.axes)})"
            )
        values[bound] = plain(value)
    return {"class": class_name, "values": dict(sorted(values.items()))}
```

- [ ] **Step 6: Run the tests**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_world_schema.py tests/unit/test_world_kinds.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add sensoryforge/world/kinds.py sensoryforge/world/schema.py tests/fixtures/worlds/tactile_small.yml tests/unit/test_world_schema.py tests/unit/test_world_kinds.py
git commit -m "feat(world): world schema with registered class kinds

A world is classes x axes x distributions with shared defaults. Axis names
bind to episode fields, amplitude, placement, then shape/pattern/modulation
fields by bare name or dotted path; ambiguous or unknown names are errors.
A class's own axes beat its layer's fixed fields, which beat world
defaults. Class kinds are registered (layered and quiet built in). The
world id hashes the normalised world, description excluded.

Refs: D-94c08c7

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Sampling — draws, fixed draws, sessions

**Files:**
- Create: `sensoryforge/world/sampling.py`, `tests/unit/test_world_sampling.py`

**Interfaces:**
- Consumes: `rng.draw_seeds`, `rng.uniforms`, `rng.seed53` (Task 2); `World`, `ClassSpec` (Task 4).
- Produces: `Draw(world, class_name, values, seed=None, index=None, draw_seed=None, sampling="declared", out_of_range=())` (frozen) with `spec`, `end_ms`, `timeline`, `to_layer()`, `to_dict()`, `Draw.from_dict(data, world)`; `sample(world, n=None, seed=0, *, indices=None, classes=None) -> List[Draw]`; `fixed_draw(world, name) -> Draw`; `Session(world, seed, index, session_seed, duration_ms, items)` (frozen; `items: Tuple[Tuple[float, Draw], ...]`) with `end_ms`, `truncated`, `contact_ms`, `quiet_fraction`, `to_dict()`, `Session.from_dict(data, world)`; `session(world, duration_ms, seed, index=0) -> Session`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_world_sampling.py`:

```python
"""Sampling: independence from n, weights, ranges, records, fixed draws, sessions."""

import json
from pathlib import Path

import pytest

from sensoryforge.world.sampling import Draw, Session, sample, session
from sensoryforge.world.schema import World, load_world

WORLD = load_world(
    Path(__file__).resolve().parents[1] / "fixtures" / "worlds" / "tactile_small.yml"
)


def test_draw_i_does_not_depend_on_n():
    many = sample(WORLD, n=200, seed=7)
    some = sample(WORLD, indices=[5, 150], seed=7)
    assert many[5].to_dict() == some[0].to_dict()
    assert many[150].to_dict() == some[1].to_dict()
    assert many[5].index == 5 and many[5].seed == 7


def test_exactly_one_of_n_and_indices():
    with pytest.raises(ValueError, match="exactly one of n or indices"):
        sample(WORLD, n=3, indices=[1], seed=0)


def test_class_frequencies_follow_the_weights():
    draws = sample(WORLD, n=20000, seed=1)
    total = sum(c.weight for c in WORLD.classes.values())
    for name, cls in WORLD.classes.items():
        share = sum(d.class_name == name for d in draws) / len(draws)
        assert abs(share - cls.weight / total) < 0.015, name


def test_values_lie_in_their_ranges_and_constants_are_kept():
    for d in sample(WORLD, n=500, seed=2):
        assert set(d.values) == set(d.spec.axes)
        for name, axis in d.spec.axes.items():
            assert axis.contains(d.values[name]), (d.class_name, name, d.values[name])


def test_restricting_classes_reaches_held_out_ones():
    draws = sample(WORLD, n=50, seed=3, classes=["gratings"])
    assert {d.class_name for d in draws} == {"gratings"}


def test_records_round_trip_through_json():
    for d in sample(WORLD, n=30, seed=4):
        record = json.loads(json.dumps(d.to_dict()))
        assert Draw.from_dict(record, WORLD).to_dict() == d.to_dict()
        assert record["end_ms"] == pytest.approx(d.end_ms)
        assert record["timeline"][-1][2] == pytest.approx(d.end_ms)


def test_a_record_from_another_world_is_refused():
    record = sample(WORLD, n=1, seed=0)[0].to_dict()
    record["world_id"] = "w-000000000000"
    with pytest.raises(ValueError, match="belongs to world"):
        Draw.from_dict(record, WORLD)


def test_fixed_draws_take_midpoints_and_flag_out_of_range():
    h = WORLD.fixed_draw("braille_H")
    assert h.values["dots"] == "125" and h.values["hold_ms"] == 40
    assert h.values["touch_ms"] == pytest.approx(10.0)  # midpoint of [5, 15]
    assert h.out_of_range == () and h.sampling == "fixed" and h.seed is None
    wide = WORLD.fixed_draw("wide_dot")
    assert wide.out_of_range == ("sigma_mm",)


def test_a_session_lays_draws_end_to_end():
    s = session(WORLD, duration_ms=1000.0, seed=11, index=0)
    assert s.items[0][0] == 0.0
    for (a, da), (b, _) in zip(s.items, s.items[1:]):
        assert b == pytest.approx(a + da.end_ms)
    last_start, last = s.items[-1]
    assert last_start < 1000.0 <= last_start + last.end_ms
    assert 0.0 < s.quiet_fraction < 1.0
    assert s.end_ms == 1000.0


def test_sessions_are_reproducible_and_differ_by_index():
    a = session(WORLD, 500.0, seed=11, index=0)
    b = session(WORLD, 500.0, seed=11, index=0)
    c = session(WORLD, 500.0, seed=11, index=1)
    assert a.to_dict() == b.to_dict() and a.to_dict() != c.to_dict()
    record = json.loads(json.dumps(a.to_dict()))
    assert Session.from_dict(record, WORLD).to_dict() == a.to_dict()


def test_a_zero_length_draw_stops_a_session():
    world = World.from_dict(
        {"world": {"classes": {"q": {"kind": "quiet", "axes": {"quiet_ms": {"value": 0}}}}}}
    )
    with pytest.raises(ValueError, match="zero length"):
        session(world, 100.0, seed=0)
```

- [ ] **Step 2: Run to see them fail**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_world_sampling.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sensoryforge.world.sampling'`.

- [ ] **Step 3: Implement `sensoryforge/world/sampling.py`**

```python
"""Sampling a world: draws, fixed draws and sessions (spec §4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sensoryforge.world import rng
from sensoryforge.world.schema import ClassSpec, World

_CONTACT_PHASES = {"touch", "hold", "slide", "release"}


@dataclass(frozen=True)
class Draw:
    """One draw from a world: a class and a value for every axis it binds.

    Attributes:
        world: The world it belongs to (not part of its record).
        class_name: The class drawn.
        values: Every bound axis name -> its value, constants included.
        seed: The sampling seed (``None`` for a fixed draw).
        index: The draw's index under that seed.
        draw_seed: ``H(seed, index)``; every random choice derives from it.
        sampling: ``declared``, ``stratified``, ``probe`` or ``fixed``.
        out_of_range: Axes set outside their declared range.
    """

    world: World = field(repr=False, compare=False)
    class_name: str
    values: Dict[str, Any]
    seed: Optional[int] = None
    index: Optional[int] = None
    draw_seed: Optional[int] = None
    sampling: str = "declared"
    out_of_range: Tuple[str, ...] = ()

    @property
    def spec(self) -> ClassSpec:
        """The draw's class."""
        return self.world.class_spec(self.class_name)

    @property
    def end_ms(self) -> float:
        """When the draw ends (the last release; ``quiet_ms`` for quiet), ms."""
        return float(self.spec.kind_obj.end_ms(self.values))

    @property
    def timeline(self) -> List[List[Any]]:
        """``[[phase, start_ms, end_ms], ...]`` from time 0, the entry's start."""
        return self.spec.kind_obj.timeline(self.values)

    def to_layer(self) -> Optional[Dict[str, Any]]:
        """The draw as an ordinary layered layer dict (``None`` for quiet)."""
        return self.spec.kind_obj.to_layer(self.spec, self.values)

    def to_dict(self) -> Dict[str, Any]:
        """The JSON-ready record (spec §4.4)."""
        return {
            "world_id": self.world.world_id,
            "seed": self.seed,
            "index": self.index,
            "draw_seed": self.draw_seed,
            "class": self.class_name,
            "sampling": self.sampling,
            "values": dict(sorted(self.values.items())),
            "timeline": self.timeline,
            "end_ms": self.end_ms,
            "out_of_range": list(self.out_of_range),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], world: World) -> "Draw":
        """Rebuild a draw from its record and the world it came from."""
        if data.get("world_id") != world.world_id:
            raise ValueError(
                f"this draw belongs to world {data.get('world_id')!r}, not {world.world_id!r}"
            )
        spec = world.class_spec(data["class"])
        missing = set(spec.axes) - set(data["values"])
        if missing:
            raise ValueError(f"draw record lacks values for {sorted(missing)}")
        return cls(
            world=world,
            class_name=data["class"],
            values=dict(data["values"]),
            seed=data.get("seed"),
            index=data.get("index"),
            draw_seed=data.get("draw_seed"),
            sampling=data.get("sampling", "declared"),
            out_of_range=tuple(data.get("out_of_range", ())),
        )


def sample(
    world: World,
    n: Optional[int] = None,
    seed: int = 0,
    *,
    indices: Optional[Iterable[int]] = None,
    classes: Optional[Sequence[str]] = None,
) -> List[Draw]:
    """Draws from the world's declared distribution (spec §4.2).

    Draw ``i`` depends only on ``(world, seed, i)`` and on ``classes``: ask
    for ``indices=[i]`` to regenerate it alone.

    Args:
        world: The world.
        n: How many draws (indices ``0 .. n-1``); or give ``indices``.
        seed: The sampling seed.
        indices: Which draw indices to produce.
        classes: Restrict to these classes (held-out ones allowed); their
            weights are renormalised (equal if they sum to 0).

    Returns:
        One :class:`Draw` per index, in order.
    """
    if (n is None) == (indices is None):
        raise ValueError("give exactly one of n or indices")
    idx = (
        np.arange(int(n), dtype=np.int64)
        if indices is None
        else np.asarray(list(indices), dtype=np.int64)
    )
    pool = [world.class_spec(c) for c in classes] if classes else list(world.classes.values())
    if not pool:
        raise ValueError("no classes to sample from")
    seeds = rng.draw_seeds(seed, idx)
    weights = np.array([c.weight for c in pool], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones(len(pool))
    cum = np.cumsum(weights) / weights.sum()
    chosen = np.minimum(
        np.searchsorted(cum, rng.uniforms(seeds, "class"), side="right"), len(pool) - 1
    )
    draws: List[Optional[Draw]] = [None] * idx.size
    for ci, cls in enumerate(pool):
        rows = np.nonzero(chosen == ci)[0]
        if rows.size == 0:
            continue
        sub = seeds[rows]
        columns = {
            name: axis.sample(rng.uniforms(sub, name)) if axis.is_random else [axis.value] * rows.size
            for name, axis in cls.axes.items()
        }
        for j, row in enumerate(rows.tolist()):
            draws[row] = Draw(
                world=world,
                class_name=cls.name,
                values={name: columns[name][j] for name in cls.axes},
                seed=int(seed),
                index=int(idx[row]),
                draw_seed=int(seeds[row]),
                sampling="declared",
            )
    return draws  # type: ignore[return-value]


def fixed_draw(world: World, name: str) -> Draw:
    """A named fixed draw: its given values, every other axis at its midpoint (spec §3.5)."""
    if name not in world.fixed:
        raise ValueError(f"no fixed draw {name!r}; the world has {sorted(world.fixed)}")
    entry = world.fixed[name]
    spec = world.class_spec(entry["class"])
    values = {axis_name: axis.midpoint() for axis_name, axis in spec.axes.items()}
    outside = []
    for axis_name, value in entry["values"].items():
        values[axis_name] = value
        if not spec.axes[axis_name].contains(value):
            outside.append(axis_name)
    return Draw(
        world=world,
        class_name=spec.name,
        values=values,
        sampling="fixed",
        out_of_range=tuple(sorted(outside)),
    )


@dataclass(frozen=True)
class Session:
    """Draws laid end to end over ``duration_ms`` (spec §4.5).

    Attributes:
        world: The world (not part of the record).
        seed: The sampling seed.
        index: The session's index under that seed.
        session_seed: ``H(seed, index)``; draw ``k`` is ``sample(world, indices=[k], seed=session_seed)``.
        duration_ms: The session's length.
        items: ``((start_ms, draw), ...)``; the last draw may run past the end.
    """

    world: World = field(repr=False, compare=False)
    seed: int
    index: int
    session_seed: int
    duration_ms: float
    items: Tuple[Tuple[float, Draw], ...]

    @property
    def end_ms(self) -> float:
        """The session's length, ms."""
        return self.duration_ms

    @property
    def truncated(self) -> bool:
        """True when the last draw is cut at ``duration_ms``."""
        start, last = self.items[-1]
        return start + last.end_ms > self.duration_ms

    @property
    def contact_ms(self) -> float:
        """Time in contact (touch, hold, slide, release) within the session, ms."""
        total = 0.0
        for start, draw in self.items:
            for phase, a, b in draw.timeline:
                if phase in _CONTACT_PHASES:
                    lo = min(start + a, self.duration_ms)
                    hi = min(start + b, self.duration_ms)
                    total += max(hi - lo, 0.0)
        return total

    @property
    def quiet_fraction(self) -> float:
        """The share of the session with nothing touching."""
        return 1.0 - self.contact_ms / self.duration_ms

    def to_dict(self) -> Dict[str, Any]:
        """The JSON-ready record."""
        return {
            "world_id": self.world.world_id,
            "sampling": "session",
            "seed": self.seed,
            "index": self.index,
            "session_seed": self.session_seed,
            "duration_ms": self.duration_ms,
            "items": [[start, draw.to_dict()] for start, draw in self.items],
            "truncated": self.truncated,
            "quiet_fraction": self.quiet_fraction,
            "end_ms": self.end_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], world: World) -> "Session":
        """Rebuild a session from its record and its world."""
        if data.get("world_id") != world.world_id:
            raise ValueError(
                f"this session belongs to world {data.get('world_id')!r}, not {world.world_id!r}"
            )
        items = tuple((float(start), Draw.from_dict(d, world)) for start, d in data["items"])
        return cls(
            world=world,
            seed=int(data["seed"]),
            index=int(data["index"]),
            session_seed=int(data["session_seed"]),
            duration_ms=float(data["duration_ms"]),
            items=items,
        )


def session(world: World, duration_ms: float, seed: int, index: int = 0) -> Session:
    """Draws from the world laid end to end until ``duration_ms`` is filled.

    Raises:
        ValueError: If a draw has zero length (the session would never fill).
    """
    if duration_ms <= 0:
        raise ValueError(f"duration_ms must be > 0, got {duration_ms}")
    session_seed = rng.seed53(seed, index)
    items: List[Tuple[float, Draw]] = []
    t = 0.0
    k = 0
    while t < duration_ms:
        draw = sample(world, indices=[k], seed=session_seed)[0]
        if draw.end_ms <= 0:
            raise ValueError(
                f"session draw {k} (class {draw.class_name!r}) has zero length; "
                "give its class a positive duration"
            )
        items.append((t, draw))
        t += draw.end_ms
        k += 1
    return Session(
        world=world,
        seed=int(seed),
        index=int(index),
        session_seed=int(session_seed),
        duration_ms=float(duration_ms),
        items=tuple(items),
    )
```

- [ ] **Step 4: Run the tests**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_world_sampling.py tests/unit/test_world_schema.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sensoryforge/world/sampling.py tests/unit/test_world_sampling.py
git commit -m "feat(world): deterministic draws, fixed draws and sessions

sample(world, n | indices, seed, classes) gives JSON-ready draw records
whose every value derives from H(seed, i), so draw i is the same however
many are asked for. Fixed draws take their given values and axis
midpoints, flagging values outside the declared ranges. A session lays
draws end to end; its quiet comes from lead-ins and quiet draws.

Refs: D-37c5247

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Rendering — `Canvas`, `render`, `render_movie`, and `layered` groups

**Files:**
- Create: `sensoryforge/world/render.py`, `tests/unit/test_world_render.py`
- Modify: `sensoryforge/world/kinds.py` (add `LayeredKind.render_group` and `_group_params`), `sensoryforge/world/__init__.py` (public API)

**Interfaces:**
- Consumes: `contact_terms`, `span_progress` (Task 1); `kernel.SHAPE_KINDS`, `kernel.MODULATION_KINDS`, `kernel.pattern_batch`, `kernel.motion_offsets` (Task 3); `LayeredKind.part_values`, `EPISODE_FIELDS` (Task 4); `Draw`, `Session` (Task 5); `stimulus_canvas` (`sensoryforge/stimuli/canvas.py`).
- Produces: `Canvas(xx, yy)` (float64, CPU) with `shape`, `Canvas.from_grid(rows, cols, spacing_mm, center_mm=(0.0, 0.0))`, `Canvas.from_grid_config(grid_cfg)`, `Canvas.from_points(xy)`; `movie_times(dt_ms, duration_ms) -> Tensor[T] float64` (`t_k = k * dt_ms`); `render(items, canvas, times_ms, *, dtype=torch.float32, device="cpu", world=None, max_elements=2**23) -> Tensor [n, K, *S]` (or `[n, K, C, *S]`); `render_movie(item, canvas, dt_ms, duration_ms, *, dtype=torch.float32, device="cpu", world=None) -> Tensor [T, *S]`. Package `sensoryforge.world` exports these plus `load_world`, `World`, `ClassSpec`, `Draw`, `Session`, `sample`, `session`, `fixed_draw`, `AxisSpec`, and the `register_*` functions.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_world_render.py`:

```python
"""Rendering: equal to layered; batch invariance; windows; grids; quiet; sessions; devices."""

import math
from pathlib import Path

import pytest
import torch

from sensoryforge.config.schema import GridConfig
from sensoryforge.stimuli.canvas import stimulus_canvas
from sensoryforge.stimuli.layered import render_layers
from sensoryforge.world import (
    Canvas,
    World,
    load_world,
    movie_times,
    render,
    render_movie,
    sample,
    session,
)

WORLD = load_world(
    Path(__file__).resolve().parents[1] / "fixtures" / "worlds" / "tactile_small.yml"
)
CANVAS = Canvas.from_grid(rows=24, cols=24, spacing_mm=0.05)
DT = 1.0


def _layered(draw, total_ms):
    xx, yy = CANVAS.xx.float(), CANVAS.yy.float()
    return render_layers([draw.to_layer()], xx, yy, dt_ms=DT, total_ms=total_ms).double()


@pytest.mark.parametrize(
    "class_name", sorted(n for n in WORLD.classes if n != "quiet") + ["gratings"]
)
def test_world_render_equals_layered(class_name):
    draws = sample(WORLD, n=3, seed=21, classes=[class_name])
    total = math.ceil(max(d.end_ms for d in draws)) + 5.0
    frames = render(draws, CANVAS, movie_times(DT, total), dtype=torch.float64)
    for i, draw in enumerate(draws):
        torch.testing.assert_close(frames[i], _layered(draw, total), atol=1e-5, rtol=0)


def test_draw_i_alone_equals_draw_i_in_a_batch_bit_for_bit():
    draws = sample(WORLD, n=40, seed=8)
    times = movie_times(DT, 120.0)
    batch = render(draws, CANVAS, times, dtype=torch.float64)
    one_per_chunk = render(draws, CANVAS, times, dtype=torch.float64, max_elements=1)
    assert torch.equal(batch, one_per_chunk)
    for i in (0, 7, 39):
        assert torch.equal(render([draws[i]], CANVAS, times, dtype=torch.float64)[0], batch[i])


def test_windows_equal_movie_frames_bit_for_bit():
    draws = sample(WORLD, n=12, seed=9)
    movie = render(draws, CANVAS, movie_times(DT, 120.0), dtype=torch.float64)
    steps = [(k - 8, k, k + 8) for k in range(10, 22)]
    windows = render(
        draws, CANVAS, torch.tensor(steps, dtype=torch.float64), dtype=torch.float64
    )
    for i, triple in enumerate(steps):
        for j, step in enumerate(triple):
            assert torch.equal(windows[i, j], movie[i, step])


def test_a_draw_agrees_on_40x40_and_80x80_where_they_overlap():
    small = Canvas.from_grid(40, 40, 0.15)
    large = Canvas.from_grid(80, 80, 0.15)
    torch.testing.assert_close(small.xx, large.xx[20:60, 20:60], atol=1e-12, rtol=0)
    draws = sample(WORLD, n=6, seed=10)
    times = torch.tensor([30.0, 60.0], dtype=torch.float64)
    a = render(draws, small, times, dtype=torch.float64)
    b = render(draws, large, times, dtype=torch.float64)[:, :, 20:60, 20:60]
    torch.testing.assert_close(a, b, atol=1e-12, rtol=0)


def test_quiet_is_exactly_zero():
    draws = sample(WORLD, n=200, seed=12)
    frames = render(draws, CANVAS, movie_times(DT, 220.0), dtype=torch.float64)
    for i, d in enumerate(draws):
        if d.class_name == "quiet":
            assert torch.count_nonzero(frames[i]) == 0
            continue
        lead = int(d.values["delay_ms"])  # frames k < delay_ms
        assert torch.count_nonzero(frames[i, :lead]) == 0
        end = int(math.ceil(d.end_ms + 1e-9))
        assert torch.count_nonzero(frames[i, end:]) == 0
        if d.class_name == "twice":
            pause_from = d.values["delay_ms"] + sum(
                d.values[f] for f in ("touch_ms", "hold_ms", "slide_ms", "release_ms")
            )
            first, last = math.ceil(pause_from + 1e-9), math.floor(pause_from + d.values["pause_ms"] - 1e-9)
            assert torch.count_nonzero(frames[i, first : last + 1]) == 0


def test_a_session_renders_each_draw_from_its_start():
    s = session(WORLD, duration_ms=400.0, seed=13)
    times = movie_times(DT, 400.0)
    frames = render([s], CANVAS, times, dtype=torch.float64)[0]
    assert torch.equal(frames, render_movie(s, CANVAS, DT, 400.0, dtype=torch.float64))
    for j, (start, draw) in enumerate(s.items):
        stop = s.items[j + 1][0] if j + 1 < len(s.items) else 400.0
        ks = [k for k in range(400) if start <= k < stop]
        if not ks:
            continue
        local = torch.tensor([k - start for k in ks], dtype=torch.float64)
        alone = render([draw], CANVAS, local, dtype=torch.float64)[0]
        assert torch.equal(frames[ks], alone)


def test_records_render_with_their_world():
    draws = sample(WORLD, n=5, seed=14)
    times = torch.tensor([20.0], dtype=torch.float64)
    direct = render(draws, CANVAS, times, dtype=torch.float64)
    from_records = render([d.to_dict() for d in draws], CANVAS, times, dtype=torch.float64, world=WORLD)
    assert torch.equal(direct, from_records)
    with pytest.raises(ValueError, match="needs world="):
        render([draws[0].to_dict()], CANVAS, times)


def test_random_pattern_seed_axis_matches_layered():
    world = World.from_dict({"world": {"classes": {"bumps": {
        "layer": {"shape": {"kind": "gaussian", "sigma_mm": 0.1},
                  "pattern": {"kind": "random", "count": 5, "width_mm": 1.0, "height_mm": 1.0}},
        "axes": {"seed": {"range": [0, 1000], "int": True}, "hold_ms": {"value": 20}},
    }}}})
    draws = sample(world, n=4, seed=1)
    assert len({d.values["seed"] for d in draws}) > 1
    frames = render(draws, CANVAS, movie_times(DT, 25.0), dtype=torch.float64)
    for i, draw in enumerate(draws):
        torch.testing.assert_close(frames[i], _layered(draw, 25.0), atol=1e-5, rtol=0)


def test_a_string_axis_splits_groups_without_changing_results():
    world = World.from_dict({"world": {"classes": {"bars": {
        "layer": {"shape": {"kind": "bar", "length_mm": 0.0}},
        "axes": {"profile": {"values": ["gaussian", "flat"]},
                 "width_mm": {"range": [0.1, 0.3]}, "hold_ms": {"value": 20}},
    }}}})
    draws = sample(world, n=10, seed=2)
    assert {d.values["profile"] for d in draws} == {"gaussian", "flat"}
    times = movie_times(DT, 25.0)
    batch = render(draws, CANVAS, times, dtype=torch.float64)
    for i, draw in enumerate(draws):
        assert torch.equal(batch[i], render([draw], CANVAS, times, dtype=torch.float64)[0])


def test_a_multi_channel_world_draws_each_class_on_its_plane():
    world = World.from_dict({"world": {
        "channels": ["pressure", "vibration"],
        "classes": {
            "press": {"layer": {"shape": {"kind": "gaussian"}}, "axes": {"hold_ms": {"value": 20}}},
            "buzz": {"channel": "vibration",
                     "layer": {"shape": {"kind": "gaussian"},
                               "modulation": {"kind": "sine", "frequency_hz": 100.0}},
                     "axes": {"hold_ms": {"value": 20}}},
        },
    }})
    draws = sample(world, n=20, seed=3)
    frames = render(draws, CANVAS, [2.0], dtype=torch.float64)
    assert frames.shape == (20, 1, 2, 24, 24)
    for i, draw in enumerate(draws):
        on = 0 if draw.class_name == "press" else 1
        assert float(frames[i, 0, on].abs().max()) > 0.0
        assert float(frames[i, 0, 1 - on].abs().max()) == 0.0


def test_float64_on_mps_is_refused():
    with pytest.raises(ValueError, match="MPS has no float64"):
        render(sample(WORLD, n=1, seed=0), CANVAS, [0.0], dtype=torch.float64, device="mps")


def test_canvas_from_a_grid_config_spans_the_stimulus_canvas():
    grid = GridConfig(name="g", rows=8, cols=8, spacing=0.15)
    canvas = Canvas.from_grid_config(grid)
    torch.testing.assert_close(canvas.xx, stimulus_canvas(grid).xx.double(), atol=1e-6, rtol=0)
    assert canvas.shape == (8, 8) and canvas.xx.dtype == torch.float64
    assert Canvas.from_points(torch.tensor([[0.0, 0.0], [0.1, 0.2]])).shape == (2,)
```

- [ ] **Step 2: Run to see them fail**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_world_render.py -q`
Expected: FAIL — `ImportError: cannot import name 'Canvas' from 'sensoryforge.world'`.

- [ ] **Step 3: Add `render_group` to `LayeredKind`** (in `sensoryforge/world/kinds.py`)

Add the import `from sensoryforge.stimuli.episode import contact_terms, span_progress` and, above `class LayeredKind`, the helper:

```python
def _group_params(dicts, view, dtype, device) -> Dict[str, Any]:
    """Per-draw parameter dicts -> one dict: numbers become tensors of shape ``view``.

    Non-numeric values (strings, bools) are equal across a group (the
    renderer groups by them) and pass through as plain values.
    """
    out: Dict[str, Any] = {}
    for key in dicts[0]:
        if key == "kind":
            continue
        values = [d[key] for d in dicts]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            out[key] = torch.tensor(
                [float(v) for v in values], dtype=dtype, device=device
            ).view(view)
        else:
            out[key] = values[0]
    return out
```

Then add this method to `LayeredKind`:

```python
    def render_group(self, spec, draws, X, Y, times):
        """Frames ``[g, K, *S]``: amplitude x envelope x modulation x sum of shapes.

        Args:
            spec: The class.
            draws: The group's draws (same class, same non-numeric values).
            X, Y: Canvas coordinates ``[*S]`` in mm, in the output dtype/device.
            times: ``[g, K]`` ms since each draw's start (negative: before it).
        """
        dtype, device = X.dtype, X.device
        g, k_count = times.shape
        ones = (1,) * X.ndim
        lead = (g, k_count) + ones
        per_draw = (g, 1) + ones

        def column(name: str) -> torch.Tensor:
            values = [float(d.values[name]) for d in draws]
            return torch.tensor(values, dtype=dtype, device=device).view(g, 1)

        ep = {name: column(name) for name in EPISODE_FIELDS}
        env, tau, k, local = contact_terms(
            times,
            ep["delay_ms"],
            ep["touch_ms"],
            ep["hold_ms"],
            ep["slide_ms"],
            ep["release_ms"],
            ep["contacts"],
            ep["pause_ms"],
        )
        parts = [self.part_values(spec, d.values) for d in draws]
        modulation = kernel.MODULATION_KINDS[spec.layer["modulation"]["kind"]]
        if modulation.fn is not None:
            params = _group_params([p["modulation"] for p in parts], (g, 1), dtype, device)
            env = env * modulation.fn(tau, params)
        progress = span_progress(
            tau, k, local, ep["contacts"], ep["touch_ms"] + ep["hold_ms"], ep["slide_ms"]
        )
        motion = spec.layer.get("motion")
        if motion is None:
            travel = ep["speed_mm_per_ms"] * ep["contacts"] * ep["slide_ms"]
            theta = torch.deg2rad(ep["direction_deg"])
            off_x = progress * (travel * torch.cos(theta))
            off_y = progress * (travel * torch.sin(theta))
        else:
            offsets = kernel.motion_offsets(motion, progress)
            off_x, off_y = offsets[..., 0], offsets[..., 1]
        off_x, off_y = off_x.reshape(lead), off_y.reshape(lead)

        shape = kernel.SHAPE_KINDS[spec.layer["shape"]["kind"]]
        params = _group_params([p["shape"] for p in parts], per_draw, dtype, device)
        amplitude = params.pop("amplitude")
        if shape.unbounded:
            total = shape.fn(X - off_x, Y - off_y, params)
        else:
            pos, scales = kernel.pattern_batch(
                spec.layer["pattern"]["kind"], [p["pattern"] for p in parts], dtype, device
            )
            total = torch.zeros((g, k_count) + tuple(X.shape), dtype=dtype, device=device)
            for slot in range(pos.shape[1]):
                px = pos[:, slot, 0].reshape(per_draw)
                py = pos[:, slot, 1].reshape(per_draw)
                weight = scales[:, slot].reshape(per_draw)
                total = total + weight * shape.fn(X - px - off_x, Y - py - off_y, params)
        return amplitude * env.reshape(lead) * total
```

- [ ] **Step 4: Implement `sensoryforge/world/render.py`**

```python
"""Rendering draws at arbitrary times on arbitrary coordinates (spec §5).

``render`` groups draws by class (and by any non-numeric values, such as a
profile axis), evaluates each group as one broadcast computation, chunked to
a memory budget, and adds each group's frames into its rows. Every operation
is elementwise or a fixed-order sum, so a draw renders to the same bits alone,
in a batch, or in any chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch

from sensoryforge.config.schema import GridConfig
from sensoryforge.stimuli.canvas import stimulus_canvas
from sensoryforge.world.sampling import Draw, Session
from sensoryforge.world.schema import World

Item = Union[Draw, Session]

#: Elements per chunk of one group's ``[g, K, *S]`` output (float64: 64 MB).
DEFAULT_MAX_ELEMENTS = 2**23


@dataclass(frozen=True)
class Canvas:
    """Where to render: x and y coordinates in mm (float64, CPU), any common shape ``S``."""

    xx: torch.Tensor
    yy: torch.Tensor

    def __post_init__(self) -> None:
        xx = torch.as_tensor(self.xx, dtype=torch.float64).detach().cpu()
        yy = torch.as_tensor(self.yy, dtype=torch.float64).detach().cpu()
        if xx.shape != yy.shape:
            raise ValueError(f"xx {list(xx.shape)} and yy {list(yy.shape)} differ in shape")
        object.__setattr__(self, "xx", xx)
        object.__setattr__(self, "yy", yy)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The coordinate shape ``S`` frames take."""
        return tuple(self.xx.shape)

    @classmethod
    def from_grid(
        cls,
        rows: int,
        cols: int,
        spacing_mm: float,
        center_mm: Tuple[float, float] = (0.0, 0.0),
    ) -> "Canvas":
        """A centred ``rows x cols`` lattice, ``indexing="ij"`` (dim 0 is x), in float64."""
        if rows < 1 or cols < 1 or spacing_mm <= 0:
            raise ValueError(f"need rows, cols >= 1 and spacing > 0; got {rows}, {cols}, {spacing_mm}")
        cx, cy = float(center_mm[0]), float(center_mm[1])
        half_x = (rows - 1) * spacing_mm / 2.0
        half_y = (cols - 1) * spacing_mm / 2.0
        x = torch.linspace(cx - half_x, cx + half_x, rows, dtype=torch.float64)
        y = torch.linspace(cy - half_y, cy + half_y, cols, dtype=torch.float64)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        return cls(xx, yy)

    @classmethod
    def from_grid_config(cls, grid_cfg: GridConfig) -> "Canvas":
        """The canvas :func:`~sensoryforge.stimuli.canvas.stimulus_canvas` builds, in float64."""
        if grid_cfg.coords_file:
            old = stimulus_canvas(grid_cfg)
            return cls(old.xx, old.yy)
        return cls.from_grid(
            grid_cfg.rows or 40,
            grid_cfg.cols or 40,
            grid_cfg.spacing,
            (grid_cfg.center_x, grid_cfg.center_y),
        )

    @classmethod
    def from_points(cls, xy: Any) -> "Canvas":
        """Scattered points ``[M, 2]`` (x, y) in mm; frames are ``[..., M]``."""
        xy = torch.as_tensor(xy, dtype=torch.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"points must be [M, 2], got {list(xy.shape)}")
        return cls(xy[:, 0], xy[:, 1])


def movie_times(dt_ms: float, duration_ms: float) -> torch.Tensor:
    """``t_k = k * dt_ms`` for ``k < round(duration_ms / dt_ms)`` (at least 1), float64."""
    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be > 0, got {dt_ms}")
    steps = max(int(round(float(duration_ms) / float(dt_ms))), 1)
    return torch.arange(steps, dtype=torch.float64) * float(dt_ms)


def _as_item(item: Any, world: Optional[World]) -> Item:
    if isinstance(item, (Draw, Session)):
        return item
    if isinstance(item, dict):
        if world is None:
            raise ValueError("rendering a draw record needs world=")
        if "items" in item:
            return Session.from_dict(item, world)
        return Draw.from_dict(item, world)
    raise ValueError(f"cannot render {type(item).__name__}; give Draw, Session or their records")


def _jobs(items: Sequence[Item], times: torch.Tensor) -> Iterator[Tuple[int, Draw, torch.Tensor]]:
    """``(row, draw, local times)``: a session becomes one job per draw, windowed."""
    for row, item in enumerate(items):
        if isinstance(item, Session):
            starts = [start for start, _ in item.items] + [item.duration_ms]
            for j, (start, draw) in enumerate(item.items):
                stop = min(starts[j + 1], item.duration_ms)
                window = (times[row] >= start) & (times[row] < stop)
                local = torch.where(window, times[row] - start, torch.full_like(times[row], -1.0))
                yield row, draw, local
        else:
            yield row, item, times[row]


def _group_key(draw: Draw) -> Tuple[Any, ...]:
    spec = draw.spec
    fixed = tuple(
        sorted(
            (name, value)
            for name, value in draw.values.items()
            if isinstance(value, (str, bool)) and spec.bindings[name][0] in ("shape", "modulation")
        )
    )
    return (id(draw.world), draw.class_name, fixed)


def render(
    items: Sequence[Any],
    canvas: Canvas,
    times_ms: Any,
    *,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
    world: Optional[World] = None,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> torch.Tensor:
    """Render draws (or sessions) at arbitrary times (spec §5.1).

    Args:
        items: Draws, sessions, or their records (records need ``world``).
        canvas: Where to render.
        times_ms: ``[K]`` times shared by every item, or ``[n, K]`` per item;
            ms from each item's start. Times before 0 or after the item's end
            render exactly zero.
        dtype: ``torch.float32`` or ``torch.float64``.
        device: ``cpu``, ``cuda`` or ``mps`` (float32 only on MPS).
        world: The world records belong to.
        max_elements: Chunk budget per group, in output elements.

    Returns:
        ``[n, K, *S]``, or ``[n, K, C, *S]`` when the world has ``C > 1`` channels.

    Raises:
        ValueError: For a bad dtype, float64 on MPS, mismatched ``times_ms``,
            or items from worlds with different channels.
    """
    device = torch.device(device)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}")
    if dtype == torch.float64 and device.type == "mps":
        raise ValueError("MPS has no float64: render on cpu or cuda, or in float32")
    resolved = [_as_item(item, world) for item in items]
    n = len(resolved)
    times = torch.as_tensor(times_ms, dtype=torch.float64).detach().cpu()
    if times.ndim == 1:
        times = times.unsqueeze(0).expand(n, -1)
    if times.ndim != 2 or times.shape[0] != n:
        raise ValueError(f"times_ms must be [K] or [n, K] with n = {n}, got {list(times.shape)}")
    k_count = times.shape[1]
    channels = resolved[0].world.channels if resolved else ["value"]
    if any(item.world.channels != channels for item in resolved):
        raise ValueError("render: items come from worlds with different channels")
    multi = len(channels) > 1
    shape = (n, k_count, len(channels)) + canvas.shape if multi else (n, k_count) + canvas.shape
    out = torch.zeros(shape, dtype=dtype, device=device)
    if n == 0 or k_count == 0:
        return out

    groups: Dict[Tuple[Any, ...], List[Tuple[int, Draw, torch.Tensor]]] = {}
    for job in _jobs(resolved, times):
        groups.setdefault(_group_key(job[1]), []).append(job)
    X = canvas.xx.to(device=device, dtype=dtype)
    Y = canvas.yy.to(device=device, dtype=dtype)
    per_chunk = max(1, int(max_elements) // max(1, k_count * X.numel()))
    for members in groups.values():
        spec = members[0][1].spec
        target = out[:, :, channels.index(spec.channel)] if multi else out
        for start in range(0, len(members), per_chunk):
            chunk = members[start : start + per_chunk]
            local = torch.stack([m[2] for m in chunk]).to(device=device, dtype=dtype)
            frames = spec.kind_obj.render_group(spec, [m[1] for m in chunk], X, Y, local)
            rows = torch.tensor([m[0] for m in chunk], device=device)
            target.index_add_(0, rows, frames)
    return out


def render_movie(
    item: Any,
    canvas: Canvas,
    dt_ms: float,
    duration_ms: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
    world: Optional[World] = None,
) -> torch.Tensor:
    """One item's frames at ``t_k = k * dt_ms``: ``[T, *S]`` (or ``[T, C, *S]``)."""
    times = movie_times(dt_ms, duration_ms)
    return render([item], canvas, times, dtype=dtype, device=device, world=world)[0]
```

- [ ] **Step 5: Fill in the package API** — replace `sensoryforge/world/__init__.py` with:

```python
"""Declared stimulus worlds: sample them, render them, build data sets on them.

A world (``world:`` YAML) declares classes of stimuli, each a layered layer
whose fields are drawn from axes. :func:`sample` gives deterministic draw
records, :func:`render` evaluates any draws at any times on any coordinates,
and :mod:`sensoryforge.world.dataset` builds data sets on a world. See
``docs/user_guide/worlds.md`` and ``docs/reference/world_contract.md``.
"""

from sensoryforge.world.distributions import AxisSpec, register_distribution
from sensoryforge.world.kernel import register_modulation, register_pattern, register_shape
from sensoryforge.world.kinds import ClassKind, register_class_kind
from sensoryforge.world.schema import ClassSpec, World, load_world
from sensoryforge.world.sampling import Draw, Session, fixed_draw, sample, session
from sensoryforge.world.render import Canvas, movie_times, render, render_movie

__all__ = [
    "AxisSpec",
    "Canvas",
    "ClassKind",
    "ClassSpec",
    "Draw",
    "Session",
    "World",
    "fixed_draw",
    "load_world",
    "movie_times",
    "register_class_kind",
    "register_distribution",
    "register_modulation",
    "register_pattern",
    "register_shape",
    "render",
    "render_movie",
    "sample",
    "session",
]
```

- [ ] **Step 6: Run the world tests**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_world_render.py tests/unit/test_world_sampling.py tests/unit/test_world_schema.py tests/unit/test_world_kinds.py tests/unit/test_world_kernel.py -q`
Expected: all pass. If `test_world_render_equals_layered` fails for one class only, compare that class's `to_layer()` with `render_group`'s formula before loosening anything: the tolerance (1e-5) covers float32 time arithmetic, not a different formula.

- [ ] **Step 7: Commit**

```bash
git add sensoryforge/world/render.py sensoryforge/world/kinds.py sensoryforge/world/__init__.py tests/unit/test_world_render.py
git commit -m "feat(world): vectorised rendering of draws at arbitrary times

render(draws, canvas, times) evaluates each class group as one broadcast
computation, chunked to a memory budget, in float32 or float64 on CPU or
CUDA, on any coordinates (a grid, scattered points, PS's own mesh).
Pinned by tests: equal to layered within float32 tolerance, a draw alone
equals the same draw in a batch bit for bit, windows equal movie frames,
40x40 and 80x80 agree where they overlap, quiet is exactly zero.

Refs: D-cccbd6a

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Benchmark — 4.1 M triples at 40×40

**Files:**
- Create: `benchmarks/world_render.py`
- Modify: `docs/reference/benchmarks.md` (a "World renderer" section with the measured numbers)

**Interfaces:**
- Consumes: `World`, `sample`, `render`, `Canvas`, `rng` (Tasks 2–6).
- Produces: a printed timing; the result recorded in the docs.

- [ ] **Step 1: Write the benchmark**

`benchmarks/world_render.py`:

```python
"""Time the world renderer on pressure-simulation's RA design load (spec §5.6).

4.1 M triples (t - tau, t, t + tau) at 40x40, 0.15 mm, float64, in chunks of
8000 draws, from a four-class world shaped like pressure-simulation's today
(dots, edges, random braille cells, signed textures, always sliding at
0.02 mm/ms). Times a few chunks after a warm-up and extrapolates.

    conda run -n sensoryforge python benchmarks/world_render.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import torch

from sensoryforge.world import Canvas, World, render, sample
from sensoryforge.world import rng

WORLD = {
    "world": {
        "name": "ps_today_like",
        "defaults": {
            "hold_ms": {"value": 0},
            "slide_ms": {"value": 1000},
            "speed_mm_per_ms": {"value": 0.02},
            "direction_deg": {"range": [0, 360], "circular": True},
            "x_mm": {"range": [-2.925, 2.925]},
            "y_mm": {"range": [-2.925, 2.925]},
        },
        "classes": {
            "dots": {
                "layer": {"shape": {"kind": "gaussian"}},
                "axes": {"sigma_mm": {"range": [0.15, 0.45]}},
            },
            "edges": {
                "layer": {"shape": {"kind": "bar", "length_mm": 0}},
                "axes": {
                    "width_mm": {"range": [0.05, 0.15]},
                    "orientation_deg": {"range": [0, 180], "circular": True},
                },
            },
            "braille": {
                "layer": {
                    "shape": {"kind": "gaussian", "sigma_mm": 0.15},
                    "pattern": {"kind": "braille", "dot_spacing_mm": 0.35},
                },
                "axes": {"dots": {"dist": "braille_cells"}},
            },
            "textures": {
                "layer": {"shape": {"kind": "gabor", "signed": True}},
                "axes": {
                    "wavelength_mm": {"range": [0.3, 0.8]},
                    "sigma_mm": {"range": [0.3, 0.6]},
                    "orientation_deg": {"range": [0, 360], "circular": True},
                    "phase_deg": {"range": [0, 360], "circular": True},
                },
            },
        },
    }
}
TRIPLES = 4_096_000
CHUNK = 8_000
TAU_MS = 8.0
TIMED_CHUNKS = 6
TARGET_S = 600.0


def main() -> None:
    world = World.from_dict(WORLD)
    canvas = Canvas.from_grid(40, 40, 0.15)
    seconds = []
    for c in range(TIMED_CHUNKS + 1):
        start = time.perf_counter()
        indices = range(c * CHUNK, (c + 1) * CHUNK)
        draws = sample(world, indices=indices, seed=0)
        t = 100.0 + 800.0 * rng.uniforms(rng.draw_seeds(1, np.arange(c * CHUNK, (c + 1) * CHUNK)), "t")
        times = torch.from_numpy(np.stack([t - TAU_MS, t, t + TAU_MS], axis=1))
        frames = render(draws, canvas, times, dtype=torch.float64)
        assert frames.shape == (CHUNK, 3, 40, 40)
        seconds.append(time.perf_counter() - start)
    per_chunk = statistics.median(seconds[1:])  # the first chunk warms up
    total = per_chunk * TRIPLES / CHUNK
    print(f"threads: {torch.get_num_threads()}, torch {torch.__version__}")
    print(f"per chunk of {CHUNK} triples: {per_chunk:.3f} s (median of {TIMED_CHUNKS})")
    print(f"4.1 M triples at 40x40, float64: {total:.0f} s ({total / 60:.1f} min)")
    print("PASS" if total <= TARGET_S else f"OVER the {TARGET_S:.0f} s target")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `conda run -n sensoryforge python benchmarks/world_render.py`
Expected: `PASS`, with an extrapolated total under 600 s (a throwaway probe before planning measured ~55 s for a single-Gaussian class and ~330 s for 6-dot braille alone, so a four-class mix should land near 2–4 min).

**Gate:** if it prints `OVER`, stop here and report the per-chunk time and the per-class times (time each class with `classes=[name]`) to Ben before going on. The spec's fallback (a separable Gaussian path) is a design change that needs his yes.

- [ ] **Step 3: Record the result** — append to `docs/reference/benchmarks.md`:

```markdown
## World renderer

`benchmarks/world_render.py` renders pressure-simulation's RA design load: 4.1 M triples
(t − τ, t, t + τ, τ = 8 ms) at 40×40, 0.15 mm, in float64, from a four-class world (dots,
edges, random braille cells, signed textures), in chunks of 8000 draws.

| Machine | Threads | Per 8000 triples | 4.1 M triples |
|---|---|---|---|
| <the machine, e.g. MacBook Pro M-series> | <threads printed> | <s printed> | <total printed> |

The target (spec §5.6) is at most 10 minutes on a laptop CPU.
```

Fill the table row with the values the run printed (machine name from `sysctl -n machdep.cpu.brand_string`).

- [ ] **Step 4: Commit**

```bash
git add benchmarks/world_render.py docs/reference/benchmarks.md
git commit -m "perf(world): benchmark the renderer on PS's 4.1 M-triple RA load

Finding: the world renderer draws pressure-simulation's 4.1 M RA design triples at 40x40 in float64 in <total printed> s on <machine> (<threads> threads), within the 10-minute target

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(Write the printed numbers into the trailer; a `Finding:` must state the measured fact.)

---

### Task 8: SF's own sha, and bundle schema 2.2.0

**Files:**
- Create: `sensoryforge/provenance.py`, `tests/unit/test_provenance.py`, `tests/unit/test_bundle_world_entry.py`
- Modify: `sensoryforge/io/bundle.py`, `tests/unit/test_bundle_io.py:76-77`, `tests/integration/test_event_encoders_engine.py:8,127`

**Interfaces:**
- Produces: `read_source_info(package_dir=<sensoryforge package dir>) -> {"sha", "dirty", "source"}`; `source_info()` (cached copy for this process); in `io/bundle.py`: `SCHEMA_VERSION = "2.2.0"`, `WORLD_ENTRY_KIND = "sensoryforge_world_entry"`; `config.json` gains `sensoryforge_sha` (every bundle) and `world: {world_id, dataset_id, entry}` (world entries); `data.h5` root attribute `sensoryforge_sha`; `load_bundle(...).meta["sensoryforge_sha"]`; `stimuli/stimulus.json` for a world entry is `{"schema_version", "kind": "sensoryforge_world_entry", "entry", "layer", "dt_ms", "total_ms", "n_frames", "grid", "reconstructible_by_pressure_simulation": false}`.
- A caller marks a world entry by passing `stimulus_config={"kind": "sensoryforge_world_entry", "entry": <manifest row>, "layer": <layer or [[start, layer], ...]>}` to `SimulationEngine.run`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_provenance.py`:

```python
"""SensoryForge's own sha: from git in a checkout, from pip's direct_url.json otherwise."""

import json
import subprocess
from pathlib import Path

from sensoryforge import provenance
from sensoryforge.provenance import read_source_info, source_info

ROOT = Path(__file__).resolve().parents[2]


def test_a_git_checkout_reports_its_head():
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    info = read_source_info()
    assert info["source"] == "git" and info["sha"] == head
    assert isinstance(info["dirty"], bool)
    assert source_info()["sha"] == head


def test_a_pip_install_from_git_reports_its_commit(tmp_path, monkeypatch):
    class FakeDistribution:
        def read_text(self, name):
            if name == "direct_url.json":
                return json.dumps(
                    {"url": "file:///x", "vcs_info": {"vcs": "git", "commit_id": "abc123"}}
                )
            return None

    monkeypatch.setattr(provenance.metadata, "distribution", lambda name: FakeDistribution())
    (tmp_path / "sensoryforge").mkdir()
    info = read_source_info(tmp_path / "sensoryforge")
    assert info == {"sha": "abc123", "dirty": False, "source": "direct_url"}


def test_with_neither_the_sha_is_unknown(tmp_path, monkeypatch):
    def missing(name):
        raise provenance.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(provenance.metadata, "distribution", missing)
    (tmp_path / "sensoryforge").mkdir()
    info = read_source_info(tmp_path / "sensoryforge")
    assert info == {"sha": "unknown", "dirty": None, "source": "unknown"}
```

`tests/unit/test_bundle_world_entry.py`:

```python
"""Bundle schema 2.2.0: SensoryForge's sha in every bundle; world entries carry their record."""

import json
from pathlib import Path

import h5py
import torch

from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.io.bundle import SCHEMA_VERSION, load_bundle
from sensoryforge.io.design import load_design
from sensoryforge.provenance import source_info

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "design_8x8"


def _run(tmp_path, stimulus_config):
    engine = SimulationEngine(load_design(FIXTURE))
    bundle = tmp_path / "bundle"
    engine.run(torch.zeros(1, 10, 8, 8), bundle_dir=bundle, stimulus_config=stimulus_config, seed=1)
    return bundle


def test_every_bundle_records_sensoryforge_sha(tmp_path):
    bundle = _run(tmp_path, {"type": "gaussian"})
    sha = source_info()["sha"]
    cfg = json.loads((bundle / "config.json").read_text())
    assert SCHEMA_VERSION == "2.2.0" and cfg["schema_version"] == "2.2.0"
    assert cfg["sensoryforge_sha"] == sha and "world" not in cfg
    with h5py.File(bundle / "data.h5", "r") as f:
        assert f.attrs["sensoryforge_sha"] == sha
    assert load_bundle(bundle).meta["sensoryforge_sha"] == sha


def test_a_world_entry_bundle_carries_its_record(tmp_path):
    entry = {"entry": "test/dots/0001", "world_id": "w-abc", "dataset_id": "d-def", "class": "dots"}
    layer = {"shape": {"kind": "gaussian"}}
    bundle = _run(
        tmp_path, {"kind": "sensoryforge_world_entry", "entry": entry, "layer": layer}
    )
    cfg = json.loads((bundle / "config.json").read_text())
    assert cfg["world"] == {"world_id": "w-abc", "dataset_id": "d-def", "entry": "test/dots/0001"}
    payload = json.loads((bundle / "stimuli" / "stimulus.json").read_text())
    assert payload["kind"] == "sensoryforge_world_entry"
    assert payload["schema_version"] == "2.2.0"
    assert payload["entry"] == entry and payload["layer"] == layer
    assert payload["n_frames"] == 10
    assert payload["reconstructible_by_pressure_simulation"] is False
```

- [ ] **Step 2: Run to see them fail**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_provenance.py tests/unit/test_bundle_world_entry.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sensoryforge.provenance'`.

- [ ] **Step 3: Implement `sensoryforge/provenance.py`**

```python
"""Where this SensoryForge came from: its git sha (spec §7.4).

In a checkout (an editable install, or a worktree on ``PYTHONPATH``) the sha
is ``git rev-parse HEAD``. In a pip install from git -- pressure-simulation's
pinned ``pip install "sensoryforge @ git+file:///...@<sha>"`` -- pip records
the commit in PEP 610's ``direct_url.json``. Otherwise it is ``"unknown"``.
"""

from __future__ import annotations

import functools
import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

_PACKAGE_DIR = Path(__file__).resolve().parent


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _direct_url_sha() -> Optional[str]:
    try:
        text = metadata.distribution("sensoryforge").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    try:
        info = json.loads(text)
    except json.JSONDecodeError:
        return None
    return (info.get("vcs_info") or {}).get("commit_id")


def read_source_info(package_dir: Path = _PACKAGE_DIR) -> Dict[str, Any]:
    """``{"sha", "dirty", "source"}`` for the SensoryForge package in ``package_dir``.

    Args:
        package_dir: The ``sensoryforge`` package directory (default: this one).

    Returns:
        ``source`` is ``"git"`` (``dirty``: tracked files modified),
        ``"direct_url"`` (``dirty`` False) or ``"unknown"`` (``sha`` "unknown",
        ``dirty`` None).
    """
    root = Path(package_dir).resolve().parent
    if (root / ".git").exists():
        sha = _git(root, "rev-parse", "HEAD")
        if sha:
            status = _git(root, "status", "--porcelain", "--untracked-files=no")
            return {"sha": sha, "dirty": bool(status), "source": "git"}
    sha = _direct_url_sha()
    if sha:
        return {"sha": sha, "dirty": False, "source": "direct_url"}
    return {"sha": "unknown", "dirty": None, "source": "unknown"}


@functools.lru_cache(maxsize=1)
def _cached() -> Dict[str, Any]:
    return read_source_info()


def source_info() -> Dict[str, Any]:
    """This process's SensoryForge sha (read once, then cached); a fresh copy each call."""
    return dict(_cached())
```

- [ ] **Step 4: Change `sensoryforge/io/bundle.py`**

1. `SCHEMA_VERSION = "2.2.0"`; add `WORLD_ENTRY_KIND = "sensoryforge_world_entry"` beside it; add `from sensoryforge.provenance import source_info` to the imports. In the module docstring's layout notes add: "2.2.0 (2026-10-02, additive): every bundle records ``sensoryforge_sha`` (``config.json``, ``data.h5`` attributes); a world data-set entry's ``stimuli/stimulus.json`` is ``kind: sensoryforge_world_entry`` with the entry's record, and ``config.json`` gains ``world``."
2. In `build_stimulus_payload`, right after `stim_type = ...`, add:

```python
    if cfg.get("kind") == WORLD_ENTRY_KIND:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": WORLD_ENTRY_KIND,
            "entry": cfg.get("entry"),
            "layer": cfg.get("layer"),
            "dt_ms": dt_ms,
            "total_ms": total_ms,
            "n_frames": int(n_frames),
            "grid": grid,
            "reconstructible_by_pressure_simulation": False,
        }
```

and mention the third kind in its docstring.
3. In `write_bundle`, after `config_json = {...}` is built: `config_json["sensoryforge_sha"] = source_info()["sha"]`, and

```python
    if stimulus_config and stimulus_config.get("kind") == WORLD_ENTRY_KIND:
        entry = stimulus_config.get("entry") or {}
        config_json["world"] = {
            "world_id": entry.get("world_id"),
            "dataset_id": entry.get("dataset_id"),
            "entry": entry.get("entry"),
        }
```

4. Where the `data.h5` root attributes are written (after `f.attrs["sensoryforge_version"] = ...`): `f.attrs["sensoryforge_sha"] = source_info()["sha"]`.
5. In `load_bundle`, after `meta["sensoryforge_version"] = ...`: `meta["sensoryforge_sha"] = f.attrs.get("sensoryforge_sha")`. Add `sensoryforge_sha` to the `Bundle.meta` docstring.

- [ ] **Step 5: Move the two schema pins to 2.2.0**

`tests/unit/test_bundle_io.py` lines 76-77 become:

```python
        # 2.1.0 (2026-10-01): signed event populations (level_crossing).
        # 2.2.0 (2026-10-02): sensoryforge_sha; world data-set entries.
        assert cfg["schema_version"] == "2.2.0"
```

`tests/integration/test_event_encoders_engine.py` line 127 becomes `assert cfg["schema_version"] == SCHEMA_VERSION == "2.2.0"`, and line 8's "(schema 2.1.0)" becomes "(schema 2.1.0 and later)".

- [ ] **Step 6: Run the bundle suites**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_provenance.py tests/unit/test_bundle_world_entry.py tests/unit/test_bundle_io.py tests/integration/test_event_encoders_engine.py tests/integration/test_bundle_stimulus_payload.py tests/integration/test_bundle_pressure_sim_compat.py tests/integration/test_cli_run_design.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add sensoryforge/provenance.py sensoryforge/io/bundle.py tests/unit/test_provenance.py tests/unit/test_bundle_world_entry.py tests/unit/test_bundle_io.py tests/integration/test_event_encoders_engine.py
git commit -m "feat(bundle): schema 2.2.0 records SensoryForge's sha and world entries

Every bundle records the sha of the SensoryForge that wrote it (git in a
checkout, pip's direct_url.json in a pinned install). A world data-set
entry's stimulus.json carries the entry's full record and its layered
form, and config.json gains world {world_id, dataset_id, entry}. Additive:
2.x readers keep working; pressure-simulation's reader does not check the
version.

Fixed: bundles recorded only sensoryforge_version 1.0.0, never the git sha of the SensoryForge that wrote them, so two bundles from different commits could not be told apart

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Data sets, `sensoryforge dataset build`, `sensoryforge world`

**Files:**
- Create: `sensoryforge/world/dataset.py`, `tests/fixtures/worlds/dataset_small.yml`, `tests/unit/test_world_dataset.py`, `tests/integration/test_cli_world_dataset.py`
- Modify: `sensoryforge/world/__init__.py` (export the data-set API), `sensoryforge/cli.py` (`dataset` and `world` subcommands)

**Interfaces:**
- Consumes: `rng` (Task 2); `World`, `ClassSpec`, `load_world` (Task 4); `Draw`, `Session`, `sample`, `session` (Task 5); `source_info` (Task 8).
- Produces: `SPLIT_KINDS`; `SplitSpec(name, kind, n, repeats, noise_repeats, bins, per_bin, duration_ms, draws)`; `DatasetSpec(name, world, seed, duration_ms, splits, source, dataset_id)`; `load_dataset(source, base_dir=None) -> DatasetSpec`; `Entry(entry, split, repeat, noise_repeat, class_name, item, bins, probe, seeds, duration_ms, world_id, dataset_id, base)` with `truncated`, `to_dict()`; `stratify_class(world, cls, bins, per_bin, class_seed) -> (draws, labels)`; `build_dataset(spec) -> List[Entry]`; `skipped_probes(spec) -> List[str]`; `write_dataset(spec, entries, out_dir) -> Path`; `load_manifest(path) -> List[dict]`. CLI: `cmd_dataset(args)`, `cmd_world(args)`.

- [ ] **Step 1: Create the test data set** `tests/fixtures/worlds/dataset_small.yml`

```yaml
# A data set on the test world, small enough to build in a second.
dataset:
  name: small
  world: tactile_small.yml
  seed: 20261002
  duration_ms: 120
  splits:
    train:      {n: 6, repeats: 2}
    validation: {n: 4, noise_repeats: 2}
    test:       {stratified: {bins: 2, per_bin: 2}}
    probes:     {per_bin: 1}
    held_out:   {stratified: {bins: 2, per_bin: 1}}
    sessions:   {n: 1, duration_ms: 300}
    fixed:      {draws: [braille_H]}
```

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_world_dataset.py`:

```python
"""Data sets: ids, counts, seeds, strata, probes, truncation, the world pin, the manifest."""

import copy
import json
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from sensoryforge.world.dataset import (
    build_dataset,
    load_dataset,
    load_manifest,
    skipped_probes,
    write_dataset,
)

WORLDS = Path(__file__).resolve().parents[1] / "fixtures" / "worlds"
SPEC = load_dataset(WORLDS / "dataset_small.yml")
ENTRIES = build_dataset(SPEC)
RAW = yaml.safe_load((WORLDS / "dataset_small.yml").read_text())


def _split(name):
    return [e for e in ENTRIES if e.split == name]


def test_ids_and_counts():
    assert re.fullmatch(r"d-[0-9a-f]{12}", SPEC.dataset_id)
    assert len({e.entry for e in ENTRIES}) == len(ENTRIES)
    assert len(_split("train")) == 12 and {e.repeat for e in _split("train")} == {0, 1}
    assert len(_split("validation")) == 8
    assert {e.noise_repeat for e in _split("validation")} == {0, 1}
    assert len(_split("test")) == len(SPEC.world.classes) * 4
    assert len(_split("held_out")) == 2
    assert len(_split("sessions")) == 1 and len(_split("fixed")) == 1
    assert _split("train")[0].entry == "train/r0/00000"
    assert _split("validation")[1].entry == "validation/r0/00000.n1"
    assert _split("test")[0].entry == "test/dots/0000"
    assert _split("sessions")[0].entry == "sessions/000"
    assert _split("fixed")[0].entry == "fixed/braille_H"


def test_building_is_deterministic():
    again = build_dataset(load_dataset(WORLDS / "dataset_small.yml"))
    assert [e.to_dict() for e in again] == [e.to_dict() for e in ENTRIES]


def test_no_draw_seed_is_in_two_splits_and_noise_seeds_are_unique():
    noise = [e.seeds["noise"] for e in ENTRIES]
    assert len(set(noise)) == len(noise)
    owners = {}
    for e in ENTRIES:
        if e.seeds["draw"] is not None:
            owners.setdefault(e.seeds["draw"], set()).add(e.split)
    assert all(len(splits) == 1 for splits in owners.values())
    first, second = _split("validation")[:2]
    assert first.seeds["draw"] == second.seeds["draw"]
    assert first.seeds["noise"] != second.seeds["noise"]


def test_train_repeats_draw_fresh_stimuli():
    r0 = {e.seeds["draw"] for e in _split("train") if e.repeat == 0}
    r1 = {e.seeds["draw"] for e in _split("train") if e.repeat == 1}
    assert not r0 & r1


def test_every_test_bin_holds_per_bin_draws_per_class():
    for cls in SPEC.world.classes.values():
        rows = [e for e in _split("test") if e.class_name == cls.name]
        assert len(rows) == 4
        for axis in cls.random_axes:
            counts = Counter(e.bins[axis.name] for e in rows)
            if axis.support() is None:
                assert sorted(counts.values()) == [2, 2], (cls.name, axis.name)
                for e in rows:
                    lo, hi = (float(x) for x in e.bins[axis.name][1:-1].split(", "))
                    value = e.item.values[axis.name]
                    assert lo - 1e-9 <= value <= hi + 1e-9
            else:
                assert max(counts.values()) - min(counts.values()) <= 1


def test_probes_are_labelled_and_outside_their_range():
    probes = _split("probes")
    assert probes
    for e in probes:
        name = e.probe["axis"]
        axis = e.item.spec.axes[name]
        value = e.item.values[name]
        assert e.bins == {name: e.probe["side"]}
        assert value < axis.lo if e.probe["side"] == "below" else value > axis.hi
        assert e.item.out_of_range == (name,)
    skipped = skipped_probes(SPEC)
    assert "dots/delay_ms-below" in skipped  # delay_ms starts at 0
    assert not any(e.entry.startswith("probes/dots/delay_ms-below") for e in probes)
    assert not any(e.probe["axis"] == "direction_deg" for e in probes)  # circular


def test_long_draws_are_marked_truncated():
    # 'twice' (two contacts) reaches 215 ms in 120 ms entries.
    twice = [e for e in ENTRIES if e.class_name == "twice"]
    assert any(e.truncated for e in twice)
    for e in twice:
        assert e.truncated == (e.item.end_ms > 120.0)
    assert all(e.duration_ms == 120.0 for e in ENTRIES if e.split != "sessions")
    assert _split("sessions")[0].duration_ms == 300.0


def test_world_pin_mismatch_fails():
    raw = copy.deepcopy(RAW)
    raw["dataset"]["world_id"] = "w-000000000000"
    with pytest.raises(ValueError, match="w-000000000000.*" + SPEC.world.world_id):
        load_dataset(raw, base_dir=WORLDS)
    raw["dataset"]["world_id"] = SPEC.world.world_id
    assert load_dataset(raw, base_dir=WORLDS).dataset_id != ""


def test_manifest_and_dataset_json(tmp_path):
    out = write_dataset(SPEC, ENTRIES, tmp_path / "ds")
    info = json.loads((out / "dataset.json").read_text())
    assert info["dataset_id"] == SPEC.dataset_id and info["world_id"] == SPEC.world.world_id
    assert info["n_entries"] == len(ENTRIES) and info["counts"]["test"]["dots"] == 4
    assert info["sensoryforge"]["source"] in ("git", "direct_url", "unknown")
    assert "dots/delay_ms-below" in info["skipped_probes"]
    assert load_manifest(out) == [json.loads(json.dumps(e.to_dict())) for e in ENTRIES]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["splits"].update(colours={"n": 3}), "unknown kind"),
        (lambda d: d["splits"]["train"].update(n=0), "n >= 1"),
        (lambda d: d["splits"]["fixed"].update(draws=["nope"]), "no fixed draw"),
        (lambda d: d["splits"]["test"].update(stratified={"bins": 0, "per_bin": 2}), "bins"),
        (lambda d: d.pop("seed"), "seed"),
        (lambda d: d["splits"]["train"].update(repeats=0), "repeats"),
    ],
)
def test_invalid_specs_are_named(mutate, message):
    raw = copy.deepcopy(RAW)
    mutate(raw["dataset"])
    with pytest.raises(ValueError, match=message):
        load_dataset(raw, base_dir=WORLDS)
```

`tests/integration/test_cli_world_dataset.py`:

```python
"""`sensoryforge dataset build` and `sensoryforge world validate|sample`."""

import json
from pathlib import Path

from sensoryforge.cli import cmd_dataset, cmd_world, create_parser
from sensoryforge.world.dataset import load_dataset

WORLDS = Path(__file__).resolve().parents[1] / "fixtures" / "worlds"


def _cli(argv):
    args = create_parser().parse_args(argv)
    return {"dataset": cmd_dataset, "world": cmd_world}[args.command](args)


def test_dataset_build_writes_the_manifest(tmp_path, capsys):
    spec = WORLDS / "dataset_small.yml"
    assert _cli(["dataset", "build", str(spec), "--out", str(tmp_path / "ds")]) == 0
    assert (tmp_path / "ds" / "manifest.jsonl").exists()
    assert load_dataset(spec).dataset_id in capsys.readouterr().out


def test_world_validate_and_sample(capsys):
    world = str(WORLDS / "tactile_small.yml")
    assert _cli(["world", "validate", world]) == 0
    out = capsys.readouterr().out
    assert out.startswith("w-") and "class dots" in out and "held out gratings" in out
    assert _cli(["world", "sample", world, "-n", "3", "--seed", "5"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["index"] == 0


def test_a_bad_world_is_a_message_not_a_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.yml"
    bad.write_text("world: {classes: {}}\n")
    assert _cli(["world", "validate", str(bad)]) == 1
    assert "declare at least one class" in capsys.readouterr().err
```

- [ ] **Step 3: Run to see them fail**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_world_dataset.py tests/integration/test_cli_world_dataset.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sensoryforge.world.dataset'`.

- [ ] **Step 4: Implement `sensoryforge/world/dataset.py`**

```python
"""Data sets on a world: splits, strata, probes, seeds and the manifest (spec §6)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import yaml

from sensoryforge.provenance import source_info
from sensoryforge.world import rng
from sensoryforge.world.distributions import AxisSpec
from sensoryforge.world.sampling import Draw, Session, sample, session
from sensoryforge.world.schema import ClassSpec, World, load_world

FORMAT = "sensoryforge-dataset/1"
#: Split name -> its kind when the split gives no ``kind:``.
SPLIT_KINDS = {
    "train": "declared",
    "validation": "declared",
    "test": "stratified",
    "probes": "probes",
    "held_out": "held_out",
    "sessions": "sessions",
    "fixed": "fixed",
}
_KINDS = set(SPLIT_KINDS.values())
_SPLIT_KEYS = {
    "kind", "n", "repeats", "noise_repeats", "stratified", "per_bin", "bins",
    "duration_ms", "draws",
}
_DATASET_KEYS = {"name", "world", "world_id", "seed", "duration_ms", "splits"}
#: An int axis with more values than this cannot be stratified one bin per value.
MAX_INT_STRATA = 64

Item = Union[Draw, Session]


@dataclass(frozen=True)
class SplitSpec:
    """One split of a data set (spec §6.1)."""

    name: str
    kind: str
    n: int = 0
    repeats: int = 1
    noise_repeats: int = 1
    bins: int = 0
    per_bin: int = 0
    duration_ms: Optional[float] = None
    draws: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """The normalised split (what the data-set id hashes)."""
        out: Dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "repeats": self.repeats,
            "noise_repeats": self.noise_repeats,
        }
        if self.n:
            out["n"] = self.n
        if self.bins:
            out["bins"] = self.bins
        if self.per_bin:
            out["per_bin"] = self.per_bin
        if self.duration_ms is not None:
            out["duration_ms"] = self.duration_ms
        if self.draws:
            out["draws"] = list(self.draws)
        return out


@dataclass
class DatasetSpec:
    """A parsed data-set spec. Build it with :func:`load_dataset`."""

    name: str
    world: World
    seed: int
    duration_ms: float
    splits: List[SplitSpec]
    source: Dict[str, Any]
    dataset_id: str = ""


def _positive_int(raw: Dict[str, Any], key: str, where: str, default: Optional[int] = None) -> int:
    value = raw.get(key, default)
    if value is None or int(value) < 1:
        raise ValueError(f"{where}: {key} >= 1 required, got {value!r}")
    return int(value)


def _strata(raw: Dict[str, Any], where: str) -> Tuple[int, int]:
    strat = raw.get("stratified")
    if not isinstance(strat, dict):
        raise ValueError(f"{where}: give stratified: {{bins: B, per_bin: m}}")
    return (
        _positive_int(strat, "bins", f"{where}.stratified"),
        _positive_int(strat, "per_bin", f"{where}.stratified"),
    )


def _parse_split(name: str, raw: Any, world: World, test_bins: Optional[int]) -> SplitSpec:
    where = f"dataset.splits.{name}"
    raw = dict(raw or {})
    unknown = set(raw) - _SPLIT_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(_SPLIT_KEYS)}")
    kind = raw.get("kind", SPLIT_KINDS.get(name))
    if kind not in _KINDS:
        raise ValueError(
            f"{where}: unknown kind {kind!r}; name the split one of "
            f"{sorted(SPLIT_KINDS)} or give kind: one of {sorted(_KINDS)}"
        )
    common = {
        "name": name,
        "kind": kind,
        "repeats": _positive_int(raw, "repeats", where, 1),
        "noise_repeats": _positive_int(raw, "noise_repeats", where, 1),
    }
    if kind == "declared":
        return SplitSpec(**common, n=_positive_int(raw, "n", where))
    if kind == "stratified":
        bins, per_bin = _strata(raw, where)
        return SplitSpec(**common, bins=bins, per_bin=per_bin)
    if kind == "probes":
        return SplitSpec(
            **common,
            per_bin=_positive_int(raw, "per_bin", where),
            bins=_positive_int(raw, "bins", where, test_bins or 5),
        )
    if kind == "held_out":
        if not world.held_out:
            raise ValueError(f"{where}: the world declares no held_out classes")
        if "stratified" in raw:
            bins, per_bin = _strata(raw, where)
            return SplitSpec(**common, bins=bins, per_bin=per_bin)
        return SplitSpec(**common, n=_positive_int(raw, "n", where))
    if kind == "sessions":
        duration = float(raw.get("duration_ms", 0))
        if duration <= 0:
            raise ValueError(f"{where}: duration_ms must be > 0")
        return SplitSpec(**common, n=_positive_int(raw, "n", where), duration_ms=duration)
    draws = tuple(str(d) for d in raw.get("draws") or [])
    if not draws:
        raise ValueError(f"{where}: list the fixed draws, e.g. draws: [braille_H]")
    missing = [d for d in draws if d not in world.fixed]
    if missing:
        raise ValueError(f"{where}: no fixed draw {missing}; the world has {sorted(world.fixed)}")
    return SplitSpec(**common, draws=draws)


def load_dataset(source: Union[str, Path, Dict[str, Any]], base_dir: Optional[Union[str, Path]] = None) -> DatasetSpec:
    """Load a data-set spec (``dataset:`` YAML) and the world it names.

    Args:
        source: A YAML path, or the mapping itself.
        base_dir: Where a relative ``world:`` path is resolved for a mapping
            (default: the current directory); a file resolves against its own
            directory.

    Raises:
        ValueError: For an invalid spec, or a ``world_id`` pin that does not
            match the world's content.
    """
    if isinstance(source, dict):
        data = source
        base = Path(base_dir) if base_dir is not None else Path.cwd()
    else:
        path = Path(source)
        data = yaml.safe_load(path.read_text())
        base = path.resolve().parent
    raw = data.get("dataset", data) if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        raise ValueError("a data set is a mapping (optionally under a top-level 'dataset:' key)")
    unknown = set(raw) - _DATASET_KEYS
    if unknown:
        raise ValueError(f"dataset: unknown keys {sorted(unknown)}; allowed: {sorted(_DATASET_KEYS)}")
    world_ref = raw.get("world")
    if isinstance(world_ref, dict):
        world = load_world(world_ref)
    elif isinstance(world_ref, (str, Path)):
        world_path = Path(world_ref)
        world = load_world(world_path if world_path.is_absolute() else base / world_path)
    else:
        raise ValueError("dataset.world: give a path to a world file, or a world mapping")
    pin = raw.get("world_id")
    if pin is not None and pin != world.world_id:
        raise ValueError(
            f"dataset.world_id pins {pin} but the world's content gives {world.world_id}: "
            "the world changed since this data set was declared"
        )
    if "seed" not in raw:
        raise ValueError("dataset.seed: required (an integer)")
    seed = int(raw["seed"])
    duration_ms = float(raw.get("duration_ms", 0))
    if duration_ms <= 0:
        raise ValueError("dataset.duration_ms: required, > 0")
    splits_raw = raw.get("splits") or {}
    if not splits_raw:
        raise ValueError("dataset.splits: declare at least one split")
    test_raw = splits_raw.get("test") or {}
    test_bins = (test_raw.get("stratified") or {}).get("bins") if isinstance(test_raw, dict) else None
    splits = [_parse_split(str(n), s, world, test_bins) for n, s in splits_raw.items()]
    name = str(raw.get("name", "dataset"))
    normal = {
        "format": FORMAT,
        "name": name,
        "seed": seed,
        "duration_ms": duration_ms,
        "world_id": world.world_id,
        "splits": [s.to_dict() for s in splits],
    }
    canonical = json.dumps(normal, sort_keys=True, separators=(",", ":"))
    return DatasetSpec(
        name=name,
        world=world,
        seed=seed,
        duration_ms=duration_ms,
        splits=splits,
        source=json.loads(json.dumps(raw, default=str)),
        dataset_id="d-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12],
    )


@dataclass
class Entry:
    """One row of a data set's manifest (spec §6.4-§6.5)."""

    entry: str
    split: str
    repeat: int
    noise_repeat: int
    class_name: Optional[str]
    item: Item
    bins: Dict[str, Any]
    probe: Optional[Dict[str, str]]
    seeds: Dict[str, Optional[int]]
    duration_ms: float
    world_id: str
    dataset_id: str
    base: str = field(default="", repr=False)

    @property
    def truncated(self) -> bool:
        """True when the draw runs past ``duration_ms`` (a session: its last draw)."""
        if isinstance(self.item, Session):
            return self.item.truncated
        return self.item.end_ms > self.duration_ms

    def to_dict(self) -> Dict[str, Any]:
        """The manifest row."""
        return {
            "entry": self.entry,
            "split": self.split,
            "repeat": self.repeat,
            "noise_repeat": self.noise_repeat,
            "class": self.class_name,
            "draw": self.item.to_dict(),
            "bins": dict(self.bins),
            "probe": self.probe,
            "seeds": dict(self.seeds),
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
            "world_id": self.world_id,
            "dataset_id": self.dataset_id,
        }


def stratify_class(
    world: World, cls: ClassSpec, bins: int, per_bin: int, class_seed: int
) -> Tuple[List[Draw], List[Dict[str, Any]]]:
    """A Latin hypercube of ``bins * per_bin`` draws of one class (spec §6.2).

    Returns:
        ``(draws, labels)``: each draw and its bin label per random axis.
    """
    n = bins * per_bin
    seeds = rng.draw_seeds(class_seed, np.arange(n))
    columns: Dict[str, List[Any]] = {}
    labels: List[Dict[str, Any]] = [{} for _ in range(n)]
    for name, axis in cls.axes.items():
        if not axis.is_random:
            columns[name] = [axis.value] * n
            continue
        order = rng.permutation(n, class_seed, name)
        support = axis.support()
        if support is not None:
            if axis.form == "int" and len(support) > MAX_INT_STRATA:
                raise ValueError(
                    f"class {cls.name!r}: int axis {name!r} has {len(support)} values; "
                    f"stratifying allows at most {MAX_INT_STRATA} (declare it as a float range)"
                )
            picks = (np.arange(n) % len(support))[order]
            columns[name] = [support[i] for i in picks]
            for j, i in enumerate(picks):
                labels[j][name] = support[i]
        elif axis.form == "numeric":
            picks = np.repeat(np.arange(bins), per_bin)[order]
            columns[name] = axis.stratum_values(picks, bins, rng.uniforms(seeds, name))
            for j, b in enumerate(picks):
                labels[j][name] = axis.bin_label(int(b), bins)
        else:
            raise ValueError(
                f"class {cls.name!r}: axis {name!r} uses distribution {axis.dist!r}, "
                "which declares no finite support, so it cannot be stratified"
            )
    draws = [
        Draw(
            world=world,
            class_name=cls.name,
            values={k: columns[k][j] for k in cls.axes},
            seed=int(class_seed),
            index=j,
            draw_seed=int(seeds[j]),
            sampling="stratified",
        )
        for j in range(n)
    ]
    return draws, labels


def _probe_axes(cls: ClassSpec) -> List[AxisSpec]:
    return [a for a in cls.random_axes if a.form == "numeric" and not a.circular and a.probes]


def _probe_draws(
    world: World, cls: ClassSpec, axis: AxisSpec, side: str, split: SplitSpec, seed: int
) -> Optional[List[Draw]]:
    n = split.per_bin
    seeds = rng.draw_seeds(seed, np.arange(n))
    values = axis.probe_values(side, split.bins, rng.uniforms(seeds, axis.name))
    if values is None:
        return None
    columns: Dict[str, List[Any]] = {}
    for name, other in cls.axes.items():
        if name == axis.name:
            columns[name] = values
        elif other.is_random:
            columns[name] = other.sample(rng.uniforms(seeds, name))
        else:
            columns[name] = [other.value] * n
    return [
        Draw(
            world=world,
            class_name=cls.name,
            values={k: columns[k][j] for k in cls.axes},
            seed=int(seed),
            index=j,
            draw_seed=int(seeds[j]),
            sampling="probe",
            out_of_range=(axis.name,),
        )
        for j in range(n)
    ]


def _split_items(
    spec: DatasetSpec, split: SplitSpec, split_seed: int
) -> Iterator[Tuple[str, Item, Dict[str, Any], Optional[Dict[str, str]]]]:
    """``(stem, item, bins, probe)`` for every entry of one (split, repeat)."""
    world = spec.world
    if split.kind == "declared":
        for i, draw in enumerate(sample(world, n=split.n, seed=split_seed)):
            yield f"{i:05d}", draw, {}, None
    elif split.kind in ("stratified", "held_out") and split.bins:
        classes = world.classes if split.kind == "stratified" else world.held_out
        for cls in classes.values():
            draws, labels = stratify_class(
                world, cls, split.bins, split.per_bin, rng.seed53(split_seed, cls.name)
            )
            for i, (draw, label) in enumerate(zip(draws, labels)):
                yield f"{cls.name}/{i:04d}", draw, label, None
    elif split.kind == "held_out":
        held = list(world.held_out)
        for i, draw in enumerate(sample(world, n=split.n, seed=split_seed, classes=held)):
            yield f"{draw.class_name}/{i:04d}", draw, {}, None
    elif split.kind == "probes":
        for cls in world.classes.values():
            for axis in _probe_axes(cls):
                for side in ("below", "above"):
                    side_seed = rng.seed53(split_seed, cls.name, axis.name, side)
                    draws = _probe_draws(world, cls, axis, side, split, side_seed)
                    for i, draw in enumerate(draws or []):
                        yield (
                            f"{cls.name}/{axis.name}-{side}/{i:03d}",
                            draw,
                            {axis.name: side},
                            {"axis": axis.name, "side": side},
                        )
    elif split.kind == "sessions":
        for i in range(split.n):
            yield f"{i:03d}", session(world, split.duration_ms, seed=split_seed, index=i), {}, None
    else:
        for name in split.draws:
            yield name, world.fixed_draw(name), {}, None


def build_dataset(spec: DatasetSpec) -> List[Entry]:
    """Every entry of the data set, in split order, then id order (spec §6.4).

    Raises:
        ValueError: If any draw seed or noise seed appears twice.
    """
    entries: List[Entry] = []
    for split in spec.splits:
        for repeat in range(split.repeats):
            split_seed = rng.seed53(spec.seed, split.name, repeat)
            if split.kind == "declared" or split.repeats > 1:
                prefix = f"{split.name}/r{repeat}"
            else:
                prefix = split.name
            duration = split.duration_ms if split.kind == "sessions" else spec.duration_ms
            for stem, item, bins, probe in _split_items(spec, split, split_seed):
                base = f"{prefix}/{stem}"
                is_session = isinstance(item, Session)
                draw_seed = item.session_seed if is_session else item.draw_seed
                for noise_repeat in range(split.noise_repeats):
                    entry_id = base if split.noise_repeats == 1 else f"{base}.n{noise_repeat}"
                    entries.append(
                        Entry(
                            entry=entry_id,
                            split=split.name,
                            repeat=repeat,
                            noise_repeat=noise_repeat,
                            class_name=None if is_session else item.class_name,
                            item=item,
                            bins=bins,
                            probe=probe,
                            seeds={
                                "draw": draw_seed,
                                "noise": rng.seed53(spec.seed, "noise", entry_id),
                            },
                            duration_ms=float(duration),
                            world_id=spec.world.world_id,
                            dataset_id=spec.dataset_id,
                            base=base,
                        )
                    )
    _check_unique(entries)
    return entries


def _check_unique(entries: List[Entry]) -> None:
    noise_owner: Dict[int, str] = {}
    draw_owner: Dict[int, str] = {}
    for e in entries:
        noise = e.seeds["noise"]
        if noise in noise_owner:
            raise ValueError(
                f"noise seed {noise} of {e.entry} repeats {noise_owner[noise]}; "
                "choose another data-set seed"
            )
        noise_owner[noise] = e.entry
        draw = e.seeds["draw"]
        if draw is None:
            continue
        owner = draw_owner.setdefault(draw, e.base)
        if owner != e.base:
            raise ValueError(
                f"draw seed {draw} of {e.entry} repeats {owner}; choose another data-set seed"
            )


def skipped_probes(spec: DatasetSpec) -> List[str]:
    """``"<class>/<axis>-<side>"`` for probe sides with no room in the field's domain."""
    out: List[str] = []
    for split in spec.splits:
        if split.kind != "probes":
            continue
        for cls in spec.world.classes.values():
            for axis in _probe_axes(cls):
                for side in ("below", "above"):
                    if axis.probe_values(side, split.bins, np.zeros(1)) is None:
                        out.append(f"{cls.name}/{axis.name}-{side}")
    return sorted(set(out))


def write_dataset(spec: DatasetSpec, entries: List[Entry], out_dir: Union[str, Path]) -> Path:
    """Write ``dataset.json`` and ``manifest.jsonl`` (one row per entry) into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, Dict[str, int]] = {}
    for e in entries:
        per_split = counts.setdefault(e.split, {})
        key = e.class_name or "session"
        per_split[key] = per_split.get(key, 0) + 1
    info = {
        "format": FORMAT,
        "name": spec.name,
        "dataset_id": spec.dataset_id,
        "world_id": spec.world.world_id,
        "seed": spec.seed,
        "duration_ms": spec.duration_ms,
        "splits": [s.to_dict() for s in spec.splits],
        "spec": spec.source,
        "world": spec.world.to_dict(),
        "world_description": spec.world.description,
        "sensoryforge": source_info(),
        "n_entries": len(entries),
        "counts": counts,
        "skipped_probes": skipped_probes(spec),
    }
    (out / "dataset.json").write_text(json.dumps(info, indent=2, sort_keys=True))
    with open(out / "manifest.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict(), sort_keys=True) + "\n")
    return out


def load_manifest(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """The rows of ``manifest.jsonl`` (``path`` is the file or its directory)."""
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
```

Then export the data-set API from `sensoryforge/world/__init__.py`: add

```python
from sensoryforge.world.dataset import (
    DatasetSpec,
    Entry,
    build_dataset,
    load_dataset,
    load_manifest,
    write_dataset,
)
```

and add `"DatasetSpec", "Entry", "build_dataset", "load_dataset", "load_manifest", "write_dataset"` to `__all__`.

- [ ] **Step 5: Add the `dataset` and `world` subcommands to `sensoryforge/cli.py`**

Add `import json` and `import yaml` to the imports. Add the handlers before `create_parser`:

```python
def cmd_dataset(args: argparse.Namespace) -> int:
    """``sensoryforge dataset build SPEC --out DIR``: write the manifest, no simulation."""
    from sensoryforge.world.dataset import build_dataset, load_dataset, write_dataset

    if args.dataset_command != "build":
        print("usage: sensoryforge dataset build SPEC --out DIR", file=sys.stderr)
        return 1
    try:
        spec = load_dataset(args.spec)
        entries = build_dataset(spec)
        out = write_dataset(spec, entries, args.out)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"Error building data set: {exc}", file=sys.stderr)
        return 1
    print(f"{spec.dataset_id}: {len(entries)} entries on world {spec.world.world_id} -> {out}")
    return 0


def cmd_world(args: argparse.Namespace) -> int:
    """``sensoryforge world validate WORLD`` / ``world sample WORLD -n N --seed S``."""
    from sensoryforge.world import load_world, sample

    if args.world_command not in ("validate", "sample"):
        print("usage: sensoryforge world {validate,sample} WORLD", file=sys.stderr)
        return 1
    try:
        world = load_world(args.world)
        if args.world_command == "validate":
            print(f"{world.world_id}  {world.name}  channels={world.channels}")
            for group, table in (("class", world.classes), ("held out", world.held_out)):
                for name, cls in table.items():
                    axes = ", ".join(a.name for a in cls.random_axes) or "-"
                    weight = "" if cls.held_out else f" weight={cls.weight:g}"
                    print(f"  {group} {name} ({cls.kind}){weight}: {axes}")
            for name in world.fixed:
                print(f"  fixed draw {name}")
            return 0
        classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
        for draw in sample(world, n=args.n, seed=args.seed, classes=classes):
            print(json.dumps(draw.to_dict(), sort_keys=True))
        return 0
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
```

In `create_parser`, after the batch parser block, add:

```python
    # Data sets and worlds
    dataset_parser = subparsers.add_parser(
        "dataset", help="Build a data set's manifest from a world (no simulation)"
    )
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command")
    build_parser = dataset_sub.add_parser(
        "build", help="Write dataset.json and manifest.jsonl"
    )
    build_parser.add_argument("spec", help="Data-set YAML (a 'dataset:' mapping)")
    build_parser.add_argument("--out", required=True, help="Output directory")

    world_parser = subparsers.add_parser("world", help="Inspect a world file")
    world_sub = world_parser.add_subparsers(dest="world_command")
    validate_world = world_sub.add_parser("validate", help="Print the world's id, classes and axes")
    validate_world.add_argument("world", help="World YAML (a 'world:' mapping)")
    sample_world = world_sub.add_parser("sample", help="Print draws as JSON lines")
    sample_world.add_argument("world", help="World YAML")
    sample_world.add_argument("-n", type=int, default=10, help="How many draws")
    sample_world.add_argument("--seed", type=int, default=0, help="Sampling seed")
    sample_world.add_argument("--classes", help="Comma-separated classes to sample from")
```

and register `"dataset": cmd_dataset, "world": cmd_world` in `main()`'s `commands`.

- [ ] **Step 6: Run the tests**

Run: `conda run -n sensoryforge python -m pytest tests/unit/test_world_dataset.py tests/integration/test_cli_world_dataset.py tests/unit/test_world_render.py -q`
Expected: all pass. If `test_long_draws_are_marked_truncated` finds no truncated `twice` entry, check `end_ms` for the `twice` draws before touching the fixture: with these ranges most two-contact episodes exceed 120 ms.

- [ ] **Step 7: Commit**

```bash
git add sensoryforge/world/dataset.py sensoryforge/world/__init__.py sensoryforge/cli.py tests/fixtures/worlds/dataset_small.yml tests/unit/test_world_dataset.py tests/integration/test_cli_world_dataset.py
git commit -m "feat(world): data sets with seeded splits, stratified test, probes and a manifest

A dataset: YAML names a world and its splits: train/validation by the
declared distribution, a Latin-hypercube test per class, out-of-range
probes, held-out classes, sessions and fixed draws, with repeats (fresh
draws) and noise_repeats (same draws, new noise). Every entry gets a draw
and a noise seed; building fails if any seed repeats. 'sensoryforge
dataset build' writes dataset.json and manifest.jsonl; 'sensoryforge
world validate|sample' inspects a world.

Refs: D-d3605dd

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: `sensoryforge batch --dataset` — the runner

**Files:**
- Create: `sensoryforge/world/runner.py`, `tests/integration/test_world_batch.py`
- Modify: `sensoryforge/cli.py` (shared sensor resolution, `batch` routing and flags), `sensoryforge/world/__init__.py` (export `run_dataset`, `read_batch_index`)

**Interfaces:**
- Consumes: `build_dataset`, `DatasetSpec`, `Entry` (Task 9); `Canvas`, `render_movie` (Task 6); `Session` (Task 5); `rng.seed53` (Task 2); `source_info`, `WORLD_ENTRY_KIND` semantics (Task 8); `SimulationEngine(config).run(stimulus, bundle_dir=, stimulus_config=, seed=, bundle_overwrite=, design_manifest=)`; `load_design`, `read_manifest` (`sensoryforge/io/design.py`).
- Produces: `task_slice(n, tasks, task_index) -> range`; `select_entries(entries, splits) -> List[Entry]`; `channel_layout(world, grid_cfg) -> Optional[List[int]]`; `to_grid_frames(movie, layout, n_grid_channels) -> Tensor`; `stimulus_payload(entry) -> dict`; `run_dataset(config, spec, out_dir, *, design_manifest=None, splits=None, tasks=1, task_index=0, entry_range=None, resume=False, log=None) -> {"ok", "failed", "skipped"}`; `read_batch_index(out_dir) -> List[dict]`. CLI: `_resolve_sensor_config(args) -> (config_dict, design_manifest, source_desc)`, `cmd_batch_dataset(args)`; `batch` gains `--dataset`, `--design`/`--preset`, `--splits`, `--tasks`, `--entries`, `--print-tasks`; `--resume` takes an optional value (a sweep's checkpoint path, or nothing with `--dataset`).

- [ ] **Step 1: Write the failing tests**

`tests/integration/test_world_batch.py`:

```python
"""`sensoryforge batch --dataset`: bundles, index, tasks, resume, seeds, failures."""

import json
from pathlib import Path

import pytest
import yaml

from sensoryforge.cli import cmd_batch, create_parser
from sensoryforge.io.bundle import load_bundle
from sensoryforge.io.design import load_design
from sensoryforge.world import rng
from sensoryforge.world.dataset import build_dataset, load_dataset
from sensoryforge.world.runner import read_batch_index, task_slice

TESTS = Path(__file__).resolve().parents[1]
FIXTURE = TESTS / "fixtures" / "design_8x8"
WORLD = TESTS / "fixtures" / "worlds" / "tactile_small.yml"


@pytest.fixture
def dataset_file(tmp_path):
    spec = {
        "dataset": {
            "name": "tiny",
            "world": str(WORLD),
            "seed": 5,
            "duration_ms": 120,
            "splits": {
                "test": {"stratified": {"bins": 1, "per_bin": 1}},
                "fixed": {"draws": ["braille_H"]},
                "sessions": {"n": 1, "duration_ms": 200},
            },
        }
    }
    path = tmp_path / "tiny.yml"
    path.write_text(yaml.safe_dump(spec))
    return path


def _batch(*argv):
    return cmd_batch(create_parser().parse_args(["batch", *map(str, argv)]))


def test_batch_writes_one_bundle_per_entry(tmp_path, dataset_file):
    out = tmp_path / "out"
    assert _batch("--design", FIXTURE, "--dataset", dataset_file, "--output", out) == 0
    entries = build_dataset(load_dataset(dataset_file))
    rows = read_batch_index(out)
    assert [r["entry"] for r in rows] == [e.entry for e in entries]
    assert {r["status"] for r in rows} == {"ok"}
    base = load_design(FIXTURE)
    for e in entries:
        bundle = load_bundle(out / e.entry)
        cfg = bundle.meta["config_json"]
        assert cfg["world"]["entry"] == e.entry
        sim = cfg["config"]["simulation"]
        assert sim["receptor_noise_seed"] == e.seeds["noise"]
        for i, pop in enumerate(cfg["config"]["populations"]):
            if base.populations[i].noise_seed is None:
                assert pop.get("noise_seed") is None
            else:
                assert pop["noise_seed"] == rng.seed53(e.seeds["noise"], "population", i)
        assert bundle.stimulus.shape[0] == int(round(e.duration_ms))
    info = json.loads((out / "batch.json").read_text())
    assert info["dataset_id"] == entries[0].dataset_id
    assert info["design"]["design_id"] == json.loads((FIXTURE / "design.json").read_text())["design_id"]
    partial = out / ".partial"
    assert not partial.exists() or not list(partial.rglob("data.h5"))


def test_splits_select_entries(tmp_path, dataset_file):
    out = tmp_path / "out"
    assert _batch("--design", FIXTURE, "--dataset", dataset_file, "--output", out, "--splits", "fixed") == 0
    assert [r["entry"] for r in read_batch_index(out)] == ["fixed/braille_H"]
    assert _batch("--design", FIXTURE, "--dataset", dataset_file, "--output", out, "--splits", "nope") == 1


def test_tasks_partition_the_entries(tmp_path, dataset_file):
    n = len(build_dataset(load_dataset(dataset_file)))
    covered = [i for t in range(3) for i in task_slice(n, 3, t)]
    assert covered == list(range(n))
    out = tmp_path / "out"
    for t in range(3):
        argv = ["--design", FIXTURE, "--dataset", dataset_file, "--output", out]
        assert _batch(*argv, "--tasks", 3, "--task-index", t) == 0
    names = sorted(p.name for p in (out / "index").iterdir())
    assert names == ["task_0000.jsonl", "task_0001.jsonl", "task_0002.jsonl"]
    assert len(read_batch_index(out)) == n


def test_print_tasks_gives_one_command_per_task(tmp_path, dataset_file, capsys):
    out = tmp_path / "o"
    argv = ["--design", FIXTURE, "--dataset", dataset_file, "--output", out]
    assert _batch(*argv, "--tasks", 4, "--print-tasks") == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4
    assert lines[2].startswith("sensoryforge batch ") and lines[2].endswith("--tasks 4 --task-index 2")
    assert all(str(FIXTURE.resolve()) in line for line in lines)
    assert not out.exists()


def test_rerun_replaces_resume_skips_and_partials_are_cleaned(tmp_path, dataset_file):
    out = tmp_path / "out"
    argv = ["--design", FIXTURE, "--dataset", dataset_file, "--output", out, "--entries", "0:2"]
    assert _batch(*argv) == 0
    first = read_batch_index(out)[0]["entry"]
    marker = out / first / "marker.txt"
    marker.write_text("x")
    stale = out / ".partial" / first
    stale.mkdir(parents=True)
    (stale / "junk").write_text("x")
    assert _batch(*argv, "--resume") == 0
    assert marker.exists()  # resumed: the finished bundle was skipped
    assert _batch(*argv) == 0
    assert not marker.exists()  # rerun: the bundle was replaced
    assert not (stale / "junk").exists()


def test_a_failed_entry_is_recorded_and_the_exit_status_says_so(tmp_path, dataset_file, monkeypatch):
    import sensoryforge.world.runner as runner

    real = runner.render_movie

    def flaky(item, *args, **kwargs):
        if getattr(item, "class_name", None) == "edges":
            raise ValueError("boom")
        return real(item, *args, **kwargs)

    monkeypatch.setattr(runner, "render_movie", flaky)
    out = tmp_path / "out"
    argv = ["--design", FIXTURE, "--dataset", dataset_file, "--output", out, "--entries", "0:3"]
    assert _batch(*argv) == 1
    rows = {r["entry"]: r for r in read_batch_index(out)}
    assert rows["test/edges/0000"]["status"] == "failed"
    assert "boom" in rows["test/edges/0000"]["error"]
    assert rows["test/dots/0000"]["status"] == "ok"
    assert not (out / "test" / "edges" / "0000").exists()


def test_a_sweep_batch_still_needs_its_config(capsys):
    assert _batch() == 1
    assert "batch needs a config" in capsys.readouterr().err
```

- [ ] **Step 2: Run to see them fail**

Run: `conda run -n sensoryforge python -m pytest tests/integration/test_world_batch.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sensoryforge.world.runner'`.

- [ ] **Step 3: Implement `sensoryforge/world/runner.py`**

```python
"""Run a data set through the sensor: one bundle per entry (spec §7).

One process builds the engine once (receptive fields loaded once) and loops
over its entries. Each entry is rendered in float64 on the design's canvas,
cast to float32, simulated with its own noise seeds, and written to
``<out>/.partial/<entry>/`` before an atomic rename to ``<out>/<entry>/``.
Each task appends to its own ``index/task_<i>.jsonl``, so array tasks never
share a file.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch

from sensoryforge.config.schema import GridConfig, SensoryForgeConfig
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.provenance import source_info
from sensoryforge.world import rng
from sensoryforge.world.dataset import DatasetSpec, Entry, build_dataset
from sensoryforge.world.render import Canvas, render_movie
from sensoryforge.world.sampling import Session
from sensoryforge.world.schema import World

WORLD_ENTRY_KIND = "sensoryforge_world_entry"
#: What one failing entry may raise without stopping the run.
_ENTRY_ERRORS = (RuntimeError, ValueError, OSError, KeyError, IndexError, TypeError)


def task_slice(n: int, tasks: int, task_index: int) -> range:
    """The contiguous slice of ``n`` entries that task ``task_index`` of ``tasks`` runs."""
    if tasks < 1:
        raise ValueError(f"--tasks must be >= 1, got {tasks}")
    if not 0 <= task_index < tasks:
        raise ValueError(f"--task-index must be in [0, {tasks}), got {task_index}")
    return range(task_index * n // tasks, (task_index + 1) * n // tasks)


def select_entries(entries: Sequence[Entry], splits: Optional[Sequence[str]] = None) -> List[Entry]:
    """The entries of the named splits (all when ``splits`` is empty)."""
    if not splits:
        return list(entries)
    known = {e.split for e in entries}
    unknown = sorted(set(splits) - known)
    if unknown:
        raise ValueError(f"--splits {unknown} are not in this data set (it has {sorted(known)})")
    keep = set(splits)
    return [e for e in entries if e.split in keep]


def channel_layout(world: World, grid: GridConfig) -> Optional[List[int]]:
    """Where each world channel goes among the grid's channels (``None``: plain frames)."""
    grid_channels = list(grid.channels or ["value"])
    if len(grid_channels) == 1:
        if len(world.channels) > 1:
            raise ValueError(
                f"the world has channels {world.channels} but grid {grid.name!r} has one plane"
            )
        return None
    missing = [c for c in world.channels if c not in grid_channels]
    if missing:
        raise ValueError(
            f"world channels {missing} are not among grid {grid.name!r}'s channels {grid_channels}"
        )
    return [grid_channels.index(c) for c in world.channels]


def to_grid_frames(movie: torch.Tensor, layout: Optional[List[int]], n_grid_channels: int) -> torch.Tensor:
    """``[T, H, W]`` as is, or the world's planes placed in a ``[T, C_grid, H, W]`` stack."""
    if layout is None:
        return movie
    if movie.ndim == 3:
        movie = movie.unsqueeze(1)
    out = torch.zeros(
        (movie.shape[0], n_grid_channels) + tuple(movie.shape[2:]),
        dtype=movie.dtype,
        device=movie.device,
    )
    for c, target in enumerate(layout):
        out[:, target] = movie[:, c]
    return out


def stimulus_payload(entry: Entry) -> Dict[str, Any]:
    """The ``stimulus_config`` that marks a bundle as this world entry (spec §7.3)."""
    item = entry.item
    if isinstance(item, Session):
        layer: Any = [[start, draw.to_layer()] for start, draw in item.items]
    else:
        layer = item.to_layer()
    return {"kind": WORLD_ENTRY_KIND, "entry": entry.to_dict(), "layer": layer}


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def run_dataset(
    config: SensoryForgeConfig,
    spec: DatasetSpec,
    out_dir: Union[str, Path],
    *,
    design_manifest: Optional[Dict[str, Any]] = None,
    splits: Optional[Sequence[str]] = None,
    tasks: int = 1,
    task_index: int = 0,
    entry_range: Optional[slice] = None,
    resume: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, int]:
    """Simulate a data set's entries, one bundle each (spec §7.2).

    Args:
        config: The sensor (grids and populations); mutated per entry
            (noise seeds), so pass one that only this run uses.
        spec: The data set.
        out_dir: Output directory (created).
        design_manifest: A design directory's manifest, stamped into every bundle.
        splits: Only these splits.
        tasks: Split the (selected) entries into this many contiguous tasks...
        task_index: ...and run this one.
        entry_range: Run this slice of the selected entries instead of a task.
        resume: Skip entries whose bundle already holds ``data.h5``.
        log: Called with one line per entry.

    Returns:
        ``{"ok": int, "failed": int, "skipped": int}``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = select_entries(build_dataset(spec), splits)
    if entry_range is not None:
        chosen = entries[entry_range]
    else:
        chosen = [entries[i] for i in task_slice(len(entries), tasks, task_index)]
    provenance = source_info()
    _write_json_atomic(
        out / "batch.json",
        {
            "format": "sensoryforge-batch/1",
            "dataset_id": spec.dataset_id,
            "dataset_name": spec.name,
            "world_id": spec.world.world_id,
            "n_entries": len(entries),
            "splits": sorted({e.split for e in entries}),
            "design": design_manifest,
            "config": config.to_dict(),
            "sensoryforge": provenance,
        },
    )
    grid = config.grids[0]
    layout = channel_layout(spec.world, grid)
    n_grid_channels = len(grid.channels or ["value"])
    engine = SimulationEngine(config)
    canvas = Canvas.from_grid_config(grid)
    render_device = torch.device("cpu") if engine.device.type == "mps" else engine.device
    base_noise = [pop.noise_seed for pop in config.populations]
    index_path = out / "index" / f"task_{task_index:04d}.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    design_id = (design_manifest or {}).get("design_id")
    summary = {"ok": 0, "failed": 0, "skipped": 0}

    for entry in chosen:
        final = out / entry.entry
        if resume and (final / "data.h5").exists():
            summary["skipped"] += 1
            continue
        started = time.perf_counter()
        row: Dict[str, Any] = {
            "entry": entry.entry,
            "bundle": entry.entry,
            "task": task_index,
            "design_id": design_id,
            "sensoryforge_sha": provenance["sha"],
        }
        try:
            noise = int(entry.seeds["noise"])
            config.simulation.receptor_noise_seed = noise
            for i, (pop, base) in enumerate(zip(config.populations, base_noise)):
                pop.noise_seed = None if base is None else rng.seed53(noise, "population", i)
            movie = render_movie(
                entry.item,
                canvas,
                config.simulation.dt_ms,
                entry.duration_ms,
                dtype=torch.float64,
                device=render_device,
            )
            frames = to_grid_frames(movie, layout, n_grid_channels)
            frames = frames.to(device=engine.device, dtype=torch.float32)
            partial = out / ".partial" / entry.entry
            if partial.exists():
                shutil.rmtree(partial)
            partial.parent.mkdir(parents=True, exist_ok=True)
            engine.run(
                frames.unsqueeze(0),
                bundle_dir=partial,
                stimulus_config=stimulus_payload(entry),
                seed=noise,
                bundle_overwrite=True,
                design_manifest=design_manifest,
            )
            if final.exists():
                shutil.rmtree(final)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(partial, final)
            row.update(status="ok", error=None)
            summary["ok"] += 1
        except _ENTRY_ERRORS as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            summary["failed"] += 1
        row["seconds"] = round(time.perf_counter() - started, 3)
        row["finished_at"] = datetime.now(timezone.utc).isoformat()
        with open(index_path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        if log is not None:
            log(f"{row['status']:>6}  {entry.entry}  ({row['seconds']:.2f} s)")
    return summary


def read_batch_index(out_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Every task's index rows merged: one row per entry, the latest run winning.

    Rows keep the order in which entries first appear (task files in task
    order), which is the data set's order for a complete run.
    """
    latest: Dict[str, Dict[str, Any]] = {}
    for path in sorted((Path(out_dir) / "index").glob("task_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            old = latest.get(row["entry"])
            if old is None or row["finished_at"] >= old["finished_at"]:
                latest[row["entry"]] = row
    return list(latest.values())
```

Export from `sensoryforge/world/__init__.py`: `from sensoryforge.world.runner import read_batch_index, run_dataset` and add both to `__all__`.

- [ ] **Step 4: Wire the CLI** (`sensoryforge/cli.py`)

1. Extract the sensor resolution from `cmd_run` into a helper (place it above `cmd_run`):

```python
def _resolve_sensor_config(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], str]:
    """The sensor a command runs: a design directory, a preset or a config file.

    A config file given with ``--design``/``--preset`` is deep-merged over it.

    Returns:
        ``(config dict, design manifest or None, a description for messages)``.

    Raises:
        ValueError: When none of the three is given.
    """
    preset_name = getattr(args, "preset", None)
    design_dir = getattr(args, "design", None)
    config_path = getattr(args, "config", None)
    design_manifest: Optional[Dict[str, Any]] = None
    if design_dir:
        from sensoryforge.io.design import load_design, read_manifest

        design_manifest = read_manifest(design_dir)
        config = load_design(design_dir).to_dict()
        if config_path:
            config = _deep_merge_preset(config, load_config_file(config_path))
    elif preset_name:
        from sensoryforge.presets import load_preset

        config = load_preset(preset_name)
        if config_path:
            config = _deep_merge_preset(config, load_config_file(config_path))
    elif config_path:
        config = load_config_file(config_path)
    else:
        raise ValueError("no config file and no --preset given (no --design given either)")
    source_desc = config_path or (
        f"--design {design_dir}" if design_dir else f"--preset {preset_name}"
    )
    return config, design_manifest, source_desc
```

In `cmd_run`, replace the block from `preset_name = getattr(args, "preset", None)` through the `else: print("Error running simulation: no config file and no --preset ..."); return 1` with:

```python
        try:
            config, design_manifest, source_desc = _resolve_sensor_config(args)
        except ValueError as exc:
            print(f"Error running simulation: {exc}", file=sys.stderr)
            return 1
```

and delete the later `source_desc = args.config or (...)` assignment (it is computed by the helper now). Run `grep -n "preset_name\|design_dir" sensoryforge/cli.py` and replace any remaining use inside `cmd_run` with `getattr(args, "preset", None)` / `getattr(args, "design", None)`. Add `Tuple` to the `typing` import.

2. The batch parser: make `config` optional and add the data-set flags:

```python
    batch_parser.add_argument(
        "config",
        nargs="?",
        help="Sweep batch YAML; with --dataset, an optional sensor config "
        "merged over --design/--preset",
    )
    batch_sensor = batch_parser.add_mutually_exclusive_group()
    batch_sensor.add_argument("--design", help="A design directory (with --dataset)")
    batch_sensor.add_argument("--preset", help="A preset name (with --dataset)")
    batch_parser.add_argument("--dataset", help="Run a world data set (dataset: YAML)")
    batch_parser.add_argument("--splits", help="Comma-separated splits to run (with --dataset)")
    batch_parser.add_argument(
        "--tasks", type=int, default=1, help="Split the entries into this many tasks (with --dataset)"
    )
    batch_parser.add_argument("--entries", help="Run entries a:b only (with --dataset)")
    batch_parser.add_argument(
        "--print-tasks",
        action="store_true",
        help="Print one 'sensoryforge batch' command per task and exit (with --dataset)",
    )
```

and change `--resume` to take an optional value:

```python
    batch_parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help="Sweep batch: the checkpoint.json to resume from. With --dataset: "
        "no value; skip entries whose bundle is complete.",
    )
```

3. At the top of `cmd_batch`, before `try:`:

```python
    if getattr(args, "dataset", None):
        return cmd_batch_dataset(args)
    if args.design or args.preset:
        print("Error: --design/--preset need --dataset", file=sys.stderr)
        return 1
    if not args.config:
        print("Error: batch needs a config file (or --dataset with a sensor)", file=sys.stderr)
        return 1
    if args.resume is True:
        print("Error: --resume for a sweep batch needs the checkpoint path", file=sys.stderr)
        return 1
```

4. Add the data-set handler and its helpers after `cmd_batch`:

```python
def _parse_range(text: str) -> slice:
    """``"a:b"`` -> ``slice(a, b)``."""
    try:
        start, stop = (int(part) for part in text.split(":"))
    except ValueError:
        raise ValueError(f"--entries must look like a:b, got {text!r}") from None
    if start < 0 or stop < start:
        raise ValueError(f"--entries needs 0 <= a <= b, got {text!r}")
    return slice(start, stop)


def _task_commands(args: argparse.Namespace) -> List[str]:
    """One ``sensoryforge batch`` command per task, with absolute paths."""
    import shlex

    base = ["sensoryforge", "batch"]
    if args.config:
        base.append(shlex.quote(str(Path(args.config).resolve())))
    if args.design:
        base += ["--design", shlex.quote(str(Path(args.design).resolve()))]
    if args.preset:
        base += ["--preset", shlex.quote(args.preset)]
    base += ["--dataset", shlex.quote(str(Path(args.dataset).resolve()))]
    base += ["--output", shlex.quote(str(Path(args.output).resolve()))]
    if args.splits:
        base += ["--splits", shlex.quote(args.splits)]
    if args.device:
        base += ["--device", args.device]
    if args.resume:
        base.append("--resume")
    return [" ".join(base + ["--tasks", str(args.tasks), "--task-index", str(i)]) for i in range(args.tasks)]


def cmd_batch_dataset(args: argparse.Namespace) -> int:
    """``sensoryforge batch --dataset``: one bundle per data-set entry (spec §7)."""
    from sensoryforge.world.dataset import load_dataset
    from sensoryforge.world.runner import run_dataset

    if not args.output:
        print("Error: --output is required with --dataset", file=sys.stderr)
        return 1
    if args.print_tasks:
        for line in _task_commands(args):
            print(line)
        return 0
    try:
        config_dict, design_manifest, source_desc = _resolve_sensor_config(args)
        if args.device:
            config_dict.setdefault("simulation", {})["device"] = args.device
        config = SensoryForgeConfig.from_dict(config_dict)
        spec = load_dataset(args.dataset)
        splits = [s.strip() for s in args.splits.split(",") if s.strip()] if args.splits else None
        entry_range = _parse_range(args.entries) if args.entries else None
        summary = run_dataset(
            config,
            spec,
            args.output,
            design_manifest=design_manifest,
            splits=splits,
            tasks=args.tasks,
            task_index=args.task_index or 0,
            entry_range=entry_range,
            resume=bool(args.resume),
            log=print,
        )
    except (ValueError, OSError, KeyError, yaml.YAMLError) as exc:
        print(f"Error running data set: {exc}", file=sys.stderr)
        return 1
    print(
        f"{summary['ok']} ok, {summary['failed']} failed, {summary['skipped']} skipped "
        f"({source_desc}, data set {spec.dataset_id} -> {args.output})"
    )
    return 1 if summary["failed"] else 0
```

Add `List` to the `typing` import.

- [ ] **Step 5: Run the batch tests and the existing batch/CLI suites**

Run: `conda run -n sensoryforge python -m pytest tests/integration/test_world_batch.py tests/unit/test_batch_executor.py tests/unit/test_batch_executor_bundles.py tests/unit/test_batch_executor_canonical.py tests/integration/test_cli_run_design.py tests/integration/test_cli_stimulus_block.py tests/integration/test_examples_smoke.py -q`
Expected: all pass (the old sweep path is unchanged).

Two engine facts the per-entry seeds rely on, which `test_batch_writes_one_bundle_per_entry` checks: `SimulationEngine.run` reads `config.simulation.receptor_noise_seed` and each population's `noise_seed` at run time (`core/simulation_engine.py` ~1006 and ~1119), and `PopulationConfig.to_dict()` writes `noise_seed` when it is set. If the bundle's `config.json` does not show the entry's seeds, find which of the two is false before changing the runner.

- [ ] **Step 6: Commit**

```bash
git add sensoryforge/world/runner.py sensoryforge/world/__init__.py sensoryforge/cli.py tests/integration/test_world_batch.py
git commit -m "feat(cli): batch --dataset runs a world data set, one bundle per entry

One process builds the engine once and loops over its entries; each is
rendered in float64 on the design's canvas, cast to float32 and simulated
with its own receptor (and population) noise seeds, then written
atomically. --tasks/--task-index split the entries into contiguous array
tasks, --print-tasks emits one command per task (for an LSF manifest),
--resume skips finished bundles, each task appends to its own index file,
and a failed entry makes the exit status non-zero. Without --dataset,
batch runs the old sweep executor unchanged.

Refs: D-cccbd6a

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Pressure-simulation's contract tests, here and under PS's stack

**Files:**
- Create: `tests/contract/test_world_contract.py`

**Interfaces:**
- Consumes: everything above; `read_manifest` (`sensoryforge/io/design.py`); `cmd_batch`, `create_parser` (`sensoryforge/cli.py`).
- Produces: the eight tests of PS's brief §6, named `test_<n>_...`, which Task 12's contract document cites.

- [ ] **Step 1: Write the contract tests**

`tests/contract/test_world_contract.py`:

```python
"""Pressure-simulation's contract with the world engine (its brief, §6), run here.

PS's Phase 2b runs these against the tagged sha; running them in SensoryForge
first means the tag is known to pass them. Each test carries the brief's
number. docs/reference/world_contract.md cites them.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
import yaml

from sensoryforge.cli import cmd_batch, create_parser
from sensoryforge.io.bundle import load_bundle
from sensoryforge.io.design import load_design, read_manifest
from sensoryforge.provenance import source_info
from sensoryforge.world import (
    Canvas,
    Draw,
    Session,
    build_dataset,
    load_dataset,
    load_world,
    movie_times,
    read_batch_index,
    render,
    render_movie,
    sample,
    session,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "design_8x8"
WORLDS = ROOT / "tests" / "fixtures" / "worlds"
WORLD_PATH = WORLDS / "tactile_small.yml"
WORLD = load_world(WORLD_PATH)

_DIGEST_SCRIPT = """
import hashlib, json, sys, torch
from sensoryforge.world import Canvas, load_world, render, sample
world = load_world(sys.argv[1])
draws = sample(world, n=50, seed=7)
frames = render(draws, Canvas.from_grid(16, 16, 0.1), torch.arange(0.0, 120.0, 5.0),
                dtype=torch.float64)
records = json.dumps([d.to_dict() for d in draws], sort_keys=True).encode()
print(json.dumps({"draws": hashlib.sha256(records).hexdigest(),
                  "frames": hashlib.sha256(frames.numpy().tobytes()).hexdigest()}))
"""


def _digests(draws, frames):
    records = json.dumps([d.to_dict() for d in draws], sort_keys=True).encode()
    return {
        "draws": hashlib.sha256(records).hexdigest(),
        "frames": hashlib.sha256(frames.numpy().tobytes()).hexdigest(),
    }


def _digests_in_a_new_process():
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    out = subprocess.run(
        [sys.executable, "-c", _DIGEST_SCRIPT, str(WORLD_PATH)],
        capture_output=True, text=True, env=env, cwd=ROOT, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_1_determinism_across_processes_and_batch_sizes():
    first, second = _digests_in_a_new_process(), _digests_in_a_new_process()
    assert first == second
    draws = sample(WORLD, n=50, seed=7)
    canvas = Canvas.from_grid(16, 16, 0.1)
    times = torch.arange(0.0, 120.0, 5.0)
    frames = render(draws, canvas, times, dtype=torch.float64)
    assert _digests(draws, frames) == first
    alone = sample(WORLD, indices=[31], seed=7)[0]
    assert alone.to_dict() == draws[31].to_dict()
    assert torch.equal(render([alone], canvas, times, dtype=torch.float64)[0], frames[31])


@pytest.fixture(scope="module")
def batch_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("contract")
    spec_path = root / "dataset.yml"
    spec_path.write_text(yaml.safe_dump({"dataset": {
        "name": "contract",
        "world": str(WORLD_PATH),
        "seed": 2026,
        "duration_ms": 120,
        "splits": {
            "test": {"stratified": {"bins": 1, "per_bin": 1}},
            "validation": {"n": 2, "noise_repeats": 2},
            "sessions": {"n": 1, "duration_ms": 250},
            "fixed": {"draws": ["braille_H"]},
        },
    }}))
    sensor = root / "sensor.yml"
    sensor.write_text(yaml.safe_dump({"simulation": {"receptor_noise_std": 0.05}}))
    out = root / "run"
    argv = ["batch", str(sensor), "--design", str(FIXTURE), "--dataset", str(spec_path),
            "--output", str(out)]
    assert cmd_batch(create_parser().parse_args(argv)) == 0
    spec = load_dataset(spec_path)
    return {"out": out, "spec": spec, "entries": build_dataset(spec),
            "spec_path": spec_path, "sensor": sensor}


def _item_from_bundle(bundle_dir):
    """What PS does: rebuild the draw (or session) from the bundle's own record."""
    record = json.loads((bundle_dir / "stimuli" / "stimulus.json").read_text())["entry"]["draw"]
    if record.get("sampling") == "session":
        return Session.from_dict(record, WORLD)
    return Draw.from_dict(record, WORLD)


def test_2_the_bundle_records_exactly_the_in_process_render(batch_run):
    config = load_design(FIXTURE)
    canvas = Canvas.from_grid_config(config.grids[0])
    for e in batch_run["entries"]:
        bundle_dir = batch_run["out"] / e.entry
        item = _item_from_bundle(bundle_dir)
        want = render_movie(
            item, canvas, config.simulation.dt_ms, e.duration_ms, dtype=torch.float64
        ).to(torch.float32)
        assert torch.equal(load_bundle(bundle_dir).stimulus, want), e.entry


def test_3_windows_agree_with_movies():
    draws = sample(WORLD, n=8, seed=3)
    canvas = Canvas.from_grid(8, 8, 0.15)
    movie = render(draws, canvas, movie_times(1.0, 120.0), dtype=torch.float64)
    triples = torch.tensor([[42.0, 50.0, 58.0]] * 8, dtype=torch.float64)
    windows = render(draws, canvas, triples, dtype=torch.float64)
    for j, step in enumerate((42, 50, 58)):
        assert torch.equal(windows[:, j], movie[:, step])


def test_4_noise_seeds(batch_run, tmp_path):
    validation = [e for e in batch_run["entries"] if e.split == "validation"]
    a, b = validation[0], validation[1]  # one draw, two noise repeats
    assert a.seeds["draw"] == b.seeds["draw"] and a.seeds["noise"] != b.seeds["noise"]
    bundle_a = load_bundle(batch_run["out"] / a.entry)
    bundle_b = load_bundle(batch_run["out"] / b.entry)
    assert torch.equal(bundle_a.stimulus, bundle_b.stimulus)
    assert any(
        not torch.equal(bundle_a.populations[p]["filtered"], bundle_b.populations[p]["filtered"])
        for p in bundle_a.populations
    )
    rerun = tmp_path / "rerun"
    argv = ["batch", str(batch_run["sensor"]), "--design", str(FIXTURE), "--dataset",
            str(batch_run["spec_path"]), "--output", str(rerun), "--splits", "validation",
            "--entries", "0:1"]
    assert cmd_batch(create_parser().parse_args(argv)) == 0
    again = load_bundle(rerun / a.entry)
    for name, data in bundle_a.populations.items():
        for key, tensor in data.items():
            assert torch.equal(tensor, again.populations[name][key]), (name, key)


def test_5_splits_strata_and_probes():
    spec = load_dataset(WORLDS / "dataset_small.yml")
    entries = build_dataset(spec)
    owners = {}
    for e in entries:
        if e.seeds["draw"] is not None:
            owners.setdefault(e.seeds["draw"], set()).add(e.split)
    assert all(len(splits) == 1 for splits in owners.values())
    noise = [e.seeds["noise"] for e in entries]
    assert len(noise) == len(set(noise))
    for cls in spec.world.classes.values():
        rows = [e for e in entries if e.split == "test" and e.class_name == cls.name]
        for axis in cls.random_axes:
            counts = Counter(e.bins[axis.name] for e in rows)
            if axis.support() is None:
                assert set(counts.values()) == {2}, (cls.name, axis.name)
            else:
                assert max(counts.values()) - min(counts.values()) <= 1
    probes = [e for e in entries if e.split == "probes"]
    assert probes
    for e in probes:
        axis = e.item.spec.axes[e.probe["axis"]]
        value = e.item.values[e.probe["axis"]]
        assert e.bins[e.probe["axis"]] == e.probe["side"]
        assert value < axis.lo if e.probe["side"] == "below" else value > axis.hi


def test_6_one_draw_on_40x40_and_80x80():
    draws = sample(WORLD, n=6, seed=10)
    times = torch.tensor([30.0, 60.0], dtype=torch.float64)
    small = render(draws, Canvas.from_grid(40, 40, 0.15), times, dtype=torch.float64)
    large = render(draws, Canvas.from_grid(80, 80, 0.15), times, dtype=torch.float64)
    torch.testing.assert_close(small, large[:, :, 20:60, 20:60], atol=1e-12, rtol=0)


def test_7_session_quiet_stretches_are_exactly_zero():
    s = session(WORLD, duration_ms=600.0, seed=17)
    times = movie_times(1.0, 600.0)
    frames = render_movie(s, Canvas.from_grid(8, 8, 0.15), 1.0, 600.0, dtype=torch.float64)
    contact = torch.zeros(len(times), dtype=torch.bool)
    for start, draw in s.items:
        for phase, a, b in draw.timeline:
            if phase in ("touch", "hold", "slide", "release"):
                contact |= (times >= start + a) & (times < start + b)
    assert torch.count_nonzero(frames[~contact]) == 0
    assert torch.count_nonzero(frames[contact]) > 0


def test_8_every_bundle_carries_its_provenance(batch_run):
    manifest = read_manifest(FIXTURE)
    sha = source_info()["sha"]
    assert re.fullmatch(r"[0-9a-f]{40}", sha)
    for e in batch_run["entries"]:
        bundle_dir = batch_run["out"] / e.entry
        bundle = load_bundle(bundle_dir)
        cfg = bundle.meta["config_json"]
        assert cfg["design"] == manifest
        assert cfg["world"] == {
            "world_id": WORLD.world_id, "dataset_id": batch_run["spec"].dataset_id,
            "entry": e.entry,
        }
        assert cfg["sensoryforge_sha"] == sha and bundle.meta["sensoryforge_sha"] == sha
        payload = json.loads((bundle_dir / "stimuli" / "stimulus.json").read_text())
        assert payload["entry"] == json.loads(json.dumps(e.to_dict()))
    rows = read_batch_index(batch_run["out"])
    assert {r["sensoryforge_sha"] for r in rows} == {sha}
    assert {r["design_id"] for r in rows} == {manifest["design_id"]}
```

- [ ] **Step 2: Run them in SensoryForge's env**

Run: `conda run -n sensoryforge python -m pytest tests/contract/test_world_contract.py -q`
Expected: 8 passed. Any failure here is a defect in Tasks 1–10, not in the test: find it with `superpowers:systematic-debugging`, fix it in the owning module, and re-run that task's tests too.

- [ ] **Step 3: Sabotage check — the contract tests must bite**

Make each change, run the named test, see it FAIL, then revert with `git checkout -- <file>`:

1. In `sensoryforge/world/rng.py`, change `_SECOND` to `np.uint64(0xD1B54A32D192ED05)` → `tests/unit/test_world_rng.py::test_hashes_are_pinned` must fail. (`test_1_…` will still pass: it compares two processes running the same code, so it pins determinism, not stability across versions — that is the pinned-hash test's job.)
2. In `sensoryforge/world/runner.py`, change `dtype=torch.float64` in the `render_movie` call to `torch.float32` → `test_2_…` must fail.
3. In `sensoryforge/world/runner.py`, comment out `config.simulation.receptor_noise_seed = noise` → `test_4_…` must fail.
4. In `sensoryforge/stimuli/episode.py`, change `active = (local >= 0) & (k < contacts)` to `active = k < contacts` → `test_7_…` must fail.

If a sabotage does not make its test fail, the test does not pin its guarantee: strengthen the test before going on.

- [ ] **Step 4: Run under pressure-simulation's stack (Python 3.10, torch 2.2.2)**

Nothing is installed: the worktree goes on `PYTHONPATH`, ahead of the pinned copy in `bio-encoding`'s site-packages.

```bash
PYTHONPATH="$PWD" conda run -n bio-encoding python -c "import sensoryforge, torch, sys; print(sensoryforge.__file__, torch.__version__, sys.version.split()[0])"
```

Expected: a path inside this worktree, `2.2.2`, `3.10.18`. Then:

```bash
PYTHONPATH="$PWD" conda run -n bio-encoding python -m pytest -q -p no:cacheprovider tests/unit/test_world_rng.py tests/unit/test_world_axes.py tests/unit/test_world_kernel.py tests/unit/test_world_schema.py tests/unit/test_world_kinds.py tests/unit/test_world_sampling.py tests/unit/test_world_render.py tests/unit/test_world_dataset.py tests/unit/test_layered_episode.py tests/unit/test_layered_golden.py tests/unit/test_provenance.py tests/unit/test_bundle_world_entry.py tests/integration/test_world_batch.py tests/contract/test_world_contract.py
```

Expected: all pass. A failure only here is a torch-2.2/Python-3.10 incompatibility (an API added after torch 2.2, or syntax newer than 3.10): fix it in the module so both envs pass.

- [ ] **Step 5: Commit**

```bash
git add tests/contract/test_world_contract.py
git commit -m "test(contract): pressure-simulation's eight world-engine contract tests

The brief's section 6 tests, run in SensoryForge so the tag is known to
pass them: determinism across processes and batch sizes, bundle frames
equal to the in-process render of the bundle's own record, windows equal
to movies, per-entry noise seeds, splits/strata/probes, 40x40 vs 80x80,
exactly-zero quiet in sessions, and provenance in every bundle. Each was
sabotaged once and failed. They also pass under pressure-simulation's
bio-encoding env (Python 3.10, torch 2.2.2) with the worktree on
PYTHONPATH.

Finding: the world engine passes pressure-simulation's eight contract tests in SensoryForge's env (Python 3.11, torch 2.5.1) and in bio-encoding (Python 3.10, torch 2.2.2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(Write the sabotage results you actually saw into the body; if any test needed strengthening, say which.)

---

### Task 12: Documentation, the contract, version 1.1.0, the tag

**Files:**
- Create: `docs/user_guide/worlds.md`, `docs/reference/world_contract.md`, `.claude/rules/world-engine.md`
- Modify: `docs/user_guide/designing_stimuli.md`, `mkdocs.yml` (nav), `CLAUDE.md`, `docs_root/DECISIONS.md`, `CHANGELOG.md`, `pyproject.toml` (version), `sensoryforge/__init__.py` (`__version__`)

**Interfaces:**
- Consumes: the API and test names of Tasks 1–11 (cite them exactly; re-read the code for any name before quoting it).
- Produces: the contract document pressure-simulation's Phase 2b plan is written against; the tag `v1.1.0`.

- [ ] **Step 1: Write `docs/reference/world_contract.md`**

```markdown
# World engine contract

This page is what pressure-simulation's Phase 2b is written against: what SensoryForge
**v1.1.0** guarantees about declared worlds, sampling, rendering, data sets and batch runs,
and the SensoryForge test that pins each guarantee. Where this page and the code disagree,
the tests decide.

## Getting it

```bash
conda run -n bio-encoding pip install --no-deps --force-reinstall \
  "sensoryforge @ git+file:///Users/benefron/sensoryforge@v1.1.0"
```

The in-process API needs Python ≥ 3.10, torch ≥ 2.2, numpy, PyYAML and h5py; it is tested
on Python 3.10.18 / torch 2.2.2 (`bio-encoding`) and Python 3.11 / torch 2.5.1. A bundle
records the sha of the SensoryForge that wrote it (`config.json["sensoryforge_sha"]`), read
from pip's `direct_url.json` in a pinned install.

## 1. A world

A world file has one key, `world:`. The user guide (`docs/user_guide/worlds.md`) explains
each part; the rules that matter to a caller:

- **Classes** (`classes:`) carry a `weight`; **held-out classes** (`held_out:`) carry none
  and are sampled only when named. A class has a `kind`: `layered` (default) or `quiet`
  (renders exactly 0 for `quiet_ms`); plugins may register more.
- A `layered` class is a layered **layer** (`shape`, `pattern`, `modulation`, optional
  `motion`) whose fields its **axes** draw. An axis is `{value: v}`, `{range: [lo, hi]}`
  (`dist: uniform | log_uniform`, `int: true`, `circular: true`, `probes: false`),
  `{values: [...], weights: [...]}`, or `{dist: <registered>}` (`braille_cells`: the 63
  non-empty cells as dot-number strings, uniform).
- An axis name binds, in order, to an episode field (`delay_ms`, `touch_ms`, `hold_ms`,
  `slide_ms`, `release_ms`, `contacts`, `pause_ms`, `speed_mm_per_ms`, `direction_deg`),
  `amplitude`, `x_mm`/`y_mm`, then a shape, pattern or modulation field by bare name, or a
  dotted path (`shape.width_mm`). Ambiguous or unknown names fail at load time.
- Precedence: a class's own axes, then the fields its layer fixes, then the world's
  `defaults`, then built-ins (all durations 0, `contacts` 1, `amplitude` 1, `x_mm`/`y_mm` 0).
- `fixed_draws:` names draws by explicit values; unnamed axes take their midpoint
  (geometric for `log_uniform`); values outside a range are allowed and listed in the
  draw's `out_of_range`.
- **Identity:** `world.world_id` is `w-` + 12 hex digits of SHA-256 over the normalised
  world (everything except `description`).

## 2. One draw's time course

Time 0 is the start of the entry. A draw is quiet for `delay_ms`, then `contacts` contacts,
`pause_ms` apart; each is a linear ramp up over `touch_ms`, still for `hold_ms`, moving for
`slide_ms`, a linear ramp down over `release_ms`. Motion runs only during slides, at
`speed_mm_per_ms` towards `direction_deg` (0° = +x, 90° = +y), spread over all contacts so a
re-touch lands where the last contact ended. A modulation multiplies the whole contact
envelope, measured from each touch: `sine` (`frequency_hz`, `depth`, `phase_deg`;
1 − depth·(1 − cos(2πft + φ))/2) or `pulses` (`rate_hz`, `duty`, `edge_ms`, `depth`). Before
0, after the draw's `end_ms`, in lead-ins and in pauses the stimulus is **exactly 0**.

## 3. Sampling

```python
from sensoryforge.world import load_world, sample, session
world = load_world("world.yml")
draws = sample(world, n=1000, seed=7)                    # draw i depends only on (world, 7, i)
same  = sample(world, indices=range(500, 600), seed=7)
some  = sample(world, n=200, seed=7, classes=["dots"])   # held-out classes may be named
demo  = world.fixed_draw("braille_H")
s     = session(world, duration_ms=10_000, seed=7, index=0)
```

A draw's record (`draw.to_dict()`, rebuilt by `Draw.from_dict(record, world)`):
`world_id, seed, index, draw_seed, class, sampling, values, timeline, end_ms, out_of_range`.
`values` holds every axis, constants included. A session's record:
`world_id, sampling: "session", seed, index, session_seed, duration_ms, items: [[start_ms,
draw], ...], truncated, quiet_fraction, end_ms`; its draws are laid end to end.

## 4. Rendering

```python
from sensoryforge.world import Canvas, render, render_movie, movie_times
canvas = Canvas(xx, yy)                                   # any coordinates, mm
canvas = Canvas.from_grid(40, 40, 0.15)                   # SF's centred layout, dim 0 = x
frames = render(draws, canvas, times_ms, dtype=torch.float64, device="cpu")
movie  = render_movie(draw, canvas, dt_ms=1.0, duration_ms=500.0, dtype=torch.float64)
```

- `times_ms` is `[K]` (shared) or `[n, K]` (per draw), ms from each item's start.
- Output `[n, K, *S]` (`S` = the canvas shape), or `[n, K, C, *S]` for a world with `C > 1`
  channels. `render_movie` uses `t_k = k · dt_ms`.
- **Coordinates:** SensoryForge's canvases are centred with x on dim 0. To render on
  pressure-simulation's corner-origin `[y, x]` frames, pass that mesh: `Canvas(xx_ps, yy_ps)`
  where `xx_ps`, `yy_ps` are your `(H, W)` coordinate arrays — the world's positions are in
  its own mm, so choose the world's `x_mm`/`y_mm` ranges in the same frame as the mesh.
- Rendering is chunked (`max_elements`); render 4.1 M triples in slices (e.g. 8000 draws
  per call). Measured: see `docs/reference/benchmarks.md` ("World renderer").

## 5. Data sets

```yaml
dataset:
  name: dev_set
  world: world.yml            # relative to this file, or an inline world
  world_id: w-…               # optional pin
  seed: 20261002
  duration_ms: 500
  splits:
    train:      {n: 400, repeats: 3}            # repeats: fresh draws and noise each
    validation: {n: 200, noise_repeats: 2}      # noise_repeats: same draws, new noise
    test:       {stratified: {bins: 5, per_bin: 20}}
    probes:     {per_bin: 20}                    # bins default to test's
    held_out:   {stratified: {bins: 5, per_bin: 20}}
    sessions:   {n: 5, duration_ms: 10000}
    fixed:      {draws: [braille_H]}
```

- **test**, per class: `bins × per_bin` draws; each numeric axis cut into `bins` equal bins
  on its sampling scale, each bin holding exactly `per_bin`; categorical axes (values, `int`,
  `braille_cells`) one bin per value, balanced to ±1; axes shuffled independently.
- **probes**, per class and numeric non-circular axis: `per_bin` draws one bin-width below
  and above the range, inside the field's valid domain; a side with no room is skipped and
  listed in `dataset.json["skipped_probes"]`.
- **Seeds:** split seed `H(seed, split, repeat)`; draw seed per draw; noise seed
  `H(seed, "noise", entry id)`. Building fails if any seed appears twice.
- **Entry ids:** `train/r0/00017`, `validation/r0/00003.n1`, `test/dots/0042`,
  `probes/dots/sigma_mm-below/007`, `held_out/gratings/0003`, `sessions/002`,
  `fixed/braille_H`.
- `sensoryforge dataset build dataset.yml --out DIR` writes `dataset.json` and
  `manifest.jsonl`. A row: `entry, split, repeat, noise_repeat, class, draw (the record),
  bins, probe, seeds {draw, noise}, duration_ms, truncated, world_id, dataset_id`. Bin labels
  are `"[a, b)"` (last bin `"[a, b]"`, 6 significant digits), a categorical value as itself,
  or `"below"`/`"above"`.

## 6. Batch runs

```bash
sensoryforge batch --design DIR --dataset dataset.yml --output OUT
sensoryforge batch --design DIR --dataset dataset.yml --output OUT --tasks 50 --task-index 7
sensoryforge batch --design DIR --dataset dataset.yml --output OUT --tasks 50 --print-tasks
sensoryforge batch sensor.yml --design DIR --dataset dataset.yml --output OUT --splits test,probes --resume
```

- The sensor is `--design`, `--preset` or a config file (a file given with either is merged
  over it). `--entries a:b` runs a slice; `--print-tasks` prints one command per task (rows
  for `experiments/lsf/make_manifest.py`).
- Each entry: rendered with `render_movie(..., dtype=torch.float64)` on
  `Canvas.from_grid_config(grids[0])`, cast to float32, simulated with
  `simulation.receptor_noise_seed` and the run seed set to the entry's noise seed (and each
  population's `noise_seed`, if the design sets one, replaced by `H(noise, "population", i)`).
- Output: `OUT/<entry id>/` (a schema-2.2.0 bundle, written atomically), `OUT/batch.json`,
  `OUT/index/task_<i>.jsonl` (`entry, bundle, status, error, seconds, finished_at,
  design_id, sensoryforge_sha, task`); `read_batch_index(OUT)` merges them. The exit status
  is non-zero if any entry failed.
- Until v1.1.0 is merged into `~/sensoryforge`'s `main`, the `sensoryforge` env's CLI is
  older; run the batch from `bio-encoding` with `python -m sensoryforge.cli batch …`.

## 7. Bundles (schema 2.2.0)

Additive over 2.1.0: `config.json` gains `sensoryforge_sha` and, for an entry,
`world: {world_id, dataset_id, entry}`; `data.h5` attributes gain `sensoryforge_sha`;
`stimuli/stimulus.json` for an entry is `{schema_version, kind: "sensoryforge_world_entry",
entry (the manifest row), layer (the draw as a layered layer; a session: [[start, layer],
...]), dt_ms, total_ms, n_frames, grid, reconstructible_by_pressure_simulation: false}`.
`/stimulus/frames` holds the clean (noise-free) float32 frames.

## 8. Guarantees and the tests that pin them

| # | Guarantee | Test |
|---|---|---|
| 1 | Same world and seed give identical draws and frames in two processes; draw *i* alone equals draw *i* in a batch (bit for bit, same machine) | `tests/contract/test_world_contract.py::test_1_determinism_across_processes_and_batch_sizes`, `tests/unit/test_world_render.py::test_draw_i_alone_equals_draw_i_in_a_batch_bit_for_bit` |
| 2 | A bundle's `/stimulus/frames` equals `render_movie(<the bundle's own record>, Canvas.from_grid_config(grid), dt, duration, dtype=float64).to(float32)` bit for bit | `test_2_the_bundle_records_exactly_the_in_process_render` |
| 3 | Frames rendered at `[t−τ, t, t+τ]` equal those steps of the movie (times `k·dt`) | `test_3_windows_agree_with_movies` |
| 4 | Different noise seeds give different responses; the same seed gives identical spikes | `test_4_noise_seeds` |
| 5 | No draw seed in two splits; every numeric test bin holds `per_bin` per class (categorical ±1); probes labelled and outside the range | `test_5_splits_strata_and_probes` |
| 6 | One draw on 40×40 and 80×80 at 0.15 mm agrees on the shared points to 1e-12 | `test_6_one_draw_on_40x40_and_80x80` |
| 7 | Quiet stretches of a session are exactly 0 | `test_7_session_quiet_stretches_are_exactly_zero` |
| 8 | Every bundle carries the design manifest, the world id, the entry's record and SensoryForge's sha | `test_8_every_bundle_carries_its_provenance` |
| — | A world render equals the `layered` render of `draw.to_layer()` to 1e-5 | `tests/unit/test_world_render.py::test_world_render_equals_layered` |

**Across machines:** uniform, integer and categorical values are bit-identical everywhere;
`log_uniform` values and rendered frames may differ in the last bit between platforms (the
platform maths library), so a manifest's stored record is the canonical draw (as for
SensoryForge's golden fixtures, F-071).

## 9. Conventions pressure-simulation must map

- `bar` (an edge) uses `p = x·sinθ + y·cosθ`; `grating`/`gabor` stripes vary along
  `x·cosθ + y·sinθ`; both take degrees.
- Shapes peak at `amplitude` and are non-negative unless `signed: true` (`grating`, `gabor`:
  `cos` instead of `(1 + cos)/2`).
- Braille cells are dot numbers 1–6 (1–3 down the left column, 4–6 down the right), dot
  pitch `dot_spacing_mm`, centred on the cell.
- Batch frames are float32 (the engine's dtype); render in float64 and cast to compare.
```

- [ ] **Step 2: Write `docs/user_guide/worlds.md`**

A user guide, in this order, each section with a runnable example taken from the tests:

1. *What a world is* — one paragraph: classes of layered stimuli whose fields are drawn from
   axes; worlds are for data sets, hypothesis tests and simulation-based inference.
2. *A first world* — a three-class YAML (`dots`, `edges`, `quiet`) and
   `sensoryforge world validate world.yml` output.
3. *Axes* — the five forms and the binding order (copy §1 of the contract, then the
   ambiguity example from `test_ambiguous_names_need_a_dotted_path`).
4. *Time* — the episode (lead-in, touch, hold, slide, release, contacts, pauses), with the
   `sliders` and `twice` classes from `tests/fixtures/worlds/tactile_small.yml`.
5. *Vibration and taps* — `modulation: sine | pulses`, the `taps` and `vibes` classes.
6. *Sampling and rendering in Python* — `sample`, `render`, `render_movie`, `Canvas`
   variants; "draw *i* never depends on n".
7. *Sessions* — draws end to end; the quiet share.
8. *Data sets* — the spec, the splits, `sensoryforge dataset build`, reading the manifest.
9. *Running a data set* — `sensoryforge batch --dataset`, tasks, `--print-tasks`, resume,
   reading `index/`.
10. *Extending* — `register_shape` (the ring example from `tests/unit/test_world_kernel.py`),
    `register_distribution`, `register_class_kind`; a channel per class.

Link the contract (`../reference/world_contract.md`) for the exact guarantees.

- [ ] **Step 3: Document the new layered fields** in `docs/user_guide/designing_stimuli.md`

Add a section "Slides, repeated contacts and vibration" after the stacking example:
`timing.slide_ms` with `motion.span: slide`; `timing.contacts` / `pause_ms` (motion spread
over all contacts); `modulation: {kind: sine, frequency_hz, depth, phase_deg}` and
`{kind: pulses, rate_hz, duty, edge_ms, depth}`; braille `dots: "125 14"`; `signed: true` on
`grating`/`gabor`. Use the YAML from `tests/unit/test_layered_episode.py`. State that every
new field defaults off and old stimuli render unchanged (`tests/unit/test_layered_golden.py`).

- [ ] **Step 4: Navigation** — in `mkdocs.yml`, under `User Guide`, add
`- Worlds and Data Sets: user_guide/worlds.md` after `Designing Stimuli`; under
`API Reference`, add `- World Engine Contract: reference/world_contract.md` after
`Benchmarks`.

- [ ] **Step 5: A path rule for future sessions** — create `.claude/rules/world-engine.md`:

```markdown
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
```

- [ ] **Step 6: CLAUDE.md** — under "Architecture", after "Receptive fields (Phase 2, Wave I)", add:

```markdown
### Worlds and data sets (v1.1.0)

`sensoryforge/world/` declares stimulus **worlds** (`world:` YAML: classes of layered
stimuli whose fields are drawn from axes), samples them deterministically
(`sample(world, n, seed)`: counter-based splitmix64 hashing, so draw *i* never depends on
*n*), renders any draws at any times on any coordinates (`render`, vectorised, float32/64,
CPU/CUDA, kept equal to `layered` by tests), builds data sets (`dataset:` YAML: seeded
splits, a Latin-hypercube test split, probes, held-out classes, sessions, fixed draws;
`sensoryforge dataset build`), and runs them (`sensoryforge batch --dataset`, one bundle per
entry, `--tasks/--task-index`, `--print-tasks`, `--resume`). Layered gained default-off
fields (`slide_ms`, `contacts`, `pause_ms`, `modulation`, braille `dots`, `signed`); their
timing math is `stimuli/episode.py`. Bundles are schema 2.2.0 (SF's sha in every bundle;
world entries carry their record). pressure-simulation's contract:
`docs/reference/world_contract.md`, pinned by `tests/contract/test_world_contract.py`; see
`.claude/rules/world-engine.md` before changing any of it.
```

- [ ] **Step 7: DECISIONS.md** — read the top of `docs_root/DECISIONS.md` for its section format, then append (newest-first if the file is newest-first) one dated section, "World engine (2026-10-02)", recording for D-94c08c7, D-99d39b6, D-37c5247, D-cccbd6a, D-d3605dd and D-46a52e6 the reasoning from the spec's §2 (why a class is a layered layer and not a new vocabulary; why touch/hold/slide/release with contacts; why sessions are draws end to end; why a separate renderer pinned by tests rather than rewriting layered or extending `BatchExecutor`; why both repeat semantics; why a branch tag, not a merge), plus the tolerance correction (layered equality is 1e-5, not the spec's first 1e-6, because layered computes time in float32).

- [ ] **Step 8: Version and changelog**

`pyproject.toml`: `version = "1.1.0"`. `sensoryforge/__init__.py`: `__version__ = "1.1.0"`.
`CHANGELOG.md`: a `## 1.1.0 — 2026-10-02` entry listing: the world engine (worlds, sampling,
rendering, data sets, `batch --dataset`, `dataset build`, `world validate|sample`); layered's
new fields; bundle schema 2.2.0 with SF's sha; the contract page.

- [ ] **Step 9: Verify everything**

```bash
conda run -n sensoryforge python -m pytest tests -q -m "not gui"
conda run -n sensoryforge python -m pytest tests -q -m gui
conda run -n sensoryforge black --check sensoryforge/world sensoryforge/provenance.py sensoryforge/stimuli/episode.py sensoryforge/stimuli/layered.py sensoryforge/io/bundle.py sensoryforge/cli.py benchmarks/world_render.py tests/unit/test_world_*.py tests/contract/test_world_contract.py tests/integration/test_world_batch.py tests/integration/test_cli_world_dataset.py
conda run -n sensoryforge flake8 sensoryforge/world sensoryforge/provenance.py sensoryforge/stimuli/episode.py
conda run -n sensoryforge mkdocs build --strict
```

Expected: every test passes (report the counts); `black --check` clean (run `black` on the
listed files and re-run the tests if not); `flake8` clean; the strict docs build succeeds.
Report any pre-existing failure separately, with evidence that it fails on `main` too
(`git stash` is shared with other sessions — use a temporary second worktree at `main`
instead to check).

- [ ] **Step 10: Commit the docs and release**

```bash
git add docs/user_guide/worlds.md docs/reference/world_contract.md docs/user_guide/designing_stimuli.md mkdocs.yml .claude/rules/world-engine.md CLAUDE.md docs_root/DECISIONS.md CHANGELOG.md pyproject.toml sensoryforge/__init__.py
git commit -m "docs(world): user guide, contract for pressure-simulation, version 1.1.0

The world engine's user guide, the contract page pressure-simulation's
Phase 2b is written against (schema, sampling and rendering API, data-set
spec, batch command, each guarantee with the test that pins it), a path
rule for future edits, the reasoning in DECISIONS.md, and version 1.1.0.

Refs: D-46a52e6

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 11: Tag**

```bash
git tag -a v1.1.0 -m "SensoryForge 1.1.0: the world engine (declared worlds, sampling, rendering, data sets, batch --dataset)"
git rev-parse v1.1.0^{commit}
```

Report the sha. Do not push and do not merge into `main` (D-46a52e6): Ben merges once
pressure-simulation's Phase 1b no longer simulates from `~/sensoryforge`.
