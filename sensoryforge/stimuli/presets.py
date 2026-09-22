"""Named starting points for layered stimuli.

Each preset is a function of the run's duration (some stimuli sweep a
distance that depends on how long the run is) returning a layered stimulus
dict ``{"combine": ..., "layers": [...]}`` -- the same form
``StimulusConfig(type="layered")`` stores, ready to edit.

The first group re-expresses SensoryForge's named stimulus types in the
layered model. The four ported from pressure-simulation (``moving_edge``,
``braille``, ``drifting_grating``, ``ramp_gaussian``) reproduce their named
type to within 2% of its peak; the named types themselves are unchanged and
still exact, so data made with them is not affected. The second group are
new examples of what layering is for.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List

from sensoryforge.stimuli.layered import TIMING_SPECS, defaults

Preset = Dict[str, Any]


def _timing(**values) -> Dict[str, Any]:
    return {**defaults(TIMING_SPECS), **values}


def _one(shape, pattern=None, motion=None, timing=None) -> Preset:
    return {
        "combine": "sum",
        "layers": [
            {
                "shape": shape,
                "pattern": pattern or {"kind": "single", "x_mm": 0.0, "y_mm": 0.0},
                "motion": motion or {"kind": "none"},
                "timing": timing or _timing(),
            }
        ],
    }


def _eighth(duration_ms: float) -> float:
    return duration_ms / 8.0


# ---------------------------------------------------- the named stimulus types


def moving_edge(duration_ms: float) -> Preset:
    """pressure-simulation's moving edge: 50 deg, sweeping x -7.11 -> 7 mm."""
    return _one(
        {
            "kind": "bar",
            "amplitude": 1.0,
            "width_mm": 1.0,
            "length_mm": 0.0,
            "orientation_deg": 50.0,
            "profile": "gaussian",
        },
        motion={
            "kind": "linear",
            "start": [-7.11, 0.0],
            "end": [7.0, 0.0],
            "span": "hold",
        },
        timing=_timing(ramp_up_ms=20.0, hold_ms=None, ramp_down_ms=10.0),
    )


def braille(duration_ms: float) -> Preset:
    """pressure-simulation's braille "H": three Gaussian dots moving up at 20 mm/s."""
    travel = 20.0 * duration_ms / 1000.0
    return _one(
        {"kind": "gaussian", "amplitude": 1.0, "sigma_mm": 0.40},
        pattern={"kind": "list", "positions": [[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5]]},
        motion={
            "kind": "linear",
            "start": [0.0, -7.0],
            "end": [0.0, -7.0 + travel],
            "span": "all",
        },
        timing=_timing(ramp_up_ms=75.0, hold_ms=None, ramp_down_ms=75.0),
    )


def drifting_grating(duration_ms: float) -> Preset:
    """pressure-simulation's drifting grating: 4 mm stripes drifting at 15 mm/s."""
    travel = 15.0 * duration_ms / 1000.0
    return _one(
        {
            "kind": "grating",
            "amplitude": 1.0,
            "wavelength_mm": 4.0,
            "orientation_deg": 0.0,
            "phase_deg": 0.0,
            "profile": "sine",
        },
        motion={
            "kind": "linear",
            "start": [0.0, 0.0],
            "end": [-travel, 0.0],
            "span": "all",
        },
        timing=_timing(ramp_up_ms=100.0, hold_ms=None, ramp_down_ms=100.0),
    )


def ramp_gaussian(duration_ms: float) -> Preset:
    """pressure-simulation's ramp-and-hold probe: ramps up over 50 ms, holds."""
    return _one(
        {"kind": "gaussian", "amplitude": 1.0, "sigma_mm": 1.0},
        timing=_timing(ramp_up_ms=50.0, hold_ms=None, ramp_down_ms=0.0),
    )


def gaussian(duration_ms: float) -> Preset:
    """A still Gaussian probe (the ``gaussian`` type's look: 30, sigma 1 mm)."""
    ramp = _eighth(duration_ms)
    return _one(
        {"kind": "gaussian", "amplitude": 30.0, "sigma_mm": 1.0},
        timing=_timing(ramp_up_ms=ramp, hold_ms=None, ramp_down_ms=ramp),
    )


def moving(duration_ms: float) -> Preset:
    """The ``moving`` type's look: a Gaussian crossing from x = -2 to 2 mm."""
    return _one(
        {"kind": "gaussian", "amplitude": 30.0, "sigma_mm": 1.0},
        motion={
            "kind": "linear",
            "start": [-2.0, 0.0],
            "end": [2.0, 0.0],
            "span": "all",
        },
        timing=_timing(ramp_up_ms=0.0, hold_ms=None, ramp_down_ms=0.0),
    )


def gabor(duration_ms: float) -> Preset:
    """A Gabor patch (raised cosine under a Gaussian window, never negative)."""
    ramp = _eighth(duration_ms)
    return _one(
        {
            "kind": "gabor",
            "amplitude": 1.0,
            "sigma_mm": 1.0,
            "wavelength_mm": 1.0,
            "orientation_deg": 0.0,
            "phase_deg": 0.0,
        },
        timing=_timing(ramp_up_ms=ramp, hold_ms=None, ramp_down_ms=ramp),
    )


def edge_grating(duration_ms: float) -> Preset:
    """Parallel edges: a square grating."""
    ramp = _eighth(duration_ms)
    return _one(
        {
            "kind": "grating",
            "amplitude": 1.0,
            "wavelength_mm": 1.0,
            "orientation_deg": 0.0,
            "phase_deg": 0.0,
            "profile": "square",
            "duty": 0.2,
        },
        timing=_timing(ramp_up_ms=ramp, hold_ms=None, ramp_down_ms=ramp),
    )


def repeated_pattern(duration_ms: float) -> Preset:
    """The ``repeated_pattern`` type's look: 2 x 3 Gaussians 0.5 mm apart."""
    ramp = _eighth(duration_ms)
    return _one(
        {"kind": "gaussian", "amplitude": 30.0, "sigma_mm": 0.5},
        pattern={
            "kind": "grid",
            "rows": 2,
            "cols": 3,
            "spacing_mm": 0.5,
            "row_spacing_mm": 0.5,
            "x_mm": 0.0,
            "y_mm": 0.0,
            "mask": "",
        },
        timing=_timing(ramp_up_ms=ramp, hold_ms=None, ramp_down_ms=ramp),
    )


# ----------------------------------------------------------------- examples


def braille_word(duration_ms: float) -> Preset:
    """The word "hello" in braille dots, swept across the skin."""
    return _one(
        {"kind": "disc", "amplitude": 1.0, "diameter_mm": 0.8, "edge_mm": 0.2},
        pattern={
            "kind": "braille",
            "text": "hello",
            "dot_spacing_mm": 2.5,
            "cell_spacing_mm": 6.0,
            "x_mm": 0.0,
            "y_mm": 0.0,
        },
        motion={
            "kind": "linear",
            "start": [5.0, 0.0],
            "end": [-30.0, 0.0],
            "span": "hold",
        },
        timing=_timing(ramp_up_ms=50.0, hold_ms=None, ramp_down_ms=50.0),
    )


def bumpy_texture(duration_ms: float) -> Preset:
    """A field of random Gaussian bumps sliding across the skin."""
    return _one(
        {"kind": "gaussian", "amplitude": 1.0, "sigma_mm": 0.3},
        pattern={
            "kind": "random",
            "count": 120,
            "width_mm": 30.0,
            "height_mm": 12.0,
            "x_mm": 0.0,
            "y_mm": 0.0,
            "min_distance_mm": 0.8,
            "amplitude_jitter": 0.3,
            "seed": 0,
        },
        motion={
            "kind": "linear",
            "start": [8.0, 0.0],
            "end": [-8.0, 0.0],
            "span": "hold",
        },
        timing=_timing(ramp_up_ms=100.0, hold_ms=None, ramp_down_ms=100.0),
    )


def probe_sequence(duration_ms: float) -> Preset:
    """Three disc probes pressed one after the other at different places."""
    step = duration_ms / 3.0
    layers: List[Dict[str, Any]] = []
    for index, x in enumerate((-2.0, 0.0, 2.0)):
        layers.append(
            {
                "shape": {
                    "kind": "disc",
                    "amplitude": 1.0,
                    "diameter_mm": 1.0,
                    "edge_mm": 0.1,
                },
                "pattern": {"kind": "single", "x_mm": x, "y_mm": 0.0},
                "motion": {"kind": "none"},
                "timing": _timing(
                    onset_ms=index * step,
                    ramp_up_ms=step / 6,
                    hold_ms=step / 2,
                    ramp_down_ms=step / 6,
                ),
            }
        )
    return {"combine": "sum", "layers": layers}


#: Preset name -> builder; the first group mirrors the named stimulus types.
PRESETS: Dict[str, Callable[[float], Preset]] = {
    "moving_edge": moving_edge,
    "braille": braille,
    "drifting_grating": drifting_grating,
    "ramp_gaussian": ramp_gaussian,
    "gaussian": gaussian,
    "moving": moving,
    "gabor": gabor,
    "edge_grating": edge_grating,
    "repeated_pattern": repeated_pattern,
    "braille_word": braille_word,
    "bumpy_texture": bumpy_texture,
    "probe_sequence": probe_sequence,
}


def preset(name: str, duration_ms: float = 1000.0) -> Preset:
    """A fresh copy of preset ``name`` for a run of ``duration_ms``.

    Raises:
        KeyError: If there is no preset of that name.
    """
    if name not in PRESETS:
        raise KeyError(f"no stimulus preset {name!r}; known: {sorted(PRESETS)}")
    return copy.deepcopy(PRESETS[name](float(duration_ms)))
