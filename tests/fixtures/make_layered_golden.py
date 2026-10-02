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
        [
            _layer(
                {"kind": "grating", "wavelength_mm": 0.8, "phase_deg": 40.0}, timing=_T
            )
        ],
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
                pattern={
                    "kind": "grid",
                    "rows": 2,
                    "cols": 3,
                    "spacing_mm": 0.8,
                    "mask": "101 011",
                },
                timing=_T,
            )
        ],
        "sum",
    ),
    "list": (
        [
            _layer(
                {"kind": "gaussian", "sigma_mm": 0.2},
                pattern={
                    "kind": "list",
                    "positions": [[0.5, 0.5], [-0.5, 0.2]],
                    "amplitudes": [1.0, 0.5],
                },
                timing=_T,
            )
        ],
        "sum",
    ),
    "random": (
        [
            _layer(
                {"kind": "gaussian", "sigma_mm": 0.15},
                pattern={
                    "kind": "random",
                    "count": 6,
                    "width_mm": 3.0,
                    "height_mm": 3.0,
                    "seed": 3,
                    "amplitude_jitter": 0.3,
                },
                timing=_T,
            )
        ],
        "sum",
    ),
    "braille_text": (
        [
            _layer(
                {"kind": "gaussian", "sigma_mm": 0.15},
                pattern={
                    "kind": "braille",
                    "text": "hi",
                    "dot_spacing_mm": 0.4,
                    "cell_spacing_mm": 1.2,
                    "x_mm": -0.6,
                },
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
                motion={
                    "kind": "linear",
                    "start": [0.0, -1.0],
                    "end": [0.0, 1.0],
                    "span": "all",
                },
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
            chosen["layers"],
            XX,
            YY,
            dt_ms=1.0,
            total_ms=TOTAL_MS,
            combine=chosen["combine"],
        )
        out[f"preset_{name}"] = frames[::EVERY].clone()
    return out


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "layered_golden.pt"
    torch.save(render_all(), target)
    print(f"wrote {target}")
