"""Layered stimuli: a few primitive shapes, placed by patterns, moved and timed.

A layered stimulus is a stack of **layers**, combined by ``sum`` or ``max``.
Each layer is four independent choices:

- **shape** -- what one element looks like: ``gaussian``, ``disc``, ``bar``,
  ``grating`` or ``gabor``, with an ``amplitude``;
- **pattern** -- where copies of it go: ``single``, ``grid`` (with an optional
  on/off ``mask``), ``list`` of positions, ``random`` scatter, or ``braille``
  text;
- **motion** -- how the whole pattern moves: ``none``, ``linear``,
  ``circular`` or along a ``path`` of waypoints;
- **timing** -- ``onset_ms``, ``ramp_up_ms``, ``hold_ms``, ``ramp_down_ms``.

A braille word is a ``braille`` pattern of ``disc`` shapes moving
``linear``; a bumpy texture is a ``random`` pattern of ``gaussian`` shapes; a
sequence of probes is several layers with different onsets.

Every value is non-negative: shapes are pressures in ``[0, amplitude]`` (the
grating and Gabor use a raised cosine, not a signed one). Coordinates are
``(x, y)`` in mm, times in ms, as everywhere in SensoryForge.

Config form (``StimulusConfig`` with ``type: layered``)::

    combine: sum
    layers:
      - shape:   {kind: disc, amplitude: 1.0, diameter_mm: 0.6}
        pattern: {kind: braille, text: "hi", dot_spacing_mm: 2.5}
        motion:  {kind: linear, start: [-10, 0], end: [10, 0]}
        timing:  {onset_ms: 0, ramp_up_ms: 50, hold_ms: 800, ramp_down_ms: 50}

Each part's parameters are described by :data:`SHAPES`, :data:`PATTERNS`,
:data:`MOTIONS` (``{kind: [ParamSpec, ...]}``) and :data:`TIMING_SPECS`, so
a form can be generated for any of them.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sensoryforge.stimuli.base import BaseStimulus, ParamSpec

# --------------------------------------------------------------------- specs


def _label(name: str) -> str:
    """``diameter_mm`` -> ``Diameter``: the unit is shown in the box itself."""
    for suffix in ("_mm", "_ms", "_deg"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", " ").capitalize()


def _f(name, default, lo=None, hi=None, unit="", help_="", advanced=False):
    return ParamSpec(
        name,
        label=_label(name),
        dtype="float",
        default=default,
        min_val=lo,
        max_val=hi,
        unit=unit,
        help=help_,
        tooltip=help_,
        advanced=advanced,
    )


def _i(name, default, lo=None, hi=None, help_=""):
    return ParamSpec(
        name,
        label=_label(name),
        dtype="int",
        default=default,
        min_val=lo,
        max_val=hi,
        help=help_,
        tooltip=help_,
    )


def _c(name, default, choices, help_=""):
    return ParamSpec(
        name,
        label=_label(name),
        dtype="str",
        default=default,
        choices=list(choices),
        help=help_,
        tooltip=help_,
    )


def _s(name, default, help_=""):
    return ParamSpec(
        name,
        label=_label(name),
        dtype="str",
        default=default,
        help=help_,
        tooltip=help_,
    )


def _v(name, default, help_=""):
    """A list-valued parameter (a point ``[x, y]``, a list of points)."""
    return ParamSpec(
        name,
        label=_label(name),
        dtype="float",
        default=default,
        help=help_,
        tooltip=help_,
    )


_AMPLITUDE = _f("amplitude", 1.0, 0.0, 1.0e4, "", "Peak value of one element.")
_ORIENTATION = _f("orientation_deg", 0.0, -360.0, 360.0, "deg", "Rotation.")

#: Shape kind -> its parameters.
SHAPES: Dict[str, List[ParamSpec]] = {
    "gaussian": [
        _AMPLITUDE,
        _f("sigma_mm", 0.5, 0.001, 50.0, "mm", "Standard deviation of the bump."),
    ],
    "disc": [
        _AMPLITUDE,
        _f("diameter_mm", 1.0, 0.001, 100.0, "mm", "Diameter of the flat top."),
        _f("edge_mm", 0.1, 0.0, 10.0, "mm", "Width of the soft edge (0 = hard)."),
    ],
    "bar": [
        _AMPLITUDE,
        _f(
            "width_mm",
            0.3,
            0.001,
            100.0,
            "mm",
            "Width across the bar (sigma for " "a gaussian profile).",
        ),
        _f(
            "length_mm",
            0.0,
            0.0,
            1000.0,
            "mm",
            "Length along the bar; 0 = an " "infinite edge.",
        ),
        _ORIENTATION,
        _c("profile", "gaussian", ["gaussian", "flat"], "Cross-section."),
    ],
    "grating": [
        _AMPLITUDE,
        _f("wavelength_mm", 1.0, 0.001, 100.0, "mm", "Period of the stripes."),
        _ORIENTATION,
        _f("phase_deg", 0.0, -360.0, 360.0, "deg", "Phase of the stripes."),
        _c("profile", "sine", ["sine", "square"], "Raised cosine or on/off."),
        _f(
            "duty",
            0.5,
            0.01,
            0.99,
            "",
            "Fraction of a period that is on " "(square profile).",
            advanced=True,
        ),
    ],
    "gabor": [
        _AMPLITUDE,
        _f("sigma_mm", 1.0, 0.001, 50.0, "mm", "Width of the Gaussian window."),
        _f("wavelength_mm", 1.0, 0.001, 100.0, "mm", "Period of the stripes."),
        _ORIENTATION,
        _f("phase_deg", 0.0, -360.0, 360.0, "deg", "Phase of the stripes."),
    ],
}

#: Pattern kind -> its parameters.
PATTERNS: Dict[str, List[ParamSpec]] = {
    "single": [
        _f("x_mm", 0.0, -1000.0, 1000.0, "mm", "Position."),
        _f("y_mm", 0.0, -1000.0, 1000.0, "mm", "Position."),
    ],
    "grid": [
        _i("rows", 3, 1, 1000, "Rows (along y, first row on top)."),
        _i("cols", 2, 1, 1000, "Columns (along x, first column on the left)."),
        _f("spacing_mm", 2.5, 0.0, 1000.0, "mm", "Distance between columns."),
        _f(
            "row_spacing_mm",
            0.0,
            0.0,
            1000.0,
            "mm",
            "Distance between rows; 0 = " "same as spacing_mm.",
        ),
        _f("x_mm", 0.0, -1000.0, 1000.0, "mm", "Centre of the grid."),
        _f("y_mm", 0.0, -1000.0, 1000.0, "mm", "Centre of the grid."),
        _s(
            "mask",
            "",
            "Which cells are on, row by row: '1'/'0' (spaces "
            "ignored); empty = all on.",
        ),
    ],
    "list": [
        _v("positions", [[0.0, 0.0]], "[[x, y], ...] in mm."),
        _v("amplitudes", [], "Optional scale per position (default 1)."),
    ],
    "random": [
        _i("count", 30, 1, 100000, "How many elements."),
        _f("width_mm", 10.0, 0.0, 10000.0, "mm", "Width of the region."),
        _f("height_mm", 10.0, 0.0, 10000.0, "mm", "Height of the region."),
        _f("x_mm", 0.0, -1000.0, 1000.0, "mm", "Centre of the region."),
        _f("y_mm", 0.0, -1000.0, 1000.0, "mm", "Centre of the region."),
        _f(
            "min_distance_mm",
            0.0,
            0.0,
            1000.0,
            "mm",
            "No two elements closer " "than this (as far as the region allows).",
        ),
        _f(
            "amplitude_jitter",
            0.0,
            0.0,
            1.0,
            "",
            "Each element's amplitude is " "scaled by 1 - jitter * U(0, 1).",
        ),
        _i("seed", 0, 0, 2**31 - 1, "Random seed (same seed, same pattern)."),
    ],
    "braille": [
        _s("text", "h", "Letters a-z and spaces."),
        _f("dot_spacing_mm", 2.5, 0.1, 100.0, "mm", "Dot pitch within a cell."),
        _f(
            "cell_spacing_mm",
            6.0,
            0.1,
            100.0,
            "mm",
            "Distance between cell " "centres.",
        ),
        _f("x_mm", 0.0, -1000.0, 1000.0, "mm", "Centre of the first cell."),
        _f("y_mm", 0.0, -1000.0, 1000.0, "mm", "Centre of the first cell."),
    ],
}

_SPAN = _c(
    "span",
    "hold",
    ["hold", "all"],
    "Move during the hold only, or " "from onset to the end of the ramp down.",
)

#: Motion kind -> its parameters.
MOTIONS: Dict[str, List[ParamSpec]] = {
    "none": [],
    "linear": [
        _v("start", [-5.0, 0.0], "[x, y] offset at the start, in mm."),
        _v("end", [5.0, 0.0], "[x, y] offset at the end, in mm."),
        _SPAN,
    ],
    "circular": [
        _f("radius_mm", 2.0, 0.0, 1000.0, "mm", "Radius of the circle."),
        _f("revolutions", 1.0, -1000.0, 1000.0, "", "Turns (negative = clockwise)."),
        _f("start_deg", 0.0, -360.0, 360.0, "deg", "Starting angle."),
        _SPAN,
    ],
    "path": [
        _v(
            "waypoints",
            [[-5.0, 0.0], [5.0, 0.0]],
            "[[x, y], ...] offsets, " "visited at constant speed.",
        ),
        _SPAN,
    ],
}

#: The timing of one layer, in ms.
TIMING_SPECS: List[ParamSpec] = [
    _f("onset_ms", 0.0, 0.0, 1.0e7, "ms", "When the layer starts."),
    _f("ramp_up_ms", 50.0, 0.0, 1.0e7, "ms", "Rise time (0 = a step)."),
    _f(
        "hold_ms",
        None,
        0.0,
        1.0e7,
        "ms",
        "Time at full amplitude; unset = " "until the ramp down ends the run.",
    ),
    _f("ramp_down_ms", 50.0, 0.0, 1.0e7, "ms", "Fall time (0 = a step)."),
]

COMBINE_MODES = ("sum", "max")


def defaults(specs: Sequence[ParamSpec]) -> Dict[str, Any]:
    """``{name: default}`` for a list of specs."""
    return {spec.name: spec.default for spec in specs}


def default_layer(shape: str = "gaussian", pattern: str = "single") -> Dict[str, Any]:
    """A complete layer dict with every default filled in."""
    return {
        "shape": {"kind": shape, **defaults(SHAPES[shape])},
        "pattern": {"kind": pattern, **defaults(PATTERNS[pattern])},
        "motion": {"kind": "none"},
        "timing": defaults(TIMING_SPECS),
    }


# -------------------------------------------------------------------- shapes


def _gaussian(x, y, p):
    s = float(p["sigma_mm"])
    return torch.exp(-(x**2 + y**2) / (2.0 * s**2))


def _disc(x, y, p):
    radius = float(p["diameter_mm"]) / 2.0
    edge = float(p.get("edge_mm", 0.0))
    r = torch.sqrt(x**2 + y**2)
    if edge <= 0:
        return (r <= radius).to(x.dtype)
    return ((radius - r) / edge + 0.5).clamp(0.0, 1.0)


def _rotated(x, y, orientation_deg):
    """``(across, along)``: across the stripes / bar, and along them."""
    theta = torch.tensor(math.radians(orientation_deg), dtype=x.dtype, device=x.device)
    sin_t, cos_t = torch.sin(theta), torch.cos(theta)
    # The moving-edge convention of pressure-simulation: p = x sin + y cos.
    across = x * sin_t + y * cos_t
    along = x * cos_t - y * sin_t
    return across, along


def _bar(x, y, p):
    across, along = _rotated(x, y, float(p.get("orientation_deg", 0.0)))
    width = float(p["width_mm"])
    if p.get("profile", "gaussian") == "flat":
        value = (across.abs() <= width / 2.0).to(x.dtype)
    else:
        value = torch.exp(-(across**2) / (2.0 * width**2))
    length = float(p.get("length_mm", 0.0) or 0.0)
    if length > 0:
        value = value * (along.abs() <= length / 2.0).to(x.dtype)
    return value


def _stripes(across, p):
    wavelength = float(p["wavelength_mm"])
    phase = torch.remainder(
        2.0 * math.pi * across / wavelength
        + math.radians(float(p.get("phase_deg", 0.0))),
        2.0 * math.pi,
    )
    if p.get("profile", "sine") == "square":
        duty = float(p.get("duty", 0.5))
        # On for the part of each period centred on phase 0.
        centred = torch.minimum(phase, 2.0 * math.pi - phase)
        return (centred <= math.pi * duty).to(across.dtype)
    return 0.5 * (1.0 + torch.cos(phase))


def _grating(x, y, p):
    # At 0 deg the stripes are vertical: the value varies along x.
    theta = math.radians(float(p.get("orientation_deg", 0.0)))
    across = x * math.cos(theta) + y * math.sin(theta)
    return _stripes(across, p)


def _gabor(x, y, p):
    return _gaussian(x, y, p) * _grating(x, y, p)


_SHAPE_FUNCTIONS: Dict[str, Callable] = {
    "gaussian": _gaussian,
    "disc": _disc,
    "bar": _bar,
    "grating": _grating,
    "gabor": _gabor,
}

#: Shapes that fill the whole plane: drawn once, not at every position.
_UNBOUNDED = {"grating"}

# ------------------------------------------------------------------ patterns

#: Dots of letters a-z, numbered 1-6 (1-3 down the left column, 4-6 right).
_BRAILLE = {
    "a": "1",
    "b": "12",
    "c": "14",
    "d": "145",
    "e": "15",
    "f": "124",
    "g": "1245",
    "h": "125",
    "i": "24",
    "j": "245",
    "k": "13",
    "l": "123",
    "m": "134",
    "n": "1345",
    "o": "135",
    "p": "1234",
    "q": "12345",
    "r": "1235",
    "s": "234",
    "t": "2345",
    "u": "136",
    "v": "1236",
    "w": "2456",
    "x": "1346",
    "y": "13456",
    "z": "1356",
}


def _grid_positions(p) -> Tuple[List[Tuple[float, float]], List[float]]:
    rows, cols = int(p["rows"]), int(p["cols"])
    dx = float(p["spacing_mm"])
    dy = float(p.get("row_spacing_mm") or 0.0) or dx
    cx, cy = float(p.get("x_mm", 0.0)), float(p.get("y_mm", 0.0))
    mask = "".join(ch for ch in str(p.get("mask") or "") if ch in "01")
    if mask and len(mask) != rows * cols:
        raise ValueError(
            f"grid mask has {len(mask)} cells but the grid has rows x cols = "
            f"{rows * cols}"
        )
    positions = []
    for r in range(rows):
        for c in range(cols):
            if mask and mask[r * cols + c] != "1":
                continue
            x = cx + (c - (cols - 1) / 2.0) * dx
            y = cy + ((rows - 1) / 2.0 - r) * dy
            positions.append((x, y))
    return positions, [1.0] * len(positions)


def _random_positions(p) -> Tuple[List[Tuple[float, float]], List[float]]:
    generator = torch.Generator().manual_seed(int(p.get("seed", 0)))
    count = int(p["count"])
    w, h = float(p["width_mm"]), float(p["height_mm"])
    cx, cy = float(p.get("x_mm", 0.0)), float(p.get("y_mm", 0.0))
    min_d = float(p.get("min_distance_mm", 0.0))
    positions: List[Tuple[float, float]] = []
    attempts = 0
    while len(positions) < count and attempts < 100 * count:
        attempts += 1
        u = torch.rand(2, generator=generator)
        x = cx + (float(u[0]) - 0.5) * w
        y = cy + (float(u[1]) - 0.5) * h
        if min_d > 0 and any(
            (x - px) ** 2 + (y - py) ** 2 < min_d**2 for px, py in positions
        ):
            continue
        positions.append((x, y))
    jitter = float(p.get("amplitude_jitter", 0.0))
    scales = (1.0 - jitter * torch.rand(len(positions), generator=generator)).tolist()
    return positions, scales


def _braille_positions(p) -> Tuple[List[Tuple[float, float]], List[float]]:
    text = str(p.get("text", "")).lower()
    pitch = float(p["dot_spacing_mm"])
    step = float(p["cell_spacing_mm"])
    x0, y0 = float(p.get("x_mm", 0.0)), float(p.get("y_mm", 0.0))
    positions = []
    for index, letter in enumerate(text):
        if letter == " ":
            continue
        if letter not in _BRAILLE:
            raise ValueError(f"braille pattern: no cell for {letter!r} (a-z only)")
        cell_x = x0 + index * step
        for dot in _BRAILLE[letter]:
            n = int(dot) - 1
            col, row = divmod(n, 3)  # 1-3 left column, 4-6 right; top to bottom
            positions.append((cell_x + (col - 0.5) * pitch, y0 + (1 - row) * pitch))
    return positions, [1.0] * len(positions)


def pattern_positions(
    pattern: Dict[str, Any],
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """``(positions, amplitude scales)`` a pattern places elements at, in mm.

    Args:
        pattern: ``{"kind": ..., <params>}``; missing params take defaults.

    Raises:
        ValueError: For an unknown kind or an inconsistent pattern.
    """
    kind = pattern.get("kind", "single")
    if kind not in PATTERNS:
        raise ValueError(f"unknown pattern kind {kind!r}; known: {sorted(PATTERNS)}")
    p = {**defaults(PATTERNS[kind]), **pattern}
    if kind == "single":
        return [(float(p["x_mm"]), float(p["y_mm"]))], [1.0]
    if kind == "grid":
        return _grid_positions(p)
    if kind == "list":
        positions = [(float(x), float(y)) for x, y in p["positions"]]
        scales = [float(a) for a in (p.get("amplitudes") or [])] or [1.0] * len(
            positions
        )
        if len(scales) != len(positions):
            raise ValueError(
                f"list pattern: {len(scales)} amplitudes for {len(positions)} positions"
            )
        return positions, scales
    if kind == "random":
        return _random_positions(p)
    return _braille_positions(p)


# -------------------------------------------------------------------- timing


def layer_envelope(
    timing: Dict[str, Any], time_ms: torch.Tensor, run_ms: float
) -> torch.Tensor:
    """The layer's amplitude over time, ``[T]`` in ``[0, 1]``.

    Zero before ``onset_ms``; a linear ramp up over ``ramp_up_ms``; 1 for
    ``hold_ms`` (unset: until the ramp down ends at ``run_ms``); a linear ramp
    down over ``ramp_down_ms``; zero after.
    """
    t = {**defaults(TIMING_SPECS), **(timing or {})}
    onset = float(t["onset_ms"] or 0.0)
    up = float(t["ramp_up_ms"] or 0.0)
    down = float(t["ramp_down_ms"] or 0.0)
    hold = t["hold_ms"]
    hold = max(run_ms - onset - up - down, 0.0) if hold is None else float(hold)
    local = time_ms - onset
    amp = torch.zeros_like(time_ms)
    rising = (local >= 0) & (local < up)
    if up > 0:
        amp[rising] = local[rising] / up
    held = (local >= up) & (local < up + hold)
    amp[held] = 1.0
    if down > 0:
        falling = (local >= up + hold) & (local < up + hold + down)
        amp[falling] = 1.0 - (local[falling] - up - hold) / down
    return amp.clamp(0.0, 1.0)


def _motion_fraction(
    timing, span: str, time_ms: torch.Tensor, run_ms: float
) -> torch.Tensor:
    """0 -> 1 progress of the motion over its span, ``[T]``."""
    t = {**defaults(TIMING_SPECS), **(timing or {})}
    onset = float(t["onset_ms"] or 0.0)
    up = float(t["ramp_up_ms"] or 0.0)
    down = float(t["ramp_down_ms"] or 0.0)
    hold = t["hold_ms"]
    hold = max(run_ms - onset - up - down, 0.0) if hold is None else float(hold)
    if span == "all":
        start, length = onset, up + hold + down
    else:
        start, length = onset + up, hold
    if length <= 0:
        return torch.zeros_like(time_ms)
    return ((time_ms - start) / length).clamp(0.0, 1.0)


def motion_offsets(
    motion, timing, time_ms: torch.Tensor, run_ms: float
) -> Optional[torch.Tensor]:
    """``[T, 2]`` translation of the pattern over time in mm, or ``None``."""
    motion = motion or {"kind": "none"}
    kind = motion.get("kind", "none")
    if kind not in MOTIONS:
        raise ValueError(f"unknown motion kind {kind!r}; known: {sorted(MOTIONS)}")
    if kind == "none":
        return None
    m = {**defaults(MOTIONS[kind]), **motion}
    s = _motion_fraction(timing, m.get("span", "hold"), time_ms, run_ms)
    if kind == "linear":
        (x0, y0), (x1, y1) = m["start"], m["end"]
        return torch.stack([x0 + s * (x1 - x0), y0 + s * (y1 - y0)], dim=1)
    if kind == "circular":
        angle = (
            math.radians(float(m["start_deg"]))
            + 2 * math.pi * float(m["revolutions"]) * s
        )
        r = float(m["radius_mm"])
        return torch.stack([r * torch.cos(angle), r * torch.sin(angle)], dim=1)
    points = torch.tensor(m["waypoints"], dtype=time_ms.dtype)
    if points.shape[0] < 2:
        return points[:1].expand(time_ms.numel(), 2).clone()
    seg = (points[1:] - points[:-1]).norm(dim=1)
    cum = torch.cat([torch.zeros(1), seg.cumsum(0)])
    total = float(cum[-1]) or 1.0
    d = s * total
    idx = torch.searchsorted(cum, d.clamp(max=total - 1e-9), right=True) - 1
    idx = idx.clamp(0, len(seg) - 1)
    frac = ((d - cum[idx]) / seg[idx].clamp(min=1e-12)).unsqueeze(1)
    return points[idx] + frac * (points[idx + 1] - points[idx])


# ------------------------------------------------------------------- render


def render_layer(layer: Dict[str, Any], xx, yy, time_ms, run_ms: float) -> torch.Tensor:
    """One layer's frames ``[T, H, W]``."""
    shape = layer.get("shape") or {"kind": "gaussian"}
    kind = shape.get("kind", "gaussian")
    if kind not in SHAPES:
        raise ValueError(f"unknown shape kind {kind!r}; known: {sorted(SHAPES)}")
    params = {**defaults(SHAPES[kind]), **shape}
    fn = _SHAPE_FUNCTIONS[kind]
    amplitude = float(params["amplitude"])
    positions, scales = pattern_positions(layer.get("pattern") or {"kind": "single"})
    envelope = layer_envelope(layer.get("timing"), time_ms, run_ms)
    offsets = motion_offsets(layer.get("motion"), layer.get("timing"), time_ms, run_ms)

    def draw(dx: float, dy: float) -> torch.Tensor:
        if kind in _UNBOUNDED:
            return fn(xx - dx, yy - dy, params)
        frame = torch.zeros_like(xx)
        for (px, py), scale in zip(positions, scales):
            frame = frame + scale * fn(xx - px - dx, yy - py - dy, params)
        return frame

    frames = torch.zeros(time_ms.numel(), *xx.shape, dtype=xx.dtype, device=xx.device)
    if offsets is None:
        frames = draw(0.0, 0.0).unsqueeze(0) * envelope.view(-1, 1, 1)
    else:
        for k in torch.nonzero(envelope > 0).flatten().tolist():
            frames[k] = draw(float(offsets[k, 0]), float(offsets[k, 1])) * envelope[k]
    return frames * amplitude


def render_layers(
    layers: Sequence[Dict[str, Any]],
    xx: torch.Tensor,
    yy: torch.Tensor,
    *,
    dt_ms: float,
    total_ms: float,
    combine: str = "sum",
) -> torch.Tensor:
    """Frames ``[T, H, W]`` of a stack of layers, ``T = round(total_ms / dt_ms)``."""
    if combine not in COMBINE_MODES:
        raise ValueError(f"combine must be one of {COMBINE_MODES}, got {combine!r}")
    n = max(int(round(float(total_ms) / float(dt_ms))), 1)
    xx = xx.to(torch.float32)
    yy = yy.to(torch.float32)
    time_ms = torch.arange(n, dtype=torch.float32) * float(dt_ms)
    out = torch.zeros(n, *xx.shape, dtype=torch.float32, device=xx.device)
    for layer in layers:
        if not layer.get("enabled", True):
            continue
        frames = render_layer(layer, xx, yy, time_ms.to(xx.device), float(total_ms))
        out = out + frames if combine == "sum" else torch.maximum(out, frames)
    return out


class LayeredStimulus(BaseStimulus):
    """A stack of layers (see the module docstring); registered as ``layered``.

    Args:
        layers: Layer dicts (``shape``, ``pattern``, ``motion``, ``timing``).
        combine: ``"sum"`` or ``"max"``.
        total_ms: Duration in ms (the renderer passes the run's).
        dt_ms: Time step in ms (the renderer passes the run's).
    """

    def __init__(
        self,
        layers: Optional[List[Dict[str, Any]]] = None,
        combine: str = "sum",
        total_ms: float = 1000.0,
        dt_ms: float = 1.0,
    ) -> None:
        super().__init__()
        self.layers = list(layers) if layers else [default_layer()]
        self.combine = combine
        self.total_ms = float(total_ms)
        self.dt_ms = float(dt_ms)

    def forward(self, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """Frames ``[T, H, W]`` on the ``(xx, yy)`` canvas (mm)."""
        return render_layers(
            self.layers,
            xx,
            yy,
            dt_ms=self.dt_ms,
            total_ms=self.total_ms,
            combine=self.combine,
        )

    def reset_state(self) -> None:
        """Stateless."""

    def to_dict(self) -> Dict[str, Any]:
        """Every constructor argument."""
        return {
            "layers": self.layers,
            "combine": self.combine,
            "total_ms": self.total_ms,
            "dt_ms": self.dt_ms,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LayeredStimulus":
        """Build from :meth:`to_dict` output (or a config's params)."""
        return cls(**config)

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        """Only ``combine`` is a scalar; layers are edited by a layer editor."""
        return [_c("combine", "sum", COMBINE_MODES, "How overlapping layers add up.")]
