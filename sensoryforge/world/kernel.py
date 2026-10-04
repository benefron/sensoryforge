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
    """A registered modulation: ``fn(tc, params) -> [0, 1]``; ``None`` means none."""

    fn: Optional[ModulationFn]
    specs: List[ParamSpec]


SHAPE_KINDS: Dict[str, ShapeKind] = {}
PATTERN_KINDS: Dict[str, PatternKind] = {}
MODULATION_KINDS: Dict[str, ModulationKind] = {}


def _check_free(table: Dict[str, Any], name: str, what: str, replace: bool) -> None:
    if name in table and not replace:
        raise ValueError(
            f"{what} {name!r} is already registered; pass replace=True to replace it"
        )


def register_shape(
    name: str,
    fn: ShapeFn,
    specs: Sequence[ParamSpec],
    unbounded: bool = False,
    *,
    replace: bool = False,
) -> None:
    """Register a shape kind for worlds and layered stimuli.

    Args:
        name: The ``shape.kind`` that selects it.
        fn: ``fn(x, y, params) -> values``. ``x``, ``y`` are mm offsets from the
            element's centre, shape ``[..., *S]``; numeric ``params`` are
            tensors broadcasting against them; strings and bools are plain.
        specs: Its parameters; must include ``amplitude``.
        unbounded: Drawn once over the plane, not at each pattern position.
        replace: Replace a shape already registered under ``name``.

    Raises:
        ValueError: If ``specs`` lacks ``amplitude``, or the name is taken
            and ``replace`` is false.
    """
    if not any(s.name == "amplitude" for s in specs):
        raise ValueError(f"shape {name!r}: its specs must include 'amplitude'")
    _check_free(SHAPE_KINDS, name, "shape", replace)
    SHAPE_KINDS[name] = ShapeKind(fn=fn, specs=list(specs), unbounded=bool(unbounded))


def register_pattern(
    name: str, fn: PatternFn, specs: Sequence[ParamSpec], *, replace: bool = False
) -> None:
    """Register a pattern kind: ``fn(params) -> (positions, scales)``, mm, at 0.

    Raises:
        ValueError: If the name is taken and ``replace`` is false.
    """
    _check_free(PATTERN_KINDS, name, "pattern", replace)
    PATTERN_KINDS[name] = PatternKind(fn=fn, specs=list(specs))


def register_modulation(
    name: str,
    fn: Optional[ModulationFn],
    specs: Sequence[ParamSpec],
    *,
    replace: bool = False,
) -> None:
    """Register a modulation: ``fn(tc, params) -> [0, 1]``; ``tc`` ms since touch.

    Raises:
        ValueError: If the name is taken and ``replace`` is false.
    """
    _check_free(MODULATION_KINDS, name, "modulation", replace)
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
        return layered.pattern_positions(
            {**params, "kind": kind, "x_mm": 0.0, "y_mm": 0.0}
        )

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
    return positions.to(device=device, dtype=dtype), scales.to(
        device=device, dtype=dtype
    )


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
        points = torch.tensor(
            motion["waypoints"], dtype=progress.dtype, device=progress.device
        )
        if points.shape[0] < 2:
            return points[0].expand(*progress.shape, 2).clone()
        seg = (points[1:] - points[:-1]).norm(dim=1)
        cum = torch.cat(
            [torch.zeros(1, dtype=seg.dtype, device=seg.device), seg.cumsum(0)]
        )
        total = float(cum[-1]) or 1.0
        d = progress * total
        idx = (
            torch.searchsorted(cum, d.clamp(max=total - 1e-9).contiguous(), right=True)
            - 1
        )
        idx = idx.clamp(0, len(seg) - 1)
        frac = ((d - cum[idx]) / seg[idx].clamp(min=1e-12)).unsqueeze(-1)
        return points[idx] + frac * (points[idx + 1] - points[idx])
    raise ValueError(
        f"unknown motion kind {kind!r}; known: none, linear, circular, path"
    )


# ------------------------------------------------------ patch-filling surfaces

from sensoryforge.world import surfaces  # noqa: E402

register_shape("self_affine", surfaces.self_affine, surfaces.SELF_AFFINE_SPECS)
register_shape("dot_array", surfaces.dot_array, surfaces.DOT_ARRAY_SPECS)
