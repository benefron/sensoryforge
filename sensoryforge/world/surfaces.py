"""World surfaces: shapes that fill the whole patch (spec addendum 2026-10-04).

``self_affine`` is a zero-mean random rough surface with Persson's isotropic
power spectrum, synthesised as a sum of cosines. The surface is a continuous
function of position (it renders the same on any canvas), a pure function of
its parameters and its own ``seed`` (no stream, no global state).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import torch

from sensoryforge.stimuli.base import ParamSpec
from sensoryforge.world import rng

#: Seeds and component counts are exact in float32 up to this bound.
SEED_MAX = 2**24 - 1

_TABLE_CACHE: Dict[Tuple[int, float, float, float, int], np.ndarray] = {}
_CACHE_LIMIT = 10_000


def _spec(name, dtype, default, lo, hi, unit="", help_=""):
    return ParamSpec(
        name,
        label=name.replace("_mm", "").replace("_", " ").capitalize(),
        dtype=dtype,
        default=default,
        min_val=lo,
        max_val=hi,
        unit=unit,
        help=help_,
        tooltip=help_,
    )


SELF_AFFINE_SPECS = [
    _spec(
        "amplitude",
        "float",
        1.0,
        0.0,
        1.0e4,
        help_="RMS of the surface is amplitude / sqrt(2), like a unit-peak sinusoid",
    ),
    _spec("hurst", "float", 0.8, 0.0, 1.0, help_="Hurst exponent H"),
    _spec(
        "rolloff_mm",
        "float",
        2.0,
        0.01,
        1000.0,
        "mm",
        "wavelength of the roll-off wavenumber q0 = 2 pi / rolloff_mm",
    ),
    _spec(
        "cutoff_mm",
        "float",
        0.3,
        0.001,
        100.0,
        "mm",
        "no power above q1 = 2 pi / cutoff_mm",
    ),
    _spec(
        "components",
        "int",
        256,
        16,
        4096,
        help_="number of cosines in the synthesis",
    ),
    _spec(
        "seed",
        "int",
        0,
        0,
        SEED_MAX,
        help_="the surface's own seed (integers stay exact in float32)",
    ),
]


def _radial_wavenumbers(u: np.ndarray, hurst: float, q0: float, q1: float):
    """Inverse CDF of the radial density proportional to ``q C(q)`` on ``[0, q1]``.

    ``C`` is flat below ``q0`` and ``(q / q0) ** (-2 (H + 1))`` from ``q0`` to
    ``q1``. When ``q1 <= q0`` the flat piece alone runs to ``q1``.

    Args:
        u: Stratified uniforms in ``[0, 1)``, shape ``[N]``.
        hurst: Hurst exponent in ``[0, 1]``.
        q0: Roll-off wavenumber, rad/mm.
        q1: Cutoff wavenumber, rad/mm.

    Returns:
        Wavenumbers ``[N]`` in rad/mm, float64.
    """
    if q1 <= q0:
        return q1 * np.sqrt(u)
    a1 = 0.5 * q0**2
    if hurst < 1e-9:
        a2 = q0**2 * math.log(q1 / q0)
    else:
        a2 = q0**2 * (1.0 - (q0 / q1) ** (2.0 * hurst)) / (2.0 * hurst)
    m = u * (a1 + a2)
    low = np.sqrt(2.0 * np.minimum(m, a1))
    m2 = np.maximum(m - a1, 0.0)
    if hurst < 1e-9:
        high = q0 * np.exp(m2 / q0**2)
    else:
        high = q0 * (1.0 - 2.0 * hurst * m2 / q0**2) ** (-1.0 / (2.0 * hurst))
    return np.where(m <= a1, low, np.minimum(high, q1))


def self_affine_table(
    seed: int, hurst: float, rolloff_mm: float, cutoff_mm: float, components: int
) -> np.ndarray:
    """The cosines of one self-affine surface: ``[N, 3]`` of ``qx, qy, phase``.

    ``h(x, y) = N**-0.5 * sum_j cos(qx_j x + qy_j y + phase_j)`` (x, y in mm).
    Radial wavenumbers come from a stratified inverse CDF (stratum ``j`` at
    ``(j + v_j) / N``), directions and phases are uniform; all uniforms are
    ``sensoryforge.world.rng`` values keyed by ``(seed, j, slot)``.

    Args:
        seed: The surface's seed, ``0 <= seed <= 2**24 - 1``.
        hurst: Hurst exponent in ``[0, 1]``.
        rolloff_mm: Roll-off wavelength, mm.
        cutoff_mm: Cutoff wavelength, mm.
        components: Number of cosines ``N``.

    Returns:
        Float64 array ``[N, 3]``; ``qx``, ``qy`` in rad/mm, ``phase`` in rad.
    """
    n = int(components)
    key = (int(seed), float(hurst), float(rolloff_mm), float(cutoff_mm), n)
    if key not in _TABLE_CACHE:
        if len(_TABLE_CACHE) >= _CACHE_LIMIT:
            _TABLE_CACHE.clear()
        seeds = rng.draw_seeds(int(seed), np.arange(n))
        v = rng.uniforms(seeds, "self_affine.stratum")
        psi = 2.0 * math.pi * rng.uniforms(seeds, "self_affine.direction")
        phase = 2.0 * math.pi * rng.uniforms(seeds, "self_affine.phase")
        u = (np.arange(n) + v) / n
        q = _radial_wavenumbers(
            u, float(hurst), 2.0 * math.pi / rolloff_mm, 2.0 * math.pi / cutoff_mm
        )
        _TABLE_CACHE[key] = np.stack([q * np.cos(psi), q * np.sin(psi), phase], axis=1)
    return _TABLE_CACHE[key]


def _per_draw(value: Any) -> list:
    """A parameter (python number, 0-d or ``[g, 1, ...]`` tensor) as a flat list."""
    if isinstance(value, torch.Tensor):
        return value.flatten().tolist()
    return [value]


def self_affine(x: torch.Tensor, y: torch.Tensor, p: Dict[str, Any]) -> torch.Tensor:
    """Self-affine relief, signed, zero mean, RMS ``1 / sqrt(2)`` per unit amplitude.

    The component loop runs to the group's largest ``components``; each draw's
    surplus terms have weight exactly 0, and terms accumulate in a fixed order
    from zero, so a draw renders the same alone or in a batch.

    Args:
        x, y: mm offsets from the element's centre, ``[g, K, *S]`` (world) or
            ``[*S]`` (layered).
        p: ``hurst``, ``rolloff_mm``, ``cutoff_mm``, ``components`` and
            ``seed``, each a tensor of shape ``[g, 1, ...]`` (or 0-d).

    Returns:
        Values, same shape as ``x``; multiply by ``amplitude``.
    """
    cols = [
        _per_draw(p[k])
        for k in ("seed", "hurst", "rolloff_mm", "cutoff_mm", "components")
    ]
    g = len(cols[0])
    tables, counts = [], []
    for seed, hurst, rolloff, cutoff, comps in zip(*cols):
        n = int(round(comps))
        tables.append(self_affine_table(int(round(seed)), hurst, rolloff, cutoff, n))
        counts.append(n)
    size = max(counts)
    table = np.zeros((g, size, 3), dtype=np.float64)
    weight = np.zeros((g, size), dtype=np.float64)
    for i, (t, n) in enumerate(zip(tables, counts)):
        table[i, :n] = t
        weight[i, :n] = n**-0.5
    table = torch.tensor(table, dtype=x.dtype, device=x.device)
    weight = torch.tensor(weight, dtype=x.dtype, device=x.device)
    view = (g,) + (1,) * (x.ndim - 1) if x.ndim > 2 else ()
    total = torch.zeros_like(x)
    for j in range(size):
        col = table[:, j]
        phase = (
            col[:, 0].view(view) * x + col[:, 1].view(view) * y + col[:, 2].view(view)
        )
        total = total + weight[:, j].view(view) * torch.cos(phase)
    return total


# ------------------------------------------------------------------ dot_array

#: Most lattice sites summed each way around the nearest one.
DOT_NEIGHBOURS_MAX = 32

DOT_ARRAY_SPECS = [
    _spec("amplitude", "float", 1.0, 0.0, 1.0e4, help_="peak of one isolated dot"),
    _spec("sigma_mm", "float", 0.2, 0.001, 50.0, "mm", "spread of one dot"),
    _spec("spacing_mm", "float", 2.0, 0.01, 1000.0, "mm", "distance along a row"),
    _spec(
        "row_spacing_mm",
        "float",
        0.0,
        0.0,
        1000.0,
        "mm",
        "distance between rows; 0 = spacing_mm (square) or spacing_mm * sqrt(3) / 2"
        " (hexagonal)",
    ),
    _spec("orientation_deg", "float", 0.0, -3600.0, 3600.0, "deg", "lattice rotation"),
    ParamSpec(
        "arrangement",
        label="Arrangement",
        dtype="str",
        default="square",
        choices=["square", "hexagonal"],
        help="square, or hexagonal (every other row offset by half a spacing)",
        tooltip="square, or hexagonal (every other row offset by half a spacing)",
    ),
]


def dot_array(x: torch.Tensor, y: torch.Tensor, p: Dict[str, Any]) -> torch.Tensor:
    """A lattice of Gaussian bumps, each peaking at 1, overlaps summed.

    In the lattice frame (rotated by ``orientation_deg`` about the element's
    centre) each point is wrapped to its nearest site in every nearby row, and
    the bumps of the ``m`` sites each way are summed, ``m = ceil(7.5 sigma /
    min(a, b))`` (truncation below 1e-12 of the peak). The loop runs to the
    group's largest ``m``; each draw's surplus terms are masked to exact zeros
    and terms accumulate in a fixed order from zero, so a draw renders the
    same alone or in a batch.

    Args:
        x, y: mm offsets from the element's centre, ``[g, K, *S]`` (world) or
            ``[*S]`` (layered).
        p: ``sigma_mm``, ``spacing_mm``, ``row_spacing_mm``, ``orientation_deg``
            (tensors broadcasting against ``x``) and ``arrangement`` (a string).

    Returns:
        Values, same shape as ``x``; multiply by ``amplitude``.

    Raises:
        ValueError: If a dot is so wide for its spacing that more than
            ``DOT_NEIGHBOURS_MAX`` sites each way are needed.
    """
    arrangement = p.get("arrangement", "square")
    if arrangement not in ("square", "hexagonal"):
        raise ValueError(
            "dot_array arrangement must be 'square' or 'hexagonal', "
            f"got {arrangement!r}"
        )
    hexagonal = arrangement == "hexagonal"
    sigma, a, row = p["sigma_mm"], p["spacing_mm"], p["row_spacing_mm"]
    default_b = a * (math.sqrt(3.0) / 2.0) if hexagonal else a
    b = torch.where(row > 0, row, default_b)
    m = torch.ceil(7.5 * sigma / torch.minimum(a, b))
    m_max = int(m.max().item())
    if m_max > DOT_NEIGHBOURS_MAX:
        raise ValueError(
            f"dot_array needs {m_max} lattice sites each way (limit "
            f"{DOT_NEIGHBOURS_MAX}): sigma_mm={float(sigma.max()):g} is too wide "
            f"for spacing_mm={float(a.min()):g}"
        )
    theta = torch.deg2rad(p["orientation_deg"])
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    u = x * cos_t + y * sin_t
    v = y * cos_t - x * sin_t
    j0 = torch.round(v / b)
    inv = 1.0 / (2.0 * sigma**2)
    total = torch.zeros_like(x)
    for dj in range(-m_max, m_max + 1):
        j = j0 + dj
        off = 0.5 * a * torch.remainder(j, 2.0) if hexagonal else 0.0
        i0 = torch.round((u - off) / a)
        dv = v - j * b
        row_on = m >= abs(dj)
        for di in range(-m_max, m_max + 1):
            du = u - off - (i0 + di) * a
            bump = torch.exp(-(du**2 + dv**2) * inv)
            on = row_on & (m >= abs(di))
            total = total + torch.where(on, bump, torch.zeros_like(bump))
    return total


# -------------------------------------------------------------- curved_contact

CURVED_CONTACT_SPECS = [
    _spec(
        "amplitude",
        "float",
        1.0,
        0.0,
        1.0e4,
        help_="peak indentation depth, in units (unit_mm mm each)",
    ),
    _spec("radius_mm", "float", 20.0, 0.1, 1000.0, "mm", "radius of curvature"),
    ParamSpec(
        "form",
        label="Form",
        dtype="str",
        default="sphere",
        choices=["sphere", "cylinder"],
        help="a sphere, or a cylinder whose axis lies along orientation_deg",
        tooltip="a sphere, or a cylinder whose axis lies along orientation_deg",
    ),
    _spec(
        "orientation_deg",
        "float",
        0.0,
        -3600.0,
        3600.0,
        "deg",
        "cylinder axis direction (the bar convention: across = x sin t + y cos t)",
    ),
    _spec(
        "unit_mm",
        "float",
        1.0,
        0.001,
        100.0,
        "mm",
        "millimetres of indentation per unit of amplitude",
    ),
]


def curved_contact(x: torch.Tensor, y: torch.Tensor, p: Dict[str, Any]) -> torch.Tensor:
    """Indentation by a convex rigid sphere or cylinder pressed to depth ``depth``.

    With ``r`` the distance from the centre (sphere) or from the axis
    (cylinder), the sag of the surface is ``R - sqrt(R**2 - r**2)`` (written
    ``r**2 / (R + sqrt(R**2 - r**2))``, free of cancellation for large ``R``)
    and the indentation is ``max(0, depth - sag / unit_mm)`` for ``r <= R``,
    0 beyond. Depth 0 gives exactly 0 everywhere.

    Args:
        x, y: mm offsets from the contact's centre, any shape.
        p: ``depth`` (units, ``amplitude x envelope x modulation``, broadcasting
            against ``x``), ``radius_mm``, ``unit_mm``, ``orientation_deg``
            (tensors) and ``form`` (a string).

    Returns:
        Indentation in units, same shape as ``x``.

    Raises:
        ValueError: If ``form`` is not ``sphere`` or ``cylinder``.
    """
    form = p.get("form", "sphere")
    if form == "sphere":
        r2 = x**2 + y**2
    elif form == "cylinder":
        theta = torch.deg2rad(p["orientation_deg"])
        r2 = (x * torch.sin(theta) + y * torch.cos(theta)) ** 2
    else:
        raise ValueError(
            f"curved_contact form must be sphere or cylinder, got {form!r}"
        )
    radius = p["radius_mm"]
    inside = r2 <= radius**2
    root = torch.sqrt((radius**2 - r2).clamp(min=0.0))
    sag = r2 / (radius + root)
    value = (p["depth"] - sag / p["unit_mm"]).clamp(min=0.0)
    return torch.where(inside, value, torch.zeros_like(value))
