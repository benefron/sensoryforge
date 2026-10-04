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
