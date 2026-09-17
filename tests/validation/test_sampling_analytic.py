"""Analytic validation of the Wave L ``grid_sample`` receptor-sampling path
(Wave S, S1).

Reference: a 2-D Gaussian stimulus frame has a closed-form value at any
(x, y) position: ``A * exp(-((x-x0)^2/(2 sigmax^2) + (y-y0)^2/(2 sigmay^2)))``.
``SimulationEngine._sample_stimulus_at_receptors`` bilinearly interpolates a
*discretised* frame at each receptor's continuous position, so it cannot
reproduce the closed form exactly -- bilinear interpolation of a smooth,
twice-differentiable function has local error O(h^2) in the grid spacing
``h``, so the discretisation error shrinks as the frame is refined. We use
an asymmetric sigma (different in x and y) precisely because a transposed
x/y mapping is exactly correct on a symmetric Gaussian and only exposed on
an asymmetric one (see the docstring of ``_sample_stimulus_at_receptors``,
which names this same test rationale).

Tolerance: RMS error relative to the Gaussian's peak amplitude. Bilinear
interpolation error near a smooth peak scales like ``(h/sigma)^2`` for grid
spacing ``h``; with a 121x121 frame over a 6mm span (h ~ 0.05mm) and
sigma ~ 0.6mm, ``(h/sigma)^2 ~ 0.007``, so we allow 1% RMS/peak -- generous
enough to absorb the mixed x/y sigma and off-lattice receptor positions, but
far tighter than the ~50% (transposed-axis) or ~100% (zero-order/nearest,
no interpolation) error a broken sampling axis produces (see perturbation
below).

Perturbation proof (recorded 2026-09-16, applied to
``sensoryforge/core/simulation_engine.py``, tested, then reverted):
swapping the sample_grid axis order from
``torch.stack([norm_y, norm_x], dim=-1)`` to
``torch.stack([norm_x, norm_y], dim=-1)`` (the classic x/y transpose bug the
function's own docstring warns about) makes
``test_grid_sample_recovers_analytic_gaussian_at_hex_positions`` fail: RMS
error relative to peak jumps from < 1% to ~11% -- an order of magnitude
over the 1% bound, driven entirely by the asymmetric sigma_x != sigma_y (a
symmetric Gaussian would pass even with the axes swapped, which is exactly
why this test uses different sigmas per axis).
"""

from __future__ import annotations

import math

import torch

from sensoryforge.core.simulation_engine import SimulationEngine


def _hex_receptor_coords(
    n_rows: int, n_cols: int, spacing_mm: float, xlim, ylim
) -> torch.Tensor:
    """Hexagonal (offset-row) lattice of (x, y) positions in mm, centred in the frame."""
    row_pitch = spacing_mm * math.sqrt(3) / 2.0
    pts = []
    for i in range(n_rows):
        x = i * row_pitch
        offset = (spacing_mm / 2.0) if (i % 2 == 1) else 0.0
        for j in range(n_cols):
            y = offset + j * spacing_mm
            pts.append((x, y))
    coords = torch.tensor(pts, dtype=torch.float64)
    # Centre within the frame extent.
    span_x = xlim[1] - xlim[0]
    span_y = ylim[1] - ylim[0]
    coords[:, 0] = coords[:, 0] - coords[:, 0].mean() + (xlim[0] + xlim[1]) / 2.0
    coords[:, 1] = coords[:, 1] - coords[:, 1].mean() + (ylim[0] + ylim[1]) / 2.0
    # Clip a small margin inside the frame so no receptor lands out of bounds.
    margin = 0.05 * min(span_x, span_y)
    coords[:, 0] = coords[:, 0].clamp(xlim[0] + margin, xlim[1] - margin)
    coords[:, 1] = coords[:, 1].clamp(ylim[0] + margin, ylim[1] - margin)
    return coords


def _analytic_gaussian(
    coords: torch.Tensor,
    amp: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
) -> torch.Tensor:
    dx2 = (coords[:, 0] - x0) ** 2 / (2.0 * sigma_x**2)
    dy2 = (coords[:, 1] - y0) ** 2 / (2.0 * sigma_y**2)
    return amp * torch.exp(-(dx2 + dy2))


def test_grid_sample_recovers_analytic_gaussian_at_hex_positions():
    """Bilinear sampling at hex receptor positions recovers the analytic Gaussian.

    See module docstring for the reference, tolerance and its justification.
    """
    amp = 1.0
    x0, y0 = 3.0, 3.1
    sigma_x, sigma_y = 0.6, 0.9  # asymmetric on purpose (catches an x/y swap)
    xlim = (0.0, 6.0)
    ylim = (0.0, 6.0)
    n_pix = 121  # h = 6/120 = 0.05 mm

    xs = torch.linspace(xlim[0], xlim[1], n_pix, dtype=torch.float64)
    ys = torch.linspace(ylim[0], ylim[1], n_pix, dtype=torch.float64)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")  # frame[i, j] at (xs[i], ys[j])
    frame = amp * torch.exp(
        -((xx - x0) ** 2 / (2 * sigma_x**2) + (yy - y0) ** 2 / (2 * sigma_y**2))
    )
    frames = (
        frame.unsqueeze(0).unsqueeze(0).to(torch.float32)
    )  # [batch=1, time=1, H, W]

    coords = _hex_receptor_coords(15, 15, spacing_mm=0.35, xlim=xlim, ylim=ylim).to(
        torch.float32
    )

    sampled = SimulationEngine._sample_stimulus_at_receptors(frames, coords, xlim, ylim)
    sampled = sampled.squeeze(0).squeeze(0).to(torch.float64)  # [M]

    analytic = _analytic_gaussian(
        coords.to(torch.float64), amp, x0, y0, sigma_x, sigma_y
    )

    rms_error = torch.sqrt(torch.mean((sampled - analytic) ** 2))
    rel_rms = float(rms_error / amp)
    assert rel_rms < 0.01, f"RMS error relative to peak = {rel_rms}"
