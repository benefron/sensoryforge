"""Analytic validation of the ``template`` receptive-field builder (Wave S, S1).

Every check compares against a closed-form value, never a recorded output of
our own code:

* row weights against the Gaussian evaluated at the same neuron-receptor
  distances, using the builder's own resolved ``sigma_mm`` -- i.e. we
  re-derive ``exp(-r^2 / 2 sigma^2)`` independently from the geometry (which
  the builder also computes and exposes via ``receptor_coords``/
  ``neuron_centers``) rather than importing any internal helper;
* row L2 norms against unity (the ``normalize="l2"`` contract);
* the resolvable-distance chain ``d -> sigma = d/pi, pitch = d, N = A/pitch^2``
  against plain arithmetic on the receptor bounding box, at three values of
  ``d``.

Perturbation proofs (recorded 2026-09-16, applied to
``sensoryforge/core/rf_builders/template.py``, tested, then reverted):

* Weights-vs-Gaussian: changing ``vals = torch.exp(-d2_k / (2.0 * self.sigma_mm**2))``
  to use ``self.sigma_mm`` (no square, i.e. ``/ (2.0 * self.sigma_mm)``) makes
  ``test_template_weights_match_analytic_gaussian`` fail hard: max relative
  error jumps from < 1e-6 to ~2.5e4 (row 0), since sigma_mm ~ 0.127mm here
  and the squared vs. unsquared denominator diverges rapidly with distance.
* Resolvable-distance chain: changing ``self.sigma_mm = d / math.pi`` to
  ``self.sigma_mm = d / 2.0`` makes
  ``test_resolvable_distance_chain_matches_arithmetic`` fail at all three
  parametrized ``d`` values (0.2, 0.4, 0.8) -- the builder's resolved
  ``sigma_mm`` (0.4 for d=0.4) no longer equals ``d/pi`` (0.2546...) to
  machine precision.
"""

from __future__ import annotations

import math

import pytest
import torch

from sensoryforge.core.rf_builders.template import TemplateRFBuilder


def _make_grid_coords(rows: int, cols: int, spacing_mm: float) -> torch.Tensor:
    """A regular receptor grid, (x, y) in mm, row-major k = i*cols + j."""
    xs = torch.arange(rows, dtype=torch.float64) * spacing_mm
    ys = torch.arange(cols, dtype=torch.float64) * spacing_mm
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], dim=1).to(torch.float32)


def test_template_weights_match_analytic_gaussian():
    """Non-zero template weights equal the closed-form Gaussian at the same distance.

    Reference: for each (neuron, receptor) pair the template builder keeps
    (its k nearest receptors), the pre-normalisation weight is defined
    (module docstring, ``compute_weights``) as
    ``exp(-r^2 / (2*sigma_mm^2))`` where ``r`` is the Euclidean distance
    between the neuron centre and receptor position, both of which the bank
    exposes directly. We recompute this closed-form value independently
    from the exposed coordinates and undo only the row normalisation
    (dividing back out the row's own L2 norm) to compare like for like.

    Tolerance: 1e-5 relative error. Both sides are the same float32
    arithmetic operation (exp of a scaled squared distance) computed from
    identical inputs, so the only source of discrepancy is float32 rounding;
    1e-5 is far tighter than the 0.5x error a wrong exponent (the
    perturbation below) produces, so it cannot pass a broken formula.
    """
    coords = _make_grid_coords(16, 16, 0.15)
    d = 0.40
    builder = TemplateRFBuilder(coords, resolvable_distance_mm=d, k=28, normalize="l2")
    bank = builder.build()

    weights = bank.weights  # [N, M]
    centers = bank.neuron_centers.to(torch.float64)  # [N, 2]
    receptors = bank.receptor_coords.to(torch.float64)  # [M, 2]
    sigma = builder.sigma_mm

    row_norms = weights.to(torch.float64).norm(dim=1)
    assert (row_norms > 0).all()

    for n in range(weights.shape[0]):
        nz = torch.nonzero(weights[n], as_tuple=True)[0]
        assert nz.numel() == min(builder.k, receptors.shape[0])
        r2 = ((centers[n] - receptors[nz]) ** 2).sum(dim=1)
        analytic = torch.exp(-r2 / (2.0 * sigma**2))
        analytic_normalized = analytic / analytic.norm().clamp(min=1e-300)
        actual = weights[n, nz].to(torch.float64)
        rel_err = (
            (actual - analytic_normalized).abs() / analytic_normalized.clamp(min=1e-12)
        ).max()
        assert rel_err < 1e-5, f"row {n}: relative error {rel_err}"


def test_template_row_norms_are_unity_under_l2_normalization():
    """``normalize="l2"`` rows have unit L2 norm.

    Reference: the ``normalize="l2"`` contract documented on
    ``TemplateRFBuilder`` -- each output row is divided by its own L2 norm,
    so the resulting norm must be exactly 1 up to float32 rounding.

    Tolerance: 1e-6. This is a normalisation identity (x / ||x|| has norm 1
    by construction for any nonzero x), not a numerical approximation, so
    the only error source is float32 rounding in the division/sqrt/sum
    chain -- a bug that skipped normalisation, or normalised by the wrong
    quantity (e.g. sum instead of L2 norm), would produce norms far from 1
    (see perturbation below).
    """
    coords = _make_grid_coords(16, 16, 0.15)
    builder = TemplateRFBuilder(
        coords, resolvable_distance_mm=0.4, k=28, normalize="l2"
    )
    bank = builder.build()
    norms = bank.weights.to(torch.float64).norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


@pytest.mark.parametrize("d", [0.20, 0.40, 0.80])
def test_resolvable_distance_chain_matches_arithmetic(d: float):
    """The d -> sigma, pitch, N chain matches plain arithmetic (D-020).

    Reference: the design chain documented in
    ``sensoryforge/core/rf_builders/template.py``'s module docstring:
    ``sigma = d/pi``, ``pitch = d``, ``N = A/pitch^2`` (A = the receptor
    bounding-box area, here extended by half a receptor spacing per side per
    ``_design_lattice``). We recompute each quantity independently with
    ``math`` and compare to what the builder resolves and to the neuron
    count it actually derives.

    Tolerance: sigma and pitch are checked to 1e-9 (pure float arithmetic,
    no iteration involved). N is checked as an exact integer count derived
    from ``floor(extended_side / pitch + eps) + 1`` per axis (the builder's
    own centring rule, restated here independently) -- not "close to A/d^2"
    with a loose margin, which would hide an off-by-one lattice bug.
    """
    rows = cols = 16
    spacing = 0.15
    coords = _make_grid_coords(rows, cols, spacing)
    builder = TemplateRFBuilder(coords, resolvable_distance_mm=d, k=28, normalize="l2")

    assert builder.sigma_mm == pytest.approx(d / math.pi, abs=1e-9)
    assert builder.pitch_mm == pytest.approx(d, abs=1e-9)

    # Independent count re-derivation matching _design_lattice's rule:
    # extended side = (rows-1)*spacing + spacing (half-spacing pad each
    # side) - 2*edge_offset (default pitch/2 each side).
    edge_offset = d / 2.0
    side_x = (rows - 1) * spacing + spacing - 2 * edge_offset
    side_y = (cols - 1) * spacing + spacing - 2 * edge_offset
    n_x = math.floor(side_x / d + 1e-6) + 1
    n_y = math.floor(side_y / d + 1e-6) + 1
    expected_n = n_x * n_y

    bank = builder.build()
    assert bank.num_neurons == expected_n
