"""Tests for the ``template`` receptive-field builder (Phase 2, I4).

Designed receptive fields: one resolvable distance ``d`` gives
``sigma = d / pi`` and ``pitch = d``; one Gaussian template is translated
over a square neuron lattice, truncated to the ``k`` nearest receptors, with
analytic weights and unit-L2 rows by default (pressure-simulation's
``ConstructedRF``).
"""

import math

import pytest
import torch

from sensoryforge.core.grid import ReceptorGrid
from sensoryforge.core.rf_bank import ReceptiveFieldBank
from sensoryforge.core.rf_builders.template import TemplateRFBuilder
from sensoryforge.register_components import register_all
from sensoryforge.registry import INNERVATION_REGISTRY
from sensoryforge.testing.contracts import check_component

register_all()


@pytest.fixture(scope="module")
def coords16():
    """16x16 receptor grid at 0.15 mm: extreme-to-extreme side 2.25 mm."""
    return ReceptorGrid(grid_size=(16, 16), spacing=0.15).get_receptor_coordinates()


def test_registered_under_template():
    assert INNERVATION_REGISTRY.get_class("template") is TemplateRFBuilder


def test_d_gives_sigma_and_pitch(coords16):
    b = TemplateRFBuilder(coords16, resolvable_distance_mm=0.40)
    assert b.sigma_mm == pytest.approx(0.12732, abs=1e-5)
    assert b.sigma_mm == pytest.approx(0.40 / math.pi)
    assert b.pitch_mm == 0.40
    assert b.edge_offset_mm == pytest.approx(0.20)


def test_16x16_grid_at_d_040_gives_36_neurons(coords16):
    b = TemplateRFBuilder(coords16, resolvable_distance_mm=0.40)
    assert b.num_neurons == 36
    assert b.lattice_shape == (6, 6)
    bank = b.build()
    assert isinstance(bank, ReceptiveFieldBank)
    assert tuple(bank.weights.shape) == (36, 256)
    assert tuple(bank.neuron_centers.shape) == (36, 2)


def test_explicit_sigma_and_pitch_form(coords16):
    b = TemplateRFBuilder(coords16, sigma_mm=0.1, pitch_mm=0.5)
    assert b.sigma_mm == 0.1 and b.pitch_mm == 0.5
    assert b.resolvable_distance_mm is None


def test_both_parameter_forms_raise(coords16):
    with pytest.raises(ValueError, match="exactly one"):
        TemplateRFBuilder(
            coords16, resolvable_distance_mm=0.4, sigma_mm=0.1, pitch_mm=0.4
        )
    with pytest.raises(ValueError, match="exactly one"):
        TemplateRFBuilder(coords16, resolvable_distance_mm=0.4, sigma_mm=0.1)


def test_neither_parameter_form_raises(coords16):
    with pytest.raises(ValueError, match="exactly one"):
        TemplateRFBuilder(coords16)
    with pytest.raises(ValueError, match="exactly one"):
        TemplateRFBuilder(coords16, sigma_mm=0.1)


def test_bad_normalize_raises(coords16):
    with pytest.raises(ValueError, match="normalize"):
        TemplateRFBuilder(coords16, resolvable_distance_mm=0.4, normalize="max")


def test_every_row_has_exactly_k_nonzeros(coords16):
    for k in (1, 7, 28):
        w = TemplateRFBuilder(
            coords16, resolvable_distance_mm=0.40, k=k
        ).compute_weights()
        assert torch.all((w != 0).sum(dim=1) == k)


def test_k_is_clamped_to_receptor_count():
    coords = ReceptorGrid(grid_size=(3, 3), spacing=0.15).get_receptor_coordinates()
    w = TemplateRFBuilder(coords, resolvable_distance_mm=0.4, k=28).compute_weights()
    assert torch.all((w != 0).sum(dim=1) == 9)


def test_l2_rows_have_unit_norm(coords16):
    w = TemplateRFBuilder(coords16, resolvable_distance_mm=0.40).compute_weights()
    assert torch.allclose(w.norm(dim=1), torch.ones(w.shape[0]), atol=1e-6)


def test_sum_rows_sum_to_one(coords16):
    w = TemplateRFBuilder(
        coords16, resolvable_distance_mm=0.40, normalize="sum"
    ).compute_weights()
    assert torch.allclose(w.sum(dim=1), torch.ones(w.shape[0]), atol=1e-6)


def test_weight_scale_multiplies(coords16):
    w1 = TemplateRFBuilder(coords16, resolvable_distance_mm=0.40).compute_weights()
    w3 = TemplateRFBuilder(
        coords16, resolvable_distance_mm=0.40, weight_scale=3.0
    ).compute_weights()
    assert torch.allclose(w3, 3.0 * w1)


def _interior_neurons(builder):
    """Indices of neurons whose k nearest receptors are far from the border."""
    centers = builder.neuron_centers
    x_lo, x_hi = (
        builder.receptor_coords[:, 0].min(),
        builder.receptor_coords[:, 0].max(),
    )
    y_lo, y_hi = (
        builder.receptor_coords[:, 1].min(),
        builder.receptor_coords[:, 1].max(),
    )
    margin = 4 * builder.sigma_mm
    inside = (
        (centers[:, 0] > x_lo + margin)
        & (centers[:, 0] < x_hi - margin)
        & (centers[:, 1] > y_lo + margin)
        & (centers[:, 1] < y_hi - margin)
    )
    return inside.nonzero(as_tuple=True)[0].tolist()


def test_interior_weights_equal_analytic_gaussian(coords16):
    b = TemplateRFBuilder(coords16, resolvable_distance_mm=0.40, normalize="none")
    w = b.compute_weights()
    interior = _interior_neurons(b)
    assert interior, "expected interior neurons on a 16x16 grid"
    n = interior[0]
    nz = (w[n] != 0).nonzero(as_tuple=True)[0]
    assert nz.numel() == 28
    r2 = ((coords16[nz] - b.neuron_centers[n]) ** 2).sum(dim=1)
    expected = torch.exp(-r2 / (2 * b.sigma_mm**2))
    assert torch.allclose(w[n, nz], expected, atol=1e-6)
    # the k nearest: every zero-weight receptor is at least as far as the
    # farthest non-zero one
    all_r2 = ((coords16 - b.neuron_centers[n]) ** 2).sum(dim=1)
    assert all_r2[w[n] == 0].min() >= r2.max() - 1e-9


def test_interior_neurons_share_one_template(coords16):
    b = TemplateRFBuilder(coords16, resolvable_distance_mm=0.40)
    w = b.compute_weights()
    interior = _interior_neurons(b)
    assert len(interior) >= 2
    a, c = interior[0], interior[-1]
    assert a != c
    ta = torch.sort(w[a][w[a] != 0]).values
    tc = torch.sort(w[c][w[c] != 0]).values
    assert torch.allclose(ta, tc, atol=1e-6)


def test_two_builds_are_bit_identical(coords16):
    w1 = TemplateRFBuilder(coords16, resolvable_distance_mm=0.40).build()
    w2 = TemplateRFBuilder(coords16, resolvable_distance_mm=0.40).build()
    assert torch.equal(w1.weights, w2.weights)
    assert torch.equal(w1.neuron_centers, w2.neuron_centers)


def test_lattice_is_row_major_in_xy_order(coords16):
    # Section 2 ordering: the first index is x (slow), the second is y (fast),
    # like torch.meshgrid(x, y, indexing="ij").flatten().
    b = TemplateRFBuilder(coords16, resolvable_distance_mm=0.40)
    c = b.neuron_centers
    nx, ny = b.lattice_shape
    assert torch.allclose(c[:ny, 0], c[0, 0].expand(ny))  # first ny share x
    assert torch.all(c[1:ny, 1] > c[:-1][: ny - 1, 1])  # y increasing
    assert c[ny, 0] > c[0, 0]  # next block moves in x
    # lattice pitch is d in both directions, inset by d/2 from the extended box
    assert (c[ny, 0] - c[0, 0]).item() == pytest.approx(0.40, abs=1e-6)
    assert (c[1, 1] - c[0, 1]).item() == pytest.approx(0.40, abs=1e-6)
    side = 15 * 0.15 + 0.15  # extreme-to-extreme plus half a spacing each side
    assert (c[:, 0].min() - (coords16[:, 0].min() - 0.075)).item() == pytest.approx(
        0.20, abs=1e-6
    )
    assert (c[:, 0].max() - c[:, 0].min()).item() == pytest.approx(
        side - 0.40, abs=1e-6
    )


def test_ties_break_toward_lower_receptor_index():
    # one neuron exactly between two receptors on a 1-D line: with k=1 the
    # lower index wins.
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    b = TemplateRFBuilder(
        coords, sigma_mm=0.5, pitch_mm=1.0, k=1, edge_offset_mm=0.0, normalize="none"
    )
    # force a single neuron centre at the midpoint by hand
    b.neuron_centers = torch.tensor([[0.5, 0.0]])
    b.num_neurons = 1
    w = b.compute_weights()
    assert w[0, 0] != 0 and w[0, 1] == 0


def test_neuron_centers_argument_is_ignored_with_warning(coords16):
    with pytest.warns(UserWarning, match="neuron_centers"):
        b = TemplateRFBuilder(coords16, torch.zeros(3, 2), resolvable_distance_mm=0.40)
    assert b.num_neurons == 36


def test_round_trip_and_contract(coords16):
    b = TemplateRFBuilder(coords16, resolvable_distance_mm=0.40, k=12)
    d = b.to_dict()
    assert d["method"] == "template"
    rebuilt = TemplateRFBuilder.from_config({**d, "receptor_coords": coords16})
    assert torch.equal(rebuilt.compute_weights(), b.compute_weights())
    check_component("innervation", TemplateRFBuilder, b)
    bank = b.build()
    assert bank.provenance["builder"] == "template"
    assert bank.provenance["derived"]["sigma_mm"] == pytest.approx(0.40 / math.pi)
    assert bank.provenance["derived"]["num_neurons"] == 36


def test_registry_create_and_build(coords16):
    b = INNERVATION_REGISTRY.create(
        "template", receptor_coords=coords16, resolvable_distance_mm=0.40
    )
    assert b.build().num_neurons == 36
