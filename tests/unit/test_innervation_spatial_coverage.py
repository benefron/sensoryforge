"""
Tests for spatial coverage and bias of neuron innervation.

Validates that SA and RA innervation fields cover the receptor grid
uniformly and that neuron centers span the expected spatial extent.
"""
import pytest
import torch
import numpy as np
from sensoryforge.core.grid import GridManager
from sensoryforge.core.innervation import create_sa_innervation, create_ra_innervation

# ── Shared fixtures ──────────────────────────────────────────────────

GRID_SIZE = 40
SPACING = 0.15
SA_NEURONS_PER_ROW = 7   # 7² = 49 neurons
RA_NEURONS_PER_ROW = 10  # 10² = 100 neurons
NUM_SA_NEURONS = SA_NEURONS_PER_ROW ** 2
NUM_RA_NEURONS = RA_NEURONS_PER_ROW ** 2
RECEPTORS_PER_NEURON = 14
SA_SPREAD = 0.3
RA_SPREAD = 0.39
CONNECTION_STRENGTH = (0.1, 1.0)
SEED = 33
DEVICE = "cpu"


@pytest.fixture
def grid_manager():
    return GridManager(grid_size=GRID_SIZE, spacing=SPACING, device=DEVICE)


@pytest.fixture
def sa_innervation(grid_manager):
    return create_sa_innervation(
        grid_manager,
        neurons_per_row=SA_NEURONS_PER_ROW,
        connections_per_neuron=RECEPTORS_PER_NEURON,
        sigma_d_mm=SA_SPREAD,
        weight_range=CONNECTION_STRENGTH,
        seed=SEED,
    )


@pytest.fixture
def ra_innervation(grid_manager):
    return create_ra_innervation(
        grid_manager,
        neurons_per_row=RA_NEURONS_PER_ROW,
        connections_per_neuron=RECEPTORS_PER_NEURON,
        sigma_d_mm=RA_SPREAD,
        weight_range=CONNECTION_STRENGTH,
        seed=SEED,
    )


# ── Tests ────────────────────────────────────────────────────────────

class TestSpatialCoverage:
    """Verify that innervation fields cover the receptor grid adequately."""

    def test_sa_innervation_shape(self, sa_innervation):
        """SA innervation weights have correct shape [neurons, grid_h, grid_w]."""
        w = sa_innervation.innervation_weights
        assert w.shape == (NUM_SA_NEURONS, GRID_SIZE, GRID_SIZE)

    def test_ra_innervation_shape(self, ra_innervation):
        """RA innervation weights have correct shape [neurons, grid_h, grid_w]."""
        w = ra_innervation.innervation_weights
        assert w.shape == (NUM_RA_NEURONS, GRID_SIZE, GRID_SIZE)

    def test_sa_centers_within_grid(self, sa_innervation, grid_manager):
        """SA neuron centers should lie within the physical grid extent."""
        centers = sa_innervation.neuron_centers.cpu()
        x_1d, y_1d = grid_manager.get_1d_coordinates()
        x_min, x_max = x_1d.min().item(), x_1d.max().item()
        y_min, y_max = y_1d.min().item(), y_1d.max().item()
        assert centers[:, 0].min().item() >= x_min
        assert centers[:, 0].max().item() <= x_max
        assert centers[:, 1].min().item() >= y_min
        assert centers[:, 1].max().item() <= y_max

    def test_ra_centers_within_grid(self, ra_innervation, grid_manager):
        """RA neuron centers should lie within the physical grid extent."""
        centers = ra_innervation.neuron_centers.cpu()
        x_1d, y_1d = grid_manager.get_1d_coordinates()
        x_min, x_max = x_1d.min().item(), x_1d.max().item()
        y_min, y_max = y_1d.min().item(), y_1d.max().item()
        assert centers[:, 0].min().item() >= x_min
        assert centers[:, 0].max().item() <= x_max
        assert centers[:, 1].min().item() >= y_min
        assert centers[:, 1].max().item() <= y_max

    def test_sa_weights_nonnegative(self, sa_innervation):
        """All SA innervation weights should be non-negative."""
        assert (sa_innervation.innervation_weights >= 0).all()

    def test_ra_weights_nonnegative(self, ra_innervation):
        """All RA innervation weights should be non-negative."""
        assert (ra_innervation.innervation_weights >= 0).all()

    def test_most_grid_points_innervated(self, sa_innervation, ra_innervation):
        """A majority of grid points should receive nonzero innervation.

        With sparse populations (7×7 SA + 10×10 RA on 40×40 grid) we
        expect ~75 % coverage.  The threshold is set conservatively at 60 %.
        """
        sa_sum = sa_innervation.innervation_weights.sum(dim=0)
        ra_sum = ra_innervation.innervation_weights.sum(dim=0)
        total = sa_sum + ra_sum
        coverage = (total > 0).float().mean().item()
        assert coverage >= 0.60, f"Only {coverage*100:.1f}% of grid points innervated"

    def test_center_spread(self, sa_innervation, ra_innervation):
        """Neuron centers should span a reasonable fraction of the grid."""
        for inn, label in [(sa_innervation, "SA"), (ra_innervation, "RA")]:
            centers = inn.neuron_centers.cpu()
            x_range = (centers[:, 0].max() - centers[:, 0].min()).item()
            y_range = (centers[:, 1].max() - centers[:, 1].min()).item()
            grid_extent = (GRID_SIZE - 1) * SPACING
            # Centers should span at least 50 % of the grid in each axis
            assert x_range > 0.5 * grid_extent, f"{label} X range too small: {x_range:.3f}"
            assert y_range > 0.5 * grid_extent, f"{label} Y range too small: {y_range:.3f}"
