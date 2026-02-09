"""Tests for grid spacing computation (resolves ReviewFinding#H3).

Verifies get_grid_spacing returns correct values with ij-indexed meshgrids.
"""

import torch
import pytest

from sensoryforge.core.grid import create_grid_torch, get_grid_spacing, GridManager


class TestGetGridSpacing:
    """Test suite for get_grid_spacing."""

    def test_square_grid_spacing(self):
        """Regression test for ReviewFinding#H3: spacing must match constructor."""
        spacing = 0.15
        gm = GridManager(grid_size=10, spacing=spacing)
        dx, dy = get_grid_spacing(gm.xx, gm.yy)
        assert abs(dx.item() - spacing) < 1e-5
        assert abs(dy.item() - spacing) < 1e-5

    def test_rectangular_grid_spacing(self):
        """Different nx/ny should still return correct dx, dy."""
        xx, yy, x, y = create_grid_torch(grid_size=(5, 10), spacing=0.25)
        dx, dy = get_grid_spacing(xx, yy)
        assert abs(dx.item() - 0.25) < 1e-5
        assert abs(dy.item() - 0.25) < 1e-5

    def test_nonzero_spacing(self):
        """Previously returned (0, 0) due to wrong axis access."""
        xx, yy, _, _ = create_grid_torch(grid_size=20, spacing=0.3)
        dx, dy = get_grid_spacing(xx, yy)
        assert dx.item() > 0.0, "dx must not be zero"
        assert dy.item() > 0.0, "dy must not be zero"

    def test_grid_manager_cached_spacing_matches(self):
        """GridManager.dx/dy must agree with spacing constructor arg."""
        gm = GridManager(grid_size=40, spacing=0.15)
        assert abs(gm.dx.item() - 0.15) < 1e-5
        assert abs(gm.dy.item() - 0.15) < 1e-5
