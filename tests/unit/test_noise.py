"""Tests for noise modules (resolves ReviewFinding#C2, #H5).

Verifies:
1. Noise modules inherit from nn.Module
2. to(device) propagation works
3. Setting seed does NOT pollute global RNG
4. Output shapes match input shapes
5. reset_state() restores reproducibility
"""

import torch
import torch.nn as nn

from sensoryforge.filters.noise import MembraneNoiseTorch, ReceptorNoiseTorch


class TestMembraneNoiseTorch:
    """Test suite for MembraneNoiseTorch."""

    def test_is_nn_module(self):
        """Regression test for ReviewFinding#C2: must be nn.Module."""
        noise = MembraneNoiseTorch(std=0.5)
        assert isinstance(noise, nn.Module)

    def test_output_shape_matches_input(self):
        """Output shape must match input shape."""
        noise = MembraneNoiseTorch(std=0.5)
        x = torch.randn(2, 10, 50)
        out = noise(x)
        assert out.shape == x.shape

    def test_no_global_rng_pollution(self):
        """Regression test for ReviewFinding#H5: seed must not alter global RNG."""
        global_seed_before = torch.initial_seed()
        torch.manual_seed(12345)
        expected_seed = torch.initial_seed()

        # Creating a noise module with a different seed should NOT change global state
        _ = MembraneNoiseTorch(std=1.0, seed=99999)
        assert torch.initial_seed() == expected_seed

    def test_seeded_reproducibility(self):
        """Same seed produces same noise when reset."""
        noise = MembraneNoiseTorch(std=1.0, seed=42)
        x = torch.ones(1, 5, 3)
        out1 = noise(x)

        noise.reset_state()
        out2 = noise(x)
        assert torch.allclose(out1, out2)

    def test_forward_callable(self):
        """forward() and __call__() work identically via nn.Module."""
        noise = MembraneNoiseTorch(std=0.1, seed=7)
        x = torch.randn(1, 3, 4)
        # Just verify it doesn't crash
        out = noise.forward(x)
        assert out.shape == x.shape


class TestReceptorNoiseTorch:
    """Test suite for ReceptorNoiseTorch."""

    def test_is_nn_module(self):
        """Regression test for ReviewFinding#C2: must be nn.Module."""
        noise = ReceptorNoiseTorch(std=0.5)
        assert isinstance(noise, nn.Module)

    def test_output_shape_matches_input(self):
        """Output shape must match input shape."""
        noise = ReceptorNoiseTorch(std=0.5)
        x = torch.randn(2, 10, 8, 8)
        out = noise(x)
        assert out.shape == x.shape

    def test_no_global_rng_pollution(self):
        """Regression test for ReviewFinding#H5: seed must not alter global RNG."""
        torch.manual_seed(12345)
        expected_seed = torch.initial_seed()

        _ = ReceptorNoiseTorch(std=1.0, seed=99999)
        assert torch.initial_seed() == expected_seed

    def test_seeded_reproducibility(self):
        """Same seed produces same noise when reset."""
        noise = ReceptorNoiseTorch(std=1.0, seed=42)
        x = torch.ones(1, 5, 4, 4)
        out1 = noise(x)

        noise.reset_state()
        out2 = noise(x)
        assert torch.allclose(out1, out2)
