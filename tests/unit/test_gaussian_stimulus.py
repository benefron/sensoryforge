"""Unit tests for Gaussian stimulus generation.

Tests for sensoryforge.stimuli.gaussian module including:
- Basic Gaussian stimulus generation
- Parameter validation
- Multi-Gaussian superposition
- Batched processing
- Device handling
- GaussianStimulus module
"""

import pytest
import torch
import math

from sensoryforge.stimuli.gaussian import (
    gaussian_stimulus,
    multi_gaussian_stimulus,
    batched_gaussian_stimulus,
    GaussianStimulus,
)


class TestGaussianStimulus:
    """Tests for gaussian_stimulus function."""
    
    def test_output_shape(self):
        """Test that output shape matches input coordinate grids."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 32),
            torch.linspace(-1, 1, 32),
            indexing='ij'
        )
        
        result = gaussian_stimulus(xx, yy, 0.0, 0.0, amplitude=1.0, sigma=0.5)
        
        assert result.shape == xx.shape
        assert result.shape == yy.shape
        assert result.shape == (32, 32)
    
    def test_peak_amplitude(self):
        """Test that peak amplitude matches specified value at center."""
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 100),
            torch.linspace(-2, 2, 100),
            indexing='ij'
        )
        
        amplitude = 2.5
        result = gaussian_stimulus(xx, yy, 0.0, 0.0, amplitude=amplitude, sigma=0.3)
        
        # Peak should be very close to amplitude (at center)
        # Allow tolerance due to discrete grid
        assert torch.abs(result.max() - amplitude) < 0.02
    
    def test_centered_at_specified_location(self):
        """Test that Gaussian is centered at the specified coordinates."""
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 100),
            torch.linspace(-2, 2, 100),
            indexing='ij'
        )
        
        center_x, center_y = 0.5, -0.5
        result = gaussian_stimulus(xx, yy, center_x, center_y, amplitude=1.0, sigma=0.2)
        
        # Find the location of maximum value
        max_idx = torch.argmax(result)
        max_i = max_idx // result.shape[1]
        max_j = max_idx % result.shape[1]
        
        # Check that maximum is near specified center
        assert torch.abs(xx[max_i, max_j] - center_x) < 0.1
        assert torch.abs(yy[max_i, max_j] - center_y) < 0.1
    
    def test_sigma_controls_width(self):
        """Test that sigma parameter controls the width of the Gaussian."""
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 100),
            torch.linspace(-2, 2, 100),
            indexing='ij'
        )
        
        narrow = gaussian_stimulus(xx, yy, 0.0, 0.0, amplitude=1.0, sigma=0.1)
        wide = gaussian_stimulus(xx, yy, 0.0, 0.0, amplitude=1.0, sigma=0.5)
        
        # Narrow Gaussian should have higher peak concentration
        # (more mass near center, less at edges)
        center_idx = xx.shape[0] // 2
        edge_idx = 5
        
        narrow_center = narrow[center_idx, center_idx]
        wide_center = wide[center_idx, center_idx]
        
        narrow_edge = narrow[edge_idx, edge_idx]
        wide_edge = wide[edge_idx, edge_idx]
        
        # Both should have similar peak (amplitude=1.0)
        assert torch.abs(narrow_center - wide_center) < 0.1
        
        # But narrow should decay faster (smaller at edges)
        assert narrow_edge < wide_edge
    
    def test_invalid_sigma_raises_error(self):
        """Test that non-positive sigma raises ValueError."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 10),
            torch.linspace(-1, 1, 10),
            indexing='ij'
        )
        
        with pytest.raises(ValueError, match="sigma must be positive"):
            gaussian_stimulus(xx, yy, 0.0, 0.0, sigma=0.0)
        
        with pytest.raises(ValueError, match="sigma must be positive"):
            gaussian_stimulus(xx, yy, 0.0, 0.0, sigma=-0.5)
    
    def test_mismatched_shapes_raises_error(self):
        """Test that mismatched xx and yy shapes raise ValueError."""
        xx = torch.zeros(10, 10)
        yy = torch.zeros(10, 5)
        
        with pytest.raises(ValueError, match="same shape"):
            gaussian_stimulus(xx, yy, 0.0, 0.0)
    
    def test_device_handling(self):
        """Test that device parameter works correctly."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing='ij'
        )
        
        # Test with explicit device
        result_cpu = gaussian_stimulus(xx, yy, 0.0, 0.0, device='cpu')
        assert result_cpu.device.type == 'cpu'
        
        # Test that it inherits device from input when device=None
        result_inherit = gaussian_stimulus(xx, yy, 0.0, 0.0, device=None)
        assert result_inherit.device == xx.device


class TestMultiGaussianStimulus:
    """Tests for multi_gaussian_stimulus function."""
    
    def test_superposition_of_gaussians(self):
        """Test that multiple Gaussians are correctly superposed."""
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 64),
            torch.linspace(-2, 2, 64),
            indexing='ij'
        )
        
        centers = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0)]
        result = multi_gaussian_stimulus(xx, yy, centers, amplitudes=[1.0, 0.5, 0.5])
        
        # Should have three peaks
        # This is a basic test - just check shape and that it's non-zero
        assert result.shape == xx.shape
        assert result.max() > 0
    
    def test_default_amplitudes_and_sigmas(self):
        """Test that default amplitudes and sigmas are applied."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 32),
            torch.linspace(-1, 1, 32),
            indexing='ij'
        )
        
        centers = [(0.0, 0.0), (0.5, 0.5)]
        
        # Should use amplitude=1.0 and sigma=0.2 for both
        result = multi_gaussian_stimulus(xx, yy, centers)
        
        assert result.shape == xx.shape
        assert result.max() > 0
    
    def test_length_mismatch_raises_error(self):
        """Test that mismatched parameter lengths raise ValueError."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing='ij'
        )
        
        centers = [(0.0, 0.0), (1.0, 1.0)]
        
        # Mismatched amplitudes
        with pytest.raises(ValueError, match="amplitudes"):
            multi_gaussian_stimulus(xx, yy, centers, amplitudes=[1.0])
        
        # Mismatched sigmas
        with pytest.raises(ValueError, match="sigmas"):
            multi_gaussian_stimulus(xx, yy, centers, sigmas=[0.2])


class TestBatchedGaussianStimulus:
    """Tests for batched_gaussian_stimulus function."""
    
    def test_batched_output_shape(self):
        """Test that batched output has correct shape [batch, H, W]."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 32),
            torch.linspace(-1, 1, 32),
            indexing='ij'
        )
        
        batch_size = 5
        centers = torch.randn(batch_size, 2)
        
        result = batched_gaussian_stimulus(xx, yy, centers)
        
        assert result.shape == (batch_size, 32, 32)
    
    def test_batched_with_different_parameters(self):
        """Test batched generation with different amplitudes and sigmas."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 32),
            torch.linspace(-1, 1, 32),
            indexing='ij'
        )
        
        centers = torch.tensor([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5]])
        amplitudes = torch.tensor([1.0, 2.0, 0.5])
        sigmas = torch.tensor([0.2, 0.3, 0.1])
        
        result = batched_gaussian_stimulus(xx, yy, centers, amplitudes, sigmas)
        
        assert result.shape == (3, 32, 32)
        
        # Check that different amplitudes produce different peaks
        assert not torch.allclose(result[0].max(), result[1].max())
    
    def test_invalid_centers_shape_raises_error(self):
        """Test that invalid centers shape raises ValueError."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing='ij'
        )
        
        # Wrong shape: should be [batch, 2]
        with pytest.raises(ValueError, match="centers must have shape"):
            batched_gaussian_stimulus(xx, yy, torch.randn(5, 3))
    
    def test_batch_size_mismatch_raises_error(self):
        """Test that mismatched batch sizes raise ValueError."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing='ij'
        )
        
        centers = torch.randn(5, 2)
        amplitudes = torch.ones(3)  # Wrong batch size
        
        with pytest.raises(ValueError, match="batch size"):
            batched_gaussian_stimulus(xx, yy, centers, amplitudes=amplitudes)


class TestGaussianStimulusModule:
    """Tests for GaussianStimulus nn.Module."""
    
    def test_module_creation(self):
        """Test that module can be created with valid parameters."""
        module = GaussianStimulus(
            center_x=0.5,
            center_y=-0.5,
            amplitude=2.0,
            sigma=0.3
        )
        
        assert isinstance(module, torch.nn.Module)
    
    def test_module_forward(self):
        """Test that module forward pass produces correct output."""
        module = GaussianStimulus(center_x=0.0, center_y=0.0, amplitude=1.5, sigma=0.4)
        
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 32),
            torch.linspace(-1, 1, 32),
            indexing='ij'
        )
        
        result = module(xx, yy)
        
        assert result.shape == (32, 32)
        assert result.max() <= 1.5  # Should not exceed amplitude
        assert result.max() > 1.4  # Should be close to amplitude
    
    def test_module_device_movement(self):
        """Test that module can be moved between devices."""
        module = GaussianStimulus(center_x=0.0, center_y=0.0, amplitude=1.0, sigma=0.2)
        
        # Move to CPU (should already be there)
        module_cpu = module.to('cpu')
        assert module_cpu.center_x.device.type == 'cpu'
        
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing='ij'
        )
        
        result = module_cpu(xx, yy)
        assert result.device.type == 'cpu'
    
    def test_invalid_sigma_in_module_raises_error(self):
        """Test that invalid sigma in module creation raises ValueError."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            GaussianStimulus(sigma=0.0)
        
        with pytest.raises(ValueError, match="sigma must be positive"):
            GaussianStimulus(sigma=-0.1)


class TestGaussianStimulusIntegration:
    """Integration tests for Gaussian stimulus generation."""
    
    def test_reproducibility(self):
        """Test that same parameters produce same output."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 50),
            torch.linspace(-1, 1, 50),
            indexing='ij'
        )
        
        result1 = gaussian_stimulus(xx, yy, 0.3, -0.2, amplitude=1.5, sigma=0.25)
        result2 = gaussian_stimulus(xx, yy, 0.3, -0.2, amplitude=1.5, sigma=0.25)
        
        assert torch.allclose(result1, result2)
    
    def test_gaussian_properties(self):
        """Test mathematical properties of Gaussian function."""
        xx, yy = torch.meshgrid(
            torch.linspace(-3, 3, 200),
            torch.linspace(-3, 3, 200),
            indexing='ij'
        )
        
        amplitude = 1.0
        sigma = 0.5
        result = gaussian_stimulus(xx, yy, 0.0, 0.0, amplitude=amplitude, sigma=sigma)
        
        # At center, should equal amplitude
        center = result[100, 100]
        assert torch.abs(center - amplitude) < 0.01
        
        # At distance sigma from center, should be approximately amplitude * exp(-0.5)
        # Find point at approximately (sigma, 0) from center
        idx_sigma = int(100 + sigma / 6.0 * 200)  # Scale to grid
        value_at_sigma = result[100, idx_sigma]
        expected = amplitude * math.exp(-0.5)
        
        # Allow some tolerance due to discretization
        assert torch.abs(value_at_sigma - expected) < 0.15
