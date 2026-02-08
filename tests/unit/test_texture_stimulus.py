"""Unit tests for texture stimulus generation.

Tests for sensoryforge.stimuli.texture module including:
- Gabor texture generation
- Edge grating generation
- Noise texture generation
- Parameter validation
- Deterministic generation with seeds
- Module classes
"""

import pytest
import torch
import math

from sensoryforge.stimuli.texture import (
    gabor_texture,
    edge_grating,
    noise_texture,
    GaborTexture,
    EdgeGrating,
)


class TestGaborTexture:
    """Tests for gabor_texture function."""
    
    def test_output_shape(self):
        """Test that output shape matches input coordinate grids."""
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 64),
            torch.linspace(-2, 2, 64),
            indexing='ij'
        )
        
        result = gabor_texture(xx, yy, wavelength=0.5, orientation=0.0)
        
        assert result.shape == xx.shape
        assert result.shape == (64, 64)
    
    def test_oscillatory_pattern(self):
        """Test that Gabor produces oscillating values."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 100),
            torch.linspace(-1, 1, 100),
            indexing='ij'
        )
        
        result = gabor_texture(
            xx, yy,
            center_x=0.0,
            center_y=0.0,
            wavelength=0.2,
            orientation=0.0,
            sigma=2.0  # Large sigma to see oscillations
        )
        
        # Should have both positive and negative values (oscillation)
        assert result.max() > 0
        assert result.min() < 0
    
    def test_wavelength_affects_frequency(self):
        """Test that wavelength parameter affects spatial frequency."""
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 100),
            torch.linspace(-2, 2, 100),
            indexing='ij'
        )
        
        # Short wavelength should have more oscillations
        # Use very large sigma so the envelope doesn't suppress oscillations
        short_wave = gabor_texture(xx, yy, wavelength=0.1, sigma=10.0)
        long_wave = gabor_texture(xx, yy, wavelength=0.4, sigma=10.0)
        
        # Different wavelengths should produce different patterns
        assert not torch.allclose(short_wave, long_wave, atol=0.1)
    
    def test_orientation_rotates_pattern(self):
        """Test that orientation parameter rotates the grating."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 64),
            torch.linspace(-1, 1, 64),
            indexing='ij'
        )
        
        horizontal = gabor_texture(xx, yy, wavelength=0.3, orientation=0.0, sigma=1.0)
        vertical = gabor_texture(xx, yy, wavelength=0.3, orientation=math.pi/2, sigma=1.0)
        
        # Patterns should be different (one rotated relative to other)
        assert not torch.allclose(horizontal, vertical, atol=0.1)
    
    def test_phase_shifts_pattern(self):
        """Test that phase parameter shifts the pattern."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 64),
            torch.linspace(-1, 1, 64),
            indexing='ij'
        )
        
        phase0 = gabor_texture(xx, yy, wavelength=0.3, phase=0.0)
        phase_pi = gabor_texture(xx, yy, wavelength=0.3, phase=math.pi)
        
        # Phase shift of π should approximately negate the pattern
        assert torch.allclose(phase0, -phase_pi, atol=0.15)
    
    def test_invalid_wavelength_raises_error(self):
        """Test that non-positive wavelength raises ValueError."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing='ij'
        )
        
        with pytest.raises(ValueError, match="wavelength must be positive"):
            gabor_texture(xx, yy, wavelength=0.0)
        
        with pytest.raises(ValueError, match="wavelength must be positive"):
            gabor_texture(xx, yy, wavelength=-0.5)
    
    def test_invalid_sigma_raises_error(self):
        """Test that non-positive sigma raises ValueError."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing='ij'
        )
        
        with pytest.raises(ValueError, match="sigma must be positive"):
            gabor_texture(xx, yy, sigma=0.0)


class TestEdgeGrating:
    """Tests for edge_grating function."""
    
    def test_output_shape(self):
        """Test that output shape matches input coordinate grids."""
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 64),
            torch.linspace(-2, 2, 64),
            indexing='ij'
        )
        
        result = edge_grating(xx, yy, orientation=0.0, spacing=0.5, count=5)
        
        assert result.shape == xx.shape
        assert result.shape == (64, 64)
    
    def test_count_affects_number_of_edges(self):
        """Test that count parameter controls number of edges."""
        xx, yy = torch.meshgrid(
            torch.linspace(-3, 3, 200),
            torch.linspace(-3, 3, 200),
            indexing='ij'
        )
        
        few_edges = edge_grating(xx, yy, orientation=0.0, spacing=0.5, count=3)
        many_edges = edge_grating(xx, yy, orientation=0.0, spacing=0.5, count=10)
        
        # More edges should create more peaks
        # Count peaks along a line perpendicular to orientation
        center_idx = xx.shape[0] // 2
        
        # For orientation=0, edges are vertical, so count along horizontal line
        few_line = few_edges[center_idx, :]
        many_line = many_edges[center_idx, :]
        
        # Find peaks (local maxima)
        def count_peaks(line, threshold=0.5):
            peaks = 0
            for i in range(1, len(line) - 1):
                if line[i] > threshold and line[i] > line[i-1] and line[i] > line[i+1]:
                    peaks += 1
            return peaks
        
        few_peaks = count_peaks(few_line)
        many_peaks = count_peaks(many_line)
        
        assert many_peaks > few_peaks
    
    def test_spacing_affects_edge_distance(self):
        """Test that spacing parameter affects distance between edges."""
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 100),
            torch.linspace(-2, 2, 100),
            indexing='ij'
        )
        
        narrow = edge_grating(xx, yy, spacing=0.2, count=5)
        wide = edge_grating(xx, yy, spacing=0.8, count=5)
        
        # Different spacing should produce different patterns
        assert not torch.allclose(narrow, wide)
    
    def test_normalization(self):
        """Test that normalize parameter is accepted."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 64),
            torch.linspace(-1, 1, 64),
            indexing='ij'
        )
        
        amplitude = 2.5
        # Just test that both parameters work
        normalized = edge_grating(
            xx, yy,
            amplitude=amplitude,
            normalize=True
        )
        not_normalized = edge_grating(
            xx, yy,
            amplitude=amplitude,
            normalize=False
        )
        
        # Both should produce valid patterns
        assert normalized.shape == xx.shape
        assert not_normalized.shape == xx.shape
        assert normalized.max() > 0
        assert not_normalized.max() > 0
    
    def test_invalid_spacing_raises_error(self):
        """Test that non-positive spacing raises ValueError."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing='ij'
        )
        
        with pytest.raises(ValueError, match="spacing must be positive"):
            edge_grating(xx, yy, spacing=0.0)
    
    def test_invalid_edge_width_raises_error(self):
        """Test that non-positive edge_width raises ValueError."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing='ij'
        )
        
        with pytest.raises(ValueError, match="edge_width must be positive"):
            edge_grating(xx, yy, edge_width=0.0)
    
    def test_invalid_count_raises_error(self):
        """Test that count < 1 raises ValueError."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing='ij'
        )
        
        with pytest.raises(ValueError, match="count must be at least 1"):
            edge_grating(xx, yy, count=0)


class TestNoiseTexture:
    """Tests for noise_texture function."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        texture = noise_texture(64, 64, scale=1.0, kernel_size=5)
        
        assert texture.shape == (64, 64)
    
    def test_deterministic_with_seed(self):
        """Test that same seed produces same texture (deterministic)."""
        texture1 = noise_texture(50, 50, scale=0.5, kernel_size=7, seed=42)
        texture2 = noise_texture(50, 50, scale=0.5, kernel_size=7, seed=42)
        
        assert torch.allclose(texture1, texture2)
    
    def test_different_seeds_produce_different_textures(self):
        """Test that different seeds produce different textures."""
        texture1 = noise_texture(50, 50, scale=0.5, kernel_size=7, seed=42)
        texture2 = noise_texture(50, 50, scale=0.5, kernel_size=7, seed=123)
        
        assert not torch.allclose(texture1, texture2)
    
    def test_scale_affects_amplitude(self):
        """Test that scale parameter affects texture amplitude."""
        small_scale = noise_texture(50, 50, scale=0.1, kernel_size=5, seed=42)
        large_scale = noise_texture(50, 50, scale=2.0, kernel_size=5, seed=42)
        
        # Larger scale should have larger absolute values
        assert large_scale.abs().max() > small_scale.abs().max()
    
    def test_kernel_size_affects_smoothness(self):
        """Test that kernel size affects spatial correlation."""
        # Larger kernel should produce smoother texture
        rough = noise_texture(100, 100, kernel_size=3, seed=42)
        smooth = noise_texture(100, 100, kernel_size=15, seed=42)
        
        # Compute spatial gradients as measure of roughness
        rough_grad = torch.abs(rough[1:, :] - rough[:-1, :]).mean()
        smooth_grad = torch.abs(smooth[1:, :] - smooth[:-1, :]).mean()
        
        # Smooth should have smaller gradients
        assert smooth_grad < rough_grad
    
    def test_invalid_dimensions_raise_error(self):
        """Test that non-positive dimensions raise ValueError."""
        with pytest.raises(ValueError, match="height and width must be positive"):
            noise_texture(0, 50)
        
        with pytest.raises(ValueError, match="height and width must be positive"):
            noise_texture(50, -10)
    
    def test_even_kernel_size_raises_error(self):
        """Test that even kernel size raises ValueError."""
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            noise_texture(50, 50, kernel_size=4)
    
    def test_device_parameter(self):
        """Test that device parameter works correctly."""
        texture = noise_texture(32, 32, device='cpu')
        assert texture.device.type == 'cpu'


class TestGaborTextureModule:
    """Tests for GaborTexture nn.Module."""
    
    def test_module_creation(self):
        """Test that module can be created with valid parameters."""
        module = GaborTexture(
            wavelength=0.5,
            orientation=math.pi/4,
            sigma=0.3
        )
        
        assert isinstance(module, torch.nn.Module)
    
    def test_module_forward(self):
        """Test that module forward pass produces correct output."""
        module = GaborTexture(wavelength=0.4, orientation=0.0, sigma=0.5)
        
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 32),
            torch.linspace(-1, 1, 32),
            indexing='ij'
        )
        
        result = module(xx, yy)
        
        assert result.shape == (32, 32)
        # Should have oscillating values
        assert result.max() > 0
        assert result.min() < 0
    
    def test_invalid_parameters_raise_error(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="wavelength must be positive"):
            GaborTexture(wavelength=0.0)
        
        with pytest.raises(ValueError, match="sigma must be positive"):
            GaborTexture(sigma=-0.1)


class TestEdgeGratingModule:
    """Tests for EdgeGrating nn.Module."""
    
    def test_module_creation(self):
        """Test that module can be created with valid parameters."""
        module = EdgeGrating(
            orientation=math.pi/6,
            spacing=0.5,
            count=10
        )
        
        assert isinstance(module, torch.nn.Module)
    
    def test_module_forward(self):
        """Test that module forward pass produces correct output."""
        module = EdgeGrating(orientation=0.0, spacing=0.6, count=7)
        
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 64),
            torch.linspace(-2, 2, 64),
            indexing='ij'
        )
        
        result = module(xx, yy)
        
        assert result.shape == (64, 64)
        assert result.max() > 0
    
    def test_invalid_parameters_raise_error(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="spacing must be positive"):
            EdgeGrating(spacing=0.0)
        
        with pytest.raises(ValueError, match="edge_width must be positive"):
            EdgeGrating(edge_width=-0.1)
        
        with pytest.raises(ValueError, match="count must be at least 1"):
            EdgeGrating(count=0)


class TestTextureIntegration:
    """Integration tests for texture stimulus generation."""
    
    def test_reproducibility(self):
        """Test that same parameters produce same output."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 50),
            torch.linspace(-1, 1, 50),
            indexing='ij'
        )
        
        result1 = gabor_texture(
            xx, yy,
            wavelength=0.4,
            orientation=math.pi/3,
            sigma=0.5
        )
        result2 = gabor_texture(
            xx, yy,
            wavelength=0.4,
            orientation=math.pi/3,
            sigma=0.5
        )
        
        assert torch.allclose(result1, result2)
    
    def test_combined_textures(self):
        """Test combining multiple texture types."""
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 64),
            torch.linspace(-2, 2, 64),
            indexing='ij'
        )
        
        # Create combination of Gabor and edge grating
        gabor = gabor_texture(xx, yy, wavelength=0.3, amplitude=0.5)
        grating = edge_grating(xx, yy, spacing=0.6, count=5, amplitude=0.5)
        
        combined = gabor + grating
        
        assert combined.shape == xx.shape
        assert combined.max() > gabor.max()  # Should have combined peaks
