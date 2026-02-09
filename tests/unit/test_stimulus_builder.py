"""Unit tests for composable stimulus builder API.

Tests for Phase 1.4: Composable stimulus architecture with builder pattern,
motion attachment, and composition.
"""

import pytest
import torch
import math
from sensoryforge.stimuli import (
    Stimulus,
    StaticStimulus,
    MovingStimulus,
    CompositeStimulus,
    with_motion,
)


class TestStaticStimulus:
    """Test suite for StaticStimulus class."""
    
    @pytest.fixture
    def coordinate_grid(self):
        """Create simple 32x32 coordinate grid."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 32),
            torch.linspace(-1, 1, 32),
            indexing='ij',
        )
        return xx, yy
    
    def test_gaussian_static_stimulus(self, coordinate_grid):
        """Test Gaussian static stimulus generation."""
        xx, yy = coordinate_grid
        stim = Stimulus.gaussian(amplitude=1.0, sigma=0.3, center=(0.0, 0.0))
        
        output = stim(xx, yy)
        
        assert output.shape == xx.shape
        assert torch.isfinite(output).all()
        assert output.max() <= 1.0 + 1e-5  # Peak ≈ amplitude
    
    def test_point_static_stimulus(self, coordinate_grid):
        """Test point (binary disc) static stimulus."""
        xx, yy = coordinate_grid
        stim = Stimulus.point(amplitude=2.0, diameter_mm=0.4, center=(0.0, 0.0))
        
        output = stim(xx, yy)
        
        assert output.shape == xx.shape
        # Should be binary (0 or amplitude)
        unique_vals = torch.unique(output)
        assert len(unique_vals) <= 3  # 0, amplitude, maybe edge artifacts
    
    def test_edge_static_stimulus(self, coordinate_grid):
        """Test oriented edge stimulus."""
        xx, yy = coordinate_grid
        stim = Stimulus.edge(amplitude=1.5, orientation=math.pi/4, width=0.05)
        
        output = stim(xx, yy)
        
        assert output.shape == xx.shape
        assert torch.isfinite(output).all()
    
    def test_gabor_static_stimulus(self, coordinate_grid):
        """Test Gabor texture stimulus."""
        xx, yy = coordinate_grid
        stim = Stimulus.gabor(
            amplitude=1.0,
            sigma=0.4,
            wavelength=0.3,
            orientation=math.pi/6,
            phase=0.0,
        )
        
        output = stim(xx, yy)
        
        assert output.shape == xx.shape
        assert torch.isfinite(output).all()
        # Gabor should have positive and negative values
        assert output.min() < 0
        assert output.max() > 0
    
    def test_edge_grating_stimulus(self, coordinate_grid):
        """Test edge grating stimulus."""
        xx, yy = coordinate_grid
        stim = Stimulus.edge_grating(
            amplitude=1.0,
            orientation=0.0,
            spacing=0.5,
            count=5,
            edge_width=0.03,
        )
        
        output = stim(xx, yy)
        
        assert output.shape == xx.shape
        assert torch.isfinite(output).all()
    
    def test_device_placement(self):
        """Test stimulus device placement."""
        stim_cpu = Stimulus.gaussian(amplitude=1.0, sigma=0.3, device='cpu')
        assert stim_cpu.device == torch.device('cpu')
        
        if torch.cuda.is_available():
            stim_gpu = Stimulus.gaussian(amplitude=1.0, sigma=0.3, device='cuda')
            assert stim_gpu.device.type == 'cuda'
    
    def test_static_stimulus_serialization(self):
        """Test to_dict and from_config for StaticStimulus."""
        stim = Stimulus.gaussian(amplitude=2.0, sigma=0.4, center=(0.5, 0.5))
        
        config = stim.to_dict()
        
        assert config['type'] == 'gaussian'
        assert config['params']['amplitude'] == 2.0
        assert config['params']['sigma'] == 0.4
        assert config['params']['center_x'] == 0.5
        assert config['params']['center_y'] == 0.5
        
        # Reconstruct
        stim_reloaded = StaticStimulus.from_config(config)
        assert stim_reloaded.stim_type == stim.stim_type
        assert stim_reloaded.params == stim.params
    
    def test_reset_state(self):
        """Test reset_state (static stimuli have no state)."""
        stim = Stimulus.gaussian(amplitude=1.0, sigma=0.3)
        stim.reset_state()  # Should not raise


class TestMovingStimulus:
    """Test suite for MovingStimulus class."""
    
    @pytest.fixture
    def coordinate_grid(self):
        """Create simple coordinate grid."""
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 64),
            torch.linspace(-2, 2, 64),
            indexing='ij',
        )
        return xx, yy
    
    @pytest.fixture
    def base_gaussian(self):
        """Create base Gaussian stimulus."""
        return Stimulus.gaussian(amplitude=1.0, sigma=0.2, center=(0.0, 0.0))
    
    def test_linear_motion(self, base_gaussian, coordinate_grid):
        """Test linear motion trajectory."""
        xx, yy = coordinate_grid
        
        moving_stim = base_gaussian.with_motion(
            'linear',
            start=(-1.0, -1.0),
            end=(1.0, 1.0),
            num_steps=50,
        )
        
        assert isinstance(moving_stim, MovingStimulus)
        assert moving_stim.motion_type == 'linear'
        assert moving_stim.trajectory.shape == (50, 2)
        
        # First position should be start
        assert torch.allclose(moving_stim.trajectory[0], torch.tensor([-1.0, -1.0]))
        # Last position should be end
        assert torch.allclose(moving_stim.trajectory[-1], torch.tensor([1.0, 1.0]))
        
        # Generate stimulus at initial position
        output = moving_stim(xx, yy)
        assert output.shape == xx.shape
        assert torch.isfinite(output).all()
    
    def test_circular_motion(self, base_gaussian, coordinate_grid):
        """Test circular motion trajectory."""
        xx, yy = coordinate_grid
        
        moving_stim = base_gaussian.with_motion(
            'circular',
            center=(0.0, 0.0),
            radius=1.0,
            num_steps=100,
            start_angle=0.0,
            end_angle=2 * math.pi,
        )
        
        assert moving_stim.motion_type == 'circular'
        assert moving_stim.trajectory.shape == (100, 2)
        
        # First position should be at (radius, 0) relative to center
        assert torch.allclose(
            moving_stim.trajectory[0],
            torch.tensor([1.0, 0.0]),
            atol=1e-5,
        )
        
        # Verify circular path (all points at radius distance from center)
        distances = torch.norm(moving_stim.trajectory, dim=1)
        assert torch.allclose(distances, torch.tensor(1.0), atol=1e-3)
        
        output = moving_stim(xx, yy)
        assert output.shape == xx.shape
    
    def test_stationary_motion(self, base_gaussian, coordinate_grid):
        """Test stationary (no motion) trajectory."""
        xx, yy = coordinate_grid
        
        moving_stim = base_gaussian.with_motion(
            'stationary',
            center=(0.5, 0.5),
            num_steps=10,
        )
        
        assert moving_stim.motion_type == 'stationary'
        assert moving_stim.trajectory.shape == (10, 2)
        
        # All positions should be the same
        for i in range(10):
            assert torch.allclose(
                moving_stim.trajectory[i],
                torch.tensor([0.5, 0.5]),
            )
    
    def test_step_advance(self, base_gaussian, coordinate_grid):
        """Test step() advances time step."""
        xx, yy = coordinate_grid
        
        moving_stim = base_gaussian.with_motion(
            'linear',
            start=(0.0, 0.0),
            end=(1.0, 0.0),
            num_steps=10,
        )
        
        assert moving_stim.current_step == 0
        
        # Step forward
        moving_stim.step()
        assert moving_stim.current_step == 1
        
        # Step multiple times
        for _ in range(5):
            moving_stim.step()
        assert moving_stim.current_step == 6
        
        # Cannot step beyond trajectory length
        for _ in range(10):
            moving_stim.step()
        assert moving_stim.current_step == 9  # Capped at len-1
    
    def test_reset_state(self, base_gaussian):
        """Test reset_state returns to initial step."""
        moving_stim = base_gaussian.with_motion(
            'linear',
            start=(0.0, 0.0),
            end=(1.0, 1.0),
            num_steps=20,
        )
        
        # Advance several steps
        for _ in range(10):
            moving_stim.step()
        assert moving_stim.current_step == 10
        
        # Reset
        moving_stim.reset_state()
        assert moving_stim.current_step == 0
    
    def test_missing_motion_params(self, base_gaussian):
        """Test error handling for missing motion parameters."""
        with pytest.raises(ValueError, match="Linear motion requires"):
            base_gaussian.with_motion('linear', num_steps=10)  # Missing start/end
        
        with pytest.raises(ValueError, match="Circular motion requires"):
            base_gaussian.with_motion('circular', num_steps=10)  # Missing center/radius
    
    def test_unknown_motion_type(self, base_gaussian):
        """Test error handling for unknown motion type."""
        with pytest.raises(ValueError, match="Unknown motion type"):
            base_gaussian.with_motion('spiral', num_steps=10)
    
    def test_moving_stimulus_serialization(self, base_gaussian):
        """Test serialization of MovingStimulus."""
        moving_stim = base_gaussian.with_motion(
            'linear',
            start=(0.0, 0.0),
            end=(1.0, 1.0),
            num_steps=50,
        )
        
        config = moving_stim.to_dict()
        
        assert config['class'] == 'MovingStimulus'
        assert config['motion_type'] == 'linear'
        assert config['motion_params']['start'] == (0.0, 0.0)
        assert config['motion_params']['end'] == (1.0, 1.0)
        
        # Reconstruct
        reloaded = MovingStimulus.from_config(config)
        assert reloaded.motion_type == moving_stim.motion_type
        assert torch.allclose(reloaded.trajectory, moving_stim.trajectory)


class TestCompositeStimulus:
    """Test suite for CompositeStimulus class."""
    
    @pytest.fixture
    def coordinate_grid(self):
        """Create coordinate grid."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 32),
            torch.linspace(-1, 1, 32),
            indexing='ij',
        )
        return xx, yy
    
    def test_compose_two_static_stimuli_add(self, coordinate_grid):
        """Test composing two static stimuli with addition."""
        xx, yy = coordinate_grid
        
        s1 = Stimulus.gaussian(amplitude=1.0, sigma=0.2, center=(-0.5, 0.0))
        s2 = Stimulus.gaussian(amplitude=1.0, sigma=0.2, center=(0.5, 0.0))
        
        composed = Stimulus.compose([s1, s2], mode='add')
        
        assert isinstance(composed, CompositeStimulus)
        assert len(composed.stimuli) == 2
        assert composed.mode == 'add'
        
        output = composed(xx, yy)
        assert output.shape == xx.shape
        assert torch.isfinite(output).all()
        
        # Should be sum of individual outputs
        out1 = s1(xx, yy)
        out2 = s2(xx, yy)
        expected = out1 + out2
        assert torch.allclose(output, expected)
    
    def test_compose_mode_max(self, coordinate_grid):
        """Test compose with max mode."""
        xx, yy = coordinate_grid
        
        s1 = Stimulus.gaussian(amplitude=1.0, sigma=0.2, center=(-0.5, 0.0))
        s2 = Stimulus.gaussian(amplitude=2.0, sigma=0.2, center=(0.5, 0.0))
        
        composed = Stimulus.compose([s1, s2], mode='max')
        
        output = composed(xx, yy)
        
        # Should be element-wise maximum
        out1 = s1(xx, yy)
        out2 = s2(xx, yy)
        expected = torch.maximum(out1, out2)
        assert torch.allclose(output, expected)
    
    def test_compose_mode_mean(self, coordinate_grid):
        """Test compose with mean mode."""
        xx, yy = coordinate_grid
        
        s1 = Stimulus.gaussian(amplitude=1.0, sigma=0.2, center=(0.0, 0.0))
        s2 = Stimulus.point(amplitude=2.0, diameter_mm=0.4, center=(0.0, 0.0))
        
        composed = Stimulus.compose([s1, s2], mode='mean')
        
        output = composed(xx, yy)
        
        # Should be element-wise mean
        out1 = s1(xx, yy)
        out2 = s2(xx, yy)
        expected = (out1 + out2) / 2
        assert torch.allclose(output, expected)
    
    def test_compose_mode_multiply(self, coordinate_grid):
        """Test compose with multiply mode."""
        xx, yy = coordinate_grid
        
        s1 = Stimulus.gaussian(amplitude=1.0, sigma=0.3, center=(0.0, 0.0))
        s2 = Stimulus.gaussian(amplitude=0.5, sigma=0.3, center=(0.0, 0.0))
        
        composed = Stimulus.compose([s1, s2], mode='multiply')
        
        output = composed(xx, yy)
        
        # Should be element-wise product
        out1 = s1(xx, yy)
        out2 = s2(xx, yy)
        expected = out1 * out2
        assert torch.allclose(output, expected)
    
    def test_compose_three_stimuli(self, coordinate_grid):
        """Test composing three stimuli."""
        xx, yy = coordinate_grid
        
        s1 = Stimulus.gaussian(amplitude=1.0, sigma=0.2, center=(-0.6, 0.0))
        s2 = Stimulus.gaussian(amplitude=1.0, sigma=0.2, center=(0.0, 0.0))
        s3 = Stimulus.gaussian(amplitude=1.0, sigma=0.2, center=(0.6, 0.0))
        
        composed = Stimulus.compose([s1, s2, s3], mode='add')
        
        output = composed(xx, yy)
        assert output.shape == xx.shape
    
    def test_compose_static_and_moving(self, coordinate_grid):
        """Test composing static and moving stimuli."""
        xx, yy = coordinate_grid
        
        static = Stimulus.gaussian(amplitude=1.0, sigma=0.3, center=(0.0, 0.0))
        moving = Stimulus.gaussian(amplitude=1.0, sigma=0.2).with_motion(
            'linear',
            start=(-0.5, -0.5),
            end=(0.5, 0.5),
            num_steps=10,
        )
        
        composed = Stimulus.compose([static, moving], mode='add')
        
        output = composed(xx, yy)
        assert output.shape == xx.shape
        assert torch.isfinite(output).all()
    
    def test_empty_stimuli_list_raises(self):
        """Test that empty stimulus list raises error."""
        with pytest.raises(ValueError, match="requires at least one stimulus"):
            Stimulus.compose([], mode='add')
    
    def test_unknown_composition_mode_raises(self, coordinate_grid):
        """Test that unknown mode raises error."""
        s1 = Stimulus.gaussian(amplitude=1.0, sigma=0.2)
        
        with pytest.raises(ValueError, match="Unknown composition mode"):
            Stimulus.compose([s1], mode='invalid_mode')
    
    def test_reset_state(self, coordinate_grid):
        """Test reset_state cascades to all constituent stimuli."""
        s1 = Stimulus.gaussian(amplitude=1.0, sigma=0.2).with_motion(
            'linear', start=(0, 0), end=(1, 1), num_steps=10
        )
        s2 = Stimulus.gaussian(amplitude=1.0, sigma=0.2).with_motion(
            'linear', start=(0, 0), end=(1, 1), num_steps=10
        )
        
        composed = Stimulus.compose([s1, s2], mode='add')
        
        # Advance both stimuli
        s1.step()
        s2.step()
        assert s1.current_step == 1
        assert s2.current_step == 1
        
        # Reset composite
        composed.reset_state()
        assert s1.current_step == 0
        assert s2.current_step == 0
    
    def test_composite_stimulus_serialization(self):
        """Test serialization of CompositeStimulus."""
        s1 = Stimulus.gaussian(amplitude=1.0, sigma=0.2, center=(0.0, 0.0))
        s2 = Stimulus.point(amplitude=2.0, diameter_mm=0.4, center=(0.5, 0.5))
        
        composed = Stimulus.compose([s1, s2], mode='max')
        
        config = composed.to_dict()
        
        assert config['class'] == 'CompositeStimulus'
        assert config['mode'] == 'max'
        assert len(config['stimuli']) == 2
        
        # Reconstruct
        reloaded = CompositeStimulus.from_config(config)
        assert reloaded.mode == composed.mode
        assert len(reloaded.stimuli) == len(composed.stimuli)


class TestBuilderPatterns:
    """Test advanced builder patterns and method chaining."""
    
    @pytest.fixture
    def coordinate_grid(self):
        """Create coordinate grid."""
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 64),
            torch.linspace(-2, 2, 64),
            indexing='ij',
        )
        return xx, yy
    
    def test_chained_gaussian_with_linear_motion(self, coordinate_grid):
        """Test fluent interface: Gaussian + linear motion."""
        xx, yy = coordinate_grid
        
        stim = Stimulus.gaussian(amplitude=1.0, sigma=0.3).with_motion(
            'linear',
            start=(-1.0, 0.0),
            end=(1.0, 0.0),
            num_steps=50,
        )
        
        assert isinstance(stim, MovingStimulus)
        output = stim(xx, yy)
        assert output.shape == xx.shape
    
    def test_chained_gabor_with_circular_motion(self, coordinate_grid):
        """Test fluent interface: Gabor + circular motion."""
        xx, yy = coordinate_grid
        
        stim = Stimulus.gabor(
            wavelength=0.4,
            orientation=0.0,
            sigma=0.5,
        ).with_motion(
            'circular',
            center=(0.0, 0.0),
            radius=0.8,
            num_steps=100,
        )
        
        assert isinstance(stim, MovingStimulus)
        output = stim(xx, yy)
        assert output.shape == xx.shape
    
    def test_complex_composition(self, coordinate_grid):
        """Test complex multi-stimulus composition."""
        xx, yy = coordinate_grid
        
        # Three static Gaussians at different positions
        s1 = Stimulus.gaussian(amplitude=1.0, sigma=0.2, center=(-0.8, 0.0))
        s2 = Stimulus.gaussian(amplitude=1.5, sigma=0.3, center=(0.0, 0.0))
        s3 = Stimulus.gaussian(amplitude=1.0, sigma=0.2, center=(0.8, 0.0))
        
        # One moving Gabor
        s4 = Stimulus.gabor(wavelength=0.3, orientation=math.pi/4).with_motion(
            'circular', center=(0.0, 0.0), radius=0.5, num_steps=100
        )
        
        # Compose all
        composed = Stimulus.compose([s1, s2, s3, s4], mode='add')
        
        output = composed(xx, yy)
        assert output.shape == xx.shape
        assert torch.isfinite(output).all()
    
    def test_functional_with_motion_api(self, coordinate_grid):
        """Test functional with_motion API (non-chained)."""
        xx, yy = coordinate_grid
        
        base = Stimulus.gaussian(amplitude=1.0, sigma=0.3)
        moving = with_motion(
            base,
            'linear',
            start=(0.0, 0.0),
            end=(1.0, 1.0),
            num_steps=20,
        )
        
        assert isinstance(moving, MovingStimulus)
        output = moving(xx, yy)
        assert output.shape == xx.shape


class TestBackwardCompatibility:
    """Test that new API doesn't break existing functionality."""
    
    def test_can_still_use_legacy_functions(self):
        """Test that legacy functional API still works."""
        from sensoryforge.stimuli import (
            gaussian_pressure_torch,
            point_pressure_torch,
        )
        
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 32),
            torch.linspace(-1, 1, 32),
            indexing='ij',
        )
        
        # Legacy function should still work
        output = gaussian_pressure_torch(xx, yy, 0.0, 0.0, amplitude=1.0, sigma=0.2)
        assert output.shape == xx.shape
        
        output2 = point_pressure_torch(xx, yy, 0.0, 0.0, amplitude=1.0, diameter_mm=0.6)
        assert output2.shape == xx.shape
    
    def test_can_still_use_modular_apis(self):
        """Test that modular functional APIs still work."""
        from sensoryforge.stimuli.gaussian import gaussian_stimulus
        from sensoryforge.stimuli.texture import gabor_texture
        
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 32),
            torch.linspace(-1, 1, 32),
            indexing='ij',
        )
        
        output1 = gaussian_stimulus(xx, yy, 0.0, 0.0, amplitude=1.0, sigma=0.2)
        assert output1.shape == xx.shape
        
        output2 = gabor_texture(xx, yy, wavelength=0.3, orientation=0.0)
        assert output2.shape == xx.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
