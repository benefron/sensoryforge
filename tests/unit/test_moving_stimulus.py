"""Unit tests for moving stimulus generation.

Tests for sensoryforge.stimuli.moving module including:
- Linear motion trajectories
- Circular motion trajectories
- Custom path motion
- Velocity profiles
- MovingStimulus module
- Tap sequences
- Slide trajectories
"""

import pytest
import torch
import math

from sensoryforge.stimuli.moving import (
    linear_motion,
    circular_motion,
    custom_path_motion,
    velocity_profile,
    MovingStimulus,
    tap_sequence,
    slide_trajectory,
)


class TestLinearMotion:
    """Tests for linear_motion function."""
    
    def test_output_shape(self):
        """Test that output has correct shape [num_steps, 2]."""
        positions = linear_motion((0.0, 0.0), (1.0, 1.0), 100)
        
        assert positions.shape == (100, 2)
    
    def test_start_and_end_positions(self):
        """Test that trajectory starts and ends at specified positions."""
        start = (0.5, -0.5)
        end = (2.0, 1.5)
        positions = linear_motion(start, end, 50)
        
        # First position should match start
        assert torch.allclose(positions[0], torch.tensor(start), atol=1e-6)
        
        # Last position should match end
        assert torch.allclose(positions[-1], torch.tensor(end), atol=1e-6)
    
    def test_linear_interpolation(self):
        """Test that positions are linearly interpolated."""
        positions = linear_motion((0.0, 0.0), (10.0, 0.0), 11)
        
        # Should have points at 0, 1, 2, ..., 10
        expected_x = torch.linspace(0, 10, 11)
        assert torch.allclose(positions[:, 0], expected_x, atol=1e-6)
        
        # Y should remain 0
        assert torch.allclose(positions[:, 1], torch.zeros(11), atol=1e-6)
    
    def test_diagonal_motion(self):
        """Test diagonal motion trajectory."""
        positions = linear_motion((0.0, 0.0), (1.0, 1.0), 100)
        
        # X and Y should be equal (diagonal)
        assert torch.allclose(positions[:, 0], positions[:, 1], atol=1e-6)
    
    def test_invalid_num_steps_raises_error(self):
        """Test that num_steps < 2 raises ValueError."""
        with pytest.raises(ValueError, match="num_steps must be at least 2"):
            linear_motion((0.0, 0.0), (1.0, 1.0), 1)
    
    def test_device_parameter(self):
        """Test that device parameter works correctly."""
        positions = linear_motion((0.0, 0.0), (1.0, 1.0), 10, device='cpu')
        assert positions.device.type == 'cpu'


class TestCircularMotion:
    """Tests for circular_motion function."""
    
    def test_output_shape(self):
        """Test that output has correct shape [num_steps, 2]."""
        positions = circular_motion((0.0, 0.0), 1.0, 100)
        
        assert positions.shape == (100, 2)
    
    def test_full_circle(self):
        """Test full circle trajectory."""
        radius = 1.0
        positions = circular_motion((0.0, 0.0), radius, 100, 0.0, 2*math.pi)
        
        # All points should be approximately at distance radius from center
        distances = torch.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        assert torch.allclose(distances, torch.ones(100) * radius, atol=1e-5)
        
        # First and last points should be approximately the same (closed circle)
        assert torch.allclose(positions[0], positions[-1], atol=1e-2)
    
    def test_quarter_circle(self):
        """Test quarter circle from right to top."""
        positions = circular_motion((0.0, 0.0), 1.0, 50, 0.0, math.pi/2)
        
        # First position should be at (radius, 0)
        assert torch.allclose(positions[0], torch.tensor([1.0, 0.0]), atol=1e-5)
        
        # Last position should be at (0, radius)
        assert torch.allclose(positions[-1], torch.tensor([0.0, 1.0]), atol=1e-5)
    
    def test_center_offset(self):
        """Test circular motion with non-zero center."""
        center = (2.0, -1.0)
        radius = 0.5
        positions = circular_motion(center, radius, 100)
        
        # All points should be at distance radius from center
        distances = torch.sqrt(
            (positions[:, 0] - center[0])**2 + (positions[:, 1] - center[1])**2
        )
        assert torch.allclose(distances, torch.ones(100) * radius, atol=1e-5)
    
    def test_invalid_num_steps_raises_error(self):
        """Test that num_steps < 2 raises ValueError."""
        with pytest.raises(ValueError, match="num_steps must be at least 2"):
            circular_motion((0.0, 0.0), 1.0, 1)
    
    def test_negative_radius_raises_error(self):
        """Test that negative radius raises ValueError."""
        with pytest.raises(ValueError, match="radius must be non-negative"):
            circular_motion((0.0, 0.0), -1.0, 10)


class TestCustomPathMotion:
    """Tests for custom_path_motion function."""
    
    def test_output_shape(self):
        """Test that output has correct shape [num_steps, 2]."""
        waypoints = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        positions = custom_path_motion(waypoints, 100)
        
        assert positions.shape == (100, 2)
    
    def test_passes_through_waypoints(self):
        """Test that trajectory approximately passes through waypoints."""
        waypoints = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
        # Use same number of steps as waypoints for exact match
        positions = custom_path_motion(waypoints, 3)
        
        waypoints_tensor = torch.tensor(waypoints, dtype=torch.float32)
        assert torch.allclose(positions, waypoints_tensor, atol=1e-5)
    
    def test_interpolation_between_waypoints(self):
        """Test that intermediate points are interpolated."""
        waypoints = [(0.0, 0.0), (2.0, 0.0)]
        positions = custom_path_motion(waypoints, 5)
        
        # Should have: (0, 0), (0.5, 0), (1, 0), (1.5, 0), (2, 0)
        expected_x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        assert torch.allclose(positions[:, 0], expected_x, atol=1e-5)
    
    def test_invalid_waypoints_raises_error(self):
        """Test that fewer than 2 waypoints raises ValueError."""
        with pytest.raises(ValueError, match="At least 2 waypoints required"):
            custom_path_motion([(0.0, 0.0)], 10)
    
    def test_invalid_num_steps_raises_error(self):
        """Test that num_steps < 2 raises ValueError."""
        waypoints = [(0.0, 0.0), (1.0, 1.0)]
        with pytest.raises(ValueError, match="num_steps must be at least 2"):
            custom_path_motion(waypoints, 1)


class TestVelocityProfile:
    """Tests for velocity_profile function."""
    
    def test_constant_profile(self):
        """Test constant velocity profile."""
        profile = velocity_profile(100, "constant")
        
        assert profile.shape == (100,)
        assert torch.allclose(profile, torch.ones(100))
    
    def test_ramp_up_profile(self):
        """Test ramp up profile."""
        profile = velocity_profile(100, "ramp_up")
        
        assert profile.shape == (100,)
        assert profile[0] < profile[-1]  # Should increase
        assert torch.abs(profile[0]) < 0.1  # Start near 0
        assert torch.abs(profile[-1] - 1.0) < 0.1  # End near 1
    
    def test_ramp_down_profile(self):
        """Test ramp down profile."""
        profile = velocity_profile(100, "ramp_down")
        
        assert profile.shape == (100,)
        assert profile[0] > profile[-1]  # Should decrease
        assert torch.abs(profile[0] - 1.0) < 0.1  # Start near 1
        assert torch.abs(profile[-1]) < 0.1  # End near 0
    
    def test_trapezoidal_profile(self):
        """Test trapezoidal profile."""
        profile = velocity_profile(100, "trapezoidal")
        
        assert profile.shape == (100,)
        # Should start near 0, reach 1, then return to 0
        assert profile[0] < 0.1
        assert profile.max() > 0.9
        assert profile[-1] < 0.2
    
    def test_sinusoidal_profile(self):
        """Test sinusoidal profile."""
        profile = velocity_profile(100, "sinusoidal")
        
        assert profile.shape == (100,)
        # Should be bounded in [0, 1]
        assert (profile >= 0).all()
        assert (profile <= 1).all()
        # Should oscillate
        assert profile.max() > 0.9
        assert profile.min() < 0.1
    
    def test_invalid_profile_type_raises_error(self):
        """Test that invalid profile type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown profile_type"):
            velocity_profile(100, "invalid_type")
    
    def test_invalid_num_steps_raises_error(self):
        """Test that num_steps < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_steps must be at least 1"):
            velocity_profile(0, "constant")


class TestMovingStimulus:
    """Tests for MovingStimulus module."""
    
    def test_module_creation(self):
        """Test that module can be created."""
        trajectory = linear_motion((0.0, 0.0), (1.0, 1.0), 50)
        
        def dummy_generator(xx, yy, cx, cy):
            return torch.ones_like(xx)
        
        module = MovingStimulus(trajectory, dummy_generator)
        
        assert isinstance(module, torch.nn.Module)
    
    def test_moving_stimulus_updates_position(self):
        """Test that stimulus position changes over time."""
        # Create trajectory with significant movement
        trajectory = linear_motion((-0.6, 0.0), (0.6, 0.0), 20)
        
        # Gaussian bump stimulus
        from sensoryforge.stimuli.gaussian import gaussian_stimulus
        
        def gauss_generator(xx, yy, cx, cy):
            return gaussian_stimulus(xx, yy, cx, cy, amplitude=1.0, sigma=0.15)
        
        module = MovingStimulus(trajectory, gauss_generator)
        
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 64),
            torch.linspace(-1, 1, 64),
            indexing='ij'
        )
        
        result = module(xx, yy)
        
        assert result.shape == (20, 64, 64)
        
        # Simple check: stimulus should be present at different frames
        # and the overall pattern should change
        first_frame = result[0]
        last_frame = result[-1]
        
        # Both should have non-zero values
        assert first_frame.max() > 0.5
        assert last_frame.max() > 0.5
        
        # But they should be different (stimulus moved)
        assert not torch.allclose(first_frame, last_frame, atol=0.1)
    
    def test_amplitude_modulation(self):
        """Test that amplitude profile modulates stimulus strength."""
        trajectory = linear_motion((0.0, 0.0), (0.0, 0.0), 10)  # Stationary
        amplitude_profile = torch.linspace(0, 1, 10)  # Increasing amplitude
        
        def const_generator(xx, yy, cx, cy):
            return torch.ones_like(xx)
        
        module = MovingStimulus(trajectory, const_generator, amplitude_profile)
        
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing='ij'
        )
        
        result = module(xx, yy)
        
        # Amplitude should increase over time
        assert result[0].max() < result[-1].max()
    
    def test_invalid_trajectory_shape_raises_error(self):
        """Test that invalid trajectory shape raises ValueError."""
        invalid_trajectory = torch.randn(10, 3)  # Wrong shape
        
        def dummy_gen(xx, yy, cx, cy):
            return torch.ones_like(xx)
        
        with pytest.raises(ValueError, match="trajectory must have shape"):
            MovingStimulus(invalid_trajectory, dummy_gen)
    
    def test_amplitude_profile_length_mismatch_raises_error(self):
        """Test that mismatched amplitude profile length raises ValueError."""
        trajectory = linear_motion((0.0, 0.0), (1.0, 1.0), 50)
        amplitude_profile = torch.ones(30)  # Wrong length
        
        def dummy_gen(xx, yy, cx, cy):
            return torch.ones_like(xx)
        
        with pytest.raises(ValueError, match="amplitude_profile length"):
            MovingStimulus(trajectory, dummy_gen, amplitude_profile)


class TestTapSequence:
    """Tests for tap_sequence function."""
    
    def test_tap_sequence_shape(self):
        """Test that tap sequence has correct total duration."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 32),
            torch.linspace(-1, 1, 32),
            indexing='ij'
        )
        
        def dummy_gen(xx, yy, cx, cy):
            return torch.ones_like(xx)
        
        # 3 taps, 10 steps each, 5 steps between
        # Total: 10 + 5 + 10 + 5 + 10 = 40 steps
        sequence = tap_sequence((0.0, 0.0), 3, 10, 5, dummy_gen, xx, yy)
        
        assert sequence.shape == (40, 32, 32)
    
    def test_tap_on_off_pattern(self):
        """Test that taps have correct on/off pattern."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing='ij'
        )
        
        def ones_gen(xx, yy, cx, cy):
            return torch.ones_like(xx)
        
        sequence = tap_sequence((0.0, 0.0), 2, 5, 3, ones_gen, xx, yy)
        
        # Steps 0-4: tap (should be 1)
        assert sequence[2].max() > 0.9
        
        # Steps 5-7: interval (should be 0)
        assert sequence[6].max() < 0.1
        
        # Steps 8-12: tap (should be 1)
        assert sequence[10].max() > 0.9
    
    def test_invalid_parameters_raise_error(self):
        """Test that invalid parameters raise ValueError."""
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 16),
            torch.linspace(-1, 1, 16),
            indexing='ij'
        )
        
        def dummy_gen(xx, yy, cx, cy):
            return torch.ones_like(xx)
        
        with pytest.raises(ValueError, match="tap_duration must be at least 1"):
            tap_sequence((0.0, 0.0), 2, 0, 5, dummy_gen, xx, yy)
        
        with pytest.raises(ValueError, match="interval_duration must be at least 1"):
            tap_sequence((0.0, 0.0), 2, 5, 0, dummy_gen, xx, yy)
        
        with pytest.raises(ValueError, match="num_taps must be at least 1"):
            tap_sequence((0.0, 0.0), 0, 5, 5, dummy_gen, xx, yy)


class TestSlideTrajectory:
    """Tests for slide_trajectory function."""
    
    def test_slide_trajectory_shape(self):
        """Test that slide trajectory has correct shape."""
        trajectory = slide_trajectory((0.0, 0.0), (1.0, 1.0), 100)
        
        assert trajectory.shape == (100, 2)
    
    def test_slide_starts_and_ends_correctly(self):
        """Test that slide starts and ends at specified positions."""
        start = (0.0, 0.0)
        end = (2.0, 1.0)
        trajectory = slide_trajectory(start, end, 50)
        
        assert torch.allclose(trajectory[0], torch.tensor(start), atol=1e-5)
        assert torch.allclose(trajectory[-1], torch.tensor(end), atol=1e-5)
    
    def test_velocity_modulation(self):
        """Test that velocity profile affects trajectory."""
        constant = slide_trajectory((0.0, 0.0), (1.0, 0.0), 100, "constant")
        trapezoidal = slide_trajectory((0.0, 0.0), (1.0, 0.0), 100, "trapezoidal")
        
        # Different velocity profiles should produce different trajectories
        # (except at start and end)
        assert not torch.allclose(constant[1:-1], trapezoidal[1:-1], atol=0.01)


class TestMovingStimuliIntegration:
    """Integration tests for moving stimuli."""
    
    def test_moving_gaussian_stimulus(self):
        """Test moving Gaussian stimulus integration."""
        from sensoryforge.stimuli.gaussian import gaussian_stimulus
        
        # Create circular trajectory
        trajectory = circular_motion((0.0, 0.0), 0.5, 30)
        
        def gauss_gen(xx, yy, cx, cy):
            return gaussian_stimulus(xx, yy, cx, cy, amplitude=1.0, sigma=0.2)
        
        module = MovingStimulus(trajectory, gauss_gen)
        
        xx, yy = torch.meshgrid(
            torch.linspace(-1, 1, 32),
            torch.linspace(-1, 1, 32),
            indexing='ij'
        )
        
        result = module(xx, yy)
        
        assert result.shape == (30, 32, 32)
        # All frames should have non-zero stimulation
        assert (result.max(dim=-1)[0].max(dim=-1)[0] > 0.5).all()
