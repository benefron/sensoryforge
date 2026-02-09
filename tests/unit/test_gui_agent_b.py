"""Tests for GUI Agent B: Extended stimuli (texture and moving) in StimulusDesignerTab."""

import math
import torch
import pytest


class TestTextureStimuli:
    """Test texture stimulus generation functions."""
    
    def test_gabor_texture_import(self):
        """Test that gabor_texture can be imported."""
        from sensoryforge.stimuli.texture import gabor_texture
        assert gabor_texture is not None
    
    def test_edge_grating_import(self):
        """Test that edge_grating can be imported."""
        from sensoryforge.stimuli.texture import edge_grating
        assert edge_grating is not None
    
    def test_noise_texture_import(self):
        """Test that noise_texture can be imported."""
        from sensoryforge.stimuli.texture import noise_texture
        assert noise_texture is not None
    
    def test_gabor_texture_shape(self):
        """Test that gabor_texture produces expected shape."""
        from sensoryforge.stimuli.texture import gabor_texture
        
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 64),
            torch.linspace(-2, 2, 64),
            indexing='ij'
        )
        result = gabor_texture(xx, yy, wavelength=0.5, orientation=0.0, sigma=0.3)
        
        assert result.shape == (64, 64)
        assert result.dtype == xx.dtype
    
    def test_edge_grating_shape(self):
        """Test that edge_grating produces expected shape."""
        from sensoryforge.stimuli.texture import edge_grating
        
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 64),
            torch.linspace(-2, 2, 64),
            indexing='ij'
        )
        result = edge_grating(xx, yy, orientation=0.0, spacing=0.6, count=5)
        
        assert result.shape == (64, 64)
        assert result.dtype == xx.dtype
    
    def test_noise_texture_shape(self):
        """Test that noise_texture produces expected shape."""
        from sensoryforge.stimuli.texture import noise_texture
        
        result = noise_texture(64, 64, scale=1.0, kernel_size=5, seed=42)
        
        assert result.shape == (64, 64)


class TestMovingStimuli:
    """Test moving stimulus generation functions."""
    
    def test_linear_motion_import(self):
        """Test that linear_motion can be imported."""
        from sensoryforge.stimuli.moving import linear_motion
        assert linear_motion is not None
    
    def test_circular_motion_import(self):
        """Test that circular_motion can be imported."""
        from sensoryforge.stimuli.moving import circular_motion
        assert circular_motion is not None
    
    def test_moving_stimulus_import(self):
        """Test that MovingStimulus can be imported."""
        from sensoryforge.stimuli.moving import MovingStimulus
        assert MovingStimulus is not None
    
    def test_slide_trajectory_import(self):
        """Test that slide_trajectory can be imported."""
        from sensoryforge.stimuli.moving import slide_trajectory
        assert slide_trajectory is not None
    
    def test_linear_motion_shape(self):
        """Test that linear_motion produces expected path shape."""
        from sensoryforge.stimuli.moving import linear_motion
        
        path = linear_motion(start=(0.0, 0.0), end=(2.0, 0.0), num_steps=100)
        
        assert path.shape == (100, 2)
        assert torch.allclose(path[0], torch.tensor([0.0, 0.0]))
        assert torch.allclose(path[-1], torch.tensor([2.0, 0.0]))
    
    def test_circular_motion_shape(self):
        """Test that circular_motion produces expected path shape."""
        from sensoryforge.stimuli.moving import circular_motion
        
        path = circular_motion(center=(0.0, 0.0), radius=1.0, num_steps=50)
        
        assert path.shape == (50, 2)
    
    def test_moving_stimulus_shape(self):
        """Test that MovingStimulus produces expected frame tensor shape."""
        from sensoryforge.stimuli.moving import MovingStimulus, linear_motion
        from sensoryforge.stimuli.gaussian import gaussian_stimulus
        
        # Create trajectory
        trajectory = linear_motion((0.0, 0.0), (1.0, 0.0), 20)
        
        # Create spatial generator
        def spatial_gen(xx, yy, cx, cy):
            return gaussian_stimulus(xx, yy, cx, cy, amplitude=1.0, sigma=0.2)
        
        # Create moving stimulus
        moving = MovingStimulus(trajectory, spatial_gen)
        
        # Generate on grid
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 32),
            torch.linspace(-2, 2, 32),
            indexing='ij'
        )
        frames = moving(xx, yy)
        
        assert frames.shape == (20, 32, 32)
        assert frames.dtype == xx.dtype
    
    def test_slide_trajectory_shape(self):
        """Test that slide_trajectory produces expected shape."""
        from sensoryforge.stimuli.moving import slide_trajectory
        
        trajectory = slide_trajectory(
            start=(0.0, 0.0),
            end=(2.0, 0.0),
            num_steps=50,
            velocity_type="constant"
        )
        
        assert trajectory.shape == (50, 2)
        assert torch.allclose(trajectory[0], torch.tensor([0.0, 0.0]), atol=1e-5)
        assert torch.allclose(trajectory[-1], torch.tensor([2.0, 0.0]), atol=1e-5)


class TestGaussianStimulus:
    """Test new gaussian stimulus module."""
    
    def test_gaussian_stimulus_import(self):
        """Test that gaussian_stimulus can be imported."""
        from sensoryforge.stimuli.gaussian import gaussian_stimulus
        assert gaussian_stimulus is not None
    
    def test_gaussian_stimulus_shape(self):
        """Test that gaussian_stimulus produces expected shape."""
        from sensoryforge.stimuli.gaussian import gaussian_stimulus
        
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 2, 50),
            torch.linspace(-2, 2, 50),
            indexing='ij'
        )
        stim = gaussian_stimulus(xx, yy, center_x=0.0, center_y=0.0, amplitude=1.0, sigma=0.5)
        
        assert stim.shape == (50, 50)
        assert stim.dtype == xx.dtype
