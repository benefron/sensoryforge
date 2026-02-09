"""Stimulus generation and pattern synthesis module.

This module provides utilities for generating spatial pressure patterns,
temporal trajectories, and motion profiles for sensory encoding experiments.

**Recommended: Composable Builder API (Phase 1.4)**
    - Stimulus.gaussian(): Create Gaussian bump
    - Stimulus.point(): Create binary disc
    - Stimulus.edge(): Create oriented edge
   - Stimulus.gabor(): Create Gabor texture
    - Stimulus.edge_grating(): Create edge grating
    - Stimulus.compose(): Combine multiple stimuli
    - .with_motion(): Add motion to any stimulus

Example (Builder API):
    >>> from sensoryforge.stimuli import Stimulus
    >>> 
    >>> # Static Gaussian
    >>> s1 = Stimulus.gaussian(amplitude=1.0, sigma=0.3, center=(0.5, 0.5))
    >>> 
    >>> # Gaussian with linear motion
    >>> s2 = Stimulus.gaussian(amplitude=2.0, sigma=0.2).with_motion(
    ...     'linear', start=(0.0, 0.0), end=(1.0, 1.0), num_steps=100
    ... )
    >>> 
    >>> # Gabor with circular motion
    >>> s3 = Stimulus.gabor(wavelength=0.5).with_motion(
    ...     'circular', center=(0.0, 0.0), radius=0.5, num_steps=200
    ... )
    >>> 
    >>> # Composite stimulus
    >>> combined = Stimulus.compose([s1, s2, s3], mode='add')

Modules:
    gaussian: Gaussian bump stimuli with superposition support
    texture: Gabor textures, edge gratings, and noise patterns
    moving: Moving contacts and temporal trajectories
    builder: Composable stimulus builder API (NEW)

Legacy Functions (from stimulus.py):
    point_pressure_torch: Binary disc stimulus
    gaussian_pressure_torch: Gaussian bump stimulus
    gabor_texture_torch: Gabor texture pattern
    edge_stimulus_torch: Sharp edge pattern
    edge_grating_stimulus_torch: Edge grating pattern
    create_temporal_profile_torch: Temporal amplitude trajectories

Classes:
    Stimulus: Fluent builder for composable stimuli (RECOMMENDED)
    StaticStimulus: Static spatial stimulus
    MovingStimulus: Stimulus with motion trajectory
    CompositeStimulus: Combination of multiple stimuli
    BaseStimulus: Abstract base class
    StimulusGenerator: High-level stimulus synthesis interface
    GaussianStimulus: Modular Gaussian stimulus generator
    GaborTexture: Modular Gabor texture generator
    EdgeGrating: Modular edge grating generator
    MovingStimulus: Moving stimulus with trajectory (legacy)
"""

# Base class
from sensoryforge.stimuli.base import BaseStimulus

# New composable builder API (Phase 1.4)
from sensoryforge.stimuli.builder import (
    Stimulus,
    StaticStimulus,
    MovingStimulus,
    CompositeStimulus,
    with_motion,
)

# Legacy stimulus functions (backward compatibility)
from sensoryforge.stimuli.stimulus import (
    point_pressure_torch,
    gaussian_pressure_torch,
    gabor_texture_torch,
    edge_stimulus_torch,
    edge_grating_stimulus_torch,
    create_temporal_profile_torch,
    StimulusGenerator,
)

# New modular stimulus APIs
from sensoryforge.stimuli.gaussian import (
    gaussian_stimulus,
    multi_gaussian_stimulus,
    batched_gaussian_stimulus,
    GaussianStimulus,
)

from sensoryforge.stimuli.texture import (
    gabor_texture,
    edge_grating,
    noise_texture,
    GaborTexture,
    EdgeGrating,
)

from sensoryforge.stimuli.moving import (
    linear_motion,
    circular_motion,
    custom_path_motion,
    velocity_profile,
    tap_sequence,
    slide_trajectory,
)

__all__ = [
    # Base class
    "BaseStimulus",
    # Composable builder API (Phase 1.4 - RECOMMENDED)
    "Stimulus",
    "StaticStimulus",
    "MovingStimulus",
    "CompositeStimulus",
    "with_motion",
    # Legacy API (backward compatibility)
    "point_pressure_torch",
    "gaussian_pressure_torch",
    "gabor_texture_torch",
    "edge_stimulus_torch",
    "edge_grating_stimulus_torch",
    "create_temporal_profile_torch",
    "StimulusGenerator",
    # Gaussian stimuli
    "gaussian_stimulus",
    "multi_gaussian_stimulus",
    "batched_gaussian_stimulus",
    "GaussianStimulus",
    # Texture stimuli
    "gabor_texture",
    "edge_grating",
    "noise_texture",
    "GaborTexture",
    "EdgeGrating",
    # Moving stimuli
    "linear_motion",
    "circular_motion",
    "custom_path_motion",
    "velocity_profile",
    "tap_sequence",
    "slide_trajectory",
]
