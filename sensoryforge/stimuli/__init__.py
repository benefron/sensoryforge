"""Stimulus generation and pattern synthesis module.

This module provides utilities for generating spatial pressure patterns,
temporal trajectories, and motion profiles for sensory encoding experiments.

Modules:
    gaussian: Gaussian bump stimuli with superposition support
    texture: Gabor textures, edge gratings, and noise patterns
    moving: Moving contacts and temporal trajectories

Legacy Functions (from stimulus.py):
    point_pressure_torch: Binary disc stimulus
    gaussian_pressure_torch: Gaussian bump stimulus
    gabor_texture_torch: Gabor texture pattern
    edge_stimulus_torch: Sharp edge pattern
    edge_grating_stimulus_torch: Edge grating pattern
    create_temporal_profile_torch: Temporal amplitude trajectories

Classes:
    StimulusGenerator: High-level stimulus synthesis interface
    GaussianStimulus: Modular Gaussian stimulus generator
    GaborTexture: Modular Gabor texture generator
    EdgeGrating: Modular edge grating generator
    MovingStimulus: Moving stimulus with trajectory

Example:
    >>> from sensoryforge.stimuli import gaussian_pressure_torch, StimulusGenerator
    >>> stimulus = gaussian_pressure_torch(xx, yy, center_x=0, center_y=0, amplitude=1.0)
    
    >>> # Using new modular API
    >>> from sensoryforge.stimuli.gaussian import gaussian_stimulus, GaussianStimulus
    >>> stim = gaussian_stimulus(xx, yy, 0.0, 0.0, amplitude=2.0, sigma=0.5)
"""

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
    MovingStimulus,
    tap_sequence,
    slide_trajectory,
)

__all__ = [
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
    "MovingStimulus",
    "tap_sequence",
    "slide_trajectory",
]
