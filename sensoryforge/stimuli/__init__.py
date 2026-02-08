"""Stimulus generation and pattern synthesis module.

This module provides utilities for generating spatial pressure patterns,
temporal trajectories, and motion profiles for sensory encoding experiments.

Functions:
    point_pressure_torch: Binary disc stimulus
    gaussian_pressure_torch: Gaussian bump stimulus
    gabor_texture_torch: Gabor texture pattern
    edge_stimulus_torch: Sharp edge pattern
    edge_grating_stimulus_torch: Edge grating pattern
    create_temporal_profile_torch: Temporal amplitude trajectories

Classes:
    StimulusGenerator: High-level stimulus synthesis interface

Example:
    >>> from sensoryforge.stimuli import gaussian_pressure_torch, StimulusGenerator
    >>> stimulus = gaussian_pressure_torch(xx, yy, center_x=0, center_y=0, amplitude=1.0)
"""

from sensoryforge.stimuli.stimulus import (
    point_pressure_torch,
    gaussian_pressure_torch,
    gabor_texture_torch,
    edge_stimulus_torch,
    edge_grating_stimulus_torch,
    create_temporal_profile_torch,
    StimulusGenerator,
)

__all__ = [
    "point_pressure_torch",
    "gaussian_pressure_torch",
    "gabor_texture_torch",
    "edge_stimulus_torch",
    "edge_grating_stimulus_torch",
    "create_temporal_profile_torch",
    "StimulusGenerator",
]
