"""Unit tests for extended stimuli modules (texture, moving)."""

from __future__ import annotations

import math
import pytest
import torch

from sensoryforge.stimuli.texture import gabor_texture, edge_grating, noise_texture
from sensoryforge.stimuli.moving import (
    linear_motion,
    circular_motion,
    custom_path_motion,
    velocity_profile,
    MovingStimulus,
    tap_sequence,
    slide_trajectory,
)
from sensoryforge.stimuli.gaussian import gaussian_stimulus


def _grid(size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-1.0, 1.0, size)
    return torch.meshgrid(axis, axis, indexing="ij")


def test_gabor_texture_shape() -> None:
    xx, yy = _grid(32)
    output = gabor_texture(xx, yy, wavelength=0.4, orientation=0.3)
    assert output.shape == xx.shape
    assert torch.isfinite(output).all()


def test_gabor_texture_invalid_params() -> None:
    xx, yy = _grid(16)
    with pytest.raises(ValueError, match="wavelength must be positive"):
        gabor_texture(xx, yy, wavelength=0.0)
    with pytest.raises(ValueError, match="sigma must be positive"):
        gabor_texture(xx, yy, sigma=-0.1)


def test_edge_grating_shape() -> None:
    xx, yy = _grid(24)
    output = edge_grating(xx, yy, orientation=0.5, spacing=0.4, count=6)
    assert output.shape == xx.shape
    assert torch.isfinite(output).all()


def test_edge_grating_invalid_params() -> None:
    xx, yy = _grid(16)
    with pytest.raises(ValueError, match="spacing must be positive"):
        edge_grating(xx, yy, spacing=0.0)
    with pytest.raises(ValueError, match="edge_width must be positive"):
        edge_grating(xx, yy, edge_width=-0.1)
    with pytest.raises(ValueError, match="count must be at least 1"):
        edge_grating(xx, yy, count=0)


def test_noise_texture_deterministic_seed() -> None:
    texture_a = noise_texture(32, 32, scale=0.5, kernel_size=5, seed=42)
    texture_b = noise_texture(32, 32, scale=0.5, kernel_size=5, seed=42)
    assert torch.allclose(texture_a, texture_b)


def test_linear_motion_endpoints() -> None:
    trajectory = linear_motion((0.0, 0.0), (1.0, 1.0), 10)
    assert torch.allclose(trajectory[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(trajectory[-1], torch.tensor([1.0, 1.0]))


def test_linear_motion_invalid_steps() -> None:
    with pytest.raises(ValueError, match="num_steps must be at least 2"):
        linear_motion((0.0, 0.0), (1.0, 0.0), 1)


def test_circular_motion_radius() -> None:
    center = (0.0, 0.0)
    radius = 1.2
    trajectory = circular_motion(center, radius, 25, 0.0, math.pi / 2)
    distances = torch.norm(trajectory, dim=1)
    assert torch.allclose(distances, torch.full_like(distances, radius), atol=1e-4)


def test_circular_motion_invalid_params() -> None:
    with pytest.raises(ValueError, match="num_steps must be at least 2"):
        circular_motion((0.0, 0.0), 1.0, 1)
    with pytest.raises(ValueError, match="radius must be non-negative"):
        circular_motion((0.0, 0.0), -1.0, 5)


def test_custom_path_motion_shape() -> None:
    trajectory = custom_path_motion([(0.0, 0.0), (1.0, 0.0)], 8)
    assert trajectory.shape == (8, 2)


def test_custom_path_motion_invalid_params() -> None:
    with pytest.raises(ValueError, match="At least 2 waypoints"):
        custom_path_motion([(0.0, 0.0)], 10)
    with pytest.raises(ValueError, match="num_steps must be at least 2"):
        custom_path_motion([(0.0, 0.0), (1.0, 0.0)], 1)


def test_velocity_profile_range() -> None:
    for profile_type in ["constant", "ramp_up", "ramp_down", "trapezoidal", "sinusoidal"]:
        profile = velocity_profile(50, profile_type=profile_type)
        assert profile.min() >= 0
        assert profile.max() <= 1


def test_velocity_profile_invalid_type() -> None:
    with pytest.raises(ValueError, match="Unknown profile_type"):
        velocity_profile(10, profile_type="invalid")


def test_moving_stimulus_output_shape() -> None:
    trajectory = linear_motion((0.0, 0.0), (0.5, 0.5), 12)

    def stim_gen(xx: torch.Tensor, yy: torch.Tensor, cx: float, cy: float) -> torch.Tensor:
        return gaussian_stimulus(xx, yy, cx, cy, amplitude=1.0, sigma=0.2)

    moving = MovingStimulus(trajectory, stim_gen)
    xx, yy = _grid(20)
    output = moving(xx, yy)
    assert output.shape == (trajectory.shape[0], 20, 20)


def test_tap_sequence_length() -> None:
    xx, yy = _grid(16)

    def stim_gen(xx: torch.Tensor, yy: torch.Tensor, cx: float, cy: float) -> torch.Tensor:
        return gaussian_stimulus(xx, yy, cx, cy, amplitude=1.0, sigma=0.2)

    taps = tap_sequence((0.0, 0.0), num_taps=3, tap_duration=4, interval_duration=2, stimulus_generator=stim_gen, xx=xx, yy=yy)
    assert taps.shape[0] == 3 * 4 + 2 * 2


def test_slide_trajectory_endpoints() -> None:
    trajectory = slide_trajectory((0.0, 0.0), (1.0, 0.0), 20, velocity_type="trapezoidal")
    assert torch.allclose(trajectory[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(trajectory[-1], torch.tensor([1.0, 0.0]))


def test_slide_trajectory_invalid_velocity() -> None:
    with pytest.raises(ValueError, match="Unknown profile_type"):
        slide_trajectory((0.0, 0.0), (1.0, 0.0), 10, velocity_type="invalid")
