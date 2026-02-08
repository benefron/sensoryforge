"""Moving stimulus generation module.

This module provides functions and classes for generating moving tactile stimuli
with temporal trajectories. Supports various motion patterns including linear
motion, circular motion, and custom paths.

Example:
    >>> import torch
    >>> from sensoryforge.stimuli.moving import linear_motion, circular_motion
    >>> # Create a stimulus that moves horizontally
    >>> positions = linear_motion(start=(0.0, 0.0), end=(2.0, 0.0), num_steps=100)
    >>> positions.shape
    torch.Size([100, 2])
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import Callable


def linear_motion(
    start: tuple[float, float],
    end: tuple[float, float],
    num_steps: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate a linear motion trajectory between two points.
    
    Args:
        start: Starting (x, y) position. Units: mm.
        end: Ending (x, y) position. Units: mm.
        num_steps: Number of time steps in the trajectory.
        device: Device to create the trajectory on (cpu, cuda, mps).
    
    Returns:
        Trajectory positions, shape [num_steps, 2]. Units: mm.
    
    Raises:
        ValueError: If num_steps is less than 2.
    
    Example:
        >>> positions = linear_motion((0.0, 0.0), (1.0, 1.0), 10)
        >>> positions.shape
        torch.Size([10, 2])
        >>> # First and last positions match start and end
        >>> torch.allclose(positions[0], torch.tensor([0.0, 0.0]))
        True
        >>> torch.allclose(positions[-1], torch.tensor([1.0, 1.0]))
        True
    """
    if num_steps < 2:
        raise ValueError(f"num_steps must be at least 2, got {num_steps}")
    
    # Create linearly spaced positions from start to end
    t = torch.linspace(0, 1, num_steps, device=device).unsqueeze(1)  # [num_steps, 1]
    
    start_tensor = torch.tensor(start, device=device, dtype=torch.float32)
    end_tensor = torch.tensor(end, device=device, dtype=torch.float32)
    
    # Linear interpolation: pos = start + t * (end - start)
    positions = start_tensor + t * (end_tensor - start_tensor)
    
    return positions


def circular_motion(
    center: tuple[float, float],
    radius: float,
    num_steps: int,
    start_angle: float = 0.0,
    end_angle: float = 2 * math.pi,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate a circular motion trajectory around a center point.
    
    Args:
        center: Center (x, y) of the circular path. Units: mm.
        radius: Radius of the circular path. Units: mm.
        num_steps: Number of time steps in the trajectory.
        start_angle: Starting angle in radians (0 is right, π/2 is up).
        end_angle: Ending angle in radians. Default is full circle (2π).
        device: Device to create the trajectory on (cpu, cuda, mps).
    
    Returns:
        Trajectory positions, shape [num_steps, 2]. Units: mm.
    
    Raises:
        ValueError: If num_steps is less than 2.
        ValueError: If radius is negative.
    
    Example:
        >>> # Quarter circle from right to top
        >>> positions = circular_motion((0.0, 0.0), 1.0, 50, 0.0, math.pi/2)
        >>> positions.shape
        torch.Size([50, 2])
        >>> # First position is at (radius, 0)
        >>> torch.allclose(positions[0], torch.tensor([1.0, 0.0]), atol=1e-6)
        True
    """
    if num_steps < 2:
        raise ValueError(f"num_steps must be at least 2, got {num_steps}")
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    
    # Generate angles
    angles = torch.linspace(start_angle, end_angle, num_steps, device=device)
    
    # Convert to Cartesian coordinates
    x = center[0] + radius * torch.cos(angles)
    y = center[1] + radius * torch.sin(angles)
    
    positions = torch.stack([x, y], dim=1)  # [num_steps, 2]
    
    return positions


def custom_path_motion(
    waypoints: list[tuple[float, float]],
    num_steps: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate motion along a custom path defined by waypoints.
    
    Interpolates smoothly between waypoints to create a trajectory with
    the specified number of steps. Uses linear interpolation between waypoints.
    
    Args:
        waypoints: List of (x, y) waypoints defining the path. Units: mm.
        num_steps: Number of time steps in the trajectory.
        device: Device to create the trajectory on (cpu, cuda, mps).
    
    Returns:
        Trajectory positions, shape [num_steps, 2]. Units: mm.
    
    Raises:
        ValueError: If fewer than 2 waypoints provided.
        ValueError: If num_steps is less than 2.
    
    Example:
        >>> waypoints = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        >>> positions = custom_path_motion(waypoints, 100)
        >>> positions.shape
        torch.Size([100, 2])
    """
    if len(waypoints) < 2:
        raise ValueError(f"At least 2 waypoints required, got {len(waypoints)}")
    if num_steps < 2:
        raise ValueError(f"num_steps must be at least 2, got {num_steps}")
    
    # Convert waypoints to tensor [num_waypoints, 2]
    waypoints_tensor = torch.tensor(waypoints, device=device, dtype=torch.float32)
    
    # Use linear interpolation to resample to desired number of steps
    # Transpose to [2, num_waypoints] for interpolate
    waypoints_transposed = waypoints_tensor.t().unsqueeze(0)  # [1, 2, num_waypoints]
    
    # Interpolate to num_steps
    interpolated = F.interpolate(
        waypoints_transposed,
        size=num_steps,
        mode="linear",
        align_corners=True,
    )
    
    # Transpose back to [num_steps, 2]
    positions = interpolated.squeeze(0).t()
    
    return positions


def velocity_profile(
    num_steps: int,
    profile_type: str = "constant",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate a velocity profile for motion modulation.
    
    Creates temporal velocity profiles that can be used to modulate
    motion speed or stimulus amplitude over time.
    
    Args:
        num_steps: Number of time steps.
        profile_type: Type of velocity profile. Options:
            - "constant": Constant velocity (value 1.0)
            - "ramp_up": Linearly increasing from 0 to 1
            - "ramp_down": Linearly decreasing from 1 to 0
            - "trapezoidal": Ramp up, plateau, ramp down
            - "sinusoidal": Sinusoidal modulation
        device: Device to create the profile on (cpu, cuda, mps).
    
    Returns:
        Velocity profile, shape [num_steps]. Values normalized to [0, 1].
    
    Raises:
        ValueError: If profile_type is not recognized.
        ValueError: If num_steps is less than 1.
    
    Example:
        >>> profile = velocity_profile(100, profile_type="trapezoidal")
        >>> profile.shape
        torch.Size([100])
        >>> # Profile values are in [0, 1]
        >>> (profile >= 0).all() and (profile <= 1).all()
        True
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be at least 1, got {num_steps}")
    
    if profile_type == "constant":
        return torch.ones(num_steps, device=device)
    
    elif profile_type == "ramp_up":
        return torch.linspace(0, 1, num_steps, device=device)
    
    elif profile_type == "ramp_down":
        return torch.linspace(1, 0, num_steps, device=device)
    
    elif profile_type == "trapezoidal":
        # Divide into three segments: ramp up (25%), plateau (50%), ramp down (25%)
        ramp_steps = max(1, num_steps // 4)
        plateau_steps = num_steps - 2 * ramp_steps
        
        ramp_up = torch.linspace(0, 1, ramp_steps, device=device)
        plateau = torch.ones(plateau_steps, device=device)
        ramp_down = torch.linspace(1, 0, ramp_steps, device=device)
        
        # Concatenate and trim/pad to exact num_steps
        profile = torch.cat([ramp_up, plateau, ramp_down])
        if len(profile) > num_steps:
            profile = profile[:num_steps]
        elif len(profile) < num_steps:
            # Pad with ones
            profile = torch.cat([profile, torch.ones(num_steps - len(profile), device=device)])
        
        return profile
    
    elif profile_type == "sinusoidal":
        t = torch.linspace(0, 2 * math.pi, num_steps, device=device)
        # Map from [-1, 1] to [0, 1]
        return 0.5 * (1 + torch.sin(t - math.pi / 2))
    
    else:
        raise ValueError(
            f"Unknown profile_type: {profile_type}. "
            f"Must be one of: constant, ramp_up, ramp_down, trapezoidal, sinusoidal"
        )


class MovingStimulus(torch.nn.Module):
    """PyTorch module for generating moving stimuli with temporal trajectories.
    
    This module combines a spatial stimulus generator with a motion trajectory
    to create time-varying stimuli that move across the tactile field.
    
    Attributes:
        trajectory: Tensor of (x, y) positions over time, shape [num_steps, 2].
        stimulus_generator: Callable that generates spatial stimulus given (xx, yy, cx, cy).
        amplitude_profile: Optional temporal amplitude modulation, shape [num_steps].
    
    Example:
        >>> import torch
        >>> from sensoryforge.stimuli.gaussian import gaussian_stimulus
        >>> # Create trajectory
        >>> trajectory = linear_motion((0.0, 0.0), (1.0, 0.0), 50)
        >>> # Create moving Gaussian
        >>> def gauss_gen(xx, yy, cx, cy):
        ...     return gaussian_stimulus(xx, yy, cx, cy, amplitude=1.0, sigma=0.2)
        >>> moving = MovingStimulus(trajectory, gauss_gen)
        >>> # Generate on grid
        >>> xx, yy = torch.meshgrid(torch.linspace(-2, 2, 32), torch.linspace(-2, 2, 32), indexing='ij')
        >>> temporal_stimulus = moving(xx, yy)
        >>> temporal_stimulus.shape
        torch.Size([50, 32, 32])
    """
    
    def __init__(
        self,
        trajectory: torch.Tensor,
        stimulus_generator: Callable[[torch.Tensor, torch.Tensor, float, float], torch.Tensor],
        amplitude_profile: torch.Tensor | None = None,
    ) -> None:
        """Initialize moving stimulus with trajectory and generator.
        
        Args:
            trajectory: Tensor of (x, y) positions over time, shape [num_steps, 2]. Units: mm.
            stimulus_generator: Function that takes (xx, yy, center_x, center_y) and
                returns a spatial stimulus pattern.
            amplitude_profile: Optional temporal amplitude modulation, shape [num_steps].
                If None, uses constant amplitude of 1.0.
        
        Raises:
            ValueError: If trajectory doesn't have shape [num_steps, 2].
            ValueError: If amplitude_profile length doesn't match trajectory.
        """
        super().__init__()
        
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError(
                f"trajectory must have shape [num_steps, 2], got {trajectory.shape}"
            )
        
        num_steps = trajectory.shape[0]
        
        if amplitude_profile is None:
            amplitude_profile = torch.ones(num_steps, device=trajectory.device)
        
        if amplitude_profile.shape[0] != num_steps:
            raise ValueError(
                f"amplitude_profile length ({amplitude_profile.shape[0]}) must match "
                f"trajectory length ({num_steps})"
            )
        
        self.register_buffer("trajectory", trajectory)
        self.register_buffer("amplitude_profile", amplitude_profile)
        self.stimulus_generator = stimulus_generator
    
    def forward(self, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """Generate temporal stimulus sequence on the given coordinate grid.
        
        Args:
            xx: Tensor of x-coordinates, shape [H, W]. Units: mm.
            yy: Tensor of y-coordinates, shape [H, W]. Units: mm.
        
        Returns:
            Temporal stimulus sequence, shape [num_steps, H, W].
        
        Example:
            >>> trajectory = circular_motion((0.0, 0.0), 0.5, 20)
            >>> def stim_gen(xx, yy, cx, cy):
            ...     from sensoryforge.stimuli.gaussian import gaussian_stimulus
            ...     return gaussian_stimulus(xx, yy, cx, cy, amplitude=1.0, sigma=0.3)
            >>> moving = MovingStimulus(trajectory, stim_gen)
            >>> xx = torch.linspace(-1, 1, 16).unsqueeze(0).expand(16, -1)
            >>> yy = torch.linspace(-1, 1, 16).unsqueeze(1).expand(-1, 16)
            >>> output = moving(xx, yy)
            >>> output.shape
            torch.Size([20, 16, 16])
        """
        num_steps = self.trajectory.shape[0]
        height, width = xx.shape
        
        # Pre-allocate output tensor
        temporal_stimulus = torch.zeros(
            num_steps, height, width,
            device=xx.device,
            dtype=xx.dtype,
        )
        
        # Generate stimulus at each time step
        for t in range(num_steps):
            center_x = float(self.trajectory[t, 0])
            center_y = float(self.trajectory[t, 1])
            
            # Generate spatial stimulus at current position
            spatial_stim = self.stimulus_generator(xx, yy, center_x, center_y)
            
            # Apply amplitude modulation
            temporal_stimulus[t] = spatial_stim * self.amplitude_profile[t]
        
        return temporal_stimulus


def tap_sequence(
    position: tuple[float, float],
    num_taps: int,
    tap_duration: int,
    interval_duration: int,
    stimulus_generator: Callable[[torch.Tensor, torch.Tensor, float, float], torch.Tensor],
    xx: torch.Tensor,
    yy: torch.Tensor,
) -> torch.Tensor:
    """Generate a sequence of taps at a fixed location.
    
    Creates a temporal pattern of repeated tap stimuli with on/off intervals,
    useful for simulating rhythmic tactile input.
    
    Args:
        position: (x, y) position of taps. Units: mm.
        num_taps: Number of taps in the sequence.
        tap_duration: Duration of each tap in time steps.
        interval_duration: Duration between taps in time steps.
        stimulus_generator: Function to generate spatial stimulus.
        xx: X-coordinates grid, shape [H, W]. Units: mm.
        yy: Y-coordinates grid, shape [H, W]. Units: mm.
    
    Returns:
        Temporal tap sequence, shape [total_steps, H, W].
    
    Raises:
        ValueError: If tap_duration or interval_duration is less than 1.
        ValueError: If num_taps is less than 1.
    
    Example:
        >>> import torch
        >>> from sensoryforge.stimuli.gaussian import gaussian_stimulus
        >>> xx, yy = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing='ij')
        >>> def gen(xx, yy, cx, cy):
        ...     return gaussian_stimulus(xx, yy, cx, cy, 1.0, 0.2)
        >>> taps = tap_sequence((0.0, 0.0), 3, 10, 5, gen, xx, yy)
        >>> # Total duration: 3 taps * (10 on + 5 off) - 5 (no interval after last)
        >>> taps.shape[0] == 3 * 10 + 2 * 5
        True
    """
    if tap_duration < 1:
        raise ValueError(f"tap_duration must be at least 1, got {tap_duration}")
    if interval_duration < 1:
        raise ValueError(f"interval_duration must be at least 1, got {interval_duration}")
    if num_taps < 1:
        raise ValueError(f"num_taps must be at least 1, got {num_taps}")
    
    # Calculate total time steps
    total_steps = num_taps * tap_duration + (num_taps - 1) * interval_duration
    
    height, width = xx.shape
    sequence = torch.zeros(total_steps, height, width, device=xx.device, dtype=xx.dtype)
    
    # Generate tap stimulus once
    tap_stimulus = stimulus_generator(xx, yy, position[0], position[1])
    
    # Place taps at appropriate intervals
    t = 0
    for i in range(num_taps):
        sequence[t : t + tap_duration] = tap_stimulus
        t += tap_duration + interval_duration
    
    return sequence


def slide_trajectory(
    start: tuple[float, float],
    end: tuple[float, float],
    num_steps: int,
    velocity_type: str = "constant",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate a sliding motion trajectory with velocity modulation.
    
    Creates a linear trajectory with controllable velocity profile,
    useful for simulating sliding or dragging tactile stimulation.
    
    Args:
        start: Starting (x, y) position. Units: mm.
        end: Ending (x, y) position. Units: mm.
        num_steps: Number of time steps.
        velocity_type: Type of velocity profile (see velocity_profile function).
        device: Device to create the trajectory on.
    
    Returns:
        Trajectory with velocity modulation, shape [num_steps, 2]. Units: mm.
    
    Example:
        >>> trajectory = slide_trajectory((0.0, 0.0), (2.0, 0.0), 50, "trapezoidal")
        >>> trajectory.shape
        torch.Size([50, 2])
        >>> # Starts at origin
        >>> torch.allclose(trajectory[0], torch.tensor([0.0, 0.0]))
        True
    """
    # Generate base linear trajectory
    base_trajectory = linear_motion(start, end, num_steps, device=device)
    
    # For constant velocity, just return the linear trajectory
    if velocity_type == "constant":
        return base_trajectory
    
    # Generate velocity profile
    vel_profile = velocity_profile(num_steps, velocity_type, device=device)
    
    # Apply velocity modulation through non-uniform time sampling
    # Compute cumulative distance traveled
    cum_vel = torch.cumsum(vel_profile, dim=0)
    cum_vel = cum_vel / cum_vel[-1]  # Normalize to [0, 1]
    
    # Create uniform time samples
    t_uniform = torch.linspace(0, 1, num_steps, device=device)
    
    # Manual linear interpolation since torch.interp may not be available
    # For each position in cum_vel, find corresponding value in base_trajectory
    modulated_trajectory = torch.zeros_like(base_trajectory)
    
    for i in range(num_steps):
        # Find where cum_vel[i] falls in t_uniform
        t_val = cum_vel[i]
        
        # Find the two indices in t_uniform that bracket t_val
        if t_val <= 0:
            modulated_trajectory[i] = base_trajectory[0]
        elif t_val >= 1:
            modulated_trajectory[i] = base_trajectory[-1]
        else:
            # Find index where t_uniform >= t_val
            idx = torch.searchsorted(t_uniform, t_val)
            if idx == 0:
                modulated_trajectory[i] = base_trajectory[0]
            elif idx >= num_steps:
                modulated_trajectory[i] = base_trajectory[-1]
            else:
                # Linear interpolation between idx-1 and idx
                t0, t1 = t_uniform[idx-1], t_uniform[idx]
                alpha = (t_val - t0) / (t1 - t0) if t1 > t0 else 0.0
                modulated_trajectory[i] = (
                    (1 - alpha) * base_trajectory[idx-1] + alpha * base_trajectory[idx]
                )
    
    return modulated_trajectory


__all__ = [
    "linear_motion",
    "circular_motion",
    "custom_path_motion",
    "velocity_profile",
    "MovingStimulus",
    "tap_sequence",
    "slide_trajectory",
]
