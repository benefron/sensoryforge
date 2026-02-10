"""Composable stimulus builder API.

This module provides a fluent builder interface for creating stimuli with optional
movement and composition. Stimuli can be created with factory methods, augmented
with motion patterns, and combined together.

Architecture:
    BaseStimulus (ABC from base.py)
    ├── StaticStimulus: Wraps functional stimulus generators
    ├── MovingStimulus: Adds temporal motion to static stimuli
    └── CompositeStimulus: Combines multiple stimuli

Example:
    >>> # Static stimulus
    >>> stim1 = Stimulus.gaussian(amplitude=1.0, sigma=0.3, center=(0.5, 0.5))
    >>> 
    >>> # Static stimulus with linear motion
    >>> stim2 = Stimulus.gaussian(amplitude=2.0, sigma=0.2).with_motion(
    ...     'linear', start=(0.0, 0.0), end=(1.0, 1.0), num_steps=100
    ... )
    >>> 
    >>> # Gabor texture with circular motion
    >>> stim3 = Stimulus.gabor(
    ...     wavelength=0.5, orientation=0.0
    ... ).with_motion(
    ...     'circular', center=(0.0, 0.0), radius=0.5, num_steps=200
    ... )
    >>> 
    >>> # Composite stimulus
    >>> combined = Stimulus.compose([stim1, stim2, stim3], mode='add')
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Literal, Tuple
import torch
import torch.nn as nn

from sensoryforge.stimuli.base import BaseStimulus
from sensoryforge.stimuli import gaussian as gauss_module
from sensoryforge.stimuli import texture as texture_module
from sensoryforge.stimuli import moving as motion_module
from sensoryforge.stimuli.stimulus import (
    point_pressure_torch,
    edge_stimulus_torch,
)


CompositionMode = Literal["add", "max", "mean", "multiply"]
MotionType = Literal["linear", "circular", "stationary"]


class StaticStimulus(BaseStimulus):
    """Static spatial stimulus without temporal dynamics.
    
    This class wraps the functional stimulus generation APIs and provides
    a BaseStimulus-compliant interface.
    
    Attributes:
        stim_type: Type of stimulus ('gaussian', 'point', 'edge', 'gabor', etc.)
        params: Dictionary of stimulus parameters
        device: Target device for computation
    """
    
    def __init__(
        self,
        stim_type: str,
        params: Dict[str, Any],
        device: torch.device | str = 'cpu',
    ):
        """Initialize static stimulus.
        
        Args:
            stim_type: Stimulus type identifier
            params: Stimulus-specific parameters (amplitude, sigma, center, etc.)
            device: Device for tensor creation
        """
        super().__init__()
        self.stim_type = stim_type
        self.params = params
        self.device = torch.device(device) if isinstance(device, str) else device
    
    def forward(self, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """Generate static stimulus on coordinate grid.
        
        Args:
            xx: X-coordinates, shape [H, W] or [batch, H, W]. Units: mm.
            yy: Y-coordinates, shape [H, W] or [batch, H, W]. Units: mm.
        
        Returns:
            Stimulus tensor with same shape as xx and yy.
        
        Raises:
            ValueError: If stim_type is unknown.
        """
        # Ensure coordinates are on correct device
        xx = xx.to(self.device)
        yy = yy.to(self.device)
        
        if self.stim_type == 'gaussian':
            return gauss_module.gaussian_stimulus(
                xx, yy,
                center_x=self.params.get('center_x', 0.0),
                center_y=self.params.get('center_y', 0.0),
                amplitude=self.params.get('amplitude', 1.0),
                sigma=self.params.get('sigma', 0.2),
                device=self.device,
            )
        
        elif self.stim_type == 'point':
            return point_pressure_torch(
                xx, yy,
                center_x=self.params.get('center_x', 0.0),
                center_y=self.params.get('center_y', 0.0),
                amplitude=self.params.get('amplitude', 1.0),
                diameter_mm=self.params.get('diameter_mm', 0.6),
            )
        
        elif self.stim_type == 'edge':
            center_x = self.params.get('center_x', 0.0)
            center_y = self.params.get('center_y', 0.0)
            return edge_stimulus_torch(
                xx - center_x,
                yy - center_y,
                theta=self.params.get('orientation', 0.0),
                w=self.params.get('width', 0.05),
                amplitude=self.params.get('amplitude', 1.0),
            )
        
        elif self.stim_type == 'gabor':
            return texture_module.gabor_texture(
                xx, yy,
                center_x=self.params.get('center_x', 0.0),
                center_y=self.params.get('center_y', 0.0),
                amplitude=self.params.get('amplitude', 1.0),
                sigma=self.params.get('sigma', 0.3),
                wavelength=self.params.get('wavelength', 0.5),
                orientation=self.params.get('orientation', 0.0),
                phase=self.params.get('phase', 0.0),
                device=self.device,
            )
        
        elif self.stim_type == 'edge_grating':
            return texture_module.edge_grating(
                xx, yy,
                orientation=self.params.get('orientation', 0.0),
                spacing=self.params.get('spacing', 0.6),
                count=self.params.get('count', 5),
                edge_width=self.params.get('edge_width', 0.05),
                amplitude=self.params.get('amplitude', 1.0),
                device=self.device,
            )
        
        else:
            raise ValueError(f"Unknown stimulus type: {self.stim_type}")
    
    def reset_state(self):
        """Static stimuli have no state to reset."""
        pass
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'StaticStimulus':
        """Create StaticStimulus from configuration dict.
        
        Args:
            config: Configuration dictionary with 'type', 'params', 'device'
        
        Returns:
            StaticStimulus instance
        
        Example:
            >>> config = {
            ...     'type': 'gaussian',
            ...     'params': {'amplitude': 1.0, 'sigma': 0.3, 'center_x': 0.5},
            ...     'device': 'cpu'
            ... }
            >>> stim = StaticStimulus.from_config(config)
        """
        return cls(
            stim_type=config['type'],
            params=config.get('params', {}),
            device=config.get('device', 'cpu'),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to configuration dictionary."""
        return {
            'type': self.stim_type,
            'params': self.params.copy(),
            'device': str(self.device),
        }


class MovingStimulus(BaseStimulus):
    """Stimulus with temporal motion trajectory.
    
    Wraps a static stimulus and applies a motion pattern to it over time.
    
    Attributes:
        base_stimulus: Underlying static stimulus
        motion_type: Type of motion ('linear', 'circular', 'stationary')
        motion_params: Motion-specific parameters
        trajectory: Pre-computed trajectory positions [num_steps, 2]. Units: mm.
        current_step: Current time step index
    """
    
    def __init__(
        self,
        base_stimulus: BaseStimulus,
        motion_type: MotionType = 'stationary',
        motion_params: Dict[str, Any] = None,
    ):
        """Initialize moving stimulus.
        
        Args:
            base_stimulus: Static stimulus to apply motion to
            motion_type: Type of motion trajectory
            motion_params: Motion-specific parameters (start, end, center, radius, etc.)
        
        Raises:
            ValueError: If motion_type is unknown or required params missing
        """
        super().__init__()
        self.base_stimulus = base_stimulus
        self.motion_type = motion_type
        self.motion_params = motion_params or {}
        
        # Inherit device from base stimulus BEFORE generating trajectory
        self.device = getattr(base_stimulus, 'device', torch.device('cpu'))
        
        # Generate trajectory
        self.trajectory = self._generate_trajectory()
        self.current_step = 0
    
    def _generate_trajectory(self) -> torch.Tensor:
        """Generate motion trajectory based on motion_type.
        
        Returns:
            Trajectory positions, shape [num_steps, 2]. Units: mm.
        
        Raises:
            ValueError: If motion_type is unknown or required params missing.
        """
        if self.motion_type == 'stationary':
            # No motion, single position
            center = self.motion_params.get('center', (0.0, 0.0))
            num_steps = self.motion_params.get('num_steps', 1)
            positions = torch.tensor([list(center)] * num_steps, dtype=torch.float32)
            return positions
        
        elif self.motion_type == 'linear':
            start = self.motion_params.get('start')
            end = self.motion_params.get('end')
            num_steps = self.motion_params.get('num_steps', 100)
            
            if start is None or end is None:
                raise ValueError("Linear motion requires 'start' and 'end' parameters")
            
            return motion_module.linear_motion(
                start=start,
                end=end,
                num_steps=num_steps,
                device=self.device,
            )
        
        elif self.motion_type == 'circular':
            center = self.motion_params.get('center')
            radius = self.motion_params.get('radius')
            num_steps = self.motion_params.get('num_steps', 100)
            start_angle = self.motion_params.get('start_angle', 0.0)
            end_angle = self.motion_params.get('end_angle', 2 * math.pi)
            
            if center is None or radius is None:
                raise ValueError("Circular motion requires 'center' and 'radius' parameters")
            
            return motion_module.circular_motion(
                center=center,
                radius=radius,
                num_steps=num_steps,
                start_angle=start_angle,
                end_angle=end_angle,
                device=self.device,
            )
        
        else:
            raise ValueError(f"Unknown motion type: {self.motion_type}")
    
    def forward(self, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """Generate stimulus at current time step.
        
        Translates the base stimulus according to the current trajectory position.
        
        Args:
            xx: X-coordinates, shape [H, W]. Units: mm.
            yy: Y-coordinates, shape [H, W]. Units: mm.
        
        Returns:
            Stimulus at current position, shape [H, W].
        """
        # Get current position from trajectory
        if self.current_step >= len(self.trajectory):
            raise RuntimeError(
                f"Current step {self.current_step} exceeds trajectory length {len(self.trajectory)}"
            )
        
        position = self.trajectory[self.current_step]
        dx, dy = position[0].item(), position[1].item()
        
        # Translate coordinates by current position offset
        xx_shifted = xx - dx
        yy_shifted = yy - dy
        
        # Generate stimulus at shifted position
        return self.base_stimulus(xx_shifted, yy_shifted)
    
    def reset_state(self):
        """Reset to initial time step."""
        self.current_step = 0
        self.base_stimulus.reset_state()
    
    def step(self):
        """Advance to next time step."""
        if self.current_step < len(self.trajectory) - 1:
            self.current_step += 1
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MovingStimulus':
        """Create MovingStimulus from configuration dict.
        
        Args:
            config: Configuration with 'base_stimulus', 'motion_type', 'motion_params'
        
        Returns:
            MovingStimulus instance
        """
        # Recursively construct base stimulus
        base_config = config['base_stimulus']
        if base_config.get('class') == 'StaticStimulus':
            base_stimulus = StaticStimulus.from_config(base_config)
        else:
            raise ValueError(f"Unknown base stimulus class: {base_config.get('class')}")
        
        return cls(
            base_stimulus=base_stimulus,
            motion_type=config.get('motion_type', 'stationary'),
            motion_params=config.get('motion_params', {}),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to configuration dictionary."""
        return {
            'class': 'MovingStimulus',
            'base_stimulus': {
                'class': 'StaticStimulus',
                **self.base_stimulus.to_dict(),
            },
            'motion_type': self.motion_type,
            'motion_params': self.motion_params.copy(),
        }


class CompositeStimulus(BaseStimulus):
    """Composition of multiple stimuli with configurable combination mode.
    
    Combines multiple stimulus sources (static or moving) using element-wise
    operations (add, max, mean, multiply).
    
    Attributes:
        stimuli: List of constituent stimuli
        mode: Combination mode ('add', 'max', 'mean', 'multiply')
        device: Target device for computation
    """
    
    def __init__(
        self,
        stimuli: List[BaseStimulus],
        mode: CompositionMode = 'add',
        device: torch.device | str = 'cpu',
    ):
        """Initialize composite stimulus.
        
        Args:
            stimuli: List of stimuli to combine
            mode: Combination mode ('add', 'max', 'mean', 'multiply')
            device: Target device
        
        Raises:
            ValueError: If stimuli list is empty or mode is unknown
        """
        super().__init__()
        
        if not stimuli:
            raise ValueError("CompositeStimulus requires at least one stimulus")
        
        if mode not in ('add', 'max', 'mean', 'multiply'):
            raise ValueError(f"Unknown composition mode: {mode}")
        
        self.stimuli = nn.ModuleList(stimuli)
        self.mode = mode
        self.device = torch.device(device) if isinstance(device, str) else device
    
    def forward(self, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """Generate composite stimulus by combining constituents.
        
        Args:
            xx: X-coordinates, shape [H, W]. Units: mm.
            yy: Y-coordinates, shape [H, W]. Units: mm.
        
        Returns:
            Combined stimulus, shape [H, W].
        """
        # Generate all constituent stimuli
        outputs = [stim(xx, yy) for stim in self.stimuli]
        
        # Stack for combination
        stacked = torch.stack(outputs, dim=0)  # [num_stimuli, H, W]
        
        # Apply combination mode
        if self.mode == 'add':
            return torch.sum(stacked, dim=0)
        elif self.mode == 'max':
            return torch.max(stacked, dim=0).values
        elif self.mode == 'mean':
            return torch.mean(stacked, dim=0)
        elif self.mode == 'multiply':
            return torch.prod(stacked, dim=0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def reset_state(self):
        """Reset all constituent stimuli."""
        for stim in self.stimuli:
            stim.reset_state()
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CompositeStimulus':
        """Create CompositeStimulus from configuration dict.
        
        Args:
            config: Configuration with 'stimuli' list, 'mode', 'device'
        
        Returns:
            CompositeStimulus instance
        """
        # Recursively construct all constituent stimuli
        stimuli = []
        for stim_config in config['stimuli']:
            stim_class = stim_config.get('class', 'StaticStimulus')
            if stim_class == 'StaticStimulus':
                stimuli.append(StaticStimulus.from_config(stim_config))
            elif stim_class == 'MovingStimulus':
                stimuli.append(MovingStimulus.from_config(stim_config))
            else:
                raise ValueError(f"Unknown stimulus class: {stim_class}")
        
        return cls(
            stimuli=stimuli,
            mode=config.get('mode', 'add'),
            device=config.get('device', 'cpu'),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to configuration dictionary."""
        return {
            'class': 'CompositeStimulus',
            'stimuli': [stim.to_dict() for stim in self.stimuli],
            'mode': self.mode,
            'device': str(self.device),
        }


# ============================================================================
# Timeline and Pattern Stimuli (Phase 3)
# ============================================================================


class TimelineStimulus(BaseStimulus):
    """Composite stimulus with per-sub-stimulus onset, duration, and envelope.

    Each sub-stimulus is active only during its own ``[onset, onset + duration]``
    window within a global timeline defined by ``total_time_ms`` and ``dt_ms``.
    When evaluated at a specific time step the inactive stimuli contribute zero.

    Attributes:
        sub_stimuli: List of (stimulus, onset_ms, duration_ms, envelope) tuples.
        total_time_ms: Total timeline duration in ms.
        dt_ms: Time step in ms.
        composition_mode: How active stimuli are combined.
        current_step: Current discrete time index.
    """

    def __init__(
        self,
        sub_stimuli: List[Dict[str, Any]],
        total_time_ms: float = 500.0,
        dt_ms: float = 0.5,
        composition_mode: CompositionMode = 'add',
        device: torch.device | str = 'cpu',
    ):
        """Initialize timeline stimulus.

        Args:
            sub_stimuli: List of dicts, each with keys:
                - ``'stimulus'``: a :class:`BaseStimulus` instance.
                - ``'onset_ms'``: start time within the timeline (ms).
                - ``'duration_ms'``: how long it is active (ms).
                - ``'envelope'`` (optional): dict with ``ramp_up_ms``,
                  ``plateau_ms``, ``ramp_down_ms`` for a trapezoidal
                  amplitude envelope.
            total_time_ms: Global timeline duration in ms.
            dt_ms: Discrete time step in ms.
            composition_mode: Combination mode for overlapping stimuli.
            device: Target device.
        """
        super().__init__()
        if not sub_stimuli:
            raise ValueError("TimelineStimulus requires at least one sub-stimulus")

        self.total_time_ms = total_time_ms
        self.dt_ms = dt_ms
        self.composition_mode = composition_mode
        self.device = torch.device(device) if isinstance(device, str) else device
        self.current_step = 0
        self.num_steps = max(1, int(total_time_ms / dt_ms))

        # Store sub-stimuli with timing metadata
        self._entries: List[Dict[str, Any]] = []
        stim_modules = []
        for entry in sub_stimuli:
            stim = entry['stimulus']
            onset = entry.get('onset_ms', 0.0)
            duration = entry.get('duration_ms', total_time_ms)
            envelope = entry.get('envelope', None)
            self._entries.append({
                'onset_ms': onset,
                'duration_ms': duration,
                'envelope': envelope,
            })
            stim_modules.append(stim)
        self.stimuli = nn.ModuleList(stim_modules)

    def _envelope_gain(self, entry: Dict[str, Any], local_time_ms: float) -> float:
        """Compute amplitude gain from trapezoidal envelope.

        Args:
            entry: Sub-stimulus timing metadata.
            local_time_ms: Time elapsed since this stimulus's onset.

        Returns:
            Gain in [0, 1].
        """
        env = entry.get('envelope')
        if env is None:
            return 1.0
        ramp_up = env.get('ramp_up_ms', 0.0)
        ramp_down = env.get('ramp_down_ms', 0.0)
        duration = entry['duration_ms']
        if local_time_ms < ramp_up:
            return local_time_ms / max(ramp_up, 1e-9)
        elif local_time_ms > duration - ramp_down:
            remaining = duration - local_time_ms
            return max(remaining / max(ramp_down, 1e-9), 0.0)
        return 1.0

    def forward(self, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """Generate stimulus at the current time step.

        Only sub-stimuli whose ``[onset, onset + duration]`` window
        includes the current time are evaluated; others contribute zero.

        Args:
            xx: X-coordinates [H, W]. Units: mm.
            yy: Y-coordinates [H, W]. Units: mm.

        Returns:
            Stimulus tensor [H, W].
        """
        current_time_ms = self.current_step * self.dt_ms
        outputs = []

        for stim, entry in zip(self.stimuli, self._entries):
            onset = entry['onset_ms']
            duration = entry['duration_ms']
            if onset <= current_time_ms < onset + duration:
                local_t = current_time_ms - onset
                gain = self._envelope_gain(entry, local_t)

                # Advance moving stimuli if needed
                frame = stim(xx, yy)
                outputs.append(frame * gain)

        if not outputs:
            return torch.zeros_like(xx)

        stacked = torch.stack(outputs, dim=0)
        if self.composition_mode == 'add':
            return stacked.sum(dim=0)
        elif self.composition_mode == 'max':
            return stacked.max(dim=0).values
        elif self.composition_mode == 'mean':
            return stacked.mean(dim=0)
        elif self.composition_mode == 'multiply':
            return stacked.prod(dim=0)
        return stacked.sum(dim=0)

    def step(self):
        """Advance timeline by one dt step."""
        self.current_step += 1
        # Also step any moving sub-stimuli
        for stim in self.stimuli:
            if hasattr(stim, 'step'):
                stim.step()

    def reset_state(self):
        """Reset timeline to t=0."""
        self.current_step = 0
        for stim in self.stimuli:
            stim.reset_state()

    def generate_all_frames(
        self, xx: torch.Tensor, yy: torch.Tensor,
    ) -> torch.Tensor:
        """Generate the entire timeline as a 3D tensor.

        Args:
            xx: X-coordinates [H, W].
            yy: Y-coordinates [H, W].

        Returns:
            ``[num_steps, H, W]`` stimulus movie.
        """
        self.reset_state()
        frames = []
        for _ in range(self.num_steps):
            frames.append(self.forward(xx, yy))
            self.step()
        self.reset_state()
        return torch.stack(frames, dim=0)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TimelineStimulus':
        """Create from configuration dict."""
        sub_entries = []
        for sc in config.get('sub_stimuli', []):
            stim_cfg = sc['stimulus']
            stim_class = stim_cfg.get('class', 'StaticStimulus')
            if stim_class == 'StaticStimulus':
                stim = StaticStimulus.from_config(stim_cfg)
            elif stim_class == 'MovingStimulus':
                stim = MovingStimulus.from_config(stim_cfg)
            else:
                raise ValueError(f"Unknown stimulus class: {stim_class}")
            sub_entries.append({
                'stimulus': stim,
                'onset_ms': sc.get('onset_ms', 0.0),
                'duration_ms': sc.get('duration_ms', config.get('total_time_ms', 500.0)),
                'envelope': sc.get('envelope'),
            })
        return cls(
            sub_stimuli=sub_entries,
            total_time_ms=config.get('total_time_ms', 500.0),
            dt_ms=config.get('dt_ms', 0.5),
            composition_mode=config.get('composition_mode', 'add'),
            device=config.get('device', 'cpu'),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to configuration dictionary."""
        sub_list = []
        for stim, entry in zip(self.stimuli, self._entries):
            sub_list.append({
                'stimulus': stim.to_dict(),
                'onset_ms': entry['onset_ms'],
                'duration_ms': entry['duration_ms'],
                'envelope': entry.get('envelope'),
            })
        return {
            'class': 'TimelineStimulus',
            'total_time_ms': self.total_time_ms,
            'dt_ms': self.dt_ms,
            'composition_mode': self.composition_mode,
            'sub_stimuli': sub_list,
            'device': str(self.device),
        }


class RepeatedPatternStimulus(BaseStimulus):
    """Tile a base stimulus on an N×M grid with configurable spacing.

    This creates textures (dot arrays, braille patterns, etc.) by copying
    a single stimulus shape at regular spatial offsets.

    Attributes:
        base_stimulus: The prototype shape to replicate.
        copies_x: Number of copies along x.
        copies_y: Number of copies along y.
        spacing_x: Distance between copies along x (mm).
        spacing_y: Distance between copies along y (mm).
        center: Center of the pattern grid (mm).
    """

    def __init__(
        self,
        base_stimulus: BaseStimulus,
        copies_x: int = 3,
        copies_y: int = 2,
        spacing_x: float = 0.5,
        spacing_y: float = 0.5,
        center: Tuple[float, float] = (0.0, 0.0),
        device: torch.device | str = 'cpu',
    ):
        """Initialize repeated pattern.

        Args:
            base_stimulus: Prototype stimulus to copy.
            copies_x: Number of repetitions along x-axis.
            copies_y: Number of repetitions along y-axis.
            spacing_x: Spacing between copies in x (mm).
            spacing_y: Spacing between copies in y (mm).
            center: Center of the overall pattern (mm).
            device: Target device.
        """
        super().__init__()
        self.base_stimulus = base_stimulus
        self.copies_x = copies_x
        self.copies_y = copies_y
        self.spacing_x = spacing_x
        self.spacing_y = spacing_y
        self.center = center
        self.device = torch.device(device) if isinstance(device, str) else device

        # Pre-compute offsets relative to pattern center
        offsets = []
        for ix in range(copies_x):
            for iy in range(copies_y):
                ox = (ix - (copies_x - 1) / 2.0) * spacing_x + center[0]
                oy = (iy - (copies_y - 1) / 2.0) * spacing_y + center[1]
                offsets.append((ox, oy))
        self._offsets = offsets

    def forward(self, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """Generate repeated pattern by summing shifted copies.

        Args:
            xx: X-coordinates [H, W]. Units: mm.
            yy: Y-coordinates [H, W]. Units: mm.

        Returns:
            Combined stimulus [H, W].
        """
        result = torch.zeros_like(xx)
        for ox, oy in self._offsets:
            result = result + self.base_stimulus(xx - ox, yy - oy)
        return result

    def reset_state(self):
        """Reset base stimulus state."""
        self.base_stimulus.reset_state()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RepeatedPatternStimulus':
        """Create from configuration dict."""
        base_cfg = config['base_stimulus']
        stim_class = base_cfg.get('class', 'StaticStimulus')
        if stim_class == 'StaticStimulus':
            base = StaticStimulus.from_config(base_cfg)
        else:
            raise ValueError(f"Unsupported base class for pattern: {stim_class}")
        return cls(
            base_stimulus=base,
            copies_x=config.get('copies_x', 3),
            copies_y=config.get('copies_y', 2),
            spacing_x=config.get('spacing_x', 0.5),
            spacing_y=config.get('spacing_y', 0.5),
            center=tuple(config.get('center', (0.0, 0.0))),
            device=config.get('device', 'cpu'),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to configuration dictionary."""
        return {
            'class': 'RepeatedPatternStimulus',
            'base_stimulus': self.base_stimulus.to_dict(),
            'copies_x': self.copies_x,
            'copies_y': self.copies_y,
            'spacing_x': self.spacing_x,
            'spacing_y': self.spacing_y,
            'center': list(self.center),
            'device': str(self.device),
        }


class Stimulus:
    """Fluent builder interface for creating stimuli.
    
    This class provides static factory methods for creating various stimulus types
    and supports method chaining for adding motion and composing multiple stimuli.
    
    Example:
        >>> # Static Gaussian
        >>> s1 = Stimulus.gaussian(amplitude=1.0, sigma=0.3, center=(0.5, 0.5))
        >>> 
        >>> # Gaussian with linear motion
        >>> s2 = Stimulus.gaussian(amplitude=2.0, sigma=0.2).with_motion(
        ...     'linear', start=(0.0, 0.0), end=(1.0, 1.0), num_steps=100
        ... )
        >>> 
        >>> # Gabor with circular motion
        >>> s3 = Stimulus.gabor(wavelength=0.5, orientation=0.0).with_motion(
        ...     'circular', center=(0.0, 0.0), radius=0.5, num_steps=200
        ... )
        >>> 
        >>> # Composite
        >>> combined = Stimulus.compose([s1, s2, s3], mode='add')
    """
    
    @staticmethod
    def gaussian(
        amplitude: float = 1.0,
        sigma: float = 0.2,
        center: Tuple[float, float] = (0.0, 0.0),
        device: torch.device | str = 'cpu',
    ) -> StaticStimulus:
        """Create Gaussian bump stimulus.
        
        Args:
            amplitude: Peak amplitude of the Gaussian.
            sigma: Standard deviation. Units: mm.
            center: (x, y) center coordinates. Units: mm.
            device: Target device ('cpu', 'cuda', 'mps').
        
        Returns:
            StaticStimulus configured as Gaussian bump.
        
        Example:
            >>> stim = Stimulus.gaussian(amplitude=2.0, sigma=0.3, center=(0.5, 0.5))
        """
        return StaticStimulus(
            stim_type='gaussian',
            params={
                'amplitude': amplitude,
                'sigma': sigma,
                'center_x': center[0],
                'center_y': center[1],
            },
            device=device,
        )
    
    @staticmethod
    def point(
        amplitude: float = 1.0,
        diameter_mm: float = 0.6,
        center: Tuple[float, float] = (0.0, 0.0),
        device: torch.device | str = 'cpu',
    ) -> StaticStimulus:
        """Create binary disc (point) stimulus.
        
        Args:
            amplitude: Amplitude inside the disc.
            diameter_mm: Disc diameter. Units: mm.
            center: (x, y) center coordinates. Units: mm.
            device: Target device.
        
        Returns:
            StaticStimulus configured as binary disc.
        
        Example:
            >>> stim = Stimulus.point(amplitude=1.0, diameter_mm=0.8, center=(0, 0))
        """
        return StaticStimulus(
            stim_type='point',
            params={
                'amplitude': amplitude,
                'diameter_mm': diameter_mm,
                'center_x': center[0],
                'center_y': center[1],
            },
            device=device,
        )
    
    @staticmethod
    def edge(
        amplitude: float = 1.0,
        orientation: float = 0.0,
        width: float = 0.05,
        center: Tuple[float, float] = (0.0, 0.0),
        device: torch.device | str = 'cpu',
    ) -> StaticStimulus:
        """Create orientation-tuned edge stimulus.
        
        Args:
            amplitude: Edge amplitude.
            orientation: Edge orientation. Units: radians.
            width: Edge width (standard deviation). Units: mm.
            center: (x, y) center coordinates. Units: mm.
            device: Target device.
        
        Returns:
            StaticStimulus configured as oriented edge.
        
        Example:
            >>> import math
            >>> stim = Stimulus.edge(orientation=math.pi/4, width=0.03)
        """
        return StaticStimulus(
            stim_type='edge',
            params={
                'amplitude': amplitude,
                'orientation': orientation,
                'width': width,
                'center_x': center[0],
                'center_y': center[1],
            },
            device=device,
        )
    
    @staticmethod
    def gabor(
        amplitude: float = 1.0,
        sigma: float = 0.3,
        wavelength: float = 0.5,
        orientation: float = 0.0,
        phase: float = 0.0,
        center: Tuple[float, float] = (0.0, 0.0),
        device: torch.device | str = 'cpu',
    ) -> StaticStimulus:
        """Create Gabor texture stimulus (localized sinusoidal pattern).
        
        Args:
            amplitude: Peak amplitude.
            sigma: Gaussian envelope standard deviation. Units: mm.
            wavelength: Sinusoidal wavelength. Units: mm.
            orientation: Grating orientation. Units: radians.
            phase: Sinusoidal phase offset. Units: radians.
            center: (x, y) center coordinates. Units: mm.
            device: Target device.
        
        Returns:
            StaticStimulus configured as Gabor patch.
        
        Example:
            >>> import math
            >>> stim = Stimulus.gabor(
            ...     wavelength=0.4, orientation=math.pi/6, sigma=0.5
            ... )
        """
        return StaticStimulus(
            stim_type='gabor',
            params={
                'amplitude': amplitude,
                'sigma': sigma,
                'wavelength': wavelength,
                'orientation': orientation,
                'phase': phase,
                'center_x': center[0],
                'center_y': center[1],
            },
            device=device,
        )
    
    @staticmethod
    def edge_grating(
        amplitude: float = 1.0,
        orientation: float = 0.0,
        spacing: float = 0.6,
        count: int = 5,
        edge_width: float = 0.05,
        device: torch.device | str = 'cpu',
    ) -> StaticStimulus:
        """Create parallel edge grating stimulus.
        
        Args:
            amplitude: Grating amplitude.
            orientation: Grating orientation. Units: radians.
            spacing: Distance between edges. Units: mm.
            count: Number of parallel edges.
            edge_width: Width of each edge. Units: mm.
            device: Target device.
        
        Returns:
            StaticStimulus configured as edge grating.
        
        Example:
            >>> import math
            >>> stim = Stimulus.edge_grating(
            ...     orientation=0.0, spacing=0.8, count=7
            ... )
        """
        return StaticStimulus(
            stim_type='edge_grating',
            params={
                'amplitude': amplitude,
                'orientation': orientation,
                'spacing': spacing,
                'count': count,
                'edge_width': edge_width,
            },
            device=device,
        )
    
    @staticmethod
    def compose(
        stimuli: List[BaseStimulus],
        mode: CompositionMode = 'add',
        device: torch.device | str = 'cpu',
    ) -> CompositeStimulus:
        """Compose multiple stimuli with a combination mode.
        
        Args:
            stimuli: List of stimuli to combine.
            mode: Combination mode ('add', 'max', 'mean', 'multiply').
            device: Target device.
        
        Returns:
            CompositeStimulus combining all inputs.
        
        Raises:
            ValueError: If stimuli list is empty.
        
        Example:
            >>> s1 = Stimulus.gaussian(amplitude=1.0, sigma=0.3)
            >>> s2 = Stimulus.point(amplitude=0.5, diameter_mm=0.6, center=(1.0, 1.0))
            >>> combined = Stimulus.compose([s1, s2], mode='add')
        """
        return CompositeStimulus(stimuli=stimuli, mode=mode, device=device)

    @staticmethod
    def repeat_pattern(
        base_stimulus: BaseStimulus,
        copies_x: int = 3,
        copies_y: int = 2,
        spacing_x: float = 0.5,
        spacing_y: float = 0.5,
        center: Tuple[float, float] = (0.0, 0.0),
        device: torch.device | str = 'cpu',
    ) -> RepeatedPatternStimulus:
        """Tile a stimulus on an N×M grid.

        Creates textures (dot arrays, braille patterns, etc.) by copying
        a base stimulus at regular spatial offsets.

        Args:
            base_stimulus: Prototype stimulus shape.
            copies_x: Copies along x-axis.
            copies_y: Copies along y-axis.
            spacing_x: Spacing between copies in x (mm).
            spacing_y: Spacing between copies in y (mm).
            center: Center of the pattern grid (mm).
            device: Target device.

        Returns:
            RepeatedPatternStimulus tiling the base shape.

        Example:
            >>> dot = Stimulus.gaussian(amplitude=1.0, sigma=0.1)
            >>> braille = Stimulus.repeat_pattern(dot, 3, 2, 0.5, 0.5)
        """
        return RepeatedPatternStimulus(
            base_stimulus=base_stimulus,
            copies_x=copies_x, copies_y=copies_y,
            spacing_x=spacing_x, spacing_y=spacing_y,
            center=center, device=device,
        )

    @staticmethod
    def timeline(
        sub_stimuli: List[Dict[str, Any]],
        total_time_ms: float = 500.0,
        dt_ms: float = 0.5,
        composition_mode: CompositionMode = 'add',
        device: torch.device | str = 'cpu',
    ) -> TimelineStimulus:
        """Create a timeline with per-sub-stimulus onset and duration.

        Args:
            sub_stimuli: List of dicts with ``'stimulus'``,
                ``'onset_ms'``, ``'duration_ms'``, and optional
                ``'envelope'`` keys.
            total_time_ms: Global timeline length in ms.
            dt_ms: Discrete time step in ms.
            composition_mode: How overlapping regions are combined.
            device: Target device.

        Returns:
            TimelineStimulus managing the temporal arrangement.

        Example:
            >>> g = Stimulus.gaussian(amplitude=1.0, sigma=0.3)
            >>> tl = Stimulus.timeline([
            ...     {'stimulus': g, 'onset_ms': 0, 'duration_ms': 200},
            ... ], total_time_ms=500)
        """
        return TimelineStimulus(
            sub_stimuli=sub_stimuli,
            total_time_ms=total_time_ms, dt_ms=dt_ms,
            composition_mode=composition_mode, device=device,
        )


def with_motion(
    stimulus: BaseStimulus,
    motion_type: MotionType,
    **motion_params,
) -> MovingStimulus:
    """Add motion trajectory to a stimulus (functional API).
    
    This is a standalone function alternative to the method chaining pattern.
    
    Args:
        stimulus: Base stimulus to add motion to.
        motion_type: Type of motion ('linear', 'circular', 'stationary').
        **motion_params: Motion-specific parameters (start, end, center, radius, num_steps).
    
    Returns:
        MovingStimulus wrapping the base stimulus with motion.
    
    Example:
        >>> base = Stimulus.gaussian(amplitude=1.0, sigma=0.3)
        >>> moving = with_motion(
        ...     base, 'linear', start=(0, 0), end=(1, 1), num_steps=100
        ... )
    """
    return MovingStimulus(
        base_stimulus=stimulus,
        motion_type=motion_type,
        motion_params=motion_params,
    )


# Monkey-patch method onto BaseStimulus for fluent interface
BaseStimulus.with_motion = lambda self, motion_type, **params: with_motion(
    self, motion_type, **params
)
