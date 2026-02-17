"""encoding.stimulus_torch
===========================

Stimulus synthesis utilities used by the tactile encoding pipelines.  The
helpers in this module generate spatial pressure patterns (point, Gaussian,
edges, gratings, Gabor textures) as well as temporal amplitude trajectories and
motion profiles.  :class:`StimulusGenerator` wraps these primitives and
constructs batched tensors ready to feed into innervation modules.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from sensoryforge.core.grid import GridManager

import torch
import torch.nn.functional as F


def point_pressure_torch(
    xx: torch.Tensor,
    yy: torch.Tensor,
    center_x: float,
    center_y: float,
    amplitude: float = 1.0,
    diameter_mm: float = 0.6,
) -> torch.Tensor:
    """Generate a binary disc stimulus centred at ``(center_x, center_y)``."""
    r = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    mask = r <= (diameter_mm / 2)
    return amplitude * mask.float()


def gaussian_pressure_torch(
    xx: torch.Tensor,
    yy: torch.Tensor,
    center_x: float,
    center_y: float,
    amplitude: float = 1.0,
    sigma: float = 0.2,
) -> torch.Tensor:
    """Return a Gaussian bump with spatial standard deviation ``sigma``."""
    return amplitude * torch.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma**2)
    )


def gabor_texture_torch(
    xx: torch.Tensor,
    yy: torch.Tensor,
    center_x: float,
    center_y: float,
    *,
    amplitude: float = 1.0,
    sigma: float = 0.3,
    wavelength: float = 0.5,
    orientation: float = 0.0,
    phase: float = 0.0,
) -> torch.Tensor:
    """Generate a localised sinusoidal (Gabor) texture."""
    dx = xx - center_x
    dy = yy - center_y
    cos_theta = math.cos(orientation)
    sin_theta = math.sin(orientation)
    x_rot = dx * cos_theta + dy * sin_theta
    envelope = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    carrier = torch.cos(2 * math.pi * x_rot / max(wavelength, 1e-6) + phase)
    return amplitude * envelope * carrier


def edge_stimulus_torch(
    xx: torch.Tensor,
    yy: torch.Tensor,
    theta: float | torch.Tensor,
    w: float = 0.05,
    amplitude: float = 1.0,
) -> torch.Tensor:
    """Generate an orientation-tuned edge stimulus."""
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, device=xx.device, dtype=xx.dtype)
    else:
        theta = theta.to(device=xx.device, dtype=xx.dtype)

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    projection = xx * sin_theta + yy * cos_theta
    return amplitude * torch.exp(-(projection**2) / (2 * w**2))


def edge_grating_stimulus_torch(
    xx: torch.Tensor,
    yy: torch.Tensor,
    theta: float | torch.Tensor,
    *,
    spacing: float = 0.6,
    count: int = 10,
    w: float = 0.05,
    amplitude: float = 1.0,
    normalise: bool = True,
) -> torch.Tensor:
    """Generate a stack of parallel edge lobes (spatial grating)."""

    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, device=xx.device, dtype=xx.dtype)
    else:
        theta = theta.to(device=xx.device, dtype=xx.dtype)

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    projection = xx * sin_theta + yy * cos_theta

    offsets = (
        torch.linspace(
            -0.5 * (count - 1),
            0.5 * (count - 1),
            int(count),
            device=projection.device,
            dtype=projection.dtype,
        )
        * spacing
    )

    stack = torch.zeros_like(projection)
    for offset in offsets:
        stack += torch.exp(-((projection - offset) ** 2) / (2 * w**2))

    if normalise:
        peak = torch.max(stack)
        if peak > 0:
            stack = stack / peak

    return amplitude * stack


def create_temporal_profile_torch(
    total_time: float = 300.0,
    ramp_time: float = 50.0,
    dt: float = 0.1,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a trapezoidal temporal envelope (ramp → hold → ramp)."""
    time = torch.arange(0, total_time, dt, device=device)
    n_steps = len(time)
    amplitude = torch.zeros(n_steps, device=device)

    # Ramp up
    ramp_up_mask = time < ramp_time
    amplitude[ramp_up_mask] = time[ramp_up_mask] / ramp_time

    # Hold
    hold_mask = (time >= ramp_time) & (time < total_time - ramp_time)
    amplitude[hold_mask] = 1.0

    # Ramp down
    ramp_down_mask = time >= total_time - ramp_time
    ramp_down_time = time[ramp_down_mask] - (total_time - ramp_time)
    amplitude[ramp_down_mask] = 1.0 - ramp_down_time / ramp_time

    return time, amplitude


class StimulusGenerator:
    """Create batched spatial-temporal stimuli aligned to a grid."""

    def __init__(self, grid_manager: "GridManager") -> None:
        """Cache grid references for future stimulus creation."""
        self.grid_manager = grid_manager
        self.device = grid_manager.device
        if hasattr(grid_manager, "xx") and grid_manager.xx is not None:
            self.xx, self.yy = grid_manager.get_coordinates()
        else:
            # Poisson/hex/blue_noise: create regular grid from bounds
            props = grid_manager.get_grid_properties()
            n_x, n_y = grid_manager.grid_size
            xlim, ylim = props["xlim"], props["ylim"]
            x = torch.linspace(xlim[0], xlim[1], n_x, device=self.device)
            y = torch.linspace(ylim[0], ylim[1], n_y, device=self.device)
            self.xx, self.yy = torch.meshgrid(x, y, indexing="ij")

    def generate_batch_stimuli(
        self,
        stimulus_configs: Sequence[Dict[str, object]],
        time_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate a batch of static or temporal stimuli.

        Args:
            stimulus_configs: Sequence of stimulus configuration dicts.
                Each dict should contain ``'type'`` and relevant spatial
                parameters (``center_x``, ``center_y``, ``amplitude``,
                ``sigma``).  For temporal stimuli, include
                ``'time_steps'``.
            time_steps: Number of time steps for temporal stimuli.  If
                ``None``, static stimuli of shape ``[B, H, W]`` are
                returned. Individual configs may override with their own
                ``'time_steps'`` entry.

        Returns:
            Static: ``[batch, grid_h, grid_w]`` or
            Temporal: ``[batch, time_steps, grid_h, grid_w]``.

        Raises:
            ValueError: If temporal stimuli are requested but
                ``time_steps`` is not provided globally or per config,
                or if configs specify inconsistent time step counts.

        Example:
            >>> gen = StimulusGenerator(grid_manager)
            >>> configs = [
            ...     {'type': 'gaussian', 'amplitude': 20},
            ...     {'type': 'gaussian', 'amplitude': 40},
            ... ]
            >>> batch = gen.generate_batch_stimuli(configs)
            >>> batch.shape
            torch.Size([2, 80, 80])
        """  # (resolves ReviewFinding#L2)
        batch_size = len(stimulus_configs)
        grid_h, grid_w = self.grid_manager.grid_size

        if time_steps is None:
            # Static stimuli
            stimuli = torch.zeros(batch_size, grid_h, grid_w, device=self.device)
            for i, config in enumerate(stimulus_configs):
                stimuli[i] = self._generate_single_stimulus(config)
            return stimuli

        # Temporal stimuli
        temporal_batches: List[torch.Tensor] = []
        final_steps = time_steps
        for config in stimulus_configs:
            local_steps = config.get("time_steps", final_steps)
            if local_steps is None:
                raise ValueError(
                    "Temporal stimuli require 'time_steps' either globally or "
                    "per stimulus."
                )
            if final_steps is None:
                final_steps = local_steps
            elif local_steps != final_steps:
                raise ValueError(
                    "All temporal stimuli in the batch must share the same "
                    "number of time steps."
                )
            temporal_batches.append(
                self._generate_temporal_stimulus(config, local_steps)
            )

        stimuli = torch.stack(temporal_batches, dim=0)

        return stimuli

    def _generate_single_stimulus(
        self,
        config: Dict[str, object],
    ) -> torch.Tensor:
        """Materialise a single frame based on ``config``."""
        stim_type = config.get("type", "gaussian")

        if stim_type == "point":
            return point_pressure_torch(
                self.xx,
                self.yy,
                config.get("center_x", 0.0),
                config.get("center_y", 0.0),
                config.get("amplitude", 1.0),
                config.get("diameter_mm", 0.6),
            )
        elif stim_type == "gaussian":
            return gaussian_pressure_torch(
                self.xx,
                self.yy,
                config.get("center_x", 0.0),
                config.get("center_y", 0.0),
                config.get("amplitude", 1.0),
                config.get("sigma", 0.2),
            )
        elif stim_type == "edge":
            center_x = config.get("center_x", 0.0)
            center_y = config.get("center_y", 0.0)
            return edge_stimulus_torch(
                self.xx - center_x,
                self.yy - center_y,
                config.get("theta", 0.0),
                config.get("w", 0.05),
                config.get("amplitude", 1.0),
            )
        elif stim_type == "edge_grating":
            center_x = config.get("center_x", 0.0)
            center_y = config.get("center_y", 0.0)
            return edge_grating_stimulus_torch(
                self.xx - center_x,
                self.yy - center_y,
                config.get("theta", 0.0),
                spacing=config.get("spacing", 0.6),
                count=config.get("count", 5),
                w=config.get("w", 0.05),
                amplitude=config.get("amplitude", 1.0),
                normalise=config.get("normalise", True),
            )
        elif stim_type == "gabor":
            return gabor_texture_torch(
                self.xx,
                self.yy,
                config.get("center_x", 0.0),
                config.get("center_y", 0.0),
                amplitude=config.get("amplitude", 1.0),
                sigma=config.get("sigma", 0.3),
                wavelength=config.get("wavelength", 0.5),
                orientation=config.get("theta", 0.0),
                phase=config.get("phase", 0.0),
            )
        elif stim_type == "texture":
            kernel = int(config.get("kernel_size", 5))
            kernel = kernel if kernel % 2 == 1 else kernel + 1
            scale = config.get("scale", 0.3)
            noise = torch.randn(
                1, 1, self.xx.shape[0], self.xx.shape[1], device=self.device
            )
            texture = F.avg_pool2d(
                noise,
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            )
            return scale * texture.squeeze()
        else:
            raise ValueError(f"Unknown stimulus type: {stim_type}")

    def _generate_temporal_stimulus(
        self,
        config: Dict[str, object],
        time_steps: int,
    ) -> torch.Tensor:
        """Generate a temporal stimulus with amplitude and motion profiles."""
        amplitude_profile = self._build_amplitude_profile(config, time_steps)
        centers = self._build_motion_profile(config, time_steps)

        temporal_stimulus = torch.zeros(
            time_steps,
            self.grid_manager.grid_size[0],
            self.grid_manager.grid_size[1],
            device=self.device,
        )

        base_config = dict(config)
        for t in range(time_steps):
            frame_cfg = dict(base_config)
            frame_cfg["center_x"], frame_cfg["center_y"] = centers[t]
            frame = self._generate_single_stimulus(frame_cfg)
            temporal_stimulus[t] = frame * amplitude_profile[t]

        return temporal_stimulus

    def _build_amplitude_profile(
        self,
        config: Dict[str, object],
        time_steps: int,
    ) -> torch.Tensor:
        profile_cfg = config.get("temporal_profile", {})
        kind = profile_cfg.get("kind", "ramp").lower()

        if kind == "constant":
            return torch.ones(time_steps, device=self.device)

        if kind == "ramp":
            up = profile_cfg.get("up_steps", max(1, time_steps // 4))
            down = profile_cfg.get("down_steps", up)
            plateau = max(time_steps - up - down, 0)
            segments: List[torch.Tensor] = [
                torch.linspace(0, 1, up, device=self.device)
            ]
            if plateau > 0:
                segments.append(torch.ones(plateau, device=self.device))
            segments.append(torch.linspace(1, 0, down, device=self.device))
            return torch.cat(segments)[:time_steps]

        if kind == "square":
            start = profile_cfg.get("start_step", time_steps // 4)
            end = profile_cfg.get("end_step", start + time_steps // 2)
            amp = torch.zeros(time_steps, device=self.device)
            amp[start : min(end, time_steps)] = profile_cfg.get("level", 1.0)
            return amp

        if kind == "sin":
            cycles = profile_cfg.get("cycles", 1)
            phase = profile_cfg.get("phase", 0.0)
            t = torch.linspace(
                0,
                cycles * 2 * math.pi,
                time_steps,
                device=self.device,
            )
            return 0.5 * (1 + torch.sin(t + phase))

        if kind == "pulse_train":
            period = profile_cfg.get("period", max(1, time_steps // 5))
            width = profile_cfg.get("width", max(1, period // 2))
            amp = torch.zeros(time_steps, device=self.device)
            level = profile_cfg.get("level", 1.0)
            for start in range(0, time_steps, period):
                amp[start : min(start + width, time_steps)] = level
            return amp

        if kind == "stairs":
            levels = profile_cfg.get("levels", [0.0, 0.5, 1.0])
            segment = max(1, time_steps // len(levels))
            amp = torch.zeros(time_steps, device=self.device)
            for idx, level in enumerate(levels):
                start = idx * segment
                end = min(time_steps, start + segment)
                amp[start:end] = level
            return amp

        # Fallback linear ramp
        return torch.linspace(0, 1, time_steps, device=self.device)

    def _build_motion_profile(
        self,
        config: Dict[str, object],
        time_steps: int,
    ) -> List[Tuple[float, float]]:
        motion_cfg = config.get("motion", {})
        kind = motion_cfg.get("kind", "static").lower()
        base_x = config.get("center_x", 0.0)
        base_y = config.get("center_y", 0.0)

        if kind == "static":
            return [(base_x, base_y)] * time_steps

        span = motion_cfg.get("span", 1.0)
        offsets = torch.linspace(-span / 2, span / 2, time_steps)

        if kind == "horizontal":
            return [(base_x + float(dx), base_y) for dx in offsets]

        if kind == "vertical":
            return [(base_x, base_y + float(dy)) for dy in offsets]

        if kind == "diagonal":
            return [(base_x + float(d), base_y + float(d)) for d in offsets]

        if kind == "circular":
            radius = span
            theta = torch.linspace(0, 2 * math.pi, time_steps)
            return [
                (
                    base_x + radius * math.cos(float(t)),
                    base_y + radius * math.sin(float(t)),
                )
                for t in theta
            ]

        if kind == "path":
            path = motion_cfg.get("coordinates", [])
            if not path:
                return [(base_x, base_y)] * time_steps
            coords = torch.tensor(path, dtype=torch.float32)
            if coords.shape[0] != time_steps:
                coords = (
                    F.interpolate(
                        coords.unsqueeze(0).transpose(1, 2),
                        size=time_steps,
                        mode="linear",
                        align_corners=True,
                    )
                    .squeeze()
                    .transpose(0, 1)
                )
            return [(float(x), float(y)) for x, y in coords]

        # fallback
        return [(base_x, base_y)] * time_steps

    def to_device(self, device: torch.device | str) -> "StimulusGenerator":
        """Move stimulus generator to ``device`` and return ``self``."""
        self.device = device
        self.xx = self.xx.to(device)
        self.yy = self.yy.to(device)
        return self
