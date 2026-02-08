"""Texture and grating stimulus generation module.

This module provides functions for generating textured patterns and spatial
gratings commonly used in tactile encoding experiments. Includes Gabor patches,
edge gratings, and noise-based textures.

Example:
    >>> import torch
    >>> from sensoryforge.stimuli.texture import gabor_texture, edge_grating
    >>> xx, yy = torch.meshgrid(torch.linspace(-2, 2, 50), torch.linspace(-2, 2, 50), indexing='ij')
    >>> gabor = gabor_texture(xx, yy, wavelength=0.5, orientation=0.0)
    >>> grating = edge_grating(xx, yy, orientation=0.0, spacing=0.6)
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def gabor_texture(
    xx: torch.Tensor,
    yy: torch.Tensor,
    center_x: float = 0.0,
    center_y: float = 0.0,
    amplitude: float = 1.0,
    sigma: float = 0.3,
    wavelength: float = 0.5,
    orientation: float = 0.0,
    phase: float = 0.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a Gabor texture (localized sinusoidal pattern).
    
    A Gabor patch is a sinusoidal grating modulated by a Gaussian envelope,
    commonly used to study orientation and spatial frequency sensitivity.
    
    Args:
        xx: Tensor of x-coordinates, shape [H, W]. Units: mm.
        yy: Tensor of y-coordinates, shape [H, W]. Units: mm.
        center_x: X-coordinate of pattern center. Units: mm.
        center_y: Y-coordinate of pattern center. Units: mm.
        amplitude: Peak amplitude of the pattern.
        sigma: Standard deviation of Gaussian envelope. Units: mm.
        wavelength: Spatial wavelength of the sinusoid. Units: mm.
        orientation: Orientation angle of the grating. Units: radians.
        phase: Phase offset of the sinusoid. Units: radians.
        device: Device to create the texture on. If None, uses xx.device.
    
    Returns:
        Gabor texture pattern, shape [H, W].
    
    Raises:
        ValueError: If wavelength is non-positive.
        ValueError: If sigma is non-positive.
    
    Example:
        >>> import torch
        >>> xx, yy = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing='ij')
        >>> texture = gabor_texture(xx, yy, wavelength=0.3, orientation=math.pi/4, sigma=0.4)
        >>> texture.shape
        torch.Size([64, 64])
    """
    if wavelength <= 0:
        raise ValueError(f"wavelength must be positive, got {wavelength}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    
    # Determine target device
    target_device = device if device is not None else xx.device
    xx = xx.to(target_device)
    yy = yy.to(target_device)
    
    # Translate coordinates to center
    dx = xx - center_x
    dy = yy - center_y
    
    # Rotate coordinates to align with grating orientation
    cos_theta = math.cos(orientation)
    sin_theta = math.sin(orientation)
    x_rot = dx * cos_theta + dy * sin_theta
    
    # Gaussian envelope
    envelope = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
    
    # Sinusoidal carrier wave
    carrier = torch.cos(2 * math.pi * x_rot / wavelength + phase)
    
    return amplitude * envelope * carrier


def edge_grating(
    xx: torch.Tensor,
    yy: torch.Tensor,
    orientation: float = 0.0,
    spacing: float = 0.6,
    count: int = 5,
    edge_width: float = 0.05,
    amplitude: float = 1.0,
    normalize: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a grating of parallel edge-like lobes.
    
    Creates a series of parallel Gaussian ridges (edges) oriented at a specific
    angle, useful for studying spatial frequency and orientation tuning.
    
    Args:
        xx: Tensor of x-coordinates, shape [H, W]. Units: mm.
        yy: Tensor of y-coordinates, shape [H, W]. Units: mm.
        orientation: Orientation angle of the grating. Units: radians.
        spacing: Distance between adjacent edges. Units: mm.
        count: Number of parallel edges to generate.
        edge_width: Width (std dev) of each edge. Units: mm.
        amplitude: Peak amplitude of the grating.
        normalize: If True, normalize peak to amplitude.
        device: Device to create the grating on. If None, uses xx.device.
    
    Returns:
        Edge grating pattern, shape [H, W].
    
    Raises:
        ValueError: If spacing is non-positive.
        ValueError: If edge_width is non-positive.
        ValueError: If count is less than 1.
    
    Example:
        >>> import torch
        >>> xx, yy = torch.meshgrid(torch.linspace(-2, 2, 64), torch.linspace(-2, 2, 64), indexing='ij')
        >>> grating = edge_grating(xx, yy, orientation=math.pi/6, spacing=0.5, count=7)
        >>> grating.shape
        torch.Size([64, 64])
    """
    if spacing <= 0:
        raise ValueError(f"spacing must be positive, got {spacing}")
    if edge_width <= 0:
        raise ValueError(f"edge_width must be positive, got {edge_width}")
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    
    # Determine target device
    target_device = device if device is not None else xx.device
    xx = xx.to(target_device)
    yy = yy.to(target_device)
    
    # Project coordinates onto grating orientation
    sin_theta = math.sin(orientation)
    cos_theta = math.cos(orientation)
    projection = xx * sin_theta + yy * cos_theta
    
    # Generate edge positions centered around origin
    offsets = torch.linspace(
        -0.5 * (count - 1) * spacing,
        0.5 * (count - 1) * spacing,
        count,
        device=target_device,
    )
    
    # Accumulate edges
    grating_pattern = torch.zeros_like(projection)
    for offset in offsets:
        grating_pattern += torch.exp(-((projection - offset) ** 2) / (2 * edge_width ** 2))
    
    # Normalize if requested
    if normalize:
        peak = torch.max(grating_pattern)
        if peak > 0:
            grating_pattern = grating_pattern / peak
    
    return amplitude * grating_pattern


def noise_texture(
    height: int,
    width: int,
    scale: float = 1.0,
    kernel_size: int = 5,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate a spatially correlated noise texture.
    
    Creates a random texture by smoothing white noise with a spatial filter.
    The resulting pattern has controlled spatial correlation, useful for
    naturalistic tactile stimulation.
    
    Args:
        height: Height of the texture in pixels.
        width: Width of the texture in pixels.
        scale: Amplitude scaling factor for the noise.
        kernel_size: Size of smoothing kernel (must be odd). Larger values
            produce smoother, more correlated textures.
        seed: Random seed for reproducibility. If None, uses random state.
        device: Device to create the texture on (cpu, cuda, mps).
    
    Returns:
        Noise texture, shape [height, width].
    
    Raises:
        ValueError: If kernel_size is even.
        ValueError: If height or width is non-positive.
    
    Example:
        >>> import torch
        >>> texture = noise_texture(64, 64, scale=0.5, kernel_size=7, seed=42)
        >>> texture.shape
        torch.Size([64, 64])
        >>> # Texture is deterministic with same seed
        >>> texture2 = noise_texture(64, 64, scale=0.5, kernel_size=7, seed=42)
        >>> torch.allclose(texture, texture2)
        True
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {height}, {width}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    
    # Set random seed if provided for deterministic generation
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None
    
    # Generate white noise
    noise = torch.randn(1, 1, height, width, device=device, generator=generator)
    
    # Smooth noise with average pooling to create spatial correlation
    texture = F.avg_pool2d(
        noise,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    
    # Remove batch and channel dimensions and scale
    return scale * texture.squeeze()


class GaborTexture(torch.nn.Module):
    """PyTorch module for generating Gabor texture stimuli.
    
    Encapsulates Gabor texture generation as a stateful torch.nn.Module.
    
    Attributes:
        center_x: X-coordinate of pattern center. Units: mm.
        center_y: Y-coordinate of pattern center. Units: mm.
        amplitude: Peak amplitude of the pattern.
        sigma: Standard deviation of Gaussian envelope. Units: mm.
        wavelength: Spatial wavelength of the sinusoid. Units: mm.
        orientation: Orientation angle of the grating. Units: radians.
        phase: Phase offset of the sinusoid. Units: radians.
    
    Example:
        >>> import torch
        >>> gabor = GaborTexture(wavelength=0.4, orientation=math.pi/3)
        >>> xx, yy = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing='ij')
        >>> texture = gabor(xx, yy)
        >>> texture.shape
        torch.Size([32, 32])
    """
    
    def __init__(
        self,
        center_x: float = 0.0,
        center_y: float = 0.0,
        amplitude: float = 1.0,
        sigma: float = 0.3,
        wavelength: float = 0.5,
        orientation: float = 0.0,
        phase: float = 0.0,
    ) -> None:
        """Initialize Gabor texture parameters.
        
        Args:
            center_x: X-coordinate of pattern center. Units: mm.
            center_y: Y-coordinate of pattern center. Units: mm.
            amplitude: Peak amplitude of the pattern.
            sigma: Standard deviation of Gaussian envelope. Units: mm.
            wavelength: Spatial wavelength of the sinusoid. Units: mm.
            orientation: Orientation angle of the grating. Units: radians.
            phase: Phase offset of the sinusoid. Units: radians.
        
        Raises:
            ValueError: If wavelength or sigma is non-positive.
        """
        super().__init__()
        
        if wavelength <= 0:
            raise ValueError(f"wavelength must be positive, got {wavelength}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        
        # Register parameters as buffers
        self.register_buffer("center_x", torch.tensor(center_x))
        self.register_buffer("center_y", torch.tensor(center_y))
        self.register_buffer("amplitude", torch.tensor(amplitude))
        self.register_buffer("sigma", torch.tensor(sigma))
        self.register_buffer("wavelength", torch.tensor(wavelength))
        self.register_buffer("orientation", torch.tensor(orientation))
        self.register_buffer("phase", torch.tensor(phase))
    
    def forward(self, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """Generate Gabor texture on the given coordinate grid.
        
        Args:
            xx: Tensor of x-coordinates, shape [H, W]. Units: mm.
            yy: Tensor of y-coordinates, shape [H, W]. Units: mm.
        
        Returns:
            Gabor texture pattern, shape [H, W].
        """
        return gabor_texture(
            xx,
            yy,
            center_x=float(self.center_x),
            center_y=float(self.center_y),
            amplitude=float(self.amplitude),
            sigma=float(self.sigma),
            wavelength=float(self.wavelength),
            orientation=float(self.orientation),
            phase=float(self.phase),
            device=xx.device,
        )


class EdgeGrating(torch.nn.Module):
    """PyTorch module for generating edge grating stimuli.
    
    Encapsulates edge grating generation as a stateful torch.nn.Module.
    
    Attributes:
        orientation: Orientation angle of the grating. Units: radians.
        spacing: Distance between adjacent edges. Units: mm.
        count: Number of parallel edges.
        edge_width: Width (std dev) of each edge. Units: mm.
        amplitude: Peak amplitude of the grating.
        normalize: Whether to normalize peak to amplitude.
    
    Example:
        >>> import torch
        >>> grating = EdgeGrating(orientation=0.0, spacing=0.5, count=10)
        >>> xx, yy = torch.meshgrid(torch.linspace(-2, 2, 64), torch.linspace(-2, 2, 64), indexing='ij')
        >>> pattern = grating(xx, yy)
        >>> pattern.shape
        torch.Size([64, 64])
    """
    
    def __init__(
        self,
        orientation: float = 0.0,
        spacing: float = 0.6,
        count: int = 5,
        edge_width: float = 0.05,
        amplitude: float = 1.0,
        normalize: bool = True,
    ) -> None:
        """Initialize edge grating parameters.
        
        Args:
            orientation: Orientation angle of the grating. Units: radians.
            spacing: Distance between adjacent edges. Units: mm.
            count: Number of parallel edges to generate.
            edge_width: Width (std dev) of each edge. Units: mm.
            amplitude: Peak amplitude of the grating.
            normalize: If True, normalize peak to amplitude.
        
        Raises:
            ValueError: If spacing or edge_width is non-positive.
            ValueError: If count is less than 1.
        """
        super().__init__()
        
        if spacing <= 0:
            raise ValueError(f"spacing must be positive, got {spacing}")
        if edge_width <= 0:
            raise ValueError(f"edge_width must be positive, got {edge_width}")
        if count < 1:
            raise ValueError(f"count must be at least 1, got {count}")
        
        # Register parameters as buffers
        self.register_buffer("orientation", torch.tensor(orientation))
        self.register_buffer("spacing", torch.tensor(spacing))
        self.register_buffer("edge_width", torch.tensor(edge_width))
        self.register_buffer("amplitude", torch.tensor(amplitude))
        self.count = count
        self.normalize = normalize
    
    def forward(self, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """Generate edge grating on the given coordinate grid.
        
        Args:
            xx: Tensor of x-coordinates, shape [H, W]. Units: mm.
            yy: Tensor of y-coordinates, shape [H, W]. Units: mm.
        
        Returns:
            Edge grating pattern, shape [H, W].
        """
        return edge_grating(
            xx,
            yy,
            orientation=float(self.orientation),
            spacing=float(self.spacing),
            count=self.count,
            edge_width=float(self.edge_width),
            amplitude=float(self.amplitude),
            normalize=self.normalize,
            device=xx.device,
        )


__all__ = [
    "gabor_texture",
    "edge_grating",
    "noise_texture",
    "GaborTexture",
    "EdgeGrating",
]
