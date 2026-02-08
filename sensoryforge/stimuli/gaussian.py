"""Gaussian stimulus generation module.

This module provides functions and classes for generating Gaussian bump stimuli
on spatial grids. Gaussian stimuli are smooth, localized pressure patterns
commonly used in tactile encoding experiments.

Example:
    >>> import torch
    >>> from sensoryforge.stimuli.gaussian import gaussian_stimulus
    >>> xx, yy = torch.meshgrid(torch.linspace(-2, 2, 50), torch.linspace(-2, 2, 50), indexing='ij')
    >>> stim = gaussian_stimulus(xx, yy, center_x=0.0, center_y=0.0, amplitude=1.0, sigma=0.5)
    >>> stim.shape
    torch.Size([50, 50])
"""

from __future__ import annotations

import torch


def gaussian_stimulus(
    xx: torch.Tensor,
    yy: torch.Tensor,
    center_x: float,
    center_y: float,
    amplitude: float = 1.0,
    sigma: float = 0.2,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a 2D Gaussian bump stimulus centered at (center_x, center_y).
    
    The Gaussian is defined as:
        amplitude * exp(-((x - center_x)^2 + (y - center_y)^2) / (2 * sigma^2))
    
    Args:
        xx: Tensor of x-coordinates, shape [H, W] or [batch, H, W]. Units: mm.
        yy: Tensor of y-coordinates, shape [H, W] or [batch, H, W]. Units: mm.
        center_x: X-coordinate of Gaussian center. Units: mm.
        center_y: Y-coordinate of Gaussian center. Units: mm.
        amplitude: Peak amplitude of the Gaussian. Units: arbitrary (e.g., mA for pressure).
        sigma: Standard deviation of the Gaussian. Units: mm.
        device: Device to create the stimulus on (cpu, cuda, mps). If None, uses xx.device.
    
    Returns:
        Gaussian stimulus tensor with same shape as xx and yy. Units: same as amplitude.
    
    Raises:
        ValueError: If sigma is non-positive.
        ValueError: If xx and yy shapes don't match.
    
    Example:
        >>> import torch
        >>> xx, yy = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing='ij')
        >>> stim = gaussian_stimulus(xx, yy, 0.0, 0.0, amplitude=2.0, sigma=0.3)
        >>> stim.max().item() > 1.9  # Peak is close to amplitude
        True
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    
    if xx.shape != yy.shape:
        raise ValueError(f"xx and yy must have the same shape, got {xx.shape} and {yy.shape}")
    
    # Determine target device
    target_device = device if device is not None else xx.device
    
    # Move coordinates to target device if needed
    if xx.device != target_device:
        xx = xx.to(target_device)
    if yy.device != target_device:
        yy = yy.to(target_device)
    
    # Compute Gaussian
    r_squared = (xx - center_x) ** 2 + (yy - center_y) ** 2
    gaussian = amplitude * torch.exp(-r_squared / (2 * sigma ** 2))
    
    return gaussian


def multi_gaussian_stimulus(
    xx: torch.Tensor,
    yy: torch.Tensor,
    centers: list[tuple[float, float]],
    amplitudes: list[float] | None = None,
    sigmas: list[float] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a superposition of multiple Gaussian stimuli.
    
    This function creates multiple Gaussian bumps and adds them together,
    useful for creating complex spatial patterns.
    
    Args:
        xx: Tensor of x-coordinates, shape [H, W]. Units: mm.
        yy: Tensor of y-coordinates, shape [H, W]. Units: mm.
        centers: List of (x, y) tuples specifying Gaussian centers. Units: mm.
        amplitudes: List of amplitudes for each Gaussian. If None, uses 1.0 for all.
        sigmas: List of standard deviations for each Gaussian. If None, uses 0.2 for all.
        device: Device to create the stimulus on. If None, uses xx.device.
    
    Returns:
        Superposition of all Gaussian stimuli, shape [H, W].
    
    Raises:
        ValueError: If lengths of centers, amplitudes, and sigmas don't match.
    
    Example:
        >>> import torch
        >>> xx, yy = torch.meshgrid(torch.linspace(-2, 2, 64), torch.linspace(-2, 2, 64), indexing='ij')
        >>> centers = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0)]
        >>> stim = multi_gaussian_stimulus(xx, yy, centers, amplitudes=[1.0, 0.5, 0.5])
        >>> stim.shape
        torch.Size([64, 64])
    """
    n_gaussians = len(centers)
    
    # Set default amplitudes and sigmas if not provided
    if amplitudes is None:
        amplitudes = [1.0] * n_gaussians
    if sigmas is None:
        sigmas = [0.2] * n_gaussians
    
    # Validate lengths
    if len(amplitudes) != n_gaussians:
        raise ValueError(
            f"Number of amplitudes ({len(amplitudes)}) must match number of centers ({n_gaussians})"
        )
    if len(sigmas) != n_gaussians:
        raise ValueError(
            f"Number of sigmas ({len(sigmas)}) must match number of centers ({n_gaussians})"
        )
    
    # Determine target device
    target_device = device if device is not None else xx.device
    
    # Initialize output tensor
    result = torch.zeros_like(xx, device=target_device)
    
    # Add each Gaussian
    for (cx, cy), amp, sig in zip(centers, amplitudes, sigmas):
        result += gaussian_stimulus(xx, yy, cx, cy, amp, sig, device=target_device)
    
    return result


class GaussianStimulus(torch.nn.Module):
    """PyTorch module for generating Gaussian stimuli.
    
    This module encapsulates Gaussian stimulus generation as a stateful
    torch.nn.Module, allowing it to be composed with other PyTorch layers
    and moved between devices.
    
    Attributes:
        center_x: X-coordinate of Gaussian center. Units: mm.
        center_y: Y-coordinate of Gaussian center. Units: mm.
        amplitude: Peak amplitude of the Gaussian.
        sigma: Standard deviation of the Gaussian. Units: mm.
    
    Example:
        >>> import torch
        >>> gaussian = GaussianStimulus(center_x=0.5, center_y=-0.5, amplitude=2.0, sigma=0.3)
        >>> xx, yy = torch.meshgrid(torch.linspace(-2, 2, 32), torch.linspace(-2, 2, 32), indexing='ij')
        >>> stim = gaussian(xx, yy)
        >>> stim.shape
        torch.Size([32, 32])
    """
    
    def __init__(
        self,
        center_x: float = 0.0,
        center_y: float = 0.0,
        amplitude: float = 1.0,
        sigma: float = 0.2,
    ) -> None:
        """Initialize Gaussian stimulus parameters.
        
        Args:
            center_x: X-coordinate of Gaussian center. Units: mm.
            center_y: Y-coordinate of Gaussian center. Units: mm.
            amplitude: Peak amplitude of the Gaussian.
            sigma: Standard deviation of the Gaussian. Units: mm.
        
        Raises:
            ValueError: If sigma is non-positive.
        """
        super().__init__()
        
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        
        # Register parameters as buffers (non-trainable tensors that move with module)
        self.register_buffer("center_x", torch.tensor(center_x))
        self.register_buffer("center_y", torch.tensor(center_y))
        self.register_buffer("amplitude", torch.tensor(amplitude))
        self.register_buffer("sigma", torch.tensor(sigma))
    
    def forward(self, xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """Generate Gaussian stimulus on the given coordinate grid.
        
        Args:
            xx: Tensor of x-coordinates, shape [H, W] or [batch, H, W]. Units: mm.
            yy: Tensor of y-coordinates, shape [H, W] or [batch, H, W]. Units: mm.
        
        Returns:
            Gaussian stimulus tensor with same shape as xx and yy.
        
        Example:
            >>> gaussian = GaussianStimulus(0.0, 0.0, 1.0, 0.5)
            >>> xx = torch.linspace(-1, 1, 16).unsqueeze(0).expand(16, -1)
            >>> yy = torch.linspace(-1, 1, 16).unsqueeze(1).expand(-1, 16)
            >>> output = gaussian(xx, yy)
            >>> output.shape
            torch.Size([16, 16])
        """
        return gaussian_stimulus(
            xx,
            yy,
            center_x=float(self.center_x),
            center_y=float(self.center_y),
            amplitude=float(self.amplitude),
            sigma=float(self.sigma),
            device=xx.device,
        )


def batched_gaussian_stimulus(
    xx: torch.Tensor,
    yy: torch.Tensor,
    centers: torch.Tensor,
    amplitudes: torch.Tensor | None = None,
    sigmas: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a batch of Gaussian stimuli with different parameters.
    
    This function efficiently generates multiple Gaussian stimuli in parallel,
    with each item in the batch having potentially different centers, amplitudes,
    and sigmas. Useful for batch processing in neural network training.
    
    Args:
        xx: Tensor of x-coordinates, shape [H, W]. Units: mm.
        yy: Tensor of y-coordinates, shape [H, W]. Units: mm.
        centers: Batch of center coordinates, shape [batch, 2]. Units: mm.
        amplitudes: Batch of amplitudes, shape [batch]. If None, uses 1.0 for all.
        sigmas: Batch of standard deviations, shape [batch]. If None, uses 0.2 for all.
        device: Device to create stimuli on. If None, uses xx.device.
    
    Returns:
        Batch of Gaussian stimuli, shape [batch, H, W].
    
    Raises:
        ValueError: If centers doesn't have shape [batch, 2].
        ValueError: If amplitudes or sigmas don't match batch size.
    
    Example:
        >>> import torch
        >>> xx, yy = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing='ij')
        >>> centers = torch.tensor([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5]])
        >>> batch = batched_gaussian_stimulus(xx, yy, centers)
        >>> batch.shape
        torch.Size([3, 32, 32])
    """
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"centers must have shape [batch, 2], got {centers.shape}")
    
    batch_size = centers.shape[0]
    
    # Set defaults
    if amplitudes is None:
        amplitudes = torch.ones(batch_size, device=centers.device)
    if sigmas is None:
        sigmas = torch.full((batch_size,), 0.2, device=centers.device)
    
    # Validate shapes
    if amplitudes.shape[0] != batch_size:
        raise ValueError(
            f"amplitudes batch size ({amplitudes.shape[0]}) must match centers ({batch_size})"
        )
    if sigmas.shape[0] != batch_size:
        raise ValueError(
            f"sigmas batch size ({sigmas.shape[0]}) must match centers ({batch_size})"
        )
    
    # Determine target device
    target_device = device if device is not None else xx.device
    
    # Move everything to target device
    xx = xx.to(target_device)
    yy = yy.to(target_device)
    centers = centers.to(target_device)
    amplitudes = amplitudes.to(target_device)
    sigmas = sigmas.to(target_device)
    
    # Expand spatial grids to batch dimension: [1, H, W]
    xx_batch = xx.unsqueeze(0)  # [1, H, W]
    yy_batch = yy.unsqueeze(0)  # [1, H, W]
    
    # Expand parameters to spatial dimensions: [batch, 1, 1]
    centers_x = centers[:, 0].view(batch_size, 1, 1)  # [batch, 1, 1]
    centers_y = centers[:, 1].view(batch_size, 1, 1)  # [batch, 1, 1]
    amps = amplitudes.view(batch_size, 1, 1)  # [batch, 1, 1]
    sigs = sigmas.view(batch_size, 1, 1)  # [batch, 1, 1]
    
    # Compute batched Gaussians using broadcasting
    r_squared = (xx_batch - centers_x) ** 2 + (yy_batch - centers_y) ** 2
    gaussians = amps * torch.exp(-r_squared / (2 * sigs ** 2))
    
    return gaussians


__all__ = [
    "gaussian_stimulus",
    "multi_gaussian_stimulus",
    "batched_gaussian_stimulus",
    "GaussianStimulus",
]
