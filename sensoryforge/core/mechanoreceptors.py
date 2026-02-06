"""
PyTorch-based mechanoreceptor response computation using conv2d.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_gaussian_kernel_torch(
    sigma_x, sigma_y, kernel_size, device="cpu", normalize=False
):
    """
    Create 2D Gaussian kernel for mechanoreceptor response convolution.

    Args:
        sigma_x, sigma_y: Gaussian standard deviations in pixels
        kernel_size: int, size of the square kernel (should be odd)
        device: torch device
        normalize: bool, whether to normalize kernel to sum=1
                  (default: False for biological realism)

    Returns:
        kernel: (1, 1, kernel_size, kernel_size) torch tensor
    """
    # Create coordinate grids centered at kernel center
    center = kernel_size // 2
    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - center
    y = torch.arange(kernel_size, device=device, dtype=torch.float32) - center
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    # Gaussian kernel (sensitivity profile)
    kernel = torch.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))

    # Biologically realistic normalization: peak value = 1, preserves shape
    # This ensures maximum response cannot exceed maximum input pressure
    if not normalize:
        kernel = kernel / kernel.max()  # Peak normalization for bio-realism
    else:
        kernel = kernel / kernel.sum()  # Sum normalization (old behavior)

    # Add batch and channel dimensions for conv2d
    return kernel.unsqueeze(0).unsqueeze(0)


def sigma_mm_to_pixels(sigma_mm, grid_spacing_mm):
    """
    Convert sigma from millimeters to pixels.

    Args:
        sigma_mm: standard deviation in millimeters
        grid_spacing_mm: spacing between grid points in millimeters

    Returns:
        sigma_pixels: standard deviation in pixels
    """
    return sigma_mm / grid_spacing_mm


class MechanoreceptorModule(nn.Module):
    """
    PyTorch module for mechanoreceptor response computation using conv2d.

    Biologically realistic implementation:
    - Uses unnormalized Gaussian kernels (sensitivity profiles)
    - Response represents mechanoreceptor activation, not weighted average
    - Maximum response cannot exceed maximum pressure applied
    """

    def __init__(
        self,
        sigma_x_mm=0.05,
        sigma_y_mm=0.05,
        grid_spacing_mm=0.15,
        kernel_size=None,
        device="cpu",
        normalize_kernel=False,
    ):
        """
        Args:
            sigma_x_mm, sigma_y_mm: Gaussian spread in mm (Parvizi-Fard)
            grid_spacing_mm: spacing between mechanoreceptors in mm
            kernel_size: convolution kernel size (auto-calculated if None)
            device: torch device
            normalize_kernel: bool, normalize Gaussian kernel (default: False)
        """
        super().__init__()

        self.sigma_x_mm = sigma_x_mm
        self.sigma_y_mm = sigma_y_mm
        self.grid_spacing_mm = grid_spacing_mm
        self.device = device
        self.normalize_kernel = normalize_kernel

        # Convert sigma to pixels
        self.sigma_x_pix = sigma_mm_to_pixels(sigma_x_mm, grid_spacing_mm)
        self.sigma_y_pix = sigma_mm_to_pixels(sigma_y_mm, grid_spacing_mm)

        # Calculate kernel size (6 sigma rule for Gaussian)
        if kernel_size is None:
            kernel_size = int(6 * max(self.sigma_x_pix, self.sigma_y_pix))
            if kernel_size % 2 == 0:  # Ensure odd kernel size
                kernel_size += 1

        self.kernel_size = kernel_size

        # Create Gaussian kernel (unnormalized by default for bio-realism)
        self.gaussian_kernel = create_gaussian_kernel_torch(
            self.sigma_x_pix,
            self.sigma_y_pix,
            kernel_size,
            device,
            normalize=normalize_kernel,
        )

        # Make it a parameter so it moves with the module
        self.register_buffer("kernel", self.gaussian_kernel)

    def forward(self, stimuli):
        """
        Compute mechanoreceptor responses using 2D convolution.

        Args:
            stimuli: torch tensor with shape:
                - (batch_size, grid_h, grid_w) for static stimuli
                - (batch_size, time_steps, grid_h, grid_w) for temporal

        Returns:
            responses: mechanoreceptor responses with same shape as input
        """
        original_shape = stimuli.shape

        if len(original_shape) == 3:
            # Static stimuli: (batch_size, grid_h, grid_w)
            batch_size, grid_h, grid_w = original_shape

            # Add channel dimension for conv2d: (batch_size, 1, grid_h, grid_w)
            stimuli_conv = stimuli.unsqueeze(1)

            # Apply convolution with padding to maintain size
            padding = self.kernel_size // 2
            responses_conv = F.conv2d(stimuli_conv, self.kernel, padding=padding)

            # Remove channel dimension: (batch_size, grid_h, grid_w)
            responses = responses_conv.squeeze(1)

        elif len(original_shape) == 4:
            # Temporal stimuli: (batch_size, time_steps, grid_h, grid_w)
            batch_size, time_steps, grid_h, grid_w = original_shape

            # Reshape for batch processing: (batch_size * time_steps, 1, grid_h, grid_w)
            stimuli_conv = stimuli.reshape(batch_size * time_steps, 1, grid_h, grid_w)

            # Apply convolution
            padding = self.kernel_size // 2
            responses_conv = F.conv2d(stimuli_conv, self.kernel, padding=padding)

            # Reshape back: (batch_size, time_steps, grid_h, grid_w)
            responses = responses_conv.reshape(batch_size, time_steps, grid_h, grid_w)
        else:
            raise ValueError(f"Unsupported stimulus shape: {original_shape}")

        return responses

    def update_parameters(self, sigma_x_mm=None, sigma_y_mm=None, grid_spacing_mm=None):
        """
        Update mechanoreceptor parameters and regenerate kernel.

        Args:
            sigma_x_mm, sigma_y_mm: new Gaussian spreads in mm
            grid_spacing_mm: new grid spacing in mm
        """
        if sigma_x_mm is not None:
            self.sigma_x_mm = sigma_x_mm
        if sigma_y_mm is not None:
            self.sigma_y_mm = sigma_y_mm
        if grid_spacing_mm is not None:
            self.grid_spacing_mm = grid_spacing_mm

        # Recalculate pixel sigmas
        self.sigma_x_pix = sigma_mm_to_pixels(self.sigma_x_mm, self.grid_spacing_mm)
        self.sigma_y_pix = sigma_mm_to_pixels(self.sigma_y_mm, self.grid_spacing_mm)

        # Recreate kernel
        new_kernel = create_gaussian_kernel_torch(
            self.sigma_x_pix,
            self.sigma_y_pix,
            self.kernel_size,
            self.device,
            normalize=self.normalize_kernel,
        )

        # Update the registered buffer
        self.kernel.data = new_kernel.squeeze(0).squeeze(0)
        self.gaussian_kernel = new_kernel

    def to_device(self, device):
        """Move module to a different device."""
        self.device = device
        return self.to(device)

    def get_kernel_info(self):
        """Return information about the current kernel."""
        return {
            "sigma_x_mm": self.sigma_x_mm,
            "sigma_y_mm": self.sigma_y_mm,
            "sigma_x_pix": self.sigma_x_pix,
            "sigma_y_pix": self.sigma_y_pix,
            "kernel_size": self.kernel_size,
            "grid_spacing_mm": self.grid_spacing_mm,
        }


def compute_mechanoreceptor_responses_torch(
    stimuli,
    sigma_x_mm=0.05,
    sigma_y_mm=0.05,
    grid_spacing_mm=0.15,
    device="cpu",
    normalize_kernel=False,
):
    """
    Standalone function for mechanoreceptor response computation.

    Args:
        stimuli: input stimuli tensor
        sigma_x_mm, sigma_y_mm: Gaussian spreads in mm
        grid_spacing_mm: grid spacing in mm
        device: torch device
        normalize_kernel: bool, normalize Gaussian kernel (default: False)

    Returns:
        responses: mechanoreceptor responses tensor
    """
    # Create temporary module
    mech_module = MechanoreceptorModule(
        sigma_x_mm,
        sigma_y_mm,
        grid_spacing_mm,
        device=device,
        normalize_kernel=normalize_kernel,
    )

    # Move stimuli to same device
    stimuli = stimuli.to(device)

    # Compute responses
    with torch.no_grad():
        responses = mech_module(stimuli)

    return responses
