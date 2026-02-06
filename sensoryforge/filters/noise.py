"""
encoding/noise_torch.py
----------------------
Additive membrane noise module for PyTorch tactile encoding pipeline.
Implements Gaussian white noise for filtered currents, as in biological neurons.
"""
import torch


class MembraneNoiseTorch:
    """
    Additive Gaussian membrane noise for filtered currents.
    Args:
        std (float): Standard deviation of the noise (same units as current)
        mean (float): Mean of the noise (default 0.0)
        seed (int, optional): Random seed for reproducibility
    """

    def __init__(self, std: float = 1.0, mean: float = 0.0, seed: int = None):
        self.std = std
        self.mean = mean
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, current: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to the input current tensor.
        Args:
            current (torch.Tensor): [batch, time, neurons] filtered current
        Returns:
            torch.Tensor: Noisy current, same shape as input
        """
        noise = torch.randn_like(current) * self.std + self.mean
        return current + noise


class ReceptorNoiseTorch:
    """
    Additive Gaussian noise for mechanoreceptor (receptor) responses.
    Args:
        std (float): Standard deviation of the noise (same units as response)
        mean (float): Mean of the noise (default 0.0)
        seed (int, optional): Random seed for reproducibility
    """

    def __init__(self, std: float = 1.0, mean: float = 0.0, seed: int = None):
        self.std = std
        self.mean = mean
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def __call__(self, responses: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to the input mechanoreceptor response tensor.
        Args:
            responses (torch.Tensor): [batch, time, height, width] mechanoreceptor responses
        Returns:
            torch.Tensor: Noisy responses, same shape as input
        """
        noise = torch.randn_like(responses) * self.std + self.mean
        return responses + noise
