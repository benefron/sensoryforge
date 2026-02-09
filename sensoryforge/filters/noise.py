"""
encoding/noise_torch.py
-----------------------
Additive membrane noise module for PyTorch tactile encoding pipeline.
Implements Gaussian white noise for filtered currents, as in biological neurons.
"""

from typing import Optional

import torch
import torch.nn as nn


class MembraneNoiseTorch(nn.Module):
    """Additive Gaussian membrane noise for filtered currents.

    Inherits from ``nn.Module`` so that ``model.to(device)`` propagates
    correctly through the pipeline.  Uses a per-instance ``torch.Generator``
    to avoid polluting the global RNG state (resolves ReviewFinding#C2, #H5).

    Args:
        std: Standard deviation of the noise (same units as current).
        mean: Mean of the noise (default 0.0).
        seed: Optional random seed for reproducibility via local Generator.
    """

    def __init__(
        self,
        std: float = 1.0,
        mean: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.std = std
        self.mean = mean
        self.seed = seed
        # Per-instance generator avoids global RNG pollution
        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to the input current tensor.

        Args:
            current: ``[batch, time, neurons]`` filtered current.

        Returns:
            Noisy current, same shape as input.
        """
        if self._generator is not None:
            noise = torch.randn(
                current.shape,
                generator=self._generator,
                device=current.device,
                dtype=current.dtype,
            ) * self.std + self.mean
        else:
            noise = torch.randn_like(current) * self.std + self.mean
        return current + noise

    def reset_state(self) -> None:
        """Reset generator to initial seed for reproducibility."""
        if self.seed is not None and self._generator is not None:
            self._generator.manual_seed(self.seed)


class ReceptorNoiseTorch(nn.Module):
    """Additive Gaussian noise for mechanoreceptor (receptor) responses.

    Inherits from ``nn.Module`` so that ``model.to(device)`` propagates
    correctly through the pipeline.  Uses a per-instance ``torch.Generator``
    to avoid polluting the global RNG state (resolves ReviewFinding#C2, #H5).

    Args:
        std: Standard deviation of the noise (same units as response).
        mean: Mean of the noise (default 0.0).
        seed: Optional random seed for reproducibility via local Generator.
    """

    def __init__(
        self,
        std: float = 1.0,
        mean: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.std = std
        self.mean = mean
        self.seed = seed
        # Per-instance generator avoids global RNG pollution
        self._generator: Optional[torch.Generator] = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

    def forward(self, responses: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to the input mechanoreceptor response tensor.

        Args:
            responses: ``[batch, time, height, width]`` mechanoreceptor responses.

        Returns:
            Noisy responses, same shape as input.
        """
        if self._generator is not None:
            noise = torch.randn(
                responses.shape,
                generator=self._generator,
                device=responses.device,
                dtype=responses.dtype,
            ) * self.std + self.mean
        else:
            noise = torch.randn_like(responses) * self.std + self.mean
        return responses + noise

    def reset_state(self) -> None:
        """Reset generator to initial seed for reproducibility."""
        if self.seed is not None and self._generator is not None:
            self._generator.manual_seed(self.seed)
