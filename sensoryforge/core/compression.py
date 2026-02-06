"""Utilities for describing the tactile pipeline's compressed sensing stage.

The tactile encoding stack maps high-dimensional mechanoreceptor grids onto
comparatively small SA/RA neuron populations via the innervation tensors.  This
module exposes lightweight helpers for inspecting that linear operator and
reusing it downstream (e.g., when building analytical decoders).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline_torch import TactileEncodingPipelineTorch


@dataclass
class CompressionOperator:
    """Dense linear operator describing SA/RA spatial compression."""

    sa_weights: torch.Tensor
    ra_weights: torch.Tensor
    grid_shape: Tuple[int, int]

    def __post_init__(self) -> None:
        self.sa_weights = self.sa_weights.detach().clone()
        self.ra_weights = self.ra_weights.detach().clone()
        self.sa_weights.requires_grad_(False)
        self.ra_weights.requires_grad_(False)
        self._combined = torch.cat(
            [self.sa_weights.view(self.sa_weights.shape[0], -1),
             self.ra_weights.view(self.ra_weights.shape[0], -1)],
            dim=0,
        )

    @property
    def combined_weights(self) -> torch.Tensor:
        """Return the concatenated SA+RA weight matrix [neurons, receptors]."""

        return self._combined

    @property
    def num_receptors(self) -> int:
        return int(self.grid_shape[0] * self.grid_shape[1])

    def compression_ratio(self, population: str = "combined") -> float:
        """Ratio of neurons to receptors for the requested population."""

        population = population.lower()
        if population == "sa":
            count = self.sa_weights.shape[0]
        elif population == "ra":
            count = self.ra_weights.shape[0]
        elif population == "combined":
            count = self._combined.shape[0]
        else:
            raise ValueError(f"Unknown population '{population}'")
        return float(count) / float(self.num_receptors)

    def project(
        self, stimuli: torch.Tensor, population: str = "combined"
    ) -> torch.Tensor:
        """Project stimuli into neuron space using the stored weights."""

        population = population.lower()
        if stimuli.ndim == 3:
            batch, height, width = stimuli.shape
            flat = stimuli.reshape(batch, height * width)
            temporal = False
            time_steps = 1
        elif stimuli.ndim == 4:
            batch, time_steps, height, width = stimuli.shape
            flat = stimuli.reshape(batch * time_steps, height * width)
            temporal = True
        else:
            raise ValueError(
                "Stimuli must be shaped [batch, H, W] or [batch, T, H, W], "
                f"received {stimuli.shape}."
            )

        weight_matrix = self._select_matrix(population)
        projections = flat @ weight_matrix.T
        if temporal:
            projections = projections.view(batch, time_steps, -1)
        return projections

    def _select_matrix(self, population: str) -> torch.Tensor:
        if population == "sa":
            return self.sa_weights.view(self.sa_weights.shape[0], -1)
        if population == "ra":
            return self.ra_weights.view(self.ra_weights.shape[0], -1)
        if population == "combined":
            return self._combined
        raise ValueError(f"Unknown population '{population}'")

    def to(self, device: torch.device | str) -> "CompressionOperator":
        """Return a copy of the operator moved to ``device``."""

        return CompressionOperator(
            sa_weights=self.sa_weights.to(device),
            ra_weights=self.ra_weights.to(device),
            grid_shape=self.grid_shape,
        )

    def as_dict(self) -> Dict[str, torch.Tensor]:
        """Expose the raw innervation tensors for downstream modules."""

        return {"sa": self.sa_weights, "ra": self.ra_weights}


def build_compression_operator(
    pipeline: "TactileEncodingPipelineTorch",
) -> CompressionOperator:
    """Construct a :class:`CompressionOperator` from a tactile pipeline."""

    sa_weights = pipeline.sa_innervation.innervation_weights
    ra_weights = pipeline.ra_innervation.innervation_weights
    grid_shape = tuple(sa_weights.shape[1:])
    return CompressionOperator(
        sa_weights=sa_weights,
        ra_weights=ra_weights,
        grid_shape=grid_shape,
    )
