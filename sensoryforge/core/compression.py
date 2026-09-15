"""Utilities for describing the tactile pipeline's compressed sensing stage.

The tactile encoding stack maps high-dimensional mechanoreceptor grids onto
comparatively small SA/RA neuron populations via the innervation tensors.  This
module exposes lightweight helpers for inspecting that linear operator and
reusing it downstream (e.g., when building analytical decoders).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import TactileEncodingPipelineTorch


@dataclass
class CompressionOperator:
    """Dense linear operator describing SA/RA spatial compression.

    Supports both grid-shaped innervation weights [num_neurons, H, W] and
    flat innervation weights [num_neurons, num_receptors].

    Attributes:
        sa_weights: SA innervation weights
        ra_weights: RA innervation weights
        grid_shape: Grid dimensions (H, W) or None for flat innervation
        num_receptors: Total receptor count (explicit for flat innervation)
    """

    sa_weights: torch.Tensor
    ra_weights: torch.Tensor
    grid_shape: Tuple[int, int] | None
    _num_receptors: int | None = None  # Explicit count for flat innervation

    def __post_init__(self) -> None:
        self.sa_weights = self.sa_weights.detach().clone()
        self.ra_weights = self.ra_weights.detach().clone()
        self.sa_weights.requires_grad_(False)
        self.ra_weights.requires_grad_(False)
        self._combined = torch.cat(
            [
                self.sa_weights.view(self.sa_weights.shape[0], -1),
                self.ra_weights.view(self.ra_weights.shape[0], -1),
            ],
            dim=0,
        )

    @property
    def combined_weights(self) -> torch.Tensor:
        """Return the concatenated SA+RA weight matrix [neurons, receptors]."""

        return self._combined

    @property
    def num_receptors(self) -> int:
        """Return total receptor count.

        For grid-shaped innervation, computes from grid_shape.
        For flat innervation, uses explicit _num_receptors.
        """
        if self._num_receptors is not None:
            return self._num_receptors
        if self.grid_shape is None:
            raise ValueError(
                "Cannot determine num_receptors: both grid_shape and _num_receptors are None"
            )
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
            _num_receptors=self._num_receptors,
        )

    def as_dict(self) -> Dict[str, torch.Tensor]:
        """Expose the raw innervation tensors for downstream modules."""

        return {"sa": self.sa_weights, "ra": self.ra_weights}


def _innervation_weights(pipeline: Any, module: Any) -> torch.Tensor:
    """Weights of a population as the legacy modules exposed them.

    A ``ReceptiveFieldBank`` (Phase 2, I6) holds ``weights [N, M]``; when the
    pipeline's grid has ``M == H * W`` receptors they are viewed as the
    grid-shaped ``[N, H, W]`` the legacy ``InnervationModule`` returned, so
    downstream grid detection is unchanged. Deprecated modules (and mocks)
    still expose ``innervation_weights`` directly.
    """
    weights = getattr(module, "innervation_weights", None)
    if weights is not None:
        return weights
    weights = module.weights
    grid_manager = getattr(pipeline, "grid_manager", None)
    grid_size = getattr(grid_manager, "grid_size", None)
    if (
        isinstance(grid_size, (tuple, list))
        and len(grid_size) == 2
        and all(isinstance(n, int) for n in grid_size)
        and int(grid_size[0]) * int(grid_size[1]) == weights.shape[1]
    ):
        return weights.view(weights.shape[0], int(grid_size[0]), int(grid_size[1]))
    return weights


def build_compression_operator(
    pipeline: "TactileEncodingPipelineTorch",
) -> CompressionOperator:
    """Construct a :class:`CompressionOperator` from a tactile pipeline.

    Automatically detects flat innervation [num_neurons, num_receptors] vs.
    grid-shaped innervation [num_neurons, grid_h, grid_w].
    """
    sa_weights = _innervation_weights(pipeline, pipeline.sa_innervation)
    ra_weights = _innervation_weights(pipeline, pipeline.ra_innervation)

    # Detect flat innervation (resolves ReviewFinding#H1)
    if sa_weights.ndim == 2:
        # Flat innervation: [num_neurons, num_receptors]
        num_receptors = int(sa_weights.shape[1])
        return CompressionOperator(
            sa_weights=sa_weights,
            ra_weights=ra_weights,
            grid_shape=None,
            _num_receptors=num_receptors,
        )
    else:
        # Grid-shaped innervation: [num_neurons, grid_h, grid_w]
        grid_shape = tuple(sa_weights.shape[1:])
        return CompressionOperator(
            sa_weights=sa_weights,
            ra_weights=ra_weights,
            grid_shape=grid_shape,
            _num_receptors=None,
        )
