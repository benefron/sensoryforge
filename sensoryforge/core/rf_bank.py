"""Receptive fields as one component: :class:`ReceptiveFieldBank` (Phase 2, I2).

A bank is the *result* of building receptive fields for one population: a
weight matrix from receptors to neurons, the two coordinate sets it was built
from, and a provenance record saying how. Builders (the
:class:`~sensoryforge.core.innervation.BaseInnervation` strategies,
``template``, ``imported``) produce banks; the engine, pipelines and GUI
consume them through :meth:`ReceptiveFieldBank.forward`.

Conventions (see ``docs/user_guide/receptive_fields.md``):

- Coordinates are ``(x, y)`` in mm everywhere inside SensoryForge.
- Receptor index ``k`` of a ``[rows, cols]`` grid built with
  ``indexing="ij"`` is ``k = i * cols + j`` -- the same row-major flattening
  that ``stimulus.reshape(T, rows * cols)`` uses, so a bank's weight column
  ``k`` is the receptor at ``xx[i, j], yy[i, j]``.
- Weights are ``float32`` ``[N, M]`` (``N`` neurons, ``M`` receptors),
  dimensionless multipliers applied to receptor responses.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

import sensoryforge

_WEIGHT_KEYS = ("innervation_weights", "weights", "W")


class ReceptiveFieldBank(nn.Module):
    """One population's receptive fields: weights, geometry and provenance.

    Attributes:
        weights: Buffer ``[N, M]`` ``float32`` -- ``weights[n, m]`` is the
            dimensionless gain from receptor ``m`` to neuron ``n``.
        neuron_centers: Buffer ``[N, 2]`` neuron centres ``(x, y)`` in mm.
        receptor_coords: Buffer ``[M, 2]`` receptor positions ``(x, y)`` in mm.
        provenance: Plain dict describing how the bank was built: ``builder``
            (registry name), ``builder_config`` (the builder's ``to_dict()``),
            ``seed``, ``source_path`` (imported banks), ``sensoryforge_version``.
            Always carries ``sensoryforge_version``; the rest depends on the
            builder.

    Example:
        >>> weights = torch.eye(4)                    # 4 neurons, 4 receptors
        >>> centers = torch.zeros(4, 2)
        >>> coords = torch.zeros(4, 2)
        >>> bank = ReceptiveFieldBank(weights, centers, coords)
        >>> bank(torch.ones(2, 10, 4)).shape          # [batch, time, N]
        torch.Size([2, 10, 4])
    """

    def __init__(
        self,
        weights: torch.Tensor,
        neuron_centers: torch.Tensor,
        receptor_coords: torch.Tensor,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Wrap a weight matrix and its geometry.

        Args:
            weights: ``[N, M]`` receptor-to-neuron weights (any float dtype;
                stored as ``float32``).
            neuron_centers: ``[N, 2]`` neuron centres ``(x, y)`` in mm.
            receptor_coords: ``[M, 2]`` receptor positions ``(x, y)`` in mm.
            provenance: How the bank was built (see class docstring). Copied;
                ``sensoryforge_version`` is filled in when absent.

        Raises:
            ValueError: If ``weights`` is not 2-D, the coordinate tensors are
                not ``[*, 2]``, or their row counts do not match ``weights``.
        """
        super().__init__()
        if weights.ndim != 2:
            raise ValueError(f"weights must be [N, M], got shape {list(weights.shape)}")
        n, m = weights.shape
        if neuron_centers.ndim != 2 or neuron_centers.shape[1] != 2:
            raise ValueError(
                "neuron_centers must be [N, 2] (x, y) in mm, got shape "
                f"{list(neuron_centers.shape)}"
            )
        if receptor_coords.ndim != 2 or receptor_coords.shape[1] != 2:
            raise ValueError(
                "receptor_coords must be [M, 2] (x, y) in mm, got shape "
                f"{list(receptor_coords.shape)}"
            )
        if neuron_centers.shape[0] != n:
            raise ValueError(
                f"neuron_centers has {neuron_centers.shape[0]} rows but weights "
                f"has N={n} neurons"
            )
        if receptor_coords.shape[0] != m:
            raise ValueError(
                f"receptor_coords has {receptor_coords.shape[0]} rows but weights "
                f"has M={m} receptors"
            )
        self.register_buffer("weights", weights.detach().to(torch.float32).contiguous())
        self.register_buffer(
            "neuron_centers", neuron_centers.detach().to(torch.float32).contiguous()
        )
        self.register_buffer(
            "receptor_coords",
            receptor_coords.detach().to(torch.float32).contiguous(),
        )
        prov: Dict[str, Any] = dict(provenance or {})
        prov.setdefault("sensoryforge_version", sensoryforge.__version__)
        self.provenance: Dict[str, Any] = prov

    # ------------------------------------------------------------------ #
    # Shape accessors
    # ------------------------------------------------------------------ #

    @property
    def num_neurons(self) -> int:
        """Number of neurons ``N``."""
        return int(self.weights.shape[0])

    @property
    def num_receptors(self) -> int:
        """Number of receptors ``M``."""
        return int(self.weights.shape[1])

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, receptor_responses: torch.Tensor) -> torch.Tensor:
        """Project receptor responses onto the neurons.

        Args:
            receptor_responses: ``[batch, M]`` or ``[batch, time, M]``
                receptor activations (any unit; the bank is dimensionless).

        Returns:
            ``[batch, N]`` or ``[batch, time, N]`` neuron drive in the same
            unit as the input.

        Raises:
            ValueError: If the input is not 2-D or 3-D, or its last dimension
                is not ``M``.
        """
        ndim = receptor_responses.ndim
        if ndim not in (2, 3):
            raise ValueError(
                "receptor_responses must be [batch, M] or [batch, time, M], "
                f"got shape {list(receptor_responses.shape)}"
            )
        if receptor_responses.shape[-1] != self.num_receptors:
            raise ValueError(
                f"receptor_responses has last dimension "
                f"{receptor_responses.shape[-1]} but the bank has "
                f"M={self.num_receptors} receptors "
                f"(input shape {list(receptor_responses.shape)}, "
                f"weights shape {list(self.weights.shape)})"
            )
        return torch.matmul(receptor_responses, self.weights.T)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: Union[str, os.PathLike]) -> None:
        """Write the bank as a ``.pt`` dict.

        The file holds ``innervation_weights`` ``[N, M]``, ``neuron_centers``
        ``[N, 2]``, ``receptor_coords`` ``[M, 2]`` (all CPU ``float32``) and
        ``provenance``. The key names match pressure-simulation's population
        files, so its viewer can open a bank that also carries a grid.

        Args:
            path: Destination file; parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "innervation_weights": self.weights.detach().cpu().clone(),
                "neuron_centers": self.neuron_centers.detach().cpu().clone(),
                "receptor_coords": self.receptor_coords.detach().cpu().clone(),
                "provenance": dict(self.provenance),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Union[str, os.PathLike],
        receptor_coords: Optional[torch.Tensor] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> "ReceptiveFieldBank":
        """Read a bank from :meth:`save` output or a pressure-simulation file.

        Accepted layouts:

        - weights under ``innervation_weights`` (preferred), ``weights`` or
          ``W``, shaped ``[N, M]`` or ``[N, H, W]`` (flattened row-major to
          ``[N, H * W]``);
        - ``neuron_centers`` ``[N, 2]`` (required);
        - ``receptor_coords`` ``[M, 2]`` (optional in the file; then the
          ``receptor_coords`` argument is required);
        - ``provenance`` (optional).

        Args:
            path: ``.pt`` file.
            receptor_coords: ``[M, 2]`` receptor positions ``(x, y)`` in mm,
                used when the file has none.
            device: Device to move the bank to (default: leave on CPU).

        Returns:
            The bank, with the file's ``provenance`` (or an empty one plus
            ``sensoryforge_version``) unchanged -- so ``load(save(bank))``
            reproduces ``bank.provenance`` exactly. The ``imported`` builder
            is what records a source path and file hash.

        Raises:
            ValueError: If no weight key is present, ``neuron_centers`` is
                missing, receptor coordinates are neither in the file nor
                given, or the receptor count does not match the weights.
        """
        path = Path(path)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict):
            raise ValueError(
                f"{path}: expected a dict with 'innervation_weights', got "
                f"{type(raw).__name__}"
            )
        weights = None
        for key in _WEIGHT_KEYS:
            if key in raw:
                weights = torch.as_tensor(raw[key])
                break
        if weights is None:
            raise ValueError(
                f"{path}: no weight tensor under any of {list(_WEIGHT_KEYS)} "
                f"(keys present: {sorted(raw)}); expected 'innervation_weights'"
            )
        if weights.ndim == 3:
            weights = weights.reshape(weights.shape[0], -1)
        elif weights.ndim != 2:
            raise ValueError(
                f"{path}: weights must be [N, M] or [N, H, W], got shape "
                f"{list(weights.shape)}"
            )
        if "neuron_centers" not in raw:
            raise ValueError(
                f"{path}: missing 'neuron_centers' [N, 2] (keys present: "
                f"{sorted(raw)})"
            )
        centers = torch.as_tensor(raw["neuron_centers"])
        file_coords = raw.get("receptor_coords")
        if file_coords is not None:
            coords = torch.as_tensor(file_coords)
        elif receptor_coords is not None:
            coords = receptor_coords.detach().cpu()
        else:
            raise ValueError(
                f"{path}: file has no 'receptor_coords'; pass receptor_coords="
                f"[{weights.shape[1]}, 2] (x, y) in mm"
            )
        if coords.ndim != 2 or coords.shape[0] != weights.shape[1]:
            raise ValueError(
                f"{path}: receptor_coords has shape {list(coords.shape)} but the "
                f"weights have M={weights.shape[1]} receptors"
            )
        provenance = dict(raw.get("provenance") or {})
        bank = cls(weights, centers, coords, provenance=provenance)
        if device is not None:
            bank = bank.to(device)
        return bank

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"num_neurons={self.num_neurons}, num_receptors={self.num_receptors}, "
            f"builder={self.provenance.get('builder')!r}"
        )
