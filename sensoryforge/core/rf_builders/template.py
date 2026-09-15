"""``template`` builder: designed receptive fields from one resolvable distance.

The design chain (pressure-simulation's ``ConstructedRF``):

.. math::

    d \\;\\to\\; f_c = \\frac{1}{2d},\\quad \\sigma = \\frac{d}{\\pi},\\quad
    \\Delta = d,\\quad N = \\frac{A}{\\Delta^2}

One Gaussian template of width :math:`\\sigma` is translated over a square
neuron lattice of pitch :math:`\\Delta` covering the receptor area
:math:`A`, truncated to the ``k`` nearest receptors of each neuron, with
analytic weights :math:`\\exp(-r^2 / 2\\sigma^2)` and (by default) unit-L2
rows. With ``d = 0.40 mm`` on a 16x16 grid at 0.15 mm (extended side
2.4 mm) this gives :math:`\\sigma \\approx 0.1273` mm and ``N = 36``.

Deterministic: no seed, two builds are bit-identical.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from sensoryforge.core.innervation import BaseInnervation
from sensoryforge.stimuli.base import ParamSpec

if TYPE_CHECKING:
    from sensoryforge.core.rf_bank import ReceptiveFieldBank

_NORMALIZE_CHOICES = ("none", "l2", "sum")


def _infer_receptor_spacing(coords: torch.Tensor) -> float:
    """Smallest positive step between distinct x (or y) values, in mm.

    Exact for a regular lattice; for an irregular layout it is a lower bound
    that only nudges the lattice's bounding box outward slightly.
    """
    best = math.inf
    for axis in (0, 1):
        unique = torch.unique(coords[:, axis])
        if unique.numel() > 1:
            steps = unique[1:] - unique[:-1]
            steps = steps[steps > 1e-9]
            if steps.numel():
                best = min(best, float(steps.min()))
    return 0.0 if math.isinf(best) else best


class TemplateRFBuilder(BaseInnervation):
    """Designed receptive fields: one Gaussian template on a square lattice.

    Give either ``resolvable_distance_mm`` (``d``; then ``sigma_mm = d / pi``
    and ``pitch_mm = d``) or both ``sigma_mm`` and ``pitch_mm`` -- exactly one
    of the two forms.

    Neuron centres form a square lattice at ``pitch_mm`` over the receptor
    bounding box extended by half a receptor spacing on each side and inset
    by ``edge_offset_mm``; the lattice is centred in that domain and ordered
    row-major with x as the slow index and y as the fast one (the receptor
    ordering of ``ReceptorGrid``). The number of neurons is derived, so any
    ``neurons_per_row``/``neuron_centers`` a population config supplies is
    ignored.

    Attributes:
        sigma_mm: Gaussian width in mm (resolved).
        pitch_mm: Lattice pitch in mm (resolved).
        edge_offset_mm: Lattice inset from the extended bounding box in mm
            (resolved; default ``pitch_mm / 2``).
        lattice_shape: ``(n_x, n_y)`` neurons along x and y.
        k: Nearest receptors kept per neuron.
        normalize: ``"none"``, ``"l2"`` (unit-norm rows) or ``"sum"``.
        weight_scale: Multiplier applied after normalization.
    """

    _TO_DICT_EXCLUDE_PARAMS = ("receptor_coords", "neuron_centers")
    DERIVES_NEURON_CENTERS = True

    def __init__(
        self,
        receptor_coords: torch.Tensor,
        neuron_centers: Optional[torch.Tensor] = None,
        resolvable_distance_mm: Optional[float] = None,
        sigma_mm: Optional[float] = None,
        pitch_mm: Optional[float] = None,
        k: int = 28,
        normalize: str = "l2",
        weight_scale: float = 1.0,
        edge_offset_mm: Optional[float] = None,
        receptor_spacing_mm: Optional[float] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Design the neuron lattice; weights are computed in :meth:`build`.

        Args:
            receptor_coords: ``[M, 2]`` receptor positions ``(x, y)`` in mm.
            neuron_centers: Ignored (the lattice is derived); a
                ``UserWarning`` is emitted when given.
            resolvable_distance_mm: ``d`` in mm. Sets ``sigma_mm = d / pi``
                and ``pitch_mm = d``.
            sigma_mm: Gaussian width in mm (explicit form, with ``pitch_mm``).
            pitch_mm: Lattice pitch in mm (explicit form, with ``sigma_mm``).
            k: Nearest receptors per neuron (clamped to ``M``). Ties are
                broken toward the lower receptor index.
            normalize: ``"none"``, ``"l2"`` (default, unit-L2 rows as in
                pressure-simulation) or ``"sum"``.
            weight_scale: Multiplier applied after normalization.
            edge_offset_mm: Lattice inset from the extended receptor bounding
                box, in mm. Default ``pitch_mm / 2``.
            receptor_spacing_mm: Receptor spacing used to extend the bounding
                box by half a spacing per side. Default: inferred from the
                coordinates (exact for a regular grid).
            device: Device for the weights and centres.

        Raises:
            ValueError: If not exactly one parameter form is given, ``k < 1``,
                ``normalize`` is unknown, or the derived ``sigma``/``pitch``
                are not positive.
        """
        explicit = sigma_mm is not None or pitch_mm is not None
        if resolvable_distance_mm is not None and explicit:
            raise ValueError(
                "template builder takes exactly one parameter form: either "
                "resolvable_distance_mm, or sigma_mm together with pitch_mm "
                f"(got resolvable_distance_mm={resolvable_distance_mm}, "
                f"sigma_mm={sigma_mm}, pitch_mm={pitch_mm})"
            )
        if resolvable_distance_mm is None and (sigma_mm is None or pitch_mm is None):
            raise ValueError(
                "template builder takes exactly one parameter form: either "
                "resolvable_distance_mm, or sigma_mm together with pitch_mm "
                f"(got resolvable_distance_mm={resolvable_distance_mm}, "
                f"sigma_mm={sigma_mm}, pitch_mm={pitch_mm})"
            )
        if normalize not in _NORMALIZE_CHOICES:
            raise ValueError(
                f"normalize must be one of {_NORMALIZE_CHOICES}, got {normalize!r}"
            )
        if int(k) < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if neuron_centers is not None:
            warnings.warn(
                "template builder derives its own neuron lattice; the given "
                "neuron_centers are ignored",
                UserWarning,
                stacklevel=2,
            )

        # Raw constructor arguments, for to_dict()/from_config() round trip.
        self.resolvable_distance_mm = resolvable_distance_mm
        self._sigma_mm_arg = sigma_mm
        self._pitch_mm_arg = pitch_mm
        self._edge_offset_mm_arg = edge_offset_mm
        self._receptor_spacing_mm_arg = receptor_spacing_mm
        self.k = int(k)
        self.normalize = normalize
        self.weight_scale = float(weight_scale)

        if resolvable_distance_mm is not None:
            d = float(resolvable_distance_mm)
            self.sigma_mm = d / math.pi
            self.pitch_mm = d
        else:
            self.sigma_mm = float(sigma_mm)  # type: ignore[arg-type]
            self.pitch_mm = float(pitch_mm)  # type: ignore[arg-type]
        if self.sigma_mm <= 0 or self.pitch_mm <= 0:
            raise ValueError(
                f"sigma_mm and pitch_mm must be positive, got sigma_mm="
                f"{self.sigma_mm}, pitch_mm={self.pitch_mm}"
            )
        self.edge_offset_mm = (
            float(edge_offset_mm) if edge_offset_mm is not None else self.pitch_mm / 2
        )

        coords64 = receptor_coords.detach().to("cpu", torch.float64)
        if coords64.ndim != 2 or coords64.shape[1] != 2:
            raise ValueError(
                "receptor_coords must be [M, 2] (x, y) in mm, got shape "
                f"{list(receptor_coords.shape)}"
            )
        self.receptor_spacing_mm = (
            float(receptor_spacing_mm)
            if receptor_spacing_mm is not None
            else _infer_receptor_spacing(coords64)
        )
        centers64, self.lattice_shape = self._design_lattice(coords64)
        super().__init__(receptor_coords, centers64.to(torch.float32), device=device)

    # ------------------------------------------------------------------ #
    # Lattice design
    # ------------------------------------------------------------------ #

    def _design_lattice(
        self, coords64: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Square lattice at ``pitch_mm`` inside the extended, inset box."""
        half = self.receptor_spacing_mm / 2
        lo = coords64.min(dim=0).values - half + self.edge_offset_mm
        hi = coords64.max(dim=0).values + half - self.edge_offset_mm
        avail = (hi - lo).clamp(min=0.0)
        # 1e-6 of a pitch of slack so float32 receptor coordinates (whose
        # inferred spacing can be one ulp short) still fill the box exactly.
        counts = torch.floor(avail / self.pitch_mm + 1e-6).long() + 1
        n_x, n_y = int(counts[0]), int(counts[1])
        # Centre the lattice in the available span.
        start = lo + (avail - (counts.to(torch.float64) - 1) * self.pitch_mm) / 2
        xs = start[0] + self.pitch_mm * torch.arange(n_x, dtype=torch.float64)
        ys = start[1] + self.pitch_mm * torch.arange(n_y, dtype=torch.float64)
        xx, yy = torch.meshgrid(xs, ys, indexing="ij")  # x slow, y fast
        centers = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        return centers, (n_x, n_y)

    # ------------------------------------------------------------------ #
    # Weights
    # ------------------------------------------------------------------ #

    def compute_weights(self, **kwargs: Any) -> torch.Tensor:
        """Analytic Gaussian weights on each neuron's ``k`` nearest receptors.

        Returns:
            ``[N, M]`` ``float32`` weights on ``device``; each row has exactly
            ``min(k, M)`` non-zeros, ``exp(-r^2 / 2 sigma^2)`` before
            normalization, then ``normalize`` and ``weight_scale``.
        """
        coords = self.receptor_coords.to(torch.float64)
        centers = self.neuron_centers.to(torch.float64)
        d2 = ((centers.unsqueeze(1) - coords.unsqueeze(0)) ** 2).sum(-1)  # [N, M]
        k = min(self.k, self.num_receptors)
        # stable sort keeps the lower receptor index first among equal distances
        order = torch.sort(d2, dim=1, stable=True).indices[:, :k]
        d2_k = torch.gather(d2, 1, order)
        vals = torch.exp(-d2_k / (2.0 * self.sigma_mm**2))
        if self.normalize == "l2":
            vals = vals / vals.norm(dim=1, keepdim=True).clamp(min=1e-300)
        elif self.normalize == "sum":
            vals = vals / vals.sum(dim=1, keepdim=True).clamp(min=1e-300)
        vals = vals * self.weight_scale
        weights = torch.zeros(
            self.num_neurons, self.num_receptors, dtype=torch.float64, device=d2.device
        )
        weights.scatter_(1, order, vals)
        return weights.to(torch.float32)

    def build(
        self,
        receptor_coords: Optional[torch.Tensor] = None,
        neuron_centers: Optional[torch.Tensor] = None,
        device: Optional[torch.device | str] = None,
    ) -> "ReceptiveFieldBank":
        """Build the bank; ``neuron_centers`` is ignored (lattice is derived).

        Provenance additionally carries ``derived`` with the resolved
        ``sigma_mm``, ``pitch_mm``, ``edge_offset_mm``, ``receptor_spacing_mm``,
        ``lattice_shape`` and ``num_neurons``.
        """
        bank = super().build(receptor_coords, None, device)
        if "derived" not in bank.provenance:
            bank.provenance["derived"] = {
                "sigma_mm": self.sigma_mm,
                "pitch_mm": self.pitch_mm,
                "edge_offset_mm": self.edge_offset_mm,
                "receptor_spacing_mm": self.receptor_spacing_mm,
                "lattice_shape": list(self.lattice_shape),
                "num_neurons": bank.num_neurons,
            }
        return bank

    # ------------------------------------------------------------------ #
    # Config round trip
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TemplateRFBuilder":
        """Create from a config dict (``neuron_centers`` optional, ignored).

        Args:
            config: ``receptor_coords`` plus constructor keywords; the
                derived ``method``/``num_neurons``/``num_receptors`` keys of
                :meth:`to_dict` are dropped.
        """
        config = dict(config)
        receptor_coords = config.pop("receptor_coords")
        config.pop("neuron_centers", None)
        for derived_key in cls._DERIVED_DICT_KEYS:
            config.pop(derived_key, None)
        return cls(receptor_coords, None, **config)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the raw constructor arguments (fixed point under
        :meth:`from_config`).

        Returns:
            ``method="template"``, ``num_neurons``, ``num_receptors``,
            ``device`` and every constructor parameter as given (so
            ``sigma_mm``/``pitch_mm`` are ``None`` when ``d`` was used).
        """
        result = super().to_dict()
        result["method"] = "template"
        result.update(
            {
                "resolvable_distance_mm": self.resolvable_distance_mm,
                "sigma_mm": self._sigma_mm_arg,
                "pitch_mm": self._pitch_mm_arg,
                "k": self.k,
                "normalize": self.normalize,
                "weight_scale": self.weight_scale,
                "edge_offset_mm": self._edge_offset_mm_arg,
                "receptor_spacing_mm": self._receptor_spacing_mm_arg,
            }
        )
        return result

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        """ParamSpecs for the GUI (``resolvable_distance_mm`` form first)."""
        return [
            ParamSpec(
                "resolvable_distance_mm",
                dtype="float",
                default=0.40,
                min_val=0.01,
                max_val=50.0,
                unit="mm",
                tooltip="Resolvable distance d: sigma = d/pi, pitch = d.",
                group="Design",
            ),
            ParamSpec(
                "k",
                dtype="int",
                default=28,
                min_val=1,
                max_val=10000,
                tooltip="Nearest receptors kept per neuron.",
                group="Design",
            ),
            ParamSpec(
                "normalize",
                dtype="str",
                default="l2",
                choices=list(_NORMALIZE_CHOICES),
                tooltip="Row normalization: unit-L2 (default), sum-to-one or none.",
                group="Design",
                advanced=True,
            ),
            ParamSpec(
                "weight_scale",
                dtype="float",
                default=1.0,
                min_val=0.0,
                max_val=1e6,
                tooltip="Multiplier applied after normalization.",
                group="Design",
                advanced=True,
            ),
            ParamSpec(
                "edge_offset_mm",
                dtype="float",
                default=None,
                min_val=0.0,
                max_val=50.0,
                unit="mm",
                tooltip="Lattice inset from the receptor box (default pitch/2).",
                group="Design",
                advanced=True,
            ),
        ]
