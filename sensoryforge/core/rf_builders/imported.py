"""``imported`` builder: receptive fields read from files.

Three sources are accepted, selected by what ``path`` points at:

(a) **CSV folder** -- the GUI's population export: ``manifest.json`` naming
    ``neuron_positions.csv`` (``x_mm,y_mm`` header, one row per neuron) and
    ``innervation_weights.csv`` (``N`` rows x ``M`` columns, SensoryForge
    receptor order).
(b) **``.pt`` file** readable by :meth:`ReceptiveFieldBank.load`: a saved
    bank, or a pressure-simulation population file (``innervation_weights``
    ``[N, H, W]`` or ``[N, M]``, ``neuron_centers``; ``receptor_coords``
    optional).
(c) **``.npz`` file** shaped like pressure-simulation's ``ConstructedRF``:
    ``H [N, N_grid]``, ``centers [N, 2]`` in ``[y, x]``, optional ``sigma``
    and ``pitch``. Centres are converted to ``(x, y)`` and the columns of
    ``H`` are re-ordered from that repo's y-slow row-major grid ordering
    to SensoryForge's x-slow receptor ordering (see ``rf_bank.py``).

In every case the receptor count must equal the target grid's; a mismatch
raises ``ValueError`` naming both counts (no silent zero-fill). Provenance
records the absolute source path and a SHA-256 of the file(s).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sensoryforge.core.innervation import BaseInnervation
from sensoryforge.core.rf_bank import ReceptiveFieldBank
from sensoryforge.stimuli.base import ParamSpec



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lattice_shape(coords: torch.Tensor) -> Optional[Tuple[int, int]]:
    """``(n_x, n_y)`` if ``coords`` is a full regular lattice, else ``None``."""
    n_x = int(torch.unique(coords[:, 0]).numel())
    n_y = int(torch.unique(coords[:, 1]).numel())
    if n_x * n_y != coords.shape[0]:
        return None
    return n_x, n_y


class ImportedRFBuilder(BaseInnervation):
    """Receptive fields loaded from a CSV folder, a ``.pt`` bank or an ``.npz``.

    The neuron centres come from the file; a ``neuron_centers`` argument is
    ignored with a ``UserWarning``. Deterministic.

    Attributes:
        path: Source path as given.
        source_format: ``"csv_folder"``, ``"pt"`` or ``"npz"``.
        source_files: ``{file name: sha256}`` of every file read.
    """

    _TO_DICT_EXCLUDE_PARAMS = ("receptor_coords", "neuron_centers")
    DERIVES_NEURON_CENTERS = True

    def __init__(
        self,
        receptor_coords: torch.Tensor,
        neuron_centers: Optional[torch.Tensor] = None,
        path: Optional[str] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Read the file(s) and validate them against the target grid.

        Args:
            receptor_coords: ``[M, 2]`` receptor positions ``(x, y)`` in mm
                of the target grid.
            neuron_centers: Ignored (centres come from the file).
            path: CSV export folder, ``.pt`` bank file, or ``.npz``.
            device: Device for the weights and centres.

        Raises:
            ValueError: If ``path`` is missing, the format is not recognised,
                the file lacks the required arrays, or its receptor count
                differs from ``receptor_coords``.
            FileNotFoundError: If ``path`` does not exist.
        """
        if path is None:
            raise ValueError("imported builder requires path=<csv folder|.pt|.npz>")
        if neuron_centers is not None:
            warnings.warn(
                "imported builder takes its neuron centres from the file; the "
                "given neuron_centers are ignored",
                UserWarning,
                stacklevel=2,
            )
        self.path = path
        source = Path(path).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"imported builder: {source} does not exist")

        m_target = int(receptor_coords.shape[0])
        if source.is_dir():
            weights, centers, extra = self._read_csv_folder(source)
        elif source.suffix == ".pt":
            weights, centers, extra = self._read_pt(source, receptor_coords)
        elif source.suffix == ".npz":
            weights, centers, extra = self._read_npz(source, receptor_coords)
        else:
            raise ValueError(
                f"imported builder: {source} is not a csv folder, a .pt bank "
                "file or an .npz ConstructedRF file"
            )
        if weights.shape[1] != m_target:
            raise ValueError(
                f"imported builder: {source} has {weights.shape[1]} receptors "
                f"but the target grid has {m_target}; the receptor counts must "
                "match (no zero-fill)"
            )
        self.source_format: str = extra.pop("source_format")
        self.source_files: Dict[str, str] = extra.pop("source_files")
        self._source_extra: Dict[str, Any] = extra
        self._weights = weights.to(torch.float32)
        super().__init__(receptor_coords, centers.to(torch.float32), device=device)
        self._weights = self._weights.to(self.device)

    # ------------------------------------------------------------------ #
    # Readers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_csv_folder(
        folder: Path,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(
                f"imported builder: {folder} has no manifest.json (expected the "
                "GUI's population CSV export folder)"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            positions_path = folder / manifest["positions_file"]
            weights_path = folder / manifest["weights_file"]
        except KeyError as exc:
            raise ValueError(
                f"imported builder: {manifest_path} lacks key {exc}"
            ) from exc
        centers_np = np.loadtxt(positions_path, delimiter=",", skiprows=1, ndmin=2)
        weights_np = np.loadtxt(weights_path, delimiter=",", ndmin=2)
        files = {
            manifest_path.name: _sha256(manifest_path),
            positions_path.name: _sha256(positions_path),
            weights_path.name: _sha256(weights_path),
        }
        return (
            torch.from_numpy(np.ascontiguousarray(weights_np, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(centers_np, dtype=np.float32)),
            {"source_format": "csv_folder", "source_files": files},
        )

    @staticmethod
    def _read_pt(
        path: Path, receptor_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        bank = ReceptiveFieldBank.load(path, receptor_coords=receptor_coords.cpu())
        extra: Dict[str, Any] = {
            "source_format": "pt",
            "source_files": {path.name: _sha256(path)},
        }
        source_prov = {
            k: v for k, v in bank.provenance.items() if k != "sensoryforge_version"
        }
        if source_prov:
            extra["source_provenance"] = source_prov
        return bank.weights, bank.neuron_centers, extra

    @staticmethod
    def _read_npz(
        path: Path, receptor_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        with np.load(path) as data:
            keys = set(data.files)
            if "H" not in keys or "centers" not in keys:
                raise ValueError(
                    f"imported builder: {path} must hold 'H' [N, N_grid] and "
                    f"'centers' [N, 2] (has {sorted(keys)})"
                )
            H = torch.from_numpy(np.ascontiguousarray(data["H"], dtype=np.float32))
            centers_yx = torch.from_numpy(
                np.ascontiguousarray(data["centers"], dtype=np.float32)
            )
            extra: Dict[str, Any] = {
                "source_format": "npz",
                "source_files": {path.name: _sha256(path)},
            }
            for key, name in (
                ("sigma", "source_sigma_mm"),
                ("pitch", "source_pitch_mm"),
            ):
                if key in keys:
                    extra[name] = float(np.asarray(data[key]).reshape(-1)[0])
        if H.ndim != 2:
            raise ValueError(
                f"imported builder: {path} 'H' must be [N, N_grid], got "
                f"{list(H.shape)}"
            )
        if centers_yx.ndim != 2 or centers_yx.shape[1] != 2:
            raise ValueError(
                f"imported builder: {path} 'centers' must be [N, 2] in [y, x], "
                f"got {list(centers_yx.shape)}"
            )
        # [y, x] -> (x, y)
        centers_xy = centers_yx[:, [1, 0]].contiguous()
        # pressure-simulation flattens its (height, width) grid y-slow:
        # column j = iy * n_x + ix. SensoryForge receptors are x-slow:
        # k = ix * n_y + iy. Re-index when the target is a full lattice.
        shape = _lattice_shape(receptor_coords.cpu())
        if H.shape[1] == receptor_coords.shape[0] and shape is not None:
            n_x, n_y = shape
            H = H.reshape(H.shape[0], n_y, n_x).transpose(1, 2).reshape(H.shape[0], -1)
            extra["column_order"] = "converted yx-row-major -> xy-row-major"
        else:
            extra["column_order"] = "unchanged (target is not a full lattice)"
        return H.contiguous(), centers_xy, extra

    # ------------------------------------------------------------------ #
    # BaseInnervation API
    # ------------------------------------------------------------------ #

    def compute_weights(self, **kwargs: Any) -> torch.Tensor:
        """Return the imported ``[N, M]`` weights (a copy on ``device``)."""
        return self._weights.clone()

    def build(
        self,
        receptor_coords: Optional[torch.Tensor] = None,
        neuron_centers: Optional[torch.Tensor] = None,
        device: Optional[torch.device | str] = None,
    ) -> ReceptiveFieldBank:
        """Build the bank; ``neuron_centers`` is ignored (centres come from the file).

        Provenance carries ``source_path`` (absolute), ``source_format``,
        ``source_files`` (``{name: sha256}``), ``source_sha256`` (one digest
        over the files, in that order) and, when present in the source,
        ``source_provenance``, ``source_sigma_mm``, ``source_pitch_mm`` and
        ``column_order``.
        """
        bank = super().build(receptor_coords, None, device)
        combined = hashlib.sha256()
        for name in sorted(self.source_files):
            combined.update(name.encode("utf-8"))
            combined.update(self.source_files[name].encode("ascii"))
        bank.provenance.update(
            {
                "source_path": str(Path(self.path).expanduser().resolve()),
                "source_format": self.source_format,
                "source_files": dict(self.source_files),
                "source_sha256": (
                    next(iter(self.source_files.values()))
                    if len(self.source_files) == 1
                    else combined.hexdigest()
                ),
                **self._source_extra,
            }
        )
        return bank

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImportedRFBuilder":
        """Create from a config dict (``neuron_centers`` optional, ignored)."""
        config = dict(config)
        receptor_coords = config.pop("receptor_coords")
        config.pop("neuron_centers", None)
        for derived_key in cls._DERIVED_DICT_KEYS:
            config.pop(derived_key, None)
        return cls(receptor_coords, None, **config)

    def to_dict(self) -> Dict[str, Any]:
        """``method="imported"``, ``path`` (as given), ``device`` and counts."""
        result = super().to_dict()
        result["method"] = "imported"
        result["path"] = self.path
        return result

    @classmethod
    def get_param_spec(cls) -> List[ParamSpec]:
        """One parameter: the source path."""
        return [
            ParamSpec(
                "path",
                dtype="str",
                default="",
                tooltip=(
                    "CSV export folder, .pt bank file, or .npz ConstructedRF "
                    "file to import receptive fields from."
                ),
                group="Source",
            )
        ]
