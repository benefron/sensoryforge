"""Project-scoped registry for STA protocols, runs, neuron manifests, and
analysis artifacts.

This module implements the shared persistence layer described in
``docs_root/STA_pipeline_plan.md``.
It provides dataclasses that mirror the JSON examples in the plan and a
:class:`ProjectRegistry` helper for managing serialized metadata on disk.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    # Torch is optional for registry usage, but device detection relies on it.
    import torch
except Exception:  # pragma: no cover - optional dependency guard
    torch = None  # type: ignore

__all__ = [
    "ProtocolDefinition",
    "NeuronModuleManifest",
    "ProtocolRunRecord",
    "STAConfigurationResult",
    "STAConfiguration",
    "STAAnalysisRecord",
    "DecodingAnalysisRecord",
    "ProjectRegistry",
    "detect_compute_devices",
    "suggest_worker_count",
]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_identifier(label: str, value: str) -> None:
    if not value:
        raise ValueError(f"{label} must be non-empty")
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{label} '{value}' contains invalid characters; "
            "only letters, numbers, underscores, dashes, and dots are allowed."
        )


def _strip_none(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively drop keys with ``None`` values from nested dictionaries."""

    def _clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [_clean(v) for v in value if v is not None]
        return value

    return _clean(data)


@dataclass
class ProtocolDefinition:
    """Metadata describing a stimulus protocol used for STA collection."""

    protocol_id: str
    version: str
    name: str
    description: str
    stimulus: Dict[str, Any]
    execution: Dict[str, Any]
    tags: Sequence[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("protocol_id", self.protocol_id)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["protocol_id"] = self.protocol_id
        data["tags"] = list(self.tags)
        return _strip_none(data)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProtocolDefinition":
        return cls(
            protocol_id=payload["protocol_id"],
            version=payload.get("version", "1.0"),
            name=payload.get("name", payload["protocol_id"]),
            description=payload.get("description", ""),
            stimulus=payload.get("stimulus", {}),
            execution=payload.get("execution", {}),
            tags=payload.get("tags", []),
            metadata=payload.get("metadata", {}),
        )


@dataclass
class NeuronModuleManifest:
    """Serialized neuron bundle metadata discoverable by the Spiking tab."""

    name: str
    model: str
    filter: Optional[str]
    parameters: Dict[str, Any]
    file: str
    tags: Sequence[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("manifest name", self.name)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags)
        return _strip_none(data)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NeuronModuleManifest":
        return cls(
            name=payload["name"],
            model=payload.get("model", ""),
            filter=payload.get("filter"),
            parameters=payload.get("parameters", {}),
            file=payload.get("file", ""),
            tags=payload.get("tags", []),
            metadata=payload.get("metadata", {}),
        )


@dataclass
class ProtocolRunRecord:
    """Metadata and tensor locations produced by Protocol Suite execution."""

    run_id: str
    protocol_id: str
    created_at: str
    version: str
    stimulus_reference: str
    neuron_modules: Sequence[NeuronModuleManifest]
    tensors: Dict[str, str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("run_id", self.run_id)
        _validate_identifier("protocol_id", self.protocol_id)

    @classmethod
    def new(
        cls,
        run_id: str,
        protocol_id: str,
        stimulus_reference: str,
        neuron_modules: Sequence[NeuronModuleManifest],
        tensors: Dict[str, str],
        version: str = "1.0",
        metrics: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ProtocolRunRecord":
        created_at = datetime.now(timezone.utc).isoformat()
        return cls(
            run_id=run_id,
            protocol_id=protocol_id,
            created_at=created_at,
            version=version,
            stimulus_reference=stimulus_reference,
            neuron_modules=list(neuron_modules),
            tensors=dict(tensors),
            metrics=metrics or {},
            notes=notes,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["neuron_modules"] = [
            manifest.to_dict() for manifest in self.neuron_modules
        ]
        return _strip_none(data)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProtocolRunRecord":
        manifests = [
            NeuronModuleManifest.from_dict(m) for m in payload.get("neuron_modules", [])
        ]
        return cls(
            run_id=payload["run_id"],
            protocol_id=payload.get("protocol_id", ""),
            created_at=payload.get(
                "created_at",
                datetime.now(timezone.utc).isoformat(),
            ),
            version=payload.get("version", "1.0"),
            stimulus_reference=payload.get("stimulus_reference", ""),
            neuron_modules=manifests,
            tensors=payload.get("tensors", {}),
            metrics=payload.get("metrics", {}),
            notes=payload.get("notes"),
            metadata=payload.get("metadata", {}),
        )


@dataclass
class STAConfigurationResult:
    """Single configuration result inside an STA analysis record."""

    kernel: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none(asdict(self))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "STAConfigurationResult":
        return cls(
            kernel=payload.get("kernel"),
            metrics=payload.get("metrics", {}),
            parameters=payload.get("parameters", {}),
        )


@dataclass
class STAConfiguration:
    """Configuration entry describing how an STA kernel was computed."""

    name: str
    method: str
    signal_source: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    kernel: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    results: Sequence[STAConfigurationResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_identifier("configuration name", self.name)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["results"] = [result.to_dict() for result in self.results]
        return _strip_none(data)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "STAConfiguration":
        results_payload = payload.get("results") or []
        results: List[STAConfigurationResult] = []
        for entry in results_payload:
            results.append(STAConfigurationResult.from_dict(entry))
        return cls(
            name=payload["name"],
            method=payload.get("method", "regular"),
            signal_source=payload.get("signal_source", "membrane"),
            parameters=payload.get("parameters", {}),
            hyperparameters=payload.get("hyperparameters", {}),
            kernel=payload.get("kernel"),
            metrics=payload.get("metrics", {}),
            results=results,
        )


@dataclass
class STAAnalysisRecord:
    """Top-level record for STA analysis outputs."""

    analysis_id: str
    source_run: str
    population: str
    created_at: str
    version: str
    configurations: Sequence[STAConfiguration]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("analysis_id", self.analysis_id)
        _validate_identifier("source_run", self.source_run)

    @classmethod
    def new(
        cls,
        analysis_id: str,
        source_run: str,
        population: str,
        configurations: Sequence[STAConfiguration],
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "STAAnalysisRecord":
        created_at = datetime.now(timezone.utc).isoformat()
        return cls(
            analysis_id=analysis_id,
            source_run=source_run,
            population=population,
            created_at=created_at,
            version=version,
            configurations=list(configurations),
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["configurations"] = [cfg.to_dict() for cfg in self.configurations]
        return _strip_none(data)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "STAAnalysisRecord":
        configs_payload = payload.get("configurations") or []
        configs = [STAConfiguration.from_dict(cfg) for cfg in configs_payload]
        return cls(
            analysis_id=payload["analysis_id"],
            source_run=payload.get("source_run", ""),
            population=payload.get("population", ""),
            created_at=payload.get(
                "created_at",
                datetime.now(timezone.utc).isoformat(),
            ),
            version=payload.get("version", "1.0"),
            configurations=configs,
            metadata=payload.get("metadata", {}),
        )


@dataclass
class DecodingAnalysisRecord:
    """Record of a decoding analysis run."""

    analysis_id: str
    source_run: str
    decoder_type: str
    dynamics_model: str
    Q_scale: float
    R_scale: float
    posterior_covariance_trace: Optional[float]
    metrics: Dict[str, float]
    created_at: str
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("analysis_id", self.analysis_id)

    @classmethod
    def new(
        cls,
        analysis_id: str,
        source_run: str,
        decoder_type: str,
        dynamics_model: str,
        Q_scale: float,
        R_scale: float,
        metrics: Dict[str, float],
        posterior_covariance_trace: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DecodingAnalysisRecord":
        created_at = datetime.now(timezone.utc).isoformat()
        return cls(
            analysis_id=analysis_id,
            source_run=source_run,
            decoder_type=decoder_type,
            dynamics_model=dynamics_model,
            Q_scale=Q_scale,
            R_scale=R_scale,
            posterior_covariance_trace=posterior_covariance_trace,
            metrics=metrics,
            created_at=created_at,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return _strip_none(asdict(self))

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DecodingAnalysisRecord":
        return cls(
            analysis_id=payload["analysis_id"],
            source_run=payload.get("source_run", ""),
            decoder_type=payload.get("decoder_type", "unknown"),
            dynamics_model=payload.get("dynamics_model", "unknown"),
            Q_scale=payload.get("Q_scale", 1.0),
            R_scale=payload.get("R_scale", 1.0),
            posterior_covariance_trace=payload.get("posterior_covariance_trace"),
            metrics=payload.get("metrics", {}),
            created_at=payload.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            version=payload.get("version", "1.0"),
            metadata=payload.get("metadata", {}),
        )


class ProjectRegistry:
    """File-backed registry that manages project persistence."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.paths = {
            "protocols": self.root / "protocols",
            "runs": self.root / "runs",
            "analyses": self.root / "sta_analyses",
            "decoding_analyses": self.root / "decoding_analyses",
            "neuron_modules": self.root / "neuron_modules",
        }
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Protocols
    # ------------------------------------------------------------------
    def save_protocol(
        self,
        definition: ProtocolDefinition,
        overwrite: bool = False,
    ) -> Path:
        path = self.paths["protocols"] / f"{definition.protocol_id}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Protocol '{definition.protocol_id}' already exists")
        self._write_json(path, definition.to_dict())
        return path

    def load_protocol(self, protocol_id: str) -> ProtocolDefinition:
        path = self.paths["protocols"] / f"{protocol_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Protocol '{protocol_id}' not found")
        payload = self._read_json(path)
        return ProtocolDefinition.from_dict(payload)

    def list_protocols(self) -> List[ProtocolDefinition]:
        definitions: List[ProtocolDefinition] = []
        for path in sorted(self.paths["protocols"].glob("*.json")):
            payload = self._read_json(path)
            definitions.append(ProtocolDefinition.from_dict(payload))
        return definitions

    # ------------------------------------------------------------------
    # Neuron module manifests
    # ------------------------------------------------------------------
    def save_neuron_manifest(
        self,
        manifest: NeuronModuleManifest,
        overwrite: bool = False,
    ) -> Path:
        path = self.paths["neuron_modules"] / f"{manifest.name}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Neuron manifest '{manifest.name}' already exists")
        self._write_json(path, manifest.to_dict())
        return path

    def load_neuron_manifest(self, name: str) -> NeuronModuleManifest:
        path = self.paths["neuron_modules"] / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Neuron manifest '{name}' not found")
        payload = self._read_json(path)
        return NeuronModuleManifest.from_dict(payload)

    def list_neuron_manifests(self) -> List[NeuronModuleManifest]:
        manifests: List[NeuronModuleManifest] = []
        for path in sorted(self.paths["neuron_modules"].glob("*.json")):
            manifest_payload = self._read_json(path)
            manifests.append(NeuronModuleManifest.from_dict(manifest_payload))
        return manifests

    # ------------------------------------------------------------------
    # Protocol run records
    # ------------------------------------------------------------------
    def save_run(
        self,
        record: ProtocolRunRecord,
        overwrite: bool = False,
    ) -> Path:
        path = self.paths["runs"] / f"{record.run_id}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Run '{record.run_id}' already exists")
        self._write_json(path, record.to_dict())
        return path

    def load_run(self, run_id: str) -> ProtocolRunRecord:
        path = self.paths["runs"] / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Run '{run_id}' not found")
        return ProtocolRunRecord.from_dict(self._read_json(path))

    def list_runs(
        self,
        protocol_id: Optional[str] = None,
    ) -> List[ProtocolRunRecord]:
        records: List[ProtocolRunRecord] = []
        for path in sorted(self.paths["runs"].glob("*.json")):
            record = ProtocolRunRecord.from_dict(self._read_json(path))
            if protocol_id is None or record.protocol_id == protocol_id:
                records.append(record)
        return records

    # ------------------------------------------------------------------
    # STA analyses
    # ------------------------------------------------------------------
    def save_sta_analysis(
        self,
        analysis: STAAnalysisRecord,
        overwrite: bool = False,
    ) -> Path:
        path = self.paths["analyses"] / f"{analysis.analysis_id}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"STA analysis '{analysis.analysis_id}' already exists"
            )
        self._write_json(path, analysis.to_dict())
        return path

    def load_sta_analysis(self, analysis_id: str) -> STAAnalysisRecord:
        path = self.paths["analyses"] / f"{analysis_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"STA analysis '{analysis_id}' not found")
        return STAAnalysisRecord.from_dict(self._read_json(path))

    def list_sta_analyses(
        self,
        source_run: Optional[str] = None,
    ) -> List[STAAnalysisRecord]:
        analyses: List[STAAnalysisRecord] = []
        for path in sorted(self.paths["analyses"].glob("*.json")):
            analysis = STAAnalysisRecord.from_dict(self._read_json(path))
            if source_run is None or analysis.source_run == source_run:
                analyses.append(analysis)
        return analyses

    # ------------------------------------------------------------------
    # Decoding analyses
    # ------------------------------------------------------------------
    def save_decoding_analysis(
        self,
        analysis: DecodingAnalysisRecord,
        overwrite: bool = False,
    ) -> Path:
        path = self.paths["decoding_analyses"] / f"{analysis.analysis_id}.json"
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Decoding analysis '{analysis.analysis_id}' already exists"
            )
        self._write_json(path, analysis.to_dict())
        return path

    def load_decoding_analysis(
        self, analysis_id: str
    ) -> DecodingAnalysisRecord:
        path = self.paths["decoding_analyses"] / f"{analysis_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Decoding analysis '{analysis_id}' not found"
            )
        return DecodingAnalysisRecord.from_dict(self._read_json(path))

    def list_decoding_analyses(
        self,
        source_run: Optional[str] = None,
    ) -> List[DecodingAnalysisRecord]:
        analyses: List[DecodingAnalysisRecord] = []
        for path in sorted(self.paths["decoding_analyses"].glob("*.json")):
            analysis = DecodingAnalysisRecord.from_dict(self._read_json(path))
            if source_run is None or analysis.source_run == source_run:
                analyses.append(analysis)
        return analyses

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _read_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def detect_compute_devices() -> Dict[str, Any]:
    """Return a dictionary describing available compute backends."""

    info: Dict[str, Any] = {"cpu": True, "cpu_count": _safe_cpu_count()}
    if torch is None:
        info.update({"cuda": False, "cuda_count": 0, "mps": False})
        return info

    cuda_available = torch.cuda.is_available()
    info["cuda"] = cuda_available
    info["cuda_count"] = torch.cuda.device_count() if cuda_available else 0

    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    info["mps"] = mps_available
    return info


def suggest_worker_count(batch_size: Optional[int] = None) -> int:
    """Return a conservative worker count for parallel protocol execution.

    Args:
        batch_size: Optional upper bound derived from pending protocol runs.
    """

    cpu_count = _safe_cpu_count()
    if batch_size is not None:
        cpu_count = min(cpu_count, max(1, batch_size))

    if torch is None:
        return max(1, cpu_count)

    if torch.cuda.is_available():
        # Start with two workers per GPU but respect CPU limits.
        gpu_workers = max(1, torch.cuda.device_count()) * 2
        return max(1, min(cpu_count, gpu_workers))

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS benefits from modest parallelism without saturating CPUs.
        return max(1, min(cpu_count, 4))

    return max(1, cpu_count)


def _safe_cpu_count() -> int:
    try:
        import multiprocessing

        return max(1, multiprocessing.cpu_count())
    except Exception:  # pragma: no cover - extremely unlikely
        return 1
