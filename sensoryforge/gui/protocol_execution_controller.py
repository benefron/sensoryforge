"""Execution controller linking Protocol Suite runs to STA extraction."""
from __future__ import annotations

import json
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch
from PyQt5 import QtCore

from GUIs.protocol_backend import ProtocolWorker, RunResult
from utils.project_registry import (
    NeuronModuleManifest,
    ProjectRegistry,
    ProtocolRunRecord,
)

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from GUIs.tabs.mechanoreceptor_tab import (
        MechanoreceptorTab,
        NeuronPopulation,
    )
    from GUIs.tabs.protocol_suite_tab import ProtocolSuiteTab
    from GUIs.tabs.spiking_tab import PopulationConfig, SpikingNeuronTab


class ProtocolExecutionController(QtCore.QObject):
    """Bridge Protocol Suite queue entries with the protocol execution backend."""

    run_completed = QtCore.pyqtSignal(str)
    run_failed = QtCore.pyqtSignal(str)
    batch_finished = QtCore.pyqtSignal()

    def __init__(
        self,
        mechanoreceptor_tab: MechanoreceptorTab,
        spiking_tab: SpikingNeuronTab,
        protocol_tab: ProtocolSuiteTab,
        registry: ProjectRegistry,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._mechanoreceptor_tab = mechanoreceptor_tab
        self._spiking_tab = spiking_tab
        self._protocol_tab = protocol_tab
        self._registry = registry

        self._queue: List[Dict[str, Any]] = []
        self._current_entry: Optional[Dict[str, Any]] = None
        self._current_configs: Dict[str, dict] = {}
        self._worker: Optional[ProtocolWorker] = None
        self._worker_thread: Optional[QtCore.QThread] = None
        self._last_stage: Optional[str] = None
        self._analysis_runs: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    def bootstrap_analysis_runs(self) -> Dict[str, dict]:
        """Load any persisted STA summaries from disk."""

        summaries: Dict[str, dict] = {}
        for record in self._registry.list_runs():
            summary = self._load_run_summary(record.run_id)
            if summary:
                summaries[record.run_id] = summary
        if summaries:
            self._analysis_runs.update(summaries)
        return dict(summaries)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def start_batch(self, entries: Sequence[Dict[str, Any]]) -> None:
        """Begin executing a batch of queued protocol entries."""
        self._log(f"start_batch called with {len(entries)} entries")
        if self._worker is not None:
            self._log("Worker already active; new batch request ignored")
            return
        payload = [dict(entry) for entry in entries if entry]
        if not payload:
            self._log("No entries to queue; start_batch exiting")
            return
        self._queue = payload
        self._last_stage = None
        message = (
            f"[{self._timestamp()}] Queued {len(payload)} protocol(s) " "for execution"
        )
        self._log(message)
        self._start_next()

    def cancel(self) -> None:
        """Request cancellation of the active STA extraction worker."""
        if self._worker is not None:
            self._log("Cancellation requested; signalling worker")
            self._worker.request_stop()

    def analysis_runs(self) -> Dict[str, dict]:
        """Return the cached run summaries generated this session."""

        return dict(self._analysis_runs)

    # ------------------------------------------------------------------
    # Worker orchestration
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Worker orchestration
    # ------------------------------------------------------------------
    def _start_next(self) -> None:
        """Kick off the next queued protocol run if resources are idle."""
        if self._worker is not None:
            return
        if not self._queue:
            self._log(f"[{self._timestamp()}] Protocol queue finished.")
            self._protocol_tab.clear_queue_after_run()
            self.batch_finished.emit()
            return

        entry = self._queue.pop(0)
        self._current_entry = entry
        self._current_configs = {}
        self._last_stage = None
        self._log(
            "Starting next protocol: "
            f"{entry.get('protocol_key')} (row={entry.get('row')})"
        )

        protocol_key = str(entry.get("protocol_key", "")).strip()
        row = int(entry.get("row", -1))

        try:
            spec = self._protocol_tab.protocol_spec(protocol_key)
        except KeyError:
            self._fail_current_entry(f"Unknown protocol '{protocol_key}'", row)
            self._start_next()
            return

        grid_manager = getattr(self._mechanoreceptor_tab, "grid_manager", None)
        if grid_manager is None:
            self._fail_current_entry(
                "Mechanoreceptor grid not configured",
                row,
            )
            self._start_next()
            return

        configs = self._collect_population_configs()
        populations = self._enabled_populations(configs)
        if not populations:
            self._fail_current_entry("No enabled populations", row)
            self._start_next()
            return
        self._current_configs = configs
        pop_label = ", ".join(sorted(configs.keys())) or "Populations"
        self._protocol_tab.set_row_population(row, pop_label)
        self._log(
            f"Resolved populations {list(configs.keys())} for protocol "
            f"{protocol_key}"
        )

        overrides = entry.get("overrides") or {}
        base_dt_ms = self._resolve_dt(overrides)
        device = self._spiking_tab.current_device()

        self._protocol_tab.set_row_status(row, "Running…")
        self._protocol_tab.append_log(
            f"[{self._timestamp()}] {spec.key} → {pop_label}"
            f" (dt={base_dt_ms:.3f} ms, device={device})"
        )

        try:
            worker = ProtocolWorker(
                grid_manager=grid_manager,
                populations=populations,
                population_configs=configs,
                protocol_specs=[spec],
                base_dt_ms=base_dt_ms,
                device=device,
                debug=True,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self._log_exception("Failed to construct ProtocolWorker", exc)
            self._fail_current_entry(str(exc), row)
            self._start_next()
            return
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress_updated.connect(self._handle_progress)
        worker.finished.connect(self._handle_finished)
        worker.failed.connect(self._handle_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._cleanup_worker)
        thread.start()
        self._worker = worker
        self._worker_thread = thread

    def _cleanup_worker(self) -> None:
        self._worker = None
        self._worker_thread = None
        if self._queue:
            QtCore.QTimer.singleShot(0, self._start_next)
        elif self._current_entry is None:
            # Queue drained and state reset
            self._protocol_tab.clear_queue_after_run()
            self.batch_finished.emit()

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------
    def _handle_progress(self, payload: Dict[str, Any]) -> None:
        """Update UI status rows based on incremental worker progress."""
        if self._current_entry is None:
            return
        row = int(self._current_entry.get("row", -1))
        stage = str(payload.get("stage") or "Running…")
        stimulus = payload.get("stimulus")
        population = payload.get("population")
        stop_requested = bool(payload.get("stop_requested"))
        if population and stimulus:
            status = f"{population}: {stimulus}"
        elif population:
            status = f"{population}: {stage}"
        else:
            status = stage
        if stop_requested:
            status = "Stopping after current stimulus…"
        if row >= 0:
            self._protocol_tab.set_row_status(row, status)
        if stage != self._last_stage:
            self._log(f"[{self._timestamp()}] {stage}")
            self._last_stage = stage

    def _handle_finished(self, results: Dict[str, RunResult]) -> None:
        """Persist worker outputs, refresh UI tables, and queue next run."""
        if self._current_entry is None:
            return
        row = int(self._current_entry.get("row", -1))
        protocol_key = str(self._current_entry.get("protocol_key", ""))
        overrides = self._current_entry.get("overrides") or {}

        if not results:
            if row >= 0:
                self._protocol_tab.set_row_status(row, "No results")
            self._log(f"[{self._timestamp()}] {protocol_key} finished without data")
            self._current_entry = None
            self._current_configs = {}
            return

        run_id, summary = self._persist_run(protocol_key, results, overrides)
        if row >= 0:
            self._protocol_tab.set_row_status(row, "Completed")
        self._log(f"[{self._timestamp()}] {protocol_key} complete → run {run_id}")
        self._log(
            f"Run {run_id} produced {len(summary.get('populations', {}))} "
            "population summaries"
        )
        self._protocol_tab.refresh_run_records(self._registry.list_runs())
        self._analysis_runs[run_id] = summary
        
        self.run_completed.emit(run_id)
        self._current_entry = None
        self._current_configs = {}

    def _handle_failed(self, message: str) -> None:
        """Record worker failure and reset controller state."""
        if self._current_entry is None:
            self._log(f"[{self._timestamp()}] ERROR: {message}")
            self.run_failed.emit(message)
            return
        row = int(self._current_entry.get("row", -1))
        if row >= 0:
            self._protocol_tab.set_row_status(row, "Failed")
        self._log(f"[{self._timestamp()}] ERROR: {message}")
        self.run_failed.emit(message)
        self._current_entry = None
        self._current_configs = {}

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _persist_run(
        self,
        protocol_key: str,
        results: Dict[str, RunResult],
        overrides: Dict[str, Any],
    ) -> Tuple[str, dict]:
        """Write artifacts to disk and register the run."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_id = f"{protocol_key}_{timestamp}"
        run_folder = self._registry.paths["runs"] / run_id
        run_folder.mkdir(parents=True, exist_ok=True)

        tensors: Dict[str, str] = {}
        metrics: Dict[str, Any] = {}
        populations_summary: Dict[str, dict] = {}

        for pop_name, result in results.items():
            sanitized = self._sanitize_token(pop_name)
            drive_path = run_folder / f"{sanitized}_drive.pt"
            filtered_path = run_folder / f"{sanitized}_filtered.pt"
            unfiltered_path = run_folder / f"{sanitized}_unfiltered.pt"
            spikes_path = run_folder / f"{sanitized}_spikes.pt"
            stimulus_path = run_folder / f"{sanitized}_stimulus.pt"

            torch.save(result.drive_tensor.cpu(), drive_path)
            torch.save(result.filtered_tensor.cpu(), filtered_path)
            torch.save(result.unfiltered_tensor.cpu(), unfiltered_path)
            torch.save(result.spike_tensor.cpu(), spikes_path)
            if result.stimulus_tensor is not None:
                torch.save(result.stimulus_tensor.cpu(), stimulus_path)
                tensors[f"{sanitized}_stimulus"] = self._relative_path(stimulus_path)

            tensors[f"{sanitized}_drive"] = self._relative_path(drive_path)
            tensors[f"{sanitized}_filtered"] = self._relative_path(filtered_path)
            tensors[f"{sanitized}_unfiltered"] = self._relative_path(unfiltered_path)
            tensors[f"{sanitized}_spikes"] = self._relative_path(spikes_path)

            if result.spike_counts is not None:
                metrics[f"{sanitized}_spike_counts"] = result.spike_counts.tolist()
            if result.valid_mask is not None:
                metrics[f"{sanitized}_valid"] = result.valid_mask.astype(
                    bool
                ).tolist()

            populations_summary[pop_name] = {
                "neuron_type": result.neuron_type,
                "dt_ms": result.dt_ms,
                "spike_count": int(result.spike_counts.sum()) if result.spike_counts is not None else 0,
            }

        manifests = self._build_manifests(self._current_configs)
        summary_path = self._summary_path(run_id)
        stimulus_reference = str(
            overrides.get("stimulus_reference")
            or overrides.get("stimulus_key")
            or "protocol_suite_capture"
        )
        record = ProtocolRunRecord.new(
            run_id=run_id,
            protocol_id=protocol_key,
            stimulus_reference=stimulus_reference,
            neuron_modules=manifests,
            tensors=tensors,
            metrics=metrics,
            metadata={
                "device": str(self._spiking_tab.current_device()),
                "populations": list(results.keys()),
                "overrides": overrides,
                "summary_file": self._relative_path(summary_path),
            },
        )
        record_path = self._registry.save_run(record)
        manifest_message = (
            f"[{self._timestamp()}] Saved run manifest → "
            f"{self._relative_path(record_path)}"
        )
        self._log(manifest_message)
        self._log(
            "["  # build string in parts to stay within line limits
            f"{self._timestamp()}] Saved summary to "
            f"{self._relative_path(summary_path)}"
        )

        summary = {
            "protocol_id": protocol_key,
            "created_at": datetime.utcnow().isoformat(),
            "populations": populations_summary,
        }
        self._write_summary(summary_path, summary)
        return run_id, summary

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------
    def _collect_population_configs(self) -> Dict[str, dict]:
        """Aggregate enabled neuron population configs from the GUI tabs."""
        configs: Dict[str, dict] = {}
        raw_configs = getattr(self._spiking_tab, "population_configs", {}) or {}
        for name, config in raw_configs.items():
            if config is None or not getattr(config, "enabled", True):
                continue
            configs[name] = self._serialize_population_config(config)
        if configs:
            return configs
        # Fallback: synthesize defaults from mechanoreceptor populations
        for population in getattr(
            self._mechanoreceptor_tab,
            "populations",
            [],
        ):
            name = getattr(population, "name", None)
            if not name:
                continue
            configs[name] = {
                "model": "Izhikevich",
                "filter_method": "none",
                "input_gain": 1.0,
                "noise_std": 0.0,
                "model_params": {},
                "filter_params": {},
                "neuron_type": getattr(population, "neuron_type", "SA"),
            }
        return configs

    def _enabled_populations(self, configs: Dict[str, dict]) -> List[NeuronPopulation]:
        """Filter mechanoreceptor populations to those with active configs."""
        populations: List[NeuronPopulation] = []
        for population in getattr(
            self._mechanoreceptor_tab,
            "populations",
            [],
        ):
            name = getattr(population, "name", None)
            if not name or name not in configs:
                continue
            populations.append(population)
        return populations

    def _serialize_population_config(self, config: PopulationConfig) -> Dict[str, Any]:
        """Convert a population configuration dataclass into a JSON payload."""
        model_params = self._coerce_mapping(getattr(config, "model_params", {}))
        filter_params = self._coerce_mapping(getattr(config, "filter_params", {}))
        return {
            "model": getattr(config, "model", "Izhikevich"),
            "filter_method": getattr(config, "filter_method", "none"),
            "input_gain": getattr(config, "input_gain", 1.0),
            "noise_std": getattr(config, "noise_std", 0.0),
            "model_params": model_params,
            "filter_params": filter_params,
            "neuron_type": getattr(config, "neuron_type", "SA"),
        }

    def _build_manifests(self, configs: Dict[str, dict]) -> List[NeuronModuleManifest]:
        """Generate manifest records describing the neuron populations used."""
        manifests: List[NeuronModuleManifest] = []
        for name, cfg in configs.items():
            manifests.append(
                NeuronModuleManifest(
                    name=self._sanitize_token(name),
                    model=str(cfg.get("model", "Izhikevich")),
                    filter=cfg.get("filter_method"),
                    parameters=dict(cfg.get("model_params", {})),
                    file="",
                    tags=[str(cfg.get("neuron_type", ""))],
                )
            )
        return manifests

    @staticmethod
    def _coerce_mapping(candidate: Any) -> Dict[str, Any]:
        """Best-effort conversion of user-provided structures to dicts."""
        if isinstance(candidate, dict):
            return dict(candidate)
        if candidate is None:
            return {}
        try:
            return dict(candidate)
        except Exception:
            return {}

    def _summary_path(self, run_id: str) -> Path:
        """Return the on-disk path that should hold the STA summary file."""
        return self._registry.paths["runs"] / run_id / "analysis_summary.json"

    def _write_summary(self, path: Path, summary: dict) -> None:
        """Persist a JSON summary, tolerating filesystem errors for logging."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
        except OSError as exc:
            message = f"[{self._timestamp()}] Failed to save STA summary: {exc}"
            self._log(message)

    def _load_run_summary(self, run_id: str) -> Optional[dict]:
        """Load a cached summary from disk if one exists for the run."""
        path = self._summary_path(run_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            message = (
                f"[{self._timestamp()}] Failed to read STA summary for "
                f"{run_id}: {exc}"
            )
            self._log(message)
            return None

    def _log(self, message: str) -> None:
        """Emit a message to both the GUI log and the terminal."""

        try:
            self._protocol_tab.append_log(message)
        except Exception:
            print(f"[ProtocolExecutionController] {message}", flush=True)

    def _log_exception(self, context: str, exc: BaseException) -> None:
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._log(f"{context}: {exc}\n{details}")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _fail_current_entry(self, message: str, row: int) -> None:
        """Mark the active queue entry as failed and emit a log entry."""
        if row >= 0:
            self._protocol_tab.set_row_status(row, f"Failed – {message}")
        self._protocol_tab.append_log(f"[{self._timestamp()}] ERROR: {message}")
        self.run_failed.emit(message)
        self._current_entry = None
        self._current_configs = {}

    def _resolve_dt(self, overrides: Dict[str, Any]) -> float:
        """Resolve the time-step override for STA extraction, if valid."""
        value = overrides.get("dt_ms")
        if value is not None:
            try:
                dt = float(value)
                if dt > 0:
                    return dt
            except (TypeError, ValueError):
                pass
        return float(self._spiking_tab.current_dt_ms())

    @staticmethod
    def _sanitize_token(value: str) -> str:
        """Normalize names for filesystem-safe tensor filenames."""
        token = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
        return token.strip("_") or "population"

    def _protocol_reference(self, protocol_id: str) -> str:
        """Return the relative path to a protocol manifest if it exists."""
        path = self._registry.paths["protocols"] / f"{protocol_id}.json"
        if path.exists():
            return self._relative_path(path)
        return ""

    def _relative_path(self, path: Path) -> str:
        """Render ``path`` relative to the registry root when possible."""
        try:
            return str(path.relative_to(self._registry.root))
        except ValueError:
            return str(path)

    @staticmethod
    def _timestamp() -> str:
        """Return a human-readable timestamp for log entries."""
        return datetime.now().strftime("%H:%M:%S")
