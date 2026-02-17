"""Shared protocol execution backend utilities.

This module exposes the data containers and background worker that power
protocol execution.
"""

from __future__ import annotations

import inspect
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PyQt5 import QtCore

try:  # noqa: E402 - optional dependency for memory snapshots
    import resource
except ImportError:  # pragma: no cover - platform without resource module
    resource = None

# Ensure repository root on sys.path for package imports when run as a script
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sensoryforge.core.grid import GridManager  # noqa: E402
from sensoryforge.stimuli.stimulus import (  # noqa: E402
    edge_stimulus_torch,
    gaussian_pressure_torch,
    point_pressure_torch,
)
from sensoryforge.filters.sa_ra import RAFilterTorch, SAFilterTorch  # noqa: E402
from sensoryforge.neurons.adex import AdExNeuronTorch  # noqa: E402
from sensoryforge.neurons.fa import FANeuronTorch  # noqa: E402
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch  # noqa: E402
from sensoryforge.neurons.mqif import MQIFNeuronTorch  # noqa: E402
from sensoryforge.neurons.sa import SANeuronTorch  # noqa: E402
from sensoryforge.gui.filter_utils import normalize_filter_method  # noqa: E402


DEFAULT_DT_MS = 1.0
GAUSSIAN_SIGMA_FRACTION = 0.25


@dataclass
class ProtocolSpec:
    """Metadata describing a single stimulus protocol."""

    key: str
    title: str
    description: str


@dataclass
class StimulusPacket:
    """Container for a single stimulus instance within a protocol."""

    protocol_key: str
    protocol_name: str
    label: str
    frames: torch.Tensor  # [time, rows, cols]
    dt_ms: float
    meta: Dict[str, float]


@dataclass
class RunResult:
    """Holds execution data for a population."""

    population_name: str
    neuron_type: str
    dt_ms: float
    spike_counts: Optional[np.ndarray]
    valid_mask: Optional[np.ndarray]
    tau_ms: Optional[float] = None
    drive_tensor: Optional[torch.Tensor] = None
    filtered_tensor: Optional[torch.Tensor] = None
    unfiltered_tensor: Optional[torch.Tensor] = None
    spike_tensor: Optional[torch.Tensor] = None
    stimulus_tensor: Optional[torch.Tensor] = None


class ProtocolWorker(QtCore.QObject):
    """Background worker that runs stimulus protocols."""

    progress_updated = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        grid_manager: GridManager,
        populations: Sequence,
        population_configs: Dict[str, dict],
        protocol_specs: Sequence[ProtocolSpec],
        base_dt_ms: float,
        device: torch.device,
        debug: bool = False,
        perform_fit: bool = False,
    ) -> None:
        super().__init__()
        self._grid_manager = grid_manager
        self._populations = list(populations)
        self._population_configs = dict(population_configs)
        self._protocol_specs = list(protocol_specs)
        self._packet_dt_ms = max(float(base_dt_ms), 1e-6)
        self._device = device
        self._stop_requested = False
        self._last_packets: List[StimulusPacket] = []
        self._perform_fit = bool(perform_fit)  # Resolves ReviewFinding#C1
        env_flag = os.getenv("PROTOCOL_WORKER_DEBUG", "")
        env_debug = str(env_flag).strip().lower() in {"1", "true", "yes", "on"}
        self._debug_enabled = bool(debug) or env_debug
        if self._debug_enabled:
            self._debug(
                "ProtocolWorker debug logging enabled",
                via_env=env_debug,
                population_count=len(self._populations),
                protocol_count=len(self._protocol_specs),
                device=str(self._device),
            )

    def request_stop(self) -> None:
        self._stop_requested = True

    @property
    def last_packets(self) -> List[StimulusPacket]:
        """Return a copy of the packets generated in the most recent run."""

        return list(self._last_packets)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def _debug(self, message: str, **fields: object) -> None:
        if not getattr(self, "_debug_enabled", False):
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        details = " ".join(
            f"{key}={self._format_debug_value(value)}" for key, value in fields.items()
        )
        if details:
            print(f"[STAWorker {timestamp}] {message} | {details}", flush=True)
        else:
            print(f"[STAWorker {timestamp}] {message}", flush=True)

    def _format_debug_value(self, value: object) -> str:
        if isinstance(value, torch.Tensor):
            shape = tuple(value.shape)
            return f"Tensor(shape={shape}, dtype={value.dtype})"
        if isinstance(value, (list, tuple)):
            preview = ", ".join(str(v) for v in list(value)[:5])
            if len(value) > 5:
                preview += ", …"
            return f"[{preview}]"
        if isinstance(value, dict):
            keys = list(value.keys())
            preview = ", ".join(str(k) for k in keys[:5])
            if len(keys) > 5:
                preview += ", …"
            return f"dict(keys={preview})"
        return str(value)

    def _memory_snapshot_mb(self) -> Optional[float]:
        if resource is None:
            return None
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
        except Exception:  # pragma: no cover - fallback if unsupported
            return None
        rss = getattr(usage, "ru_maxrss", 0)
        if rss <= 0:
            return None
        if sys.platform == "darwin":
            rss_bytes = float(rss)
        else:
            rss_bytes = float(rss) * 1024.0
        return rss_bytes / (1024.0 * 1024.0)

    def run(self) -> None:  # pragma: no cover - executed in worker thread
        self._debug(
            "Worker run invoked",
            populations=[getattr(p, "name", "Population") for p in self._populations],
            protocols=[spec.key for spec in self._protocol_specs],
            base_dt_ms=self._packet_dt_ms,
            device=str(self._device),
            perform_fit=self._perform_fit,
        )
        if self._grid_manager is None:
            self.failed.emit("Grid manager unavailable for STA extraction.")
            return
        if not self._populations:
            self.failed.emit("No populations available for STA extraction.")
            return
        torch.set_grad_enabled(False)
        self._last_packets = []
        results: Dict[str, RunResult] = {}
        try:
            packets = self._build_packets()
            self._debug("Stimulus packets built", packet_count=len(packets))
            self._last_packets = list(packets)
            if not packets:
                raise RuntimeError("STA protocol suite produced no stimuli.")
            self._debug(
                "Executing packets",
                total_jobs=len(self._populations) * len(packets),
            )
            results = self._execute_packets(packets)
        except Exception as exc:  # pragma: no cover - safeguard
            self._debug("Worker execution failed", error=str(exc))
            self.failed.emit(str(exc))
            return
        finally:
            torch.set_grad_enabled(True)
        self.finished.emit(results)

    # ------------------------------------------------------------------
    # Stimulus packet generation
    # ------------------------------------------------------------------
    def _build_packets(self) -> List[StimulusPacket]:
        packets: List[StimulusPacket] = []
        for spec in self._protocol_specs:
            if spec.key == "slow_gaussian":
                packets.extend(self._pack_slow_gaussians(spec))
            elif spec.key == "fast_gaussian":
                packets.extend(self._pack_fast_gaussians(spec))
            elif spec.key == "moving_gaussian":
                packets.extend(self._pack_moving_gaussians(spec))
            elif spec.key == "macro_gaussian":
                packets.extend(self._pack_macro_gaussians(spec))
            elif spec.key == "edge_steps":
                packets.extend(self._pack_edge_steps(spec))
            elif spec.key == "point_pulses":
                packets.extend(self._pack_point_pulses(spec))
            elif spec.key == "large_disk_pulses":
                packets.extend(self._pack_large_disk_pulses(spec))
            elif spec.key == "texture_noise":
                packets.extend(self._pack_texture_sequences(spec))
            elif spec.key == "shear_wave":
                packets.extend(self._pack_shear_sweeps(spec))
            elif spec.key == "rotating_edge":
                packets.extend(self._pack_rotating_edges(spec))
            elif spec.key == "center_surround":
                packets.extend(self._pack_center_surround(spec))
        return packets

    def _execute_packets(
        self, packets: List[StimulusPacket]
    ) -> Dict[str, RunResult]:
        if not packets:
            return {}
        total_jobs = max(len(self._populations) * len(packets), 1)
        results: Dict[str, RunResult] = {}
        processed = 0
        start_time = time.perf_counter()
        self._debug(
            "Executing packets",
            population_count=len(self._populations),
            packet_count=len(packets),
            total_jobs=total_jobs,
        )
        for population in self._populations:
            if self._stop_requested:
                break
            pop_name = str(getattr(population, "name", "Population"))
            # Use module or flat_module (Poisson/composite use flat_module)
            module = getattr(population, "module", None) or getattr(
                population, "flat_module", None
            )
            if module is None and hasattr(population, "instantiate"):
                if self._grid_manager is None:
                    raise RuntimeError(
                        "Grid manager unavailable for population instantiation."
                    )
                population.instantiate(self._grid_manager)
                module = getattr(population, "module", None) or getattr(
                    population, "flat_module", None
                )
            if module is None:
                raise RuntimeError(
                    "Innervation module unavailable for population " f"'{pop_name}'."
                )
            self._debug(
                "Population simulation starting",
                population=pop_name,
                module_type=type(module).__name__,
                packet_count=len(packets),
            )
            original_device = module.innervation_weights.device
            module = module.to(self._device)
            module.eval()
            config = self._prepare_population_config(population, pop_name)
            drive_segments: List[torch.Tensor] = []
            filtered_segments: List[torch.Tensor] = []
            raw_segments: List[torch.Tensor] = []
            spike_segments: List[torch.Tensor] = []
            stimulus_segments: List[torch.Tensor] = []
            dt_ms: Optional[float] = None
            tau_ms: Optional[float] = None
            try:
                for packet in packets:
                    if self._stop_requested:
                        break
                    self._debug(
                        "Simulating packet",
                        population=pop_name,
                        packet_label=packet.label,
                        protocol=packet.protocol_key,
                        frames_shape=tuple(packet.frames.shape),
                    )
                    (
                        drive_seg,
                        spike_seg,
                        packet_dt,
                        packet_tau,
                        filtered_seg,
                        raw_seg,
                    ) = self._simulate_packet(
                        module,
                        population,
                        config,
                        packet,
                    )
                    drive_segments.append(drive_seg)
                    filtered_segments.append(filtered_seg)
                    raw_segments.append(raw_seg)
                    spike_segments.append(spike_seg)
                    stim_frames = packet.frames.to(
                        device=self._device,
                        dtype=torch.float32,
                    )
                    if stim_frames.ndim == 3:
                        stim_frames = stim_frames.unsqueeze(0)
                    elif stim_frames.ndim != 4:
                        raise RuntimeError("Unexpected stimulus tensor shape.")
                    stimulus_segments.append(stim_frames.cpu())
                    if dt_ms is None:
                        dt_ms = packet_dt
                    elif abs(packet_dt - dt_ms) > 1e-6:
                        raise RuntimeError(
                            "Inconsistent dt across protocols; "
                            "ensure stimuli share the same sampling interval."
                        )
                    if tau_ms is None and packet_tau is not None:
                        tau_ms = packet_tau
                    processed += 1
                    progress = processed / total_jobs
                    eta = self._estimate_eta(start_time, processed, total_jobs)
                    self.progress_updated.emit(
                        {
                            "stage": f"Simulating {pop_name}",
                            "population": pop_name,
                            "stimulus": packet.label,
                            "progress": progress,
                            "eta": eta,
                            "stop_requested": self._stop_requested,
                        }
                    )
                if not drive_segments or not spike_segments:
                    continue
                
                # Concatenate segments
                drive_tensor = torch.cat(drive_segments, dim=1)
                filtered_tensor = torch.cat(filtered_segments, dim=1)
                raw_tensor = torch.cat(raw_segments, dim=1)
                spike_tensor = torch.cat(spike_segments, dim=1)
                stimulus_tensor = (
                    torch.cat(stimulus_segments, dim=1) if stimulus_segments else None
                )
                
                # Clear buffers
                drive_segments.clear()
                filtered_segments.clear()
                raw_segments.clear()
                spike_segments.clear()
                stimulus_segments.clear()

                neuron_type = getattr(population, "neuron_type", None)
                
                results[pop_name] = RunResult(
                    population_name=pop_name,
                    neuron_type=str(neuron_type or ""),
                    dt_ms=float(dt_ms or self._packet_dt_ms),
                    spike_counts=spike_tensor.sum(dim=1).cpu().numpy(),
                    valid_mask=None,
                    tau_ms=tau_ms,
                    drive_tensor=drive_tensor,
                    filtered_tensor=filtered_tensor,
                    unfiltered_tensor=raw_tensor,
                    spike_tensor=spike_tensor,
                    stimulus_tensor=stimulus_tensor,
                )
                self._debug(
                    "Population simulation complete",
                    population=pop_name,
                    total_time_steps=drive_tensor.shape[1],
                    spike_count=int(spike_tensor.sum().item()),
                )
            finally:
                module.to(original_device)
        if self._stop_requested:
            self.progress_updated.emit(
                {
                    "stage": "Cancelled",
                    "progress": processed / total_jobs,
                    "stop_requested": True,
                }
            )
        self._debug("Packet execution finished", population_count=len(results))
        return results

    def _prepare_population_config(
        self, population, pop_name: str
    ) -> Dict[str, object]:
        raw = dict(self._population_configs.get(pop_name, {}))
        return {
            "model": raw.get("model", "Izhikevich"),
            "filter_method": raw.get("filter_method", "auto"),
            "input_gain": raw.get("input_gain", 1.0),
            "noise_std": raw.get("noise_std", 0.0),
            "model_params": dict(raw.get("model_params", {})),
            "filter_params": dict(raw.get("filter_params", {})),
            "neuron_type": getattr(population, "neuron_type", "SA"),
        }

    def _resolve_dt_ms(self, config: Dict[str, object], packet_dt_ms: float) -> float:
        params = config.get("model_params", {})
        dt_param = params.get("dt") if isinstance(params, dict) else None
        dt_val = self._coerce_float(dt_param, default=packet_dt_ms)
        dt_val = dt_val if dt_param is not None else packet_dt_ms
        dt_val = max(float(dt_val), 1e-6)
        return dt_val

    def _extract_tau_ms(
        self, neuron_model: object, config: Dict[str, object]
    ) -> Optional[float]:
        params = config.get("model_params", {})
        if isinstance(params, dict):
            for key in ("tau_ms", "tau", "tau_m", "tau_mem", "tau_leak"):
                if key in params:
                    tau_val = self._coerce_float(params[key], default=0.0)
                    if tau_val > 0:
                        return tau_val
        attr_candidates = (
            "tau_ms",
            "tau_m",
            "tau_mem",
            "tau",
            "tau_leak",
        )
        for attr in attr_candidates:
            if hasattr(neuron_model, attr):
                value = getattr(neuron_model, attr)
                tau_val = self._coerce_float(value, default=0.0)
                if tau_val > 0:
                    return tau_val
        if hasattr(neuron_model, "tau_s"):
            tau_seconds = self._coerce_float(
                getattr(neuron_model, "tau_s"),
                default=0.0,
            )
            if tau_seconds > 0:
                return tau_seconds * 1000.0
        if self._tau_override is not None:
            return float(self._tau_override)
        return None

    def _simulate_packet(
        self,
        module,
        population,
        config: Dict[str, object],
        packet: StimulusPacket,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        float,
        Optional[float],
        torch.Tensor,
        torch.Tensor,
    ]:
        device = self._device
        pop_name = str(getattr(population, "name", "Population"))
        packet_label = getattr(packet, "label", packet.protocol_key)
        frames = packet.frames.to(device=device, dtype=torch.float32)
        if frames.ndim == 3:
            stimuli = frames.unsqueeze(0)
        elif frames.ndim == 4:
            stimuli = frames
        else:
            raise RuntimeError("Unexpected stimulus tensor shape.")
        if self._debug_enabled:
            self._debug(
                "Packet stimuli prepared",
                population=pop_name,
                packet_label=packet_label,
                frames_shape=tuple(frames.shape),
                stimuli_shape=tuple(stimuli.shape),
                packet_dt_ms=packet.dt_ms,
            )
        dt_ms = self._resolve_dt_ms(config, packet.dt_ms)
        if self._debug_enabled:
            self._debug(
                "Resolved timestep",
                population=pop_name,
                packet_label=packet_label,
                dt_ms=dt_ms,
            )
        neuron_drive = module(stimuli)
        if neuron_drive.ndim == 2:
            neuron_drive = neuron_drive.unsqueeze(1)
        raw_drive = neuron_drive
        if self._debug_enabled:
            stats = {
                "shape": tuple(neuron_drive.shape),
                "min": float(neuron_drive.min().item()),
                "max": float(neuron_drive.max().item()),
                "mean": float(neuron_drive.mean().item()),
            }
            self._debug(
                "Innervation module output",
                population=pop_name,
                packet_label=packet_label,
                **stats,
            )
        filtered = self._apply_filter(
            neuron_drive,
            str(config.get("neuron_type", "SA")),
            config,
            dt_ms,
        )
        filtered_drive = filtered
        gain = self._coerce_float(config.get("input_gain", 1.0), default=1.0)
        drive = filtered * gain
        noise_std = self._coerce_float(
            config.get("noise_std", 0.0),
            default=0.0,
        )
        if noise_std > 0.0:
            drive = drive + torch.randn_like(drive) * noise_std
        drive = drive.float()
        if self._debug_enabled:
            drive_stats = {
                "shape": tuple(drive.shape),
                "min": float(drive.min().item()),
                "max": float(drive.max().item()),
                "mean": float(drive.mean().item()),
                "gain": gain,
                "noise_std": noise_std,
            }
            self._debug(
                "Drive prepared for neuron model",
                population=pop_name,
                packet_label=packet_label,
                **drive_stats,
            )
        neuron_model = self._create_neuron_model(
            config,
            dt_ms,
        )
        neuron_model = neuron_model.to(device)
        neuron_model.eval()
        if self._debug_enabled:
            self._debug(
                "Neuron model instantiated",
                population=pop_name,
                packet_label=packet_label,
                model_type=type(neuron_model).__name__,
                dt_ms=dt_ms,
            )
        _, spikes = neuron_model(drive)
        steps = drive.shape[1]
        if spikes.ndim != 3:
            raise RuntimeError("Neuron model returned unexpected spike tensor.")
        if spikes.shape[1] < steps:
            drive = drive[:, : spikes.shape[1], :]
            steps = spikes.shape[1]
        elif spikes.shape[1] > steps:
            spikes = spikes[:, :steps, :]
        if self._debug_enabled:
            spike_stats = {
                "shape": tuple(spikes.shape),
                "sum": int(spikes.sum().item()),
                "steps": steps,
            }
            self._debug(
                "Neuron model response",
                population=pop_name,
                packet_label=packet_label,
                **spike_stats,
            )
            if device.type == "cuda" and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(device)
                self._debug(
                    "CUDA memory snapshot",
                    population=pop_name,
                    packet_label=packet_label,
                    bytes_allocated=int(allocated),
                )
        drive_cpu = drive.detach().cpu()
        spikes_cpu = spikes.detach().cpu().to(dtype=torch.bool)
        filtered_cpu = filtered_drive.detach().cpu()
        raw_cpu = raw_drive.detach().cpu()
        neuron_model.to(torch.device("cpu"))
        tau_ms = self._extract_tau_ms(neuron_model, config)
        if self._debug_enabled:
            self._debug(
                "Packet simulation completed",
                population=pop_name,
                packet_label=packet_label,
                tau_ms=tau_ms,
            )
        return (
            drive_cpu,
            spikes_cpu,
            float(dt_ms),
            tau_ms,
            filtered_cpu,
            raw_cpu,
        )

    def _apply_filter(
        self,
        inputs: torch.Tensor,
        neuron_type: str,
        config: Dict[str, object],
        dt_ms: float,
    ) -> torch.Tensor:
        method = self._resolve_filter_method(
            config.get("filter_method"),
            neuron_type,
        )
        self._debug(
            "Applying filter",
            method=method,
            neuron_type=neuron_type,
            input_shape=tuple(inputs.shape),
            dt_ms=dt_ms,
        )
        if method == "none":
            self._debug("Filter disabled", method=method)
            return inputs
        params = self._sanitize_parameters(dict(config.get("filter_params", {})))
        params.setdefault("dt", dt_ms)
        if method == "sa":
            kwargs = self._filter_kwargs(SAFilterTorch, params)
            filter_module = SAFilterTorch(**kwargs).to(self._device)
        else:
            kwargs = self._filter_kwargs(RAFilterTorch, params)
            filter_module = RAFilterTorch(**kwargs).to(self._device)
        filter_module.eval()
        if inputs.ndim == 3:
            result = filter_module(inputs, reset_states=True)
        else:
            result = filter_module(inputs.squeeze(1), reset_states=True)
            result = result.unsqueeze(1)
        if self._debug_enabled:
            stats = {
                "shape": tuple(result.shape),
                "min": float(result.min().item()),
                "max": float(result.max().item()),
                "mean": float(result.mean().item()),
            }
            self._debug("Filter output", method=method, **stats)
        return result

    def _create_neuron_model(
        self,
        config: Dict[str, object],
        dt_ms: float,
    ):
        model_name = str(config.get("model", "Izhikevich")).strip().lower()
        params = self._sanitize_parameters(dict(config.get("model_params", {})))
        noise_std = self._coerce_float(
            config.get("noise_std", 0.0),
            default=0.0,
        )
        dt_value = max(float(dt_ms), self._packet_dt_ms, 1e-6)

        def instantiate(model_cls):
            kwargs = self._filter_kwargs(model_cls, params)
            try:
                signature = inspect.signature(model_cls.__init__)
            except (ValueError, TypeError):
                signature = None
            if signature is None or "dt" in getattr(
                signature,
                "parameters",
                {},
            ):
                kwargs.setdefault("dt", dt_value)
            if signature is None or "noise_std" in getattr(signature, "parameters", {}):
                kwargs.setdefault("noise_std", noise_std)
            return model_cls(**kwargs)

        if model_name == "adex":
            return instantiate(AdExNeuronTorch)
        if model_name == "mqif":
            return instantiate(MQIFNeuronTorch)
        if model_name == "fa":
            return instantiate(FANeuronTorch)
        if model_name == "sa":
            return instantiate(SANeuronTorch)
        return instantiate(IzhikevichNeuronTorch)

    @staticmethod
    def _filter_kwargs(callable_obj, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            signature = inspect.signature(callable_obj.__init__)
        except (ValueError, TypeError):
            return dict(params)
        valid: Dict[str, Any] = {}
        for name in signature.parameters:
            if name == "self":
                continue
            if name in params:
                valid[name] = params[name]
        return valid

    @staticmethod
    def _sanitize_parameters(params: Dict[str, object]) -> Dict[str, object]:
        def convert(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, (list, tuple)):
                return type(value)(convert(v) for v in value)
            return value

        return {key: convert(val) for key, val in params.items()}

    @staticmethod
    def _coerce_float(value: object, default: float = 0.0) -> float:
        try:
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return float(value.item())
                return float(value.mean().item())
            if isinstance(value, np.generic):
                return float(value.item())
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _resolve_filter_method(
        method: Optional[object], neuron_type: Optional[str]
    ) -> str:
        if method is None:
            resolved = "auto"
        else:
            resolved = str(method).strip().lower() or "auto"
        if resolved in {"auto", "default"}:
            resolved = "ra" if (neuron_type or "SA").upper() == "RA" else "sa"
        return normalize_filter_method(resolved, neuron_type)

    @staticmethod
    def _estimate_eta(start_time: float, completed: int, total: int) -> float:
        if completed <= 0 or total <= 0 or completed >= total:
            return 0.0
        elapsed = max(time.perf_counter() - start_time, 0.0)
        average = elapsed / completed if completed else 0.0
        remaining = total - completed
        return max(average * remaining, 0.0)

    def _grid_bounds(self) -> Tuple[float, float, float, float]:
        xx = self._grid_manager.xx
        yy = self._grid_manager.yy
        return (
            float(xx.min().item()),
            float(xx.max().item()),
            float(yy.min().item()),
            float(yy.max().item()),
        )

    def _canonical_centers(self) -> List[Tuple[float, float]]:
        x_min, x_max, y_min, y_max = self._grid_bounds()
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        return [
            (x_mid, y_mid),
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_mid),
            (x_max, y_mid),
            (x_mid, y_min),
            (x_mid, y_max),
        ]

    def _pack_slow_gaussians(self, spec: ProtocolSpec) -> List[StimulusPacket]:
        xx = self._grid_manager.xx.to(self._device)
        yy = self._grid_manager.yy.to(self._device)
        x_min, x_max, y_min, y_max = self._grid_bounds()
        span = min(x_max - x_min, y_max - y_min)
        sigma = max(GAUSSIAN_SIGMA_FRACTION * span, 0.12)
        profile = self._temporal_profile(
            250,
            500,
            250,
            80,
            device=self._device,
        )
        packets: List[StimulusPacket] = []
        for amplitude in (0.6, 0.9, 1.2):
            for cx, cy in self._canonical_centers():
                base = gaussian_pressure_torch(
                    xx,
                    yy,
                    center_x=cx,
                    center_y=cy,
                    amplitude=amplitude,
                    sigma=sigma,
                )
                frames = profile[:, None, None] * base.unsqueeze(0)
                packets.append(
                    StimulusPacket(
                        protocol_key=spec.key,
                        protocol_name=spec.title,
                        label=(
                            f"Slow Gaussian amp={amplitude:.1f} "
                            f"@ ({cx:.2f},{cy:.2f})"
                        ),
                        frames=frames.cpu(),
                        dt_ms=self._packet_dt_ms,
                        meta={"amplitude": amplitude, "sigma": sigma},
                    )
                )
        return packets

    def _pack_fast_gaussians(self, spec: ProtocolSpec) -> List[StimulusPacket]:
        xx = self._grid_manager.xx.to(self._device)
        yy = self._grid_manager.yy.to(self._device)
        x_min, x_max, y_min, y_max = self._grid_bounds()
        span = min(x_max - x_min, y_max - y_min)
        packets: List[StimulusPacket] = []
        for sigma_scale in (0.16, 0.22):
            sigma = max(sigma_scale * span, 0.08)
            profile = self._temporal_profile(
                30,
                60,
                30,
                40,
                device=self._device,
            )
            for amplitude in (0.5, 0.8, 1.1):
                for cx, cy in self._canonical_centers():
                    base = gaussian_pressure_torch(
                        xx,
                        yy,
                        center_x=cx,
                        center_y=cy,
                        amplitude=amplitude,
                        sigma=sigma,
                    )
                    frames = profile[:, None, None] * base.unsqueeze(0)
                    packets.append(
                        StimulusPacket(
                            protocol_key=spec.key,
                            protocol_name=spec.title,
                            label=(
                                f"Rapid Gaussian amp={amplitude:.1f} "
                                f"σ={sigma:.2f} @ ({cx:.2f},{cy:.2f})"
                            ),
                            frames=frames.cpu(),
                            dt_ms=self._packet_dt_ms,
                            meta={"amplitude": amplitude, "sigma": sigma},
                        )
                    )

        return packets

    def _pack_shear_sweeps(self, spec: ProtocolSpec) -> List[StimulusPacket]:
        if self._grid_manager is None:
            return []
        xx = self._grid_manager.xx.to(self._device)
        yy = self._grid_manager.yy.to(self._device)
        x_span = max(float(xx.max() - xx.min()), 1e-6)
        y_span = max(float(yy.max() - yy.min()), 1e-6)
        dt_ms = max(self._packet_dt_ms, 0.2)
        duration_ms = 420.0
        steps = max(int(duration_ms / dt_ms), 1)
        time_s = (
            torch.arange(steps, device=self._device, dtype=xx.dtype) * dt_ms / 1000.0
        )
        amplitude_options = (0.45, 0.75, 1.05)
        temporal_freqs = (8.0, 14.0)
        spatial_freqs = (1.0 / x_span, 1.5 / y_span)
        directions = (
            ("X shear", xx / x_span),
            ("Y shear", yy / y_span),
            (
                "Diag shear",
                (xx + yy) / math.sqrt(x_span * x_span + y_span * y_span),
            ),
        )
        packets: List[StimulusPacket] = []
        for label_base, grid in directions:
            norm_grid = grid.clamp(-1.0, 1.0)
            for temp_freq in temporal_freqs:
                for spatial_freq in spatial_freqs:
                    phase = (
                        2.0
                        * math.pi
                        * (
                            temp_freq * time_s[:, None, None]
                            + spatial_freq * norm_grid[None, :, :]
                        )
                    )
                    waveform = torch.sin(phase)
                    envelope = torch.hann_window(
                        steps,
                        periodic=False,
                        device=self._device,
                        dtype=xx.dtype,
                    )
                    envelope = envelope[:, None, None]
                    for amplitude in amplitude_options:
                        frames = (0.5 * (waveform + 1.0) * envelope) * amplitude
                        packets.append(
                            StimulusPacket(
                                protocol_key=spec.key,
                                protocol_name=spec.title,
                                label=(
                                    f"{label_base} f_t={temp_freq:.0f}Hz "
                                    f"f_s={spatial_freq:.2f}"
                                ),
                                frames=frames.cpu(),
                                dt_ms=dt_ms,
                                meta={
                                    "temporal_hz": temp_freq,
                                    "spatial_cycles_per_mm": spatial_freq,
                                    "amplitude": amplitude,
                                },
                            )
                        )
        return packets

    def _pack_rotating_edges(self, spec: ProtocolSpec) -> List[StimulusPacket]:
        if self._grid_manager is None:
            return []
        xx = self._grid_manager.xx.to(self._device)
        yy = self._grid_manager.yy.to(self._device)
        steps = 180
        angles = torch.linspace(
            0.0,
            math.pi,
            steps,
            device=self._device,
            dtype=xx.dtype,
        )
        envelope = torch.hann_window(
            steps,
            periodic=False,
            device=self._device,
            dtype=xx.dtype,
        )
        envelope = envelope[:, None, None]
        width_options = (0.06, 0.1)
        amplitude_options = (0.6, 1.0)
        packets: List[StimulusPacket] = []
        for width in width_options:
            base_frames = []
            for theta in angles:
                edge = edge_stimulus_torch(
                    xx,
                    yy,
                    theta,
                    w=width,
                    amplitude=1.0,
                )
                base_frames.append(edge)
            base_frames_t = torch.stack(base_frames, dim=0)
            for amplitude in amplitude_options:
                frames = base_frames_t * envelope * amplitude
                packets.append(
                    StimulusPacket(
                        protocol_key=spec.key,
                        protocol_name=spec.title,
                        label=(f"Rotating edge w={width:.2f} " f"amp={amplitude:.1f}"),
                        frames=frames.cpu(),
                        dt_ms=self._packet_dt_ms,
                        meta={
                            "width_mm": width,
                            "amplitude": amplitude,
                            "steps": steps,
                        },
                    )
                )
        return packets

    def _pack_center_surround(self, spec: ProtocolSpec) -> List[StimulusPacket]:
        if self._grid_manager is None:
            return []
        xx = self._grid_manager.xx.to(self._device)
        yy = self._grid_manager.yy.to(self._device)
        span = min(
            float(xx.max() - xx.min()),
            float(yy.max() - yy.min()),
        )
        sigma_center = 0.18 * span
        sigma_surround = 0.48 * span
        surround_scale = 0.55
        profile = self._temporal_profile(
            80,
            160,
            80,
            60,
            device=self._device,
        )
        packets: List[StimulusPacket] = []
        centers = self._canonical_centers()[::2]
        for amplitude in (0.7, 1.0, 1.3):
            for cx, cy in centers:
                center = gaussian_pressure_torch(
                    xx,
                    yy,
                    center_x=cx,
                    center_y=cy,
                    amplitude=amplitude,
                    sigma=sigma_center,
                )
                surround = gaussian_pressure_torch(
                    xx,
                    yy,
                    center_x=cx,
                    center_y=cy,
                    amplitude=amplitude * surround_scale,
                    sigma=sigma_surround,
                )
                base = center - surround
                frames = profile[:, None, None] * base.unsqueeze(0)
                packets.append(
                    StimulusPacket(
                        protocol_key=spec.key,
                        protocol_name=spec.title,
                        label=(
                            "Center-surround "
                            f"amp={amplitude:.1f} @ ({cx:.2f},{cy:.2f})"
                        ),
                        frames=frames.cpu(),
                        dt_ms=self._packet_dt_ms,
                        meta={
                            "amplitude": amplitude,
                            "sigma_center": sigma_center,
                            "sigma_surround": sigma_surround,
                        },
                    )
                )
        return packets

    def _pack_moving_gaussians(self, spec: ProtocolSpec) -> List[StimulusPacket]:
        xx = self._grid_manager.xx.to(self._device)
        yy = self._grid_manager.yy.to(self._device)
        x_min, x_max, y_min, y_max = self._grid_bounds()
        sigma = 0.2 * min(x_max - x_min, y_max - y_min)
        profile = self._temporal_profile(
            40,
            200,
            40,
            60,
            device=self._device,
        )
        steps = profile.shape[0]
        t = torch.linspace(0.0, 1.0, steps, device=self._device)
        paths = [
            ((x_min, y_min), (x_max, y_max), "Diag ↗"),
            ((x_min, y_max), (x_max, y_min), "Diag ↘"),
            (
                (x_min, (y_min + y_max) / 2),
                (x_max, (y_min + y_max) / 2),
                "Horiz →",
            ),
            (
                ((x_min + x_max) / 2, y_min),
                ((x_min + x_max) / 2, y_max),
                "Vert ↑",
            ),
        ]
        packets: List[StimulusPacket] = []
        for start_pt, end_pt, label_base in paths:
            start = torch.tensor(start_pt, device=self._device)
            end = torch.tensor(end_pt, device=self._device)
            path = start + (end - start) * t.unsqueeze(1)
            cx = path[:, 0].view(-1, 1, 1)
            cy = path[:, 1].view(-1, 1, 1)
            dx = xx.unsqueeze(0) - cx
            dy = yy.unsqueeze(0) - cy
            base = torch.exp(-((dx**2 + dy**2) / (2 * sigma**2)))
            frames = profile[:, None, None] * base
            distance = torch.norm(end - start).item()
            velocity = distance / (steps * self._packet_dt_ms / 1000.0)
            packets.append(
                StimulusPacket(
                    protocol_key=spec.key,
                    protocol_name=spec.title,
                    label=f"{label_base} vel={velocity:.2f} mm/s",
                    frames=frames.cpu(),
                    dt_ms=self._packet_dt_ms,
                    meta={"velocity_mm_s": velocity, "sigma": sigma},
                )
            )
        return packets

    def _pack_macro_gaussians(self, spec: ProtocolSpec) -> List[StimulusPacket]:
        xx = self._grid_manager.xx.to(self._device)
        yy = self._grid_manager.yy.to(self._device)
        x_min, x_max, y_min, y_max = self._grid_bounds()
        span = min(x_max - x_min, y_max - y_min)
        profile = self._temporal_profile(
            120,
            400,
            120,
            80,
            device=self._device,
        )
        duration_ms = profile.shape[0] * self._packet_dt_ms
        sigma_scales = (0.28, 0.36, 0.45)
        amplitudes = (1.4, 1.8, 2.2)
        centers = self._canonical_centers()[:5]
        packets: List[StimulusPacket] = []
        for sigma_scale in sigma_scales:
            sigma = max(sigma_scale * span, 0.12)
            for amplitude in amplitudes:
                for cx, cy in centers:
                    base = gaussian_pressure_torch(
                        xx,
                        yy,
                        center_x=cx,
                        center_y=cy,
                        amplitude=amplitude,
                        sigma=sigma,
                    )
                    frames = profile[:, None, None] * base.unsqueeze(0)
                    packets.append(
                        StimulusPacket(
                            protocol_key=spec.key,
                            protocol_name=spec.title,
                            label=(
                                "Macro Gaussian amp="
                                f"{amplitude:.1f} σ={sigma:.2f} "
                                f"@ ({cx:.2f},{cy:.2f})"
                            ),
                            frames=frames.cpu(),
                            dt_ms=self._packet_dt_ms,
                            meta={
                                "amplitude": amplitude,
                                "sigma": sigma,
                                "duration_ms": duration_ms,
                            },
                        )
                    )
        return packets

    def _pack_edge_steps(self, spec: ProtocolSpec) -> List[StimulusPacket]:
        xx = self._grid_manager.xx.to(self._device)
        yy = self._grid_manager.yy.to(self._device)
        profile = self._temporal_profile(
            20,
            80,
            20,
            50,
            device=self._device,
        )
        packets: List[StimulusPacket] = []
        for amplitude in (0.6, 1.0):
            for orientation_deg in (0, 30, 60, 90, 120, 150):
                theta = orientation_deg * torch.pi / 180.0
                base = edge_stimulus_torch(
                    xx,
                    yy,
                    theta=theta,
                    w=0.08,
                    amplitude=amplitude,
                )
                frames = profile[:, None, None] * base.unsqueeze(0)
                packets.append(
                    StimulusPacket(
                        protocol_key=spec.key,
                        protocol_name=spec.title,
                        label=f"Edge {orientation_deg}° amp={amplitude:.1f}",
                        frames=frames.cpu(),
                        dt_ms=self._packet_dt_ms,
                        meta={
                            "orientation_deg": float(orientation_deg),
                            "amplitude": amplitude,
                        },
                    )
                )
        return packets

    def _pack_point_pulses(self, spec: ProtocolSpec) -> List[StimulusPacket]:
        xx = self._grid_manager.xx.to(self._device)
        yy = self._grid_manager.yy.to(self._device)
        profile = self._temporal_profile(
            10,
            15,
            10,
            40,
            device=self._device,
        )
        packets: List[StimulusPacket] = []
        for diameter in (0.4, 0.6):
            for amplitude in (0.8, 1.2):
                for cx, cy in self._canonical_centers():
                    base = point_pressure_torch(
                        xx,
                        yy,
                        center_x=cx,
                        center_y=cy,
                        amplitude=amplitude,
                        diameter_mm=diameter,
                    )
                    frames = profile[:, None, None] * base.unsqueeze(0)
                    packets.append(
                        StimulusPacket(
                            protocol_key=spec.key,
                            protocol_name=spec.title,
                            label=(
                                f"Point pulse amp={amplitude:.1f} "
                                f"D={diameter:.2f} @ ({cx:.2f},{cy:.2f})"
                            ),
                            frames=frames.cpu(),
                            dt_ms=self._packet_dt_ms,
                            meta={
                                "amplitude": amplitude,
                                "diameter_mm": diameter,
                            },
                        )
                    )
        return packets

    def _pack_large_disk_pulses(self, spec: ProtocolSpec) -> List[StimulusPacket]:
        xx = self._grid_manager.xx.to(self._device)
        yy = self._grid_manager.yy.to(self._device)
        profile = self._temporal_profile(
            40,
            160,
            40,
            80,
            device=self._device,
        )
        duration_ms = profile.shape[0] * self._packet_dt_ms
        packets: List[StimulusPacket] = []
        centers = self._canonical_centers()[:5]
        for diameter in (0.8, 1.2, 1.6):
            for amplitude in (1.4, 1.9, 2.4):
                for cx, cy in centers:
                    base = point_pressure_torch(
                        xx,
                        yy,
                        center_x=cx,
                        center_y=cy,
                        amplitude=amplitude,
                        diameter_mm=diameter,
                    )
                    frames = profile[:, None, None] * base.unsqueeze(0)
                    packets.append(
                        StimulusPacket(
                            protocol_key=spec.key,
                            protocol_name=spec.title,
                            label=(
                                "Large disk amp="
                                f"{amplitude:.1f} D={diameter:.2f} "
                                f"@ ({cx:.2f},{cy:.2f})"
                            ),
                            frames=frames.cpu(),
                            dt_ms=self._packet_dt_ms,
                            meta={
                                "amplitude": amplitude,
                                "diameter_mm": diameter,
                                "duration_ms": duration_ms,
                            },
                        )
                    )
        return packets

    def _pack_texture_sequences(self, spec: ProtocolSpec) -> List[StimulusPacket]:
        rows, cols = self._grid_manager.xx.shape
        steps = max(int(round(600.0 / self._packet_dt_ms)), 1)
        amplitude = 1.0
        duration_ms = steps * self._packet_dt_ms
        packets: List[StimulusPacket] = []
        for seed in (101, 202, 303):
            gen = torch.Generator(device=self._device)
            gen.manual_seed(seed)
            pattern = torch.rand(
                (steps, rows, cols), generator=gen, device=self._device
            )
            frames = amplitude * (pattern > 0.55).float()
            packets.append(
                StimulusPacket(
                    protocol_key=spec.key,
                    protocol_name=spec.title,
                    label=f"Frozen noise seed={seed}",
                    frames=frames.cpu(),
                    dt_ms=self._packet_dt_ms,
                    meta={
                        "seed": float(seed),
                        "duration_ms": duration_ms,
                    },
                )
            )
        return packets

    def _temporal_profile(
        self,
        ramp_up_ms: int,
        plateau_ms: int,
        ramp_down_ms: int,
        baseline_ms: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        dt_ms = self._packet_dt_ms
        ramp_up_steps = max(int(ramp_up_ms / dt_ms), 1)
        plateau_steps = max(int(plateau_ms / dt_ms), 1)
        ramp_down_steps = max(int(ramp_down_ms / dt_ms), 1)
        baseline_steps = max(int(baseline_ms / dt_ms), 1)
        device = torch.device(device)
        up = torch.linspace(0.0, 1.0, ramp_up_steps, device=device)
        plateau = torch.ones(plateau_steps, device=device)
        down = torch.linspace(1.0, 0.0, ramp_down_steps, device=device)
        baseline = torch.zeros(baseline_steps, device=device)
        profile = torch.cat([baseline, up, plateau, down, baseline])
        return profile.to(device=device, dtype=torch.float32)


__all__ = [
    "ProtocolSpec",
    "StimulusPacket",
    "RunResult",
    "ProtocolWorker",
]