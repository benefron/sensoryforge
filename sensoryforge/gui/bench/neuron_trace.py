"""Neuron trace bench: voltage trace + spikes for a step current, and an f-I curve.

Builds the population's neuron model (Izhikevich/AdEx/MQIF/FA/SA, or a
compiled DSL model) with
:func:`~sensoryforge.config.defaults.resolve_neuron_params`'s resolved
parameters at ``simulation.integrate_dt_ms`` -- the same construction
:class:`~sensoryforge.core.simulation_engine.SimulationEngine` uses -- and
runs it on a synthetic step current. No filter, no innervation, no full run.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from PyQt5 import QtWidgets

from sensoryforge.config.defaults import resolve_neuron_params
from sensoryforge.config.schema import PopulationConfig
from sensoryforge.gui import theme
from sensoryforge.gui.bench import find_population
from sensoryforge.gui.session import Session
from sensoryforge.gui.widgets import plot_factory
from sensoryforge.neurons.model_dsl import NeuronModel
from sensoryforge.registry import NEURON_REGISTRY

#: f-I curve current amplitudes, mA (the same unit as model_params/drive).
_FI_AMPLITUDES = np.linspace(0.0, 20.0, 9)
#: Trace/f-I stimulus duration, ms.
_DURATION_MS = 200.0
#: Fraction of the duration before the step current turns on.
_STEP_ONSET_FRACTION = 0.2


def _build_neuron(pop_cfg: PopulationConfig, integrate_dt_ms: float):
    """Construct this population's neuron model at ``integrate_dt_ms``.

    Raises:
        ValueError: If the neuron model is unregistered, or DSL with no
            ``dsl_config``.
    """
    model_name = pop_cfg.neuron_model or "Izhikevich"
    try:
        neuron_cls = NEURON_REGISTRY.get_class(model_name)
    except KeyError as exc:
        raise ValueError(str(exc)) from None

    if neuron_cls is NeuronModel:
        if not pop_cfg.dsl_config:
            raise ValueError(
                f"population {pop_cfg.name!r} has no dsl_config to compile"
            )
        dsl_model = NeuronModel.from_config(pop_cfg.dsl_config)
        return dsl_model.compile(
            dt=integrate_dt_ms, device="cpu", noise_std=pop_cfg.noise_std
        )

    params = resolve_neuron_params(
        model_name, pop_cfg.neuron_type, pop_cfg.model_params
    )
    params["dt"] = integrate_dt_ms
    params["noise_std"] = pop_cfg.noise_std
    return neuron_cls(**params)


def compute_step_trace(
    pop_cfg: PopulationConfig,
    integrate_dt_ms: float,
    amplitude: float,
    duration_ms: float = _DURATION_MS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voltage/state trace and spike times for a step current.

    Args:
        pop_cfg: The population whose neuron model to run.
        integrate_dt_ms: Neuron integration step (ms).
        amplitude: Step current amplitude (mA) after the onset.
        duration_ms: Total trace duration (ms).

    Returns:
        ``(time_ms, trace, spike_times_ms)`` -- ``trace`` is the voltage (or
        DSL state) trace, ``[n_steps + 1]``; ``spike_times_ms`` is empty for
        an analog (no-threshold) model.
    """
    n_steps = max(1, int(round(duration_ms / integrate_dt_ms)))
    onset = max(1, int(n_steps * _STEP_ONSET_FRACTION))
    neuron = _build_neuron(pop_cfg, integrate_dt_ms)

    current = torch.zeros(1, n_steps, 1)
    current[:, onset:, :] = amplitude

    output = neuron(current)
    if isinstance(output, tuple):
        trace, spikes = output
    else:
        trace, spikes = output, None

    trace_np = trace.detach().cpu().numpy().reshape(trace.shape[1])
    time_ms = np.arange(trace_np.shape[0], dtype=np.float64) * integrate_dt_ms

    if spikes is not None:
        spikes_np = spikes.detach().cpu().numpy().reshape(spikes.shape[1])
        spike_times = time_ms[spikes_np.astype(bool)]
    else:
        spike_times = np.array([], dtype=np.float64)

    return time_ms, trace_np, spike_times


def compute_fi_curve(
    pop_cfg: PopulationConfig,
    integrate_dt_ms: float,
    amplitudes: np.ndarray = _FI_AMPLITUDES,
    duration_ms: float = _DURATION_MS,
) -> np.ndarray:
    """Firing rate (spikes/s) at each amplitude in ``amplitudes``.

    An analog (no-threshold) model returns all zeros -- there are no spikes
    to count.

    Args:
        pop_cfg: The population whose neuron model to run.
        integrate_dt_ms: Neuron integration step (ms).
        amplitudes: Step current amplitudes (mA).
        duration_ms: Trace duration per amplitude (ms).

    Returns:
        Firing rate per amplitude, ``[len(amplitudes)]``, spikes/s.
    """
    rates = np.zeros(len(amplitudes), dtype=np.float64)
    for i, amp in enumerate(amplitudes):
        _, _, spike_times = compute_step_trace(
            pop_cfg, integrate_dt_ms, float(amp), duration_ms
        )
        rates[i] = len(spike_times) / (duration_ms / 1000.0)
    return rates


class NeuronTraceBench(QtWidgets.QWidget):
    """Step-current voltage trace + spikes, and an f-I curve.

    Args:
        session: The experiment the previewed population belongs to.
        parent: Qt parent.
    """

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._population_name: Optional[str] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Step amplitude:"))
        self.spin_amplitude = QtWidgets.QDoubleSpinBox()
        self.spin_amplitude.setRange(-1000.0, 1000.0)
        self.spin_amplitude.setDecimals(2)
        self.spin_amplitude.setSuffix(" mA")
        self.spin_amplitude.setValue(10.0)
        self.spin_amplitude.valueChanged.connect(self._on_amplitude_changed)
        header.addWidget(self.spin_amplitude)
        header.addStretch(1)
        layout.addLayout(header)

        self.trace_plot = plot_factory.make_plot(
            "Step response", "Time", "Voltage / state", x_unit="ms", y_unit="mV"
        )
        self.trace_curve = self.trace_plot.plot(pen=theme.pen(theme.PALETTE["accent"]))
        self.raster_item = plot_factory.make_raster_item(theme.PALETTE["error"])
        self.trace_plot.addItem(self.raster_item)
        layout.addWidget(self.trace_plot, 1)

        self.fi_plot = plot_factory.make_plot(
            "f-I curve", "Current", "Rate", x_unit="mA", y_unit="spikes/s"
        )
        self.fi_curve = self.fi_plot.plot(
            pen=theme.pen(theme.PALETTE["accent"]), symbol="o", symbolSize=6
        )
        layout.addWidget(self.fi_plot, 1)

        self.error_label = QtWidgets.QLabel("")
        self.error_label.setStyleSheet(f"color: {theme.PALETTE['error']};")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

    def set_population(self, population_name: Optional[str]) -> None:
        """Point the bench at a different population and recompute."""
        self._population_name = population_name
        self.refresh()

    def _on_amplitude_changed(self, _value: float) -> None:
        self._refresh_trace()

    def refresh(self) -> None:
        """Recompute the trace and the f-I curve from the current config."""
        self._refresh_trace()
        self._refresh_fi()

    def _clear(self) -> None:
        self.trace_curve.setData([], [])
        self.raster_item.setData([], [])
        self.fi_curve.setData([], [])

    def _current_population(self) -> Optional[PopulationConfig]:
        return find_population(self._session.config, self._population_name)

    def _refresh_trace(self) -> None:
        pop_cfg = self._current_population()
        if pop_cfg is None:
            self._clear()
            self.error_label.setVisible(False)
            return
        integrate_dt_ms = self._session.config.simulation.integrate_dt_ms
        try:
            time_ms, trace, spike_times = compute_step_trace(
                pop_cfg, integrate_dt_ms, self.spin_amplitude.value()
            )
        except (ValueError, KeyError, RuntimeError, TypeError) as exc:
            self._clear()
            self.error_label.setText(str(exc))
            self.error_label.setVisible(True)
            return
        self.error_label.setVisible(False)
        self.trace_curve.setData(time_ms, trace)
        if spike_times.size:
            y = np.full_like(spike_times, float(np.max(trace)) if trace.size else 0.0)
            self.raster_item.setData(x=spike_times, y=y)
        else:
            self.raster_item.setData([], [])

    def _refresh_fi(self) -> None:
        pop_cfg = self._current_population()
        if pop_cfg is None:
            self.fi_curve.setData([], [])
            return
        integrate_dt_ms = self._session.config.simulation.integrate_dt_ms
        try:
            rates = compute_fi_curve(pop_cfg, integrate_dt_ms, _FI_AMPLITUDES)
        except (ValueError, KeyError, RuntimeError, TypeError) as exc:
            self.fi_curve.setData([], [])
            self.error_label.setText(str(exc))
            self.error_label.setVisible(True)
            return
        self.fi_curve.setData(_FI_AMPLITUDES, rates)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        plot_factory.teardown(self.trace_plot)
        plot_factory.teardown(self.fi_plot)
        super().closeEvent(event)
