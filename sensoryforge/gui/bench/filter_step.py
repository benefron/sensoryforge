"""Filter step bench: the selected filter's response to a unit step and a ramp.

Instantiates the registered filter class directly with
:func:`~sensoryforge.config.defaults.resolve_filter_params`'s resolved
parameters at the config's record step (``simulation.dt_ms``) -- the same
parameters :class:`~sensoryforge.core.simulation_engine.SimulationEngine`
would build for this population's filter -- and runs it on two short
synthetic drive tensors. No full simulation, no innervation, no neuron.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from PyQt5 import QtWidgets

from sensoryforge.config.defaults import resolve_filter_params
from sensoryforge.config.schema import PopulationConfig
from sensoryforge.gui import theme
from sensoryforge.gui.bench import find_population
from sensoryforge.gui.session import Session
from sensoryforge.gui.widgets import plot_factory
from sensoryforge.registry import FILTER_REGISTRY

#: Bench trace length (steps) at the config's dt_ms.
_N_STEPS = 200
#: Fraction of the trace before the unit step rises.
_STEP_ONSET_FRACTION = 0.15


def compute_filter_curves(
    pop_cfg: PopulationConfig, dt_ms: float, n_steps: int = _N_STEPS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``pop_cfg``'s filter on a unit step and a ramp.

    Args:
        pop_cfg: The population whose ``filter_method``/``filter_params`` to
            use.
        dt_ms: Record step (ms) -- ``simulation.dt_ms``.
        n_steps: Number of samples in each synthetic drive.

    Returns:
        ``(time_ms, step_response, ramp_response)``, each ``[n_steps]``.

    Raises:
        ValueError: If ``pop_cfg.filter_method`` is ``"none"``/``"identity"``
            (nothing to preview) or is not registered.
    """
    method = (pop_cfg.filter_method or "none").strip()
    if method.lower() in ("none", "identity", ""):
        raise ValueError("no filter selected for this population")
    try:
        filter_cls = FILTER_REGISTRY.get_class(method)
    except KeyError as exc:
        raise ValueError(str(exc)) from None
    if method.lower() in ("sa", "ra"):
        params = resolve_filter_params(method, pop_cfg.filter_params)
    else:
        params = dict(pop_cfg.filter_params or {})
    params["dt"] = dt_ms

    time_ms = np.arange(n_steps, dtype=np.float64) * dt_ms
    onset = max(1, int(n_steps * _STEP_ONSET_FRACTION))

    step = torch.zeros(1, n_steps, 1)
    step[:, onset:, :] = 1.0
    ramp = torch.linspace(0.0, 1.0, n_steps).reshape(1, n_steps, 1)

    step_filter = filter_cls(**params)
    step_out = step_filter(step).detach().cpu().numpy().reshape(-1)

    ramp_filter = filter_cls(**params)
    ramp_out = ramp_filter(ramp).detach().cpu().numpy().reshape(-1)

    return time_ms, step_out, ramp_out


class FilterStepBench(QtWidgets.QWidget):
    """Unit-step and ramp response of the selected population's filter.

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

        self.plot = plot_factory.make_plot(
            "Filter step / ramp response", "Time", "Current", x_unit="ms", y_unit="mA"
        )
        self.step_curve = self.plot.plot(
            pen=theme.pen(theme.PALETTE["accent"]), name="step"
        )
        self.ramp_curve = self.plot.plot(
            pen=theme.pen(theme.PALETTE["warning"]), name="ramp"
        )
        layout.addWidget(self.plot, 1)

        self.error_label = QtWidgets.QLabel("")
        self.error_label.setStyleSheet(f"color: {theme.PALETTE['error']};")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

    def set_population(self, population_name: Optional[str]) -> None:
        """Point the bench at a different population and recompute."""
        self._population_name = population_name
        self.refresh()

    def refresh(self) -> None:
        """Recompute both curves from the session's current config."""
        pop_cfg = find_population(self._session.config, self._population_name)
        if pop_cfg is None:
            self.step_curve.setData([], [])
            self.ramp_curve.setData([], [])
            self.error_label.setVisible(False)
            return
        dt_ms = self._session.config.simulation.dt_ms
        try:
            time_ms, step_out, ramp_out = compute_filter_curves(pop_cfg, dt_ms)
        except (ValueError, KeyError, RuntimeError, TypeError) as exc:
            self.step_curve.setData([], [])
            self.ramp_curve.setData([], [])
            self.error_label.setText(str(exc))
            self.error_label.setVisible(True)
            return
        self.error_label.setVisible(False)
        self.step_curve.setData(time_ms, step_out)
        self.ramp_curve.setData(time_ms, ramp_out)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        plot_factory.teardown(self.plot)
        super().closeEvent(event)
