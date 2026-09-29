"""A population that fires no spikes is said so after a run (ledger F-93b91b1).

Runs go through the real :class:`RunController` (the run bar's path) into
the Run & Results screen, and through the Populations screen's quick run.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gui

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
    StimulusConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine  # noqa: E402
from sensoryforge.gui.execution.render import render_for_config  # noqa: E402
from sensoryforge.gui.execution.run_controller import RunController  # noqa: E402
from sensoryforge.gui.screens.populations import PopulationsScreen  # noqa: E402
from sensoryforge.gui.screens.results import ResultsScreen  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402

TIMEOUT_MS = 30000
SILENT_GAIN = 1e-6

#: A leaky integrator with no threshold/reset: an analog readout (Wave N).
LEAKY_INTEGRATOR = {
    "equations": "dv/dt = (-(v - v_rest) + R*I) / tau_m",
    "parameters": {"v_rest": -65.0, "R": 1.0, "tau_m": 10.0},
    "state_vars": {"v": -65.0},
}


def _config(*, sa_gain: float = 50.0, analog: bool = True) -> SensoryForgeConfig:
    """SA and RA fire at the default gain of 50 (135 spikes each in 30 ms)."""
    common = dict(target_grid="Main", innervation_method="gaussian", neurons_per_row=3)
    populations = [
        PopulationConfig(
            name="SA",
            neuron_type="SA",
            filter_method="sa",
            input_gain=sa_gain,
            seed=6,
            **common,
        ),
        PopulationConfig(
            name="RA", neuron_type="RA", filter_method="ra", seed=7, **common
        ),
    ]
    if analog:
        populations.append(
            PopulationConfig(
                name="Leaky",
                neuron_type="SA",
                neuron_model="dsl",
                filter_method="sa",
                dsl_config=LEAKY_INTEGRATOR,
                seed=5,
                **common,
            )
        )
    return SensoryForgeConfig(
        grids=[GridConfig(name="Main", rows=8, cols=8, spacing=0.2)],
        stimulus=StimulusConfig(
            type="gaussian", target_layer="Main", amplitude=40.0, spread=1.0
        ),
        populations=populations,
        simulation=SimulationConfig(
            device="cpu", dt_ms=1.0, integrate_dt_ms=1.0, duration_ms=30.0, seed=3
        ),
    )


def _run(qtbot, session: Session) -> None:
    controller = RunController(session)
    with qtbot.waitSignal(controller.finished, timeout=TIMEOUT_MS):
        controller.run(duration_ms=session.config.simulation.duration_ms)


def _screen(qtbot, session: Session) -> ResultsScreen:
    screen = ResultsScreen(session)
    qtbot.addWidget(screen)
    return screen


def test_a_silent_population_is_named_with_its_peak_input(qtbot):
    session = Session(_config(sa_gain=SILENT_GAIN))
    screen = _screen(qtbot, session)
    _run(qtbot, session)

    warning = screen.silent_warning()
    assert not screen.silent_banner.isHidden()
    assert "SA: peak filtered drive" in warning and "mA" in warning
    assert "RA" not in warning  # it fired
    assert "Leaky" not in warning  # analog: no threshold, nothing to fire
    peak = float(session.last_results.results["SA"]["filtered"].max())
    assert f"{peak:.3g} mA" in warning


def test_no_warning_when_every_population_fires(qtbot):
    session = Session(_config())
    screen = _screen(qtbot, session)
    _run(qtbot, session)

    assert all(
        pop["spikes"].sum() > 0
        for pop in session.last_results.results.values()
        if "spikes" in pop
    )
    assert screen.silent_warning() == ""
    assert screen.silent_banner.isHidden()


def test_an_analog_population_is_never_flagged(qtbot):
    config = _config(analog=True)
    config.populations = [pop for pop in config.populations if pop.name == "Leaky"]
    session = Session(config)
    screen = _screen(qtbot, session)
    _run(qtbot, session)

    assert "spikes" not in session.last_results.results["Leaky"]
    assert screen.silent_warning() == ""


def test_the_warning_clears_when_the_next_run_fires(qtbot):
    session = Session(_config(sa_gain=SILENT_GAIN))
    screen = _screen(qtbot, session)
    _run(qtbot, session)
    assert "SA" in screen.silent_warning()

    session.set_by_path("populations.0.input_gain", 50.0)
    _run(qtbot, session)

    assert screen.silent_warning() == ""


def test_a_screen_opened_after_the_run_shows_it(qtbot):
    session = Session(_config(sa_gain=SILENT_GAIN))
    _run(qtbot, session)
    assert "SA" in _screen(qtbot, session).silent_warning()


def test_a_saved_bundle_of_a_silent_run_shows_it(qtbot, tmp_path):
    pytest.importorskip("h5py")
    config = _config(sa_gain=SILENT_GAIN)
    rendered = render_for_config(config, duration_ms=config.simulation.duration_ms)
    SimulationEngine(config).run(
        rendered.stimulus,
        return_intermediates=True,
        bundle_dir=str(tmp_path / "bundle"),
        stimulus_config=config.stimulus.to_dict(),
        seed=config.simulation.seed,
    )
    session = Session(_config())
    screen = _screen(qtbot, session)

    screen.open_bundle(tmp_path / "bundle")
    assert "SA: peak filtered drive" in screen.silent_warning()

    screen.show_live_results()  # no live run yet
    assert screen.silent_warning() == ""


class TestQuickRun:
    def _quick_run(self, qtbot, session: Session, name: str) -> PopulationsScreen:
        screen = PopulationsScreen(session)
        qtbot.addWidget(screen)
        screen._select(name)
        with qtbot.waitSignal(screen._run_controller.finished, timeout=TIMEOUT_MS):
            screen._on_quick_run()
        return screen

    def test_a_silent_population_is_named(self, qtbot):
        screen = self._quick_run(qtbot, Session(_config(sa_gain=SILENT_GAIN)), "SA")
        assert not screen.quick_warning.isHidden()
        assert "SA: peak filtered drive" in screen.quick_warning.text()

    def test_a_firing_population_shows_no_warning(self, qtbot):
        screen = self._quick_run(qtbot, Session(_config(sa_gain=SILENT_GAIN)), "RA")
        assert screen.quick_warning.isHidden()

    def test_an_analog_population_shows_no_warning(self, qtbot):
        screen = self._quick_run(qtbot, Session(_config()), "Leaky")
        assert screen.quick_warning.isHidden()
