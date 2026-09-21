"""Tests for :mod:`sensoryforge.gui.screens.results` and its data adapter.

A two-population config -- one spiking (Izhikevich), one analog (a DSL leaky
integrator, no threshold) -- is run two ways: once directly through
:class:`~sensoryforge.core.simulation_engine.SimulationEngine` and bundled to
disk, and once through :class:`~sensoryforge.gui.execution.run_controller.RunController`
(the live path). Both are turned into a
:class:`~sensoryforge.gui.screens.results_data.ResultsView` and compared, then
driven through the real screen widgets.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gui

from PyQt5 import QtCore  # noqa: E402

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
    StimulusConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine  # noqa: E402
from sensoryforge.gui.execution.run_controller import RunController  # noqa: E402
from sensoryforge.gui.project import ProjectHandle  # noqa: E402
from sensoryforge.gui.screens import results_data  # noqa: E402
from sensoryforge.gui.screens.results import ResultsScreen  # noqa: E402
from sensoryforge.gui.screens.results_raster_panel import (  # noqa: E402
    MAX_RASTER_POINTS,
    raster_points,
)
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.io.bundle import load_bundle, write_bundle  # noqa: E402

h5py = pytest.importorskip("h5py")

TIMEOUT_MS = 30000

#: A leaky integrator with no threshold/reset: an analog readout (Wave N).
LEAKY_INTEGRATOR = {
    "equations": "dv/dt = (-(v - v_rest) + R*I) / tau_m",
    "parameters": {"v_rest": -65.0, "R": 1.0, "tau_m": 10.0},
    "state_vars": {"v": -65.0},
}


def _config() -> SensoryForgeConfig:
    return SensoryForgeConfig(
        grids=[
            GridConfig(name="Main", arrangement="grid", rows=8, cols=8, spacing=0.2)
        ],
        stimulus=StimulusConfig(
            type="gaussian", target_layer="Main", amplitude=40.0, spread=1.0
        ),
        populations=[
            PopulationConfig(
                name="Spiking",
                neuron_type="RA",
                neuron_model="izhikevich",
                filter_method="ra",
                innervation_method="gaussian",
                neurons_per_row=3,
                seed=6,
            ),
            PopulationConfig(
                name="Analog",
                neuron_type="SA",
                neuron_model="dsl",
                filter_method="sa",
                innervation_method="gaussian",
                neurons_per_row=3,
                dsl_config=LEAKY_INTEGRATOR,
                seed=5,
            ),
        ],
        simulation=SimulationConfig(
            device="cpu", dt_ms=1.0, integrate_dt_ms=1.0, duration_ms=30.0, seed=3
        ),
    )


@pytest.fixture
def bundle_dir(tmp_path):
    """A written bundle for :func:`_config`, from a direct engine run."""
    config = _config()
    engine = SimulationEngine(config)
    from sensoryforge.gui.execution.render import render_for_config

    rendered = render_for_config(config, duration_ms=config.simulation.duration_ms)
    results = engine.run(rendered.stimulus, return_intermediates=True)
    out = tmp_path / "bundle"
    write_bundle(
        out,
        config,
        engine,
        results,
        rendered.stimulus,
        stimulus_config=config.stimulus.to_dict(),
        seed=config.simulation.seed,
    )
    return out


def _run_live(qtbot, tmp_path):
    """Run :func:`_config` through :class:`RunController`; return the session."""
    config = _config()
    session = Session(config)
    session.set_project(ProjectHandle.create(tmp_path / "project", config))
    controller = RunController(session)
    with qtbot.waitSignal(controller.finished, timeout=TIMEOUT_MS):
        controller.run(duration_ms=config.simulation.duration_ms)
    return session


# --------------------------------------------------------------- results_data


def test_live_and_bundle_views_are_tensor_equal(qtbot, tmp_path, bundle_dir):
    session = _run_live(qtbot, tmp_path / "live")
    live_view = results_data.from_run_result(session.last_results)

    bundle = load_bundle(bundle_dir)
    bundle_view = results_data.from_bundle(bundle)

    assert torch.equal(
        live_view.stimulus, bundle_view.stimulus.to(live_view.stimulus.dtype)
    )
    assert torch.equal(live_view.time_ms.float(), bundle_view.time_ms.float())
    assert live_view.xlim == pytest.approx(bundle_view.xlim)
    assert live_view.ylim == pytest.approx(bundle_view.ylim)

    assert [p.name for p in live_view.populations] == [
        p.name for p in bundle_view.populations
    ]
    for live_pop, bundle_pop in zip(live_view.populations, bundle_view.populations):
        assert live_pop.is_analog == bundle_pop.is_analog
        for field in ("spikes", "state", "drive", "filtered"):
            live_t = getattr(live_pop, field)
            bundle_t = getattr(bundle_pop, field)
            if live_t is None:
                assert bundle_t is None
                continue
            assert torch.equal(live_t.float(), bundle_t.float())
        assert torch.equal(live_pop.neuron_centers, bundle_pop.neuron_centers)
        assert torch.equal(live_pop.weights, bundle_pop.weights)


def test_analog_population_has_no_spikes_and_state_matches(bundle_dir):
    bundle = load_bundle(bundle_dir)
    view = results_data.from_bundle(bundle)
    analog = view.population("Analog")
    assert analog is not None
    assert analog.is_analog
    assert analog.spikes is None
    assert analog.state is not None


def test_spiking_raster_count_matches_ground_truth(bundle_dir):
    bundle = load_bundle(bundle_dir)
    view = results_data.from_bundle(bundle)
    spiking = view.population("Spiking")
    spikes = spiking.spikes.detach().cpu().numpy()
    t_idx, n_idx, total, decimated = raster_points(spikes)
    assert total == int((spikes > 0).sum())
    assert not decimated
    assert len(t_idx) == total


def test_raster_decimation_cap():
    rng = np.random.default_rng(0)
    spikes = (rng.random((500, 500)) < 0.3).astype(np.float32)
    total_true = int((spikes > 0).sum())
    assert total_true > MAX_RASTER_POINTS
    t_idx, n_idx, total, decimated = raster_points(spikes)
    assert decimated
    assert total == total_true
    assert len(t_idx) < total
    assert len(t_idx) <= MAX_RASTER_POINTS

    # A run under the cap is never decimated, and its plotted count is exact.
    small = (rng.random((10, 10)) < 0.3).astype(np.float32)
    t_idx2, n_idx2, total2, decimated2 = raster_points(small)
    assert not decimated2
    assert len(t_idx2) == total2 == int((small > 0).sum())


# ------------------------------------------------------------------- screen


def test_screen_shows_analog_population_as_one_heatmap_band_no_raster_points(
    bundle_dir, qtbot
):
    session = Session(_config())
    screen = ResultsScreen(session)
    qtbot.addWidget(screen)
    bundle = load_bundle(bundle_dir)
    view = results_data.from_bundle(bundle)
    screen._set_view(view)

    assert "Analog" not in screen.raster_panel.spike_counts
    assert len(screen.raster_panel._image_items) == 1
    analog = view.population("Analog")
    plotted = screen.raster_panel._image_items[0].image
    assert np.array_equal(plotted, analog.state.detach().cpu().numpy())

    assert "Spiking" in screen.raster_panel.spike_counts


def test_slider_sets_stimulus_frame_and_cursor_exactly(bundle_dir, qtbot):
    session = Session(_config())
    screen = ResultsScreen(session)
    qtbot.addWidget(screen)
    bundle = load_bundle(bundle_dir)
    view = results_data.from_bundle(bundle)
    screen._set_view(view)

    k = 5
    screen.playback.set_index(k)

    expected_frame = view.stimulus[k].detach().cpu().numpy()
    assert np.array_equal(screen.stimulus_panel.image_item.image, expected_frame)

    expected_t = float(view.time_ms[k])
    assert screen.raster_panel.cursor.value() == pytest.approx(expected_t)
    assert screen.rate_panel.cursor.value() == pytest.approx(expected_t)
    assert screen.trace_panel.cursor.value() == pytest.approx(expected_t)


def test_clicking_a_neuron_in_the_map_selects_it_in_the_trace(bundle_dir, qtbot):
    session = Session(_config())
    screen = ResultsScreen(session)
    qtbot.addWidget(screen)
    bundle = load_bundle(bundle_dir)
    view = results_data.from_bundle(bundle)
    screen._set_view(view)

    spiking = view.population("Spiking")
    neuron_index = 1
    screen.map_panel.neuronClicked.emit("Spiking", neuron_index)

    assert screen.trace_panel.population_combo.currentText() == "Spiking"
    assert screen.trace_panel.neuron_spin.value() == neuron_index
    assert screen.trace_panel.active_curve is not None
    # A bundle stores drive/filtered/spikes (or state) but not voltages, so
    # the trace panel's readout curve falls back to "filtered" here.
    expected_y = spiking.filtered[:, neuron_index].detach().cpu().numpy()
    np.testing.assert_array_equal(screen.trace_panel.active_curve.yData, expected_y)


def test_stale_banner_shows_after_edit_and_clears_after_rerun(qtbot, tmp_path):
    session = _run_live(qtbot, tmp_path)
    screen = ResultsScreen(session)
    qtbot.addWidget(screen)
    screen.show()
    qtbot.waitExposed(screen)
    assert not screen.stale_banner.isVisible()

    session.set_by_path("populations.0.input_gain", 12.0)
    assert screen.stale_banner.isVisible()

    controller = RunController(session)
    with qtbot.waitSignal(controller.finished, timeout=TIMEOUT_MS):
        controller.run(duration_ms=session.config.simulation.duration_ms)
    assert not screen.stale_banner.isVisible()


def test_open_bundle_on_non_bundle_directory_shows_error_and_keeps_live_results(
    qtbot, tmp_path
):
    session = _run_live(qtbot, tmp_path)
    screen = ResultsScreen(session)
    qtbot.addWidget(screen)
    screen.show()
    qtbot.waitExposed(screen)
    live_view_populations = list(screen._view.populations)

    not_a_bundle = tmp_path / "not_a_bundle"
    not_a_bundle.mkdir()
    screen.open_bundle(not_a_bundle)

    assert screen.error_banner.isVisible()
    assert not screen.bundle_banner.isVisible()
    assert [p.name for p in screen._view.populations] == [
        p.name for p in live_view_populations
    ]
