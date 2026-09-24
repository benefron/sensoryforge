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
    assert screen.trace_panel.cursor_current.value() == pytest.approx(expected_t)
    assert screen.trace_panel.cursor_readout.value() == pytest.approx(expected_t)


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


from sensoryforge.gui.screens.results_rate_panel import smoothed_rate  # noqa: E402


def test_smoothed_rate_shape_when_window_exceeds_run_length():
    """A run shorter than the smoothing window must not grow past T samples
    (np.convolve(mode="same") pads out to the kernel length otherwise)."""
    spikes = np.zeros((10, 4), dtype=np.float32)
    rate = smoothed_rate(spikes, dt_ms=1.0, window_ms=20.0)
    assert rate.shape == (10,)


def test_smoothed_rate_shape_for_single_time_step():
    spikes = np.ones((1, 3), dtype=np.float32)
    rate = smoothed_rate(spikes, dt_ms=1.0, window_ms=20.0)
    assert rate.shape == (1,)
    assert rate[0] == pytest.approx(1000.0)  # 1 spike/neuron in a 1 ms bin


def test_smoothed_rate_mean_matches_total_spikes_for_one_bin_window():
    rng = np.random.default_rng(0)
    spikes = (rng.random((50, 6)) < 0.2).astype(np.float32)
    dt_ms = 2.0
    rate = smoothed_rate(spikes, dt_ms=dt_ms, window_ms=dt_ms)  # one-bin window
    duration_s = spikes.shape[0] * dt_ms / 1000.0
    expected_mean_rate = spikes.sum() / spikes.shape[1] / duration_s
    assert rate.mean() == pytest.approx(expected_mean_rate)


def _walk_axis_items(screen):
    """Every ``pg.AxisItem`` reachable from ``screen``'s panels."""

    plots = []
    for panel in (
        screen.stimulus_panel,
        screen.raster_panel,
        screen.rate_panel,
        screen.trace_panel,
        screen.map_panel,
    ):
        for name in ("plot", "current_plot", "readout_plot"):
            plot = getattr(panel, name, None)
            if plot is not None:
                plots.append(plot)
    axes = []
    for plot in plots:
        plot_item = plot.getPlotItem()
        for axis_name in ("bottom", "left", "right", "top"):
            axis = plot_item.getAxis(axis_name)
            if axis is not None:
                axes.append(axis)
    return axes


def test_trace_panel_splits_current_and_readout_onto_two_linked_plots(
    bundle_dir, qtbot
):
    """Defect 1: drive/filtered (mA) and voltage/state (mV) must not share
    an axis -- the trace panel must be two stacked, x-linked plots."""
    session = Session(_config())
    screen = ResultsScreen(session)
    qtbot.addWidget(screen)
    bundle = load_bundle(bundle_dir)
    view = results_data.from_bundle(bundle)
    screen._set_view(view)

    panel = screen.trace_panel
    assert panel.current_plot is not panel.readout_plot
    assert (
        panel.readout_plot.getPlotItem().getViewBox()
        is panel.current_plot.getPlotItem().getViewBox().linkedView(0)
        or panel.readout_plot.getPlotItem().vb.linkedView(0)
        is panel.current_plot.getPlotItem().vb
    )
    # The current plot's left axis must be about current, never voltage.
    current_label = panel.current_plot.getPlotItem().getAxis("left").labelText
    assert "mA" in current_label
    assert "mV" not in current_label

    screen.map_panel.neuronClicked.emit("Spiking", 0)
    readout_label = panel.readout_plot.getPlotItem().getAxis("left").labelText
    assert "mV" in readout_label

    screen.map_panel.neuronClicked.emit("Analog", 0)
    analog_label = panel.readout_plot.getPlotItem().getAxis("left").labelText
    assert analog_label != readout_label
    assert "mV" not in analog_label


def test_trace_panel_cursor_moves_on_both_plots(bundle_dir, qtbot):
    session = Session(_config())
    screen = ResultsScreen(session)
    qtbot.addWidget(screen)
    bundle = load_bundle(bundle_dir)
    view = results_data.from_bundle(bundle)
    screen._set_view(view)

    screen.trace_panel.set_cursor(12.0)
    assert screen.trace_panel.cursor_current.value() == pytest.approx(12.0)
    assert screen.trace_panel.cursor_readout.value() == pytest.approx(12.0)


def test_rate_panel_right_axis_hidden_unless_a_population_is_analog(qtbot):
    """Defect 2: no right (state) axis when nothing in the view is analog."""
    session = Session(_config())
    screen = ResultsScreen(session)
    qtbot.addWidget(screen)

    from sensoryforge.gui.screens.results_data import PopulationView, ResultsView

    spiking = PopulationView(
        name="OnlySpiking",
        index=0,
        neuron_type="RA",
        spikes=torch.zeros(3, 2),
        state=None,
        drive=None,
        filtered=None,
        voltages=None,
        neuron_centers=None,
        receptor_coords=None,
        weights=None,
    )
    view_no_analog = ResultsView(
        stimulus=torch.zeros(3, 2, 2),
        time_ms=torch.tensor([0.0, 1.0, 2.0]),
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        populations=[spiking],
    )
    screen._set_view(view_no_analog)
    assert not screen.rate_panel.plot.getPlotItem().getAxis("right").isVisible()


def test_rate_panel_right_axis_shown_when_a_population_is_analog(bundle_dir, qtbot):
    session = Session(_config())
    screen = ResultsScreen(session)
    qtbot.addWidget(screen)
    bundle = load_bundle(bundle_dir)
    view = results_data.from_bundle(bundle)
    screen._set_view(view)
    assert screen.rate_panel.plot.getPlotItem().getAxis("right").isVisible()


def test_rate_panel_does_not_connect_a_bound_method_to_sigresized():
    """Defect 2 (F-035): the right-axis sync must go through plot_factory.connect
    with a plain function, never a bound method, and be releasable by teardown."""
    from sensoryforge.gui.screens.results_rate_panel import RatePanel
    from sensoryforge.gui.widgets import plot_factory

    panel = RatePanel()
    owner = panel.plot
    connections = plot_factory._CONNECTIONS.get(owner, [])
    assert connections, "expected sigResized to be registered via plot_factory.connect"
    for signal, slot in connections:
        # functools.partial's .func must not be a bound method of the panel.
        func = getattr(slot, "func", slot)
        assert not (hasattr(func, "__self__") and isinstance(func.__self__, RatePanel))
    panel.teardown()


def test_stimulus_frame_bounding_rect_matches_view_extent(bundle_dir, qtbot):
    """Defect 3: after set_frame, the image's bounding rect in view/local
    coordinates equals the run's mm extent, not a fallback 1x1 scale."""
    session = Session(_config())
    screen = ResultsScreen(session)
    qtbot.addWidget(screen)
    bundle = load_bundle(bundle_dir)
    view = results_data.from_bundle(bundle)
    screen._set_view(view)

    item = screen.stimulus_panel.image_item
    rect = item.mapRectToParent(item.boundingRect())
    x0, x1 = view.xlim
    y0, y1 = view.ylim
    assert rect.left() == pytest.approx(x0, abs=1e-6)
    assert rect.top() == pytest.approx(y0, abs=1e-6)
    assert rect.width() == pytest.approx(x1 - x0, abs=1e-6)
    assert rect.height() == pytest.approx(y1 - y0, abs=1e-6)


def test_stimulus_frame_off_centre_peak_maps_to_its_mm_position(qtbot):
    """Defect 3: verify orientation by test, not assumption -- a stimulus
    frame with a single bright pixel off-centre must map to that pixel's
    known (x, y) mm position, within one pixel."""
    session = Session(_config())
    screen = ResultsScreen(session)
    qtbot.addWidget(screen)

    from sensoryforge.gui.screens.results_data import ResultsView

    h, w = 10, 20
    x0, x1 = -5.0, 5.0
    y0, y1 = -2.0, 2.0
    frame = torch.zeros(3, h, w)
    peak_x_idx, peak_y_idx = 8, 3  # first axis = x, second = y (indexing="ij")
    frame[0, peak_x_idx, peak_y_idx] = 99.0
    view = ResultsView(
        stimulus=frame,
        time_ms=torch.tensor([0.0, 1.0, 2.0]),
        xlim=(x0, x1),
        ylim=(y0, y1),
        populations=[],
    )
    screen._set_view(view)

    item = screen.stimulus_panel.image_item
    # Expected mm position of the peak pixel's centre.
    expected_x = x0 + (peak_x_idx + 0.5) * (x1 - x0) / h
    expected_y = y0 + (peak_y_idx + 0.5) * (y1 - y0) / w

    mapped = item.mapToParent(QtCore.QPointF(peak_x_idx + 0.5, peak_y_idx + 0.5))
    px_w = (x1 - x0) / h
    px_h = (y1 - y0) / w
    assert mapped.x() == pytest.approx(expected_x, abs=px_w)
    assert mapped.y() == pytest.approx(expected_y, abs=px_h)


def test_no_custom_axis_uses_units_kwarg_or_si_prefix(bundle_dir, qtbot):
    """Defect 4: every pg.AxisItem on the screen must have SI prefixing off
    and no '(k' in its label text (a leftover units= rescale)."""
    session = Session(_config())
    screen = ResultsScreen(session)
    qtbot.addWidget(screen)
    bundle = load_bundle(bundle_dir)
    view = results_data.from_bundle(bundle)
    screen._set_view(view)
    screen.map_panel.neuronClicked.emit("Spiking", 0)

    axes = _walk_axis_items(screen)
    assert axes, "expected at least one axis to inspect"
    for axis in axes:
        assert axis.autoSIPrefix is False
        label = axis.labelText or ""
        assert "(k" not in label


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
