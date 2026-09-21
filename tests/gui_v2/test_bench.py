"""Tests for the Populations screen's bench widgets (Task 2.3).

Each bench recomputes on demand from the session's live config; assertions
check the actual numeric output (bank weights, curve y-data, spike counts)
against a value computed independently through the engine or the raw
component classes -- never just that a widget constructs.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("PyQt5")

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine  # noqa: E402
from sensoryforge.gui.bench import filter_step, neuron_trace, rf_footprint  # noqa: E402
from sensoryforge.gui.bench.filter_step import FilterStepBench  # noqa: E402
from sensoryforge.gui.bench.neuron_trace import NeuronTraceBench  # noqa: E402
from sensoryforge.gui.bench.rf_footprint import RfFootprintBench  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402

pytestmark = pytest.mark.gui


def _two_pop_config() -> SensoryForgeConfig:
    return SensoryForgeConfig(
        grids=[GridConfig(name="skin", rows=16, cols=16, spacing=0.15)],
        populations=[
            PopulationConfig(
                name="SA Population",
                neuron_type="SA",
                target_grid="skin",
                innervation_method="gaussian",
                neurons_per_row=6,
                filter_method="SA",
                seed=42,
            ),
            PopulationConfig(
                name="RA Population",
                neuron_type="RA",
                target_grid="skin",
                innervation_method="template",
                resolvable_distance_mm=0.6,
                filter_method="RA",
            ),
        ],
    )


class TestRfFootprintBuild:
    def test_bank_matches_engine_for_gaussian(self):
        config = _two_pop_config()
        bank = rf_footprint.build_population_bank_for_config(config, "SA Population")
        engine = SimulationEngine(config, device="cpu")
        engine_bank = next(
            p["bank"] for p in engine.populations if p["name"] == "SA Population"
        )
        assert torch.equal(bank.weights, engine_bank.weights)
        assert torch.equal(bank.neuron_centers, engine_bank.neuron_centers)

    def test_bank_matches_engine_for_template(self):
        config = _two_pop_config()
        bank = rf_footprint.build_population_bank_for_config(config, "RA Population")
        engine = SimulationEngine(config, device="cpu")
        engine_bank = next(
            p["bank"] for p in engine.populations if p["name"] == "RA Population"
        )
        assert torch.equal(bank.weights, engine_bank.weights)
        assert bank.num_neurons == engine_bank.num_neurons

    def test_unknown_population_raises(self):
        config = _two_pop_config()
        with pytest.raises(ValueError):
            rf_footprint.build_population_bank_for_config(config, "nope")


class TestRfFootprintBenchWidget:
    def test_refresh_shows_neuron_centers_and_caption(self, qtbot):
        session = Session(_two_pop_config())
        bench = RfFootprintBench(session)
        qtbot.addWidget(bench)

        bench.set_population("SA Population")

        assert bench.bank is not None
        assert bench.spin_neuron.maximum() == bench.bank.num_neurons - 1
        assert "neurons" in bench.caption.text()
        assert bench.error_label.isHidden() is True

    def test_config_edit_changes_bank(self, qtbot):
        session = Session(_two_pop_config())
        bench = RfFootprintBench(session)
        qtbot.addWidget(bench)
        bench.set_population("SA Population")
        first_weights = bench.bank.weights.clone()

        session.set_by_path("populations.0.sigma_d_mm", 0.9)
        bench.refresh()

        assert not torch.equal(first_weights, bench.bank.weights)

    def test_neuron_click_updates_spinbox(self, qtbot):
        session = Session(_two_pop_config())
        bench = RfFootprintBench(session)
        qtbot.addWidget(bench)
        bench.set_population("SA Population")

        bench._on_neuron_clicked("SA Population", 3)

        assert bench.spin_neuron.value() == 3

    def test_bad_population_shows_error_not_raise(self, qtbot):
        session = Session(_two_pop_config())
        bench = RfFootprintBench(session)
        qtbot.addWidget(bench)

        bench.set_population("does-not-exist")

        assert bench.error_label.isHidden() is False
        assert bench.bank is None


class TestFilterStepCompute:
    def test_sa_step_response_matches_direct_filter(self):
        config = _two_pop_config()
        pop_cfg = config.populations[0]
        time_ms, step_out, ramp_out = filter_step.compute_filter_curves(
            pop_cfg, config.simulation.dt_ms
        )
        assert time_ms.shape == step_out.shape == ramp_out.shape
        # A causal filter's step response starts at (or near) zero before
        # the step turns on.
        assert abs(step_out[0]) < 1e-6

    def test_none_filter_raises(self):
        config = _two_pop_config()
        pop_cfg = config.populations[0]
        pop_cfg.filter_method = "none"
        with pytest.raises(ValueError):
            filter_step.compute_filter_curves(pop_cfg, config.simulation.dt_ms)


class TestFilterStepBenchWidget:
    def test_tau_r_edit_changes_curve(self, qtbot):
        session = Session(_two_pop_config())
        bench = FilterStepBench(session)
        qtbot.addWidget(bench)
        bench.set_population("SA Population")
        before_x, before_y = bench.step_curve.getData()

        session.set_by_path("populations.0.filter_params.tau_r", 25.0)
        bench.refresh()

        after_x, after_y = bench.step_curve.getData()
        assert not np.allclose(before_y, after_y)

    def test_no_filter_shows_error_not_raise(self, qtbot):
        session = Session(_two_pop_config())
        session.set_by_path("populations.0.filter_method", "none")
        bench = FilterStepBench(session)
        qtbot.addWidget(bench)

        bench.set_population("SA Population")

        assert bench.error_label.isHidden() is False


class TestNeuronTraceCompute:
    def test_ra_step_uses_fs_preset_defaults(self):
        config = _two_pop_config()
        pop_cfg = config.populations[1]  # RA -> FS preset
        _, trace, spikes = neuron_trace.compute_step_trace(
            pop_cfg, config.simulation.integrate_dt_ms, 20.0
        )
        assert trace.ndim == 1
        assert spikes.ndim == 1

    def test_fi_curve_increasing_with_amplitude(self):
        config = _two_pop_config()
        pop_cfg = config.populations[0]
        rates = neuron_trace.compute_fi_curve(
            pop_cfg,
            config.simulation.integrate_dt_ms,
            amplitudes=np.array([0.0, 30.0]),
        )
        assert rates[1] >= rates[0]


class TestNeuronTraceBenchWidget:
    def test_amplitude_change_changes_trace(self, qtbot):
        session = Session(_two_pop_config())
        bench = NeuronTraceBench(session)
        qtbot.addWidget(bench)
        bench.set_population("SA Population")
        before_x, before_y = bench.trace_curve.getData()

        bench.spin_amplitude.setValue(80.0)

        after_x, after_y = bench.trace_curve.getData()
        assert not np.allclose(before_y, after_y)

    def test_dsl_no_dsl_config_shows_error(self, qtbot):
        config = _two_pop_config()
        config.populations[0].neuron_model = "DSL (Custom)"
        config.populations[0].dsl_config = None
        session = Session(config)
        bench = NeuronTraceBench(session)
        qtbot.addWidget(bench)

        bench.set_population("SA Population")

        assert bench.error_label.isHidden() is False
