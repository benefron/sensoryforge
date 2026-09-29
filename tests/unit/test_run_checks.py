"""Tests for :mod:`sensoryforge.gui.execution.run_checks` (ledger F-93b91b1).

No Qt: the module is pure, so the CLI and the batch runner can reuse it.
"""

from __future__ import annotations

import pytest
import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
    StimulusConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.gui.execution.render import render_for_config
from sensoryforge.gui.execution.run_checks import (
    SilentPopulation,
    describe_silent,
    silent_populations,
)

#: A leaky integrator with no threshold/reset: an analog readout (Wave N).
LEAKY_INTEGRATOR = {
    "equations": "dv/dt = (-(v - v_rest) + R*I) / tau_m",
    "parameters": {"v_rest": -65.0, "R": 1.0, "tau_m": 10.0},
    "state_vars": {"v": -65.0},
}


def _spikes(fired: bool, shape=(1, 20, 4)) -> torch.Tensor:
    spikes = torch.zeros(shape)
    if fired:
        spikes[..., 3, 1] = 2.0
    return spikes


class TestSilentPopulations:
    def test_a_silent_spiking_population_is_named_with_its_peak_input(self):
        filtered = torch.linspace(-0.5, 0.25, 80).reshape(1, 20, 4)
        silent = silent_populations(
            {"SA": {"spikes": _spikes(False), "filtered": filtered}}
        )
        assert silent == [SilentPopulation("SA", pytest.approx(0.25))]

    def test_a_population_that_fired_is_not_named(self):
        results = {"RA": {"spikes": _spikes(True), "filtered": torch.ones(1, 20, 4)}}
        assert silent_populations(results) == []

    def test_an_analog_population_is_never_named(self):
        # A flat state trace and no spikes key: nothing that could fire.
        results = {
            "Leaky": {"state": torch.zeros(1, 20, 4), "filtered": torch.zeros(1, 20, 4)}
        }
        assert silent_populations(results) == []

    def test_none_counts_as_missing(self):
        # The shape ResultsView populations are passed in.
        results = {
            "Leaky": {"spikes": None, "state": torch.zeros(20, 4), "filtered": None},
            "SA": {"spikes": _spikes(False, (20, 4)), "state": None, "filtered": None},
        }
        assert silent_populations(results) == [SilentPopulation("SA", None)]

    def test_without_filtered_the_peak_is_unknown(self):
        assert silent_populations({"SA": {"spikes": _spikes(False)}}) == [
            SilentPopulation("SA", None)
        ]

    def test_every_silent_population_in_results_order(self):
        results = {
            "B": {"spikes": _spikes(False)},
            "fires": {"spikes": _spikes(True)},
            "A": {"spikes": _spikes(False)},
        }
        assert [pop.name for pop in silent_populations(results)] == ["B", "A"]

    def test_a_bundle_shaped_population_without_batch_dim(self):
        results = {
            "SA": {"spikes": _spikes(False, (20, 4)), "filtered": torch.ones(20, 4)}
        }
        assert silent_populations(results) == [SilentPopulation("SA", 1.0)]


class TestDescribeSilent:
    def test_nothing_silent_is_no_text(self):
        assert describe_silent([]) == ""

    def test_names_each_population_and_its_peak_in_ma(self):
        text = describe_silent(
            [SilentPopulation("SA", 0.000123), SilentPopulation("RA", None)]
        )
        assert "No spikes from 2 populations" in text
        assert "SA: peak neuron input 0.000123 mA" in text
        assert "RA: neuron input not recorded" in text
        assert "input gain" in text


def _engine_config(sa_gain: float) -> SensoryForgeConfig:
    common = dict(
        target_grid="Main",
        filter_method="sa",
        innervation_method="gaussian",
        neurons_per_row=3,
    )
    return SensoryForgeConfig(
        grids=[GridConfig(name="Main", rows=8, cols=8, spacing=0.2)],
        stimulus=StimulusConfig(
            type="gaussian", target_layer="Main", amplitude=40.0, spread=1.0
        ),
        populations=[
            PopulationConfig(
                name="SA", neuron_type="SA", input_gain=sa_gain, seed=6, **common
            ),
            PopulationConfig(
                name="Leaky",
                neuron_type="SA",
                neuron_model="dsl",
                dsl_config=LEAKY_INTEGRATOR,
                seed=5,
                **common,
            ),
        ],
        simulation=SimulationConfig(
            device="cpu", dt_ms=1.0, integrate_dt_ms=1.0, duration_ms=30.0, seed=3
        ),
    )


@pytest.mark.parametrize("sa_gain, silent_names", [(50.0, []), (1e-6, ["SA"])])
def test_on_a_real_engine_run(sa_gain, silent_names):
    """The engine's own result dict: the gain decides, the analog one never counts."""
    config = _engine_config(sa_gain)
    rendered = render_for_config(config, duration_ms=config.simulation.duration_ms)
    results = SimulationEngine(config).run(rendered.stimulus, return_intermediates=True)

    silent = silent_populations(results)

    assert [pop.name for pop in silent] == silent_names
    for pop in silent:
        assert pop.peak_input_ma == pytest.approx(
            float(results[pop.name]["filtered"].max())
        )
