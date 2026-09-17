"""The derived-lattice warning fires only on sizes the user set (F-069).

A receptive-field builder like ``template`` derives its own neuron lattice,
so any lattice size on the population is ignored, and the engine says so.
It used to say so unconditionally: ``neurons_per_row`` always exists on
``PopulationConfig`` with a default of 10, so every population using such a
builder warned "neurons_per_row=10 ... ignored", including every run of the
shipped tactile preset, which never sets it. A warning that fires on
defaults trains people to ignore warnings, and then the one that matters is
missed.
"""

import warnings

import pytest

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.core.simulation_engine import (
    SimulationEngine,
    _lattice_fields_set_by_user,
)


def _config(**population_kwargs):
    return SensoryForgeConfig(
        grids=[
            GridConfig(name="G", arrangement="grid", rows=16, cols=16, spacing=0.15)
        ],
        populations=[
            PopulationConfig(
                name="P",
                neuron_type="SA",
                neuron_model="izhikevich",
                filter_method="sa",
                innervation_method="template",
                innervation_params={"resolvable_distance_mm": 0.6},
                **population_kwargs,
            )
        ],
        simulation=SimulationConfig(device="cpu", dt_ms=1.0),
    )


def _lattice_warnings(config):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SimulationEngine(config)
    return [w for w in caught if "derives its own neuron lattice" in str(w.message)]


class TestWhenItFires:
    def test_untouched_defaults_do_not_warn(self):
        assert _lattice_warnings(_config()) == []

    def test_a_set_neurons_per_row_warns_and_names_it(self):
        caught = _lattice_warnings(_config(neurons_per_row=7))
        assert len(caught) == 1
        message = str(caught[0].message)
        assert "neurons_per_row=7" in message
        assert "neuron_rows" not in message, "only the fields the user set are named"

    def test_several_set_fields_are_all_named(self):
        caught = _lattice_warnings(_config(neuron_rows=4, neuron_cols=5))
        assert len(caught) == 1
        message = str(caught[0].message)
        assert "neuron_rows=4" in message and "neuron_cols=5" in message
        assert " are ignored" in message

    def test_the_shipped_tactile_preset_runs_without_this_warning(self):
        from sensoryforge.presets import load_preset

        config = SensoryForgeConfig.from_dict(load_preset("tactile_sa1_ra1"))
        assert _lattice_warnings(config) == []


class TestTheHelper:
    def test_defaults_are_read_from_the_dataclass(self):
        assert _lattice_fields_set_by_user(PopulationConfig(name="P")) == []

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({"neurons_per_row": 3}, ["neurons_per_row=3"]),
            ({"neuron_rows": 2}, ["neuron_rows=2"]),
            ({"neuron_cols": 9}, ["neuron_cols=9"]),
        ],
    )
    def test_each_field_is_detected(self, kwargs, expected):
        assert (
            _lattice_fields_set_by_user(PopulationConfig(name="P", **kwargs))
            == expected
        )
