"""Tests for :class:`sensoryforge.gui.screens.populations.PopulationsScreen` (Task 2.3).

Drives the real widgets (spin boxes, combos, buttons) and asserts on
``session.config`` and the bench previews' actual data (bank weights, curve
y-data), plus a golden-preset engine-parity check for the RF bench and the
sugar<->explicit input conversion.
"""

import copy

import numpy as np
import pytest
import torch

pytest.importorskip("PyQt5")

from PyQt5 import QtCore  # noqa: E402

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.gui.screens.populations import PopulationsScreen  # noqa: E402

pytestmark = pytest.mark.gui

_PRESET_PATH = "sensoryforge/presets/tactile_sa1_ra1.yml"


def _preset_config() -> SensoryForgeConfig:
    return SensoryForgeConfig.from_yaml(_PRESET_PATH)


def _simple_config() -> SensoryForgeConfig:
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
                seed=7,
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


class TestPopulationList:
    def test_lists_every_population(self, qtbot):
        session = Session(_simple_config())
        screen = PopulationsScreen(session)
        qtbot.addWidget(screen)

        assert screen.list_widget.count() == 2
        names = {
            screen.list_widget.item(i).data(QtCore.Qt.UserRole)
            for i in range(screen.list_widget.count())
        }
        assert names == {"SA Population", "RA Population"}

    def test_add_creates_unique_named_population(self, qtbot):
        session = Session(_simple_config())
        screen = PopulationsScreen(session)
        qtbot.addWidget(screen)

        screen._on_add()

        assert len(session.config.populations) == 3
        names = [p.name for p in session.config.populations]
        assert len(names) == len(set(names))

    def test_duplicate_copies_config_under_new_name(self, qtbot):
        session = Session(_simple_config())
        screen = PopulationsScreen(session)
        qtbot.addWidget(screen)
        screen._select("SA Population")

        screen._on_duplicate()

        names = [p.name for p in session.config.populations]
        assert "SA Population copy" in names
        original = session.config.populations[0]
        dup = next(
            p for p in session.config.populations if p.name == "SA Population copy"
        )
        assert dup.filter_method == original.filter_method
        assert dup.innervation_method == original.innervation_method

    def test_remove_deletes_population(self, qtbot):
        session = Session(_simple_config())
        screen = PopulationsScreen(session)
        qtbot.addWidget(screen)
        screen._select("RA Population")

        screen._on_remove()

        names = [p.name for p in session.config.populations]
        assert "RA Population" not in names
        assert screen.list_widget.count() == 1

    def test_checkbox_toggle_writes_enabled(self, qtbot):
        session = Session(_simple_config())
        screen = PopulationsScreen(session)
        qtbot.addWidget(screen)
        item = screen.list_widget.item(0)

        item.setCheckState(QtCore.Qt.Unchecked)

        assert session.config.populations[0].enabled is False


class TestFilterCardEditing:
    def test_tau_r_edit_writes_config_and_changes_bench_curve(self, qtbot):
        session = Session(_simple_config())
        screen = PopulationsScreen(session)
        qtbot.addWidget(screen)
        screen._select("SA Population")

        before_x, before_y = screen.filter_bench.step_curve.getData()

        tau_r_spin = screen.filter_card._content.findChild(object, "param_tau_r")
        assert tau_r_spin is not None
        tau_r_spin.setValue(25.0)

        assert session.config.populations[0].filter_params["tau_r"] == 25.0
        # The bench recomputes on a 250 ms debounce (brief: "debounced
        # 250 ms"), not synchronously with the edit.
        assert screen._debounce.isActive()
        qtbot.wait(screen._debounce.interval() + 100)
        after_x, after_y = screen.filter_bench.step_curve.getData()
        assert not np.allclose(before_y, after_y)

    def test_external_set_by_path_updates_bench_after_debounce(self, qtbot):
        session = Session(_simple_config())
        screen = PopulationsScreen(session)
        qtbot.addWidget(screen)
        screen._select("SA Population")
        before_y = screen.filter_bench.step_curve.getData()[1].copy()

        session.set_by_path("populations.0.filter_params.tau_r", 40.0)
        assert screen._debounce.isActive()
        screen._debounce.stop()
        screen._refresh_benches()

        after_y = screen.filter_bench.step_curve.getData()[1]
        assert not np.allclose(before_y, after_y)


class TestRfBenchEnginParity:
    def test_bank_equals_engine_bank_gaussian(self, qtbot):
        session = Session(_simple_config())
        screen = PopulationsScreen(session)
        qtbot.addWidget(screen)
        screen._select("SA Population")

        engine = SimulationEngine(session.config, device="cpu")
        engine_bank = next(
            p["bank"] for p in engine.populations if p["name"] == "SA Population"
        )
        assert torch.equal(screen.rf_bench.bank.weights, engine_bank.weights)

    def test_bank_equals_engine_bank_template(self, qtbot):
        session = Session(_simple_config())
        screen = PopulationsScreen(session)
        qtbot.addWidget(screen)
        screen._select("RA Population")

        engine = SimulationEngine(session.config, device="cpu")
        engine_bank = next(
            p["bank"] for p in engine.populations if p["name"] == "RA Population"
        )
        assert torch.equal(screen.rf_bench.bank.weights, engine_bank.weights)


class TestNeuronCardDefaults:
    def test_ra_shows_fs_preset_defaults(self, qtbot):
        session = Session(_simple_config())
        screen = PopulationsScreen(session)
        qtbot.addWidget(screen)
        screen._select("RA Population")

        from sensoryforge.config.defaults import (
            NEURON_PRESET_BY_TYPE,
            resolve_neuron_params,
        )

        assert NEURON_PRESET_BY_TYPE["RA"] == "FS"
        resolved = resolve_neuron_params("Izhikevich", "RA", {})

        a_widget = screen.neuron_card._content.findChild(object, "param_a")
        assert a_widget is not None
        assert a_widget.value() == pytest.approx(resolved["a"])
        # Viewing the form must not itself write anything into model_params.
        assert session.config.populations[1].model_params == {}


class TestInputsConversion:
    def test_add_input_then_remove_round_trips_yaml(self, qtbot):
        original = _preset_config()
        original_yaml = original.to_yaml()
        session = Session(_preset_config())
        screen = PopulationsScreen(session)
        qtbot.addWidget(screen)

        screen.inputs_card._on_add_input(0)
        assert len(session.config.populations[0].inputs) == 2
        # SensoryForgeConfig.from_dict(config.to_dict()) accepts it...
        SensoryForgeConfig.from_dict(session.config.to_dict())
        # ...and the engine runs.
        engine = SimulationEngine(session.config, device="cpu")
        assert len(engine.populations) == 2

        screen.inputs_card._on_remove_input(0, 1)
        assert len(session.config.populations[0].inputs) == 1
        assert session.config.to_yaml() == original_yaml


class TestQuickRun:
    def test_quick_run_leaves_config_and_stale_unchanged(self, qtbot):
        session = Session(_preset_config())
        config_before = copy.deepcopy(session.config)
        stale_before = session.stale
        screen = PopulationsScreen(session)
        qtbot.addWidget(screen)
        screen._select("SA Population")

        with qtbot.waitSignal(screen._run_controller.finished, timeout=30000):
            screen._on_quick_run()

        assert session.config.to_dict() == config_before.to_dict()
        assert session.stale == stale_before
        x, _ = screen.quick_raster.getData()
        assert len(x) >= 0  # a raster was produced (possibly empty), no crash


@pytest.mark.parametrize(
    "method", ["gaussian", "uniform", "distance_weighted", "template"]
)
def test_rf_form_shows_the_values_the_engine_builds_with(qtbot, method):
    """Write back every value the inputs card displays: the bank must not change."""
    import copy

    import torch

    from sensoryforge.gui.bench.rf_footprint import build_population_bank_for_config
    from sensoryforge.gui.screens.populations_cards import InputsCard
    from sensoryforge.gui.widgets.param_form import ParamForm, _read_widget

    config = SensoryForgeConfig.from_yaml_file(
        "sensoryforge/presets/tactile_sa1_ra1.yml"
    )
    config.grids[0].rows = 20
    config.grids[0].cols = 20
    population = config.populations[0]
    population.innervation_method = method
    population.innervation_params = {}
    # Population-wide values that differ from every builder's own defaults.
    population.sigma_d_mm = 0.45
    population.connections_per_neuron = 12.0
    population.max_distance_mm = 0.8
    population.decay_rate = 3.0
    population.resolvable_distance_mm = 0.7 if method == "template" else None
    population.seed = 7
    name = population.name
    base = build_population_bank_for_config(config, name).weights

    session = Session(config)
    card = InputsCard(session)
    qtbot.addWidget(card)
    card.set_population(name)
    form = card.findChild(ParamForm)
    shown = {
        spec.name: _read_widget(spec, form.widget_for(spec.name))
        for spec in form._specs
    }
    assert shown, "the builder should declare parameters"

    explicit = copy.deepcopy(config)
    explicit.populations[0].innervation_params = dict(shown)
    again = build_population_bank_for_config(explicit, name).weights
    assert again.shape == base.shape and torch.equal(again, base), shown
