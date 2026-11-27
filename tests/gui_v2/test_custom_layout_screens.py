"""Plugin layouts in the app: the lists, their settings and a missing plugin."""

import pytest

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

import torch  # noqa: E402

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
)
from sensoryforge.core.grid_base import BaseGrid  # noqa: E402
from sensoryforge.gui.screens.populations_cards import (  # noqa: E402
    NeuronLayoutCard,
)
from sensoryforge.gui.screens.sensors import SensorsScreen  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.gui.validation import validate  # noqa: E402
from sensoryforge.register_components import register_all  # noqa: E402
from sensoryforge.registry import GRID_REGISTRY  # noqa: E402
from sensoryforge.stimuli.base import ParamSpec  # noqa: E402


class Line(BaseGrid):
    """``n`` points on the area's diagonal."""

    def __init__(self, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), device="cpu", n=7):
        super().__init__(xlim, ylim, device)
        self.n = int(n)

    def get_all_coordinates(self):
        x = torch.linspace(*self.xlim, self.n)
        y = torch.linspace(*self.ylim, self.n)
        return torch.stack([x, y], dim=1)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def get_param_spec(cls):
        return [ParamSpec("n", dtype="int", default=7, min_val=1)]


class Random(Line):
    """Takes a seed, and lists it as a parameter as the scaffold does."""

    def __init__(self, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), device="cpu", n=7, seed=0):
        super().__init__(xlim, ylim, device, n)

    @classmethod
    def get_param_spec(cls):
        return super().get_param_spec() + [ParamSpec("seed", dtype="int", default=0)]


@pytest.fixture(autouse=True)
def _layouts():
    register_all()
    GRID_REGISTRY.register("gui_line", Line)
    GRID_REGISTRY.register("gui_random", Random)
    yield
    for name in ("gui_line", "gui_random"):
        GRID_REGISTRY._registry.pop(name, None)


def _session(grid=None, **population) -> Session:
    population.setdefault("name", "SA")
    population.setdefault("neuron_type", "SA")
    population.setdefault("target_grid", "skin")
    return Session(
        SensoryForgeConfig(
            grids=[grid or GridConfig(name="skin", rows=10, cols=10, spacing=0.15)],
            populations=[PopulationConfig(**population)],
        )
    )


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def _sensors(qtbot, session):
    screen = SensorsScreen(session)
    qtbot.addWidget(screen)
    return screen


def _card(qtbot, session):
    card = NeuronLayoutCard(session)
    qtbot.addWidget(card)
    card.set_population("SA")
    return card


def _choose(combo, text):
    index = combo.findText(text)
    assert index >= 0, (text, _items(combo))
    combo.setCurrentIndex(index)


# ----------------------------------------------------------------- sensors


def _arrangement(screen):
    return screen._param_form.widget_for("arrangement")


def test_the_sensor_list_offers_installed_layouts(qtbot):
    screen = _sensors(qtbot, _session())

    items = _items(_arrangement(screen))

    assert items[:2] == ["grid", "hex"]
    assert {"gui_line", "gui_random", "composite"} <= set(items)


def test_a_plugin_sensor_layout_shows_its_settings_and_writes_them(qtbot):
    session = _session()
    screen = _sensors(qtbot, session)

    _choose(_arrangement(screen), "gui_line")
    screen.layout_form.widget_for("n").setValue(4)

    assert session.config.grids[0].layout_params == {"n": 4}
    assert "4 receptors" in screen.grid_list.item(0).text()


def test_a_sensor_layouts_seed_is_not_offered_as_a_setting(qtbot):
    screen = _sensors(qtbot, _session())

    _choose(_arrangement(screen), "gui_random")

    with pytest.raises(KeyError):
        screen.layout_form.widget_for("seed")
    assert screen._param_form.widget_for("seed").isEnabled()


def test_choosing_a_plugin_sensor_layout_clears_what_it_cannot_take(qtbot):
    session = _session(
        GridConfig(
            name="skin", arrangement="hex", rows=10, cols=10, density=20.0, seed=3
        )
    )
    screen = _sensors(qtbot, session)

    _choose(_arrangement(screen), "gui_line")

    grid = session.config.grids[0]
    assert grid.density is None and grid.seed is None
    assert not screen._param_form.widget_for("density").isEnabled()
    assert not screen._param_form.widget_for("seed").isEnabled()
    assert validate(session.config) == {}


def test_choosing_another_sensor_layout_drops_the_old_settings(qtbot):
    session = _session(
        GridConfig(name="skin", arrangement="gui_line", layout_params={"n": 3})
    )
    screen = _sensors(qtbot, session)

    _choose(_arrangement(screen), "grid")

    assert session.config.grids[0].layout_params == {}
    assert screen.layout_form is None


def test_settings_written_before_the_layout_is_named_are_kept(qtbot):
    session = _session()
    screen = _sensors(qtbot, session)  # held: an unreferenced screen is collected

    session.set_by_path("grids.0.layout_params", {"n": 4})
    session.set_by_path("grids.0.arrangement", "gui_line")

    assert session.config.grids[0].layout_params == {"n": 4}
    assert screen.layout_form.widget_for("n").value() == 4


def test_one_plugin_layouts_settings_are_dropped_for_another(qtbot):
    session = _session(
        GridConfig(name="skin", arrangement="gui_line", layout_params={"n": 3})
    )
    screen = _sensors(qtbot, session)

    _choose(_arrangement(screen), "gui_random")

    assert session.config.grids[0].layout_params == {}


def test_a_missing_sensor_plugin_keeps_its_name_and_is_reported(qtbot):
    session = _session(GridConfig(name="skin", arrangement="gone_layout"))
    screen = _sensors(qtbot, session)

    assert _arrangement(screen).currentText() == "gone_layout"
    assert session.config.grids[0].arrangement == "gone_layout"
    assert "gone_layout" in validate(session.config)["grids.0"]


# ----------------------------------------------------------------- neurons


def test_the_neuron_list_offers_built_ins_then_installed_layouts(qtbot):
    card = _card(qtbot, _session())

    items = _items(card.arrangement_combo)

    assert items[:5] == ["grid", "poisson", "hex", "jittered_grid", "blue_noise"]
    assert {"gui_line", "gui_random"} <= set(items[5:])


def test_a_plugin_neuron_layout_disables_the_counts_and_writes_settings(qtbot):
    session = _session()
    card = _card(qtbot, session)

    _choose(card.arrangement_combo, "gui_line")

    assert session.config.populations[0].neuron_arrangement == "gui_line"
    for spin in (card.per_row_spin, card.rows_spin, card.cols_spin):
        assert not spin.isEnabled()
    card.layout_form.widget_for("n").setValue(5)
    assert session.config.populations[0].neuron_layout_params == {"n": 5}
    assert validate(session.config) == {}


def test_going_back_to_a_built_in_neuron_layout_restores_the_card(qtbot):
    session = _session(neuron_arrangement="gui_line", neuron_layout_params={"n": 5})
    card = _card(qtbot, session)

    _choose(card.arrangement_combo, "grid")

    assert session.config.populations[0].neuron_layout_params == {}
    assert card.per_row_spin.isEnabled()
    assert card.layout_form is None


def test_a_missing_neuron_plugin_keeps_its_name_and_is_reported(qtbot):
    session = _session(neuron_arrangement="gone_layout")
    card = _card(qtbot, session)

    assert card.arrangement_combo.currentText() == "gone_layout"
    assert session.config.populations[0].neuron_arrangement == "gone_layout"
    assert any(
        "gone_layout" in message for message in validate(session.config).values()
    )


def test_a_misspelled_name_is_not_shown_as_a_different_layout(qtbot):
    card = _card(qtbot, _session(neuron_arrangement="Hex"))

    assert card.arrangement_combo.currentText() == "Hex"


def test_a_duplicated_grid_has_its_own_layout_settings(qtbot):
    session = _session(
        GridConfig(name="skin", arrangement="gui_line", layout_params={"n": 3})
    )
    screen = _sensors(qtbot, session)

    screen.btn_duplicate.click()
    screen.layout_form.widget_for("n").setValue(9)

    assert [g.layout_params for g in session.config.grids] == [{"n": 3}, {"n": 9}]
