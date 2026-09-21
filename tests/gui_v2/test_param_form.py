"""Tests for :mod:`sensoryforge.gui.widgets.param_form`.

Covers :class:`ParamForm`: widget construction per ``dtype``/``choices``,
writing through the bound target (dict and dataclass) with
``session.notify``, reading back via ``session.configChanged`` and
``session.configReplaced``, advanced-row visibility, JSON round-trip for
list/dict-valued specs, and :func:`specs_for`'s case-insensitive lookup.
"""

import json

import pytest

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

from PyQt5 import QtWidgets  # noqa: E402

import sensoryforge.register_components as register_components  # noqa: E402
from sensoryforge.config.schema import SensoryForgeConfig  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.gui.widgets.param_form import ParamForm, specs_for  # noqa: E402
from sensoryforge.registry import FILTER_REGISTRY, GRID_REGISTRY  # noqa: E402
from sensoryforge.stimuli.base import ParamSpec  # noqa: E402

register_components.register_all()

PRESET_PATH = "sensoryforge/presets/tactile_sa1_ra1.yml"


@pytest.fixture
def session(qtbot) -> Session:
    """A Session over the ``tactile_sa1_ra1`` preset."""
    config = SensoryForgeConfig.from_yaml(PRESET_PATH)
    return Session(config)


def _pop_filter_form(session, *, advanced=False):
    specs = specs_for(FILTER_REGISTRY, "sa")
    target = session.config.populations[0].filter_params
    return (
        ParamForm(
            specs, target, session, "populations.0.filter_params", advanced=advanced
        ),
        specs,
    )


# ----------------------------------------------------------------- dict target


def test_dict_target_edit_writes_and_notifies(session, qtbot):
    form, _specs = _pop_filter_form(session)
    widget = form.widget_for("tau_r")
    assert isinstance(widget, QtWidgets.QDoubleSpinBox)

    with qtbot.waitSignal(session.configChanged, timeout=1000) as blocker:
        widget.setValue(widget.value() + 1.0)
    assert blocker.args == ["populations.0.filter_params.tau_r"]
    assert session.config.populations[0].filter_params["tau_r"] == pytest.approx(
        widget.value()
    )


def test_dict_target_missing_key_shows_default(session, qtbot):
    target = session.config.populations[0].filter_params
    assert "k2" not in target or True  # sanity: exercised regardless of preset content
    specs = specs_for(FILTER_REGISTRY, "sa")
    k2_spec = next(s for s in specs if s.name == "k2")
    target.pop("k2", None)
    form = ParamForm(specs, target, session, "populations.0.filter_params")
    widget = form.widget_for("k2")
    assert widget.value() == pytest.approx(k2_spec.default)
    assert "k2" not in target


def test_external_set_by_path_updates_widget_without_reemitting_write(session, qtbot):
    form, _specs = _pop_filter_form(session)
    widget = form.widget_for("tau_r")
    seen = []
    session.configChanged.connect(lambda p: seen.append(p))

    session.set_by_path("populations.0.filter_params.tau_r", 12.5)
    assert widget.value() == pytest.approx(12.5)
    # Exactly one emission (from set_by_path itself) -- the widget update was
    # applied under blockSignals, not by round-tripping through _write.
    assert seen == ["populations.0.filter_params.tau_r"]


def test_unrelated_path_change_does_not_touch_widget(session, qtbot):
    form, _specs = _pop_filter_form(session)
    widget = form.widget_for("tau_r")
    before = widget.value()
    session.notify("populations.1.filter_params.tau_r")
    assert widget.value() == pytest.approx(before)


# ------------------------------------------------------------- dataclass target


def test_dataclass_target_round_trips_field(session, qtbot):
    grid_cfg = session.config.grids[0]
    specs = [
        ParamSpec(
            "rows", dtype="int", default=80, min_val=1, max_val=500, group="Layout"
        ),
    ]
    form = ParamForm(specs, grid_cfg, session, "grids.0")
    widget = form.widget_for("rows")
    assert isinstance(widget, QtWidgets.QSpinBox)
    assert widget.value() == grid_cfg.rows

    with qtbot.waitSignal(session.configChanged, timeout=1000) as blocker:
        widget.setValue(widget.value() + 5)
    assert blocker.args == ["grids.0.rows"]
    assert grid_cfg.rows == widget.value()


# ------------------------------------------------------------------------ choices


def test_choices_spec_produces_combo_box(session, qtbot):
    specs = [
        ParamSpec(
            "arrangement",
            dtype="str",
            default="grid",
            choices=["grid", "poisson", "hex"],
        ),
    ]
    grid_cfg = session.config.grids[0]
    form = ParamForm(specs, grid_cfg, session, "grids.0")
    widget = form.widget_for("arrangement")
    assert isinstance(widget, QtWidgets.QComboBox)
    assert widget.currentData() == "grid"


# --------------------------------------------------------------------- advanced


def test_advanced_rows_hidden_by_default_and_shown_on_toggle(session, qtbot):
    # Mixed group (one advanced, one not) so the group itself stays visible
    # and the assertion actually exercises the per-row visibility, not just
    # the group-level "all advanced" collapse.
    specs = [
        ParamSpec("tau_r", dtype="float", default=5.0, group="Temporal"),
        ParamSpec("k2", dtype="float", default=3.0, advanced=True, group="Temporal"),
    ]
    target = session.config.populations[0].filter_params
    form = ParamForm(specs, target, session, "populations.0.filter_params")
    qtbot.addWidget(form)
    form.show()
    qtbot.waitExposed(form)

    adv_widget = form.widget_for("k2")
    assert adv_widget.isVisible() is False

    form.set_advanced(True)
    assert adv_widget.isVisible() is True

    form.set_advanced(False)
    assert adv_widget.isVisible() is False


def test_group_entirely_advanced_is_hidden_in_basic_mode(session, qtbot):
    specs = [
        ParamSpec("tau_r", dtype="float", default=5.0, group="Temporal"),
        ParamSpec(
            "clip_to_positive",
            dtype="bool",
            default=False,
            advanced=True,
            group="Options",
        ),
    ]
    target = session.config.populations[0].filter_params
    form = ParamForm(specs, target, session, "populations.0.filter_params")
    qtbot.addWidget(form)
    form.show()
    qtbot.waitExposed(form)

    group_box = form._group_boxes["Options"]
    assert group_box.isVisible() is False
    form.set_advanced(True)
    assert group_box.isVisible() is True


# -------------------------------------------------------------------------- JSON


def test_list_field_round_trips_through_json(session, qtbot):
    specs = [ParamSpec("weight_range", dtype="str", default=[0.0, 1.0])]
    target = {"weight_range": [0.1, 0.9]}
    form = ParamForm(specs, target, session, "populations.0.innervation_params")
    widget = form.widget_for("weight_range")
    assert isinstance(widget, QtWidgets.QLineEdit)
    assert json.loads(widget.text()) == [0.1, 0.9]

    with qtbot.waitSignal(session.configChanged, timeout=1000):
        widget.setText("[0.2, 0.8, 1.0]")
        widget.editingFinished.emit()
    assert target["weight_range"] == [0.2, 0.8, 1.0]


def test_invalid_json_leaves_value_unchanged_and_marks_widget(session, qtbot):
    specs = [ParamSpec("weight_range", dtype="str", default=[0.0, 1.0])]
    target = {"weight_range": [0.1, 0.9]}
    form = ParamForm(specs, target, session, "populations.0.innervation_params")
    widget = form.widget_for("weight_range")

    seen = []
    session.configChanged.connect(lambda p: seen.append(p))
    widget.setText("not json")
    widget.editingFinished.emit()

    assert target["weight_range"] == [0.1, 0.9]
    assert seen == []
    assert widget.objectName() == "Invalid"
    assert widget.toolTip()


# ----------------------------------------------------------------------- refresh


def test_config_replaced_refreshes_from_target(session, qtbot):
    form, _specs = _pop_filter_form(session)
    widget = form.widget_for("tau_r")
    session.config.populations[0].filter_params["tau_r"] = 99.0
    session.configReplaced.emit()
    assert widget.value() == pytest.approx(99.0)


# ------------------------------------------------------------------------ specs_for


def test_specs_for_is_case_insensitive():
    lower = specs_for(FILTER_REGISTRY, "sa")
    upper = specs_for(FILTER_REGISTRY, "SA")
    assert [s.name for s in lower] == [s.name for s in upper]
    assert len(lower) > 0


def test_specs_for_grid_registry_case_insensitive():
    lower = specs_for(GRID_REGISTRY, "grid")
    upper = specs_for(GRID_REGISTRY, "GRID")
    assert [s.name for s in lower] == [s.name for s in upper]


def test_both_tactile_filters_declare_their_parameters():
    """An empty spec renders an empty form, which reads as "no settings".

    SA gained its spec with ParamForm; RA must match, and each default must
    equal the resolver's value so the form opens on what the engine uses.
    """
    from sensoryforge.config.defaults import FILTER_DEFAULTS
    from sensoryforge.register_components import register_all
    from sensoryforge.registry import FILTER_REGISTRY

    register_all()
    for name in ("sa", "ra"):
        specs = {s.name: s for s in FILTER_REGISTRY.get_param_spec(name)}
        assert specs, f"{name} filter declares no parameters"
        for key, value in FILTER_DEFAULTS[name].items():
            assert key in specs, f"{name}: {key} missing from get_param_spec()"
            assert specs[key].default == value, f"{name}.{key} default drifted"


def test_empty_specs_show_a_notice_not_a_blank_form(session, qtbot):
    form = ParamForm([], {}, session, "populations.0.model_params")
    qtbot.addWidget(form)
    assert form.empty_notice is not None
    assert "no editable parameters" in form.empty_notice.text()


def test_nonempty_specs_show_no_notice(session, qtbot):
    from sensoryforge.registry import FILTER_REGISTRY

    form = ParamForm(
        FILTER_REGISTRY.get_param_spec("sa"),
        session.config.populations[0].filter_params,
        session,
        "populations.0.filter_params",
    )
    qtbot.addWidget(form)
    assert form.empty_notice is None
