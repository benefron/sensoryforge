"""Qt test: the Circuit tab inspector renders widgets from get_param_spec()
(Phase 3, Wave P, P1).

Marked gui; run alone (F-016, see docs/development/handover/phase1_tasks.md
appendix).
"""

import sys

import pytest

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

_APP = None


def _ensure_app():
    global _APP
    from PyQt5 import QtWidgets

    _APP = QtWidgets.QApplication.instance()
    if _APP is None:
        _APP = QtWidgets.QApplication(sys.argv[:1])


def _three_kind_specs():
    """A plugin-shaped ParamSpec list: float, bool, and an enum (choices),
    with the enum marked advanced -- P1's "Done when" asks for exactly these
    three kinds plus the advanced flag."""
    from sensoryforge.stimuli.base import ParamSpec

    return [
        ParamSpec(
            "amplitude",
            dtype="float",
            default=1.0,
            min_val=0.0,
            max_val=10.0,
            unit="mA",
            group="Amplitude",
        ),
        ParamSpec(
            "rectify",
            dtype="bool",
            default=False,
            group="Shape",
        ),
        ParamSpec(
            "mode",
            dtype="str",
            default="linear",
            choices=["linear", "log", "exp"],
            group="Shape",
            advanced=True,
        ),
    ]


def test_build_param_form_renders_one_widget_kind_per_spec():
    _ensure_app()
    from PyQt5 import QtWidgets

    from sensoryforge.gui.circuit.inspector import build_param_form

    specs = _three_kind_specs()
    form = build_param_form(specs, {}, title="PluginComponent", expert_mode=True)

    amp_widget = form.findChild(QtWidgets.QWidget, "param_amplitude")
    rectify_widget = form.findChild(QtWidgets.QWidget, "param_rectify")
    mode_widget = form.findChild(QtWidgets.QWidget, "param_mode")

    assert isinstance(amp_widget, QtWidgets.QDoubleSpinBox)
    assert amp_widget.suffix().strip() == "mA"
    assert isinstance(rectify_widget, QtWidgets.QCheckBox)
    assert isinstance(mode_widget, QtWidgets.QComboBox)
    assert [mode_widget.itemText(i) for i in range(mode_widget.count())] == [
        "linear",
        "log",
        "exp",
    ]


def test_advanced_param_hidden_unless_expert_mode():
    _ensure_app()
    from PyQt5 import QtWidgets

    from sensoryforge.gui.circuit.inspector import build_param_form

    specs = _three_kind_specs()

    basic_form = build_param_form(specs, {}, expert_mode=False)
    mode_widget_basic = basic_form.findChild(QtWidgets.QWidget, "param_mode")
    assert mode_widget_basic.isVisible() is False

    expert_form = build_param_form(specs, {}, expert_mode=True)
    mode_widget_expert = expert_form.findChild(QtWidgets.QWidget, "param_mode")
    # Not shown() by a real window, but not force-hidden either.
    assert mode_widget_expert.isHidden() is False


def test_param_form_calls_on_change_with_new_value():
    _ensure_app()
    from PyQt5 import QtWidgets

    from sensoryforge.gui.circuit.inspector import build_param_form

    changes = []
    form = build_param_form(
        _three_kind_specs(), {}, lambda name, value: changes.append((name, value))
    )
    check = form.findChild(QtWidgets.QCheckBox, "param_rectify")
    check.setChecked(True)
    assert ("rectify", True) in changes


def test_empty_param_spec_shows_visible_notice_not_a_blank_panel():
    """P1's stated risk: a component (like MovingStimulus) whose
    get_param_spec() returns [] must not render as an indistinguishable
    blank panel."""
    _ensure_app()
    from PyQt5 import QtWidgets

    from sensoryforge.gui.circuit.inspector import (
        _EMPTY_SPEC_NOTICE,
        build_param_form,
    )

    form = build_param_form([], {}, title="EmptySpecComponent")
    notice = form.findChild(QtWidgets.QLabel, "empty_param_spec_notice")
    assert notice is not None
    assert notice.text() == _EMPTY_SPEC_NOTICE
    assert notice.isHidden() is False


def test_moving_stimulus_get_param_spec_is_actually_empty():
    """Guards the premise above: MovingStimulus really has no spec today."""
    from sensoryforge.stimuli.builder import MovingStimulus

    assert MovingStimulus.get_param_spec() == []


def test_node_inspector_renders_registered_plugin_component():
    """Registers a plugin-shaped filter at test time (in-process, unlike
    the P3 entry-point proof) and asserts the Circuit inspector renders its
    three param kinds by going through the real node -> registry ->
    get_param_spec() path, not a hand-built ParamSpec list."""
    _ensure_app()
    from PyQt5 import QtWidgets

    from sensoryforge.gui.circuit.inspector import build_node_inspector
    from sensoryforge.gui.circuit.nodes import NODE_CLASSES
    from sensoryforge.registry import FILTER_REGISTRY
    from sensoryforge.stimuli.base import ParamSpec

    class _PluginFilterForInspectorTest:
        @classmethod
        def get_param_spec(cls):
            return _three_kind_specs()

    if not FILTER_REGISTRY.is_registered("_plugin_filter_for_inspector_test"):
        FILTER_REGISTRY.register(
            "_plugin_filter_for_inspector_test", _PluginFilterForInspectorTest
        )

    node = NODE_CLASSES["Filter"]("filter_under_test")
    node.filter_method = "_plugin_filter_for_inspector_test"

    panel = build_node_inspector(node, expert_mode=True)
    assert panel.findChild(QtWidgets.QWidget, "param_amplitude") is not None
    assert panel.findChild(QtWidgets.QWidget, "param_rectify") is not None
    assert panel.findChild(QtWidgets.QWidget, "param_mode") is not None
