"""The Stimulus Designer can load its own saved configuration (F-064).

``_set_spin`` cast every value to ``float`` before handing it to the
widget. ``QSpinBox`` holds an int and PyQt raises ``TypeError`` rather
than coercing, so the tab could not load the dictionary its own
``get_config()`` produced. Nothing caught it because no test called
``set_config()`` with a full config; Wave Q hit it while working around
it and reported it rather than only routing past it.

A tab that cannot reload its own saved state breaks the save-and-reopen
path for every project, so this pins the whole round trip rather than the
one spinbox that happened to raise.
"""

import pytest

pytest.importorskip("PyQt5")

import sys  # noqa: E402

from PyQt5 import QtWidgets  # noqa: E402

from sensoryforge.gui.tabs.mechanoreceptor_tab import MechanoreceptorTab  # noqa: E402
from sensoryforge.gui.tabs.stimulus_tab import StimulusDesignerTab  # noqa: E402

pytestmark = pytest.mark.gui


def _ensure_app():
    """One QApplication per process, as the other GUI tests do."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    return app


@pytest.fixture
def qapp():
    return _ensure_app()


@pytest.fixture
def tab(qapp):
    return StimulusDesignerTab(MechanoreceptorTab())


class TestConfigRoundTrip:
    def test_set_config_accepts_its_own_get_config(self, tab):
        """The failure this reproduces: set_config raised TypeError."""
        tab.set_config(tab.get_config())

    def test_the_round_trip_changes_nothing(self, tab):
        """Accepting the dict is not enough; it has to survive unchanged."""
        before = tab.get_config()
        tab.set_config(before)
        after = tab.get_config()
        changed = {k for k in before if before.get(k) != after.get(k)}
        assert not changed, f"these fields changed across the round trip: {changed}"

    def test_a_modified_value_survives(self, tab):
        """A real edit must come back, not just the defaults."""
        cfg = tab.get_config()
        cfg["texture"] = {**cfg.get("texture", {}), "edge_count": 9}
        tab.set_config(cfg)
        assert tab.get_config()["texture"]["edge_count"] == 9


class TestSetSpin:
    """The cast has to follow the widget, not the value."""

    def test_an_integer_spinbox_accepts_a_float_valued_config(self, qapp):
        widget = QtWidgets.QSpinBox()
        widget.setRange(0, 100)
        StimulusDesignerTab._set_spin(widget, 7.0)
        assert widget.value() == 7

    def test_a_double_spinbox_keeps_its_fraction(self, qapp):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(0.0, 100.0)
        widget.setDecimals(3)
        StimulusDesignerTab._set_spin(widget, 2.5)
        assert widget.value() == pytest.approx(2.5)

    def test_an_integer_spinbox_rounds_rather_than_truncating(self, qapp):
        widget = QtWidgets.QSpinBox()
        widget.setRange(0, 100)
        StimulusDesignerTab._set_spin(widget, 7.6)
        assert widget.value() == 8

    def test_signals_stay_blocked_during_the_set(self, qapp):
        """A config load must not fire the live-preview handler per field."""
        widget = QtWidgets.QSpinBox()
        widget.setRange(0, 100)
        fired = []
        widget.valueChanged.connect(fired.append)
        StimulusDesignerTab._set_spin(widget, 5)
        assert fired == [], "setting a value during a config load emitted a signal"
