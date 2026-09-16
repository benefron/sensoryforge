"""Running a graph says so when it ignores a stimulus setting (F-061).

``StimulusConfig`` carries every field the schema defines; any one stimulus
class accepts a subset. The Circuit tab discovers that subset by retrying
and discarding whatever the constructor rejects, which is fine until it
discards something the user actually set. Then the graph on screen
describes a run that did not happen, and nothing says so.

Discarding a field still at its schema default is housekeeping and stays
quiet, so that a warning, when it does appear, means something.

Both helpers are pure, so these run without Qt.
"""

import pytest

from sensoryforge.config.schema import StimulusConfig
from sensoryforge.gui.tabs.circuit_tab import (
    _dropped_params_warning,
    _is_schema_default,
)


class TestIsSchemaDefault:
    """The predicate separating housekeeping from a real loss."""

    def test_a_field_at_its_default_is_housekeeping(self):
        defaults = StimulusConfig()
        assert _is_schema_default("motion", defaults.motion) is True
        assert _is_schema_default("orientation_deg", defaults.orientation_deg) is True

    def test_a_field_the_user_changed_is_not(self):
        defaults = StimulusConfig()
        changed = defaults.orientation_deg + 30.0
        assert _is_schema_default("orientation_deg", changed) is False
        assert _is_schema_default("spread", defaults.spread + 1.0) is False

    def test_a_field_the_schema_does_not_have_is_housekeeping(self):
        assert _is_schema_default("not_a_schema_field", 123) is True

    def test_a_default_factory_field_is_compared_by_value(self):
        """Otherwise every list-valued field reads as always-changed."""
        defaults = StimulusConfig()
        assert _is_schema_default("start", list(defaults.start)) is True
        assert _is_schema_default("start", [9.0, 9.0]) is False


class TestDroppedParamsWarning:
    """What the user is actually told."""

    def test_nothing_dropped_says_nothing(self):
        assert _dropped_params_warning("gaussian", []) is None

    def test_untouched_fields_say_nothing(self):
        defaults = StimulusConfig()
        dropped = [
            ("motion", defaults.motion),
            ("spread", defaults.spread),
            ("orientation_deg", defaults.orientation_deg),
        ]
        assert _dropped_params_warning("gaussian", dropped) is None

    def test_a_changed_field_is_named_with_its_value(self):
        message = _dropped_params_warning("gaussian", [("spread", 7.5)])
        assert message is not None
        assert "gaussian" in message
        assert "spread=7.5" in message

    def test_only_the_changed_fields_are_named(self):
        defaults = StimulusConfig()
        message = _dropped_params_warning(
            "gaussian",
            [("motion", defaults.motion), ("wavelength", 5.0)],
        )
        assert message is not None
        assert "wavelength=5.0" in message
        assert "motion" not in message

    @pytest.mark.parametrize("value", [0.0, -1.0, "text", [1.0, 2.0]])
    def test_any_changed_value_type_is_reported(self, value):
        """A falsy change is still a change; don't test truthiness."""
        message = _dropped_params_warning("edge", [("spread", value)])
        assert message is not None, f"a spread of {value!r} was dropped silently"
