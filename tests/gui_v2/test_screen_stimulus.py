"""Tests for the Stimulus screen (:mod:`sensoryforge.gui.screens.stimulus`).

Covers the brief's "one rule that matters most": a parameter the user never
set shows the *stimulus type's* own default (not the schema's), is absent
from ``StimulusConfig.explicit_fields()``, and editing/resetting it drives
the live preview -- rendered exclusively through
:func:`sensoryforge.gui.execution.render.render_for_config`. Also covers the
empty-spec notice, changing the target grid, and a forced render error.
"""

from __future__ import annotations

import torch
import numpy as np
import pytest

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

from PyQt5 import QtCore  # noqa: E402

import sensoryforge.register_components as register_components  # noqa: E402
from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    SensoryForgeConfig,
    StimulusConfig,
)
from sensoryforge.gui.screens import (
    stimulus_preview as stimulus_preview_module,
)  # noqa: E402
from sensoryforge.gui.screens.stimulus import StimulusScreen  # noqa: E402
from sensoryforge.gui.screens.stimulus_paramform import (  # noqa: E402
    unconfigurable_params,
)
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.registry import STIMULUS_REGISTRY  # noqa: E402
from sensoryforge.stimuli.render import render_for_config  # noqa: E402

register_components.register_all()

#: Types :func:`sensoryforge.stimuli.render.render_for_config` renders from a
#: single ``StimulusConfig`` block on its own; ``composite``/``static``/
#: ``timeline`` need child stimuli this form does not build.
_SKIP_WYSIWYG_TYPES = frozenset({"composite", "static", "timeline"})


def _config(stimulus_type: str, **grid_kwargs) -> SensoryForgeConfig:
    grid = GridConfig(
        name="G",
        arrangement="grid",
        rows=grid_kwargs.pop("rows", 20),
        cols=grid_kwargs.pop("cols", 20),
        spacing=grid_kwargs.pop("spacing", 0.15),
    )
    config = SensoryForgeConfig(
        grids=[grid], populations=[], stimulus=StimulusConfig(type=stimulus_type)
    )
    config.simulation.duration_ms = 330.0
    config.simulation.dt_ms = 1.0
    return config


def _wait_render(qtbot, preview, *, timeout=8000):
    with qtbot.waitSignal(
        preview.renderFinished, timeout=timeout, raising=True
    ) as blocker:
        pass
    return blocker.args[0]


def _screen(qtbot, config) -> StimulusScreen:
    session = Session(config)
    screen = StimulusScreen(session)
    qtbot.addWidget(screen)
    return screen


def _center_of_mass(frame: np.ndarray):
    total = frame.sum()
    if total == 0:
        return (0.0, 0.0)
    ys, xs = np.mgrid[0 : frame.shape[0], 0 : frame.shape[1]]
    return (float((xs * frame).sum() / total), float((ys * frame).sum() / total))


# ------------------------------------------------------------------ moving_edge


def test_moving_edge_default_moves_and_uses_the_type_default(qtbot):
    screen = _screen(qtbot, _config("moving_edge"))
    _wait_render(qtbot, screen.preview)

    frames = screen.preview._frames  # the array actually fed to image_item
    early, late = _center_of_mass(frames[10]), _center_of_mass(frames[280])
    assert early != late, "a default moving_edge should sweep, not sit still"

    # The form shows MovingEdgeStimulus's own default, not the StimulusConfig
    # schema default. `spread` and `amplitude` are 1.0 in both (amplitude
    # since D-0437899), so use a field whose schema and type defaults
    # genuinely differ: orientation_deg (50 deg vs. the schema's 0).
    stim = screen._session.config.stimulus
    orientation_widget = screen.param_form.widget_for("orientation_deg")
    assert orientation_widget.value() == pytest.approx(50.0)  # MovingEdgeStimulus
    assert StimulusConfig().orientation_deg == pytest.approx(0.0)  # schema, unequal
    assert "orientation_deg" not in stim.explicit_fields()
    amplitude_widget = screen.param_form.widget_for("amplitude")
    assert amplitude_widget.value() == pytest.approx(1.0)  # MovingEdgeStimulus default
    assert "amplitude" not in stim.explicit_fields()


def test_editing_a_parameter_marks_it_explicit_and_changes_frames(qtbot):
    screen = _screen(qtbot, _config("moving_edge"))
    _wait_render(qtbot, screen.preview)
    frames_before = screen.preview._frames.copy()

    spread_widget = screen.param_form.widget_for("spread")
    assert screen.param_form.is_explicit("spread") is False
    spread_widget.setValue(spread_widget.value() + 3.0)
    assert screen.param_form.is_explicit("spread") is True
    assert "spread" in screen._session.config.stimulus.explicit_fields()

    _wait_render(qtbot, screen.preview)
    frames_after = screen.preview._frames
    assert not np.allclose(frames_before, frames_after)


def test_reset_to_default_unsets_the_field_and_restores_frames_exactly(qtbot):
    screen = _screen(qtbot, _config("moving_edge"))
    _wait_render(qtbot, screen.preview)
    frames_original = screen.preview._frames.copy()

    spread_widget = screen.param_form.widget_for("spread")
    spread_widget.setValue(spread_widget.value() + 3.0)
    _wait_render(qtbot, screen.preview)
    assert not np.allclose(frames_original, screen.preview._frames)

    screen.param_form._rows["spread"].reset_button.click()
    assert screen.param_form.is_explicit("spread") is False
    assert "spread" not in screen._session.config.stimulus.explicit_fields()

    _wait_render(qtbot, screen.preview)
    assert np.array_equal(frames_original, screen.preview._frames)


# --------------------------------------------------------------------- grids


def test_changing_target_grid_changes_the_preview_rect_in_mm(qtbot):
    grid_a = GridConfig(name="A", arrangement="grid", rows=20, cols=20, spacing=0.15)
    grid_b = GridConfig(name="B", arrangement="grid", rows=10, cols=10, spacing=0.5)
    config = SensoryForgeConfig(
        grids=[grid_a, grid_b],
        populations=[],
        stimulus=StimulusConfig(type="gaussian", target_layer="A"),
    )
    config.simulation.duration_ms = 5.0
    config.simulation.dt_ms = 1.0
    screen = _screen(qtbot, config)
    _wait_render(qtbot, screen.preview)
    rect_a = screen.preview._image_plot.viewRect()

    idx = screen.grid_combo.findData("B")
    assert idx >= 0
    screen.grid_combo.setCurrentIndex(idx)
    assert screen._session.config.stimulus.target_layer == "B"
    _wait_render(qtbot, screen.preview)
    rect_b = screen.preview._image_plot.viewRect()

    assert rect_a != rect_b
    assert (rect_a.width(), rect_a.height()) != (rect_b.width(), rect_b.height())


# -------------------------------------------------------------------- empty


def test_a_still_stimulus_with_no_params_of_its_own_shows_its_timing(qtbot):
    # "static" declares no parameters, but it is ramped in and out, so the
    # form shows the envelope rows (auto until set) instead of a notice.
    screen = _screen(qtbot, _config("static"))
    assert screen.param_form.empty_notice is None
    assert {"ramp_up_ms", "plateau_ms", "ramp_down_ms"} <= set(screen.param_form._rows)


def test_legacy_type_shows_notice_too(qtbot):
    screen = _screen(qtbot, _config("gaussian"))
    idx = screen.type_combo.findData("trapezoidal")
    assert idx >= 0, "legacy names must be selectable from the type combo"
    screen.type_combo.setCurrentIndex(idx)
    assert screen._session.config.stimulus.type == "trapezoidal"
    assert screen.param_form.empty_notice is not None


# --------------------------------------------------------------------- error


def test_render_error_shows_message_and_leaves_screen_usable(qtbot, monkeypatch):
    def boom(*_args, **_kwargs):
        raise ValueError("forced failure for the test")

    monkeypatch.setattr(stimulus_preview_module, "render_for_config", boom)

    screen = _screen(qtbot, _config("gaussian"))
    screen.show()
    qtbot.waitExposed(screen)
    with qtbot.waitSignal(screen.preview.renderFailed, timeout=8000) as blocker:
        pass
    assert "forced failure for the test" in blocker.args[0]
    assert screen.preview._error_label.isVisible() is True
    assert "forced failure for the test" in screen.preview._error_label.text()
    # The screen (and its widgets) stay interactive: nothing raised out of
    # the worker's failed slot, and the type combo can still be used.
    assert screen.isEnabled()
    assert screen.type_combo.isEnabled()
    idx = screen.type_combo.findData("moving_edge")
    screen.type_combo.setCurrentIndex(idx)
    assert screen._session.config.stimulus.type == "moving_edge"


# ------------------------------------------------------------- type registry


def test_type_combo_lists_registered_and_legacy_names_separately(qtbot):
    from sensoryforge.gui.screens.stimulus_paramform import LEGACY_STIMULUS_TYPES
    from sensoryforge.registry import STIMULUS_REGISTRY

    screen = _screen(qtbot, _config("gaussian"))
    all_data = [screen.type_combo.itemData(i) for i in range(screen.type_combo.count())]
    for name in STIMULUS_REGISTRY.list_registered():
        assert name in all_data
    for name in LEGACY_STIMULUS_TYPES:
        assert name in all_data


# ------------------------------------------------------------- config replace


def test_config_replaced_rebuilds_the_screen(qtbot):
    screen = _screen(qtbot, _config("moving_edge"))
    _wait_render(qtbot, screen.preview)

    new_config = _config("gaussian")
    screen._session.replace_config(new_config)
    assert screen.type_combo.currentData() == "gaussian"
    assert screen.param_form.widget_for("amplitude") is not None


# ------------------------------------------------------------- unconfigurable


def test_unconfigurable_params_is_empty():
    """Every declared parameter of every registered stimulus is editable now.

    ``stimulus.params`` gives a param with no ``StimulusConfig`` field
    somewhere to live, so nothing is unconfigurable any more.
    """
    assert unconfigurable_params() == []


# ------------------------------------------------------- shown == used (WYSIWYG)


def _read_row_value(spec_name, form):
    from sensoryforge.gui.widgets.param_form import _read_widget

    row = form._rows[spec_name]
    return _read_widget(row.spec, row.widget)


def _write_every_displayed_value(config, form):
    """Explicitly set every row's *currently displayed* value on *config*.

    Mirrors exactly what a user clicking through every row and leaving it
    unchanged (so it becomes explicit but numerically identical) would
    produce: a named field is set on ``stimulus.<name>``, everything else on
    ``stimulus.params.<name>``.
    """
    for name, row in form._rows.items():
        value = _read_row_value(name, form)
        if row.config_field:
            setattr(config.stimulus, name, value)
        else:
            config.stimulus.params[name] = value


@pytest.mark.parametrize(
    "stimulus_type",
    sorted(set(STIMULUS_REGISTRY.list_registered()) - _SKIP_WYSIWYG_TYPES),
)
def test_what_you_see_is_what_runs(qtbot, stimulus_type):
    """A fresh render must equal a render with every displayed value written explicitly.

    This is the screen's core promise (see the task brief): whatever value a
    row shows for an unset parameter is the value that actually renders.
    Before the ``effective_defaults``-aware fix, several rows displayed
    ``spec.default`` (the stimulus class's own constructor default) while the
    render used a different value (the legacy-generator default, e.g.
    gaussian amplitude 1.0 shown vs. 30.0 rendered) -- so writing the shown
    value explicitly changed what rendered, failing this test.
    """
    screen = _screen(qtbot, _config(stimulus_type))
    bare = screen._session.config

    frames_bare, _, _, _ = render_for_config(bare, duration_ms=50.0, dt_ms=1.0)

    explicit_config = _config(stimulus_type)
    _write_every_displayed_value(explicit_config, screen.param_form)
    frames_explicit, _, _, _ = render_for_config(
        explicit_config, duration_ms=50.0, dt_ms=1.0
    )

    assert torch.equal(frames_bare, frames_explicit), (
        f"{stimulus_type!r}: rendering with every displayed value written "
        "explicitly must be bit-identical to the bare render (shown != used)"
    )


def test_gaussian_amplitude_widget_reads_1_on_a_fresh_session(qtbot):
    # D-0437899: a unit peak (was 30 before every stimulus defaulted to 1.0).
    screen = _screen(qtbot, _config("gaussian"))
    amplitude_widget = screen.param_form.widget_for("amplitude")
    assert amplitude_widget.value() == pytest.approx(1.0)


# ------------------------------------------------------- params-backed rows


def test_editing_a_params_backed_row_through_the_widget_changes_frames(qtbot):
    screen = _screen(qtbot, _config("braille"))
    _wait_render(qtbot, screen.preview)
    frames_before = screen.preview._frames.copy()

    assert "v_mms" not in screen._session.config.stimulus.params
    v_mms_widget = screen.param_form.widget_for("v_mms")
    v_mms_widget.setValue(v_mms_widget.value() + 30.0)
    assert screen._session.config.stimulus.params.get("v_mms") == pytest.approx(
        v_mms_widget.value()
    )
    assert screen.param_form.is_explicit("v_mms") is True

    _wait_render(qtbot, screen.preview)
    frames_after = screen.preview._frames
    assert not np.allclose(frames_before, frames_after)


def test_params_backed_row_survives_yaml_round_trip(qtbot):
    screen = _screen(qtbot, _config("edge_grating"))
    count_widget = screen.param_form.widget_for("count")
    count_widget.setValue(count_widget.value() + 2)
    stim = screen._session.config.stimulus
    assert stim.params.get("count") == count_widget.value()

    yaml_text = screen._session.config.to_yaml()
    reloaded = SensoryForgeConfig.from_yaml(yaml_text)
    assert reloaded.stimulus.params.get("count") == count_widget.value()
    assert "count" in reloaded.stimulus.explicit_fields() or (
        "params" in reloaded.stimulus.explicit_fields()
    )


def test_reset_button_on_a_params_backed_row_restores_frames_and_removes_the_key(qtbot):
    screen = _screen(qtbot, _config("braille"))
    _wait_render(qtbot, screen.preview)
    frames_original = screen.preview._frames.copy()

    v_mms_widget = screen.param_form.widget_for("v_mms")
    v_mms_widget.setValue(v_mms_widget.value() + 30.0)
    _wait_render(qtbot, screen.preview)
    assert not np.allclose(frames_original, screen.preview._frames)

    screen.param_form._rows["v_mms"].reset_button.click()
    assert "v_mms" not in screen._session.config.stimulus.params
    assert screen.param_form.is_explicit("v_mms") is False

    _wait_render(qtbot, screen.preview)
    assert np.array_equal(frames_original, screen.preview._frames)


def test_changing_type_clears_params(qtbot):
    screen = _screen(qtbot, _config("braille"))
    v_mms_widget = screen.param_form.widget_for("v_mms")
    v_mms_widget.setValue(v_mms_widget.value() + 30.0)
    assert screen._session.config.stimulus.params

    idx = screen.type_combo.findData("gaussian")
    assert idx >= 0
    screen.type_combo.setCurrentIndex(idx)
    assert screen._session.config.stimulus.type == "gaussian"
    assert screen._session.config.stimulus.params == {}


def test_back_to_back_renders_do_not_destroy_a_running_thread(qtbot):
    """Replacing the config while a render runs used to abort the process."""
    from sensoryforge.gui.screens.stimulus import StimulusScreen

    session = Session(_config("gaussian"))
    screen = StimulusScreen(session)
    qtbot.addWidget(screen)
    preview = screen.preview
    for _ in range(5):
        preview.render_now()
    assert len(preview._inflight) >= 1
    with qtbot.waitSignal(preview.renderFinished, timeout=20000):
        preview.render_now()
    qtbot.waitUntil(lambda: not preview._inflight, timeout=20000)
