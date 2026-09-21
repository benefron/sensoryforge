"""Tests for the Stimulus screen (:mod:`sensoryforge.gui.screens.stimulus`).

Covers the brief's "one rule that matters most": a parameter the user never
set shows the *stimulus type's* own default (not the schema's), is absent
from ``StimulusConfig.explicit_fields()``, and editing/resetting it drives
the live preview -- rendered exclusively through
:func:`sensoryforge.gui.execution.render.render_for_config`. Also covers the
empty-spec notice, changing the target grid, and a forced render error.
"""

from __future__ import annotations

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
from sensoryforge.gui.session import Session  # noqa: E402

register_components.register_all()


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

    # The form shows MovingEdgeStimulus's own default (1.0 mm), not the
    # StimulusConfig schema default for `spread` (also 1.0 here, so use a
    # field whose schema and type defaults genuinely differ: amplitude).
    stim = screen._session.config.stimulus
    amplitude_widget = screen.param_form.widget_for("amplitude")
    assert amplitude_widget.value() == pytest.approx(1.0)  # MovingEdgeStimulus default
    assert StimulusConfig().amplitude == pytest.approx(30.0)  # schema default, unequal
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


def test_type_with_no_paramspecs_shows_notice_not_a_blank_form(qtbot):
    screen = _screen(qtbot, _config("static"))
    assert screen.param_form.empty_notice is not None
    assert "no editable parameters" in screen.param_form.empty_notice.text()


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
