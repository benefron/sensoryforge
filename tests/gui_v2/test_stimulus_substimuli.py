"""Composite and timeline stimuli are built in the Stimulus screen (Phase 3 prep)."""

import pytest

pytestmark = pytest.mark.gui

import torch  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    SensoryForgeConfig,
    StimulusConfig,
)
from sensoryforge.gui.screens.stimulus import StimulusScreen  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.stimuli.render import render_for_config  # noqa: E402


def _screen(qtbot, stim_type="gaussian"):
    config = SensoryForgeConfig(grids=[GridConfig(name="g", rows=30, cols=30)])
    config.stimulus = StimulusConfig(type=stim_type)
    session = Session(config)
    screen = StimulusScreen(session)
    qtbot.addWidget(screen)
    return session, screen


def _choose(screen, stim_type):
    screen.type_combo.setCurrentIndex(screen.type_combo.findData(stim_type))


def _frames(session, duration_ms=100.0):
    frames, time_ms, _, _ = render_for_config(
        session.config, duration_ms=duration_ms, dt_ms=1.0
    )
    return frames[0], time_ms


def _x_centroid(frame):
    profile = frame.sum(dim=1)
    index = torch.arange(frame.shape[0], dtype=torch.float32)
    return float((profile * index).sum() / profile.sum())


def test_editor_is_hidden_for_ordinary_stimuli(qtbot):
    _, screen = _screen(qtbot)
    assert not screen.sub_stimuli.isVisibleTo(screen)


def test_choosing_composite_starts_with_one_renderable_sub_stimulus(qtbot):
    session, screen = _screen(qtbot)
    _choose(screen, "composite")
    assert screen.sub_stimuli.isVisibleTo(screen)
    assert len(session.config.stimulus.stimuli) == 1
    assert "stimulus" not in session.errors
    assert screen.param_form.empty_notice is None


def test_a_second_sub_stimulus_added_and_moved_appears_in_the_frames(qtbot):
    session, screen = _screen(qtbot)
    _choose(screen, "composite")
    editor = screen.sub_stimuli
    editor.btn_add.click()
    qtbot.wait(10)  # the table rebuild is deferred to the event loop
    assert editor.table.rowCount() == 2

    x_spin = editor.table.cellWidget(1, 3)
    x_spin.setValue(3.0)
    x_spin.editingFinished.emit()
    assert session.config.stimulus.stimuli[1]["params"]["center_x"] == 3.0

    frame = _frames(session)[0][50]
    # Two blobs at x = 0 and x = 3 mm: the centroid sits right of the middle.
    assert _x_centroid(frame) > (frame.shape[0] - 1) / 2 + 1


def test_a_timeline_shows_its_sub_stimuli_in_turn(qtbot):
    session, screen = _screen(qtbot)
    _choose(screen, "timeline")
    editor = screen.sub_stimuli
    editor.btn_add.click()
    qtbot.wait(10)
    entries = session.config.stimulus.params["sub_stimuli"]
    assert [e["onset_ms"] for e in entries] == [0.0, 100.0]

    # Second one: 2 mm to the right, from 50 ms.
    editor.table.cellWidget(1, 3).setValue(2.0)
    editor.table.cellWidget(1, 3).editingFinished.emit()
    editor.table.cellWidget(0, 6).setValue(50.0)
    editor.table.cellWidget(0, 6).editingFinished.emit()
    editor.table.cellWidget(1, 5).setValue(50.0)
    editor.table.cellWidget(1, 5).editingFinished.emit()

    frames, _ = _frames(session)
    assert _x_centroid(frames[70]) > _x_centroid(frames[10]) + 5


def test_kind_change_keeps_the_size_and_leaving_composite_clears_it(qtbot):
    session, screen = _screen(qtbot)
    _choose(screen, "composite")
    editor = screen.sub_stimuli
    editor.table.cellWidget(0, 0).setCurrentText("point")
    qtbot.wait(10)
    sub = session.config.stimulus.stimuli[0]
    assert sub["stim_type"] == "point" and sub["params"]["diameter_mm"] == 1.0
    assert "stimulus" not in session.errors

    _choose(screen, "gaussian")
    assert session.config.stimulus.params == {}
    assert "stimuli" not in session.config.stimulus.explicit_fields()
    assert not editor.isVisibleTo(screen)
    assert "stimulus" not in session.errors


def test_the_composed_stimulus_round_trips_through_yaml(qtbot):
    session, screen = _screen(qtbot)
    _choose(screen, "timeline")
    screen.sub_stimuli.btn_add.click()
    qtbot.wait(10)
    again = SensoryForgeConfig.from_yaml(session.config.to_yaml())
    a, _ = _frames(session)
    b, _ = render_for_config(again, duration_ms=100.0, dt_ms=1.0)[:2]
    assert torch.equal(a, b[0])


@pytest.mark.parametrize("stim_type", ["composite", "timeline"])
def test_the_combine_mode_reaches_the_renderer(qtbot, stim_type):
    session, screen = _screen(qtbot)
    _choose(screen, stim_type)
    editor = screen.sub_stimuli
    editor.btn_add.click()
    qtbot.wait(10)
    if stim_type == "timeline":  # overlap the two so the mode matters
        editor.table.cellWidget(1, 5).setValue(0.0)
        editor.table.cellWidget(1, 5).editingFinished.emit()
    editor.table.cellWidget(1, 3).setValue(0.0)  # same place: add doubles it
    editor.table.cellWidget(1, 3).editingFinished.emit()
    added = _frames(session)[0][20].max()
    editor.mode_combo.setCurrentText("max")
    maxed = _frames(session)[0][20].max()
    assert float(added) == pytest.approx(2 * float(maxed), rel=1e-4)
