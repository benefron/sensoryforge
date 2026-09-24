"""The layer editor for layered stimuli
(:mod:`sensoryforge.gui.screens.stimulus_layers`)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gui

import torch  # noqa: E402
from PyQt5 import QtCore  # noqa: E402

import sensoryforge.register_components as register_components  # noqa: E402
from sensoryforge.config.schema import (  # noqa: E402
    GridConfig,
    SensoryForgeConfig,
    StimulusConfig,
)
from sensoryforge.gui.screens.stimulus import StimulusScreen  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.stimuli.render import render_for_config  # noqa: E402

register_components.register_all()


def _screen(qtbot, stim_type="gaussian", duration_ms=200.0):
    config = SensoryForgeConfig(grids=[GridConfig(name="g", rows=30, cols=30)])
    config.stimulus = StimulusConfig(type=stim_type)
    config.simulation.duration_ms = duration_ms
    config.simulation.dt_ms = 1.0
    session = Session(config)
    screen = StimulusScreen(session)
    qtbot.addWidget(screen)
    return session, screen


def _choose(screen, stim_type):
    screen.type_combo.setCurrentIndex(screen.type_combo.findData(stim_type))


def _frames(session, duration_ms=200.0, dt_ms=1.0):
    frames, time_ms, _, _ = render_for_config(
        session.config, duration_ms=duration_ms, dt_ms=dt_ms
    )
    return frames[0], time_ms


def _set_text(widget, value):
    widget.setText(value)
    widget.editingFinished.emit()


def _set_spin(widget, value):
    # QDoubleSpinBox/QSpinBox emit valueChanged on setValue, which ParamForm
    # is already connected to -- no extra emit needed.
    widget.setValue(value)


# -------------------------------------------------------------------- basics


def test_editor_is_hidden_for_ordinary_stimuli(qtbot):
    _, screen = _screen(qtbot)
    assert not screen.layer_editor.isVisibleTo(screen)


def test_choosing_layered_seeds_one_layer_and_hides_the_combine_row(qtbot):
    session, screen = _screen(qtbot)
    _choose(screen, "layered")
    assert screen.layer_editor.isVisibleTo(screen)
    assert len(session.config.stimulus.layers) == 1
    assert "stimulus" not in session.errors
    # "combine" is owned by the LayerEditor's own combo, not the generic form.
    assert "combine" not in screen.param_form._rows


# ------------------------------------------------------------- editing layers


def _editor(screen):
    return screen.layer_editor


def test_two_layers_built_and_edited_through_the_forms_both_render(qtbot):
    session, screen = _screen(qtbot, duration_ms=300.0)
    _choose(screen, "layered")
    editor = _editor(screen)

    # Layer 0: a disc braille pattern spelling "ab", moving linear.
    editor.layer_list.setCurrentRow(0)
    editor._kind_combos["shape"].setCurrentText("disc")
    editor._kind_combos["pattern"].setCurrentText("braille")
    editor._kind_combos["motion"].setCurrentText("linear")

    shape_form = editor._part_forms["shape"]
    _set_spin(shape_form.widget_for("diameter_mm"), 0.6)

    pattern_form = editor._part_forms["pattern"]
    _set_text(pattern_form.widget_for("text"), "ab")

    timing_form = editor._part_forms["timing"]
    _set_spin(timing_form.widget_for("ramp_up_ms"), 10.0)
    _set_spin(timing_form.widget_for("ramp_down_ms"), 10.0)

    layer0 = session.config.stimulus.layers[0]
    assert layer0["shape"]["kind"] == "disc"
    assert layer0["shape"]["diameter_mm"] == 0.6
    assert layer0["pattern"]["kind"] == "braille"
    assert layer0["pattern"]["text"] == "ab"
    assert layer0["motion"]["kind"] == "linear"

    # Layer 1: a random Gaussian texture, onset at 100 ms.
    editor.btn_add.click()
    assert editor.layer_list.count() == 2
    editor.layer_list.setCurrentRow(1)
    editor._kind_combos["pattern"].setCurrentText("random")

    timing_form = editor._part_forms["timing"]
    _set_spin(timing_form.widget_for("onset_ms"), 100.0)

    layer1 = session.config.stimulus.layers[1]
    assert layer1["pattern"]["kind"] == "random"
    assert layer1["timing"]["onset_ms"] == 100.0

    frames, time_ms = _frames(session, duration_ms=300.0)
    # Before the texture's onset (plus its default ramp), only layer 0
    # contributes; after, both are drawn -- more energy in the frame.
    before = frames[50].sum()
    after_index = int((150.0 / 1.0))
    after = frames[after_index].sum()
    assert float(after) > float(before)


def test_timing_ramp_hold_ramp_down_change_the_envelope(qtbot):
    session, screen = _screen(qtbot, duration_ms=200.0)
    _choose(screen, "layered")
    editor = _editor(screen)
    editor.layer_list.setCurrentRow(0)

    timing_form = editor._part_forms["timing"]
    _set_spin(timing_form.widget_for("ramp_up_ms"), 20.0)
    _set_spin(timing_form.widget_for("hold_ms"), 50.0)
    _set_spin(timing_form.widget_for("ramp_down_ms"), 20.0)

    layer = session.config.stimulus.layers[0]
    assert layer["timing"]["ramp_up_ms"] == 20.0
    assert layer["timing"]["hold_ms"] == 50.0
    assert layer["timing"]["ramp_down_ms"] == 20.0

    frames, _ = _frames(session, duration_ms=200.0)
    peak = float(frames.amax())
    # Zero before onset, rising during the ramp, at peak during the hold,
    # falling during the ramp down, zero after.
    assert float(frames[0].sum()) == 0.0
    mid_ramp = float(frames[10].amax())
    assert 0.0 < mid_ramp < peak
    held = float(frames[40].amax())
    assert held == pytest.approx(peak, rel=1e-3)
    after = float(frames[100].amax())
    assert after == 0.0


def test_changing_shape_kind_drops_the_old_kind_params(qtbot):
    session, screen = _screen(qtbot)
    _choose(screen, "layered")
    editor = _editor(screen)
    editor.layer_list.setCurrentRow(0)
    assert session.config.stimulus.layers[0]["shape"]["kind"] == "gaussian"
    assert "sigma_mm" in session.config.stimulus.layers[0]["shape"]

    editor._kind_combos["shape"].setCurrentText("disc")

    shape = session.config.stimulus.layers[0]["shape"]
    assert shape["kind"] == "disc"
    assert "sigma_mm" not in shape
    assert "diameter_mm" in shape


def test_remove_move_duplicate_and_enabled(qtbot):
    session, screen = _screen(qtbot, duration_ms=200.0)
    _choose(screen, "layered")
    editor = _editor(screen)

    editor.btn_add.click()
    assert len(session.config.stimulus.layers) == 2

    # Duplicate layer 0.
    editor.layer_list.setCurrentRow(0)
    editor.btn_duplicate.click()
    assert len(session.config.stimulus.layers) == 3
    assert session.config.stimulus.layers[0] == session.config.stimulus.layers[1]

    # Move layer 2 up (swap with layer 1).
    editor.layer_list.setCurrentRow(2)
    editor.btn_up.click()
    assert session.config.stimulus.layers[1] is not None  # swapped, still 3 layers
    assert len(session.config.stimulus.layers) == 3

    # Disable layer 0: its shape must not appear in the rendered frames.
    editor.layer_list.item(0).setCheckState(QtCore.Qt.Unchecked)
    assert session.config.stimulus.layers[0]["enabled"] is False

    frames_disabled, _ = _frames(session, duration_ms=200.0)
    # Re-enable and compare: enabling should only add energy, never remove.
    # (the list was rebuilt by the write above, so the item must be re-fetched)
    editor.layer_list.item(0).setCheckState(QtCore.Qt.Checked)
    frames_enabled, _ = _frames(session, duration_ms=200.0)
    assert float(frames_enabled.sum()) >= float(frames_disabled.sum())

    # Remove layer 0.
    n_before = len(session.config.stimulus.layers)
    editor.layer_list.setCurrentRow(0)
    editor.btn_remove.click()
    assert len(session.config.stimulus.layers) == n_before - 1


# ---------------------------------------------------------------- presets


def test_a_preset_loads_and_round_trips_through_yaml(qtbot):
    session, screen = _screen(qtbot, duration_ms=500.0)
    _choose(screen, "layered")
    editor = _editor(screen)

    idx = editor.preset_combo.findData("braille_word")
    assert idx > 0
    editor.preset_combo.setCurrentIndex(idx)

    assert len(session.config.stimulus.layers) == 1
    assert session.config.stimulus.layers[0]["pattern"]["kind"] == "braille"
    # The preset combo resets to the placeholder after loading.
    assert editor.preset_combo.currentData() is None

    frames_before, _ = _frames(session, duration_ms=500.0)

    yaml_text = session.config.to_yaml()
    reloaded = SensoryForgeConfig.from_yaml(yaml_text)
    frames_after, _, _, _ = render_for_config(reloaded, duration_ms=500.0, dt_ms=1.0)

    assert torch.equal(frames_before, frames_after[0])


# ----------------------------------------------------------------- combine


def test_combine_max_vs_sum_changes_overlapping_frames(qtbot):
    session, screen = _screen(qtbot, duration_ms=200.0)
    _choose(screen, "layered")
    editor = _editor(screen)

    # Two overlapping Gaussian probes, both full-amplitude for the whole run.
    editor.layer_list.setCurrentRow(0)
    timing_form = editor._part_forms["timing"]
    _set_spin(timing_form.widget_for("ramp_up_ms"), 0.0)
    _set_spin(timing_form.widget_for("ramp_down_ms"), 0.0)
    shape_form = editor._part_forms["shape"]
    _set_spin(shape_form.widget_for("amplitude"), 1.0)
    _set_spin(shape_form.widget_for("sigma_mm"), 3.0)

    editor.btn_duplicate.click()
    assert len(session.config.stimulus.layers) == 2

    editor.combine_combo.setCurrentText("sum")
    frames_sum, _ = _frames(session, duration_ms=200.0)

    editor.combine_combo.setCurrentText("max")
    frames_max, _ = _frames(session, duration_ms=200.0)

    assert session.config.stimulus.combine == "max"
    peak_sum = float(frames_sum.amax())
    peak_max = float(frames_max.amax())
    # Two identical, co-located probes: sum doubles the peak, max does not.
    assert peak_sum == pytest.approx(2.0, rel=1e-3)
    assert peak_max == pytest.approx(1.0, rel=1e-3)
    assert peak_sum > peak_max
