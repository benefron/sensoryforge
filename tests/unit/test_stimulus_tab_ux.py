"""GUI tests for StimulusDesignerTab UX fixes (stimulus_tab_updates items 1-8).

Covers:
- Saving the stack no longer clears it (item 3 fix)
- Selection is preserved after "Save to Stack" (item 2)
- Repeat pattern spinboxes trigger the preview timer (item 4)
- Expert mode toggle hides/shows advanced widgets (item 1)
- Full stack round-trips through save/load (item 5)
- Revert restores the last committed config (item 5 UX)
- Type change shows correct parameter widgets (item 6 fix)

Design: module-scoped _APP / fixtures to avoid pyqtgraph GC segfaults.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

_APP = None


def _ensure_app():
    global _APP
    try:
        from PyQt5 import QtWidgets
        _APP = QtWidgets.QApplication.instance()
        if _APP is None:
            _APP = QtWidgets.QApplication(sys.argv[:1])
    except ImportError:
        pytest.skip("PyQt5 not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mech_tab():
    _ensure_app()
    try:
        from sensoryforge.gui.tabs.mechanoreceptor_tab import MechanoreceptorTab
    except ImportError:
        pytest.skip("PyQt5 not available")
    return MechanoreceptorTab()


@pytest.fixture(scope="module")
def stim_tab(mech_tab):
    _ensure_app()
    try:
        from sensoryforge.gui.tabs.stimulus_tab import StimulusDesignerTab
    except ImportError:
        pytest.skip("GUI tabs not available")
    return StimulusDesignerTab(mechanoreceptor_tab=mech_tab)


# ---------------------------------------------------------------------------
# Item 3: Save no longer clears the stack
# ---------------------------------------------------------------------------

def test_stack_save_preserves_stack(stim_tab, tmp_path):
    """Saving the stack must not clear _stimulus_stack."""
    stim_tab._on_stack_add_new()
    stim_tab._on_stack_add_new()
    assert len(stim_tab._stimulus_stack) >= 2, "Need at least 2 items before save"

    target = tmp_path / "test_preserve.json"
    stim_tab._write_stimulus(target)

    assert len(stim_tab._stimulus_stack) >= 2, (
        "_stimulus_stack was cleared after _write_stimulus — bug still present"
    )
    assert target.exists(), "Stimulus file was not written"


def test_saved_file_contains_full_stack(stim_tab, tmp_path):
    """The saved JSON must contain all stack entries, not just the first."""
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()
    stim_tab._on_stack_add_new()
    stim_tab._on_stack_add_new()
    n = len(stim_tab._stimulus_stack)
    assert n >= 2

    target = tmp_path / "full_stack.json"
    stim_tab._write_stimulus(target)

    payload = json.loads(target.read_text())
    assert payload.get("kind") == "stimulus_stack", (
        f"Expected kind='stimulus_stack', got {payload.get('kind')!r}"
    )
    assert len(payload["stimuli"]) == n, (
        f"Saved {len(payload['stimuli'])} stimuli but stack had {n}"
    )


# ---------------------------------------------------------------------------
# Item 2: Selection preserved after "Save to Stack"
# ---------------------------------------------------------------------------

def test_update_preserves_selection(stim_tab):
    """After clicking Save to Stack, the selected row must not change."""
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()
    stim_tab._on_stack_add_new()
    row_before = stim_tab.stimulus_stack_list.currentRow()

    stim_tab._on_stack_update_selected()

    row_after = stim_tab.stimulus_stack_list.currentRow()
    assert row_after == row_before, (
        f"Selection changed from {row_before} to {row_after} after Save to Stack"
    )


def test_apply_button_label(stim_tab):
    """btn_stack_update must be labelled 'Apply'."""
    # Strip asterisk (dirty marker) before checking base label
    text = stim_tab.btn_stack_update.text().replace(" *", "")
    assert text == "Apply", (
        f"Expected 'Apply' button label, got {stim_tab.btn_stack_update.text()!r}"
    )


# ---------------------------------------------------------------------------
# Item 4: Repeat pattern spinboxes trigger preview
# ---------------------------------------------------------------------------

def test_repeat_spinbox_triggers_preview_timer(stim_tab):
    """Changing spin_repeat_nx while repeat is enabled must start the preview timer."""
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()
    stim_tab._on_stack_add_new()  # sets _active_stack_index

    # Enable repeat
    from PyQt5 import QtCore
    stim_tab.chk_repeat_enabled.setChecked(True)
    # Timer may have already fired — stop it and re-trigger via spinbox change
    stim_tab._preview_timer.stop()

    stim_tab.spin_repeat_nx.setValue(stim_tab.spin_repeat_nx.value() + 1)

    assert stim_tab._preview_timer.isActive(), (
        "Preview timer did not start after changing spin_repeat_nx"
    )

    # Cleanup
    stim_tab.chk_repeat_enabled.setChecked(False)


# ---------------------------------------------------------------------------
# Item 1: Expert mode toggle
# ---------------------------------------------------------------------------

def test_stimulus_tab_has_expert_mode_checkbox(stim_tab):
    """StimulusDesignerTab must expose chk_expert_mode."""
    assert hasattr(stim_tab, "chk_expert_mode"), "StimulusDesignerTab missing chk_expert_mode"


def test_expert_only_widgets_non_empty(stim_tab):
    """_expert_only_widgets_stimulus must contain at least one widget."""
    assert len(stim_tab._expert_only_widgets_stimulus) > 0, (
        "_expert_only_widgets_stimulus is empty — expert mode has no effect"
    )


def test_basic_mode_hides_expert_widgets(stim_tab):
    """In Basic mode (unchecked), all _expert_only_widgets_stimulus must be hidden."""
    stim_tab.chk_expert_mode.setChecked(False)
    for w in stim_tab._expert_only_widgets_stimulus:
        assert w.isHidden(), f"Widget {w!r} should be hidden in Basic mode but is visible"


def test_expert_mode_shows_advanced_widgets(stim_tab):
    """In Expert mode (checked), all _expert_only_widgets_stimulus must not be hidden."""
    stim_tab.chk_expert_mode.setChecked(True)
    for w in stim_tab._expert_only_widgets_stimulus:
        assert not w.isHidden(), f"Widget {w!r} should be visible in Expert mode but is hidden"

    # Restore to basic for subsequent tests
    stim_tab.chk_expert_mode.setChecked(False)


# ---------------------------------------------------------------------------
# Item 5: Load stack round-trip
# ---------------------------------------------------------------------------

def test_load_stack_round_trip(stim_tab, tmp_path):
    """Save two stack items, load them back, verify count and type fields."""
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()

    stim_tab._on_stack_add_new()
    # Set type for first item and update
    stim_tab.type_buttons["gaussian"].setChecked(True)
    stim_tab._selected_type = "gaussian"
    stim_tab._on_stack_update_selected()

    stim_tab._on_stack_add_new()
    stim_tab.type_buttons["noise"].setChecked(True)
    stim_tab._selected_type = "noise"
    stim_tab._on_stack_update_selected()

    assert len(stim_tab._stimulus_stack) == 2

    target = tmp_path / "round_trip.json"
    stim_tab._write_stimulus(target)

    # Reset and reload
    stim_tab._reset_stack()
    assert len(stim_tab._stimulus_stack) == 0, "Stack not cleared before reload"

    stim_tab._load_stimulus_from_file(target)

    assert len(stim_tab._stimulus_stack) == 2, (
        f"Expected 2 stack items after load, got {len(stim_tab._stimulus_stack)}"
    )
    assert stim_tab._stimulus_stack[0].stimulus_type == "gaussian"
    assert stim_tab._stimulus_stack[1].stimulus_type == "noise"


# ---------------------------------------------------------------------------
# Item 5 UX: Revert restores committed state
# ---------------------------------------------------------------------------

def test_revert_restores_committed(stim_tab):
    """After editing params, Revert must restore the last committed amplitude."""
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()
    stim_tab._on_stack_add_new()

    stim_tab.dbl_amplitude.setValue(2.5)
    stim_tab._on_stack_update_selected()
    committed_amp = stim_tab._committed_config.amplitude if stim_tab._committed_config else 2.5

    # Edit amplitude without saving
    stim_tab.dbl_amplitude.setValue(9.9)
    assert abs(stim_tab.dbl_amplitude.value() - 9.9) < 0.01

    stim_tab._on_stack_revert()

    reverted_amp = stim_tab.dbl_amplitude.value()
    assert abs(reverted_amp - committed_amp) < 0.01, (
        f"Revert did not restore amplitude: expected {committed_amp}, got {reverted_amp}"
    )


def test_dirty_indicator_appears_on_edit(stim_tab):
    """After editing, btn_stack_update text must contain an asterisk."""
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()
    stim_tab._on_stack_add_new()

    # Trigger dirty state manually via _handle_preview_request
    stim_tab._handle_preview_request()

    assert "*" in stim_tab.btn_stack_update.text(), (
        f"Expected '*' in button text after edit, got {stim_tab.btn_stack_update.text()!r}"
    )


def test_dirty_indicator_clears_on_save(stim_tab):
    """After Apply, the asterisk must be gone."""
    stim_tab._on_stack_update_selected()
    assert "*" not in stim_tab.btn_stack_update.text(), (
        f"Asterisk should be gone after Apply, got {stim_tab.btn_stack_update.text()!r}"
    )


# ---------------------------------------------------------------------------
# Item 6: Per-type parameter visibility
# ---------------------------------------------------------------------------

def test_gaussian_type_shows_spread_hides_texture_group(stim_tab):
    """Gaussian type: spread not explicitly hidden, texture_group hidden."""
    stim_tab.type_buttons["gaussian"].setChecked(True)
    stim_tab._on_type_selected(stim_tab.type_buttons["gaussian"])

    # Use isHidden() — isVisible() requires all ancestors to be shown (tab not in window)
    assert stim_tab.texture_group.isHidden(), (
        "texture_group should be hidden for gaussian type"
    )
    assert not stim_tab.dbl_spread.isHidden(), "spread should not be hidden for gaussian type"


def test_gabor_type_shows_texture_params(stim_tab):
    """Gabor type must show texture_group with gabor_params_widget not hidden."""
    stim_tab.type_buttons["gabor"].setChecked(True)
    stim_tab._on_type_selected(stim_tab.type_buttons["gabor"])

    assert not stim_tab.texture_group.isHidden(), "texture_group must not be hidden for gabor type"
    assert not stim_tab.gabor_params_widget.isHidden(), (
        "gabor_params_widget must not be hidden for gabor type"
    )
    assert stim_tab.edge_grating_params_widget.isHidden(), (
        "edge_grating_params_widget must be hidden for gabor type"
    )
    assert stim_tab.noise_params_widget.isHidden(), (
        "noise_params_widget must be hidden for gabor type"
    )
    # Subtype combo must be hidden for direct gabor type
    assert stim_tab.texture_subtype_combo.isHidden(), (
        "texture_subtype_combo must be hidden for gabor (not legacy texture) type"
    )


def test_grating_type_shows_grating_params(stim_tab):
    """Grating type must show texture_group with edge_grating_params_widget not hidden."""
    stim_tab.type_buttons["grating"].setChecked(True)
    stim_tab._on_type_selected(stim_tab.type_buttons["grating"])

    assert not stim_tab.texture_group.isHidden(), "texture_group must not be hidden for grating type"
    assert not stim_tab.edge_grating_params_widget.isHidden(), (
        "edge_grating_params_widget must not be hidden for grating type"
    )
    assert stim_tab.gabor_params_widget.isHidden(), (
        "gabor_params_widget must be hidden for grating type"
    )


def test_noise_type_shows_noise_params(stim_tab):
    """Noise type must show texture_group with noise_params_widget not hidden."""
    stim_tab.type_buttons["noise"].setChecked(True)
    stim_tab._on_type_selected(stim_tab.type_buttons["noise"])

    assert not stim_tab.texture_group.isHidden(), "texture_group must not be hidden for noise type"
    assert not stim_tab.noise_params_widget.isHidden(), (
        "noise_params_widget must not be hidden for noise type"
    )
    assert stim_tab.gabor_params_widget.isHidden(), (
        "gabor_params_widget must be hidden for noise type"
    )
    assert stim_tab.edge_grating_params_widget.isHidden(), (
        "edge_grating_params_widget must be hidden for noise type"
    )
