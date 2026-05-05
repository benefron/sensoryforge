"""Tests for the unified workflow: auto-workspace, stimulus/model libraries,
active-pair panel, auto-save results, and the Past Runs browser.

Covers all five parts of the unified-workflow plan:
  Part 1 — Auto-workspace (tested indirectly via set_experiment_manager flow)
  Part 2 — Global toolbar / mech-tab save buttons removed
  Part 3 — Stimulus tab: Save to Library, _library_dir propagation
  Part 4 — Spiking tab: Model Library list, Active Run panel, Use Live Set
  Part 5 — Auto-save results and Past Runs browser in VisualizationTab

Design: module-scoped _APP / fixtures to avoid PyQt5 GC segfaults.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
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


@pytest.fixture(scope="module")
def sp_tab(mech_tab, stim_tab):
    _ensure_app()
    try:
        from sensoryforge.gui.tabs.spiking_tab import SpikingNeuronTab
    except ImportError:
        pytest.skip("GUI tabs not available")
    return SpikingNeuronTab(mechanoreceptor_tab=mech_tab, stimulus_tab=stim_tab)


@pytest.fixture(scope="module")
def viz_tab():
    _ensure_app()
    try:
        from sensoryforge.gui.tabs.visualization_tab import VisualizationTab
    except ImportError:
        pytest.skip("GUI tabs not available")
    return VisualizationTab()


# ---------------------------------------------------------------------------
# Part 2 — Mech tab: config save buttons removed
# ---------------------------------------------------------------------------


def test_mechanoreceptor_tab_no_save_configuration_button(mech_tab):
    """btn_save_configuration must not exist — replaced by global Save Config."""
    assert not hasattr(mech_tab, "btn_save_configuration"), (
        "btn_save_configuration still present; should have been removed in Part 2"
    )


def test_mechanoreceptor_tab_no_save_as_configuration_button(mech_tab):
    """btn_save_as_configuration must not exist."""
    assert not hasattr(mech_tab, "btn_save_as_configuration"), (
        "btn_save_as_configuration still present"
    )


def test_mechanoreceptor_tab_no_load_configuration_button(mech_tab):
    """btn_load_configuration must not exist as a top-level attribute on the widget."""
    assert not hasattr(mech_tab, "btn_load_configuration"), (
        "btn_load_configuration still present"
    )


def test_mechanoreceptor_tab_csv_buttons_still_present(mech_tab):
    """CSV import/export buttons must remain (they are not config saves)."""
    assert hasattr(mech_tab, "btn_import_csv"), "btn_import_csv was removed unexpectedly"
    assert hasattr(mech_tab, "btn_export_csv"), "btn_export_csv was removed unexpectedly"


# ---------------------------------------------------------------------------
# Part 2 — set_experiment_manager emits configuration_directory_changed
# ---------------------------------------------------------------------------


def test_set_experiment_manager_emits_config_dir(mech_tab, tmp_path):
    """set_experiment_manager on mech_tab must emit configuration_directory_changed."""
    from sensoryforge.core.experiment_manager import ExperimentManager

    received = []
    mech_tab.configuration_directory_changed.connect(lambda d: received.append(d))

    em = ExperimentManager()
    em.create(tmp_path / "test_em_signal")
    mech_tab.set_experiment_manager(em)

    assert len(received) >= 1, "configuration_directory_changed was not emitted"
    assert received[-1] == tmp_path / "test_em_signal", (
        f"Expected project_dir, got {received[-1]}"
    )
    # Cleanup
    mech_tab.configuration_directory_changed.disconnect()


# ---------------------------------------------------------------------------
# Part 3 — Stimulus tab: Save to Library
# ---------------------------------------------------------------------------


def test_save_to_library_button_exists(stim_tab):
    """btn_save_to_library must be present on StimulusDesignerTab."""
    assert hasattr(stim_tab, "btn_save_to_library"), "btn_save_to_library missing"


def test_save_to_library_disabled_when_stack_empty(stim_tab):
    """btn_save_to_library must be disabled when _stimulus_stack is empty."""
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()
    stim_tab._update_library_buttons()
    assert not stim_tab.btn_save_to_library.isEnabled(), (
        "Save to Library should be disabled with empty stack"
    )


def test_save_to_library_enabled_after_add(stim_tab):
    """btn_save_to_library must be enabled after adding a stack item."""
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()
    stim_tab._on_stack_add_new()
    assert stim_tab.btn_save_to_library.isEnabled(), (
        "Save to Library should be enabled after adding a stack item"
    )


def test_save_to_library_disabled_after_remove_all(stim_tab):
    """After removing the last stack item, btn_save_to_library must disable."""
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()
    stim_tab._on_stack_add_new()
    assert stim_tab.btn_save_to_library.isEnabled()

    stim_tab.stimulus_stack_list.setCurrentRow(0)
    stim_tab._on_stack_remove()

    assert not stim_tab.btn_save_to_library.isEnabled(), (
        "Save to Library should be disabled after removing all items"
    )


def test_on_config_dir_changed_sets_library_dir(stim_tab, tmp_path):
    """_on_config_dir_changed must set _library_dir to project_dir/stimuli."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    stim_tab._on_config_dir_changed(project_dir)
    assert stim_tab._library_dir == project_dir / "stimuli", (
        f"_library_dir expected {project_dir / 'stimuli'}, got {stim_tab._library_dir}"
    )
    assert stim_tab._library_dir.exists(), "stimuli sub-directory was not created"


def test_on_config_dir_changed_none_clears_library_dir(stim_tab):
    """_on_config_dir_changed(None) must set _library_dir to None."""
    stim_tab._on_config_dir_changed(None)
    assert stim_tab._library_dir is None


def test_save_to_library_writes_json(stim_tab, tmp_path):
    """_on_save_to_library must write a valid JSON file to _library_dir."""
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()
    stim_tab._on_stack_add_new()
    stim_tab._on_stack_add_new()

    library_dir = tmp_path / "stimuli"
    library_dir.mkdir()
    stim_tab._library_dir = library_dir

    with patch(
        "sensoryforge.gui.tabs.stimulus_tab.QtWidgets.QInputDialog.getText",
        return_value=("my_set", True),
    ):
        stim_tab._on_save_to_library()

    target = library_dir / "my_set.json"
    assert target.exists(), "Save to Library did not create the JSON file"
    payload = json.loads(target.read_text())
    assert payload.get("kind") == "stimulus_stack", (
        f"Expected kind=stimulus_stack, got {payload.get('kind')!r}"
    )
    assert len(payload.get("stimuli", [])) == 2, (
        "Saved JSON must contain both stack items"
    )


def test_save_to_library_overwrites_silently(stim_tab, tmp_path):
    """A second save with the same name must overwrite the file without error."""
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()
    stim_tab._on_stack_add_new()

    library_dir = tmp_path / "stimuli_overwrite"
    library_dir.mkdir()
    stim_tab._library_dir = library_dir

    with patch(
        "sensoryforge.gui.tabs.stimulus_tab.QtWidgets.QInputDialog.getText",
        return_value=("overwrite_me", True),
    ):
        stim_tab._on_save_to_library()

    first_mtime = (library_dir / "overwrite_me.json").stat().st_mtime

    stim_tab._on_stack_add_new()  # change stack
    with patch(
        "sensoryforge.gui.tabs.stimulus_tab.QtWidgets.QInputDialog.getText",
        return_value=("overwrite_me", True),
    ):
        stim_tab._on_save_to_library()

    second_mtime = (library_dir / "overwrite_me.json").stat().st_mtime
    payload = json.loads((library_dir / "overwrite_me.json").read_text())
    assert second_mtime >= first_mtime, "File was not updated on second save"
    assert len(payload.get("stimuli", [])) >= 1, "Overwritten file is empty"


def test_save_to_library_updates_current_stimulus_path(stim_tab, tmp_path):
    """After saving, _current_stimulus_path must point to the new file."""
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()
    stim_tab._on_stack_add_new()

    library_dir = tmp_path / "stimuli_path"
    library_dir.mkdir()
    stim_tab._library_dir = library_dir

    with patch(
        "sensoryforge.gui.tabs.stimulus_tab.QtWidgets.QInputDialog.getText",
        return_value=("path_check", True),
    ):
        stim_tab._on_save_to_library()

    assert stim_tab._current_stimulus_path == library_dir / "path_check.json", (
        "_current_stimulus_path not updated after save to library"
    )


def test_save_to_library_emits_request_workspace_when_no_dir(stim_tab):
    """When _library_dir is None, _on_save_to_library must emit request_workspace."""
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()
    stim_tab._on_stack_add_new()
    stim_tab._library_dir = None

    emitted = []
    stim_tab.request_workspace.connect(lambda: emitted.append(True))

    # After emitting, library_dir is still None → dialog shown but we abort via patch
    with patch(
        "sensoryforge.gui.tabs.stimulus_tab.QtWidgets.QMessageBox.information",
    ):
        stim_tab._on_save_to_library()

    assert len(emitted) == 1, "request_workspace signal not emitted when library_dir is None"
    stim_tab.request_workspace.disconnect()


# ---------------------------------------------------------------------------
# Part 4 — Spiking tab: Model Library list widget
# ---------------------------------------------------------------------------


def test_spiking_tab_has_module_list(sp_tab):
    """SpikingNeuronTab must have a module_list QListWidget."""
    from PyQt5 import QtWidgets

    assert hasattr(sp_tab, "module_list"), "module_list missing from SpikingNeuronTab"
    assert isinstance(sp_tab.module_list, QtWidgets.QListWidget)


def test_refresh_module_library_empty_when_no_dir(sp_tab):
    """_refresh_module_library with _module_dir=None must leave list empty."""
    sp_tab._module_dir = None
    sp_tab._refresh_module_library()
    assert sp_tab.module_list.count() == 0


def test_refresh_module_library_shows_json_files(sp_tab, tmp_path):
    """_refresh_module_library must show *.json files with kind=neuron_module."""
    module_dir = tmp_path / "neuron_modules"
    module_dir.mkdir()

    bundle = {
        "schema_version": "1.0.0",
        "kind": "neuron_module",
        "population_configs": [{"name": "Pop1", "neuron_type": "Izhikevich"}],
    }
    (module_dir / "fast_sa.json").write_text(json.dumps(bundle))
    (module_dir / "slow_ra.json").write_text(json.dumps({**bundle, "population_configs": [{"name": "Pop1", "neuron_type": "AdEx"}]}))
    # File with wrong kind — must be ignored
    (module_dir / "stim.json").write_text(json.dumps({"kind": "stimulus_stack"}))

    sp_tab._module_dir = module_dir
    sp_tab._refresh_module_library()

    assert sp_tab.module_list.count() == 2, (
        f"Expected 2 model entries, got {sp_tab.module_list.count()}"
    )
    texts = [sp_tab.module_list.item(i).text() for i in range(sp_tab.module_list.count())]
    assert any("fast_sa" in t for t in texts), "fast_sa not shown in module list"
    assert any("slow_ra" in t for t in texts), "slow_ra not shown in module list"


def test_module_list_selects_current_path(sp_tab, tmp_path):
    """_refresh_module_library must auto-select the item matching _current_module_path."""
    module_dir = tmp_path / "neuron_modules_sel"
    module_dir.mkdir()
    bundle = {"schema_version": "1.0.0", "kind": "neuron_module", "population_configs": []}
    target = module_dir / "selected.json"
    target.write_text(json.dumps(bundle))

    sp_tab._module_dir = module_dir
    sp_tab._current_module_path = target
    sp_tab._refresh_module_library()

    selected = sp_tab.module_list.currentItem()
    assert selected is not None, "No item selected after refresh with matching path"
    from pathlib import Path as P

    assert P(selected.data(sp_tab.module_list.model().UserRole if hasattr(sp_tab.module_list.model(), 'UserRole') else 256)) == target or \
           selected.data(256) == str(target), "Wrong item selected"


# ---------------------------------------------------------------------------
# Part 4 — Active Run panel
# ---------------------------------------------------------------------------


def test_spiking_tab_has_active_run_widgets(sp_tab):
    """SpikingNeuronTab must expose lbl_active_stimulus, lbl_active_model, btn_run_active."""
    assert hasattr(sp_tab, "lbl_active_stimulus"), "lbl_active_stimulus missing"
    assert hasattr(sp_tab, "lbl_active_model"), "lbl_active_model missing"
    assert hasattr(sp_tab, "btn_run_active"), "btn_run_active missing"


def test_active_pair_label_defaults(sp_tab):
    """With no paths set, labels must show 'No stimulus' / 'No model'."""
    sp_tab._current_stimulus_path = None
    sp_tab._current_module_path = None
    sp_tab._update_active_pair_labels()
    assert "No stimulus" in sp_tab.lbl_active_stimulus.text() or \
           "Live set" in sp_tab.lbl_active_stimulus.text(), (
        f"Expected default stimulus label, got {sp_tab.lbl_active_stimulus.text()!r}"
    )
    assert "No model" in sp_tab.lbl_active_model.text(), (
        f"Expected 'No model', got {sp_tab.lbl_active_model.text()!r}"
    )


def test_active_pair_label_with_saved_stimulus(sp_tab, tmp_path):
    """Setting _current_stimulus_path must show its stem in lbl_active_stimulus."""
    path = tmp_path / "my_stimulus.json"
    path.touch()
    sp_tab._current_stimulus_path = path
    sp_tab._current_module_path = None
    sp_tab._update_active_pair_labels()
    assert sp_tab.lbl_active_stimulus.text() == "my_stimulus", (
        f"Expected 'my_stimulus', got {sp_tab.lbl_active_stimulus.text()!r}"
    )
    sp_tab._current_stimulus_path = None


def test_active_pair_label_with_saved_model(sp_tab, tmp_path):
    """Setting _current_module_path must show its stem in lbl_active_model."""
    path = tmp_path / "sa_fast.json"
    path.touch()
    sp_tab._current_module_path = path
    sp_tab._update_active_pair_labels()
    assert sp_tab.lbl_active_model.text() == "sa_fast", (
        f"Expected 'sa_fast', got {sp_tab.lbl_active_model.text()!r}"
    )
    sp_tab._current_module_path = None


def test_active_pair_label_live_set_count(sp_tab, stim_tab):
    """With no saved stimulus path, label shows live set count from stimulus_tab."""
    sp_tab._current_stimulus_path = None
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()
    stim_tab._on_stack_add_new()
    stim_tab._on_stack_add_new()

    sp_tab._update_active_pair_labels()
    label = sp_tab.lbl_active_stimulus.text()
    assert "Live set" in label and "2" in label, (
        f"Expected 'Live set (2 stimuli)', got {label!r}"
    )


# ---------------------------------------------------------------------------
# Part 4 — Use Live Set button
# ---------------------------------------------------------------------------


def test_spiking_tab_has_use_live_set_button(sp_tab):
    """SpikingNeuronTab must have btn_use_live_set."""
    assert hasattr(sp_tab, "btn_use_live_set"), "btn_use_live_set missing"


def test_use_live_set_clears_stimulus_path(sp_tab, tmp_path):
    """_on_use_live_set must set _current_stimulus_path to None."""
    sp_tab._current_stimulus_path = tmp_path / "some.json"
    sp_tab._on_use_live_set()
    assert sp_tab._current_stimulus_path is None, (
        "_current_stimulus_path should be None after Use Live Set"
    )


def test_use_live_set_updates_active_label(sp_tab, stim_tab):
    """After _on_use_live_set, lbl_active_stimulus should reflect live set."""
    stim_tab._stimulus_stack.clear()
    stim_tab._refresh_stack_list()
    stim_tab._on_stack_add_new()

    sp_tab._current_stimulus_path = None  # already cleared
    sp_tab._on_use_live_set()
    label = sp_tab.lbl_active_stimulus.text()
    assert "Live set" in label or "No stimulus" in label, (
        f"Unexpected label after Use Live Set: {label!r}"
    )


# ---------------------------------------------------------------------------
# Part 5 — Auto-save results
# ---------------------------------------------------------------------------


def _make_mock_em(results_dir):
    """Return a minimal fake ExperimentManager that reports is_open=True."""

    class _FakeEM:
        is_open = True

        @property
        def results_dir(self):
            return results_dir

    return _FakeEM()


def _make_fake_results(time_ms):
    """Return a minimal {pop_name: SimulationResult} dict."""
    from sensoryforge.gui.tabs.spiking_tab import SimulationResult

    n = len(time_ms)
    return {
        "Pop1": SimulationResult(
            population_name="Pop1",
            dt_ms=1.0,
            time_ms=time_ms,
            spikes=np.zeros((n, 4), dtype=bool),
            drive=np.zeros((n, 4), dtype=np.float32),
            v_trace=np.zeros((n, 4), dtype=np.float32),
        )
    }


def test_auto_save_creates_pt_file(sp_tab, tmp_path):
    """_auto_save_results must write a .pt file to results_dir."""
    results_dir = tmp_path / "results"
    sp_tab._em = _make_mock_em(results_dir)
    sp_tab._current_stimulus_path = None
    sp_tab._current_module_path = None

    time_ms = np.linspace(0, 100, 101, dtype=np.float32)
    results = _make_fake_results(time_ms)
    sp_tab._auto_save_results(results, None, time_ms, 1.0, (-5.0, 5.0), (-5.0, 5.0))

    pt_files = list(results_dir.glob("*.pt"))
    assert len(pt_files) == 1, f"Expected 1 .pt file, found {len(pt_files)}"

    import torch
    bundle = torch.load(str(pt_files[0]), map_location="cpu", weights_only=False)
    assert bundle.get("stimulus") == "live"
    assert bundle.get("model") == "unsaved"
    assert "Pop1" in bundle.get("results", {})


def test_auto_save_uses_path_stems(sp_tab, tmp_path):
    """run_id must include stimulus and model stems when paths are set."""
    import torch

    results_dir = tmp_path / "results_stems"
    sp_tab._em = _make_mock_em(results_dir)
    sp_tab._current_stimulus_path = tmp_path / "my_stim.json"
    sp_tab._current_module_path = tmp_path / "my_model.json"

    time_ms = np.linspace(0, 50, 51, dtype=np.float32)
    sp_tab._auto_save_results(
        _make_fake_results(time_ms), None, time_ms, 1.0, (-5.0, 5.0), (-5.0, 5.0)
    )
    pt_files = list(results_dir.glob("*.pt"))
    assert pt_files, "No .pt file written"
    bundle = torch.load(str(pt_files[0]), map_location="cpu", weights_only=False)
    assert bundle["stimulus"] == "my_stim"
    assert bundle["model"] == "my_model"
    assert "my_stim" in bundle["run_id"] and "my_model" in bundle["run_id"]

    sp_tab._current_stimulus_path = None
    sp_tab._current_module_path = None


def test_auto_save_no_em_is_noop(sp_tab, tmp_path):
    """Without ExperimentManager, _auto_save_results must not create any files."""
    sp_tab._em = None
    time_ms = np.linspace(0, 10, 11, dtype=np.float32)
    sp_tab._auto_save_results(
        _make_fake_results(time_ms), None, time_ms, 1.0, (-5.0, 5.0), (-5.0, 5.0)
    )
    # No results dir should have been created
    assert not (tmp_path / "results").exists(), "results dir was created unexpectedly"


def test_auto_save_emits_results_saved_signal(sp_tab, tmp_path):
    """results_saved signal must be emitted with the run_id after auto-save."""
    results_dir = tmp_path / "results_signal"
    sp_tab._em = _make_mock_em(results_dir)
    sp_tab._current_stimulus_path = None
    sp_tab._current_module_path = None

    emitted_ids = []
    sp_tab.results_saved.connect(lambda rid: emitted_ids.append(rid))

    time_ms = np.linspace(0, 10, 11, dtype=np.float32)
    sp_tab._auto_save_results(
        _make_fake_results(time_ms), None, time_ms, 1.0, (-5.0, 5.0), (-5.0, 5.0)
    )
    sp_tab.results_saved.disconnect()

    assert len(emitted_ids) == 1, "results_saved not emitted"
    assert "live" in emitted_ids[0] and "unsaved" in emitted_ids[0]


# ---------------------------------------------------------------------------
# Part 5 — Past Runs browser in VisualizationTab
# ---------------------------------------------------------------------------


def test_visualization_tab_has_results_list(viz_tab):
    """VisualizationTab must expose _results_list and refresh_results_list."""
    assert hasattr(viz_tab, "_results_list"), "_results_list widget missing"
    assert hasattr(viz_tab, "refresh_results_list"), "refresh_results_list method missing"
    assert hasattr(viz_tab, "set_experiment_manager"), "set_experiment_manager missing"


def test_set_experiment_manager_sets_results_dir(viz_tab, tmp_path):
    """set_experiment_manager must set _results_dir to em.results_dir."""
    results_dir = tmp_path / "viz_results"
    results_dir.mkdir()
    em = _make_mock_em(results_dir)
    viz_tab.set_experiment_manager(em)
    assert viz_tab._results_dir == results_dir, (
        f"Expected {results_dir}, got {viz_tab._results_dir}"
    )


def test_set_experiment_manager_none_clears_results_dir(viz_tab):
    """set_experiment_manager(None) must set _results_dir to None."""

    class _ClosedEM:
        is_open = False
        results_dir = None

    viz_tab.set_experiment_manager(_ClosedEM())
    assert viz_tab._results_dir is None


def test_refresh_past_runs_shows_pt_files(viz_tab, tmp_path):
    """_refresh_past_runs must populate _results_list with .pt bundles."""
    import torch

    results_dir = tmp_path / "past_runs"
    results_dir.mkdir()

    time_ms = np.linspace(0, 100, 101, dtype=np.float32)
    for name in ("stim_a__model_x__20260101_120000", "stim_b__model_y__20260101_130000"):
        bundle = {
            "run_id": name,
            "stimulus": name.split("__")[0],
            "model": name.split("__")[1],
            "results": {},
            "time_ms": time_ms,
            "dt_ms": 1.0,
            "frames": None,
            "xlim": (-5.0, 5.0),
            "ylim": (-5.0, 5.0),
        }
        torch.save(bundle, results_dir / f"{name}.pt")

    viz_tab._results_dir = results_dir
    viz_tab._refresh_past_runs()

    assert viz_tab._results_list.count() == 2, (
        f"Expected 2 past-run items, got {viz_tab._results_list.count()}"
    )
    texts = [viz_tab._results_list.item(i).text() for i in range(viz_tab._results_list.count())]
    assert any("stim_a" in t and "model_x" in t for t in texts), "stim_a × model_x not in list"
    assert any("stim_b" in t and "model_y" in t for t in texts), "stim_b × model_y not in list"


def test_refresh_past_runs_empty_when_no_dir(viz_tab):
    """With no results_dir set, _refresh_past_runs must produce an empty list."""
    viz_tab._results_dir = None
    viz_tab._refresh_past_runs()
    assert viz_tab._results_list.count() == 0
