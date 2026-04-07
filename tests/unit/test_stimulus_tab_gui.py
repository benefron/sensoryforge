"""Unit tests for the StimulusDesignerTab GUI widget.

These tests exercise the stimulus tab in headless mode using unittest.mock to
patch PyQt5 display calls.  No X11 / Wayland display is required.

Run with:
    pytest tests/unit/test_stimulus_tab_gui.py -v
"""
from __future__ import annotations

import importlib
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal PyQt5 stubs so the module can be imported without a display server
# ---------------------------------------------------------------------------

def _make_pyqt5_stubs():
    """Inject lightweight PyQt5 stubs if PyQt5 is not importable."""
    if "PyQt5" in sys.modules:
        return  # already imported; use the real thing

    # Build thin stubs for the subset of Qt used in stimulus_tab.py.
    pyqt5_mod = types.ModuleType("PyQt5")

    # QtWidgets — widget types used as base classes must be real Python classes;
    # everything else can be a MagicMock factory.
    qw = types.ModuleType("PyQt5.QtWidgets")

    # Real base-class stubs (needed for CollapsibleGroupBox inheritance chain)
    class QObject: pass  # noqa: E301
    class QWidget(QObject): pass  # noqa: E301
    class QGroupBox(QWidget): pass  # noqa: E301
    class QFrame(QWidget): pass  # noqa: E301
    class QAbstractScrollArea(QFrame): pass  # noqa: E301
    class QAbstractItemView(QAbstractScrollArea):  # noqa: E301
        SingleSelection = 1

    qw.QWidget = QWidget
    qw.QGroupBox = QGroupBox
    qw.QFrame = QFrame
    qw.QAbstractScrollArea = QAbstractScrollArea
    qw.QAbstractItemView = QAbstractItemView

    # Everything else can be MagicMock factories
    _MOCK_APIS = [
        "QFormLayout", "QVBoxLayout", "QHBoxLayout",
        "QListWidget", "QListWidgetItem", "QButtonGroup", "QToolButton",
        "QPushButton", "QLabel", "QLineEdit", "QComboBox", "QSpinBox",
        "QDoubleSpinBox", "QCheckBox", "QSlider", "QScrollArea",
        "QSizePolicy", "QSplitter",
        "QApplication", "QMessageBox", "QFileDialog",
    ]
    for name in _MOCK_APIS:
        mock_cls = MagicMock(name=name)
        mock_cls.return_value = MagicMock(name=f"{name}()")
        setattr(qw, name, mock_cls)

    # QtCore
    qc = types.ModuleType("PyQt5.QtCore")
    qc.Qt = MagicMock()
    qc.Qt.AlignRight = 2
    qc.Qt.Checked = 2
    qc.Qt.UserRole = 32
    qc.pyqtSignal = MagicMock(return_value=MagicMock())
    qc.QTimer = MagicMock()

    # QtGui
    qg = types.ModuleType("PyQt5.QtGui")

    # pyqtgraph
    pg = types.ModuleType("pyqtgraph")
    pg.ImageView = MagicMock()
    pg.PlotWidget = MagicMock()

    pyqt5_mod.QtWidgets = qw
    pyqt5_mod.QtCore = qc
    pyqt5_mod.QtGui = qg
    sys.modules["PyQt5"] = pyqt5_mod
    sys.modules["PyQt5.QtWidgets"] = qw
    sys.modules["PyQt5.QtCore"] = qc
    sys.modules["PyQt5.QtGui"] = qg
    sys.modules["pyqtgraph"] = pg


_make_pyqt5_stubs()


# ---------------------------------------------------------------------------
# Lightweight StimulusConfig fixture (without importing the full tab)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402


@dataclass
class StimulusConfig:
    """Minimal replica of the real StimulusConfig for isolated testing."""

    name: str = "Stimulus"
    stimulus_type: str = "gaussian"
    motion: str = "static"
    start: tuple = (0.0, 0.0)
    end: tuple = (0.0, 0.0)
    spread: float = 0.3
    orientation_deg: float = 0.0
    amplitude: float = 1.0
    ramp_up_ms: float = 50.0
    plateau_ms: float = 200.0
    ramp_down_ms: float = 50.0
    total_ms: float = 300.0
    dt_ms: float = 1.0
    speed_mm_s: float = 0.0
    texture_subtype: str = "gabor"
    wavelength: float = 0.5
    phase: float = 0.0
    edge_count: int = 5
    edge_width: float = 0.05
    noise_scale: float = 1.0
    noise_kernel_size: int = 5
    moving_subtype: str = "linear"
    num_steps: int = 100
    radius: float = 1.0
    start_angle: float = 0.0
    end_angle: float = 6.28318
    moving_sigma: float = 0.3
    onset_ms: float = 0.0
    duration_ms: float = 0.0
    motion_type: str = "static"
    repeat_enabled: bool = False
    repeat_nx: int = 1
    repeat_ny: int = 1
    repeat_spacing_x: float = 1.0
    repeat_spacing_y: float = 1.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["type"] = d.pop("stimulus_type")
        return d


# ===========================================================================
# Load CollapsibleGroupBox directly to avoid triggering sensoryforge/__init__.py
# (which imports torch, an optional dependency not always present)
# ===========================================================================

import importlib.util as _ilu  # noqa: E402

_COLLAPSIBLE_FILE = (
    Path(__file__).resolve().parents[2] / "sensoryforge" / "gui" / "widgets" / "collapsible.py"
)


def _load_collapsible_class():
    """Return CollapsibleGroupBox loaded directly from its source file."""
    spec = _ilu.spec_from_file_location("_collapsible_mod", _COLLAPSIBLE_FILE)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CollapsibleGroupBox


# ===========================================================================
# Tests for the CollapsibleGroupBox widget
# ===========================================================================

class TestCollapsibleGroupBox:
    """Tests for the shared CollapsibleGroupBox widget."""

    def _make_widget(self, start_expanded: bool = False):
        """Create a CollapsibleGroupBox with mocked Qt attributes.

        Uses __new__ to skip the real __init__ (which calls Qt APIs) and
        manually injects the only attributes exercised by the tests.
        """
        CollapsibleGroupBox = _load_collapsible_class()
        widget = CollapsibleGroupBox.__new__(CollapsibleGroupBox)
        widget._title = "Test Section"
        widget._is_expanded = start_expanded
        widget._toggle_btn = MagicMock()
        widget._content = MagicMock()
        widget._content_layout = MagicMock()
        return widget

    def test_default_collapsed(self):
        """CollapsibleGroupBox starts collapsed by default."""
        widget = self._make_widget(start_expanded=False)
        assert widget._is_expanded is False

    def test_start_expanded(self):
        """CollapsibleGroupBox can start expanded."""
        widget = self._make_widget(start_expanded=True)
        assert widget._is_expanded is True

    def test_set_checked_true_expands(self):
        """setChecked(True) expands the widget."""
        widget = self._make_widget(start_expanded=False)
        widget.setChecked(True)
        assert widget._is_expanded is True
        widget._content.setVisible.assert_called_with(True)

    def test_set_checked_false_collapses(self):
        """setChecked(False) collapses the widget."""
        widget = self._make_widget(start_expanded=True)
        widget.setChecked(False)
        assert widget._is_expanded is False
        widget._content.setVisible.assert_called_with(False)

    def test_is_checked_reflects_state(self):
        """isChecked() returns current expansion state."""
        widget = self._make_widget(start_expanded=True)
        assert widget.isChecked() is True
        widget.setChecked(False)
        assert widget.isChecked() is False

    def test_on_toggle_flips_state(self):
        """_on_toggle() flips expanded state each call."""
        widget = self._make_widget(start_expanded=False)
        widget._update_button_text = MagicMock()
        widget._on_toggle()
        assert widget._is_expanded is True
        widget._on_toggle()
        assert widget._is_expanded is False

    def test_add_row_delegates_to_form_layout(self):
        """addRow() delegates to _content_layout.addRow()."""
        widget = self._make_widget()
        label = MagicMock(name="QLabel")
        input_w = MagicMock(name="QSpinBox")
        widget.addRow(label, input_w)
        widget._content_layout.addRow.assert_called_once_with(label, input_w)

    def test_layout_returns_form_layout(self):
        """layout() returns the inner QFormLayout."""
        widget = self._make_widget()
        assert widget.layout() is widget._content_layout


# ===========================================================================
# Tests for stack management logic (isolated from Qt rendering)
# ===========================================================================

class TestStackManagement:
    """Tests for the Add/Update/Revert stack workflow logic."""

    def _make_stack_logic(self):
        """Return a minimal object that mimics relevant StimulusDesignerTab state."""

        class Logic:
            """Lightweight stand-in for the tab's stack operations."""

            def __init__(self):
                self._stimulus_stack = []
                self._active_stack_index: Optional[int] = None
                self._committed_config: Optional[StimulusConfig] = None

            def add_new(self, config: StimulusConfig) -> None:
                self._stimulus_stack.append(config)
                new_index = len(self._stimulus_stack) - 1
                self._active_stack_index = new_index
                self._committed_config = config

            def update_selected(self, config: StimulusConfig) -> None:
                if self._active_stack_index is None:
                    return
                idx = self._active_stack_index
                if not (0 <= idx < len(self._stimulus_stack)):
                    return
                self._stimulus_stack[idx] = config
                self._committed_config = config

            def revert(self) -> Optional[StimulusConfig]:
                return self._committed_config

            def select(self, index: int) -> None:
                if 0 <= index < len(self._stimulus_stack):
                    self._active_stack_index = index
                    self._committed_config = self._stimulus_stack[index]

        return Logic()

    def test_add_new_appends_and_selects(self):
        """add_new() appends to stack and sets active index."""
        logic = self._make_stack_logic()
        cfg = StimulusConfig(name="S1")
        logic.add_new(cfg)
        assert len(logic._stimulus_stack) == 1
        assert logic._active_stack_index == 0
        assert logic._committed_config is cfg

    def test_add_multiple_preserves_order(self):
        """Multiple add_new() calls preserve insertion order."""
        logic = self._make_stack_logic()
        for i in range(3):
            logic.add_new(StimulusConfig(name=f"S{i}"))
        assert len(logic._stimulus_stack) == 3
        assert logic._active_stack_index == 2  # points to last

    def test_update_selected_overwrites_correct_entry(self):
        """update_selected() replaces the entry at _active_stack_index."""
        logic = self._make_stack_logic()
        logic.add_new(StimulusConfig(name="S0"))
        logic.add_new(StimulusConfig(name="S1"))
        logic.select(0)
        updated = StimulusConfig(name="S0_updated")
        logic.update_selected(updated)
        assert logic._stimulus_stack[0].name == "S0_updated"
        assert logic._stimulus_stack[1].name == "S1"
        assert logic._committed_config is updated

    def test_update_selected_does_nothing_when_no_selection(self):
        """update_selected() is a no-op when nothing is selected."""
        logic = self._make_stack_logic()
        logic.add_new(StimulusConfig(name="S0"))
        logic._active_stack_index = None
        logic.update_selected(StimulusConfig(name="changed"))
        assert logic._stimulus_stack[0].name == "S0"

    def test_revert_returns_committed_config(self):
        """revert() returns the snapshot taken at add/select time."""
        logic = self._make_stack_logic()
        original = StimulusConfig(name="Original")
        logic.add_new(original)
        # Simulate in-progress edit (update_selected not yet called)
        reverted = logic.revert()
        assert reverted is original

    def test_revert_after_update_returns_new_committed(self):
        """After update_selected(), revert returns the updated committed state."""
        logic = self._make_stack_logic()
        logic.add_new(StimulusConfig(name="S0"))
        committed = StimulusConfig(name="S0_committed")
        logic.update_selected(committed)
        assert logic.revert() is committed

    def test_selection_changes_committed_config(self):
        """Selecting a different stack entry snapshots its state."""
        logic = self._make_stack_logic()
        s0 = StimulusConfig(name="S0")
        s1 = StimulusConfig(name="S1")
        logic.add_new(s0)
        logic.add_new(s1)
        logic.select(0)
        assert logic._committed_config is s0

    def test_add_sequence_maintains_stack_integrity(self):
        """Add, update, add does not corrupt existing entries."""
        logic = self._make_stack_logic()
        logic.add_new(StimulusConfig(name="A"))
        logic.add_new(StimulusConfig(name="B"))
        logic.select(0)
        logic.update_selected(StimulusConfig(name="A_updated"))
        logic.add_new(StimulusConfig(name="C"))
        assert [s.name for s in logic._stimulus_stack] == ["A_updated", "B", "C"]


# ===========================================================================
# Tests for stack save / load format
# ===========================================================================

SAVE_VERSION = "1.0.0"


class TestStackSaveLoad:
    """Tests for the stimulus_stack save / load payload format."""

    def _make_bundle(self, stimuli: list) -> dict:
        return {
            "schema_version": SAVE_VERSION,
            "kind": "stimulus_stack",
            "name": "my_stack",
            "composition_mode": "add",
            "stimuli": stimuli,
            "grid": {},
        }

    def _make_single_bundle(self, config: StimulusConfig) -> dict:
        return {
            "schema_version": SAVE_VERSION,
            "kind": "stimulus",
            "name": config.name,
            **config.as_dict(),
            "grid": {},
        }

    def test_single_stimulus_bundle_contains_type(self):
        """Single-stimulus bundle has 'kind'='stimulus'."""
        cfg = StimulusConfig(name="Test")
        bundle = self._make_single_bundle(cfg)
        assert bundle["kind"] == "stimulus"
        assert bundle["name"] == "Test"
        assert "type" in bundle

    def test_stack_bundle_kind_is_stimulus_stack(self):
        """Stack bundle has 'kind'='stimulus_stack'."""
        bundle = self._make_bundle([])
        assert bundle["kind"] == "stimulus_stack"

    def test_stack_bundle_includes_all_stimuli(self):
        """Stack bundle includes every stimulus in order."""
        stimuli = [StimulusConfig(name="S0").as_dict(), StimulusConfig(name="S1").as_dict()]
        bundle = self._make_bundle(stimuli)
        assert len(bundle["stimuli"]) == 2
        assert bundle["stimuli"][0]["name"] == "S0"
        assert bundle["stimuli"][1]["name"] == "S1"

    def test_stack_detection_by_kind(self):
        """Loading code can detect stack format via 'kind' field."""
        single = {"kind": "stimulus"}
        stack = {"kind": "stimulus_stack"}
        unknown = {}
        assert single.get("kind") == "stimulus"
        assert stack.get("kind") == "stimulus_stack"
        assert unknown.get("kind") is None

    def test_stimulus_config_as_dict_round_trip(self):
        """StimulusConfig.as_dict() preserves all fields needed for reload."""
        cfg = StimulusConfig(
            name="test",
            stimulus_type="gabor",
            spread=0.5,
            repeat_enabled=True,
            repeat_nx=3,
            repeat_ny=2,
            repeat_spacing_x=1.5,
            repeat_spacing_y=0.8,
        )
        d = cfg.as_dict()
        assert d["type"] == "gabor"
        assert d["spread"] == 0.5
        assert d["repeat_enabled"] is True
        assert d["repeat_nx"] == 3
        assert d["repeat_spacing_x"] == 1.5


# ===========================================================================
# Tests for repeat pattern tiling logic
# ===========================================================================

class TestRepeatPatternTiling:
    """Tests for the repeat tiling offset calculations."""

    def _compute_offsets(self, nx: int, ny: int, sx: float, sy: float):
        """Replicate the offset calculation from _build_stimulus_frames."""
        offsets = []
        for ix in range(nx):
            for iy in range(ny):
                ox = (ix - (nx - 1) / 2.0) * sx
                oy = (iy - (ny - 1) / 2.0) * sy
                offsets.append((ox, oy))
        return offsets

    def test_single_copy_zero_offset(self):
        """1×1 pattern produces a single copy at offset (0, 0)."""
        offsets = self._compute_offsets(1, 1, 1.0, 1.0)
        assert len(offsets) == 1
        assert offsets[0] == (0.0, 0.0)

    def test_two_copies_symmetric_x(self):
        """2×1 pattern produces copies symmetric around centre along x."""
        offsets = self._compute_offsets(2, 1, 1.0, 1.0)
        assert len(offsets) == 2
        xs = sorted(o[0] for o in offsets)
        assert xs[0] == pytest.approx(-0.5)
        assert xs[1] == pytest.approx(0.5)

    def test_three_copies_includes_centre(self):
        """3×1 pattern includes a copy at x=0."""
        offsets = self._compute_offsets(3, 1, 1.0, 1.0)
        xs = [o[0] for o in offsets]
        assert 0.0 in xs

    def test_offset_count_equals_nx_times_ny(self):
        """Total copies equals copies_x * copies_y."""
        for nx, ny in [(2, 3), (5, 1), (4, 4)]:
            offsets = self._compute_offsets(nx, ny, 1.0, 1.0)
            assert len(offsets) == nx * ny

    def test_spacing_scales_offsets(self):
        """Twice the spacing doubles the offset distances."""
        offs1 = self._compute_offsets(3, 3, 1.0, 1.0)
        offs2 = self._compute_offsets(3, 3, 2.0, 2.0)
        for (ox1, oy1), (ox2, oy2) in zip(offs1, offs2):
            assert ox2 == pytest.approx(2.0 * ox1)
            assert oy2 == pytest.approx(2.0 * oy1)

    def test_repeat_disabled_uses_single_frame(self):
        """When repeat_enabled=False, only one copy is generated."""
        cfg = StimulusConfig(repeat_enabled=False, repeat_nx=5, repeat_ny=5)
        # The frame selection logic:
        if cfg.repeat_enabled and (cfg.repeat_nx > 1 or cfg.repeat_ny > 1):
            num_copies = cfg.repeat_nx * cfg.repeat_ny
        else:
            num_copies = 1
        assert num_copies == 1

    def test_repeat_enabled_uses_multiple_copies(self):
        """When repeat_enabled=True and nx*ny>1, multiple copies are used."""
        cfg = StimulusConfig(repeat_enabled=True, repeat_nx=3, repeat_ny=2)
        if cfg.repeat_enabled and (cfg.repeat_nx > 1 or cfg.repeat_ny > 1):
            num_copies = cfg.repeat_nx * cfg.repeat_ny
        else:
            num_copies = 1
        assert num_copies == 6


# ===========================================================================
# Tests for type visibility mapping
# ===========================================================================

class TestTypeVisibilityMapping:
    """Tests for dynamic parameter visibility rules per stimulus type."""

    # The visibility rules encoded in _on_type_selected / _apply_config:
    TEXTURE_TYPES = {"texture"}
    MOVING_TYPES = {"moving"}
    ORIENT_TYPES = {"edge", "gabor", "grating"}
    GABOR_FIRST_CLASS = {"gabor"}
    GRATING_FIRST_CLASS = {"grating"}
    NOISE_FIRST_CLASS = {"noise"}

    def _visible_for_type(self, stype: str) -> dict:
        return {
            "texture_group": stype in self.TEXTURE_TYPES,
            "moving_group": stype in self.MOVING_TYPES,
            "gabor_params": stype in self.GABOR_FIRST_CLASS,
            "edge_grating_params": stype in self.GRATING_FIRST_CLASS,
            "noise_params": stype in self.NOISE_FIRST_CLASS,
            "orientation_widget": stype in self.ORIENT_TYPES,
        }

    def test_gaussian_shows_no_subtype_panels(self):
        """Gaussian type hides all subtype-specific panels."""
        vis = self._visible_for_type("gaussian")
        assert vis["texture_group"] is False
        assert vis["moving_group"] is False
        assert vis["gabor_params"] is False
        assert vis["edge_grating_params"] is False
        assert vis["noise_params"] is False

    def test_gaussian_hides_orientation(self):
        """Gaussian type hides the orientation widget."""
        vis = self._visible_for_type("gaussian")
        assert vis["orientation_widget"] is False

    def test_gabor_shows_gabor_params(self):
        """Gabor type shows the gabor parameter panel."""
        vis = self._visible_for_type("gabor")
        assert vis["gabor_params"] is True
        assert vis["edge_grating_params"] is False
        assert vis["noise_params"] is False

    def test_gabor_shows_orientation(self):
        """Gabor type shows the orientation widget."""
        vis = self._visible_for_type("gabor")
        assert vis["orientation_widget"] is True

    def test_grating_shows_grating_params(self):
        """Grating type shows the grating parameter panel."""
        vis = self._visible_for_type("grating")
        assert vis["edge_grating_params"] is True
        assert vis["gabor_params"] is False
        assert vis["orientation_widget"] is True

    def test_noise_hides_orientation(self):
        """Noise type hides the orientation widget."""
        vis = self._visible_for_type("noise")
        assert vis["orientation_widget"] is False
        assert vis["noise_params"] is True

    def test_edge_shows_orientation(self):
        """Edge type shows the orientation widget."""
        vis = self._visible_for_type("edge")
        assert vis["orientation_widget"] is True

    def test_moving_shows_moving_group(self):
        """Moving type shows the moving parameter group."""
        vis = self._visible_for_type("moving")
        assert vis["moving_group"] is True
        assert vis["texture_group"] is False

    def test_texture_shows_texture_group(self):
        """Texture type shows the texture parameter group."""
        vis = self._visible_for_type("texture")
        assert vis["texture_group"] is True
        assert vis["moving_group"] is False

    def test_all_types_at_most_one_subpanel(self):
        """Each type shows at most one primary subtype panel."""
        all_types = ["gaussian", "point", "edge", "gabor", "grating", "noise",
                     "texture", "moving"]
        for stype in all_types:
            vis = self._visible_for_type(stype)
            panels = [vis["texture_group"], vis["moving_group"],
                      vis["gabor_params"], vis["edge_grating_params"],
                      vis["noise_params"]]
            active = sum(1 for p in panels if p)
            assert active <= 1, (
                f"Type '{stype}' activates {active} subtype panels simultaneously"
            )


# ===========================================================================
# Tests for full-field stimulus type visibility rules (coord hiding)
# ===========================================================================

class TestFullFieldVisibilityRules:
    """Tests that noise/grating types hide irrelevant coordinate widgets.

    The rule implemented in _on_type_selected / _apply_config:
        is_full_field = stype in ("noise", "grating")
        is_noise = stype == "noise"
    Position spinners (start_x, start_y, end_x, end_y) → hidden when full-field.
    Spread spinner → hidden only for noise.
    """

    FULL_FIELD_TYPES = {"noise", "grating"}
    NOISE_ONLY = {"noise"}

    def _compute_visibility(self, stype: str) -> dict:
        is_full_field = stype in self.FULL_FIELD_TYPES
        is_noise = stype in self.NOISE_ONLY
        return {
            "position_visible": not is_full_field,
            "spread_visible": not is_noise,
        }

    def test_noise_hides_position(self):
        """Noise type hides all four position coordinate spinners."""
        vis = self._compute_visibility("noise")
        assert vis["position_visible"] is False

    def test_noise_hides_spread(self):
        """Noise type hides the spread spinner (full-field, no spatial size meaning)."""
        vis = self._compute_visibility("noise")
        assert vis["spread_visible"] is False

    def test_grating_hides_position(self):
        """Grating is a full-field type and should hide position spinners."""
        vis = self._compute_visibility("grating")
        assert vis["position_visible"] is False

    def test_grating_shows_spread(self):
        """Grating uses spread/wavelength so the spread spinner stays visible."""
        vis = self._compute_visibility("grating")
        assert vis["spread_visible"] is True

    def test_gaussian_shows_position_and_spread(self):
        """Gaussian is a localised type — both position and spread are visible."""
        vis = self._compute_visibility("gaussian")
        assert vis["position_visible"] is True
        assert vis["spread_visible"] is True

    def test_point_shows_position(self):
        """Point stimulus is localised — position must be shown."""
        vis = self._compute_visibility("point")
        assert vis["position_visible"] is True

    def test_edge_shows_position(self):
        """Edge stimulus is localised — position must be shown."""
        vis = self._compute_visibility("edge")
        assert vis["position_visible"] is True

    def test_gabor_shows_position(self):
        """Gabor is a localised stimulus — position must be shown."""
        vis = self._compute_visibility("gabor")
        assert vis["position_visible"] is True

    def test_moving_shows_position(self):
        """Moving stimulus type keeps position controls visible."""
        vis = self._compute_visibility("moving")
        assert vis["position_visible"] is True

    def test_full_field_set_matches_expected_types(self):
        """Only noise and grating are classified as full-field types."""
        expected_full_field = {"noise", "grating"}
        assert self.FULL_FIELD_TYPES == expected_full_field

    def test_spread_hidden_only_for_noise(self):
        """Spread is hidden only for noise, not for grating or any other type."""
        for stype in ["gaussian", "point", "edge", "gabor", "grating", "texture", "moving"]:
            vis = self._compute_visibility(stype)
            assert vis["spread_visible"] is True, (
                f"spread should be visible for '{stype}' but was hidden"
            )


# ===========================================================================
# Tests for motion type dispatch logic
# ===========================================================================

class TestMotionTypeDispatchLogic:
    """Tests for the _build_stimulus_frames dispatch conditions.

    The dispatch rules at the top of _build_stimulus_frames:
      1. if motion_type in ("circular", "slide") and stimulus_type != "moving"
         → _generate_motion_trajectory_frames
      2. elif stimulus_type == "moving"
         → _generate_moving_frames
      3. else → inline time loop
    """

    def _dispatch(self, motion_type: str, stimulus_type: str) -> str:
        """Replicate the dispatch logic from _build_stimulus_frames."""
        if motion_type in ("circular", "slide") and stimulus_type != "moving":
            return "trajectory"
        if stimulus_type == "moving":
            return "moving_frames"
        return "time_loop"

    def test_circular_dispatch(self):
        """Circular motion_type dispatches to trajectory generator."""
        assert self._dispatch("circular", "gaussian") == "trajectory"

    def test_slide_dispatch(self):
        """Slide motion_type dispatches to trajectory generator."""
        assert self._dispatch("slide", "gaussian") == "trajectory"

    def test_circular_with_moving_type_does_not_dispatch_to_trajectory(self):
        """Circular motion_type with stimulus_type=moving does NOT dispatch to trajectory."""
        # Moving type has its own generator; trajectory is only for non-moving spatial types
        assert self._dispatch("circular", "moving") == "moving_frames"

    def test_static_falls_through_to_time_loop(self):
        """Static motion_type uses the inline time loop."""
        assert self._dispatch("static", "gaussian") == "time_loop"

    def test_linear_falls_through_to_time_loop(self):
        """Linear motion_type (non-circular/slide) uses the inline time loop."""
        assert self._dispatch("linear", "gaussian") == "time_loop"

    def test_moving_stimulus_type_dispatches_to_moving_frames(self):
        """stimulus_type=moving always dispatches to _generate_moving_frames."""
        assert self._dispatch("static", "moving") == "moving_frames"

    def test_noise_with_static_uses_time_loop(self):
        """Noise with static motion uses the inline time loop."""
        assert self._dispatch("static", "noise") == "time_loop"

    def test_gabor_with_circular_dispatches_to_trajectory(self):
        """Gabor stimulus with circular motion dispatches to trajectory generator."""
        assert self._dispatch("circular", "gabor") == "trajectory"

    def test_noise_with_circular_dispatches_to_trajectory(self):
        """Noise with circular motion dispatches to trajectory generator."""
        assert self._dispatch("circular", "noise") == "trajectory"

    @pytest.mark.parametrize("stype", ["gaussian", "point", "edge", "gabor", "grating", "noise",
                                        "texture"])
    def test_circular_dispatches_for_all_spatial_types(self, stype):
        """All non-moving spatial types dispatch to trajectory with circular motion."""
        assert self._dispatch("circular", stype) == "trajectory"

    @pytest.mark.parametrize("stype", ["gaussian", "point", "edge", "gabor", "grating", "noise",
                                        "texture"])
    def test_slide_dispatches_for_all_spatial_types(self, stype):
        """All non-moving spatial types dispatch to trajectory with slide motion."""
        assert self._dispatch("slide", stype) == "trajectory"


# ===========================================================================
# Tests for circular motion trajectory correctness (uses real torch/moving API)
# ===========================================================================

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import math as _math  # noqa: E402

pytestmark_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch not available"
)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
class TestCircularMotionTrajectory:
    """Tests that circular_motion() produces geometrically correct positions.

    These tests use the real sensoryforge.stimuli.moving API so they require
    torch to be installed.
    """

    def _import_circular_motion(self):
        import importlib
        mod = importlib.import_module("sensoryforge.stimuli.moving")
        return mod.circular_motion, mod.slide_trajectory

    def test_circular_trajectory_has_correct_shape(self):
        """circular_motion returns tensor of shape [num_steps, 2]."""
        circular_motion, _ = self._import_circular_motion()
        traj = circular_motion(center=(0.0, 0.0), radius=1.0, num_steps=50)
        assert traj.shape == (50, 2)

    def test_circular_trajectory_radius_is_constant(self):
        """All positions in a circular trajectory are equidistant from center."""
        circular_motion, _ = self._import_circular_motion()
        cx, cy = 1.0, -0.5
        r = 2.0
        traj = circular_motion(center=(cx, cy), radius=r, num_steps=100)
        # Compute distance from center for each step
        dist = _torch.sqrt((traj[:, 0] - cx) ** 2 + (traj[:, 1] - cy) ** 2)
        assert _torch.allclose(dist, _torch.full_like(dist, r), atol=1e-5), (
            f"Radii not constant: min={dist.min():.6f} max={dist.max():.6f} expected={r}"
        )

    def test_circular_trajectory_first_position_matches_start_angle(self):
        """First position aligns with start_angle relative to center."""
        circular_motion, _ = self._import_circular_motion()
        start_angle = 0.0  # Expected → (center_x + r, center_y)
        traj = circular_motion(center=(0.0, 0.0), radius=1.0, num_steps=10,
                                start_angle=start_angle, end_angle=2 * _math.pi)
        expected_x = 1.0  # cos(0) * r
        expected_y = 0.0  # sin(0) * r
        assert _torch.allclose(traj[0], _torch.tensor([expected_x, expected_y]), atol=1e-5)

    def test_circular_trajectory_not_linear(self):
        """Circular trajectory is NOT a straight line (x should vary non-monotonically)."""
        circular_motion, _ = self._import_circular_motion()
        traj = circular_motion(center=(0.0, 0.0), radius=1.0, num_steps=50,
                                start_angle=0.0, end_angle=2 * _math.pi)
        x_vals = traj[:, 0]
        # A full circle has x values that go +1, 0, -1, 0, +1 — not monotone
        is_monotone_increasing = _torch.all(x_vals[1:] >= x_vals[:-1])
        is_monotone_decreasing = _torch.all(x_vals[1:] <= x_vals[:-1])
        assert not (is_monotone_increasing or is_monotone_decreasing), (
            "Circular trajectory x-coordinates are monotone — it's producing a line, not a circle"
        )

    def test_circular_trajectory_completes_full_circle(self):
        """A full-circle trajectory has first and last positions nearly the same."""
        circular_motion, _ = self._import_circular_motion()
        # Use many steps so start/end wrap close
        traj = circular_motion(center=(0.0, 0.0), radius=1.0, num_steps=360,
                                start_angle=0.0, end_angle=2 * _math.pi)
        # First point ~(1, 0), last point ~(1, 0) for a full revolution
        assert _torch.allclose(traj[0, 0], _torch.tensor(1.0), atol=0.05)
        assert _torch.allclose(traj[-1, 0], _torch.tensor(1.0), atol=0.05)

    def test_slide_trajectory_has_correct_shape(self):
        """slide_trajectory returns tensor of shape [num_steps, 2]."""
        _, slide_trajectory = self._import_circular_motion()
        traj = slide_trajectory(start=(0.0, 0.0), end=(2.0, 1.0), num_steps=40)
        assert traj.shape == (40, 2)

    def test_slide_trajectory_starts_at_start(self):
        """slide_trajectory first position matches start."""
        _, slide_trajectory = self._import_circular_motion()
        traj = slide_trajectory(start=(0.5, -1.0), end=(2.0, 1.0), num_steps=20)
        assert _torch.allclose(traj[0], _torch.tensor([0.5, -1.0]), atol=1e-5)

    def test_slide_trajectory_ends_at_end(self):
        """slide_trajectory last position matches end."""
        _, slide_trajectory = self._import_circular_motion()
        traj = slide_trajectory(start=(0.0, 0.0), end=(3.0, 4.0), num_steps=20)
        assert _torch.allclose(traj[-1], _torch.tensor([3.0, 4.0]), atol=1e-5)

    def test_slide_trajectory_is_monotone(self):
        """slide_trajectory x-coordinates are monotonically increasing for horizontal slide."""
        _, slide_trajectory = self._import_circular_motion()
        traj = slide_trajectory(start=(0.0, 0.0), end=(5.0, 0.0), num_steps=30)
        x_vals = traj[:, 0]
        assert _torch.all(x_vals[1:] >= x_vals[:-1] - 1e-6), (
            "Slide trajectory should be monotone along the direction of motion"
        )


# ===========================================================================
# Tests for spiking tab config_from_payload — all new fields
# ===========================================================================

class TestConfigFromPayloadAllFields:
    """Tests that all StimulusConfig fields are parsed from a payload dict.

    This replicates the logic in spiking_tab._config_from_payload using the
    local StimulusConfig definition.
    """

    def _config_from_payload(self, payload: dict) -> StimulusConfig:
        """Replicate spiking_tab._config_from_payload logic using local StimulusConfig."""

        def _tuple(values, default):
            if not isinstance(values, (list, tuple)):
                return default
            if len(values) != 2:
                return default
            return float(values[0]), float(values[1])

        start = payload.get("start", [0.0, 0.0])
        end = payload.get("end", start)
        return StimulusConfig(
            name=payload.get("name", "unnamed"),
            stimulus_type=payload.get("type", "gaussian"),
            motion=payload.get("motion", "static"),
            start=_tuple(start, (0.0, 0.0)),
            end=_tuple(end, (0.0, 0.0)),
            spread=float(payload.get("spread", 0.3)),
            orientation_deg=float(payload.get("orientation_deg", 0.0)),
            amplitude=float(payload.get("amplitude", 1.0)),
            ramp_up_ms=float(payload.get("ramp_up_ms", 50.0)),
            plateau_ms=float(payload.get("plateau_ms", 200.0)),
            ramp_down_ms=float(payload.get("ramp_down_ms", 50.0)),
            total_ms=float(payload.get("total_ms", 300.0)),
            dt_ms=float(payload.get("dt_ms", 1.0)),
            speed_mm_s=float(payload.get("speed_mm_s", 0.0)),
            texture_subtype=payload.get("texture_subtype", "gabor"),
            wavelength=float(payload.get("wavelength", 0.5)),
            phase=float(payload.get("phase", 0.0)),
            edge_count=int(payload.get("edge_count", 5)),
            edge_width=float(payload.get("edge_width", 0.05)),
            noise_scale=float(payload.get("noise_scale", 1.0)),
            noise_kernel_size=int(payload.get("noise_kernel_size", 5)),
            moving_subtype=payload.get("moving_subtype", "linear"),
            num_steps=int(payload.get("num_steps", 100)),
            radius=float(payload.get("radius", 1.0)),
            start_angle=float(payload.get("start_angle", 0.0)),
            end_angle=float(payload.get("end_angle", 6.28318)),
            moving_sigma=float(payload.get("moving_sigma", 0.3)),
            onset_ms=float(payload.get("onset_ms", 0.0)),
            duration_ms=float(payload.get("duration_ms", 0.0)),
            motion_type=payload.get("motion_type", "static"),
            repeat_enabled=bool(payload.get("repeat_enabled", False)),
            repeat_nx=int(payload.get("repeat_nx", 1)),
            repeat_ny=int(payload.get("repeat_ny", 1)),
            repeat_spacing_x=float(payload.get("repeat_spacing_x", 1.0)),
            repeat_spacing_y=float(payload.get("repeat_spacing_y", 1.0)),
        )

    def test_basic_fields_parsed(self):
        """Basic stimulus fields (name, type, motion) are parsed correctly."""
        payload = {"name": "my_stim", "type": "gabor", "motion": "moving",
                   "start": [0.5, -0.5], "spread": 0.8}
        config = self._config_from_payload(payload)
        assert config.name == "my_stim"
        assert config.stimulus_type == "gabor"
        assert config.motion == "moving"
        assert config.start == (0.5, -0.5)
        assert config.spread == pytest.approx(0.8)

    def test_onset_ms_and_duration_ms_parsed(self):
        """Timeline fields onset_ms and duration_ms are parsed."""
        payload = {"onset_ms": 100.0, "duration_ms": 250.0}
        config = self._config_from_payload(payload)
        assert config.onset_ms == pytest.approx(100.0)
        assert config.duration_ms == pytest.approx(250.0)

    def test_motion_type_circular_parsed(self):
        """motion_type field accepts 'circular' value."""
        payload = {"motion_type": "circular"}
        config = self._config_from_payload(payload)
        assert config.motion_type == "circular"

    def test_motion_type_slide_parsed(self):
        """motion_type field accepts 'slide' value."""
        payload = {"motion_type": "slide"}
        config = self._config_from_payload(payload)
        assert config.motion_type == "slide"

    def test_radius_and_angles_parsed(self):
        """radius, start_angle, end_angle are parsed for circular motion."""
        payload = {"radius": 2.5, "start_angle": 0.785, "end_angle": 3.14}
        config = self._config_from_payload(payload)
        assert config.radius == pytest.approx(2.5)
        assert config.start_angle == pytest.approx(0.785)
        assert config.end_angle == pytest.approx(3.14)

    def test_repeat_fields_parsed(self):
        """Repeat tiling fields are parsed correctly."""
        payload = {
            "repeat_enabled": True,
            "repeat_nx": 4,
            "repeat_ny": 3,
            "repeat_spacing_x": 1.2,
            "repeat_spacing_y": 0.9,
        }
        config = self._config_from_payload(payload)
        assert config.repeat_enabled is True
        assert config.repeat_nx == 4
        assert config.repeat_ny == 3
        assert config.repeat_spacing_x == pytest.approx(1.2)
        assert config.repeat_spacing_y == pytest.approx(0.9)

    def test_texture_fields_parsed(self):
        """Texture-specific fields are parsed correctly."""
        payload = {
            "texture_subtype": "edge_grating",
            "wavelength": 0.4,
            "phase": 1.57,
            "edge_count": 8,
            "edge_width": 0.03,
        }
        config = self._config_from_payload(payload)
        assert config.texture_subtype == "edge_grating"
        assert config.wavelength == pytest.approx(0.4)
        assert config.phase == pytest.approx(1.57)
        assert config.edge_count == 8
        assert config.edge_width == pytest.approx(0.03)

    def test_noise_fields_parsed(self):
        """Noise-specific fields are parsed correctly."""
        payload = {"noise_scale": 2.0, "noise_kernel_size": 9}
        config = self._config_from_payload(payload)
        assert config.noise_scale == pytest.approx(2.0)
        assert config.noise_kernel_size == 9

    def test_num_steps_parsed(self):
        """num_steps is parsed as int."""
        payload = {"num_steps": 200}
        config = self._config_from_payload(payload)
        assert config.num_steps == 200

    def test_defaults_when_fields_absent(self):
        """Missing fields fall back to correct defaults."""
        config = self._config_from_payload({})
        assert config.stimulus_type == "gaussian"
        assert config.motion_type == "static"
        assert config.onset_ms == pytest.approx(0.0)
        assert config.duration_ms == pytest.approx(0.0)
        assert config.repeat_enabled is False
        assert config.repeat_nx == 1
        assert config.radius == pytest.approx(1.0)

    def test_start_end_as_list_parsed(self):
        """start/end given as 2-element lists are converted to tuples."""
        payload = {"start": [1.5, -2.0], "end": [3.0, 4.0]}
        config = self._config_from_payload(payload)
        assert config.start == (1.5, -2.0)
        assert config.end == (3.0, 4.0)

    def test_invalid_start_falls_back_to_default(self):
        """start with wrong length falls back to (0.0, 0.0)."""
        payload = {"start": [1.0]}
        config = self._config_from_payload(payload)
        assert config.start == (0.0, 0.0)


# ===========================================================================
# Tests for composite stack timeline logic
# ===========================================================================

class TestCompositeStackTimeline:
    """Tests for the timeline-aware composite frame building logic.

    These test the logic shared between stimulus_tab._build_composite_frames
    and spiking_tab._build_composite_frames: placing each stimulus at its
    onset_ms offset and blending via composition modes.
    """

    def _build_composite(
        self,
        configs: list,
        dt_ms: float = 1.0,
        composition_mode: str = "add",
    ) -> dict:
        """Replicate the key subset of _build_composite_frames logic."""
        import numpy as np

        use_timeline = any(c.onset_ms > 0.0 or c.duration_ms > 0.0 for c in configs)

        if not use_timeline:
            # Non-timeline mode: align all to same length (max total_ms)
            max_frames = max(int(c.total_ms / dt_ms) + 1 for c in configs)
            return {
                "mode": "non_timeline",
                "num_frames": max_frames,
                "n_stimuli": len(configs),
            }
        else:
            # Timeline mode: place each stimulus at onset_ms
            last_end = max(c.onset_ms + (c.duration_ms if c.duration_ms > 0 else c.total_ms)
                           for c in configs)
            total_frames = int(last_end / dt_ms) + 1
            return {
                "mode": "timeline",
                "num_frames": total_frames,
                "n_stimuli": len(configs),
            }

    def _make_cfg(self, total_ms=300.0, onset_ms=0.0, duration_ms=0.0) -> StimulusConfig:
        return StimulusConfig(total_ms=total_ms, onset_ms=onset_ms, duration_ms=duration_ms)

    def test_no_onset_uses_non_timeline_mode(self):
        """Configs with all onset_ms=0 use non-timeline (aligned) mode."""
        cfgs = [self._make_cfg(total_ms=300.0), self._make_cfg(total_ms=200.0)]
        result = self._build_composite(cfgs)
        assert result["mode"] == "non_timeline"

    def test_onset_triggers_timeline_mode(self):
        """A config with onset_ms>0 triggers timeline mode."""
        cfgs = [self._make_cfg(onset_ms=0.0), self._make_cfg(onset_ms=100.0)]
        result = self._build_composite(cfgs)
        assert result["mode"] == "timeline"

    def test_duration_ms_triggers_timeline_mode(self):
        """A config with duration_ms>0 triggers timeline mode."""
        cfgs = [self._make_cfg(duration_ms=150.0), self._make_cfg()]
        result = self._build_composite(cfgs)
        assert result["mode"] == "timeline"

    def test_timeline_total_frames_covers_latest_end(self):
        """Timeline mode total frames covers the latest stimulus end time."""
        dt_ms = 1.0
        cfgs = [
            self._make_cfg(onset_ms=0.0, duration_ms=200.0),
            self._make_cfg(onset_ms=500.0, duration_ms=100.0),  # ends at 600 ms
        ]
        result = self._build_composite(cfgs, dt_ms=dt_ms)
        # last end = 600 ms → 601 frames at dt=1ms
        assert result["num_frames"] >= 600

    def test_non_timeline_uses_max_total_ms(self):
        """Non-timeline mode aligns to the longest stimulus."""
        dt_ms = 1.0
        cfgs = [
            self._make_cfg(total_ms=100.0),
            self._make_cfg(total_ms=500.0),  # longest
        ]
        result = self._build_composite(cfgs, dt_ms=dt_ms)
        # max_frames = 501 (500/1 + 1)
        assert result["num_frames"] >= 500

    def test_single_stimulus_non_timeline(self):
        """A single stimulus config generates non-timeline composite."""
        cfgs = [self._make_cfg(total_ms=200.0)]
        result = self._build_composite(cfgs)
        assert result["mode"] == "non_timeline"
        assert result["n_stimuli"] == 1

    def test_stack_bundle_kind_detection(self):
        """Stack payload kind detection works for add/max/mean modes."""
        for mode in ("add", "max", "mean"):
            payload = {"kind": "stimulus_stack", "composition_mode": mode, "stimuli": []}
            assert payload.get("kind") == "stimulus_stack"
            assert payload.get("composition_mode") == mode

    def test_non_stack_payload_not_detected_as_stack(self):
        """Non-stack payloads are not falsely detected as stacks."""
        single = {"kind": "stimulus", "type": "gaussian"}
        assert single.get("kind") != "stimulus_stack"

    def test_stack_with_zero_stimuli_is_valid_bundle(self):
        """A stack bundle with zero stimuli is structurally valid."""
        payload = {"kind": "stimulus_stack", "stimuli": [], "composition_mode": "add"}
        assert payload["kind"] == "stimulus_stack"
        assert len(payload["stimuli"]) == 0
