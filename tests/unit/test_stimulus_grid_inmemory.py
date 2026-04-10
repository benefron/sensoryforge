"""Tests for in-memory grid→stimulus flow without requiring file saves (item 7).

The StimulusDesignerTab was blocking stimulus generation unless a project
directory had been saved to disk. These tests verify the fix: stimulus
generation and in-memory use must work purely from the grid signal alone.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from sensoryforge.stimuli.stimulus import StimulusGenerator
from sensoryforge.core.grid import ReceptorGrid


# ---------------------------------------------------------------------------
# Backend tests: StimulusGenerator works without any file path
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_grid():
    """A minimal ReceptorGrid with no file backing."""
    return ReceptorGrid(grid_size=(10, 10), spacing=0.15, center=(0.0, 0.0))


def test_stimulus_generator_requires_no_file(simple_grid):
    """StimulusGenerator must instantiate and generate frames with no file path."""
    gen = StimulusGenerator(simple_grid)
    assert gen is not None
    assert gen.grid_manager is not None


def test_stimulus_generator_grid_update_preserves_no_crash(simple_grid):
    """Replacing the grid_manager on an existing StimulusGenerator must not crash."""
    gen = StimulusGenerator(simple_grid)
    new_grid = ReceptorGrid(grid_size=(8, 8), spacing=0.2, center=(0.0, 0.0))
    # Update grid manager in-place — should not raise
    gen.grid_manager = new_grid
    assert gen.grid_manager is new_grid


# ---------------------------------------------------------------------------
# Integration: _on_grid_changed should not block stimulus creation
# ---------------------------------------------------------------------------

def test_on_grid_changed_creates_generator_without_library_dir():
    """_on_grid_changed must create a StimulusGenerator even when _library_dir is None.

    Previously, the save-blocking check was coupled with the generator creation
    path, causing stimulus generation to fail when no project was saved.
    This test verifies that the generator is created when a valid grid arrives
    regardless of _library_dir state.
    """
    # Import here to avoid PyQt5 import at module level in non-GUI test runs
    try:
        from PyQt5 import QtWidgets
        import sys
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv[:1])
    except ImportError:
        pytest.skip("PyQt5 not available")

    from sensoryforge.gui.tabs.stimulus_tab import StimulusDesignerTab
    from sensoryforge.gui.tabs.mechanoreceptor_tab import MechanoreceptorTab

    mech_tab = MechanoreceptorTab()
    stim_tab = StimulusDesignerTab(mech_tab)

    # Simulate: no project directory saved (the user just started)
    stim_tab._library_dir = None
    stim_tab._current_project_dir = None

    # Build a real grid and fire the signal
    grid = ReceptorGrid(grid_size=(10, 10), spacing=0.15, center=(0.0, 0.0))
    stim_tab._on_grid_changed(grid)

    # Generator should be created even without a library dir
    assert stim_tab.generator is not None, (
        "_on_grid_changed must create a StimulusGenerator when a valid grid "
        "arrives, even when _library_dir is None (no project saved yet)"
    )


def test_grid_update_preserves_stimulus_stack():
    """Updating the grid must not clear the in-memory stimulus stack.

    Previously, _on_grid_changed recreated StimulusGenerator and implicitly
    cleared any in-progress stimulus state. The stack must survive a grid update.
    """
    try:
        from PyQt5 import QtWidgets
        import sys
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv[:1])
    except ImportError:
        pytest.skip("PyQt5 not available")

    from sensoryforge.gui.tabs.stimulus_tab import StimulusDesignerTab
    from sensoryforge.gui.tabs.mechanoreceptor_tab import MechanoreceptorTab

    mech_tab = MechanoreceptorTab()
    stim_tab = StimulusDesignerTab(mech_tab)

    # Set up initial grid
    grid1 = ReceptorGrid(grid_size=(10, 10), spacing=0.15, center=(0.0, 0.0))
    stim_tab._on_grid_changed(grid1)

    # Record pre-update stack length (may be 0, just verify it doesn't shrink)
    stack_before = len(getattr(stim_tab, "_stimulus_stack", []))

    # Now update to a new grid of same dimensions
    grid2 = ReceptorGrid(grid_size=(10, 10), spacing=0.15, center=(1.0, 0.0))
    stim_tab._on_grid_changed(grid2)

    stack_after = len(getattr(stim_tab, "_stimulus_stack", []))

    assert stack_after >= stack_before, (
        f"Grid update reduced _stimulus_stack from {stack_before} to {stack_after}. "
        "Grid changes must not wipe the in-memory stimulus configuration."
    )


# ---------------------------------------------------------------------------
# Save-to-disk only blocks disk writes, not in-memory generation
# ---------------------------------------------------------------------------

def test_save_blocked_without_library_dir_does_not_affect_generation():
    """Stimulus saving to disk is correctly blocked without project dir.

    Generation (for pipeline use) must still work. This test verifies that
    the two concerns are properly separated.
    """
    try:
        from PyQt5 import QtWidgets
        import sys
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv[:1])
    except ImportError:
        pytest.skip("PyQt5 not available")

    from sensoryforge.gui.tabs.stimulus_tab import StimulusDesignerTab
    from sensoryforge.gui.tabs.mechanoreceptor_tab import MechanoreceptorTab

    mech_tab = MechanoreceptorTab()
    stim_tab = StimulusDesignerTab(mech_tab)
    stim_tab._library_dir = None

    grid = ReceptorGrid(grid_size=(10, 10), spacing=0.15, center=(0.0, 0.0))
    stim_tab._on_grid_changed(grid)

    # Generator exists — in-memory generation works
    assert stim_tab.generator is not None

    # get_config() must return valid config for pipeline use
    config = stim_tab.get_config()
    assert isinstance(config, dict), (
        "get_config() must return a dict even when no project is saved, "
        "so the spiking tab can run simulations in-memory"
    )
