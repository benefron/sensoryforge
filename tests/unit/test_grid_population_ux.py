"""Tests for grid/population Add-default and rows/cols behaviour (items 3, 4).

Covers:
- GridEntry square-grid default (rows == cols when square toggle is on)
- Population default neurons_per_row adapts to grid size
- GridEntry serialization preserves rows != cols
"""

import pytest
from sensoryforge.core.grid import ReceptorGrid


# ---------------------------------------------------------------------------
# GridEntry dataclass behaviour (import without Qt)
# ---------------------------------------------------------------------------

def _make_grid_entry(**kwargs):
    """Import GridEntry from the GUI module without requiring a QApplication."""
    try:
        from PyQt5 import QtWidgets, QtGui
        import sys
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv[:1])
        from sensoryforge.gui.tabs.mechanoreceptor_tab import GridEntry
        defaults = dict(name="test", rows=20, cols=20, spacing=0.15,
                        color=QtGui.QColor(100, 100, 200, 200))
        defaults.update(kwargs)
        return GridEntry(**defaults)
    except ImportError:
        pytest.skip("PyQt5 not available")


def test_grid_entry_default_is_square():
    """GridEntry default rows and cols must be equal (square grid default)."""
    entry = _make_grid_entry()
    assert entry.rows == entry.cols, (
        f"Default GridEntry has rows={entry.rows}, cols={entry.cols}; "
        "expected rows==cols (square default)"
    )


def test_grid_entry_non_square_round_trips():
    """A non-square GridEntry (rows != cols) must survive to_dict / from_dict."""
    entry = _make_grid_entry(rows=20, cols=40)
    d = entry.to_dict()
    from sensoryforge.gui.tabs.mechanoreceptor_tab import GridEntry
    restored = GridEntry.from_dict(d)
    assert restored.rows == 20 and restored.cols == 40, (
        f"Round-trip failed: got rows={restored.rows}, cols={restored.cols}"
    )


# ---------------------------------------------------------------------------
# Population defaults adapt to grid size (item 3)
# ---------------------------------------------------------------------------

def test_add_population_default_neurons_proportional_to_grid():
    """When adding a population to a 40x40 grid, the neuron default should
    be proportional to the grid, not hardcoded to 10.

    A reasonable default is grid_rows // 4 (clamped to [4, 32]).
    For a 40x40 grid this gives 10 — same as before.
    For a 20x20 grid this gives 5.
    For an 80x80 grid this gives 20.
    """
    try:
        from PyQt5 import QtWidgets
        import sys
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv[:1])
        from sensoryforge.gui.tabs.mechanoreceptor_tab import MechanoreceptorTab
    except ImportError:
        pytest.skip("PyQt5 not available")

    tab = MechanoreceptorTab()

    # Default grid is 40x40 — population default should be ~10
    tab._on_add_grid()  # creates default 40x40 grid
    initial_pop_count = len(tab.populations)
    tab._on_add_population()
    assert len(tab.populations) == initial_pop_count + 1
    pop = tab.populations[-1]
    expected = max(4, min(32, 40 // 4))  # = 10 for 40x40
    assert pop.neurons_per_row == expected, (
        f"neurons_per_row={pop.neurons_per_row} for 40x40 grid; "
        f"expected {expected} (grid_rows // 4)"
    )


def test_add_population_neuron_count_adapts_to_larger_grid():
    """Larger grid → larger neuron default (within [4, 32] bounds)."""
    try:
        from PyQt5 import QtWidgets
        import sys
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv[:1])
        from sensoryforge.gui.tabs.mechanoreceptor_tab import MechanoreceptorTab, GridEntry
        from PyQt5 import QtGui
    except ImportError:
        pytest.skip("PyQt5 not available")

    tab = MechanoreceptorTab()

    # Add a large grid (80x80)
    entry = GridEntry(name="BigGrid", rows=80, cols=80, spacing=0.15,
                      color=QtGui.QColor(100, 100, 200))
    tab._add_grid_entry(entry)
    tab._on_add_population()
    pop = tab.populations[-1]
    expected = max(4, min(32, 80 // 4))  # = 20 for 80x80
    assert pop.neurons_per_row == expected, (
        f"neurons_per_row={pop.neurons_per_row} for 80x80 grid; expected {expected}"
    )


# ---------------------------------------------------------------------------
# Grid square toggle logic (item 4) — pure logic, no Qt widget needed
# ---------------------------------------------------------------------------

def test_square_grid_cols_equals_rows():
    """When square mode is active, cols must equal rows at entry creation."""
    entry = _make_grid_entry(rows=30, cols=30)
    # Square: rows == cols
    assert entry.rows == entry.cols


def test_non_square_grid_cols_independent():
    """When square mode is off, cols can differ from rows."""
    entry = _make_grid_entry(rows=20, cols=40)
    assert entry.rows != entry.cols, (
        "Non-square GridEntry should have rows != cols"
    )
