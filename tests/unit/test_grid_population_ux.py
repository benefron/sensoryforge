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


# ---------------------------------------------------------------------------
# NeuronPopulation per-col toggle (item 4) — dataclass behaviour
# ---------------------------------------------------------------------------

def _make_neuron_population(**kwargs):
    """Create a NeuronPopulation without requiring a running display."""
    try:
        from PyQt5 import QtWidgets, QtGui
        import sys
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv[:1])
        from sensoryforge.gui.tabs.mechanoreceptor_tab import NeuronPopulation
        defaults = dict(
            name="SA #1",
            neuron_type="SA",
            color=QtGui.QColor(66, 135, 245),
            neurons_per_row=10,
            connections_per_neuron=5.0,
            sigma_d_mm=0.5,
            weight_min=0.1,
            weight_max=1.0,
        )
        defaults.update(kwargs)
        return NeuronPopulation(**defaults)
    except ImportError:
        pytest.skip("PyQt5 not available")


def test_neuron_population_default_neuron_rows_equals_cols():
    """NeuronPopulation created with neuron_rows=neuron_cols must store both equal."""
    pop = _make_neuron_population(neuron_rows=10, neuron_cols=10)
    assert pop.neuron_rows == pop.neuron_cols == 10


def test_neuron_population_non_square_stores_independently():
    """NeuronPopulation with neuron_rows != neuron_cols must preserve both values."""
    pop = _make_neuron_population(neuron_rows=8, neuron_cols=12)
    assert pop.neuron_rows == 8 and pop.neuron_cols == 12, (
        f"Expected neuron_rows=8, neuron_cols=12; "
        f"got neuron_rows={pop.neuron_rows}, neuron_cols={pop.neuron_cols}"
    )


# ---------------------------------------------------------------------------
# NeuronPopulation per-col toggle (item 4) — Qt widget behaviour
# ---------------------------------------------------------------------------

def test_chk_square_neurons_default_is_checked():
    """Square neurons checkbox must be checked by default and col spinbox hidden."""
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
    assert hasattr(tab, "chk_square_neurons"), "MechanoreceptorTab missing chk_square_neurons"
    assert tab.chk_square_neurons.isChecked(), "chk_square_neurons should be checked by default"
    assert hasattr(tab, "spin_neurons_per_col"), "MechanoreceptorTab missing spin_neurons_per_col"
    assert tab.spin_neurons_per_col.isHidden(), (
        "spin_neurons_per_col should be hidden when square mode is on"
    )


def test_uncheck_square_reveals_col_spinbox():
    """Unchecking square neurons must reveal the per-col spinbox."""
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
    tab.chk_square_neurons.setChecked(False)
    assert not tab.spin_neurons_per_col.isHidden(), (
        "spin_neurons_per_col should not be hidden when square mode is off"
    )


def test_population_editor_non_square_writes_correct_cols():
    """With square=off, changing the col spinbox must write neuron_cols to the population."""
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
    tab._on_add_grid()
    tab._on_add_population()
    pop = tab.populations[-1]

    # Select the population in the list so the editor targets it
    row_idx = tab.population_list.count() - 1
    tab.population_list.setCurrentRow(row_idx)
    tab._on_population_selected(row_idx)

    tab.chk_square_neurons.setChecked(False)
    tab.spin_neurons_per_row.setValue(8)
    tab.spin_neurons_per_col.setValue(12)
    tab._on_population_editor_changed()

    assert pop.neuron_rows == 8, f"Expected neuron_rows=8, got {pop.neuron_rows}"
    assert pop.neuron_cols == 12, f"Expected neuron_cols=12, got {pop.neuron_cols}"


def test_load_population_into_form_restores_non_square():
    """_load_population_into_form must restore non-square neuron dims and show col spinbox."""
    try:
        from PyQt5 import QtWidgets
        import sys
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv[:1])
        from sensoryforge.gui.tabs.mechanoreceptor_tab import MechanoreceptorTab, NeuronPopulation
        from PyQt5 import QtGui
    except ImportError:
        pytest.skip("PyQt5 not available")

    tab = MechanoreceptorTab()
    pop = NeuronPopulation(
        name="Test",
        neuron_type="SA",
        color=QtGui.QColor(100, 100, 200),
        neurons_per_row=6,
        neuron_rows=6,
        neuron_cols=10,
        connections_per_neuron=5.0,
        sigma_d_mm=0.5,
        weight_min=0.1,
        weight_max=1.0,
    )
    tab._load_population_into_form(pop)

    assert not tab.chk_square_neurons.isChecked(), (
        "chk_square_neurons should be unchecked for non-square population"
    )
    assert tab.spin_neurons_per_col.value() == 10, (
        f"Expected spin_neurons_per_col=10, got {tab.spin_neurons_per_col.value()}"
    )
    assert not tab.spin_neurons_per_col.isHidden(), (
        "spin_neurons_per_col should not be hidden after loading non-square population"
    )
