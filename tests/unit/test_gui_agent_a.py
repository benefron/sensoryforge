"""Unit tests for GUI Agent A integration (CompositeGrid wiring)."""
import pytest


def test_mechanoreceptor_tab_import():
    """Verify MechanoreceptorTab is importable from gui.tabs."""
    from sensoryforge.gui.tabs import MechanoreceptorTab
    assert MechanoreceptorTab is not None


def test_composite_grid_import():
    """Verify CompositeGrid can be imported from expected location."""
    from sensoryforge.core.composite_grid import CompositeGrid
    assert CompositeGrid is not None


def test_composite_grid_basic_instantiation():
    """Verify CompositeGrid can be instantiated."""
    from sensoryforge.core.composite_grid import CompositeGrid
    grid = CompositeGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0), device="cpu")
    assert grid is not None
    assert grid.list_populations() == []


def test_composite_grid_add_population():
    """Verify populations can be added to CompositeGrid."""
    from sensoryforge.core.composite_grid import CompositeGrid
    grid = CompositeGrid(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0), device="cpu")
    grid.add_population(name="SA1", density=100.0, arrangement="grid")
    assert "SA1" in grid.list_populations()
    assert grid.get_population_count("SA1") > 0
