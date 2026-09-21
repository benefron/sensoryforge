"""Tests for :class:`sensoryforge.gui.widgets.grid_preview.GridPreview`.

Assertions check what is actually drawn (point counts against
``build_grid``'s real coordinates, brush arrays, bank weights) rather than
just that the widget constructs.
"""

import gc

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtGui  # noqa: E402

from sensoryforge.config.schema import GridConfig  # noqa: E402
from sensoryforge.core.innervation import build_population_bank  # noqa: E402
from sensoryforge.core.simulation_engine import build_grid  # noqa: E402
from sensoryforge.gui.widgets.grid_preview import GridPreview  # noqa: E402

pytestmark = pytest.mark.gui


def _bank_for_grid(grid_cfg, *, resolvable_distance_mm=0.6):
    grid = build_grid(grid_cfg, device="cpu")
    coords = grid.get_all_coordinates()
    return build_population_bank(
        receptor_coords=coords,
        innervation_method="template",
        resolvable_distance_mm=resolvable_distance_mm,
        device="cpu",
    )


class TestSetGrids:
    def test_grid_arrangement_count_matches_rows_cols(self, qtbot):
        preview = GridPreview()
        qtbot.addWidget(preview)
        grid_cfg = GridConfig(name="g", arrangement="grid", rows=20, cols=20)

        preview.set_grids([grid_cfg])

        assert preview.receptor_count() == 400
        assert len(preview._grid_scatters) == 1
        scatter = preview._grid_scatters[0]
        x, y = scatter.getData()
        assert x.shape[0] == 400
        assert y.shape[0] == 400

    def test_hex_arrangement_matches_build_grid_coordinates(self, qtbot):
        preview = GridPreview()
        qtbot.addWidget(preview)
        grid_cfg = GridConfig(
            name="g", arrangement="hex", rows=12, cols=12, spacing=0.2, seed=7
        )
        expected = build_grid(grid_cfg, device="cpu").get_all_coordinates().numpy()

        preview.set_grids([grid_cfg])

        assert preview.receptor_count() == expected.shape[0]
        scatter = preview._grid_scatters[0]
        x, y = scatter.getData()
        np.testing.assert_allclose(np.sort(x), np.sort(expected[:, 0]))
        np.testing.assert_allclose(np.sort(y), np.sort(expected[:, 1]))

    def test_poisson_arrangement_matches_build_grid_coordinates(self, qtbot):
        preview = GridPreview()
        qtbot.addWidget(preview)
        grid_cfg = GridConfig(
            name="g", arrangement="poisson", density=25.0, rows=10, cols=10, seed=3
        )
        expected = build_grid(grid_cfg, device="cpu").get_all_coordinates().numpy()

        preview.set_grids([grid_cfg])

        assert preview.receptor_count() == expected.shape[0]
        scatter = preview._grid_scatters[0]
        x, y = scatter.getData()
        assert x.shape[0] == expected.shape[0]
        np.testing.assert_allclose(np.sort(x), np.sort(expected[:, 0]))
        np.testing.assert_allclose(np.sort(y), np.sort(expected[:, 1]))

    def test_session_rows_change_redraws_with_new_count(self, qtbot):
        preview = GridPreview()
        qtbot.addWidget(preview)
        preview.set_grids([GridConfig(name="g", rows=10, cols=10)])
        assert preview.receptor_count() == 100

        # Simulate a session change to grids.0.rows: caller rebuilds the
        # GridConfig list and calls set_grids again.
        preview.set_grids([GridConfig(name="g", rows=15, cols=10)])

        assert preview.receptor_count() == 150
        assert len(preview._grid_scatters) == 1  # old scatter replaced, not stacked


class TestSetPopulation:
    def test_adds_scatter_with_n_neurons(self, qtbot):
        preview = GridPreview()
        qtbot.addWidget(preview)
        grid_cfg = GridConfig(name="g", rows=16, cols=16, spacing=0.15)
        preview.set_grids([grid_cfg])
        bank = _bank_for_grid(grid_cfg)

        preview.set_population("SA", bank, QtGui.QColor("#2563EB"))

        assert "SA" in preview._populations
        scatter = preview._populations["SA"]["scatter"]
        x, y = scatter.getData()
        assert x.shape[0] == bank.weights.shape[0]
        np.testing.assert_allclose(x, bank.neuron_centers[:, 0].numpy())
        np.testing.assert_allclose(y, bank.neuron_centers[:, 1].numpy())


class TestFootprint:
    def test_show_and_clear_footprint(self, qtbot):
        preview = GridPreview()
        qtbot.addWidget(preview)
        grid_cfg = GridConfig(name="g", rows=16, cols=16, spacing=0.15)
        preview.set_grids([grid_cfg])
        bank = _bank_for_grid(grid_cfg)
        preview.set_population("SA", bank, QtGui.QColor("#2563EB"))

        preview.show_rf_footprint("SA", 0)

        weights_row = bank.weights[0].numpy()
        expected_nonzero = int(np.count_nonzero(weights_row))
        assert preview._footprint_scatter is not None
        fx, _fy = preview._footprint_scatter.getData()
        assert fx.shape[0] == expected_nonzero

        # Brush differs from the plain grid scatter's uniform brush: the
        # footprint scatter carries per-point colors from bank.weights, so
        # more than one distinct color should appear whenever the row has
        # more than one distinct nonzero weight.
        spots = preview._footprint_scatter.points()
        brushes = {tuple(s.brush().color().getRgb()) for s in spots}
        distinct_weight_values = np.unique(weights_row[weights_row != 0.0])
        if distinct_weight_values.size > 1:
            assert len(brushes) > 1

        preview.clear_footprint()
        assert preview._footprint_scatter is None
        assert preview._footprint_ring is None
        assert preview._footprint_name is None

    def test_footprint_checked_against_bank_not_widget_bookkeeping(self, qtbot):
        """Footprint receptor positions must come from bank.receptor_coords,
        matched to bank.weights[i] != 0 -- not from the widget's own grid
        scatter data (which could, in principle, disagree)."""
        preview = GridPreview()
        qtbot.addWidget(preview)
        grid_cfg = GridConfig(name="g", rows=16, cols=16, spacing=0.15)
        preview.set_grids([grid_cfg])
        bank = _bank_for_grid(grid_cfg)
        preview.set_population("SA", bank, QtGui.QColor("#2563EB"))

        neuron_index = 3
        preview.show_rf_footprint("SA", neuron_index)

        weights_row = bank.weights[neuron_index].numpy()
        mask = weights_row != 0.0
        expected_coords = bank.receptor_coords.numpy()[mask]

        fx, fy = preview._footprint_scatter.getData()
        got = np.stack([fx, fy], axis=1)
        # Order-independent comparison of the point sets.
        got_sorted = got[np.lexsort((got[:, 1], got[:, 0]))]
        expected_sorted = expected_coords[
            np.lexsort((expected_coords[:, 1], expected_coords[:, 0]))
        ]
        np.testing.assert_allclose(got_sorted, expected_sorted)


class TestReceptorClick:
    def test_receptor_click_emits_index(self, qtbot):
        preview = GridPreview()
        qtbot.addWidget(preview)
        preview.set_grids([GridConfig(name="g", rows=5, cols=5)])
        scatter = preview._grid_scatters[0]

        received = []
        preview.receptorClicked.connect(received.append)

        points = scatter.pointsAt(scatter.pos() + scatter.points()[7].pos())
        assert points, "expected the click position to hit a point"
        scatter.sigClicked.emit(scatter, points, None)

        assert received == [7]


class TestTeardown:
    def test_close_and_gc_does_not_crash(self, qtbot):
        was_enabled = gc.isenabled()
        gc.enable()
        try:
            for _ in range(5):
                preview = GridPreview()
                qtbot.addWidget(preview)
                grid_cfg = GridConfig(name="g", rows=10, cols=10)
                preview.set_grids([grid_cfg])
                bank = _bank_for_grid(grid_cfg)
                preview.set_population("SA", bank, QtGui.QColor("#2563EB"))
                preview.show_rf_footprint("SA", 0)

                preview.show()
                qtbot.wait(1)
                preview.close()
                gc.collect()
        finally:
            if not was_enabled:
                gc.disable()
