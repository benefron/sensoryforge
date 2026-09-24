"""Tests for sensoryforge.gui.theme module.

Tests cover palette, QSS stylesheet, color functions, and pyqtgraph integration.
Marked with 'gui' for pytest-qt.
"""

import pytest
import unittest.mock

pytest.importorskip("PyQt5")

import pyqtgraph as pg  # noqa: E402
from PyQt5 import QtCore, QtGui  # noqa: E402

from sensoryforge.gui import theme  # noqa: E402

pytestmark = pytest.mark.gui


class TestPalette:
    """PALETTE dict coverage."""

    def test_palette_exists(self):
        """PALETTE is defined with expected keys."""
        assert hasattr(theme, "PALETTE")
        assert isinstance(theme.PALETTE, dict)
        expected_keys = {
            "bg_app",
            "bg_panel",
            "bg_panel_alt",
            "bg_sunken",
            "border",
            "border_strong",
            "text",
            "text_secondary",
            "text_disabled",
            "accent",
            "accent_hover",
            "accent_subtle",
            "success",
            "warning",
            "error",
        }
        assert set(theme.PALETTE.keys()) == expected_keys

    def test_palette_values_are_hex_colors(self):
        """All PALETTE values are #RRGGBB strings."""
        for key, value in theme.PALETTE.items():
            assert isinstance(value, str), f"{key} is not a string"
            assert value.startswith("#"), f"{key} does not start with #"
            assert len(value) == 7, f"{key} is not #RRGGBB format"


class TestPopulationColors:
    """POPULATION_COLORS list coverage."""

    def test_population_colors_exists(self):
        """POPULATION_COLORS list is defined."""
        assert hasattr(theme, "POPULATION_COLORS")
        assert isinstance(theme.POPULATION_COLORS, list)
        assert len(theme.POPULATION_COLORS) >= 2

    def test_first_two_colors_are_sa_ra(self):
        """First two colors match SA and RA definitions."""
        assert theme.POPULATION_COLORS[0] == "#2563EB"  # SA
        assert theme.POPULATION_COLORS[1] == "#E8630A"  # RA


class TestPopulationColor:
    """population_color() function."""

    def test_sa_returns_first_color(self):
        """population_color("SA") returns #2563EB (case-insensitive)."""
        result = theme.population_color(0, "SA")
        assert isinstance(result, QtGui.QColor)
        assert result.name() == "#2563eb"  # QColor.name() returns lowercase

    def test_ra_returns_second_color(self):
        """population_color("RA") returns #E8630A (case-insensitive)."""
        result = theme.population_color(1, "RA")
        assert isinstance(result, QtGui.QColor)
        assert result.name() == "#e8630a"

    def test_case_insensitive_neuron_type(self):
        """population_color is case-insensitive for neuron types."""
        sa_lower = theme.population_color(0, "sa")
        sa_upper = theme.population_color(0, "SA")
        sa_mixed = theme.population_color(0, "Sa")
        assert sa_lower.name() == sa_upper.name() == sa_mixed.name()

    def test_without_neuron_type_uses_index(self):
        """population_color(index) without neuron_type uses index cycling."""
        result = theme.population_color(0)
        expected = QtGui.QColor(theme.POPULATION_COLORS[0])
        assert result.name() == expected.name()

    def test_index_cycles_through_colors(self):
        """population_color cycles through POPULATION_COLORS via index."""
        for i in range(len(theme.POPULATION_COLORS)):
            result = theme.population_color(i)
            expected = QtGui.QColor(theme.POPULATION_COLORS[i])
            assert result.name() == expected.name()

    def test_large_index_wraps_around(self):
        """population_color(7) cycles when index exceeds list length."""
        index = 7
        expected_idx = index % len(theme.POPULATION_COLORS)
        result = theme.population_color(index)
        expected = QtGui.QColor(theme.POPULATION_COLORS[expected_idx])
        assert result.name() == expected.name()


class TestSpacing:
    """Spacing and dimension constants."""

    def test_spacing_constants_exist(self):
        """SPACING and height constants are defined."""
        assert hasattr(theme, "SPACING")
        assert hasattr(theme, "ROW_HEIGHT")
        assert hasattr(theme, "BUTTON_HEIGHT")
        assert hasattr(theme, "PRIMARY_HEIGHT")
        assert theme.SPACING == (4, 8, 12, 16, 24, 32)
        assert theme.ROW_HEIGHT == 28
        assert theme.BUTTON_HEIGHT == 30
        assert theme.PRIMARY_HEIGHT == 36


class TestStylesheet:
    """stylesheet() function."""

    def test_stylesheet_returns_string(self):
        """stylesheet() returns a non-empty string."""
        result = theme.stylesheet()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_stylesheet_contains_object_names(self):
        """stylesheet() contains all required object-name selectors."""
        ss = theme.stylesheet()
        required_selectors = [
            "QMainWindow",
            "#AppRoot",  # QWidget#AppRoot
            "QTabBar::tab",
            "QGroupBox",
            "#Panel",  # QFrame#Panel
            "#SectionTitle",  # QLabel#SectionTitle
            "#Primary",  # QPushButton#Primary
            "#Chip",  # QToolButton#Chip
            "QPushButton",
            "QSpinBox",
            "QDoubleSpinBox",
            "QComboBox",
            "QLineEdit",
            "QCheckBox",
            "QListWidget",
            "QProgressBar",
            "QScrollArea",
            "QSplitter",
            "QToolTip",
        ]
        for selector in required_selectors:
            assert selector in ss, f"Missing selector: {selector}"

    def test_stylesheet_contains_blue_accent_color(self):
        """stylesheet() uses the blue accent color #2563EB."""
        ss = theme.stylesheet()
        assert "#2563EB" in ss or "2563EB" in ss


class TestApply:
    """apply() function."""

    def test_apply_sets_fusion_style(self, qapp):
        """apply(qapp) sets the Fusion style."""
        theme.apply(qapp)
        # After setStyle("Fusion"), className should contain "Fusion"
        # (may be wrapped in QStyleSheetStyle after setStyleSheet)
        style_class = qapp.style().metaObject().className()
        assert "Fusion" in style_class or "StyleSheet" in style_class

    def test_apply_sets_stylesheet(self, qapp):
        """apply(qapp) sets the stylesheet."""
        theme.apply(qapp)
        ss = qapp.styleSheet()
        assert len(ss) > 0
        assert "#2563EB" in ss or "2563EB" in ss

    def test_apply_sets_pyqtgraph_config(self, qapp):
        """apply(qapp) calls pg.setConfigOptions with correct parameters."""
        with unittest.mock.patch("pyqtgraph.setConfigOptions") as mock_set:
            theme.apply(qapp)
            # Verify setConfigOptions was called with correct kwargs
            mock_set.assert_called_once()
            call_kwargs = mock_set.call_args[1]
            assert call_kwargs["antialias"] is True
            assert call_kwargs["background"] == "#FFFFFF"
            assert call_kwargs["foreground"] == "#5B6470"

    def test_apply_sets_font(self, qapp):
        """apply(qapp) sets the application font to 12px."""
        theme.apply(qapp)
        font = qapp.font()
        assert font.pixelSize() == 12


class TestPen:
    """pen() function."""

    def test_pen_returns_qpen(self):
        """pen() returns a QtGui.QPen."""
        result = theme.pen("#2563EB")
        assert isinstance(result, QtGui.QPen)

    def test_pen_with_string_color(self):
        """pen() accepts a hex color string."""
        result = theme.pen("#2563EB", width=2.0)
        assert isinstance(result, QtGui.QPen)
        assert result.width() == 2.0 or result.widthF() == 2.0

    def test_pen_with_qcolor(self):
        """pen() accepts a QtGui.QColor."""
        color = QtGui.QColor("#2563EB")
        result = theme.pen(color, width=1.5)
        assert isinstance(result, QtGui.QPen)

    def test_pen_has_round_caps(self):
        """pen() uses round caps."""
        result = theme.pen("#2563EB")
        # RoundCap is QtCore.Qt.RoundCap (enum value 1)
        assert result.capStyle() == QtCore.Qt.RoundCap

    def test_pen_default_width(self):
        """pen() uses LINE_WIDTH as default."""
        result = theme.pen("#2563EB")
        # LINE_WIDTH is 1.6
        assert abs(result.widthF() - theme.LINE_WIDTH) < 0.01


class TestGraphicsConstants:
    """Graphics and visualization constants."""

    def test_graphics_constants_exist(self):
        """Graphics constants are defined."""
        assert hasattr(theme, "AXIS_PEN")
        assert hasattr(theme, "GRID_ALPHA")
        assert hasattr(theme, "LINE_WIDTH")
        assert hasattr(theme, "RASTER_SIZE")
        assert hasattr(theme, "COLORMAP_NAME")

    def test_graphics_constants_values(self):
        """Graphics constants have expected values."""
        # AXIS_PEN is a QPen, checked in separate test
        assert theme.GRID_ALPHA == 0.12
        assert theme.LINE_WIDTH == 1.6
        assert theme.RASTER_SIZE == 4
        assert theme.COLORMAP_NAME == "viridis"

    def test_axis_pen_is_qpen(self):
        """AXIS_PEN is a QPen with border color and width 1.0."""
        assert isinstance(theme.AXIS_PEN, QtGui.QPen)
        # Color should be #B9C0CA (border color)
        assert theme.AXIS_PEN.color().name() == "#b9c0ca"
        # Width should be 1.0
        assert abs(theme.AXIS_PEN.widthF() - 1.0) < 0.01


class TestColormap:
    """colormap() function."""

    def test_colormap_returns_colormap(self):
        """colormap() returns a pg.ColorMap."""
        result = theme.colormap()
        assert isinstance(result, pg.ColorMap)

    def test_colormap_has_viridis_data(self):
        """colormap() loads viridis from matplotlib."""
        result = theme.colormap()
        # Check that it's a valid colormap with data
        assert result is not None
        lut = result.getLookupTable()
        assert len(lut) > 0


def test_apply_makes_numbers_read_like_the_yaml(qapp):
    """Spin boxes must show "1000.5", not a system-locale "1000,5"."""
    from PyQt5 import QtCore, QtWidgets

    from sensoryforge.gui import theme

    theme.apply(qapp)
    assert QtCore.QLocale().decimalPoint() == "."
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(0, 1e6)
    box.setDecimals(1)
    box.setValue(1000.5)
    assert box.text() == "1000.5"


def test_pens_are_cosmetic_so_width_is_pixels_not_data_units():
    from sensoryforge.gui import theme

    assert theme.pen("#2563EB").isCosmetic()
    assert theme.pen("#2563EB", width=3.0).isCosmetic()
    assert theme.AXIS_PEN.isCosmetic()
