"""Light theme for SensoryForge GUI v2.

Defines the complete visual system: palette, QSS stylesheet, pyqtgraph
configuration, and utility functions for colors and pens.
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

# Palette: all semantic colors for the light theme
PALETTE = {
    "bg_app": "#F4F5F7",
    "bg_panel": "#FFFFFF",
    "bg_panel_alt": "#FAFAFB",
    "bg_sunken": "#EEF0F3",
    "border": "#D8DCE2",
    "border_strong": "#B9C0CA",
    "text": "#1E2328",
    "text_secondary": "#5B6470",
    "text_disabled": "#A3ABB5",
    "accent": "#2563EB",
    "accent_hover": "#1D4FD1",
    "accent_subtle": "#E8EEFE",
    "success": "#1E9E5A",
    "warning": "#C77A11",
    "error": "#D6394A",
}

# Population colors: first two are SA and RA mechanoreceptor types
POPULATION_COLORS = [
    "#2563EB",  # SA (blue accent)
    "#E8630A",  # RA (orange)
    "#7A5AF8",  # custom 1 (purple)
    "#0E9384",  # custom 2 (teal)
    "#C0369D",  # custom 3 (magenta)
    "#B45309",  # custom 4 (brown)
]

# Spacing and component dimensions (px)
SPACING = (4, 8, 12, 16, 24, 32)
ROW_HEIGHT = 28
BUTTON_HEIGHT = 30
PRIMARY_HEIGHT = 36

# Graphics and visualization constants (partial; pen() defined below)
GRID_ALPHA = 0.12
LINE_WIDTH = 1.6
RASTER_SIZE = 4
COLORMAP_NAME = "viridis"


def pen(color: str | QtGui.QColor, width: float = LINE_WIDTH) -> QtGui.QPen:
    """Create a styled pen with the given color and width.

    Args:
        color: Hex color string (e.g. "#2563EB") or QtGui.QColor
        width: Pen width in pixels (default LINE_WIDTH = 1.6)

    Returns:
        A QtGui.QPen with round caps and the specified color and width.

    Example:
        >>> p = pen("#2563EB", width=2.0)
        >>> assert p.widthF() == 2.0
    """
    qcolor = color if isinstance(color, QtGui.QColor) else QtGui.QColor(color)
    qpen = QtGui.QPen(qcolor)
    qpen.setWidthF(width)
    qpen.setCapStyle(QtCore.Qt.RoundCap)
    return qpen


# Pen for plot axes (defined after pen() function)
AXIS_PEN = pen("#B9C0CA", 1.0)


def population_color(index: int, neuron_type: str | None = None) -> QtGui.QColor:
    """Return a color for a population by neuron type or index.

    For known neuron types (SA, RA), returns the corresponding color
    regardless of index. Otherwise returns a color by cycling through
    POPULATION_COLORS at index % len(POPULATION_COLORS).

    Args:
        index: Population index (used if neuron_type is None or unknown)
        neuron_type: Neuron type name ("SA", "RA", etc.); case-insensitive

    Returns:
        A QtGui.QColor for the population.

    Example:
        >>> c1 = population_color(0, "SA")  # blue
        >>> c2 = population_color(5, "RA")  # orange
        >>> c3 = population_color(10)       # cycles through colors
    """
    if neuron_type is not None:
        nt = neuron_type.upper()
        if nt == "SA":
            return QtGui.QColor(POPULATION_COLORS[0])
        elif nt == "RA":
            return QtGui.QColor(POPULATION_COLORS[1])

    # Fall back to index cycling
    idx = index % len(POPULATION_COLORS)
    return QtGui.QColor(POPULATION_COLORS[idx])


def stylesheet() -> str:
    """Return the complete QSS stylesheet for the light theme.

    Defines styles for all major Qt widgets and custom object names,
    using the PALETTE colors and sizing constants.

    Returns:
        A QSS string ready to pass to QApplication.setStyleSheet().
    """
    return f"""
* {{ font-size: 12px; color: {PALETTE['text']}; }}
QMainWindow, QWidget#AppRoot {{ background: {PALETTE['bg_app']}; }}
QTabBar::tab {{ background: {PALETTE['bg_app']}; padding: 8px 16px; border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ border-bottom: 2px solid {PALETTE['accent']}; color: {PALETTE['accent']}; font-weight: 600; }}
QGroupBox, QFrame#Panel {{ background: {PALETTE['bg_panel']}; border: 1px solid {PALETTE['border']}; border-radius: 6px; margin-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {PALETTE['text_secondary']}; font-weight: 600; font-size: 10.5px; }}
QLabel#SectionTitle {{ color: {PALETTE['text_secondary']}; font-weight: 600; font-size: 10.5px; }}
QPushButton {{ background: {PALETTE['bg_panel']}; border: 1px solid {PALETTE['border_strong']}; border-radius: 5px; padding: 5px 12px; min-height: 18px; }}
QPushButton:hover {{ border-color: {PALETTE['accent']}; }}
QPushButton:disabled {{ color: {PALETTE['text_disabled']}; border-color: {PALETTE['border']}; }}
QPushButton#Primary {{ background: {PALETTE['accent']}; color: white; border: none; font-weight: 600; min-height: 24px; }}
QPushButton#Primary:hover {{ background: {PALETTE['accent_hover']}; }}
QPushButton#Primary:disabled {{ background: {PALETTE['text_disabled']}; }}
QToolButton#Chip {{ background: {PALETTE['bg_panel']}; border: 1px solid {PALETTE['border']}; border-radius: 6px; padding: 4px 10px; text-align: left; }}
QToolButton#Chip:checked {{ background: {PALETTE['accent_subtle']}; border-color: {PALETTE['accent']}; }}
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{ background: {PALETTE['bg_panel']}; border: 1px solid {PALETTE['border']}; border-radius: 4px; padding: 3px 6px; min-height: 22px; }}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus {{ border: 1px solid {PALETTE['accent']}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {PALETTE['border_strong']}; border-radius: 3px; background: {PALETTE['bg_panel']}; }}
QCheckBox::indicator:checked {{ background: {PALETTE['accent']}; border-color: {PALETTE['accent']}; }}
QListWidget {{ background: {PALETTE['bg_panel']}; border: 1px solid {PALETTE['border']}; border-radius: 6px; }}
QListWidget::item {{ padding: 6px 8px; }}
QListWidget::item:selected {{ background: {PALETTE['accent_subtle']}; color: {PALETTE['text']}; }}
QProgressBar {{ border: 1px solid {PALETTE['border']}; border-radius: 4px; background: {PALETTE['bg_sunken']}; text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {PALETTE['accent']}; border-radius: 3px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
QSplitter::handle {{ background: {PALETTE['border']}; }}
QToolTip {{ background: {PALETTE['text']}; color: #FFFFFF; border: none; padding: 4px 6px; }}
"""


def apply(app: QtWidgets.QApplication) -> None:
    """Apply the light theme to a QApplication.

    Sets the Fusion style, applies the stylesheet, configures pyqtgraph
    defaults, and sets the application font.

    Args:
        app: The QApplication to theme.

    Example:
        >>> app = QtWidgets.QApplication([])
        >>> apply(app)
    """
    # Numbers read as they do in the YAML config the user edits: "1000.0",
    # never "1000,0". Qt otherwise follows the system locale, so on a
    # comma-decimal system every spin box disagreed with the file format.
    QtCore.QLocale.setDefault(QtCore.QLocale.c())

    # Set Fusion style for a modern, cross-platform look
    app.setStyle("Fusion")

    # Apply the stylesheet
    app.setStyleSheet(stylesheet())

    # Configure pyqtgraph defaults
    pg.setConfigOptions(
        antialias=True,
        background=PALETTE["bg_panel"],
        foreground=PALETTE["text_secondary"],
    )

    # Set application font to 12 px
    font = app.font()
    font.setPixelSize(12)
    app.setFont(font)


def colormap() -> pg.ColorMap:
    """Return a pyqtgraph colormap for visualization.

    Loads the viridis colormap from matplotlib, falling back to grey
    if matplotlib is not available.

    Returns:
        A pg.ColorMap ready for use with ImageItem or other displays.

    Example:
        >>> cmap = colormap()
        >>> lut = cmap.getLookupTable()
    """
    try:
        return pg.colormap.get(COLORMAP_NAME, source="matplotlib")
    except Exception:
        return pg.colormap.get("grey")
