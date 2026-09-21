"""Stimulus-frame panel: the raw drive image, on mm axes, at the cursor time.

One ``pg.ImageItem`` on a themed image plot (``plot_factory.make_image_plot``);
:meth:`StimulusFramePanel.set_frame` sets the array shown, exactly the frame
at whatever time index the shared cursor is on -- nothing here decimates or
reinterprets the stimulus.
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtWidgets

from sensoryforge.gui.screens.results_data import ResultsView
from sensoryforge.gui.widgets import plot_factory


class StimulusFramePanel(QtWidgets.QWidget):
    """Shows ``view.stimulus[index]`` on the run's mm canvas."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._view: Optional[ResultsView] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot, self.image_item, self.colorbar = plot_factory.make_image_plot(
            "Stimulus", "x", "y", x_unit="mm", y_unit="mm", colorbar_label="mA"
        )
        layout.addWidget(self.plot)

    def set_view(self, view: Optional[ResultsView]) -> None:
        """Bind a new :class:`ResultsView`, sizing the image to its canvas."""
        self._view = view
        if view is None:
            self.image_item.clear()
            return
        x0, x1 = view.xlim
        y0, y1 = view.ylim
        self.image_item.setRect(x0, y0, x1 - x0, y1 - y0)
        if view.stimulus.numel() > 0:
            lo = float(view.stimulus.min())
            hi = float(view.stimulus.max())
            if hi <= lo:
                hi = lo + 1.0
            self.colorbar.setLevels((lo, hi))
        self.set_frame(0)

    def set_frame(self, index: int) -> None:
        """Show ``view.stimulus[index]`` exactly, with no resampling.

        Args:
            index: Frame index into ``view.time_ms``.

        Raises:
            ValueError: If no view is bound, or ``index`` is out of range.
        """
        if self._view is None:
            raise ValueError("no ResultsView bound; call set_view() first")
        frame = self._view.stimulus[index]
        self.image_item.setImage(frame.detach().cpu().numpy(), autoLevels=False)

    def teardown(self) -> None:
        """Release the plot's pyqtgraph resources."""
        plot_factory.teardown(self.plot)
