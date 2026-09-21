"""The shared time cursor: a slider plus play/pause/speed.

One :class:`PlaybackBar` drives every panel on the Run & Results screen
through :attr:`PlaybackBar.frameChanged` ``(index: int)`` -- the screen
connects it to each panel's ``set_frame``/``set_cursor``.
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtWidgets

#: Playback speeds offered in the combo box, as a multiple of real time.
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)

#: How often the play timer advances, in ms (independent of the run's own dt).
_TIMER_INTERVAL_MS = 50


class PlaybackBar(QtWidgets.QWidget):
    """Slider + play/pause/speed row, emitting a frame index."""

    frameChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._n_frames = 0
        self._dt_ms = 1.0
        self._playing = False

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        layout.addWidget(self.play_button)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, stretch=1)

        self.time_label = QtWidgets.QLabel("0 ms")
        layout.addWidget(self.time_label)

        layout.addWidget(QtWidgets.QLabel("Speed"))
        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems([f"{s}x" for s in SPEEDS])
        self.speed_combo.setCurrentIndex(SPEEDS.index(1.0))
        layout.addWidget(self.speed_combo)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(_TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._advance)

    def set_range(self, n_frames: int, dt_ms: float) -> None:
        """Set how many frames there are and their spacing, resetting to 0.

        Args:
            n_frames: Number of frames the slider covers.
            dt_ms: Spacing between frames, in ms (for the play-speed timer
                and the time label).
        """
        self.stop()
        self._n_frames = max(n_frames, 0)
        self._dt_ms = dt_ms if dt_ms > 0 else 1.0
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(self._n_frames - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._update_label(0)
        self.frameChanged.emit(0)

    def current_index(self) -> int:
        """The frame index currently shown."""
        return self.slider.value()

    def set_index(self, index: int) -> None:
        """Move the slider to ``index`` (clamped), emitting once."""
        self.slider.setValue(max(0, min(index, self.slider.maximum())))

    def toggle_play(self) -> None:
        """Start playback if stopped, stop it if running."""
        self.stop() if self._playing else self.play()

    def play(self) -> None:
        """Start advancing the frame index on a timer."""
        if self._n_frames <= 1:
            return
        self._playing = True
        self.play_button.setText("Pause")
        self._timer.start()

    def stop(self) -> None:
        """Stop advancing; the slider stays where it is."""
        self._playing = False
        self.play_button.setText("Play")
        self._timer.stop()

    def _advance(self) -> None:
        speed = SPEEDS[self.speed_combo.currentIndex()]
        step = max(1, round(speed * _TIMER_INTERVAL_MS / self._dt_ms))
        next_index = self.slider.value() + step
        if next_index >= self.slider.maximum():
            self.slider.setValue(self.slider.maximum())
            self.stop()
        else:
            self.slider.setValue(next_index)

    def _on_slider_changed(self, index: int) -> None:
        self._update_label(index)
        self.frameChanged.emit(index)

    def _update_label(self, index: int) -> None:
        self.time_label.setText(f"{index * self._dt_ms:.1f} ms")
