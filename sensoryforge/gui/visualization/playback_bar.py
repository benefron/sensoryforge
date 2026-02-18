"""Playback controller bar widget.

A compact, self-contained transport bar that emits ``seek_requested(t_idx)``
signals as the user scrubs or the timer fires.  The tab connects this signal
to all visible panels.
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets


_SPEED_OPTIONS = [0.25, 0.5, 1.0, 2.0, 4.0]
_SPEED_LABELS  = ["¼×", "½×", "1×", "2×", "4×"]
_DEFAULT_SPEED_IDX = 2  # 1×
_BASE_INTERVAL_MS = 50  # timer interval at 1× speed → ~20 fps


class PlaybackController(QtWidgets.QWidget):
    """Bottom transport bar for the Visualization tab.

    Signals:
        seek_requested(int): emitted with the new time index whenever the
            playhead moves (scrub, step, or timer tick).
        speed_changed(float): emitted when playback speed changes.
    """

    seek_requested = QtCore.pyqtSignal(int)
    speed_changed  = QtCore.pyqtSignal(float)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._n_steps: int = 0
        self._t_idx:   int = 0
        self._playing: bool = False
        self._speed:   float = 1.0

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._apply_style()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_length(self, n_steps: int, dt_ms: float) -> None:
        """Configure for a new simulation with ``n_steps`` frames."""
        self._n_steps = max(1, n_steps)
        self._dt_ms   = dt_ms
        self._t_idx   = 0
        self._slider.blockSignals(True)
        self._slider.setRange(0, self._n_steps - 1)
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._update_time_label(0)
        self._update_end_label()
        self.stop()

    def seek(self, t_idx: int) -> None:
        """Programmatically move playhead (does not emit seek_requested)."""
        self._t_idx = max(0, min(t_idx, self._n_steps - 1))
        self._slider.blockSignals(True)
        self._slider.setValue(self._t_idx)
        self._slider.blockSignals(False)
        self._update_time_label(self._t_idx)

    def stop(self) -> None:
        self._timer.stop()
        self._playing = False
        self._play_btn.setText("▶")
        self._play_btn.setToolTip("Play (Space)")

    @property
    def current_step(self) -> int:
        return self._t_idx

    @property
    def speed(self) -> float:
        return self._speed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # ---- Step back
        self._step_back_btn = _icon_btn("◀◀", "Step back one frame")
        self._step_back_btn.clicked.connect(self._on_step_back)
        layout.addWidget(self._step_back_btn)

        # ---- Play / Pause
        self._play_btn = _icon_btn("▶", "Play (Space)")
        self._play_btn.setMinimumWidth(36)
        self._play_btn.clicked.connect(self._on_play_pause)
        layout.addWidget(self._play_btn)

        # ---- Step forward
        self._step_fwd_btn = _icon_btn("▶▶", "Step forward one frame")
        self._step_fwd_btn.clicked.connect(self._on_step_fwd)
        layout.addWidget(self._step_fwd_btn)

        # ---- Separator
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(sep)

        # ---- Time label current
        self._time_label = QtWidgets.QLabel("t = 0.0 ms")
        self._time_label.setFixedWidth(100)
        self._time_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._time_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self._time_label)

        # ---- Slider
        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(10)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, stretch=1)

        # ---- End label
        self._end_label = QtWidgets.QLabel("0.0 ms")
        self._end_label.setFixedWidth(70)
        self._end_label.setStyleSheet("font-size: 11px; color: #666;")
        layout.addWidget(self._end_label)

        # ---- Separator
        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.VLine)
        sep2.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(sep2)

        # ---- Speed selector
        spd_lbl = QtWidgets.QLabel("Speed:")
        spd_lbl.setStyleSheet("font-size: 11px; color: #555;")
        layout.addWidget(spd_lbl)

        self._speed_cmb = QtWidgets.QComboBox()
        self._speed_cmb.addItems(_SPEED_LABELS)
        self._speed_cmb.setCurrentIndex(_DEFAULT_SPEED_IDX)
        self._speed_cmb.setFixedWidth(52)
        self._speed_cmb.currentIndexChanged.connect(self._on_speed_changed)
        layout.addWidget(self._speed_cmb)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            PlaybackController {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ececec, stop:1 #dedede
                );
                border-top: 1px solid #bbbbbb;
            }
            QPushButton {
                padding: 3px 8px;
                border: 1px solid #aaaaaa;
                border-radius: 3px;
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f5f5f5, stop:1 #e8e8e8
                );
                font-size: 11px;
            }
            QPushButton:hover  { background: #eeeeee; border-color: #888; }
            QPushButton:pressed{ background: #d8d8d8; }
        """)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_play_pause(self) -> None:
        if self._n_steps == 0:
            return
        if self._playing:
            self.stop()
        else:
            self._playing = True
            self._play_btn.setText("⏸")
            self._play_btn.setToolTip("Pause (Space)")
            interval = max(1, int(_BASE_INTERVAL_MS / self._speed))
            self._timer.start(interval)

    def _on_step_back(self) -> None:
        self._move_to(self._t_idx - 1)

    def _on_step_fwd(self) -> None:
        self._move_to(self._t_idx + 1)

    def _on_slider_changed(self, value: int) -> None:
        if value != self._t_idx:
            self._t_idx = value
            self._update_time_label(value)
            self.seek_requested.emit(value)

    def _on_speed_changed(self, idx: int) -> None:
        self._speed = _SPEED_OPTIONS[idx]
        if self._playing:
            interval = max(1, int(_BASE_INTERVAL_MS / self._speed))
            self._timer.setInterval(interval)
        self.speed_changed.emit(self._speed)

    def _on_tick(self) -> None:
        nxt = self._t_idx + 1
        if nxt >= self._n_steps:
            nxt = 0  # loop
        self._move_to(nxt)

    def _move_to(self, t_idx: int) -> None:
        self._t_idx = max(0, min(t_idx, self._n_steps - 1))
        self._slider.blockSignals(True)
        self._slider.setValue(self._t_idx)
        self._slider.blockSignals(False)
        self._update_time_label(self._t_idx)
        self.seek_requested.emit(self._t_idx)

    def _update_time_label(self, t_idx: int) -> None:
        if hasattr(self, "_dt_ms"):
            ms = t_idx * self._dt_ms
            self._time_label.setText(f"t = {ms:7.1f} ms")
        else:
            self._time_label.setText(f"t = {t_idx}")

    def _update_end_label(self) -> None:
        if hasattr(self, "_dt_ms"):
            total_ms = (self._n_steps - 1) * self._dt_ms
            self._end_label.setText(f"{total_ms:.1f} ms")
        else:
            self._end_label.setText(f"{self._n_steps - 1}")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Space:
            self._on_play_pause()
        elif event.key() == QtCore.Qt.Key_Left:
            self._on_step_back()
        elif event.key() == QtCore.Qt.Key_Right:
            self._on_step_fwd()
        else:
            super().keyPressEvent(event)


def _icon_btn(text: str, tooltip: str) -> QtWidgets.QPushButton:
    btn = QtWidgets.QPushButton(text)
    btn.setToolTip(tooltip)
    btn.setFixedHeight(24)
    return btn
