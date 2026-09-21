"""The Stimulus screen's live preview: frame image, playhead, amplitude timeline.

:class:`StimulusPreview` is the only place the Stimulus screen renders a
stimulus, and it renders through one function,
:func:`sensoryforge.gui.execution.render.render_for_config` -- the same
renderer :class:`~sensoryforge.gui.execution.run_controller.RunController`
uses, so what this screen shows is what a run would actually produce.

Rendering happens on a ``QThread`` worker (never the GUI thread): building an
80x80-receptor stimulus is well over the ~100 ms budget a debounced keypress
should stay under. Edits are debounced 200 ms before a render is requested;
a generation counter drops a result that arrives after a newer render has
already been requested, so a slow render for stale parameters can never
clobber a fresh one.

Units: ms for time, mm for space, mA for the stimulus amplitude.
"""

from __future__ import annotations

import copy
import functools
from typing import Dict, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, sip

from sensoryforge.config.defaults import resolve_duration_ms
from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.gui import theme
from sensoryforge.gui.execution.render import RenderedStimulus, render_for_config
from sensoryforge.gui.session import Session
from sensoryforge.gui.widgets import plot_factory

#: A render preview never covers more than this many ms, however long the
#: configured run duration is -- an 80x80 grid for the full run duration is
#: not a preview cost.
PREVIEW_CAP_MS = 2000.0

#: Debounce interval between the last relevant edit and starting a render.
DEBOUNCE_MS = 200

#: Playback interval between frames when "Play" is pressed.
PLAYBACK_INTERVAL_MS = 60

#: Config path prefixes/paths whose change should trigger a re-render.
_RELEVANT_PREFIXES = ("stimulus.", "grids.")
_RELEVANT_PATHS = (
    "simulation.dt_ms",
    "simulation.duration_ms",
    "simulation.device",
)


class _RenderWorker(QtCore.QObject):
    """Renders one config on whatever thread owns it.

    Signals:
        finished(object): A :class:`RenderedStimulus`.
        failed(str): ``"TypeName: message"``.
    """

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        config: SensoryForgeConfig,
        *,
        duration_ms: float,
        dt_ms: float,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._duration_ms = duration_ms
        self._dt_ms = dt_ms

    @QtCore.pyqtSlot()
    def work(self) -> None:
        """Render. Emits exactly one of :attr:`finished`/:attr:`failed`."""
        try:
            rendered = render_for_config(
                self._config, duration_ms=self._duration_ms, dt_ms=self._dt_ms
            )
        except (ValueError, TypeError, RuntimeError) as exc:
            # A thread boundary: nothing may escape this slot. render_for_config
            # can raise ValueError (bad stimulus/grid config), TypeError (a
            # stimulus constructor rejecting a value outright, rare -- most are
            # caught and dropped inside render_for_config itself) or a torch
            # RuntimeError (e.g. an invalid device).
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(rendered)


class StimulusPreview(QtWidgets.QWidget):
    """Frame image (mm axes) + amplitude timeline + playback, for one Session.

    Signals:
        renderFinished(object): A :class:`RenderedStimulus` was drawn (tests
            wait on this instead of the debounce timer).
        renderFailed(str): A render raised; the message is also shown in the
            panel in the theme's error colour.

    Args:
        session: The session whose ``config`` is rendered.
        parent: Qt parent.
    """

    renderFinished = QtCore.pyqtSignal(object)
    renderFailed = QtCore.pyqtSignal(str)

    def __init__(
        self, session: Session, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._generation = 0
        # Every render still in flight, by generation. A new render does not
        # replace an older one's thread: dropping the last reference to a
        # running QThread aborts the process ("QThread: Destroyed while
        # thread is still running"). Each pair is released when it reports.
        self._inflight: Dict[int, Tuple[QtCore.QThread, "_RenderWorker"]] = {}
        self._rendered: Optional[RenderedStimulus] = None
        self._frame_index = 0
        self._frames = np.empty((0, 0, 0), dtype=np.float32)
        self._time_ms = np.empty((0,), dtype=np.float32)

        layout = QtWidgets.QVBoxLayout(self)

        self._capped_notice = QtWidgets.QLabel()
        self._capped_notice.setWordWrap(True)
        self._capped_notice.setStyleSheet(f"color: {theme.PALETTE['text_secondary']};")
        self._capped_notice.setVisible(False)
        layout.addWidget(self._capped_notice)

        self._warning_label = QtWidgets.QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet(f"color: {theme.PALETTE['warning']};")
        self._warning_label.setVisible(False)
        layout.addWidget(self._warning_label)

        self._error_label = QtWidgets.QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet(f"color: {theme.PALETTE['error']};")
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        self._image_plot, self.image_item, self._colorbar = (
            plot_factory.make_image_plot(
                "Stimulus frame", "x", "y", x_unit="mm", y_unit="mm"
            )
        )
        layout.addWidget(self._image_plot, 2)

        playback_row = QtWidgets.QHBoxLayout()
        self._play_button = QtWidgets.QPushButton("Play")
        self._play_button.setCheckable(True)
        self._play_button.toggled.connect(self._on_play_toggled)
        playback_row.addWidget(self._play_button)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        playback_row.addWidget(self._slider, 1)

        self._time_label = QtWidgets.QLabel("0 ms")
        playback_row.addWidget(self._time_label)
        layout.addLayout(playback_row)

        self._amplitude_plot = plot_factory.make_plot(
            "Amplitude (max over space)", "Time", "Amplitude", x_unit="ms", y_unit="mA"
        )
        self._amplitude_curve = self._amplitude_plot.plot(
            pen=theme.pen(theme.PALETTE["accent"])
        )
        self._playhead = pg.InfiniteLine(
            pos=0, angle=90, pen=theme.pen(theme.PALETTE["text_secondary"])
        )
        self._amplitude_plot.addItem(self._playhead)
        layout.addWidget(self._amplitude_plot, 1)

        self._playback_timer = QtCore.QTimer(self)
        self._playback_timer.setInterval(PLAYBACK_INTERVAL_MS)
        self._playback_timer.timeout.connect(self._advance_frame)

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self.render_now)

        session.configChanged.connect(self._on_config_changed)
        session.configReplaced.connect(self._on_config_replaced)

    # ---------------------------------------------------------------- render

    def _on_config_changed(self, path: str) -> None:
        if path.startswith(_RELEVANT_PREFIXES) or path in _RELEVANT_PATHS:
            self._debounce.start(DEBOUNCE_MS)

    def _on_config_replaced(self) -> None:
        self._debounce.stop()
        self.render_now()

    def render_now(self) -> None:
        """Start a render immediately, bypassing the debounce timer.

        Cancels any in-flight render's relevance (a stale result is dropped
        by generation, never applied) and starts a fresh worker thread.
        """
        self._debounce.stop()
        config = copy.deepcopy(self._session.config)
        run_duration = resolve_duration_ms(config.simulation.duration_ms)
        duration = min(run_duration, PREVIEW_CAP_MS)
        capped = run_duration > PREVIEW_CAP_MS
        self._capped_notice.setText(
            f"Preview capped at {PREVIEW_CAP_MS:.0f} ms of the configured "
            f"{run_duration:.0f} ms run."
            if capped
            else ""
        )
        self._capped_notice.setVisible(capped)

        self._generation += 1
        generation = self._generation

        worker = _RenderWorker(
            config, duration_ms=duration, dt_ms=float(config.simulation.dt_ms)
        )
        thread = QtCore.QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.work)
        worker.finished.connect(functools.partial(self._on_render_finished, generation))
        worker.failed.connect(functools.partial(self._on_render_failed, generation))

        self._inflight[generation] = (thread, worker)
        thread.start()

    def _on_render_finished(self, generation: int, rendered: RenderedStimulus) -> None:
        self._teardown_thread(generation)
        if sip.isdeleted(self):
            # The preview was torn down (e.g. its screen closed) while this
            # render was in flight on its own thread; the queued signal still
            # delivers after teardown, and there is nothing left to update.
            return
        if generation != self._generation:
            return  # a newer render was requested meanwhile; drop this one
        self._error_label.setVisible(False)
        self._rendered = rendered
        if rendered.warning:
            self._warning_label.setText(rendered.warning)
            self._warning_label.setVisible(True)
        else:
            self._warning_label.setVisible(False)
        self._apply_rendered(rendered)
        self.renderFinished.emit(rendered)

    def _on_render_failed(self, generation: int, message: str) -> None:
        self._teardown_thread(generation)
        if sip.isdeleted(self):
            return
        if generation != self._generation:
            return
        self._error_label.setText(message)
        self._error_label.setVisible(True)
        self.renderFailed.emit(message)

    def _teardown_thread(self, generation: int) -> None:
        pair = self._inflight.pop(generation, None)
        if pair is None:
            return
        thread, worker = pair
        thread.quit()
        thread.wait()
        worker.deleteLater()
        thread.deleteLater()

    def wait_for_renders(self) -> None:
        """Block until every in-flight render has finished and been released.

        For shutdown and tests; the event loop is not run, so no result is
        applied to the widgets.
        """
        for generation in list(self._inflight):
            self._teardown_thread(generation)

    # ----------------------------------------------------------------- draw

    def _apply_rendered(self, rendered: RenderedStimulus) -> None:
        stimulus = rendered.stimulus
        # [1, T, H, W] (single-channel) or [1, T, C, H, W]; the preview shows
        # the first channel of a multi-channel stimulus.
        if stimulus.dim() == 5:
            frames = stimulus[0, :, 0]
        else:
            frames = stimulus[0]
        self._frames = frames.detach().cpu().numpy()
        self._time_ms = np.asarray(rendered.time_ms)

        n_frames = self._frames.shape[0]
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(n_frames - 1, 0))
        self._slider.setValue(min(self._frame_index, max(n_frames - 1, 0)))
        self._slider.blockSignals(False)
        self._frame_index = self._slider.value()

        canvas = rendered.canvas
        if canvas is not None:
            xlim, ylim = canvas.xlim, canvas.ylim
        else:
            xlim, ylim = (-1.0, 1.0), (-1.0, 1.0)
        n_x, n_y = self._frames.shape[1], self._frames.shape[2]
        dx = (xlim[1] - xlim[0]) / max(n_x, 1)
        dy = (ylim[1] - ylim[0]) / max(n_y, 1)
        transform = pg.QtGui.QTransform()
        transform.translate(xlim[0], ylim[0])
        transform.scale(dx, dy)
        self.image_item.setTransform(transform)
        self._image_plot.setRange(xRange=list(xlim), yRange=list(ylim), padding=0.05)

        amplitude = self._frames.reshape(n_frames, -1).max(axis=1)
        self._amplitude_curve.setData(self._time_ms, amplitude)
        frame_min, frame_max = float(self._frames.min()), float(self._frames.max())
        self._colorbar.setLevels(
            (frame_min, frame_max if frame_max != frame_min else frame_min + 1.0)
        )

        self._render_frame(self._frame_index)

    def _render_frame(self, index: int) -> None:
        if self._rendered is None or not len(self._frames):
            return
        index = max(0, min(index, self._frames.shape[0] - 1))
        self.image_item.setImage(self._frames[index], autoLevels=False)
        t = float(self._time_ms[index]) if index < len(self._time_ms) else 0.0
        self._time_label.setText(f"{t:.1f} ms")
        self._playhead.setPos(t)

    # ------------------------------------------------------------ playback

    def _on_slider_changed(self, value: int) -> None:
        self._frame_index = value
        self._render_frame(value)

    def _advance_frame(self) -> None:
        if self._rendered is None:
            return
        n = self._frames.shape[0]
        if n <= 1:
            self._play_button.setChecked(False)
            return
        self._slider.setValue((self._slider.value() + 1) % n)

    def _on_play_toggled(self, checked: bool) -> None:
        self._play_button.setText("Pause" if checked else "Play")
        if checked:
            self._playback_timer.start()
        else:
            self._playback_timer.stop()

    # -------------------------------------------------------------- public

    @property
    def rendered(self) -> Optional[RenderedStimulus]:
        """The last successfully rendered stimulus, or ``None``."""
        return self._rendered

    def frame_index(self) -> int:
        """The slider's current frame index."""
        return self._frame_index

    def set_frame_index(self, index: int) -> None:
        """Move the slider (and playhead) to ``index``."""
        self._slider.setValue(index)
