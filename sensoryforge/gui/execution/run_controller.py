"""One interactive run, off the GUI thread, cancellable, bundled.

Every run the GUI v2 shell performs goes through :class:`RunController`: it
renders the session's stimulus, deep-copies the config so later edits cannot
change what the results mean, and hands both to a :class:`RunWorker` living on
a ``QThread``. The worker constructs
:class:`~sensoryforge.core.simulation_engine.SimulationEngine` **on that
thread** (an MPS engine must be built and used on one thread) and calls
``SimulationEngine.run(...)`` -- the same call ``sensoryforge run`` makes, so
the GUI and the CLI cannot drift apart (the guard is
``tests/integration/test_gui_engine_equality.py``).

Cancel is cooperative, never ``QThread.terminate()``: a
:class:`threading.Event` is polled in the engine's ``progress_cb``, which
fires once per population immediately *before* that population's
filter/neuron pass. Setting it raises :class:`RunCancelled` inside the engine
loop, so the remaining populations never run and, because the bundle is
written after the loop, no half-written bundle is left behind.

Units: ms at this API, mA for the stimulus, mm for space.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PyQt5 import QtCore

from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.gui.execution.render import RenderedStimulus, render_for_config
from sensoryforge.gui.session import RunResult, Session

LOGGER = logging.getLogger(__name__)

#: A quick run is capped at this many ms, however long the full run would be.
QUICK_DURATION_MS = 100.0


class RunCancelled(Exception):
    """Raised inside the engine loop when the user asked to stop.

    Not an error: :class:`RunWorker` turns it into a ``cancelled`` signal, and
    it is the only exception that does not become ``failed``.
    """


class RunWorker(QtCore.QObject):
    """Builds and runs a :class:`SimulationEngine` on whatever thread owns it.

    Signals:
        progress(int, int, str): ``(index, n_populations, population_name)``,
            re-emitted from the engine's ``progress_cb`` before each
            population runs.
        finished(object): The raw results dict plus timing, as
            ``{"results": dict, "elapsed_s": float}``.
        failed(str): ``"TypeName: message"`` for anything that went wrong; the
            traceback goes to the module logger.
        cancelled(): The run stopped early because :meth:`cancel` was asked
            for. No results, no bundle.

    Args:
        config: The config snapshot to run. Owned by the worker; nothing else
            may mutate it while the run is in flight.
        stimulus: ``[1, time, H, W]`` (or ``[1, time, C, H, W]``) in mA.
        device: Torch device string (``"cpu"``, ``"mps"``, ``"cuda"``).
        bundle_dir: Where to write the run's bundle, or ``None`` not to.
        cancel_event: Set from the GUI thread to stop the run.
        parent: Qt parent. Normally ``None``: the worker is moved to a thread
            and must not be owned by an object on another one.
    """

    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(
        self,
        config: SensoryForgeConfig,
        stimulus: torch.Tensor,
        *,
        device: str,
        bundle_dir: Optional[Path],
        cancel_event: threading.Event,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._stimulus = stimulus
        self._device = device
        self._bundle_dir = bundle_dir
        self._cancel = cancel_event

    # ------------------------------------------------------------------ work

    def _progress_cb(self, index: int, total: int, name: str) -> None:
        """The engine's per-population hook: the cancel poll and the progress.

        Args:
            index: Zero-based population index about to run.
            total: Number of populations in the run.
            name: That population's name.

        Raises:
            RunCancelled: If cancellation was requested. Raised *before* the
                progress signal, so a cancelled population is never announced
                as started.
        """
        if self._cancel.is_set():
            raise RunCancelled(
                f"run cancelled before population {index + 1}/{total} ({name!r})"
            )
        self.progress.emit(index, total, name)

    @QtCore.pyqtSlot()
    def work(self) -> None:
        """Run the simulation. Invoked by the thread's ``started`` signal.

        Emits exactly one of :attr:`finished`, :attr:`cancelled` or
        :attr:`failed`, whatever happens.
        """
        started = time.perf_counter()
        try:
            if self._cancel.is_set():
                raise RunCancelled("run cancelled before it started")
            # Built here, on the worker thread: an MPS engine's modules and
            # the tensors they allocate must belong to the thread that runs
            # them, and building an 80x80 grid's receptive fields is itself
            # slow enough to freeze the window if done on the GUI thread.
            engine = SimulationEngine(self._config, device=torch.device(self._device))
            results = engine.run(
                self._stimulus,
                return_intermediates=True,
                bundle_dir=str(self._bundle_dir) if self._bundle_dir else None,
                stimulus_config=self._config.stimulus.to_dict(),
                seed=self._config.simulation.seed,
                progress_cb=self._progress_cb,
            )
        except RunCancelled as exc:
            LOGGER.info("%s", exc)
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - thread boundary, see below
            # A worker thread is a boundary: an exception that escapes a Qt
            # slot aborts the process, so everything the engine can raise
            # (ValueError, KeyError, RuntimeError, torch's own errors, a
            # plugin component's anything) has to be turned into a signal
            # here. The traceback is logged; the message reaches the user.
            LOGGER.exception("simulation run failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(
                {"results": results, "elapsed_s": time.perf_counter() - started}
            )


class RunController(QtCore.QObject):
    """Runs the session's config on a worker thread, with progress and cancel.

    Signals:
        started(): A run has begun (the worker thread is up).
        progress(int, int, str): ``(index, n_populations, population_name)``.
        finished(object): The :class:`~sensoryforge.gui.session.RunResult`,
            already published to the session.
        failed(str): The run raised; the message names the exception.
        cancelled(): The run stopped early; the session keeps its previous
            results and no bundle was written.

    Args:
        session: The experiment to run. Its config is snapshotted per run and
            its ``last_results`` is where a finished run lands.
        parent: Qt parent.

    Example:
        >>> controller = RunController(session)          # doctest: +SKIP
        >>> controller.run(duration_ms=200.0)            # doctest: +SKIP
    """

    started = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, session: Session, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._session = session
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[RunWorker] = None
        self._cancel = threading.Event()
        self._snapshot: Optional[SensoryForgeConfig] = None
        self._rendered: Optional[RenderedStimulus] = None
        self._bundle_dir: Optional[Path] = None
        self._duration_ms: float = 0.0
        self._quick: bool = False
        self._started_at: datetime = datetime.now()

    # -------------------------------------------------------------- lifecycle

    @property
    def running(self) -> bool:
        """Whether a run is in flight."""
        return self._thread is not None

    @property
    def bundle_dir(self) -> Optional[Path]:
        """Where the run in flight (or the last one) writes its bundle."""
        return self._bundle_dir

    @property
    def last_warning(self) -> Optional[str]:
        """The stimulus render warning of the run in flight, if any (F-061)."""
        return self._rendered.warning if self._rendered is not None else None

    # ------------------------------------------------------------------- run

    def run(
        self,
        *,
        duration_ms: float,
        bundle: bool = True,
        quick_population: Optional[str] = None,
    ) -> None:
        """Start a run of the session's config.

        The config is deep-copied first, so editing a spinbox while the run is
        in flight changes the next run, not this one. The stimulus is rendered
        here, on the GUI thread: it is cheap, needs no torch device, and a
        bad stimulus should raise before a thread is ever started.

        Args:
            duration_ms: Simulated duration in ms.
            bundle: Whether to write a bundle. Only honoured when a project is
                open -- there is nowhere else to put it.
            quick_population: Name of the single population to run as a quick
                preview. The snapshot keeps only that population, the duration
                is capped at :data:`QUICK_DURATION_MS`, and no bundle is
                written whatever ``bundle`` says.

        Raises:
            RuntimeError: If a run is already in flight. Cancel it first.
            ValueError: If ``duration_ms`` is not positive, if
                ``quick_population`` names no population in the config, or if
                the stimulus cannot be rendered (no grid, unknown type) --
                nothing is started in any of these cases.
        """
        if self.running:
            raise RuntimeError(
                "a run is already in progress; cancel it before starting another"
            )
        if duration_ms <= 0:
            raise ValueError(f"duration_ms must be positive, got {duration_ms}")

        snapshot = copy.deepcopy(self._session.config)
        duration = float(duration_ms)
        if quick_population is not None:
            kept = [p for p in snapshot.populations if p.name == quick_population]
            if not kept:
                known = [p.name for p in self._session.config.populations]
                raise ValueError(
                    f"no population named {quick_population!r} to quick-run; "
                    f"the config has {known}"
                )
            snapshot.populations = kept
            duration = min(duration, QUICK_DURATION_MS)
            bundle = False
        snapshot.simulation.duration_ms = duration

        rendered = render_for_config(snapshot, duration_ms=duration)

        bundle_dir: Optional[Path] = None
        if bundle and self._session.project is not None:
            bundle_dir = self._session.project.new_run_dir(
                str(snapshot.metadata.get("name", "run"))
            )

        self._snapshot = snapshot
        self._rendered = rendered
        self._bundle_dir = bundle_dir
        self._duration_ms = duration
        self._quick = quick_population is not None
        self._started_at = datetime.now()
        self._cancel = threading.Event()

        worker = RunWorker(
            snapshot,
            rendered.stimulus,
            device=snapshot.simulation.device,
            bundle_dir=bundle_dir,
            cancel_event=self._cancel,
        )
        thread = QtCore.QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.work)
        worker.progress.connect(self.progress)
        worker.finished.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        worker.cancelled.connect(self._on_worker_cancelled)

        self._worker = worker
        self._thread = thread
        thread.start()
        self.started.emit()

    def cancel(self) -> None:
        """Ask the run in flight to stop at the next population boundary.

        Cooperative and idempotent: setting the event is all that happens
        here. The worker notices in the engine's ``progress_cb``, before the
        next population runs, and emits ``cancelled``. A run whose last
        population has already started finishes normally -- there is no safe
        way to interrupt a torch kernel, and ``QThread.terminate()`` would
        leave torch's allocator in an undefined state.
        """
        self._cancel.set()

    # ------------------------------------------------------- worker callbacks

    def _on_worker_finished(self, payload: Dict[str, Any]) -> None:
        """Publish a finished run. Runs on the GUI thread (queued signal)."""
        if self._snapshot is None or self._rendered is None:
            raise RuntimeError(
                "a run finished with no snapshot or rendered stimulus held; "
                "RunController.run() was not the one that started it"
            )
        result = RunResult(
            config_snapshot=self._snapshot,
            results=payload["results"],
            stimulus=self._rendered.stimulus,
            time_ms=self._rendered.time_ms,
            canvas=self._rendered.canvas,
            bundle_dir=self._bundle_dir,
            duration_ms=self._duration_ms,
            started=self._started_at,
            elapsed_s=float(payload["elapsed_s"]),
            quick=self._quick,
        )
        self._teardown_thread()
        # set_results is called here, on the GUI thread, never in the worker:
        # it emits resultsChanged, which every view is connected to.
        self._session.set_results(result)
        self.finished.emit(result)

    def _on_worker_failed(self, message: str) -> None:
        """Surface a failed run. Runs on the GUI thread (queued signal)."""
        self._teardown_thread()
        self.failed.emit(message)

    def _on_worker_cancelled(self) -> None:
        """Surface a cancelled run. Runs on the GUI thread (queued signal)."""
        self._teardown_thread()
        self.cancelled.emit()

    def _teardown_thread(self) -> None:
        """Stop and drop the worker thread, before any public signal goes out.

        ``running`` is therefore already ``False`` when a handler of
        ``finished``/``failed``/``cancelled`` runs, so a handler may start the
        next run straight away.
        """
        thread, worker = self._thread, self._worker
        self._thread = None
        self._worker = None
        if thread is None:
            return
        thread.quit()
        thread.wait()
        if worker is not None:
            worker.deleteLater()
        thread.deleteLater()
