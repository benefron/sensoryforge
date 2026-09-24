"""Tests for :mod:`sensoryforge.gui.execution.run_controller`.

Four properties are load-bearing and each is checked by observing the real
engine, not a stub:

1. the engine is *built and run on the worker thread*, and the GUI thread
   keeps processing events while it runs;
2. cancel is cooperative and real -- the populations after the cancel never
   execute, and no half-written bundle is left behind;
3. a failure inside the engine becomes a ``failed`` signal and leaves the
   controller reusable;
4. a finished run's bundle lands under the project's ``runs/`` directory and
   loads with :func:`sensoryforge.io.bundle.load_bundle`.
"""

from __future__ import annotations

import copy
import threading
import time

import pytest

pytestmark = pytest.mark.gui

from PyQt5 import QtCore  # noqa: E402

from sensoryforge.config.schema import SensoryForgeConfig  # noqa: E402
from sensoryforge.core.simulation_engine import SimulationEngine  # noqa: E402
from sensoryforge.gui.execution.run_controller import RunController  # noqa: E402
from sensoryforge.gui.project import ProjectHandle  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.io.bundle import load_bundle  # noqa: E402

PRESET = "sensoryforge/presets/tactile_sa1_ra1.yml"

#: How long a run may take before a test gives up (ms). Generous: the
#: instrumented runs below deliberately sleep per population.
TIMEOUT_MS = 30000


def _config(n_populations: int = 2) -> SensoryForgeConfig:
    """The preset shrunk to 20x20, with ``n_populations`` populations."""
    config = SensoryForgeConfig.from_yaml_file(PRESET)
    config.grids[0].rows = 20
    config.grids[0].cols = 20
    config.simulation.seed = 1
    while len(config.populations) < n_populations:
        extra = copy.deepcopy(config.populations[0])
        extra.name = f"Extra {len(config.populations)}"
        config.populations.append(extra)
    del config.populations[n_populations:]
    return config


class _Spy:
    """Records what each population's kernel call saw, from whichever thread.

    Patching ``SimulationEngine._run_pop_from_drive`` is the only way to
    observe *which* populations actually executed: it is the one call every
    population makes, and it makes it exactly once.
    """

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.threads: list[int] = []
        self.lock = threading.Lock()

    @property
    def calls(self) -> int:
        with self.lock:
            return len(self.threads)


@pytest.fixture
def spy(monkeypatch):
    """Patch the engine's kernel and constructor to record threads (and stall).

    Yields a :class:`_Spy`; set ``spy.delay_s`` before starting a run to make
    each population slow enough for the test to interleave with it.
    """
    recorder = _Spy()
    original_kernel = SimulationEngine._run_pop_from_drive
    original_init = SimulationEngine.__init__
    init_threads: list[int] = []

    def kernel(*args, **kwargs):
        with recorder.lock:
            recorder.threads.append(threading.get_ident())
        if recorder.delay_s:
            time.sleep(recorder.delay_s)
        return original_kernel(*args, **kwargs)

    def init(self, config, device=None):
        init_threads.append(threading.get_ident())
        original_init(self, config, device=device)

    monkeypatch.setattr(SimulationEngine, "_run_pop_from_drive", staticmethod(kernel))
    monkeypatch.setattr(SimulationEngine, "__init__", init)
    recorder.init_threads = init_threads  # type: ignore[attr-defined]
    return recorder


def _session(config: SensoryForgeConfig, project_root=None) -> Session:
    session = Session(config)
    if project_root is not None:
        session.set_project(ProjectHandle.create(project_root, config))
    return session


# ----------------------------------------------------------------- threading


def test_engine_is_built_and_run_off_the_gui_thread_which_stays_responsive(qtbot, spy):
    """Item 1: worker-thread execution, with a live GUI event loop."""
    spy.delay_s = 0.15
    session = _session(_config(3))
    controller = RunController(session)

    progress: list[tuple] = []
    controller.progress.connect(lambda *a: progress.append(a))

    # A timer on the GUI thread: if the run blocked it, this never ticks.
    ticks = []
    timer = QtCore.QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
    timer.start()

    main_thread = threading.get_ident()
    with qtbot.waitSignal(controller.finished, timeout=TIMEOUT_MS):
        controller.run(duration_ms=30.0, bundle=False)
    timer.stop()

    assert len(progress) == 3, progress
    assert [p[0] for p in progress] == [0, 1, 2]
    assert [p[2] for p in progress] == [p.name for p in session.config.populations]
    # The GUI thread processed its own events while the engine ran.
    assert len(ticks) >= 2, f"GUI thread ticked only {len(ticks)} times"
    # Nothing to do with the engine happened on the GUI thread.
    assert spy.calls == 3
    assert main_thread not in spy.threads
    assert spy.init_threads and main_thread not in spy.init_threads
    assert len(set(spy.threads)) == 1


def test_results_are_published_to_the_session_on_the_gui_thread(qtbot):
    session = _session(_config(2))
    controller = RunController(session)
    seen: list = []
    session.resultsChanged.connect(
        lambda result: seen.append((result, threading.get_ident()))
    )

    with qtbot.waitSignal(controller.finished, timeout=TIMEOUT_MS) as blocker:
        controller.run(duration_ms=30.0, bundle=False)

    result = blocker.args[0]
    assert session.last_results is result
    assert seen[-1][0] is result
    assert seen[-1][1] == threading.get_ident()
    assert set(result.results) == {p.name for p in session.config.populations}
    assert result.stimulus.shape == (1, 30, 20, 20)
    assert result.duration_ms == 30.0
    assert result.quick is False
    assert result.elapsed_s > 0


# -------------------------------------------------------------------- cancel


def test_cancel_stops_before_the_remaining_populations_and_writes_no_bundle(
    qtbot, spy, tmp_path
):
    """Item 2: cooperative cancel, proven by which populations ran."""
    spy.delay_s = 0.4
    session = _session(_config(3), project_root=tmp_path / "project")
    controller = RunController(session)

    outcomes: list[str] = []
    controller.finished.connect(lambda _r: outcomes.append("finished"))
    controller.failed.connect(lambda _m: outcomes.append("failed"))
    # Cancel as soon as the first population is announced. The engine's
    # progress_cb fires before each population runs, so population 0 is
    # already in flight and 1 and 2 have not started.
    controller.progress.connect(lambda *_a: controller.cancel())

    with qtbot.waitSignal(controller.cancelled, timeout=TIMEOUT_MS):
        controller.run(duration_ms=30.0, bundle=True)

    assert outcomes == [], "a cancelled run must not also finish or fail"
    assert (
        spy.calls == 1
    ), f"{spy.calls} populations ran; the 2 after the cancel must not have"
    assert session.last_results is None
    bundle_dir = controller.bundle_dir
    assert bundle_dir is not None
    assert not bundle_dir.exists(), "a cancelled run must leave no bundle behind"
    with pytest.raises(FileNotFoundError):
        load_bundle(bundle_dir)
    assert session.project.list_runs() == []
    assert not controller.running


def test_cancel_before_the_run_starts_is_still_a_cancel(qtbot, spy):
    spy.delay_s = 0.0
    session = _session(_config(2))
    controller = RunController(session)
    controller.started.connect(controller.cancel)

    with qtbot.waitSignal(controller.cancelled, timeout=TIMEOUT_MS):
        controller.run(duration_ms=10.0, bundle=False)

    assert spy.calls == 0
    assert session.last_results is None


def test_the_controller_is_reusable_after_a_cancel(qtbot, spy):
    spy.delay_s = 0.3
    session = _session(_config(3))
    controller = RunController(session)
    cancel_once = controller.progress.connect(lambda *_a: controller.cancel())
    del cancel_once

    with qtbot.waitSignal(controller.cancelled, timeout=TIMEOUT_MS):
        controller.run(duration_ms=20.0, bundle=False)
    controller.progress.disconnect()

    spy.delay_s = 0.0
    with qtbot.waitSignal(controller.finished, timeout=TIMEOUT_MS):
        controller.run(duration_ms=20.0, bundle=False)
    assert session.last_results is not None


# ------------------------------------------------------------------ failures


def test_a_broken_config_surfaces_as_failed_and_leaves_a_reusable_controller(
    qtbot,
):
    """Item 3: an engine-build failure reaches the GUI as ``failed``."""
    config = _config(2)
    config.populations[0].neuron_model = "no_such_neuron"
    session = _session(config)
    controller = RunController(session)

    finished: list = []
    controller.finished.connect(finished.append)

    with qtbot.waitSignal(controller.failed, timeout=TIMEOUT_MS) as blocker:
        controller.run(duration_ms=20.0, bundle=False)

    message = blocker.args[0]
    assert "no_such_neuron" in message
    assert "ValueError" in message
    assert finished == []
    assert session.last_results is None
    assert not controller.running

    # Reusable: fix the config and the same controller runs it.
    session.set_by_path("populations.0.neuron_model", "izhikevich")
    with qtbot.waitSignal(controller.finished, timeout=TIMEOUT_MS):
        controller.run(duration_ms=20.0, bundle=False)
    assert session.last_results is not None


def test_a_second_run_while_running_raises(qtbot, spy):
    spy.delay_s = 0.3
    session = _session(_config(2))
    controller = RunController(session)

    with qtbot.waitSignal(controller.finished, timeout=TIMEOUT_MS):
        controller.run(duration_ms=20.0, bundle=False)
        with pytest.raises(RuntimeError, match="already in progress"):
            controller.run(duration_ms=20.0, bundle=False)


def test_an_unknown_quick_population_raises_before_anything_starts(qtbot):
    session = _session(_config(2))
    controller = RunController(session)

    with pytest.raises(ValueError, match="Nope"):
        controller.run(duration_ms=20.0, quick_population="Nope")
    assert not controller.running


# ------------------------------------------------------------------- bundles


def test_a_bundled_run_lands_under_the_projects_runs_dir_and_loads(qtbot, tmp_path):
    """Item 5: the bundle is where the project says, and ``load_bundle`` reads it."""
    session = _session(_config(2), project_root=tmp_path / "project")
    controller = RunController(session)

    with qtbot.waitSignal(controller.finished, timeout=TIMEOUT_MS) as blocker:
        controller.run(duration_ms=30.0, bundle=True)

    result = blocker.args[0]
    assert result.bundle_dir is not None
    assert result.bundle_dir.parent == session.project.runs_dir
    assert session.project.list_runs() == [result.bundle_dir]

    bundle = load_bundle(result.bundle_dir)
    assert set(bundle.populations) == {p.name for p in session.config.populations}
    assert bundle.config.grids[0].rows == 20


def test_no_project_means_no_bundle(qtbot):
    session = _session(_config(2))
    controller = RunController(session)

    with qtbot.waitSignal(controller.finished, timeout=TIMEOUT_MS) as blocker:
        controller.run(duration_ms=20.0, bundle=True)

    assert blocker.args[0].bundle_dir is None


# ---------------------------------------------------------------- quick runs


def test_a_quick_run_keeps_one_population_caps_the_duration_and_skips_the_bundle(
    qtbot, tmp_path
):
    session = _session(_config(2), project_root=tmp_path / "project")
    controller = RunController(session)
    name = session.config.populations[1].name

    with qtbot.waitSignal(controller.finished, timeout=TIMEOUT_MS) as blocker:
        controller.run(duration_ms=5000.0, bundle=True, quick_population=name)

    result = blocker.args[0]
    assert result.quick is True
    assert list(result.results) == [name]
    assert [p.name for p in result.config_snapshot.populations] == [name]
    assert result.duration_ms == 100.0
    assert result.stimulus.shape[1] == 100
    assert result.bundle_dir is None
    assert session.project.list_runs() == []
    # The session's own config is untouched: the snapshot was a deep copy.
    assert len(session.config.populations) == 2


def test_the_snapshot_is_independent_of_later_config_edits(qtbot, spy):
    spy.delay_s = 0.2
    session = _session(_config(2))
    before = session.config.populations[0].input_gain
    controller = RunController(session)
    controller.started.connect(
        lambda: session.set_by_path("populations.0.input_gain", 999.0)
    )

    with qtbot.waitSignal(controller.finished, timeout=TIMEOUT_MS) as blocker:
        controller.run(duration_ms=20.0, bundle=False)

    assert blocker.args[0].config_snapshot.populations[0].input_gain == before
    assert session.config.populations[0].input_gain == 999.0
