"""The cyclic collector must never free a window under code still using it (F-085).

``SensoryForgeApp`` has no Qt parent, so Python owns it: freeing its wrapper
deletes the C++ window and every child. Two reference cycles made the cyclic
collector, rather than the owner, decide when a dropped window died:

* The "Open preset" actions were connected to
  ``functools.partial(self._load_config_file, path)``, a strong reference from
  a child ``QAction`` back to the window. A window its owner had dropped (a
  test that had returned) stayed alive, timers running, until a collection
  happened to run -- and a collection triggered inside one of its own slots
  (the Sensors screen's preview debounce) deleted the window under that
  slot: "wrapped C/C++ object of type ViewBox has been deleted" in
  ``GridPreview.set_grids``. The ViewBox was that same window's, never
  another window's.
* The stimulus preview connected its render worker's cross-thread signals to
  ``functools.partial(self._on_render_finished, generation)``, closing the
  cycle preview -> worker -> slot -> preview. Freeing that cycle while a
  render result was still queued made PyQt deliver the result to a slot it
  had just cleared: a segfault (``PyQtSlot::call``) at the next event loop.

Every collection here is made explicitly, mid-test; none relies on the
autouse fixture in ``tests/conftest.py`` collecting at test boundaries.
"""

import gc
import os
import subprocess
import sys
import textwrap
import weakref
from pathlib import Path

import pytest

pytestmark = pytest.mark.gui

from PyQt5 import QtCore, sip  # noqa: E402

import pyqtgraph as pg  # noqa: E402

from sensoryforge.config.schema import SensoryForgeConfig  # noqa: E402
from sensoryforge.gui import theme  # noqa: E402
from sensoryforge.gui.app import SensoryForgeApp  # noqa: E402
from sensoryforge.gui.screens import sensors as sensors_module  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.gui.widgets.grid_preview import GridPreview  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_PRESET = str(ROOT / "sensoryforge/presets/tactile_sa1_ra1.yml")


@pytest.fixture(autouse=True, scope="module")
def _matplotlib_already_imported():
    """Import matplotlib before any window here is built.

    The first ``theme.colormap()`` of a process imports matplotlib, whose
    pyparsing grammar leaves exception tracebacks in reference cycles that
    hold every frame of the importing stack -- a window's constructor and
    its caller's locals, if a window is being built. Importing it here keeps
    that one-time, third-party cycle off the windows these tests measure.
    """
    theme.colormap()


def _window():
    session = Session(SensoryForgeConfig.from_yaml_file(_PRESET))
    window = SensoryForgeApp(session)
    window.show_dialogs = False
    return window, session


def _view_boxes_of(window):
    """Every live ViewBox drawn inside ``window``."""
    boxes = []
    for box in list(pg.ViewBox.AllViews):
        if sip.isdeleted(box) or box.scene() is None:
            continue
        views = box.scene().views()
        if views and views[0].window() is window:
            boxes.append(box)
    return boxes


@pytest.fixture
def collector_off():
    """Run the body with the cyclic collector off; restore it afterwards."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def test_a_dropped_window_is_freed_at_once_without_the_collector(qtbot, collector_off):
    """Reference counting alone frees the window: nothing of its own keeps it."""
    window, session = _window()
    qtbot.addWidget(window)
    window._load_config_file(_PRESET)
    window.stage_list.setCurrentRow(2)
    alive = weakref.ref(window)

    del window, session

    assert alive() is None, (
        "a SensoryForgeApp its owner dropped is still alive, held only by a "
        "reference cycle; the cyclic collector will delete it later, "
        "possibly inside one of its own slots (F-085)"
    )


def test_a_collection_inside_a_dropped_windows_slot_deletes_nothing_under_it(
    qtbot, monkeypatch
):
    """The measured F-085 failure, made deterministic.

    The window leaves a preview update pending (as loading a config does),
    its owner drops it, and a full collection runs inside the Sensors
    screen's ``_update_preview`` slot. Before the fix that collection freed
    the window's cycle and deleted the window -- and the preview's ViewBox --
    while the slot was still drawing into it.
    """
    window, session = _window()
    qtbot.addWidget(window)
    window._screens["sensors"]._schedule_preview_update()
    del window, session

    real_build_grid = sensors_module.build_grid

    def build_grid_then_collect(*args, **kwargs):
        gc.collect()
        return real_build_grid(*args, **kwargs)

    monkeypatch.setattr(sensors_module, "build_grid", build_grid_then_collect)

    with qtbot.captureExceptions() as exceptions:
        qtbot.wait(4 * sensors_module._PREVIEW_DEBOUNCE_MS)

    assert not exceptions, "".join(
        f"{exc_type.__name__}: {exc}\n" for exc_type, exc, _tb in exceptions
    )


def test_collecting_a_dead_windows_cycles_mid_test_spares_a_live_window(qtbot):
    """A live window survives a mid-test collection of everything discarded.

    Garbage from three sources is collected while window ``live`` is in use:
    the pyqtgraph wrappers two destroyed windows left in reference cycles
    (one deleted by Qt, one closed with its deletion still pending and its
    stimulus render result queued), and a pyqtgraph widget ``live`` itself
    discarded and rebuilt at runtime (its Sensors preview replaced by a new
    ``GridPreview``, the old one detached so that Python owns it). No ViewBox
    of ``live`` is deleted, its new preview draws, and the event loop runs on.
    """

    def view_box_refs(window):
        return [
            weakref.ref(plot.getPlotItem().vb)
            for plot in window.findChildren(pg.PlotWidget)
        ]

    deleted, _ = _window()
    dead_boxes = view_box_refs(deleted)
    deleted.close()
    deleted.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    pending, _ = _window()
    dead_boxes += view_box_refs(pending)
    pending.close()  # joins its render thread: the result is queued
    pending.deleteLater()  # no event-loop turn yet
    del deleted, pending
    in_cycles = [box for box in dead_boxes if box() is not None]
    assert in_cycles, "the destroyed windows left no pyqtgraph wrapper in a cycle"

    live, session = _window()
    qtbot.addWidget(live)
    sensors = live._screens["sensors"]
    sensors._update_preview()

    old_preview = sensors.preview
    layout = old_preview.parentWidget().layout()
    new_preview = GridPreview()
    layout.replaceWidget(old_preview, new_preview)
    old_preview.close()  # GridPreview.closeEvent releases its connections
    old_preview.setParent(None)  # Python owns it now, C++ side and all
    sensors.preview = new_preview
    old_alive = weakref.ref(old_preview)
    del old_preview

    boxes = _view_boxes_of(live)
    assert len(boxes) > 1

    assert gc.collect() > 0
    qtbot.wait(50)  # deliver whatever the collected windows left queued

    assert any(box() is None for box in in_cycles)
    assert old_alive() is None
    assert [box for box in boxes if sip.isdeleted(box)] == []
    sensors._update_preview()
    assert new_preview.receptor_count() == sum(
        grid.rows * grid.cols for grid in session.config.grids
    )
    live.stage_list.setCurrentRow(2)
    assert live.stack.currentWidget() is live._screens["populations"]


def test_collecting_a_closed_window_with_a_render_result_queued_does_not_crash():
    """The segfault half of F-085, in a subprocess because it kills the process.

    Closing a window joins its stimulus render thread, so the render result
    waits in the event queue. A collection before that queue is delivered
    used to free the preview's cycle through the worker's slot, and the
    delivery then crashed inside ``PyQtSlot::call``.
    """
    script = textwrap.dedent(f"""
        import gc, sys
        from PyQt5 import QtCore, QtWidgets
        app = QtWidgets.QApplication(sys.argv)
        from sensoryforge.config.schema import SensoryForgeConfig
        from sensoryforge.gui import theme
        from sensoryforge.gui.app import SensoryForgeApp
        from sensoryforge.gui.session import Session
        theme.colormap()
        gc.disable()

        def window():
            config = SensoryForgeConfig.from_yaml_file({_PRESET!r})
            return SensoryForgeApp(Session(config))

        closed = window()
        closed.close()
        del closed
        live = window()
        gc.collect()
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(50, loop.quit)
        loop.exec_()
        print("event loop survived")
        sys.stdout.flush()
        import os
        os._exit(0)
        """)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert (
        result.returncode == 0 and "event loop survived" in result.stdout
    ), f"exit status {result.returncode}\n{result.stdout}\n{result.stderr[-3000:]}"
