"""
Test configuration and fixtures for the bio-inspired encoding project.
"""

import os
import sys
from pathlib import Path

# Thread limits must be set before torch is imported: torch fixes its
# intra-op thread count at import, so these lines, which used to sit after the
# imports below, never took effect. CI therefore ran with every core the runner
# had, and matrix reductions split across that many threads rounded
# differently from a run with fewer (seen as sub-float32-step parity
# differences on Linux, F-071).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pytest  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Hermetic GUI preferences (sensoryforge/gui/settings.py). Without this the
# GUI tests read and write the preferences of whoever runs them -- expert mode,
# which collapsible sections are expanded -- so they passed on a machine where
# someone had once expanded a section and failed on a fresh CI runner, and a
# test run could overwrite a developer's real GUI state. A fresh directory per
# session makes every run start from the application's true defaults, here and
# in CI alike. Subprocesses the tests launch inherit it.
import tempfile  # noqa: E402

os.environ["SENSORYFORGE_SETTINGS_DIR"] = tempfile.mkdtemp(
    prefix="sensoryforge-test-settings-"
)


try:
    torch.set_num_threads(1)
except Exception:  # pragma: no cover - fallback when backend disallows
    pass


_session_exit_status = 0


def _collect_qt_garbage() -> None:
    """Deliver Qt's pending deletions, then run the cyclic collector."""
    import gc

    # Widgets closed at teardown are deleted by Qt on the next event-loop
    # pass (deleteLater); their pyqtgraph cycles only become garbage then.
    threads = sys.modules.get("sensoryforge.gui.execution.threads")
    if threads is not None:
        threads.wait_all()  # a worker still running must not be collected
    qtcore = sys.modules.get("PyQt5.QtCore")
    if qtcore is not None and qtcore.QCoreApplication.instance() is not None:
        for _ in range(2):
            qtcore.QCoreApplication.sendPostedEvents(None, qtcore.QEvent.DeferredDelete)
            qtcore.QCoreApplication.processEvents()
    gc.collect()


@pytest.fixture(autouse=True)
def _collect_gui_garbage_at_the_test_boundary(request):
    """Run the cyclic collector before and after every ``gui`` test (F-035, F-085).

    The collector stays enabled for the whole session. Freeing pyqtgraph
    objects left in reference cycles by a destroyed window, in the middle of
    another test, can destroy a C++ object that test is using (measured:
    "wrapped C/C++ object of type ViewBox has been deleted" in
    GridPreview.set_grids / SensorsScreen._add_highlight). Collecting before
    a gui test starts -- when none of its windows exist -- clears leftovers of
    any earlier test, gui-marked or not (CI failed on a test whose predecessor
    was not gui-marked); collecting after it clears its own. The shipped app
    builds each plot once per window lifetime, so it has no such boundary.
    """
    is_gui = request.node.get_closest_marker("gui") is not None
    if is_gui:
        _collect_qt_garbage()
    yield
    if is_gui:
        _collect_qt_garbage()


def pytest_sessionfinish(session, exitstatus):
    """Record the session's exit status for pytest_unconfigure to reuse."""
    global _session_exit_status
    _session_exit_status = int(exitstatus)


def pytest_unconfigure(config):
    """Force-exit after a Qt session to dodge the F-016 teardown crash.

    PyQt5/pyqtgraph objects (QApplication chief among them) can segfault or
    abort during CPython's normal interpreter teardown -- observed reliably
    for Qt test files in this repo. Once pytest has already recorded the
    real exit status, skip that teardown entirely with os._exit(), but only
    when PyQt5.QtWidgets was actually imported during the session AND a
    QApplication instance exists (never do this for a non-GUI session --
    checking sys.modules instead of importing PyQt5 here means a
    non-GUI session never imports PyQt5 at all during teardown).
    """
    qtwidgets = sys.modules.get("PyQt5.QtWidgets")
    app = qtwidgets.QApplication.instance() if qtwidgets is not None else None
    if app is not None:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_session_exit_status)


@pytest.fixture
def sample_pressure_grid():
    """Create a sample pressure grid for testing."""
    return np.random.rand(80, 80) * 100


@pytest.fixture
def sample_spike_train():
    """Create a sample spike train for testing."""
    return torch.randint(0, 2, (1000, 100), dtype=torch.float32)


@pytest.fixture
def test_parameters():
    """Common test parameters."""
    return {
        "grid_size": (80, 80),
        "n_sa_neurons": 100,
        "n_ra_neurons": 196,
        "dt": 1e-4,
        "timesteps": 1000,
    }
