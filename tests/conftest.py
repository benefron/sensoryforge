"""
Test configuration and fixtures for the bio-inspired encoding project.
"""

import os
import sys
from pathlib import Path

import pytest
import numpy as np
import torch

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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    torch.set_num_threads(1)
except Exception:  # pragma: no cover - fallback when backend disallows
    pass


_session_exit_status = 0


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    """Disable the cyclic GC only for sessions that collect `gui`-marked tests.

    ``trylast=True`` so this runs after pytest's own ``-m``/``-k`` filtering
    (also a ``pytest_collection_modifyitems`` hookimpl) has already removed
    deselected items from ``items`` -- otherwise ``-m "not gui"`` would still
    see the not-yet-deselected gui items here and disable gc anyway.

    Destroying pyqtgraph ViewBox/GraphicsItem hierarchies from several Qt
    test modules in one process segfaults reliably once Python's cyclic
    collector sweeps their reference cycles (F-035: reproduced inside
    pyqtgraph's ScatterPlotItem render path, called from
    MechanoreceptorTab._add_receptor_scatter_by_weight via a ViewBox lambda
    left over from a previously destroyed tab's plot -- a real GUI bug this
    only works around for the test harness, not a fix). Refcounting alone
    still frees everything that isn't in a reference cycle, which is
    sufficient for a test-session-sized run, but a non-GUI session gets no
    benefit from paying that cost, so scope it to sessions that actually
    collect GUI tests.
    """
    import gc

    if any(item.get_closest_marker("gui") is not None for item in items):
        gc.disable()


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
