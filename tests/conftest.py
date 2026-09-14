"""
Test configuration and fixtures for the bio-inspired encoding project.
"""
import gc
import os
import sys
from pathlib import Path

import pytest
import numpy as np
import torch

# Qt/pyqtgraph widget teardown is not safe under Python's cyclic garbage
# collector: destroying pyqtgraph ViewBox/GraphicsItem hierarchies from
# several Qt test modules in one process segfaults reliably once the GC
# sweeps across their reference cycles (reproduced: test_grid_population_ux.py
# crashes mid-run when several GUI test files share a process). Disabling
# the cyclic collector avoids that; CPython's refcounting still frees
# everything that isn't in a reference cycle, so this does not leak in any
# test-session-sized run.
gc.disable()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    torch.set_num_threads(1)
except Exception:  # pragma: no cover - fallback when backend disallows
    pass


_session_exit_status = 0


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
    when a QApplication instance actually exists (never do this for a
    non-GUI session, where normal teardown is safe and desirable).
    """
    try:
        from PyQt5 import QtWidgets
        app = QtWidgets.QApplication.instance()
    except ImportError:
        app = None
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
