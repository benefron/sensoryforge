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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    torch.set_num_threads(1)
except Exception:  # pragma: no cover - fallback when backend disallows
    pass


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
