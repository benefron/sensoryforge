"""Every layered stimulus written before the world engine renders as it did (R10)."""

import importlib.util
from pathlib import Path

import pytest
import torch

from sensoryforge.testing.golden import assert_matches_golden

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_layered_golden", FIXTURES / "make_layered_golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GOLDEN = torch.load(FIXTURES / "layered_golden.pt", weights_only=True)
FRESH = _generator().render_all()


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_layered_renders_as_before(name):
    assert_matches_golden(FRESH[name], GOLDEN[name], what=f"layered {name}")
