"""Scale tests: a world file to a report folder on many draws (roadmap A11b).

Tests marked ``scale`` are skipped unless the run asks for them (``pytest -m scale``
or ``pytest --run-scale``; the option and the skip live in ``tests/conftest.py``). The
small tests beside them carry no marker and run in every CI job. The size of the large
run comes from the environment, so a laptop and a cluster use the same test::

    SF_SCALE_DRAWS=5000 SF_SCALE_DURATION_MS=100 pytest -m scale tests/scale -s

``SF_SCALE_OUT`` names a folder to keep the data set and report in (default: a temporary
folder that is removed).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
WORLD = FIXTURES / "worlds" / "tactile_small.yml"
DESIGN = FIXTURES / "design_8x8"


def env_int(name: str, default: int) -> int:
    """An integer from the environment, ``default`` when unset."""
    return int(os.environ.get(name, default))


@pytest.fixture
def scale_out(tmp_path):
    """The folder a scale run writes to: ``SF_SCALE_OUT`` or a temporary one."""
    named = os.environ.get("SF_SCALE_OUT")
    return Path(named) if named else tmp_path / "scale"
