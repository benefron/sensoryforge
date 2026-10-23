"""Scale tests: a world file to a report folder on many draws (roadmap A11b).

Tests marked ``scale`` are skipped unless the run asks for them (``pytest -m scale``
or ``pytest --run-scale``; the option and the skip live in ``tests/conftest.py``). The
small tests beside them carry no marker and run in every CI job. The size of the large
run comes from the environment, so a laptop and a cluster use the same test::

    SF_SCALE_DRAWS=5000 SF_SCALE_DURATION_MS=100 pytest -m scale tests/scale -s

``SF_SCALE_OUT`` names a folder to keep the data set and report in (default: a temporary
folder that is removed when the test ends). A bundle on the 80x80 recipes takes about 2 MB
per entry (it was 47 MB before the receptive-field weights were stored sparsely, A11d),
but the large tests still measure one entry first and refuse to start when the whole run
would not fit: ``SF_SCALE_MAX_GB`` (default 10) caps what one case may write, and the
volume of the output folder must keep ``SF_SCALE_MIN_FREE_GB`` (default 20) free after it.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
WORLD = FIXTURES / "worlds" / "tactile_small.yml"
DESIGN = FIXTURES / "design_8x8"
ELEMENTS_WORLD = FIXTURES / "worlds" / "elements_v1_2.yml"

#: The two tactile recipes the large run covers (``sensoryforge list-presets``).
RECIPES = ("tactile_sa1_ra1", "tactile_sa1_ra1_adex")


def env_int(name: str, default: int) -> int:
    """An integer from the environment, ``default`` when unset."""
    return int(os.environ.get(name, default))


@pytest.fixture
def scale_out():
    """The folder a scale run writes to: ``SF_SCALE_OUT``, or a temporary one removed after."""
    named = os.environ.get("SF_SCALE_OUT")
    if named:
        yield Path(named)
        return
    folder = Path(tempfile.mkdtemp(prefix="sf_scale_"))
    try:
        yield folder
    finally:
        shutil.rmtree(folder, ignore_errors=True)
