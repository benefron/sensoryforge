"""The large run: thousands of draws from a world file to a report (roadmap A11b/A11c).

Skipped unless asked for (``pytest -m scale``). The size comes from the environment
(``SF_SCALE_DRAWS``, default 1000; ``SF_SCALE_DURATION_MS``, default 100; see
``conftest.py``). It checks the invariants of the small run at the large size and prints
the time of every stage. Peak memory is not compared across runs (F-056); the time per
stage is printed for a person to read and asserted nowhere.
"""

from __future__ import annotations

import pytest

from .conftest import env_int
from .pipeline_runs import check_report, run_world, throughput


@pytest.mark.scale
def test_thousands_of_draws_from_a_world_to_a_report(scale_out, capsys):
    draws = env_int("SF_SCALE_DRAWS", 1000)
    duration_ms = float(env_int("SF_SCALE_DURATION_MS", 100))
    doc = run_world(scale_out, draws, duration_ms, checks=True)
    check_report(doc, scale_out, draws, checks=True)
    with capsys.disabled():
        print("\nlarge scale:", throughput(doc, draws), "->", scale_out)
