"""The large run: thousands of draws from a world file to a report (roadmap A11b/A11c).

Skipped unless asked for (``pytest -m scale``). The size comes from the environment
(``SF_SCALE_DRAWS``, default 1000, for the plain run; ``SF_SCALE_MATRIX_DRAWS``, default 500,
for the matrix, whose bundles are about 2 MB each; ``SF_SCALE_DURATION_MS``, default 100;
see ``conftest.py``). The first test is the plain run on the 8x8 test sensor; the second is
the matrix: both tactile recipes (80x80 sensors) on both test worlds, with sessions and,
where the world has them, held-out classes and probes (A11c). Both check the invariants
of the small run at the large size and print the time of every stage. Peak memory is not
compared across runs (F-056); the time per
stage is printed for a person to read and asserted nowhere. The disk guard in
``pipeline_runs.py`` refuses a case that would write more than ``SF_SCALE_MAX_GB``.
"""

from __future__ import annotations

import pytest

from .conftest import ELEMENTS_WORLD, RECIPES, WORLD, env_int
from .pipeline_runs import check_report, run_recipe, run_world, throughput


@pytest.mark.scale
def test_thousands_of_draws_from_a_world_to_a_report(scale_out, capsys):
    draws = env_int("SF_SCALE_DRAWS", 1000)
    duration_ms = float(env_int("SF_SCALE_DURATION_MS", 100))
    doc = run_world(scale_out, draws, duration_ms, checks=True, guard=True)
    check_report(doc, scale_out, draws, checks=True)
    with capsys.disabled():
        print("\nlarge scale:", throughput(doc, draws), "->", scale_out)


@pytest.mark.scale
@pytest.mark.parametrize("preset", RECIPES)
@pytest.mark.parametrize(
    "world, held_out",
    [(WORLD, True), (ELEMENTS_WORLD, False)],
    ids=["tactile_small", "elements_v1_2"],
)
def test_both_recipes_across_worlds_and_splits(
    scale_out, capsys, world, held_out, preset
):
    draws = env_int("SF_SCALE_MATRIX_DRAWS", 500)
    duration_ms = float(env_int("SF_SCALE_DURATION_MS", 100))
    out = scale_out / f"{world.stem}_{preset}"
    doc = run_recipe(out, world, preset, draws, duration_ms, held_out=held_out)
    check_report(doc, out, None, checks=True)
    assert doc["counts"]["entries"] > draws // 2  # train takes most of the budget
    with capsys.disabled():
        print(f"\n{world.stem} on {preset}:", throughput(doc, doc["counts"]["entries"]))
