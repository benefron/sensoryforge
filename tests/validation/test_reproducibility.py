"""S3: CI-side proof of the reproducibility claim (Wave S).

Runs ``scripts/reproduce_figure.py``'s ``--check`` logic in-process (not as
a subprocess against a fresh venv -- that full "clean checkout, fresh
environment, install the package" proof is ``scripts/reproduce_env.sh``,
which this suite does not invoke because F-053 forbids this project's own
test run from performing a `pip install`; see that script's own docstring)
and asserts a fresh run of the pressure-simulation recipe (Wave K5,
``examples/pressure_simulation_recipe.py``) reproduces the committed
reference under ``tests/fixtures/reference/reproducibility/`` exactly (spike
counts) and within the stated relative tolerance (mean rates) -- see
``scripts/reproduce_figure.py``'s module docstring for why those are the
right two comparisons.

Reference: the committed
``tests/fixtures/reference/reproducibility/summary_stats.json``, produced by
``python scripts/reproduce_figure.py --write-reference`` on this machine
(recorded 2026-09-16: darwin/arm64, the conda `sensoryforge` environment).
Tolerance: exact match on spike counts (integers from a fully deterministic
seeded run -- any difference is a real change, not noise); 1e-9 relative
tolerance on mean rates (float reduction-order slack only).

Perturbation proof (recorded 2026-09-16): editing a single spike count in
the committed reference JSON (braille/SA Population: 15 -> 999) and
re-running ``scripts/reproduce_figure.py --check`` fails with
``braille/SA Population: spike count 15 != reference 999`` -- i.e. the check
does distinguish a fresh run from a stale/wrong reference, not just confirm
the file exists. The reference was restored (regenerated with
``--write-reference``) immediately after.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import reproduce_figure  # noqa: E402


def test_pressure_simulation_recipe_reproduces_committed_reference():
    """A fresh --quick run of the K5 recipe matches the committed summary stats."""
    assert reproduce_figure.STATS_PATH.exists(), (
        "no committed reference at "
        f"{reproduce_figure.STATS_PATH} -- run "
        "`python scripts/reproduce_figure.py --write-reference` first"
    )

    actual = reproduce_figure.compute_summary_stats(quick=True)
    import json

    with open(reproduce_figure.STATS_PATH) as f:
        reference = json.load(f)

    problems = reproduce_figure._compare(actual, reference)
    assert not problems, "reproducibility check failed:\n" + "\n".join(problems)
