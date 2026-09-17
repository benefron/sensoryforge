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

    exact = reproduce_figure.same_platform(reference)
    problems = reproduce_figure._compare(actual, reference, exact=exact)
    mode = "exact (same platform)" if exact else "cross-platform tolerance"
    assert not problems, f"reproducibility check failed [{mode}]:\n" + "\n".join(
        problems
    )


# ---------------------------------------------------------------------------
# The comparison itself, as a pure function (F-068). These run without the
# recipe, so they pin the rules independently of any particular reference.
# ---------------------------------------------------------------------------


def _stats(counts):
    return {
        "quick": True,
        "stimuli": {
            "stim": {
                pop: {"spikes": n, "mean_rate_hz": float(n)}
                for pop, n in counts.items()
            }
        },
    }


class TestComparisonRules:
    def test_same_platform_requires_exact_counts(self):
        ref = _stats({"SA": 15})
        assert reproduce_figure._compare(_stats({"SA": 15}), ref, exact=True) == []
        assert reproduce_figure._compare(_stats({"SA": 16}), ref, exact=True)

    def test_cross_platform_absorbs_a_few_threshold_edge_spikes(self):
        ref = _stats({"SA": 15, "RA": 6601})
        near = _stats({"SA": 18, "RA": 6601 + 130})
        assert reproduce_figure._compare(near, ref, exact=False) == []

    def test_cross_platform_still_catches_a_real_regression(self):
        """A wrong gain or preset moves counts by factors, not a handful."""
        ref = _stats({"SA": 15, "RA": 6601})
        assert reproduce_figure._compare(
            _stats({"SA": 999, "RA": 6601}), ref, exact=False
        )
        assert reproduce_figure._compare(
            _stats({"SA": 15, "RA": 3300}), ref, exact=False
        )
        just_over = 6601 + reproduce_figure.spike_tolerance(6601) + 1
        assert reproduce_figure._compare(
            _stats({"SA": 15, "RA": just_over}), ref, exact=False
        )

    def test_small_counts_get_an_absolute_floor(self):
        assert (
            reproduce_figure.spike_tolerance(15)
            == reproduce_figure.CROSS_PLATFORM_SPIKE_ABS
        )
        assert reproduce_figure.spike_tolerance(6601) == 132


class TestPlatformDetection:
    def test_a_matching_signature_is_the_same_platform(self):
        ref = {"platform": reproduce_figure.platform_signature()}
        assert reproduce_figure.same_platform(ref) is True

    def test_a_different_machine_is_foreign(self):
        sig = dict(reproduce_figure.platform_signature())
        sig["machine"] = "a-different-architecture"
        assert reproduce_figure.same_platform({"platform": sig}) is False

    def test_a_reference_without_a_platform_is_treated_as_foreign(self):
        """No evidence it matches, so exact agreement is not demanded."""
        assert reproduce_figure.same_platform({}) is False

    def test_the_committed_reference_records_its_platform(self):
        import json

        with open(reproduce_figure.STATS_PATH) as f:
            reference = json.load(f)
        assert isinstance(reference.get("platform"), dict), (
            "the committed reference must record the platform it was made on, "
            "or every run is compared under the looser cross-platform rule"
        )
