#!/usr/bin/env python
"""CI performance regression guard (Wave T, T3; calibrated per F-067).

Runs the fast ``ci_guard`` benchmark cell fresh and fails (non-zero exit) if
the engine has become slower by more than a fixed factor against the
committed baseline in ``benchmarks/results/ci_baseline.json``.

What is compared, and why not raw milliseconds
-----------------------------------------------
The baseline was measured on an Apple M3 Pro laptop. The guard runs on a
GitHub ``ubuntu-latest`` runner. Those are different machines, so the same
engine takes a different number of milliseconds on each, and that difference
is a fixed offset rather than noise. Comparing raw milliseconds would make
the guard's first CI run a coin flip: pass or fail by hardware alone, with
nothing about the code having changed. A guard that raises a false alarm on
its first run is disabled soon after, and then it protects nothing.

So each run also times a fixed reference kernel, in the same process,
interleaved with the engine repeats (``run_benchmarks._reference_kernel``).
The guard compares the engine's run time *divided by* the reference time,
now against then. A uniformly slower machine slows both and the ratio holds;
a change that makes the engine itself slower moves the ratio. The kernel is
shaped like the engine's inner loop -- many small tensor operations in a
Python loop -- because that, not a large matrix product, is where a cell
this small spends its time.

This removes most of the hardware offset, not all of it. The engine and the
kernel are not identical workloads, so a different CPU architecture can
still shift their ratio somewhat. That residual is why the factor stays
generous. Measured on the guard's first run on a GitHub Linux x86_64
runner: raw engine time 1.50x the Apple M3 Pro baseline, reference kernel
1.60x, calibrated factor 0.94x -- a shift of about 6 percent.

Why this factor
---------------
Across six independent back-to-back runs on the baseline machine the
normalised ratio varied by the spread recorded in ``ci_baseline.json``. The
guard's factor is **3.0x**. It sits far above that noise and above the
residual cross-architecture shift, while still catching the scenario the
spec names -- an engine made ten times slower -- with wide margin. It will
**not** catch a regression smaller than about 3x on this cell. That is the
honest trade for a fast guard that does not cry wolf; the full distribution
across every cell is what ``docs/reference/benchmarks.md`` is for, read by a
person rather than gated in CI.

A baseline recorded before calibration (no reference timing) is still
accepted, compared on raw milliseconds, with a warning that says so.

Usage::

    python -m benchmarks.check_regression
    python -m benchmarks.check_regression --factor 5.0   # override for testing
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from benchmarks.run_benchmarks import CI_GUARD_CELL, run_cell_with_watchdog, summarize

BENCH_DIR = Path(__file__).resolve().parent
BASELINE_PATH = BENCH_DIR / "results" / "ci_baseline.json"

DEFAULT_FACTOR = 3.0
DEFAULT_SEED = 1234


@dataclass(frozen=True)
class Verdict:
    """The outcome of one comparison."""

    passed: bool
    factor: float
    calibrated: bool
    explanation: str


def evaluate(
    measured_run_s: float,
    measured_reference_s: Optional[float],
    baseline_run_s: float,
    baseline_reference_s: Optional[float],
    limit: float,
) -> Verdict:
    """Decide whether the engine has regressed.

    Pure, so the decision can be tested without timing anything.

    Args:
        measured_run_s: Engine run-phase median from this run, seconds.
        measured_reference_s: Reference kernel median from this run, or
            ``None`` if it was not timed.
        baseline_run_s: Engine run-phase median in the baseline, seconds.
        baseline_reference_s: Reference kernel median in the baseline, or
            ``None`` for a baseline recorded before calibration.
        limit: The regression factor that fails the guard.

    Returns:
        A :class:`Verdict`. When both reference timings are present the
        factor is ``(measured_run / measured_reference) /
        (baseline_run / baseline_reference)``; otherwise it is the raw
        ratio of run times and ``calibrated`` is ``False``.

    Raises:
        ValueError: If any timing that is present is not positive.
    """
    for label, value in (
        ("measured_run_s", measured_run_s),
        ("baseline_run_s", baseline_run_s),
        ("measured_reference_s", measured_reference_s),
        ("baseline_reference_s", baseline_reference_s),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{label} must be positive, got {value!r}")

    calibrated = measured_reference_s is not None and baseline_reference_s is not None
    if calibrated:
        factor = (measured_run_s / measured_reference_s) / (
            baseline_run_s / baseline_reference_s
        )
        how = "engine time relative to the reference kernel, now against baseline"
    else:
        factor = measured_run_s / baseline_run_s
        how = (
            "raw run time against baseline -- NOT calibrated for hardware, so a "
            "different machine alone can move this"
        )
    passed = factor <= limit
    return Verdict(
        passed=passed,
        factor=factor,
        calibrated=calibrated,
        explanation=f"{factor:.3f}x ({how}); limit {limit:.1f}x",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=float, default=DEFAULT_FACTOR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    args = parser.parse_args(argv)

    baseline = json.loads(args.baseline.read_text())
    baseline_run_s = baseline["baseline_run_median_s"]
    baseline_reference_s = baseline.get("baseline_reference_median_s")

    print(
        f"Running {CI_GUARD_CELL.name!r} ({CI_GUARD_CELL.repeats} repeats + 1 warm-up)..."
    )
    summary = summarize(run_cell_with_watchdog(CI_GUARD_CELL, args.seed))
    measured_run_s = summary["run"]["median_s"]
    measured_reference_s = summary.get("reference", {}).get("median_s")

    verdict = evaluate(
        measured_run_s,
        measured_reference_s,
        baseline_run_s,
        baseline_reference_s,
        args.factor,
    )

    print(f"baseline engine run median:    {baseline_run_s * 1000:.3f} ms")
    print(f"measured engine run median:    {measured_run_s * 1000:.3f} ms")
    if baseline_reference_s is not None:
        print(f"baseline reference median:     {baseline_reference_s * 1000:.3f} ms")
    if measured_reference_s is not None:
        print(f"measured reference median:     {measured_reference_s * 1000:.3f} ms")
    print(
        f"measured engine spread:        {summary['run']['spread_s'] * 1000:.3f} ms "
        f"({summary['run']['spread_s'] / measured_run_s * 100:.1f}% of median)"
    )
    if not verdict.calibrated:
        print(
            "WARNING: comparing raw milliseconds. The baseline or this run has no "
            "reference timing, so a slower machine alone can fail this guard.",
            file=sys.stderr,
        )
    print(f"regression factor: {verdict.explanation}")
    print("PASS" if verdict.passed else "FAIL")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
