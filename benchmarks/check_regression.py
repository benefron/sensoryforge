#!/usr/bin/env python
"""CI performance regression guard (Wave T, T3).

Runs the fast ``ci_guard`` benchmark cell fresh and fails (non-zero exit) if
its run-phase median has regressed by more than a fixed factor against the
committed baseline in ``benchmarks/results/ci_baseline.json``.

Why this factor, not a round number
------------------------------------
The baseline was built from six independent back-to-back runs of ``ci_guard``
on quiet development hardware (see ``ci_baseline.json``): the run-phase
median varied by **3.85%** run to run. GitHub Actions runners are shared VMs
with noisy neighbours, cold caches and no thermal control, which is
routinely 2-4x noisier than a quiet local machine for a benchmark this
small (tens of milliseconds) -- CI-provided cloud runners are known to show
single-run wall-clock swings well past 2x for cells in this size range.

The guard's factor is **3.0x** the baseline median. That is chosen to sit
comfortably above the local noise floor (3.85%, i.e. ~1.04x) with wide
headroom for CI's extra noise, while still catching the scenario the spec
names explicitly -- "a future change that makes the engine ten times slower"
-- with a large margin (10x >> 3x). It will also catch anything that
regresses this cell by 3x or worse. It will **not** catch a regression
smaller than 3x: a change that makes this particular cell 50% or 100%
slower passes silently. That is the honest trade a fast, low-false-alarm CI
cell makes; the full distribution across every cell (not just this one) is
what ``docs/reference/benchmarks.md`` is for, and that is regenerated and
read by a human, not gated in CI.

Usage::

    python -m benchmarks.check_regression
    python -m benchmarks.check_regression --factor 5.0   # override for testing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.run_benchmarks import CI_GUARD_CELL, run_cell_with_watchdog, summarize

BENCH_DIR = Path(__file__).resolve().parent
BASELINE_PATH = BENCH_DIR / "results" / "ci_baseline.json"

DEFAULT_FACTOR = 3.0
DEFAULT_SEED = 1234


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=float, default=DEFAULT_FACTOR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    args = parser.parse_args(argv)

    baseline = json.loads(args.baseline.read_text())
    baseline_median_s = baseline["baseline_run_median_s"]
    limit_s = baseline_median_s * args.factor

    print(
        f"Running {CI_GUARD_CELL.name!r} ({CI_GUARD_CELL.repeats} repeats + 1 warm-up)..."
    )
    raw = run_cell_with_watchdog(CI_GUARD_CELL, args.seed)
    summary = summarize(raw)
    measured_median_s = summary["run"]["median_s"]
    regression_factor = measured_median_s / baseline_median_s

    print(f"baseline run-phase median: {baseline_median_s * 1000:.3f} ms")
    print(f"measured run-phase median: {measured_median_s * 1000:.3f} ms")
    print(
        f"measured/baseline factor:  {regression_factor:.3f}x (limit: {args.factor:.1f}x)"
    )
    print(
        f"measured spread this run:  "
        f"{summary['run']['spread_s'] * 1000:.3f} ms "
        f"({summary['run']['spread_s'] / summary['run']['median_s'] * 100:.1f}% of median)"
    )

    if measured_median_s > limit_s:
        print(
            f"FAIL: run-phase median {measured_median_s * 1000:.3f}ms exceeds "
            f"{args.factor:.1f}x the baseline ({limit_s * 1000:.3f}ms)."
        )
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
