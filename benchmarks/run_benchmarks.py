#!/usr/bin/env python
"""Benchmark harness for SensoryForge (Phase 4, Wave T / F-022).

This is the harness the project did not previously have, and the reason it was
written: `docs_root/LEDGER.md` finding F-056 records that the memory watchdog's
peak-RSS figure for the *whole test suite* varies from roughly 800 MB to 1,500 MB
run to run for identical code (873 MB vs. 1,517 MB for the same commit), which
makes it useless for detecting a regression. This harness instead measures one
simulation cell at a time, repeats each measurement, and reports a distribution
-- not a single sample -- so a later run can actually be compared against it.

Design, stated plainly:

* Build time (constructing ``SimulationEngine``, which builds grids and
  receptive-field banks) and run time (``engine.run()``) are timed separately,
  because they scale differently with grid size vs. neuron count and mixing
  them hides both.
* Each cell is run in its own subprocess (fresh ``torch``/``sensoryforge``
  import), with one warm-up repeat discarded before the timed repeats, so the
  reported distribution reflects steady-state behaviour, not import cost.
* Peak resident memory is measured with the same technique as the Phase 1
  memory watchdog (``docs/development/handover/phase1_tasks.md`` appendix):
  sample the subprocess's RSS (and any children's) once a fraction of a
  second and keep the peak. It is re-implemented in Python here (rather than
  shelling out to a copy of ``memwatch.sh``) so it can watch exactly the one
  subprocess running exactly one cell, but it is the same ``ps``-based
  sum-of-RSS technique, not a second mechanism. See the "What this does and
  does not capture" note in `docs/reference/benchmarks.md` for what a
  wall-clock-sampled RSS figure misses.
* Every number is recorded alongside the machine, Python/torch/SensoryForge
  versions, the thread-count environment, and the seed, because a number
  without them cannot be compared to anything.

Usage::

    python -m benchmarks.run_benchmarks                      # run every cell, write JSON + docs
    python -m benchmarks.run_benchmarks --cell smoke_10x10    # run one cell
    python -m benchmarks.run_benchmarks --no-docs             # skip regenerating benchmarks.md
    python -m benchmarks.run_benchmarks --out my.json

    # internal worker mode (invoked by the harness itself as a subprocess):
    python -m benchmarks.run_benchmarks --_worker '<cell-json>'
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"
DEFAULT_JSON_OUT = RESULTS_DIR / "latest.json"
DOCS_OUT = REPO_ROOT / "docs" / "reference" / "benchmarks.md"

DEFAULT_SEED = 1234
WARMUP_REPEATS = 1

# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------
#
# Each cell builds a two-population (SA + RA) canonical config -- the same
# shape as `examples/canonical_config.yml`, i.e. "both pressure-simulation
# populations" per the Wave T spec -- on a square grid, and drives it with a
# random stimulus for `duration_ms`. Kept short deliberately (per the
# coordinator's 2026-09-17 note): several quick repeats of a modest cell
# report the spread far more reliably than one long run of a large one, and
# survive interruption. "smoke_10x10" is the fast sanity cell; "large_80x80"
# is the one T2 asks for explicitly -- an 80x80 grid with both
# pressure-simulation populations (SA + RA) simulated over one full second
# -- kept to a small repeat count (3) so it stays a ~10s cell, not "many
# minutes", per the coordinator's 2026-09-17 note.


@dataclass
class Cell:
    name: str
    rows: int
    cols: int
    neurons_per_row_sa: int
    neurons_per_row_ra: int
    duration_ms: float
    device: str = "cpu"
    repeats: int = 5
    dt_ms: float = 1.0


def default_cells() -> List[Cell]:
    cells = [
        Cell(
            name="smoke_10x10",
            rows=10,
            cols=10,
            neurons_per_row_sa=3,
            neurons_per_row_ra=4,
            duration_ms=20.0,
            repeats=7,
        ),
        Cell(
            name="medium_40x40",
            rows=40,
            cols=40,
            neurons_per_row_sa=10,
            neurons_per_row_ra=14,
            duration_ms=100.0,
            repeats=5,
        ),
        Cell(
            name="large_80x80",
            rows=80,
            cols=80,
            neurons_per_row_sa=10,
            neurons_per_row_ra=14,
            duration_ms=1000.0,
            repeats=3,
        ),
    ]
    # Device sweep: add the smoke cell on every extra device this machine
    # actually has, so "across device" is a real measurement, not a claim.
    try:
        import torch

        if torch.cuda.is_available():
            cells.append(
                Cell(
                    name="smoke_10x10_cuda",
                    rows=10,
                    cols=10,
                    neurons_per_row_sa=3,
                    neurons_per_row_ra=4,
                    duration_ms=20.0,
                    device="cuda",
                    repeats=7,
                )
            )
        if torch.backends.mps.is_available():
            cells.append(
                Cell(
                    name="smoke_10x10_mps",
                    rows=10,
                    cols=10,
                    neurons_per_row_sa=3,
                    neurons_per_row_ra=4,
                    duration_ms=20.0,
                    device="mps",
                    repeats=7,
                )
            )
    except ImportError:
        pass
    return cells


# The CI regression guard (T3) runs exactly this one cell -- it must be fast.
CI_GUARD_CELL = Cell(
    name="ci_guard",
    rows=10,
    cols=10,
    neurons_per_row_sa=3,
    neurons_per_row_ra=4,
    duration_ms=20.0,
    repeats=7,
)


# ---------------------------------------------------------------------------
# Worker: runs inside a subprocess, one cell, prints JSON to stdout.
# ---------------------------------------------------------------------------


def _run_cell_worker(cell: Cell, seed: int) -> Dict[str, Any]:
    """Build+run `cell` (warm-up + timed repeats) in the current process.

    Returns a dict with raw per-repeat timings; statistics are computed by
    the parent so the worker's job is only to produce samples.
    """
    import torch
    import numpy as np

    from sensoryforge.config.schema import (
        SensoryForgeConfig,
        GridConfig,
        PopulationConfig,
        SimulationConfig,
    )
    from sensoryforge.core.simulation_engine import SimulationEngine

    def build_config() -> SensoryForgeConfig:
        return SensoryForgeConfig(
            grids=[
                GridConfig(
                    name="Main Grid",
                    arrangement="grid",
                    rows=cell.rows,
                    cols=cell.cols,
                    spacing=0.15,
                )
            ],
            populations=[
                PopulationConfig(
                    name="SA Population",
                    neuron_type="SA",
                    neuron_model="izhikevich",
                    filter_method="sa",
                    innervation_method="gaussian",
                    neurons_per_row=cell.neurons_per_row_sa,
                ),
                PopulationConfig(
                    name="RA Population",
                    neuron_type="RA",
                    neuron_model="izhikevich",
                    filter_method="ra",
                    innervation_method="gaussian",
                    neurons_per_row=cell.neurons_per_row_ra,
                ),
            ],
            simulation=SimulationConfig(device=cell.device, dt_ms=cell.dt_ms),
        )

    def one_iteration() -> Dict[str, float]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = build_config()

        t0 = time.perf_counter()
        engine = SimulationEngine(cfg)
        t1 = time.perf_counter()

        n_steps = max(1, int(round(cell.duration_ms / cell.dt_ms)))
        stimulus = torch.rand(n_steps, cell.rows, cell.cols, device=cell.device)

        t2 = time.perf_counter()
        engine.run(stimulus)
        t3 = time.perf_counter()

        return {"build_s": t1 - t0, "run_s": t3 - t2}

    # Warm-up (discarded): pays for lazy imports, registry population,
    # any device-specific first-call cost (kernel compilation on MPS/CUDA).
    for _ in range(WARMUP_REPEATS):
        one_iteration()

    samples = [one_iteration() for _ in range(cell.repeats)]
    return {
        "cell": asdict(cell),
        "build_s": [s["build_s"] for s in samples],
        "run_s": [s["run_s"] for s in samples],
    }


# ---------------------------------------------------------------------------
# Memory watchdog (same technique as docs/development/handover/phase1_tasks.md
# appendix: sample RSS of the process + its children, keep the peak).
# ---------------------------------------------------------------------------


def _rss_mb(pid: int) -> int:
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return int(out) // 1024 if out else 0
    except (OSError, ValueError):
        return 0


def _children(pid: int) -> List[int]:
    try:
        out = subprocess.run(
            ["pgrep", "-P", str(pid)], capture_output=True, text=True, check=False
        ).stdout.strip()
        return [int(p) for p in out.splitlines() if p.strip()]
    except (OSError, ValueError):
        return []


def run_cell_with_watchdog(
    cell: Cell, seed: int, sample_interval_s: float = 0.05
) -> Dict[str, Any]:
    """Run one cell in a subprocess, watching its peak summed RSS.

    Returns the worker's JSON result plus ``peak_rss_mb`` and wall-clock
    ``started_at``/``finished_at`` timestamps (UTC, ISO 8601) so a run taken
    across a machine sleep/OOM event (see benchmarks.md) can be told apart
    from one that was not.
    """
    payload = json.dumps(asdict(cell))
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "benchmarks.run_benchmarks",
            "--_worker",
            payload,
            "--seed",
            str(seed),
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak = 0
    while proc.poll() is None:
        rss = _rss_mb(proc.pid)
        for c in _children(proc.pid):
            rss += _rss_mb(c)
        peak = max(peak, rss)
        time.sleep(sample_interval_s)
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"benchmark cell {cell.name!r} worker failed (exit {proc.returncode}):\n{stderr}"
        )
    result = json.loads(stdout)
    result["peak_rss_mb"] = peak
    result["started_at"] = started_at
    result["finished_at"] = finished_at
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _stats(samples: List[float]) -> Dict[str, float]:
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "spread_s": max(samples) - min(samples),
        "n": len(samples),
    }


def summarize(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cell": raw["cell"],
        "build": _stats(raw["build_s"]),
        "run": _stats(raw["run_s"]),
        "peak_rss_mb": raw["peak_rss_mb"],
        "started_at": raw["started_at"],
        "finished_at": raw["finished_at"],
        "raw_build_s": raw["build_s"],
        "raw_run_s": raw["run_s"],
    }


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------


def _cpu_brand() -> str:
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            if out:
                return out
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def environment_metadata(seed: int) -> Dict[str, Any]:
    import os

    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = "not installed"

    try:
        import importlib.metadata as importlib_metadata

        sf_version = importlib_metadata.version("sensoryforge")
    except Exception:
        try:
            import sensoryforge

            sf_version = getattr(sensoryforge, "__version__", "unknown")
        except ImportError:
            sf_version = "unknown"

    return {
        "machine": platform.platform(),
        "cpu": _cpu_brand(),
        "python": platform.python_version(),
        "torch": torch_version,
        "sensoryforge": sf_version,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell", action="append", help="Run only this cell (repeatable)."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument(
        "--no-docs", action="store_true", help="Skip regenerating benchmarks.md"
    )
    parser.add_argument(
        "--_worker",
        help=argparse.SUPPRESS,
        default=None,
    )
    args = parser.parse_args(argv)

    if args._worker is not None:
        # Internal: run exactly one cell in this process, print JSON, exit.
        cell_dict = json.loads(args._worker)
        cell = Cell(**cell_dict)
        result = _run_cell_worker(cell, args.seed)
        print(json.dumps(result))
        return 0

    all_cells = default_cells() + [CI_GUARD_CELL]
    cells = default_cells()
    if args.cell:
        wanted = set(args.cell)
        cells = [c for c in all_cells if c.name in wanted]
        missing = wanted - {c.name for c in cells}
        if missing:
            raise SystemExit(f"Unknown cell(s): {sorted(missing)}")

    results = []
    for cell in cells:
        print(f"Running cell {cell.name!r} ({cell.repeats} repeats + 1 warm-up)...")
        raw = run_cell_with_watchdog(cell, args.seed)
        summary = summarize(raw)
        print(
            f"  build: median={summary['build']['median_s']*1000:.2f}ms "
            f"spread={summary['build']['spread_s']*1000:.2f}ms | "
            f"run: median={summary['run']['median_s']*1000:.2f}ms "
            f"spread={summary['run']['spread_s']*1000:.2f}ms | "
            f"peak_rss={summary['peak_rss_mb']}MB"
        )
        results.append(summary)

    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment_metadata(args.seed),
        "cells": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {args.out}")

    if not args.no_docs:
        from benchmarks.generate_table import generate

        generate(output, DOCS_OUT)
        print(f"Wrote {DOCS_OUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
