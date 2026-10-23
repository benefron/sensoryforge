"""``python -m sensoryforge.evaluate.diff BASELINE NEW`` writes ``diff.json``.

BASELINE and NEW are the data-set folders of two runs of the same data set. Exit
status 0 when the runs show no difference (identical, or equal within the golden
tolerance), 1 when they differ, 2 on a usage or data error (message on stderr).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sensoryforge.evaluate.diff.arrays import DIFF_RTOL
from sensoryforge.evaluate.diff.run import DIFF_FILE, run_diff
from sensoryforge.testing.golden import FLOAT32_GOLDEN_ATOL


def format_report(doc: Dict[str, Any]) -> str:
    """The diff as text: verdict, counts, first divergence, changed metrics."""
    c = doc["counts"]
    lines = [
        f"verdict: {doc['verdict']}",
        "entries: "
        + ", ".join(f"{n} {s}" for s, n in c.items() if n)
        + f" ({doc['entries']['n']} in all)",
    ]
    for note in doc["notes"]:
        lines.append(f"note: {note}")
    first = doc["first_divergence"]
    if first:
        lines.append(
            f"first divergence: {first['entry']} at step {first['step']} "
            f"({first['ms']:g} ms)"
        )
    for name, n in doc["by_array"].items():
        lines.append(
            f"  {name}: {n['n_different']} entries differ, "
            f"{n['n_equivalent']} equivalent"
        )
    for r in doc["most_divergent"][:5]:
        lines.append(
            f"  {r['entry']}: first at {r['first_ms']} ms, largest difference "
            f"{r['largest_abs_diff']:.3g}"
        )
    changed = [m for m in (doc["metrics"] or []) if m["n_changed"]]
    if doc["metrics"] is not None:
        lines.append(f"metrics: {len(changed)} of {len(doc['metrics'])} changed")
    for m in changed[:15]:
        where = "".join(
            f" {x}"
            for x in (
                m["population"],
                None if m["channel"] is None else f"ch{m['channel']}",
            )
            if x
        )
        lines.append(
            f"  {m['id']}{where}: {m['n_changed']} of {m['n_pairs']} entries, "
            f"mean {m['mean_a']} -> {m['mean_b']}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the diff from the command line.

    Args:
        argv: Arguments; default ``sys.argv[1:]``.

    Returns:
        The exit status.
    """
    ap = argparse.ArgumentParser(
        prog="python -m sensoryforge.evaluate.diff",
        description="What moved between two runs of the same data set.",
    )
    ap.add_argument("baseline", type=Path, help="data-set folder of the baseline run")
    ap.add_argument("new", type=Path, help="data-set folder of the new run")
    ap.add_argument(
        "--out", type=Path, default=Path("evaluate_diff"), help="output folder"
    )
    ap.add_argument(
        "--atol",
        type=float,
        default=FLOAT32_GOLDEN_ATOL,
        help="absolute tolerance (default: the golden fixtures')",
    )
    ap.add_argument(
        "--rtol",
        type=float,
        default=DIFF_RTOL,
        help="relative tolerance (default: two float32 steps, 2**-22)",
    )
    ap.add_argument(
        "--no-metrics", action="store_true", help="skip the per-metric deltas"
    )
    ap.add_argument("--packs", default="core", help="metric packs of the two audits")
    ap.add_argument("--audit-baseline", type=Path, help="reuse this folder's audit")
    ap.add_argument("--audit-new", type=Path, help="reuse this folder's audit")
    ap.add_argument(
        "--quiet", action="store_true", help="write the files, print no report"
    )
    args = ap.parse_args(argv)
    try:
        doc = run_diff(
            args.baseline,
            args.new,
            args.out,
            atol=args.atol,
            rtol=args.rtol,
            metrics=not args.no_metrics,
            packs=[p for p in args.packs.split(",") if p],
            audit_baseline=args.audit_baseline,
            audit_new=args.audit_new,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"diff: {exc}", file=sys.stderr)
        return 2
    print(f"compared {doc['entries']['n']} entries -> {args.out / DIFF_FILE}")
    if not args.quiet:
        print(format_report(doc))
    return 1 if doc["verdict"] == "different" else 0


if __name__ == "__main__":
    sys.exit(main())
