"""``python -m sensoryforge.evaluate.compare AUDIT DECLARED`` writes ``compare.json``.

AUDIT and DECLARED are the ``audit.json`` and ``declared.json`` files or the folders
that hold them. The raw table of the data set is printed after the files are written.
Exit status 0 on success, 2 on a usage or data error (message on stderr).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from sensoryforge.evaluate.compare.run import COMPARE_FILE, run_compare
from sensoryforge.evaluate.compare.table import (
    format_distribution_table,
    format_entry_table,
    format_histograms,
    format_stat_table,
)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the compare stage from the command line.

    Args:
        argv: Arguments; default ``sys.argv[1:]``.

    Returns:
        The exit status.
    """
    ap = argparse.ArgumentParser(
        prog="python -m sensoryforge.evaluate.compare",
        description="Pair the audit's measured values with the declared ones.",
    )
    ap.add_argument("audit", type=Path, help="audit.json or its folder")
    ap.add_argument("declared", type=Path, help="declared.json or its folder")
    ap.add_argument("--out", type=Path, help="output folder (default: the audit's)")
    ap.add_argument(
        "--histogram-bins", type=int, default=32, help="paired histogram bins"
    )
    ap.add_argument(
        "--table",
        default="dataset",
        metavar="GROUP",
        help="'dataset' (default), 'class:LABEL', 'split:LABEL' or 'entry:ID'",
    )
    ap.add_argument(
        "--stats", help="statistics to print, comma separated (default all)"
    )
    ap.add_argument("--metrics", help="metric ids to print, comma separated")
    ap.add_argument(
        "--distributions",
        action="store_true",
        help="print the KS and Wasserstein distances of the group instead",
    )
    ap.add_argument(
        "--histogram", metavar="METRIC", help="print one metric's histograms"
    )
    ap.add_argument(
        "--quiet", action="store_true", help="write the files, print no table"
    )
    args = ap.parse_args(argv)
    try:
        audit_dir = args.audit if args.audit.is_dir() else args.audit.parent
        out = args.out or audit_dir
        compare = run_compare(
            args.audit, args.declared, out, histogram_bins=args.histogram_bins
        )
        n_pairs = sum(compare["counts"].values())
        print(
            f"compared {compare['entries']['n']} entries ({n_pairs} pairs) "
            f"-> {out / COMPARE_FILE}"
        )
        if args.quiet:
            return 0
        metrics = args.metrics.split(",") if args.metrics else None
        stats = args.stats.split(",") if args.stats else None
        if args.histogram:
            text = format_histograms(compare, args.histogram, args.table)
        elif args.distributions:
            text = format_distribution_table(compare, args.table)
        elif args.table.startswith("entry:"):
            text = format_entry_table(
                out / COMPARE_FILE, args.table.partition(":")[2], metrics
            )
        else:
            text = format_stat_table(compare, args.table, stats, metrics)
        print(text)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"compare: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
