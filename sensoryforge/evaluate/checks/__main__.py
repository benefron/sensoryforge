"""``python -m sensoryforge.evaluate.checks FOLDER`` writes and prints ``checks.json``.

Exit status 0 when no check failed (skipped checks do not fail a run), 1 when a check
failed or errored, 2 on a usage or data error (message on stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sensoryforge.evaluate.checks.run import CHECKS_FILE, run_checks

MARK = {"pass": "pass   ", "fail": "FAIL   ", "skipped": "skipped", "error": "ERROR  "}


def _params(items: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        dotted, sep, text = item.partition("=")
        pack, dot, name = dotted.partition(".")
        if not sep or not dot or not name:
            raise ValueError(f"--param expects PACK.NAME=VALUE, got {item!r}")
        try:
            value: Any = json.loads(text)
        except json.JSONDecodeError:
            value = text
        out.setdefault(pack, {})[name] = value
    return out


def _bounds(text: Optional[str]) -> Optional[Tuple[float, float]]:
    if not text:
        return None
    low, sep, high = text.partition(",")
    if not sep:
        raise ValueError(f"--voltage-bounds expects LOW,HIGH in mV, got {text!r}")
    return float(low), float(high)


def format_report(doc: Dict[str, Any]) -> str:
    """The checks as a text table, failures indented under their check."""
    lines = []
    for row in doc["checks"]:
        tail = f"{row['failed']}/{row['tested']} failed" if row["tested"] else ""
        lines.append(f"{MARK[row['status']]} {row['id']:<30} {tail}")
        if row["note"]:
            lines.append(f"           {row['note']}")
        for f in row["failures"][:5]:
            lines.append(f"           - {f['where']}: {f['detail']}")
        if row["failed"] > min(5, len(row["failures"])):
            lines.append(f"           … {row['failed']} in all")
    c = doc["counts"]
    lines.append(
        f"{c['pass']} passed, {c['fail']} failed, {c['skipped']} skipped, "
        f"{c['error']} errors"
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the checks from the command line.

    Args:
        argv: Arguments; default ``sys.argv[1:]``.

    Returns:
        The exit status.
    """
    ap = argparse.ArgumentParser(
        prog="python -m sensoryforge.evaluate.checks",
        description="Reference-free checks of the simulator on a data set.",
    )
    ap.add_argument(
        "folder", type=Path, help="data-set folder (manifest, index, bundles)"
    )
    ap.add_argument("--out", type=Path, help="output folder (default: FOLDER/evaluate)")
    ap.add_argument(
        "--manifest", type=Path, help="manifest (default: FOLDER/manifest.jsonl)"
    )
    ap.add_argument(
        "--packs", default="core", help="comma-separated check packs (default core)"
    )
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="PACK.NAME=VALUE",
        help="a pack parameter, e.g. tactile.tolerance=1 (repeatable)",
    )
    ap.add_argument(
        "--sample", type=int, default=8, help="entries the engine checks re-run"
    )
    ap.add_argument("--seed", type=int, default=12345, help="seed of every probe")
    ap.add_argument("--limit", type=int, help="read at most this many entries")
    ap.add_argument(
        "--voltage-bounds",
        metavar="LOW,HIGH",
        help="membrane-voltage limits in mV for core.run.voltage",
    )
    ap.add_argument(
        "--max-failures", type=int, default=100, help="failures listed per check"
    )
    ap.add_argument(
        "--quiet", action="store_true", help="write the file, print no table"
    )
    args = ap.parse_args(argv)
    out = args.out or args.folder / "evaluate"
    try:
        doc = run_checks(
            args.folder,
            out,
            packs=[p for p in args.packs.split(",") if p],
            params=_params(args.param),
            sample=args.sample,
            seed=args.seed,
            bounds_mv=_bounds(args.voltage_bounds),
            limit=args.limit,
            max_failures=args.max_failures,
            manifest=args.manifest,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"checks: {exc}", file=sys.stderr)
        return 2
    print(f"ran {len(doc['checks'])} checks -> {out / CHECKS_FILE}")
    if not args.quiet:
        print(format_report(doc))
    return 1 if doc["counts"]["fail"] or doc["counts"]["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
