"""``python -m sensoryforge.evaluate.audit DATA_DIR`` writes ``audit.json``.

The audit needs only the data-set folder. Exit status 0 on success, 2 on a usage or data
error (message on stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sensoryforge.evaluate.audit.run import AUDIT_FILE, run_audit
from sensoryforge.evaluate.audit.units import UnitConflict


def _parse_pairs(items: List[str], what: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise ValueError(f"{what} expects KEY=VALUE, got {item!r}")
        out[key] = value
    return out


def _parse_params(items: List[str]) -> Dict[str, Dict[str, Any]]:
    params: Dict[str, Dict[str, Any]] = {}
    for dotted, text in _parse_pairs(items, "--param").items():
        pack, sep, name = dotted.partition(".")
        if not sep or not name:
            raise ValueError(f"--param expects PACK.NAME=VALUE, got {dotted}={text}")
        try:
            value: Any = json.loads(text)
        except json.JSONDecodeError:
            value = text
        params.setdefault(pack, {})[name] = value
    return params


def main(argv: Optional[List[str]] = None) -> int:
    """Run the audit from the command line.

    Args:
        argv: Arguments; default ``sys.argv[1:]``.

    Returns:
        The exit status.
    """
    ap = argparse.ArgumentParser(
        prog="python -m sensoryforge.evaluate.audit",
        description="Measure a data set from its generated data alone.",
    )
    ap.add_argument(
        "folder", type=Path, help="data-set folder (manifest, index, bundles)"
    )
    ap.add_argument("--out", type=Path, help="output folder (default: FOLDER/evaluate)")
    ap.add_argument(
        "--manifest", type=Path, help="manifest file (default: FOLDER/manifest.jsonl)"
    )
    ap.add_argument(
        "--packs", default="core", help="comma-separated metric packs (default core)"
    )
    ap.add_argument(
        "--units",
        action="append",
        default=[],
        metavar="QUANTITY=UNIT",
        help="a unit the data does not name, e.g. frame=kPa (repeatable)",
    )
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="PACK.NAME=VALUE",
        help="a pack parameter, e.g. core.zero_tolerance=0.01 (repeatable)",
    )
    args = ap.parse_args(argv)
    try:
        audit = run_audit(
            args.folder,
            args.out or args.folder / "evaluate",
            packs=[p for p in args.packs.split(",") if p],
            params=_parse_params(args.param),
            user_units=_parse_pairs(args.units, "--units"),
            manifest=args.manifest,
        )
    except (FileNotFoundError, ValueError, KeyError, UnitConflict) as exc:
        print(f"audit: {exc}", file=sys.stderr)
        return 2
    n = audit["entries"]
    print(
        f"audited {n['n']} entries ({n['n_failed']} failed) "
        f"-> {(args.out or args.folder / 'evaluate') / AUDIT_FILE}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
