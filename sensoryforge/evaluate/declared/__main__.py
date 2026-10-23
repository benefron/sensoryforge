"""``python -m sensoryforge.evaluate.declared DATA_DIR`` writes ``declared.json``.

The stage needs the data-set folder's ``dataset.json`` and ``manifest.jsonl`` only; it
reads no bundle. Exit status 0 on success, 2 on a usage or data error (message on
stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sensoryforge.evaluate.declared.run import DECLARED_FILE, run_declared


def main(argv: Optional[List[str]] = None) -> int:
    """Run the declared stage from the command line.

    Args:
        argv: Arguments; default ``sys.argv[1:]``.

    Returns:
        The exit status.
    """
    ap = argparse.ArgumentParser(
        prog="python -m sensoryforge.evaluate.declared",
        description="Write what a data set's declaration says each entry should show.",
    )
    ap.add_argument(
        "folder", type=Path, help="data-set folder (dataset.json, manifest)"
    )
    ap.add_argument("--out", type=Path, help="output folder (default: FOLDER/evaluate)")
    ap.add_argument(
        "--manifest", type=Path, help="manifest file (default: FOLDER/manifest.jsonl)"
    )
    ap.add_argument(
        "--adapter",
        help="registered adapter (default: the first that accepts the data set)",
    )
    ap.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="an adapter parameter (repeatable)",
    )
    ap.add_argument(
        "--histogram-bins", type=int, default=32, help="summary histogram bins"
    )
    args = ap.parse_args(argv)
    try:
        params: Dict[str, Any] = {}
        for item in args.param:
            key, sep, text = item.partition("=")
            if not sep or not key:
                raise ValueError(f"--param expects NAME=VALUE, got {item!r}")
            try:
                params[key] = json.loads(text)
            except json.JSONDecodeError:
                params[key] = text
        out = args.out or args.folder / "evaluate"
        declared = run_declared(
            args.folder,
            out,
            adapter=args.adapter,
            params=params,
            manifest=args.manifest,
            histogram_bins=args.histogram_bins,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"declared: {exc}", file=sys.stderr)
        return 2
    omitted = sum(sum(v.values()) for v in declared["omitted"].values())
    print(
        f"declared {declared['entries']['n']} entries ({omitted} values left out) "
        f"-> {out / DECLARED_FILE}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
