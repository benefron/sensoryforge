"""``python -m sensoryforge.evaluate.server FOLDER``: serve a data set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from sensoryforge.evaluate.server import Api, DatasetService, serve


def main(argv: Optional[List[str]] = None) -> int:
    """Run the server.

    Args:
        argv: Arguments; default ``sys.argv[1:]``.

    Returns:
        0 on a clean stop, 2 for a folder that is not a data set.
    """
    ap = argparse.ArgumentParser(
        prog="python -m sensoryforge.evaluate.server",
        description="Serve a data set for the evaluation browser.",
    )
    ap.add_argument("folder", type=Path, help="the data-set folder")
    ap.add_argument(
        "--stages",
        type=Path,
        help="where audit.json, declared.json and compare.json live "
        "(default: FOLDER/evaluate)",
    )
    ap.add_argument("--host", default="127.0.0.1", help="address to bind")
    ap.add_argument("--port", type=int, default=8765, help="port to bind")
    ap.add_argument(
        "--frame-cache-mb",
        type=float,
        default=256.0,
        help="megabytes of decompressed stimulus frames kept between requests "
        "(default 256; 0 keeps none)",
    )
    ap.add_argument(
        "--voltage-cache-mb",
        type=float,
        default=256.0,
        help="megabytes of re-simulated membrane voltage kept between requests "
        "(default 256; 0 keeps none)",
    )
    args = ap.parse_args(argv)
    try:
        service = DatasetService(
            args.folder,
            args.stages,
            frame_cache_bytes=int(args.frame_cache_mb * (1 << 20)),
            voltage_cache_bytes=int(args.voltage_cache_mb * (1 << 20)),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    serve(Api(service), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
