"""The export command's arguments, free of heavy imports so the CLI stays fast."""

from __future__ import annotations

import argparse
from pathlib import Path

from sensoryforge.evaluate.export.video import FORMATS


def add_arguments(ap: argparse.ArgumentParser) -> None:
    """Add the export's arguments to ``ap`` (shared with the CLI's export command)."""
    ap.add_argument("folder", type=Path, help="the data-set folder")
    ap.add_argument("out", type=Path, help="the folder to write (new or empty)")
    ap.add_argument(
        "--stages",
        type=Path,
        help="where audit.json, declared.json and compare.json live "
        "(default: FOLDER/evaluate)",
    )
    ap.add_argument(
        "--draws",
        help="comma-separated entry ids to include in full "
        "(default: --n-draws draws spread over the manifest)",
    )
    ap.add_argument(
        "--n-draws", type=int, default=3, help="draws to take when --draws is not given"
    )
    ap.add_argument(
        "--sensor",
        type=int,
        action="append",
        help="a sensor index whose stimulus trace is recorded (repeatable; "
        "default: the middle sensor)",
    )
    ap.add_argument(
        "--unit",
        action="append",
        metavar="POPULATION:INDEX",
        help="a unit whose traces are recorded (repeatable; default: each "
        "population's middle unit)",
    )
    ap.add_argument(
        "--no-pins", action="store_true", help="record no traces for single points"
    )
    ap.add_argument(
        "--max-list-rows", type=int, default=1000, help="rows kept of any draw list"
    )
    ap.add_argument("--max-mb", type=float, default=500.0, help="refuse to record more")
    ap.add_argument(
        "--video",
        choices=FORMATS,
        help="also write a movie of each chosen draw (needs ffmpeg)",
    )
    ap.add_argument(
        "--video-speed", type=float, default=1.0, help="movie speed (1 is real time)"
    )
