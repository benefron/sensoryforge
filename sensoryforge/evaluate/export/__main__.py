"""``python -m sensoryforge.evaluate.export FOLDER OUT``: write a static snapshot."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

from sensoryforge.evaluate.export.build import ExportOptions, export_snapshot
from sensoryforge.evaluate.export.arguments import add_arguments
from sensoryforge.evaluate.server import DatasetService


def _units(specs: Optional[List[str]]) -> List[Tuple[str, int]]:
    out = []
    for spec in specs or []:
        name, sep, index = spec.rpartition(":")
        if not sep or not name:
            raise ValueError(f"--unit wants POPULATION:INDEX, got {spec!r}")
        try:
            out.append((name, int(index)))
        except ValueError as exc:
            raise ValueError(f"--unit wants POPULATION:INDEX, got {spec!r}") from exc
    return out


def run(args: argparse.Namespace) -> int:
    """Export from parsed arguments.

    Args:
        args: Namespace made with :func:`add_arguments`.

    Returns:
        0 on success, 2 for a folder that is not a data set or a bad option.
    """
    try:
        service = DatasetService(args.folder, args.stages)
        if args.no_pins:
            sensors, units = [], []
        else:
            sensors = args.sensor
            units = _units(args.unit) if args.unit else None
            if units is None and sensors is not None:
                units = []  # asking for sensors alone means no unit traces
        options = ExportOptions(
            draws=[d for d in (args.draws or "").split(",") if d],
            n_draws=args.n_draws,
            sensors=sensors,
            units=units,
            max_list_rows=args.max_list_rows,
            max_mb=args.max_mb,
            video=args.video,
            video_speed=args.video_speed,
        )
        result = export_snapshot(service, args.out, options)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.out}: open {args.out / 'index.html'}")
    print(f"  draws in full: {', '.join(result.draws) or 'none'}")
    for note in result.notes:
        print(f"  note: {note}")
    for video in result.videos:
        print(f"  video: {args.out / video}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Run the export command line."""
    ap = argparse.ArgumentParser(
        prog="python -m sensoryforge.evaluate.export",
        description="Write a snapshot of a data set that opens offline.",
    )
    add_arguments(ap)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
