"""The checks stage: run reference-free checks and write ``checks.json``.

``run_checks`` runs the chosen check packs over a data-set folder and writes one file,
``checks.json`` (format tag ``sensoryforge-checks/1``, report contract section 13). It
reads the data through the audit's blind reader and never reads ``audit.json``,
``declared.json`` or ``compare.json``, so it changes none of them. Two runs over the
same data with the same parameters are byte-identical apart from ``created_at``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from sensoryforge.evaluate.audit.reader import ReadError
from sensoryforge.evaluate.checks.base import (
    STATUSES,
    Check,
    CheckContext,
    CheckPack,
    CheckResult,
    CheckSkipped,
)
from sensoryforge.evaluate.checks.probe import ENGINE_ERRORS
from sensoryforge.evaluate.checks.registry import CHECK_PACK_REGISTRY
from sensoryforge.evaluate.stage_io import producer, timestamp, write_json

CHECKS_FILE = "checks.json"
CHECKS_FORMAT = "sensoryforge-checks/1"
DEFAULT_MAX_FAILURES = 100


def build_check_packs(
    names: Sequence[str], params: Optional[Mapping[str, Mapping[str, Any]]] = None
) -> List[CheckPack]:
    """Instantiate the named check packs.

    Args:
        names: Registered pack names, in the order they run.
        params: Each pack's parameters by pack name.

    Returns:
        The packs.

    Raises:
        KeyError: If a name is not registered.
        ValueError: If a name repeats, or parameters are given for a pack not run.
    """
    if len(set(names)) != len(names):
        raise ValueError(f"a check pack is named twice: {list(names)}")
    params = params or {}
    unknown = set(params) - set(names)
    if unknown:
        raise ValueError(f"parameters for packs that are not run: {sorted(unknown)}")
    return [CHECK_PACK_REGISTRY.get_class(n).from_config(params.get(n)) for n in names]


def clean(value: Any) -> Any:
    """JSON-safe copy of a value: numpy scalars become numbers, NaN becomes null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_one(check: Check, ctx: CheckContext) -> CheckResult:
    """Run a check; a skip or an engine or read error is a status, not a crash."""
    try:
        return check.run(ctx)
    except CheckSkipped as exc:
        return CheckResult("skipped", note=str(exc))
    except (ReadError, *ENGINE_ERRORS) as exc:
        return CheckResult("error", note=f"{type(exc).__name__}: {exc}")


def _row(check: Check, result: CheckResult, max_failures: int) -> Dict[str, Any]:
    listed = result.failures[:max_failures]
    row = check.definition()
    row.update(
        {
            "status": result.status,
            "tested": int(result.tested),
            "failed": int(result.failed),
            "failures": [f.to_dict() for f in listed],
            "numbers": clean(result.numbers),
            "note": result.note,
        }
    )
    return row


def run_checks(
    folder: Union[str, Path],
    out_dir: Union[str, Path],
    packs: Sequence[str] = ("core",),
    params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    sample: int = 8,
    seed: int = 12345,
    bounds_mv: Optional[Tuple[float, float]] = None,
    limit: Optional[int] = None,
    max_failures: int = DEFAULT_MAX_FAILURES,
    created_at: Optional[str] = None,
    manifest: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the check packs over a data set and write ``checks.json``.

    Args:
        folder: The data-set folder (the batch output).
        out_dir: Where to write ``checks.json`` (created if missing).
        packs: Registered check packs, in order (default ``core`` alone; the tactile
            pack is named explicitly, like the audit's).
        params: Each pack's parameters by pack name.
        sample: Entries the engine-level per-entry checks re-run.
        seed: Seed of every synthetic probe.
        bounds_mv: Membrane-voltage limits ``(low, high)`` in mV for
            ``core.run.voltage``; none by default.
        limit: Read at most this many entries in the per-entry data checks.
        max_failures: Failures listed per check (the count is always exact).
        created_at: UTC timestamp for ``created_at``; default now.
        manifest: The manifest file (default ``folder / manifest.jsonl``).

    Returns:
        The ``checks.json`` content.

    Raises:
        FileNotFoundError: If the data set has no manifest.
        KeyError: If a pack is not registered.
        ValueError: If a parameter is invalid.
    """
    if max_failures < 0:
        raise ValueError(f"max_failures must be >= 0, got {max_failures}")
    built = build_check_packs(packs, params)
    ctx = CheckContext(
        Path(folder),
        sample=sample,
        seed=seed,
        bounds_mv=bounds_mv,
        limit=limit,
        manifest=manifest,
    )
    rows: List[Dict[str, Any]] = []
    by_entry: Dict[str, List[str]] = {}
    for pack in built:
        for check in pack.checks():
            result = run_one(check, ctx)
            rows.append(_row(check, result, max_failures))
            for failure in result.failures:
                if failure.entry is not None:
                    by_entry.setdefault(failure.entry, [])
                    if check.id not in by_entry[failure.entry]:
                        by_entry[failure.entry].append(check.id)
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
    doc = {
        "format": CHECKS_FORMAT,
        "created_at": timestamp(created_at),
        "producer": producer(),
        "inputs": {
            "dataset_dir": str(folder),
            "packs": list(packs),
            "params": {
                **ctx.params(),
                "max_failures": max_failures,
                "packs": {p.name: p.to_dict() for p in built},
            },
        },
        "counts": counts,
        "checks": rows,
        "entries_with_failures": [
            {"entry": e, "checks": by_entry[e]} for e in sorted(by_entry)
        ],
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / CHECKS_FILE, clean(doc))
    return doc
