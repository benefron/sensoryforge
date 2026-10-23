"""Per-metric deltas between two audits of the same data set (roadmap A10).

The audit's per-entry rows (``audit_entries.jsonl``) of the baseline and the new run are
paired by entry and by metric key ``(id, channel, population)``. For each key: how many
pairs, how many changed beyond the tolerance, the mean of each side and the largest
change and the entry that holds it. A number changed when it moved by more than
``atol + rtol * |baseline|`` (the golden tolerance by default); a string or integer
count changed when it is not equal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

Key = Tuple[str, Optional[int], Optional[str]]


def read_rows(path: Path) -> Dict[str, Dict[Key, Any]]:
    """An audit's per-entry values by entry id.

    Args:
        path: ``audit_entries.jsonl``.

    Returns:
        ``{entry: {(metric id, channel, population): value}}``.

    Raises:
        ValueError: If a line is not JSON.
    """
    out: Dict[str, Dict[Key, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path} line {n}: {exc}") from exc
            out[row["entry"]] = {
                (v["id"], v.get("channel"), v.get("population")): v["value"]
                for v in row["values"]
            }
    return out


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def metric_deltas(
    a: Dict[str, Dict[Key, Any]],
    b: Dict[str, Dict[Key, Any]],
    order: Sequence[str],
    atol: float,
    rtol: float,
) -> List[Dict[str, Any]]:
    """Compare two audits' values, metric by metric.

    Args:
        a: The baseline's values (:func:`read_rows`).
        b: The new run's values.
        order: Metric ids in catalogue order (the baseline audit's ``metrics``); keys
            are listed in this order, then by channel and population.
        atol: Absolute tolerance of a number.
        rtol: Relative tolerance of a number.

    Returns:
        One dict per key: ``id``, ``channel``, ``population``, ``n_pairs`` (entries that
        hold the key in both audits), ``n_changed``, ``mean_a`` and ``mean_b`` (over the
        pairs, ``None`` for non-numbers), ``max_abs_delta`` and ``max_entry`` (the
        largest change among numbers; ``None`` if there is none).
    """
    rank = {m: i for i, m in enumerate(order)}
    keys = sorted(
        {k for row in a.values() for k in row} | {k for row in b.values() for k in row},
        key=lambda k: (
            rank.get(k[0], len(rank)),
            k[0],
            -1 if k[1] is None else k[1],
            k[2] or "",
        ),
    )
    out = []
    shared = sorted(set(a) & set(b))
    for key in keys:
        n = changed = 0
        xs: List[float] = []
        ys: List[float] = []
        worst: Optional[float] = None
        worst_entry: Optional[str] = None
        for entry in shared:
            ra, rb = a[entry], b[entry]
            if key not in ra and key not in rb:
                continue
            n += 1
            va, vb = ra.get(key), rb.get(key)
            na, nb = _number(va), _number(vb)
            if na is not None and nb is not None:
                xs.append(na)
                ys.append(nb)
                delta = abs(nb - na)
                if delta > atol + rtol * abs(na):
                    changed += 1
                    if worst is None or delta > worst:
                        worst, worst_entry = delta, entry
            elif va != vb:
                changed += 1
        out.append(
            {
                "id": key[0],
                "channel": key[1],
                "population": key[2],
                "n_pairs": n,
                "n_changed": changed,
                "mean_a": sum(xs) / len(xs) if xs else None,
                "mean_b": sum(ys) / len(ys) if ys else None,
                "max_abs_delta": worst,
                "max_entry": worst_entry,
            }
        )
    return out
