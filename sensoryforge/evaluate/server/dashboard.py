"""Data-set dashboards (roadmap A7): distributions over the whole data set.

Counts by split, class, kind and status; histograms of entry length (the manifest),
stimulus peak and response rate per population (the audit's per-draw rows); and a
coverage map of the values each declared axis actually took. Every count and every bar
carries a ``query``: a filter that ``/api/draws`` accepts, so the number is one click
from the draws behind it (``bin=<histogram id>:<bin index>`` names a bin; the service
resolves it with :attr:`Dashboard.bins`).

This is a viewer, not the blind audit: the counts and the coverage read the manifest
(each draw's declared record) and the world in ``dataset.json``; the histograms of
measured quantities read only the audit's output.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode

import numpy as np

#: Bins of a histogram of a measured quantity and of a numeric axis.
N_BINS = 10

#: An integer axis with at most this many values is shown value by value.
MAX_INT_VALUES = 12

#: The audit's per-draw metrics the histograms read.
PEAK_METRIC = "core.frames.peak"
RATE_METRIC = "core.response.rate_mean"


@dataclass
class Dashboard:
    """The dashboard payload and, for each histogram, the draws in each bin.

    Attributes:
        payload: The JSON the page draws (see :func:`build_dashboard`).
        bins: Histogram id -> for each bin, the manifest positions (row numbers) of
            the draws in it.
    """

    payload: Dict[str, Any]
    bins: Dict[str, List[List[int]]] = field(default_factory=dict)


def bin_query(hist_id: str, index: int) -> str:
    """The ``/api/draws`` filter that lists one bin's draws."""
    return urlencode({"bin": f"{hist_id}:{index}"})


def _assign(values: np.ndarray, lo: float, hi: float, n: int, log: bool) -> np.ndarray:
    """Bin index of each value in ``[lo, hi]`` cut in ``n`` equal bins; ``-1`` outside.

    Args:
        values: Values, ``NaN`` for missing.
        lo: Lower bound.
        hi: Upper bound (a value equal to it is in the last bin).
        n: Number of bins.
        log: Equal bins of the logarithm (needs ``0 < lo``).

    Returns:
        ``int64`` indices, ``-1`` for a missing or outside value.
    """
    out = np.full(values.shape, -1, dtype=np.int64)
    ok = np.isfinite(values) & (values >= lo) & (values <= hi)
    if log:
        ok &= values > 0
    if not ok.any():
        return out
    v = values[ok]
    if hi == lo:
        out[ok] = 0
        return out
    if log:
        t = (np.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
    else:
        t = (v - lo) / (hi - lo)
    out[ok] = np.minimum((t * n).astype(np.int64), n - 1)
    return out


def _edges(lo: float, hi: float, n: int, log: bool) -> List[float]:
    if hi == lo:
        return [float(lo), float(hi)]
    if log:
        return [
            float(x) for x in np.exp(np.linspace(math.log(lo), math.log(hi), n + 1))
        ]
    return [float(x) for x in np.linspace(lo, hi, n + 1)]


def _rows_by_bin(idx: np.ndarray, rows: np.ndarray, n: int) -> List[List[int]]:
    return [rows[idx == b].tolist() for b in range(n)]


def histogram(
    hist_id: str,
    title: str,
    source: str,
    unit: Optional[str],
    values: np.ndarray,
    n_bins: int = N_BINS,
) -> Tuple[Dict[str, Any], List[List[int]]]:
    """A histogram of one value per draw over the data's own range.

    Args:
        hist_id: The histogram's id (what ``bin=`` names).
        title: What the page calls it.
        source: Where the numbers come from (``manifest`` or ``audit``).
        unit: The values' unit, or ``None``.
        values: One value per manifest row, ``NaN`` where there is none.
        n_bins: Bins (one when every value is equal).

    Returns:
        ``(entry, rows_per_bin)``; ``entry`` has ``edges`` (``n + 1``), ``counts``,
        ``n_missing``, ``queries`` and the data's ``min`` and ``max``.
    """
    finite = np.isfinite(values)
    if not finite.any():
        return (
            {
                "id": hist_id,
                "title": title,
                "source": source,
                "unit": unit,
                "edges": [],
                "counts": [],
                "queries": [],
                "n_missing": int(values.size),
                "min": None,
                "max": None,
            },
            [],
        )
    lo, hi = float(values[finite].min()), float(values[finite].max())
    n = 1 if hi == lo else n_bins
    idx = _assign(values, lo, hi, n, log=False)
    rows = np.arange(values.size)
    per_bin = _rows_by_bin(idx, rows, n)
    return (
        {
            "id": hist_id,
            "title": title,
            "source": source,
            "unit": unit,
            "edges": _edges(lo, hi, n, False),
            "counts": [len(r) for r in per_bin],
            "queries": [bin_query(hist_id, b) for b in range(n)],
            "n_missing": int((~finite).sum()),
            "min": lo,
            "max": hi,
        },
        per_bin,
    )


def _numeric_bounds(spec: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    bounds = spec.get("range")
    if (
        isinstance(bounds, (list, tuple))
        and len(bounds) == 2
        and all(isinstance(b, (int, float)) and not isinstance(b, bool) for b in bounds)
    ):
        return float(bounds[0]), float(bounds[1])
    return None


def coverage_for_axis(
    cls: str,
    axis: str,
    spec: Mapping[str, Any],
    rows: np.ndarray,
    raw: Sequence[Any],
    n_bins: int = N_BINS,
) -> Optional[Tuple[Dict[str, Any], str, List[List[int]]]]:
    """How far the drawn values of one declared axis cover its declared domain.

    Args:
        cls: The class name.
        axis: The axis name (a key of every draw's ``values``).
        spec: The world's normalised axis declaration.
        rows: Manifest positions of the class's draws.
        raw: For each of those draws, the value it took (``None`` where it has none).
        n_bins: Bins for a numeric range.

    Returns:
        ``(entry, hist_id, rows_per_bin)``, or ``None`` for an axis with nothing to bin
        (a constant, a link) or a form the dashboard does not bin.
    """
    hist_id = f"cov:{cls}:{axis}"
    base: Dict[str, Any] = {
        "id": hist_id,
        "class": cls,
        "axis": axis,
        "circular": bool(spec.get("circular")),
    }
    if "values" in spec and isinstance(spec["values"], list):
        labels = [str(v) for v in spec["values"]]
        index = {json.dumps(v): i for i, v in enumerate(spec["values"])}
        idx = np.array([index.get(json.dumps(v), -1) for v in raw], dtype=np.int64)
        n = len(labels)
        form, scale, edges = "categorical", None, None
    elif spec.get("int") and _numeric_bounds(spec) is not None:
        lo, hi = _numeric_bounds(spec)  # type: ignore[misc]
        span = int(hi) - int(lo) + 1
        vals = np.array([np.nan if v is None else float(v) for v in raw])
        if span <= MAX_INT_VALUES:
            n = span
            labels = [str(int(lo) + i) for i in range(n)]
            ints = np.rint(vals)
            inside = np.isfinite(vals) & (ints >= lo) & (ints <= hi)
            idx = np.where(inside, ints - lo, -1).astype(np.int64)
            form, scale, edges = "int", None, None
        else:
            n, form, scale = n_bins, "int", "linear"
            idx = _assign(vals, lo, hi, n, False)
            edges, labels = _edges(lo, hi, n, False), None
    elif spec.get("dist") in ("uniform", "log_uniform") and _numeric_bounds(spec):
        lo, hi = _numeric_bounds(spec)  # type: ignore[misc]
        log = spec["dist"] == "log_uniform" and lo > 0
        vals = np.array([np.nan if v is None else float(v) for v in raw])
        n, form, scale = n_bins, "numeric", "log" if log else "linear"
        idx = _assign(vals, lo, hi, n, log)
        edges, labels = _edges(lo, hi, n, log), None
    else:
        return None
    present = np.array([v is not None for v in raw])
    per_bin = _rows_by_bin(idx, rows, n)
    counts = [len(r) for r in per_bin]
    entry = {
        **base,
        "form": form,
        "scale": scale,
        "edges": edges,
        "labels": labels,
        "counts": counts,
        "queries": [bin_query(hist_id, b) for b in range(n)],
        "n": int(present.sum()),
        "n_outside": int((present & (idx < 0)).sum()),
        "n_empty": int(sum(1 for c in counts if c == 0)),
    }
    if edges is not None:
        entry["lo"], entry["hi"] = edges[0], edges[-1]
    return entry, hist_id, per_bin


def _counts(pairs: Sequence[Tuple[str, str]], key: str) -> List[Dict[str, Any]]:
    """Counts of each ``(label, filter token)`` in *pairs*, in label order."""
    tally: Dict[Tuple[str, str], int] = {}
    for pair in pairs:
        tally[pair] = tally.get(pair, 0) + 1
    return [
        {"label": label, "n": n, "query": urlencode({key: token})}
        for (label, token), n in sorted(tally.items())
    ]


def _read_manifest(path: Path, n_rows: int) -> List[Dict[str, Any]]:
    """The ``draw`` record of each manifest line, in order (``{}`` where unreadable)."""
    out: List[Dict[str, Any]] = []
    with open(path, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                draw = json.loads(line).get("draw")
            except ValueError:
                draw = None
            out.append(draw if isinstance(draw, dict) else {})
    if len(out) != n_rows:
        raise ValueError(f"{path}: {len(out)} lines but the service indexed {n_rows}")
    return out


def _audit_columns(
    entries: Path, position: Mapping[str, int], n_rows: int
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Per-draw stimulus peak (the largest over channels) and rate per population."""
    peak = np.full(n_rows, np.nan)
    rates: Dict[str, np.ndarray] = {}
    with open(entries, "rb") as fh:
        for line in fh:
            try:
                raw = json.loads(line)
            except ValueError:
                continue
            row = position.get(raw.get("entry"))
            if row is None:
                continue
            for v in raw.get("values") or []:
                value = v.get("value")
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if v.get("id") == PEAK_METRIC and v.get("population") is None:
                    if not np.isfinite(peak[row]) or value > peak[row]:
                        peak[row] = value
                elif v.get("id") == RATE_METRIC and v.get("population"):
                    col = rates.setdefault(v["population"], np.full(n_rows, np.nan))
                    col[row] = value
    return peak, rates


def _unit(audit: Optional[Mapping[str, Any]], metric: str) -> Optional[str]:
    for m in (audit or {}).get("metrics") or []:
        if m.get("id") == metric:
            unit = m.get("unit")
            return unit.get("symbol") if isinstance(unit, dict) else None
    return None


def build_dashboard(
    rows: Sequence[Any],
    manifest: Path,
    *,
    world: Optional[Mapping[str, Any]],
    entries: Optional[Path],
    audit: Optional[Mapping[str, Any]],
    no_class: str,
) -> Dashboard:
    """The dashboard of one data set.

    Args:
        rows: The service's :class:`DrawRow` list, in manifest order.
        manifest: ``manifest.jsonl``.
        world: The normalised world (``dataset.json["world"]``), or ``None``.
        entries: The audit's ``audit_entries.jsonl``, or ``None`` if there is none.
        audit: The parsed ``audit.json`` (for units), or ``None``.
        no_class: The filter token for "no class".

    Returns:
        The :class:`Dashboard`. ``payload`` has ``n_draws``, ``counts`` (``split``,
        ``class``, ``kind``, ``status`` and, for sessions, ``session_type``: lists of
        ``{label, n, query}``), ``matrix`` (class by split), ``histograms`` (length,
        stimulus peak, a rate per spiking population), ``coverage`` and ``unavailable``
        (what could not be shown and why).

    Raises:
        ValueError: If the manifest does not have one line per row.
    """
    n = len(rows)
    draws = _read_manifest(manifest, n)
    position = {r.entry: i for i, r in enumerate(rows)}
    kinds = ["session" if d.get("sampling") == "session" else "draw" for d in draws]
    session_types = [d.get("session_type") for d in draws]

    counts: Dict[str, List[Dict[str, Any]]] = {
        "split": _counts([(r.split, r.split) for r in rows], "split"),
        "class": _counts(
            [(r.cls or "no class", r.cls or no_class) for r in rows], "class"
        ),
        "status": _counts(
            [(r.status or "unknown", r.status or "unknown") for r in rows], "status"
        ),
    }
    bins: Dict[str, List[List[int]]] = {}
    histograms: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, str]] = []

    def add_hist(entry: Dict[str, Any], per_bin: List[List[int]]) -> None:
        histograms.append(entry)
        bins[entry["id"]] = per_bin

    # kind and session type are not manifest columns, so they are bins of their own
    for hist_id, title, labels in (
        ("kind", "kind", kinds),
        ("session_type", "session type", session_types),
    ):
        names = sorted({str(x) for x in labels if x is not None})
        if hist_id == "session_type" and not names:
            continue
        per_bin = [[i for i, x in enumerate(labels) if str(x) == nm] for nm in names]
        bins[hist_id] = per_bin
        counts[hist_id] = [
            {"label": nm, "n": len(p), "query": bin_query(hist_id, b)}
            for b, (nm, p) in enumerate(zip(names, per_bin))
        ]

    matrix_classes = [c["label"] for c in counts["class"]]
    matrix_splits = [s["label"] for s in counts["split"]]
    cells = []
    for c_label in matrix_classes:
        token = no_class if c_label == "no class" else c_label
        line = []
        for s_label in matrix_splits:
            k = sum(
                1 for r in rows if r.split == s_label and (r.cls or no_class) == token
            )
            line.append(
                {"n": k, "query": urlencode({"class": token, "split": s_label})}
            )
        cells.append(line)

    length = np.array(
        [
            float(r.duration_ms) if isinstance(r.duration_ms, (int, float)) else np.nan
            for r in rows
        ]
    )
    add_hist(*histogram("length", "Entry length", "manifest", "ms", length))

    audit_present = entries is not None and entries.is_file()
    if audit_present:
        peak, rates = _audit_columns(entries, position, n)  # type: ignore[arg-type]
        add_hist(
            *histogram(
                "peak",
                "Stimulus peak (largest over channels)",
                "audit",
                _unit(audit, PEAK_METRIC),
                peak,
            )
        )
        for name in sorted(rates):
            add_hist(
                *histogram(
                    f"rate:{name}",
                    f"Mean rate, {name}",
                    "audit",
                    _unit(audit, RATE_METRIC),
                    rates[name],
                )
            )
    else:
        unavailable.append(
            {
                "what": "stimulus peak and response rates",
                "why": "no audit_entries.jsonl yet: run the audit stage (the Stages "
                "endpoint or `python -m sensoryforge.evaluate.audit`)",
            }
        )

    coverage: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    classes = (world or {}).get("classes") if isinstance(world, Mapping) else None
    if not isinstance(classes, dict):
        unavailable.append(
            {
                "what": "the coverage map",
                "why": "dataset.json holds no world, so the declared axes are unknown",
            }
        )
        classes = {}
    n_sessions = sum(1 for k in kinds if k == "session")
    for cls_name in sorted(classes):
        axes = (classes[cls_name] or {}).get("axes") or {}
        members = np.array(
            [i for i, r in enumerate(rows) if r.cls == cls_name and kinds[i] == "draw"],
            dtype=np.int64,
        )
        if not members.size:
            continue
        for axis in sorted(axes):
            spec = axes[axis]
            if not isinstance(spec, dict) or "value" in spec or "same_as" in spec:
                continue
            raw = [(draws[i].get("values") or {}).get(axis) for i in members]
            built = coverage_for_axis(cls_name, axis, spec, members, raw)
            if built is None:
                skipped.append(
                    {
                        "class": cls_name,
                        "axis": axis,
                        "why": "its declared form is not a range, an integer range or "
                        "a list of values",
                    }
                )
                continue
            entry, hist_id, per_bin = built
            bins[hist_id] = per_bin
            coverage.append(entry)

    payload = {
        "n_draws": n,
        "counts": counts,
        "matrix": {"classes": matrix_classes, "splits": matrix_splits, "cells": cells},
        "histograms": histograms,
        "coverage": coverage,
        "coverage_skipped": skipped,
        "n_sessions": n_sessions,
        "audit_present": bool(audit_present),
        "unavailable": unavailable,
    }
    return Dashboard(payload, bins)
