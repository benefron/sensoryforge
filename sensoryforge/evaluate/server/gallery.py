"""The anomaly gallery (roadmap A9): the draws most worth a look, ranked.

Random sampling misses rare failures, so every entry is scored on a handful of signals
and the worst are listed first. The signals read only the audit's per-entry rows
(``audit_entries.jsonl``, never a declaration), the batch index's status and, when it
exists, ``checks.json``; the class label is used as a grouping key and nothing more.

| signal | flags an entry whose ... | severity |
|---|---|---|
| `did_not_run` | batch index says it failed or has no row | 1.0 |
| `bad_data` | `checks.json` lists it under a failed `core.data.*` check | 1.0 |
| `silent` | population fires nothing though the stimulus was active (rarely) | 0.8 |
| `saturated` | population's mean count per bin per unit >= `saturation` | 0.7-0.9 |
| `class_mismatch` | frame statistics fit another class better than its own | 0.5-0.9 |
| `outlier_rate` | population's mean rate is a robust-z outlier in its class | 0.3-0.9 |
| `long_quiet` | longest stimulus gap is a robust-z outlier in its class | 0.3-0.9 |

The severity orders entries; it is not a probability. The thresholds are parameters
(:class:`GalleryParams`) recorded in the payload, and the saturation threshold is an
assumption with its origin: a neuron with a 2 ms refractory period cannot spike more
than once in two 1 ms bins (the tactile recipes' setting).

An outlier is a robust z-score ``(x - median) / scale`` among entries of the same class
(or split, for entries without a class), with ``scale = 1.4826 * MAD`` floored at the
metric's resolution (so a class whose members all agree exactly still has a scale) and
at a twentieth of the group's own standard deviation. A class needs
``min_peers`` entries to be judged.

This is a viewer module, not the blind audit.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PEAK = "core.frames.peak"
DT = "core.frames.dt"
ACTIVE = "core.frames.active_fraction"
GAP_MAX = "core.frames.gap_max"
BOUTS = "core.frames.bouts"
RATE = "core.response.rate_mean"
UNIT_FRACTION = "core.response.active_unit_fraction"

#: The frame statistics a class's profile is made of (per channel if several).
PROFILE_METRICS: Tuple[str, ...] = (
    "core.frames.peak",
    "core.frames.mean",
    "core.frames.rms",
    "core.frames.active_fraction",
    "core.frames.bouts",
    "core.frames.footprint_mean",
    "core.frames.spread",
    "core.frames.centroid_x",
    "core.frames.centroid_y",
    "core.frames.centroid_speed",
)

#: What the page says about each signal.
SIGNALS: Dict[str, Dict[str, str]] = {
    "did_not_run": {
        "title": "Did not run",
        "what": "The batch index says the entry failed, or has no row for it.",
    },
    "bad_data": {
        "title": "Bad data",
        "what": "checks.json lists the entry under a failed data check "
        "(NaN or infinity, non-integer counts, an irregular time axis, an unreadable "
        "bundle).",
    },
    "silent": {
        "title": "Silent",
        "what": "A population fired nothing although the stimulus was active.",
    },
    "saturated": {
        "title": "Saturated",
        "what": "A population's mean count per bin per unit reached the saturation "
        "threshold.",
    },
    "class_mismatch": {
        "title": "Class mismatch",
        "what": "The entry's frame statistics are closer to another class's profile "
        "than to its own class's.",
    },
    "outlier_rate": {
        "title": "Rate outlier",
        "what": "A population's mean rate is far from its class's typical rate.",
    },
    "long_quiet": {
        "title": "Unusually long quiet",
        "what": "The stimulus's longest quiet gap is far longer than its class's.",
    },
}


@dataclass(frozen=True)
class GalleryParams:
    """The gallery's thresholds.

    Attributes:
        z: Robust z-score at and above which a value is an outlier.
        min_peers: Entries a class (or split) needs before its outliers are judged.
        saturation: Spikes per bin per unit at and above which a population is
            saturated.
        mismatch_ratio: How much farther an entry must be from its own class's profile
            than from the nearest other class's to count as a mismatch.
        min_class: Entries a class needs to have a profile.
        common_silent: A population silent for more than this fraction of a class's
            active entries is reported once for the class, not once per entry.
    """

    z: float = 3.5
    min_peers: int = 8
    saturation: float = 0.5
    mismatch_ratio: float = 1.5
    min_class: int = 5
    common_silent: float = 0.5

    def __post_init__(self) -> None:
        if not self.z > 0 or not 0 < self.saturation <= 1 or self.mismatch_ratio < 1:
            raise ValueError(f"gallery parameters out of range: {self}")
        if not 0 <= self.common_silent <= 1:
            raise ValueError(f"common_silent must be in [0, 1]: {self}")
        if self.min_peers < 3 or self.min_class < 2:
            raise ValueError(f"min_peers must be >= 3 and min_class >= 2: {self}")


@dataclass
class Reason:
    """One signal that flagged an entry.

    Attributes:
        signal: A key of :data:`SIGNALS`.
        severity: Ordering weight in ``(0, 1]``.
        detail: What was seen, with the numbers.
        population: The population, for a population-level signal.
    """

    signal: str
    severity: float
    detail: str
    population: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """The reason as the API writes it."""
        out: Dict[str, Any] = {
            "signal": self.signal,
            "severity": round(self.severity, 4),
            "detail": self.detail,
        }
        if self.population is not None:
            out["population"] = self.population
        return out


Row = Dict[str, Any]  # one line of audit_entries.jsonl


def _scaled(excess: float, span: float) -> float:
    return min(1.0, max(0.0, excess / span))


def read_entries(path: Path) -> List[Row]:
    """The audit's per-entry rows.

    Args:
        path: ``audit_entries.jsonl``.

    Returns:
        One dict per line (``entry``, ``split``, ``class``, ``values``, ...).

    Raises:
        ValueError: If a line is not JSON.
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError(f"{path} line {n}: {exc}") from exc
    return rows


def _by_key(row: Row) -> Dict[Tuple[str, Optional[int], Optional[str]], Any]:
    return {
        (v["id"], v.get("channel"), v.get("population")): v["value"]
        for v in row["values"]
    }


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _group(row: Row) -> str:
    return row["class"] if row.get("class") else f"split:{row['split']}"


def robust_z(
    values: Sequence[Optional[float]], resolution: Optional[float]
) -> List[Optional[float]]:
    """Robust z-scores of values among peers (``None`` where a value is missing).

    Args:
        values: One group's values.
        resolution: The metric's resolution, a floor for the scale; ``None`` for none.

    Returns:
        ``(x - median) / scale`` for each value; the scale is ``1.4826 * MAD``,
        floored at ``resolution`` and at a twentieth of the group's standard
        deviation.
    """
    have = np.array([v for v in values if v is not None], dtype=np.float64)
    if have.size < 2:
        return [None] * len(values)
    med = float(np.median(have))
    scale = max(
        1.4826 * float(np.median(np.abs(have - med))),
        float(resolution or 0.0),
        0.05 * float(have.std()),
        1e-12,
    )
    return [None if v is None else (v - med) / scale for v in values]


class Gallery:
    """The ranked entries.

    Attributes:
        items: Flagged entries, worst first (each ``{entry, split, class, score,
            reasons}``).
        counts: Entries flagged by each signal.
        n_entries: Entries scored.
        unavailable: Signals that could not be computed, each ``{signal, why}``.
        systemic: Populations silent for most of a class (``group``, ``population``,
            ``n_silent``, ``n_active``), reported once instead of per entry.
        params: The thresholds used.
    """

    def __init__(
        self,
        items: List[Dict[str, Any]],
        counts: Dict[str, int],
        n_entries: int,
        unavailable: List[Dict[str, str]],
        params: GalleryParams,
        systemic: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.items = items
        self.counts = counts
        self.n_entries = n_entries
        self.unavailable = unavailable
        self.systemic = systemic or []
        self.params = params

    def payload(
        self,
        signals: Optional[Sequence[str]] = None,
        split: Optional[Sequence[str]] = None,
        classes: Optional[Sequence[str]] = None,
        limit: int = 24,
    ) -> Dict[str, Any]:
        """The JSON the page draws.

        Args:
            signals: Keep entries flagged by any of these signals (``None``: all;
                an empty list: none).
            split: Keep these splits.
            classes: Keep these classes (``~`` for none).
            limit: How many entries to return, worst first.

        Returns:
            ``{signals, n_entries, n_flagged, total, items, unavailable, systemic,
            params}``;
            ``total`` counts the matching entries, ``items`` the first ``limit``.
        """
        chosen = None if signals is None else set(signals)
        items = []
        for item in self.items:
            if split and item["split"] not in split:
                continue
            if classes and (item["class"] or "~") not in classes:
                continue
            reasons = [
                r for r in item["reasons"] if chosen is None or r["signal"] in chosen
            ]
            if not reasons:
                continue
            items.append({**item, "reasons": reasons})
        return {
            "signals": [
                {"id": k, **SIGNALS[k], "n_entries": self.counts.get(k, 0)}
                for k in SIGNALS
            ],
            "n_entries": self.n_entries,
            "n_flagged": len(self.items),
            "total": len(items),
            "items": items[:limit],
            "unavailable": self.unavailable,
            "systemic": self.systemic,
            "params": asdict(self.params),
        }


def _resolutions(audit: Optional[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    return {m["id"]: m.get("resolution") for m in (audit or {}).get("metrics", [])}


def _population_signals(
    rows: Sequence[Row],
    table: Sequence[Dict[Any, Any]],
    params: GalleryParams,
    add,
) -> List[Dict[str, Any]]:
    """Silent, saturated and rate outliers.

    Returns:
        The *systemic* silences: a population silent for more than
        ``params.common_silent`` of the active entries of a class is a property of the
        class (a population that reads a channel the class never drives), not an
        anomaly of one entry, so it is reported once instead of per entry.
    """
    seen: Dict[Tuple[str, str], List[str]] = {}  # (group, pop) -> active entries
    silent_entries: Dict[Tuple[str, str], List[str]] = {}
    for row, vals in zip(rows, table):
        active = any((_num(v) or 0.0) > 0 for k, v in vals.items() if k[0] == ACTIVE)
        dt = next((_num(v) for k, v in vals.items() if k[0] == DT), None)
        for key, value in vals.items():
            if key[2] is None:
                continue
            metric, _, pop = key
            if metric == UNIT_FRACTION and active and _num(value) is not None:
                seen.setdefault((_group(row), pop), []).append(row["entry"])
                if _num(value) == 0.0:
                    silent_entries.setdefault((_group(row), pop), []).append(
                        row["entry"]
                    )
            if metric == RATE and dt and _num(value) is not None:
                per_bin = float(value) * dt
                if per_bin >= params.saturation:
                    add(
                        row["entry"],
                        Reason(
                            "saturated",
                            0.7 + 0.2 * _scaled(per_bin - params.saturation, 0.5),
                            f"{per_bin:.3g} spikes per bin per unit "
                            f"(threshold {params.saturation:g})",
                            pop,
                        ),
                    )
    systemic = []
    for (group, pop), quiet in sorted(silent_entries.items()):
        active_n = len(seen[(group, pop)])
        if len(quiet) > params.common_silent * active_n:
            systemic.append(
                {
                    "group": group,
                    "population": pop,
                    "n_silent": len(quiet),
                    "n_active": active_n,
                }
            )
            continue
        for entry in quiet:
            add(
                entry,
                Reason(
                    "silent",
                    0.8,
                    f"no unit fired though the stimulus was active "
                    f"({len(quiet)} of {active_n} such entries of its class)",
                    pop,
                ),
            )
    # rate outliers per population within a class
    pops = sorted({k[2] for vals in table for k in vals if k[0] == RATE})
    for pop in pops:
        groups: Dict[str, List[int]] = {}
        for i, row in enumerate(rows):
            groups.setdefault(_group(row), []).append(i)
        for members in groups.values():
            if len(members) < params.min_peers:
                continue
            series = [_num(table[i].get((RATE, None, pop))) for i in members]
            for i, z in zip(members, robust_z(series, None)):
                if z is not None and abs(z) >= params.z:
                    rate = _num(table[i].get((RATE, None, pop)))
                    add(
                        rows[i]["entry"],
                        Reason(
                            "outlier_rate",
                            0.3 + 0.6 * _scaled(abs(z) - params.z, 10.0),
                            f"mean rate {rate:.3g} per ms, robust z {z:+.1f} among "
                            f"{len(members)} entries of its class",
                            pop,
                        ),
                    )
    return systemic


def _longest_gap(vals: Mapping[Any, Any]) -> Optional[float]:
    """An entry's longest stimulus gap over its channels, 0 when it has none.

    ``None`` when the entry has no bout count (the audit measured no frames).
    """
    if not any(k[0] == BOUTS for k in vals):
        return None
    found = [_num(v) for k, v in vals.items() if k[0] == GAP_MAX]
    return max([g for g in found if g is not None], default=0.0)


def _long_quiet(
    rows: Sequence[Row],
    table: Sequence[Dict[Any, Any]],
    resolution: Optional[float],
    params: GalleryParams,
    add,
) -> bool:
    # fewer than two bouts leave no gap_max value: the longest gap is then 0
    gaps = [_longest_gap(vals) for vals in table]
    if all(g is None for g in gaps):
        return False
    groups: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(_group(row), []).append(i)
    for members in groups.values():
        if len(members) < params.min_peers:
            continue
        zs = robust_z([gaps[i] for i in members], resolution)
        for i, z in zip(members, zs):
            if z is not None and z >= params.z:
                add(
                    rows[i]["entry"],
                    Reason(
                        "long_quiet",
                        0.3 + 0.6 * _scaled(z - params.z, 10.0),
                        f"longest stimulus gap {gaps[i]:g} ms, robust z {z:+.1f} among "
                        f"{len(members)} entries of its class",
                    ),
                )
    return True


def _class_profiles(
    rows: Sequence[Row], table: Sequence[Dict[Any, Any]], params: GalleryParams
) -> Tuple[List[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """Features, the data matrix ``[entries, features]`` (NaN = missing), and scales."""
    features = sorted(
        {
            (k[0], k[1])
            for vals in table
            for k in vals
            if k[2] is None and k[0] in PROFILE_METRICS
        },
        key=lambda f: (f[0], -1 if f[1] is None else f[1]),
    )
    if not features:
        return [], None, None
    x = np.full((len(rows), len(features)), np.nan)
    for i, vals in enumerate(table):
        for j, (metric, ch) in enumerate(features):
            v = _num(vals.get((metric, ch, None)))
            if v is not None:
                x[i, j] = v
    keep = [
        j
        for j in range(len(features))
        if np.isfinite(x[:, j]).mean() >= 0.7 and np.nanstd(x[:, j]) > 0
    ]
    return [f"{features[j][0]}@{features[j][1]}" for j in keep], x[:, keep], None


def _class_mismatch(
    rows: Sequence[Row], table: Sequence[Dict[Any, Any]], params: GalleryParams, add
) -> Optional[str]:
    names, x, _ = _class_profiles(rows, table, params)
    labels = [r.get("class") for r in rows]
    classes = sorted(
        c for c in set(labels) if c and labels.count(c) >= params.min_class
    )
    if len(classes) < 2:
        return "needs at least two classes with enough entries"
    if x is None or not names:
        return "no frame statistic varies across the data set"
    idx = {c: [i for i, lab in enumerate(labels) if lab == c] for c in classes}
    with warnings.catch_warnings():
        # a class that lacks a feature has a NaN centre there; it is left out below
        warnings.simplefilter("ignore", RuntimeWarning)
        centres = np.array([np.nanmedian(x[idx[c]], axis=0) for c in classes])
        within = np.array(
            [
                1.4826 * np.nanmedian(np.abs(x[idx[c]] - centres[k]), axis=0)
                for k, c in enumerate(classes)
            ]
        )
        scale = np.maximum(np.nanmedian(within, axis=0), 0.05 * np.nanstd(x, axis=0))
    scale = np.where(np.isfinite(scale), np.maximum(scale, 1e-12), np.nan)
    for i, label in enumerate(labels):
        if label not in idx:
            continue
        dist = np.full(len(classes), np.nan)
        for k in range(len(classes)):
            # the features this entry and this class's centre both have
            have = np.isfinite(x[i]) & np.isfinite(centres[k]) & np.isfinite(scale)
            if have.any():
                dist[k] = np.mean(np.abs(x[i, have] - centres[k, have]) / scale[have])
        own = classes.index(label)
        if not np.isfinite(dist[own]):
            continue
        best = int(np.nanargmin(dist))
        if best == own:
            continue
        ratio = dist[own] / max(dist[best], 1e-9)
        if ratio >= params.mismatch_ratio:
            add(
                rows[i]["entry"],
                Reason(
                    "class_mismatch",
                    0.5 + 0.4 * _scaled(ratio - params.mismatch_ratio, 5.0),
                    f"declared {label!r} (distance {dist[own]:.2f}) but its frame "
                    f"statistics fit {classes[best]!r} better "
                    f"(distance {dist[best]:.2f})",
                ),
            )
    return None


def _score(ordered: Sequence[Reason]) -> float:
    """An entry's rank key: its worst severity, a little more for each further reason.

    A soft signal never reaches 1, so an entry that did not run or holds bad data always
    ranks above one that merely looks unusual.
    """
    score = ordered[0].severity + 0.01 * (len(ordered) - 1)
    return round(score if ordered[0].severity >= 1.0 else min(score, 0.99), 4)


def build_gallery(
    entries: Sequence[Row],
    audit: Optional[Mapping[str, Any]] = None,
    checks: Optional[Mapping[str, Any]] = None,
    status: Optional[Mapping[str, Tuple[Optional[str], Optional[str]]]] = None,
    params: Optional[GalleryParams] = None,
) -> Gallery:
    """Score every entry and rank the flagged ones.

    Args:
        entries: The audit's per-entry rows (:func:`read_entries`).
        audit: ``audit.json`` (for the metrics' resolutions).
        checks: ``checks.json``, or ``None`` (then ``bad_data`` is unavailable).
        status: Entry id -> ``(status, error)`` from the batch index, for every entry of
            the manifest (including ones the audit could not read); ``None`` skips
            ``did_not_run``.
        params: Thresholds (the defaults when omitted).

    Returns:
        The :class:`Gallery`.
    """
    params = params or GalleryParams()
    found: Dict[str, List[Reason]] = {}

    def add(entry: str, reason: Reason) -> None:
        found.setdefault(entry, []).append(reason)

    unavailable: List[Dict[str, str]] = []
    info: Dict[str, Tuple[str, Optional[str]]] = {
        r["entry"]: (r["split"], r.get("class")) for r in entries
    }
    if status is None:
        unavailable.append({"signal": "did_not_run", "why": "no batch index to read"})
    else:
        for entry, (state, error) in status.items():
            if state != "ok":
                add(
                    entry,
                    Reason(
                        "did_not_run",
                        1.0,
                        f"batch status {state or 'unknown'}"
                        + (f": {error}" if error else ""),
                    ),
                )
    if checks is None:
        unavailable.append(
            {"signal": "bad_data", "why": "checks.json has not been computed"}
        )
    else:
        for row in checks.get("checks", []):
            # an unreadable bundle is flagged here even when its batch status is ok,
            # which did_not_run does not see
            if row["id"].startswith("core.data."):
                for f in row["failures"]:
                    if f.get("entry"):
                        add(
                            f["entry"],
                            Reason("bad_data", 1.0, f"{row['id']}: {f['detail']}"),
                        )
    table = [_by_key(r) for r in entries]
    res = _resolutions(audit)
    systemic = _population_signals(entries, table, params, add)
    if not _long_quiet(entries, table, res.get(GAP_MAX), params, add):
        unavailable.append(
            {"signal": "long_quiet", "why": "the audit holds no gap metric"}
        )
    why = _class_mismatch(entries, table, params, add)
    if why:
        unavailable.append({"signal": "class_mismatch", "why": why})
    items = []
    counts: Dict[str, int] = {k: 0 for k in SIGNALS}
    for entry, reasons in found.items():
        # one reason per (signal, population); keep the most severe
        best: Dict[Tuple[str, Optional[str]], Reason] = {}
        for r in reasons:
            k = (r.signal, r.population)
            if k not in best or r.severity > best[k].severity:
                best[k] = r
        ordered = sorted(
            best.values(), key=lambda r: (-r.severity, r.signal, r.population or "")
        )
        for sig in {r.signal for r in ordered}:
            counts[sig] += 1
        split, cls = info.get(entry, ("", None))
        items.append(
            {
                "entry": entry,
                "split": split,
                "class": cls,
                "score": _score(ordered),
                "reasons": [r.to_dict() for r in ordered],
            }
        )
    items.sort(key=lambda it: (-it["score"], it["entry"]))
    return Gallery(items, counts, len(entries), unavailable, params, systemic)
