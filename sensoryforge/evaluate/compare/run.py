"""The compare stage: write ``compare.json`` (report contract, section 9).

``run_compare`` reads ``audit.json`` and ``declared.json`` (and the two
``_entries.jsonl`` files beside them) and writes ``compare.json`` and
``compare_entries.jsonl``. For every value the declared side carries it records the
measured value, the declared value and the raw difference ``measured - declared``. It
gives no verdict and converts no unit. Entry rows stream in step, so memory grows with
the number of paired values (kept for the distributions), not with the rows.
"""

from __future__ import annotations

import json
import os
from array import array
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from sensoryforge.evaluate.audit.summary import summarise_values
from sensoryforge.evaluate.compare.distribution import distribution_entry
from sensoryforge.evaluate.compare.pairing import (
    COMPUTED,
    PAIRINGS,
    STATS,
    is_number,
    pair_values,
)
from sensoryforge.evaluate.stage_io import (
    file_sha256,
    producer,
    timestamp,
    write_json,
    write_rows,
)

FORMAT = "sensoryforge-compare/1"
AUDIT_FORMAT = "sensoryforge-audit/1"
DECLARED_FORMAT = "sensoryforge-declared/1"
COMPARE_FILE = "compare.json"
ENTRIES_FILE = "compare_entries.jsonl"
DEFAULT_BINS = 32

Key = Tuple[str, Optional[int], Optional[str]]


def _locate(source: Union[str, Path], name: str) -> Path:
    path = Path(source)
    return path / name if path.is_dir() else path


def _load(path: Path, tag: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"no file at {path}")
    content = json.loads(path.read_text(encoding="utf-8"))
    if content.get("format") != tag:
        raise ValueError(f"{path} is {content.get('format')!r}, expected {tag!r}")
    entries = path.parent / content["entries"]["file"]
    if not entries.is_file():
        raise FileNotFoundError(f"{path} names {entries}, which does not exist")
    if file_sha256(entries) != content["entries"]["sha256"]:
        raise ValueError(f"{entries} does not match the sha256 recorded in {path}")
    return content


def _rows(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _key(value: Mapping[str, Any]) -> Key:
    return (value["id"], value.get("channel"), value.get("population"))


def _symbol(unit: Optional[Mapping[str, Any]]) -> Optional[str]:
    return None if unit is None else unit["symbol"]


def _with_labels(base: Dict[str, Any], key: Key) -> Dict[str, Any]:
    if key[1] is not None:
        base["channel"] = int(key[1])
    if key[2] is not None:
        base["population"] = key[2]
    return base


def pair_summaries(
    measured: Optional[Sequence[Mapping[str, Any]]],
    declared: Sequence[Mapping[str, Any]],
    measured_units: Mapping[str, Optional[str]],
    declared_units: Mapping[str, Optional[str]],
) -> List[Dict[str, Any]]:
    """Pair two groups' summaries statistic by statistic.

    Args:
        measured: Summaries of the measured values of the entries the declared summaries
            cover, or ``None`` when the audit has no such group.
        declared: The declared side's summaries of the group.
        measured_units: ``{metric id: unit symbol or None}`` of the audit.
        declared_units: The same for the declared side.

    Returns:
        One ``stat_pair`` per declared summary and statistic, in the declared order.
        A summary the audit lacks pairs as ``missing_measured``.
    """
    index = {_key(s): s for s in measured or []}
    out: List[Dict[str, Any]] = []
    for summary in declared:
        key = _key(summary)
        counterpart = index.get(key)
        for stat in STATS:
            pair = pair_values(
                None if counterpart is None else counterpart.get(stat),
                summary.get(stat),
                measured_units.get(key[0]),
                declared_units.get(key[0]),
            )
            row = _with_labels({"id": key[0]}, key)
            row["stat"] = stat
            row.update(pair)
            out.append(row)
    return out


class _PairColumns:
    """Per metric key, the entries that declare it and their paired values.

    ``raw`` is the measured value of every such entry whether or not a difference was
    computed (NaN when the audit has none), for the group summaries (contract 9.1).
    """

    def __init__(self) -> None:
        self.entry = array("q")
        self.measured = array("d")
        self.declared = array("d")
        self.raw = array("d")

    def add(
        self, entry: int, measured: float, declared: float, raw: float = np.nan
    ) -> None:
        self.entry.append(entry)
        self.measured.append(measured)
        self.declared.append(declared)
        self.raw.append(raw)


def run_compare(
    audit: Union[str, Path],
    declared: Union[str, Path],
    out_dir: Union[str, Path],
    histogram_bins: int = DEFAULT_BINS,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Write ``compare.json`` and ``compare_entries.jsonl``.

    Args:
        audit: ``audit.json`` or the folder that holds it.
        declared: ``declared.json`` or the folder that holds it.
        out_dir: Where to write the two files (created if missing).
        histogram_bins: Bins of the paired histograms in ``distributions``.
        created_at: UTC timestamp for ``created_at``; default now.

    Returns:
        The ``compare.json`` content.

    Raises:
        FileNotFoundError: If an input file does not exist.
        ValueError: If an input has the wrong format tag or does not match its recorded
            hash, or the two files do not describe the same entries.
    """
    if histogram_bins < 1:
        raise ValueError(f"histogram_bins must be at least 1, got {histogram_bins}")
    audit_path = _locate(audit, "audit.json")
    declared_path = _locate(declared, "declared.json")
    a = _load(audit_path, AUDIT_FORMAT)
    d = _load(declared_path, DECLARED_FORMAT)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    a_units = {m["id"]: _symbol(m.get("unit")) for m in a["metrics"]}
    d_units = {m["id"]: _symbol(m.get("unit")) for m in d["metrics"]}
    resolution = {m["id"]: m.get("resolution") for m in a["metrics"]}
    d_order = {m["id"]: i for i, m in enumerate(d["metrics"])}

    entry_meta: List[Tuple[str, Optional[str]]] = []  # (split, class) by entry index
    counts = {name: 0 for name in PAIRINGS}
    n_pairs: Dict[str, int] = {}
    columns: Dict[Key, _PairColumns] = {}

    def entry_rows() -> Iterator[Dict[str, Any]]:
        measured_rows = _rows(audit_path.parent / a["entries"]["file"])
        for index, drow in enumerate(
            _rows(declared_path.parent / d["entries"]["file"])
        ):
            mrow = next(measured_rows, None)
            if mrow is None or mrow["entry"] != drow["entry"]:
                raise ValueError(
                    "audit and declared describe different entries: the declared "
                    f"row {index} is {drow['entry']!r}, the audit's is "
                    f"{None if mrow is None else mrow['entry']!r}"
                )
            measured = {_key(v): v["value"] for v in mrow["values"]}
            entry_meta.append((drow["split"], drow["class"]))
            pairs = []
            for v in drow["values"]:
                key = _key(v)
                pair = pair_values(
                    measured.get(key),
                    v["value"],
                    a_units.get(key[0]),
                    d_units.get(key[0]),
                )
                counts[pair["pairing"]] += 1
                row = _with_labels({"id": key[0]}, key)
                row.update(pair)
                row["resolution"] = resolution.get(key[0])
                row = {k: row[k] for k in _PAIR_ORDER if k in row}
                pairs.append(row)
                col = columns.setdefault(key, _PairColumns())
                value = measured.get(key)
                raw = float(value) if is_number(value) else np.nan
                if pair["pairing"] in COMPUTED:
                    n_pairs[key[0]] = n_pairs.get(key[0], 0) + 1
                    col.add(
                        index, float(pair["measured"]), float(pair["declared"]), raw
                    )
                else:
                    col.add(index, np.nan, np.nan, raw)
            yield {
                "entry": drow["entry"],
                "split": drow["split"],
                "class": drow["class"],
                "pairs": pairs,
            }
        if next(measured_rows, None) is not None:
            raise ValueError(
                "audit and declared describe different entries: the audit has more rows"
            )

    n, sha = write_rows(out / ENTRIES_FILE, entry_rows())

    d_ids = set(d_order)
    a_ids = [m["id"] for m in a["metrics"]]
    metrics = [
        {
            "id": i,
            "unit": None if a_units[i] is None else {"symbol": a_units[i]},
            "resolution": resolution[i],
            "n_pairs": n_pairs.get(i, 0),
        }
        for i in a_ids
        if i in d_ids
    ]
    unpaired = {
        "measured_only": [i for i in a_ids if i not in d_ids],
        "declared_only": [i for i in d_order if i not in set(a_ids)],
    }

    splits = np.array([m[0] for m in entry_meta], dtype=object)
    classes = np.array([m[1] for m in entry_meta], dtype=object)

    def measured_summaries(
        keys: Sequence[Key], selection: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """The audit's summariser over the measured values of the declared set.

        Per key, the values of the entries (of the group) whose declared row carries
        that key, so a measured and a declared statistic summarise the same entries.
        """
        out = []
        for key in keys:
            col = columns.get(key)
            if col is None:
                values = np.zeros(0)
            else:
                values = np.frombuffer(col.raw, dtype=np.float64)
                if selection is not None:
                    values = values[selection[np.frombuffer(col.entry, dtype=np.int64)]]
            gone = np.isnan(values)
            out.append(
                summarise_values(values[~gone], int(gone.sum()), histogram_bins, *key)
            )
        return out

    def group_pairs(
        measured_group: Any,
        declared_group: Mapping[str, Any],
        selection: Optional[np.ndarray],
    ) -> Any:
        keys = [_key(s) for s in declared_group["summaries"]]
        return pair_summaries(
            None if measured_group is None else measured_summaries(keys, selection),
            declared_group["summaries"],
            a_units,
            d_units,
        )

    dataset = group_pairs(a["dataset"], d["dataset"], None)
    groupings = {
        kind: {
            label: group_pairs(
                a["groupings"][kind].get(label),
                group,
                (classes if kind == "class" else splits) == label,
            )
            for label, group in d["groupings"][kind].items()
        }
        for kind in ("class", "split")
    }

    ordered_keys = sorted(
        columns,
        key=lambda k: (d_order.get(k[0], len(d_order)), k[0], k[1] or 0, k[2] or ""),
    )

    def distributions(selection: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        blocks = []
        for key in ordered_keys:
            col = columns[key]
            entry = np.frombuffer(col.entry, dtype=np.int64)
            measured = np.frombuffer(col.measured, dtype=np.float64)
            declared_values = np.frombuffer(col.declared, dtype=np.float64)
            keep = (
                np.ones(entry.size, dtype=bool)
                if selection is None
                else selection[entry]
            )
            if not keep.any():
                continue
            both = keep & ~np.isnan(measured)
            blocks.append(
                distribution_entry(
                    measured[both],
                    declared_values[both],
                    histogram_bins,
                    key[0],
                    key[1],
                    key[2],
                )
            )
        return blocks

    compare: Dict[str, Any] = {
        "format": FORMAT,
        "created_at": timestamp(created_at),
        "producer": producer(),
        "inputs": {
            "audit": {
                "path": os.path.relpath(audit_path, out),
                "sha256": file_sha256(audit_path),
            },
            "declared": {
                "path": os.path.relpath(declared_path, out),
                "sha256": file_sha256(declared_path),
            },
        },
        "difference": "measured - declared",
        "metrics": metrics,
        "unpaired": unpaired,
        "counts": counts,
        "entries": {"file": ENTRIES_FILE, "n": n, "sha256": sha},
        "dataset": dataset,
        "groupings": groupings,
        "distributions": {
            "histogram_bins": histogram_bins,
            "dataset": distributions(None),
            "class": {
                label: distributions(classes == label) for label in groupings["class"]
            },
            "split": {
                label: distributions(splits == label) for label in groupings["split"]
            },
        },
    }
    write_json(out / COMPARE_FILE, compare)
    return compare


_PAIR_ORDER = (
    "id",
    "channel",
    "population",
    "measured",
    "declared",
    "diff",
    "rel_diff",
    "resolution",
    "pairing",
)
