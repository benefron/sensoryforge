"""The diff stage: compare two runs of the same data set and write ``diff.json``.

``run_diff`` pairs the entries of a baseline run and a new run, compares every stimulus
and population array under the golden tolerance, finds each entry's first diverging
frame, and (unless told not to) audits both sides to give per-metric deltas. It writes
``diff.json`` and ``diff_entries.jsonl`` (report contract, section 14). Both sides are
read through the audit's blind reader. A change that alters nothing reports zero
differences; the last float32 step another processor rounds differently is *equivalent*,
not different.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from sensoryforge.evaluate.audit.reader import DatasetReader, EntryRow, ReadError
from sensoryforge.evaluate.diff.arrays import DIFF_RTOL
from sensoryforge.evaluate.diff.entries import compare_entry
from sensoryforge.evaluate.diff.metrics import metric_deltas, read_rows
from sensoryforge.evaluate.stage_io import (
    clean,
    producer,
    timestamp,
    write_json,
    write_rows,
)
from sensoryforge.testing.golden import FLOAT32_GOLDEN_ATOL

DIFF_FILE = "diff.json"
DIFF_ENTRIES_FILE = "diff_entries.jsonl"
DIFF_FORMAT = "sensoryforge-diff/1"
TOP_DIVERGENT = 20

#: Entry statuses: the three comparisons, an entry on one side only, and unreadable.
STATUSES = (
    "identical",
    "equivalent",
    "different",
    "only_a",
    "only_b",
    "unreadable",
    "both_failed",
)


def _side(folder: Path) -> Dict[str, Any]:
    """What a side says about itself (``dataset.json``'s ids)."""
    meta: Dict[str, Any] = {}
    path = folder / "dataset.json"
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            meta = {k: raw.get(k) for k in ("dataset_id", "world_id", "seed", "name")}
        except ValueError:
            meta = {}
    return {"dataset_dir": str(folder), **meta}


def _why(row: EntryRow) -> str:
    return f"batch status {row.status or 'unknown'}" + (
        f": {row.error}" if row.error else ""
    )


def _pair(
    entry: str,
    ra: DatasetReader,
    rb: DatasetReader,
    rows_a: Mapping[str, EntryRow],
    rows_b: Mapping[str, EntryRow],
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    """One entry's row of ``diff_entries.jsonl``.

    ``rows_a`` and ``rows_b`` are each side's index rows by entry id, built once by
    :func:`_rows` (rebuilding them per entry made pairing quadratic).
    """
    base = {"entry": entry}
    here, there = rows_a.get(entry), rows_b.get(entry)
    ref = here or there
    base.update(split=ref.split, **{"class": ref.cls})
    if here is None or there is None:
        return {**base, "status": "only_b" if here is None else "only_a"}
    if here.status != "ok" or there.status != "ok":
        wa, wb = _why(here) if here.status != "ok" else None, (
            _why(there) if there.status != "ok" else None
        )
        status = "both_failed" if wa is not None and wa == wb else "unreadable"
        return {**base, "status": status, "baseline": wa, "new": wb}
    try:
        va, vb = ra.read_entry(entry), rb.read_entry(entry)
    except ReadError as exc:
        return {**base, "status": "unreadable", "baseline": None, "new": str(exc)}
    return {**base, **compare_entry(va, vb, atol, rtol)}


def _rows(
    entries: Sequence[str],
    ra: DatasetReader,
    rb: DatasetReader,
    atol: float,
    rtol: float,
) -> Iterator[Dict[str, Any]]:
    rows_a = {r.entry: r for r in ra.rows}
    rows_b = {r.entry: r for r in rb.rows}
    for entry in entries:
        yield clean(_pair(entry, ra, rb, rows_a, rows_b, atol, rtol))


def _first_row(reader: DatasetReader) -> Optional[Dict[str, Any]]:
    for row in reader.rows:
        if row.status == "ok":
            try:
                info = reader.peek(row.entry)
            except (ReadError, KeyError):
                continue
            return {
                "schema_version": info.schema_version,
                "sensoryforge_sha": info.sensoryforge_sha,
            }
    return None


def run_diff(
    baseline: Union[str, Path],
    new: Union[str, Path],
    out_dir: Union[str, Path],
    atol: float = FLOAT32_GOLDEN_ATOL,
    rtol: float = DIFF_RTOL,
    metrics: bool = True,
    packs: Sequence[str] = ("core",),
    audit_baseline: Optional[Union[str, Path]] = None,
    audit_new: Optional[Union[str, Path]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare two runs of a data set and write ``diff.json``.

    Args:
        baseline: The data-set folder of the baseline run (batch output).
        new: The data-set folder of the new run.
        out_dir: Where to write ``diff.json`` and ``diff_entries.jsonl`` (created if
            missing); the two audits go to ``audit_baseline/`` and ``audit_new/`` there.
        atol: Absolute tolerance per value (default: the golden fixtures').
        rtol: Relative tolerance.
        metrics: Audit both sides and give per-metric deltas.
        packs: The metric packs of those audits.
        audit_baseline: A folder that already holds the baseline's ``audit.json`` and
            ``audit_entries.jsonl`` (otherwise the baseline is audited).
        audit_new: The same for the new run.
        created_at: UTC timestamp for ``created_at``; default now.

    Returns:
        The ``diff.json`` content.

    Raises:
        FileNotFoundError: If a side has no manifest or a given audit has no rows.
        ValueError: If a tolerance is negative or an audit cannot be made.
    """
    if atol < 0 or rtol < 0:
        raise ValueError(f"tolerances must be >= 0, got atol={atol}, rtol={rtol}")
    fa, fb = Path(baseline).resolve(), Path(new).resolve()
    ra, rb = DatasetReader(fa), DatasetReader(fb)
    entries = sorted({r.entry for r in ra.rows} | {r.entry for r in rb.rows})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_rows, digest = write_rows(
        out / DIFF_ENTRIES_FILE, _rows(entries, ra, rb, atol, rtol)
    )
    counts = {s: 0 for s in STATUSES}
    by_array: Dict[str, Dict[str, int]] = {}
    divergent: List[Dict[str, Any]] = []
    first: Optional[Tuple[float, int, str]] = None
    with open(out / DIFF_ENTRIES_FILE, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            counts[row["status"]] += 1
            for arr in row.get("arrays", []):
                slot = by_array.setdefault(
                    arr["name"], {"n_equivalent": 0, "n_different": 0}
                )
                slot[
                    "n_different" if arr["status"] == "different" else "n_equivalent"
                ] += 1
            if row["status"] == "different":
                divergent.append(row)
                if row.get("first_step") is not None:
                    key = (row["first_ms"], row["first_step"], row["entry"])
                    first = key if first is None or key < first else first
    divergent.sort(
        key=lambda r: (
            -sum(1 for x in r.get("arrays", []) if x["status"] == "different"),
            -(r.get("largest_abs_diff") or 0.0),
            r["entry"],
        )
    )
    n_different = (
        counts["different"] + counts["only_a"] + counts["only_b"] + counts["unreadable"]
    )
    verdict = (
        "different"
        if n_different
        else "equivalent" if counts["equivalent"] else "identical"
    )
    inputs = {"baseline": _side(fa), "new": _side(fb)}
    for key, reader in (("baseline", ra), ("new", rb)):
        inputs[key].update(_first_row(reader) or {})
    notes = []
    ia, ib = inputs["baseline"].get("dataset_id"), inputs["new"].get("dataset_id")
    if ia and ib and ia != ib:
        notes.append(f"the data sets have different ids ({ia} and {ib})")
    metric_rows: Optional[List[Dict[str, Any]]] = None
    if metrics:
        metric_rows = _metric_deltas(
            fa, fb, out, packs, audit_baseline, audit_new, atol, rtol, created_at
        )
    doc = {
        "format": DIFF_FORMAT,
        "created_at": timestamp(created_at),
        "producer": producer(),
        "inputs": {
            **inputs,
            "atol": atol,
            "rtol": rtol,
            "tolerance_source": (
                "atol: sensoryforge.testing.golden.FLOAT32_GOLDEN_ATOL; "
                "rtol: sensoryforge.evaluate.diff.arrays.DIFF_RTOL"
            ),
            "packs": list(packs) if metrics else None,
        },
        "verdict": verdict,
        "counts": counts,
        "first_divergence": (
            None
            if first is None
            else {"entry": first[2], "step": first[1], "ms": first[0]}
        ),
        "by_array": dict(sorted(by_array.items())),
        "most_divergent": [
            {
                k: r.get(k)
                for k in (
                    "entry",
                    "split",
                    "class",
                    "first_step",
                    "first_ms",
                    "largest_abs_diff",
                )
            }
            for r in divergent[:TOP_DIVERGENT]
        ],
        "metrics": metric_rows,
        "notes": notes,
        "entries": {"file": DIFF_ENTRIES_FILE, "n": n_rows, "sha256": digest},
    }
    write_json(out / DIFF_FILE, clean(doc))
    return doc


def _metric_deltas(
    fa: Path,
    fb: Path,
    out: Path,
    packs: Sequence[str],
    audit_a: Optional[Union[str, Path]],
    audit_b: Optional[Union[str, Path]],
    atol: float,
    rtol: float,
    created_at: Optional[str],
) -> List[Dict[str, Any]]:
    from sensoryforge.evaluate.audit.run import ENTRIES_FILE, run_audit

    sides = []
    for folder, given, name in (
        (fa, audit_a, "audit_baseline"),
        (fb, audit_b, "audit_new"),
    ):
        if given is None:
            doc = run_audit(folder, out / name, packs=packs, created_at=created_at)
            where = out / name
        else:
            where = Path(given)
            doc = json.loads((where / "audit.json").read_text(encoding="utf-8"))
        path = where / ENTRIES_FILE
        if not path.is_file():
            raise FileNotFoundError(f"no {ENTRIES_FILE} in {where}")
        sides.append((doc, read_rows(path)))
    order = [m["id"] for m in sides[0][0].get("metrics", [])]
    return metric_deltas(sides[0][1], sides[1][1], order, atol, rtol)
