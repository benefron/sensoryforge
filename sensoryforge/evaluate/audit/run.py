"""The audit stage: measure a data set and write ``audit.json`` (contract, section 5).

``run_audit`` reads the data-set folder through :class:`DatasetReader`, runs the chosen
metric packs on every entry, summarises each metric over the data set, per class and per
split, and writes ``audit.json`` and ``audit_entries.jsonl``. It never opens a
world file. Two runs over the same data with the same parameters are byte-identical
apart from ``created_at``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from sensoryforge.evaluate.audit.inventory import inventory
from sensoryforge.evaluate.audit.reader import DatasetReader, ReadError
from sensoryforge.evaluate.audit.summary import ValueTable, finite_or_none
from sensoryforge.evaluate.pack import DatasetRow, MetricPack, ValueKey
from sensoryforge.evaluate.registry import METRIC_PACK_REGISTRY

ENTRIES_FILE = "audit_entries.jsonl"
AUDIT_FILE = "audit.json"
DEFAULT_BINS = 32

#: What a pack may raise on an entry it cannot measure; it fails that entry only.
MEASURE_ERRORS = (
    ValueError,
    KeyError,
    IndexError,
    TypeError,
    ArithmeticError,
    RuntimeError,
)


def build_packs(
    names: Sequence[str], params: Optional[Mapping[str, Mapping[str, Any]]] = None
) -> List[MetricPack]:
    """Instantiate the named metric packs.

    Args:
        names: Registered pack names, in the order the audit runs them.
        params: Each pack's parameters by pack name.

    Returns:
        The packs.

    Raises:
        KeyError: If a name is not registered.
        ValueError: If a name repeats, or a pack rejects its parameters.
    """
    if len(set(names)) != len(names):
        raise ValueError(f"a metric pack is named twice: {list(names)}")
    params = params or {}
    unknown = set(params) - set(names)
    if unknown:
        raise ValueError(f"parameters for packs that are not run: {sorted(unknown)}")
    return [METRIC_PACK_REGISTRY.get_class(n).from_config(params.get(n)) for n in names]


def _expand_keys(
    metrics: Sequence[Mapping[str, Any]],
    n_channels: int,
    populations: Sequence[Tuple[str, str, int]],
) -> List[ValueKey]:
    """The summarised value keys, in catalogue order."""
    keys: List[ValueKey] = []
    for m in metrics:
        if m["level"] != "entry" or not m["summarised"]:
            continue
        if m["scope"] == "entry":
            keys.append((m["id"], None, None))
        elif m["scope"] == "channel":
            keys += [
                (m["id"], c if n_channels > 1 else None, None)
                for c in range(n_channels)
            ]
        else:
            applies = m.get("applies_to")
            keys += [
                (m["id"], None, name)
                for name, readout, _n in populations
                if applies is None or readout in applies
            ]
    return keys


def _key(value: Mapping[str, Any]) -> ValueKey:
    return (value["id"], value.get("channel"), value.get("population"))


def run_audit(
    folder: Path,
    out_dir: Path,
    packs: Sequence[str] = ("core",),
    params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    user_units: Optional[Mapping[str, str]] = None,
    manifest: Optional[Path] = None,
    created_at: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Audit a data set and write ``audit.json`` and ``audit_entries.jsonl``.

    Args:
        folder: The data-set folder (``manifest.jsonl``, ``index/``, one bundle per
            entry).
        out_dir: Where to write the two files (created if missing).
        packs: Registered metric packs to run, in order.
        params: Each pack's parameters by pack name.
        user_units: Units given by the user, by quantity (``{"frame": "kPa"}``).
        manifest: The manifest file; default ``folder / "manifest.jsonl"``.
        created_at: UTC timestamp for ``created_at``; default now.
        progress: Called as ``progress(done, total)`` after each entry.

    Returns:
        The ``audit.json`` content.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError: If no entry could be read, a pack is unknown or repeated, or a
            parameter is rejected.
        UnitConflict: If a user or pack unit contradicts what the data names.
    """
    instances = build_packs(packs, params)
    reader = DatasetReader(
        Path(folder).resolve(),
        manifest=None if manifest is None else Path(manifest).resolve(),
        user_units=user_units,
        pack_units=[(p.name, p.units()) for p in instances],
    )
    inv = inventory(reader)
    header = inv.header(
        [{"name": p.name, "version": p.version} for p in instances],
        {p.name: p.to_dict() for p in instances},
        created_at=created_at,
    )
    if inv.layout is None or inv.units is None:  # header() raised already
        raise ValueError("no entry could be read")
    steps = inv.lengths()["dt"]
    definitions: List[Dict[str, Any]] = []
    for pack in instances:
        definitions += pack.metrics(
            inv.layout, inv.units, steps[0] if len(steps) == 1 else None
        )
    ids = [d["id"] for d in definitions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"two metric packs define the same metric id: {sorted(ids)}")
    bins = next(
        (p.histogram_bins for p in instances if hasattr(p, "histogram_bins")),
        DEFAULT_BINS,
    )
    table = ValueTable(
        _expand_keys(definitions, inv.layout.n_channels, inv.populations), bins
    )
    wanted = {i for p in instances for i in p.dataset_inputs}
    failures = {f["entry"]: dict(f) for f in inv.failures}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    kept: List[DatasetRow] = []
    with open(out / ENTRIES_FILE, "wb") as fh:
        for done, row in enumerate(inv.rows, 1):
            values: List[Dict[str, Any]] = []
            if row.entry not in failures:
                try:
                    view = reader.read_entry(row.entry)
                except ReadError as exc:
                    failures[row.entry] = {
                        "entry": row.entry,
                        "stage": "read",
                        "error": (
                            f"{exc.code}: {exc.message}" if exc.code else exc.message
                        ),
                    }
                else:
                    try:
                        for pack in instances:
                            values += pack.measure(view)
                    except MEASURE_ERRORS as exc:
                        values = []
                        failures[row.entry] = {
                            "entry": row.entry,
                            "stage": "measure",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
            if row.entry in failures:
                table.add_failed(row.split, row.cls)
            else:
                by_key = {_key(v): v["value"] for v in values}
                table.add(row.split, row.cls, by_key)
                kept.append(
                    DatasetRow(
                        row.entry,
                        row.split,
                        row.cls,
                        row.repeat,
                        row.noise_repeat,
                        {k: v for k, v in by_key.items() if k[0] in wanted},
                    )
                )
            line = json.dumps(
                {
                    "entry": row.entry,
                    "split": row.split,
                    "class": row.cls,
                    "repeat": row.repeat,
                    "noise_repeat": row.noise_repeat,
                    "bundle": row.bundle,
                    "values": values,
                },
                allow_nan=False,
            )
            data = (line + "\n").encode("utf-8")
            digest.update(data)
            fh.write(data)
            if progress is not None:
                progress(done, inv.n_entries)

    dataset_values: List[Dict[str, Any]] = []
    for pack in instances:
        dataset_values += [
            {"id": v["id"], "value": finite_or_none(v["value"])}
            for v in pack.measure_dataset(kept)
        ]
    dataset, by_class, by_split = table.groups()
    ordered = [failures[k] for k in sorted(failures)]
    header["entries"] = {
        "file": ENTRIES_FILE,
        "n": inv.n_entries,
        "n_ok": inv.n_entries - len(ordered),
        "n_failed": len(ordered),
        "sha256": digest.hexdigest(),
    }
    header["failures"] = ordered
    audit = {
        **{
            k: header[k]
            for k in (
                "format",
                "created_at",
                "producer",
                "inputs",
                "packs",
                "params",
                "layout",
                "populations",
                "units",
            )
        },
        "metrics": definitions,
        "entries": header["entries"],
        "failures": ordered,
        "dataset": dataset,
        "groupings": {"class": by_class, "split": by_split},
        "dataset_values": dataset_values,
    }
    (out / AUDIT_FILE).write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return audit
