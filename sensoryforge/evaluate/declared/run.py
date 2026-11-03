"""The declared stage: write ``declared.json`` (contract, section 8).

``run_declared`` reads a data set's ``dataset.json`` and ``manifest.jsonl``, asks the
adapter what each entry's declaration implies, and writes ``declared.json`` and
``declared_entries.jsonl``. The data set and group summaries come from the audit's own
summariser, so the two sides compare statistic by statistic. The stage reads no bundle.
Two runs over the same data set are byte-identical apart from ``created_at``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from sensoryforge.evaluate.audit.summary import ValueTable, finite_or_none
from sensoryforge.evaluate.declared.base import DeclaredAdapter
from sensoryforge.evaluate.registry import DECLARED_ADAPTER_REGISTRY
from sensoryforge.evaluate.stage_io import (
    producer,
    timestamp,
    write_json,
    write_rows,
)

FORMAT = "sensoryforge-declared/1"
ENTRIES_FILE = "declared_entries.jsonl"
DECLARED_FILE = "declared.json"
DEFAULT_BINS = 32


def choose_adapter(
    dataset_meta: Mapping[str, Any],
    name: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> DeclaredAdapter:
    """The adapter that reads the data set.

    Args:
        dataset_meta: The parsed ``dataset.json``.
        name: A registered adapter name; default the first registered adapter that
            accepts the data set.
        params: The adapter's parameters.

    Returns:
        An adapter instance.

    Raises:
        ValueError: If a named adapter does not accept the data set, or none does.
        KeyError: If ``name`` is not registered.
    """
    if name is not None:
        cls = DECLARED_ADAPTER_REGISTRY.get_class(name)
        if not cls.accepts(dataset_meta):
            raise ValueError(
                f"declared adapter {name!r} does not accept this data set "
                f"(world format {(dataset_meta.get('world') or {}).get('format')!r})"
            )
        return cls.from_config(params)
    for registered in DECLARED_ADAPTER_REGISTRY.list_registered():
        cls = DECLARED_ADAPTER_REGISTRY.get_class(registered)
        if cls.accepts(dataset_meta):
            return cls.from_config(params)
    raise ValueError(
        "no registered declared adapter accepts this data set; registered: "
        f"{sorted(DECLARED_ADAPTER_REGISTRY.list_registered())}"
    )


def run_declared(
    folder: Path,
    out_dir: Path,
    adapter: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
    manifest: Optional[Path] = None,
    histogram_bins: int = DEFAULT_BINS,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Write ``declared.json`` and ``declared_entries.jsonl`` for a data set.

    Args:
        folder: The data-set folder (``dataset.json`` and ``manifest.jsonl``).
        out_dir: Where to write the two files (created if missing).
        adapter: A registered adapter name; default the first that accepts the data set.
        params: The adapter's parameters.
        manifest: The manifest file; default ``folder / "manifest.jsonl"``.
        histogram_bins: Histogram bins of the summaries (match the audit's).
        created_at: UTC timestamp for ``created_at``; default now.

    Returns:
        The ``declared.json`` content.

    Raises:
        FileNotFoundError: If ``dataset.json`` or the manifest does not exist.
        ValueError: If no adapter accepts the data set, the manifest has a repeated
            entry id, or the declaration does not match the data set.
    """
    folder = Path(folder).resolve()
    meta = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    manifest_path = (
        folder / "manifest.jsonl" if manifest is None else Path(manifest).resolve()
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda r: r["entry"])
    ids = [r["entry"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("the manifest names an entry twice")

    chosen = choose_adapter(meta, adapter, params)
    world = chosen.load(meta)
    units = chosen.units(world)
    catalogue = chosen.metrics(units)
    n_channels = chosen.channels(world)
    keys = [
        (m["id"], c if n_channels > 1 else None, None)
        for m in catalogue
        for c in range(n_channels)
    ]
    table = ValueTable(keys, histogram_bins)
    omitted: Dict[str, Dict[str, int]] = {}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def entry_rows() -> Any:
        for row in rows:
            declaration = chosen.declare(row, world)
            values = [
                {**v, "value": finite_or_none(v["value"])} for v in declaration.values
            ]
            for metric_id, reason in declaration.omitted:
                counts = omitted.setdefault(metric_id, {})
                counts[reason] = counts.get(reason, 0) + 1
            table.add(
                row["split"],
                row.get("class"),
                {(v["id"], v.get("channel"), None): v["value"] for v in values},
            )
            yield {
                "entry": row["entry"],
                "split": row["split"],
                "class": row.get("class"),
                "repeat": row["repeat"],
                "noise_repeat": row["noise_repeat"],
                "values": values,
            }

    n, sha = write_rows(out / ENTRIES_FILE, entry_rows())
    dataset, by_class, by_split = table.groups()
    declared: Dict[str, Any] = {
        "format": FORMAT,
        "created_at": timestamp(created_at),
        "producer": producer(),
        "adapter": {
            "name": chosen.name,
            "version": chosen.version,
            "params": chosen.to_dict(),
        },
        "inputs": {"dataset_dir": str(folder), **chosen.describe(meta, world)},
        "units": units,
        "metrics": catalogue,
        "entries": {"file": ENTRIES_FILE, "n": n, "sha256": sha},
        "omitted": {k: dict(sorted(omitted[k].items())) for k in sorted(omitted)},
        "bounds": _sorted_bounds(chosen.bounds(world)),
        "dataset": dataset,
        "groupings": {"class": by_class, "split": by_split},
    }
    write_json(out / DECLARED_FILE, declared)
    return declared


def _sorted_bounds(bounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(bounds, key=lambda b: (b["grouping"], b["label"], b["id"]))
