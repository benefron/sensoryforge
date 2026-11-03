"""Counts and metadata of a data set, measured without loading any array.

``inventory`` walks every entry once through the reader's ``peek`` (shapes and
attributes only) and collects what the front of ``audit.json`` needs: the entries and
how they are grouped, the frame step and lengths, the shared sensor layout, channels and
populations, the resolved units, and the entries that failed (in the batch or at read).
It says nothing about draws: how many distinct stimuli a data set holds is a measured
quantity (the
``core.frames.digest`` metric), not a count of what was declared.
"""

from __future__ import annotations

import datetime
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sensoryforge.evaluate.audit.layout import Layout
from sensoryforge.evaluate.audit.reader import (
    BundleInfo,
    DatasetReader,
    EntryRow,
    ReadError,
)

FORMAT = "sensoryforge-audit/1"


@dataclass
class Inventory:
    """What a data set is, without any metric.

    Attributes:
        reader: The reader that produced it (its ``folder`` and ``manifest``).
        rows: Every manifest entry, sorted by entry id.
        failures: ``{"entry", "stage", "error"}`` per failed entry, sorted by id.
        infos: ``{entry: BundleInfo}`` for the entries that opened.
        layout: The shared sensor layout (``None`` when no entry opened).
        populations: ``(name, readout, n_units)`` shared by all opened entries.
        units: Resolved units by quantity (``None`` when no entry opened).
        index_only: Entry ids the index lists and the manifest does not.
        index_skipped: ``(index file, line number)`` of each index line that was not
            JSON and was skipped.
    """

    reader: DatasetReader
    rows: List[EntryRow]
    failures: List[Dict[str, str]]
    infos: Dict[str, BundleInfo]
    layout: Optional[Layout]
    populations: Tuple[Tuple[str, str, int], ...]
    units: Optional[Dict[str, Optional[Dict[str, str]]]]
    index_only: List[str] = field(default_factory=list)
    index_skipped: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def n_entries(self) -> int:
        """Entries in the manifest."""
        return len(self.rows)

    @property
    def n_failed(self) -> int:
        """Entries that failed in the batch or at read."""
        return len(self.failures)

    @property
    def n_ok(self) -> int:
        """Entries that opened."""
        return self.n_entries - self.n_failed

    def counts(self) -> Dict[str, Any]:
        """Entry counts by split, class and both, with repeat and noise-repeat totals.

        Failed entries are counted: the manifest says they belong to the data set.

        Returns:
            Plain dicts, keys sorted; a missing class is the key ``""``.
        """
        by_split = Counter(r.split for r in self.rows)
        by_class = Counter(r.cls or "" for r in self.rows)
        by_both = Counter((r.split, r.cls or "") for r in self.rows)
        return {
            "n_entries": self.n_entries,
            "by_split": dict(sorted(by_split.items())),
            "by_class": dict(sorted(by_class.items())),
            "by_split_class": {
                s: {c: n for (s2, c), n in sorted(by_both.items()) if s2 == s}
                for s in sorted(by_split)
            },
            "n_repeats": 1 + max((r.repeat for r in self.rows), default=0),
            "n_noise_repeats": 1 + max((r.noise_repeat for r in self.rows), default=0),
        }

    def lengths(self) -> Dict[str, Any]:
        """Frame step and entry lengths over the entries that opened.

        Returns:
            ``dt`` (distinct frame steps, sorted) and ``n_frames`` (``{length:
            entries}``, keys sorted numerically).
        """
        n_frames = Counter(i.n_frames for i in self.infos.values())
        return {
            "dt": sorted({i.dt for i in self.infos.values()}),
            "n_frames": {str(k): n_frames[k] for k in sorted(n_frames)},
        }

    def identifiers(self) -> Dict[str, List[str]]:
        """Opaque ``world`` identifiers in the bundles, by name (no meaning read)."""
        out: Dict[str, set] = {}
        for info in self.infos.values():
            for key in ("world_id", "dataset_id"):
                if key in info.world_ids:
                    out.setdefault(key, set()).add(info.world_ids[key])
        return {k: sorted(v) for k, v in sorted(out.items())}

    def header(
        self,
        packs: Sequence[Mapping[str, str]],
        params: Mapping[str, Mapping[str, Any]],
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """The front of ``audit.json`` that needs no metric (contract, section 5).

        The entry-file reference (``file``, ``sha256``) and the metric sections are
        added by the stage that writes the file.

        Args:
            packs: ``{"name", "version"}`` per metric pack, in the order given.
            params: Each pack's parameters by pack name.
            created_at: UTC timestamp; default now.

        Returns:
            ``format, created_at, producer, inputs, packs, params, layout, populations,
            units, entries (n, n_ok, n_failed), failures``.

        Raises:
            ValueError: If no entry opened, so there is no layout to describe.
        """
        if self.layout is None or self.units is None:
            raise ValueError("no entry could be read, so the data set has no layout")
        import sensoryforge
        from sensoryforge.provenance import source_info

        stamp = created_at or datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return {
            "format": FORMAT,
            "created_at": stamp,
            "producer": {
                "sensoryforge_version": sensoryforge.__version__,
                "sensoryforge_sha": source_info()["sha"],
            },
            "inputs": {
                "batch_dir": str(self.reader.folder),
                "manifest": str(self.reader.manifest),
                "data_sensoryforge_sha": sorted(
                    {i.sensoryforge_sha for i in self.infos.values()}
                ),
                "bundle_schema_versions": sorted(
                    {i.schema_version for i in self.infos.values()}
                ),
            },
            "packs": [dict(p) for p in packs],
            "params": {k: dict(v) for k, v in params.items()},
            "layout": self.layout.summary(),
            "populations": [
                {"name": n, "readout": r, "n_units": u} for n, r, u in self.populations
            ],
            "units": self.units,
            "entries": {
                "n": self.n_entries,
                "n_ok": self.n_ok,
                "n_failed": self.n_failed,
            },
            "failures": list(self.failures),
        }


def inventory(reader: DatasetReader) -> Inventory:
    """Walk every entry once and collect counts and metadata.

    A failure is recorded per entry, never raised: ``batch`` when the index says the
    entry failed (or has no row and no bundle), ``read`` when the bundle is missing,
    unreadable or differs in layout, populations or units from the first one that
    opened.

    Args:
        reader: A reader over the data-set folder.

    Returns:
        The :class:`Inventory`.

    Raises:
        UnitConflict: If a user or pack unit contradicts what the data names.
    """
    failures: List[Dict[str, str]] = []
    infos: Dict[str, BundleInfo] = {}
    for row in reader.rows:
        if row.status is not None and row.status != "ok":
            failures.append(
                {"entry": row.entry, "stage": "batch", "error": row.error or row.status}
            )
            continue
        try:
            infos[row.entry] = reader.peek(row.entry)
        except ReadError as exc:
            failures.append(
                {
                    "entry": row.entry,
                    "stage": "read",
                    "error": f"{exc.code}: {exc.message}" if exc.code else exc.message,
                }
            )
    ref = reader.reference
    return Inventory(
        reader=reader,
        rows=reader.rows,
        failures=failures,
        infos=infos,
        layout=None if ref is None else ref.layout,
        populations=() if ref is None else ref.populations,
        units=None if ref is None else reader.units,
        index_only=reader.index_only,
        index_skipped=reader.index_skipped,
    )
