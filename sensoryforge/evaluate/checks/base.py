"""The reference-free checks' interface (report contract, section 13).

A check tests the simulator, not the world: it needs no declaration of what a draw
should show. A **check pack** (``core``, ``tactile``, a plugin's) is a list of
:class:`Check` objects. Each check runs against a :class:`CheckContext`, which holds the
data-set folder, the blind reader, the checks' parameters and lazily the bundle's own
configuration, and returns a :class:`CheckResult`.

The checks stage is outside the audit's import wall: ``world_checks`` imports the world
package and the engine checks import the engine. The audit never imports this package.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sensoryforge.evaluate.audit.reader import DatasetReader, ReadError

#: Status values of a check (``skipped`` means it could not be tried, with the reason).
STATUSES: Tuple[str, ...] = ("pass", "fail", "skipped", "error")


class CheckSkipped(Exception):
    """A check that cannot be tried on this data set; the message is the reason."""


@dataclass
class Failure:
    """One thing a check found wrong.

    Attributes:
        where: An entry id, or the name of a probe (``mirror_x``, ``level 3``).
        detail: What is wrong, with the numbers.
        entry: The entry id when ``where`` is an entry, else ``None`` (the page links
            an entry to its draw).
    """

    where: str
    detail: str
    entry: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """The failure as ``checks.json`` writes it."""
        out: Dict[str, Any] = {"where": self.where, "detail": self.detail}
        if self.entry is not None:
            out["entry"] = self.entry
        return out


@dataclass
class CheckResult:
    """What one check found.

    Attributes:
        status: ``pass``, ``fail``, ``skipped`` or ``error``.
        tested: How many things it tried (entries, probes or populations).
        failed: How many of them failed (exact, even when ``failures`` is capped).
        failures: The first failures, each naming where.
        numbers: Raw numbers behind the verdict (counts, largest differences).
        note: A sentence of context, or the reason for a skip or an error.
    """

    status: str
    tested: int = 0
    failed: int = 0
    failures: List[Failure] = field(default_factory=list)
    numbers: Dict[str, Any] = field(default_factory=dict)
    note: Optional[str] = None


class Check:
    """One reference-free check.

    Subclasses set the class attributes and implement :meth:`run`. The id is
    ``<pack>.<area>.<name>``.

    Attributes:
        id: The check's id.
        pack: The pack it belongs to.
        title: A short title for the page.
        what: One or two sentences: what is tried and what passes.
        scope: ``entry`` (one verdict per entry), ``probe`` (a synthetic stimulus
            through the engine) or ``dataset`` (one verdict for the data set).
    """

    id: str = ""
    pack: str = ""
    title: str = ""
    what: str = ""
    scope: str = "entry"

    def run(self, ctx: "CheckContext") -> CheckResult:
        """Run the check.

        Args:
            ctx: The context.

        Returns:
            The result. A check raises :class:`CheckSkipped` when it cannot be tried.
        """
        raise NotImplementedError

    def definition(self) -> Dict[str, str]:
        """The check's description as ``checks.json`` lists it."""
        return {
            "id": self.id,
            "pack": self.pack,
            "title": self.title,
            "what": self.what,
            "scope": self.scope,
        }


class CheckPack:
    """A named list of checks (``core``, ``tactile``, a plugin's).

    Subclasses set ``name`` and implement :meth:`checks`.
    """

    name: str = ""

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None) -> "CheckPack":
        """Build the pack (it takes no parameters today)."""
        return cls(**dict(config or {}))

    def checks(self) -> List[Check]:
        """The pack's checks, in the order the page lists them."""
        raise NotImplementedError


class CheckContext:
    """What the checks share for one data set.

    Args:
        folder: The data-set folder (the batch output).
        sample: How many entries the engine-level per-entry checks re-run, spread evenly
            over the sorted entries that ran.
        seed: The seed of every synthetic probe run.
        bounds_mv: ``(low, high)`` membrane-voltage limits for ``core.run.voltage``;
            ``None`` checks only that the voltage is finite.
        limit: Read at most this many entries in the per-entry data checks (``None``
            for all).
        manifest: The manifest file (default ``folder / manifest.jsonl``).
    """

    def __init__(
        self,
        folder: Path,
        sample: int = 8,
        seed: int = 12345,
        bounds_mv: Optional[Tuple[float, float]] = None,
        limit: Optional[int] = None,
        manifest: Optional[Path] = None,
    ) -> None:
        if sample < 1:
            raise ValueError(f"sample must be at least 1, got {sample}")
        if limit is not None and limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        if bounds_mv is not None and not bounds_mv[0] < bounds_mv[1]:
            raise ValueError(f"bounds_mv must be (low, high), got {bounds_mv}")
        self.folder = Path(folder)
        self.sample = int(sample)
        self.seed = int(seed)
        self.bounds_mv = bounds_mv
        self.limit = limit
        self.reader = DatasetReader(self.folder, manifest=manifest)
        self.cache: Dict[Any, Any] = {}

    def params(self) -> Dict[str, Any]:
        """The parameters recorded in ``checks.json``."""
        return {
            "sample": self.sample,
            "seed": self.seed,
            "limit": self.limit,
            "bounds_mv": None if self.bounds_mv is None else list(self.bounds_mv),
        }

    def ran_entries(self) -> List[str]:
        """Entries the batch index marks ``ok``, sorted by id."""
        return [r.entry for r in self.reader.rows if r.status == "ok"]

    def sampled_entries(self) -> List[str]:
        """``sample`` entries spread evenly over :meth:`ran_entries` (all if fewer)."""
        ran = self.ran_entries()
        if len(ran) <= self.sample:
            return ran
        picks = np.unique(np.linspace(0, len(ran) - 1, self.sample).round().astype(int))
        return [ran[i] for i in picks]

    def bundle_config(self, entry: Optional[str] = None) -> Dict[str, Any]:
        """The ``config.json`` of an entry's bundle (default: the first that ran).

        Raises:
            CheckSkipped: If no bundle ran or its ``config.json`` cannot be read.
        """
        entry = entry or (self.ran_entries() or [None])[0]
        if entry is None:
            raise CheckSkipped("no entry ran, so there is no bundle to read")
        key = ("config", entry)
        if key not in self.cache:
            try:
                path = self.reader.bundle_path(entry) / "config.json"
                self.cache[key] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, KeyError, ReadError) as exc:
                raise CheckSkipped(f"{entry}: config.json cannot be read ({exc})")
        return copy.deepcopy(self.cache[key])

    def peak(self) -> float:
        """The largest ``|frame value|`` over the sampled entries (the probes' scale).

        Raises:
            CheckSkipped: If the sampled frames are all zero or cannot be read.
        """
        if "peak" not in self.cache:
            best = 0.0
            for entry in self.sampled_entries():
                try:
                    view = self.reader.read_entry(entry)
                except (ReadError, KeyError):
                    continue
                if view.frames.size:
                    best = max(best, float(np.nanmax(np.abs(view.frames))))
            self.cache["peak"] = best
        if not self.cache["peak"] > 0.0:
            raise CheckSkipped(
                "the sampled entries' frames are all zero, so a probe has no scale"
            )
        return float(self.cache["peak"])


def pack_failures(
    failures: Sequence[Failure], max_failures: int
) -> Tuple[List[Failure], int]:
    """The first ``max_failures`` failures and the exact total."""
    return list(failures[:max_failures]), len(failures)
