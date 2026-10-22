"""The summariser (report contract, section 4): one metric over a set of entries.

The audit and the declared adapter both summarise through :class:`ValueTable`, so
the two sides' summaries can be compared statistic by statistic. A summary holds
``n`` (non-null numeric values), ``n_null``, ``mean``, ``std`` (ddof 0), ``min``, the
quantiles ``q05`` to ``q95`` (linear interpolation) and ``max``, and a histogram of
``histogram_bins`` equal-width bins from ``min`` to ``max`` (one bin when they are
equal).
"""

from __future__ import annotations

import math
from array import array
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sensoryforge.evaluate.pack import ValueKey

_QUANTILES = (("q05", 0.05), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q95", 0.95))
_EMPTY_STATS = ("mean", "std", "min", "q05", "q25", "q50", "q75", "q95", "max")


def finite_or_none(value: Any) -> Any:
    """A number as a plain Python value, with NaN and infinity as ``None``.

    Args:
        value: A Python or numpy number, a string or ``None``.

    Returns:
        ``int``/``float`` (finite), the string, or ``None``.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = float(value)
    return number if math.isfinite(number) else None


def summarise_values(
    values: np.ndarray,
    n_null: int,
    bins: int,
    metric_id: str,
    channel: Optional[int] = None,
    population: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarise one metric over a set of entries.

    Args:
        values: The non-null numeric values ``[n]`` (any dtype; read as float64).
        n_null: Entries of the group whose value is ``null``.
        bins: Histogram bins.
        metric_id: The metric's id.
        channel: The channel index, for a channel-scoped metric of a multi-channel set.
        population: The population name, for a population-scoped metric.

    Returns:
        The summary object of the contract (every statistic ``None`` when ``n`` is 0).
    """
    if bins < 1:
        raise ValueError(f"histogram_bins must be at least 1, got {bins}")
    v = np.asarray(values, dtype=np.float64)
    out: Dict[str, Any] = {"id": metric_id}
    if channel is not None:
        out["channel"] = int(channel)
    if population is not None:
        out["population"] = population
    out["n"] = int(v.size)
    out["n_null"] = int(n_null)
    if v.size == 0:
        out.update({k: None for k in _EMPTY_STATS})
        out["histogram"] = None
        return out
    lo, hi = float(v.min()), float(v.max())
    out["mean"] = float(v.mean())
    out["std"] = float(v.std())
    out["min"] = lo
    for key, q in _QUANTILES:
        out[key] = float(np.quantile(v, q))
    out["max"] = hi
    if lo == hi:
        edges, counts = np.array([lo, hi]), np.array([v.size])
    else:
        counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    out["histogram"] = {
        "edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
    }
    return out


def _group_names(
    labels: Sequence[Optional[str]], include_none: bool = False
) -> List[str]:
    return sorted({x for x in labels if include_none or x is not None})  # type: ignore


class ValueTable:
    """Per-entry values kept as columns, summarised by group at the end.

    Add every entry once, in any order: :meth:`add` for an entry with values,
    :meth:`add_failed` for one without (it is counted in ``n_entries`` and
    ``n_failed`` of its groups and enters no summary). A metric key an entry does not
    carry counts as a ``null`` value for that entry.

    Args:
        keys: The summarised value keys ``(id, channel, population)``, in catalogue
            order.
        bins: Histogram bins.
    """

    def __init__(self, keys: Sequence[ValueKey], bins: int = 32) -> None:
        self.keys: List[ValueKey] = list(keys)
        self.bins = int(bins)
        self._cols: Dict[ValueKey, array] = {k: array("d") for k in self.keys}
        self._split: List[str] = []
        self._cls: List[Optional[str]] = []
        self._failed_split: List[str] = []
        self._failed_cls: List[Optional[str]] = []

    @property
    def n_ok(self) -> int:
        """Entries added with values."""
        return len(self._split)

    def add(
        self,
        split: str,
        cls: Optional[str],
        values: Mapping[ValueKey, Any],
    ) -> None:
        """Add an entry that has values.

        Args:
            split: The entry's split label.
            cls: Its class label or ``None``.
            values: ``{key: value}``; a missing key or a non-number is ``null``.
        """
        self._split.append(split)
        self._cls.append(cls)
        for key, col in self._cols.items():
            v = values.get(key)
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(
                v, bool
            ):
                f = float(v)
                col.append(f if math.isfinite(f) else math.nan)
            else:
                col.append(math.nan)

    def add_failed(self, split: str, cls: Optional[str]) -> None:
        """Add an entry that has no values (it failed)."""
        self._failed_split.append(split)
        self._failed_cls.append(cls)

    def _group(self, mask: np.ndarray, failed: int) -> Dict[str, Any]:
        summaries = []
        for key in self.keys:
            col = np.frombuffer(self._cols[key], dtype=np.float64)[mask]
            ok = ~np.isnan(col)
            summaries.append(
                summarise_values(
                    col[ok],
                    int((~ok).sum()),
                    self.bins,
                    key[0],
                    channel=key[1],
                    population=key[2],
                )
            )
        return {
            "n_entries": int(mask.sum()) + failed,
            "n_failed": failed,
            "summaries": summaries,
        }

    def groups(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Summaries for the whole set, per class and per split.

        An entry without a class is in its split and in the whole set only.

        Returns:
            ``(dataset, by_class, by_split)``, each group ``{n_entries, n_failed,
            summaries}``; the label dicts are sorted by label.
        """
        split = np.array(self._split, dtype=object)
        cls = np.array(self._cls, dtype=object)
        f_split = np.array(self._failed_split, dtype=object)
        f_cls = np.array(self._failed_cls, dtype=object)
        everything = np.ones(len(self._split), dtype=bool)
        dataset = self._group(everything, len(self._failed_split))
        by_class = {
            c: self._group(cls == c, int((f_cls == c).sum()))
            for c in _group_names(self._cls + self._failed_cls)
        }
        by_split = {
            s: self._group(split == s, int((f_split == s).sum()))
            for s in _group_names(self._split + self._failed_split)
        }
        return dataset, by_class, by_split
