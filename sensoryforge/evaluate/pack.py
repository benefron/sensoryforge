"""The metric-pack interface (report contract, section 10).

A pack measures one entry at a time from an :class:`EntryView` and may add data-set
level values from the compact rows the audit keeps. It follows the component pattern
(``from_config``/``to_dict``/``get_param_spec``). Inside the audit's import wall: it
never imports ``sensoryforge.world``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: ``(metric id, channel or None, population or None)`` keys a value row by.
ValueKey = Tuple[str, Optional[int], Optional[str]]


@dataclass(frozen=True)
class DatasetRow:
    """What the audit keeps of one measured entry for ``measure_dataset``.

    Attributes:
        entry: Entry id.
        split: Split label.
        cls: Class label or ``None``.
        repeat: Repeat number.
        noise_repeat: Noise-repeat number.
        values: The pack's ``dataset_inputs`` values by :data:`ValueKey`.
    """

    entry: str
    split: str
    cls: Optional[str]
    repeat: int
    noise_repeat: int
    values: Dict[ValueKey, Any] = field(default_factory=dict)


def metric_definition(
    metric_id: str,
    unit: Optional[Dict[str, str]],
    scope: str = "channel",
    value_type: str = "number",
    level: str = "entry",
    resolution: Optional[float] = None,
    applies_to: Optional[Sequence[str]] = None,
    summarised: bool = True,
) -> Dict[str, Any]:
    """One metric's definition as ``audit.json`` lists it (contract, section 4).

    Args:
        metric_id: ``<pack>.<quantity>.<name>``.
        unit: The resolved unit (``{"symbol", "origin", ...}``) or ``None`` when nobody
            states it.
        scope: ``"entry"``, ``"channel"`` or ``"population"``.
        value_type: ``"number"``, ``"integer"`` or ``"string"``.
        level: ``"entry"`` or ``"dataset"``.
        resolution: The quantisation step of a frame-based measurement, else ``None``.
        applies_to: For ``scope="population"``, the readouts the metric applies to; omit
            for all.
        summarised: False for a metric that has no summary (digests).

    Returns:
        The definition as a plain dict, in the contract's key order.
    """
    out: Dict[str, Any] = {
        "id": metric_id,
        "pack": metric_id.split(".", 1)[0],
        "level": level,
        "value_type": value_type,
        "unit": None if unit is None else dict(unit),
        "resolution": resolution,
        "scope": scope,
    }
    if applies_to is not None:
        out["applies_to"] = list(applies_to)
    out["summarised"] = summarised
    return out


class MetricPack:
    """Base class of a metric pack.

    A subclass sets ``name`` and ``version`` and implements :meth:`metrics` and
    :meth:`measure`. Its constructor arguments are its parameters; ``to_dict()`` returns
    them and ``from_config()`` rebuilds the pack.

    Attributes:
        name: Registry name and the first part of every metric id.
        version: Pack version, recorded in ``audit.json``.
        dataset_inputs: Metric ids whose per-entry values :meth:`measure_dataset` needs;
            the audit keeps those values for every entry.
    """

    name: str = ""
    version: str = "1"
    dataset_inputs: Tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]] = None) -> "MetricPack":
        """Build the pack from its parameters.

        Args:
            config: The pack's parameters by name (``None`` for the defaults).

        Returns:
            A pack instance.
        """
        return cls(**dict(config or {}))

    def to_dict(self) -> Dict[str, Any]:
        """The pack's parameters, recorded under ``params`` in ``audit.json``."""
        return {}

    @classmethod
    def get_param_spec(cls) -> List[Any]:
        """The pack's parameters as ``ParamSpec`` objects."""
        return []

    def units(self) -> Dict[str, Tuple[str, str]]:
        """Units the pack states: ``{quantity: (symbol, source)}`` (contract, 3).

        Returns:
            Empty unless the pack knows a unit and can name its source.
        """
        return {}

    def metrics(
        self, layout: Any, units: Mapping[str, Any], dt: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """The pack's metric definitions, in catalogue order.

        Args:
            layout: The data set's shared :class:`Layout` (some metrics exist once per
                spatial dimension).
            units: The resolved units by quantity.
            dt: The data set's frame step, or ``None`` when its bundles differ.

        Returns:
            Definitions built with :func:`metric_definition`.
        """
        raise NotImplementedError

    def measure(self, view: Any) -> List[Dict[str, Any]]:
        """Measure one entry.

        Args:
            view: The entry's :class:`EntryView`.

        Returns:
            Values ``{"id", "value"[, "channel"][, "population"]}`` in catalogue order.
        """
        raise NotImplementedError

    def measure_dataset(self, rows: Sequence[Any]) -> List[Dict[str, Any]]:
        """Values of the metrics whose level is ``dataset`` (optional).

        Args:
            rows: One compact row per measured entry, sorted by id: ``entry``,
                ``split``,
                ``cls``, ``repeat``, ``noise_repeat`` and ``values``, a dict keyed by
                :data:`ValueKey` holding the pack's ``dataset_inputs``.

        Returns:
            Values ``{"id", "value"}``.
        """
        return []
