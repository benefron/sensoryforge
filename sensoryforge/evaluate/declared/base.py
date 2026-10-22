"""The declared-adapter interface (report contract, sections 8 and 10).

An adapter reads a data set's own declaration (for the first modality, a world file) and
says, per entry, which value each metric id should take. It writes a value only
where the declaration determines it and names the reason otherwise. It follows the
component pattern (``from_config``/``to_dict``/``get_param_spec``) and is registered in
``DECLARED_ADAPTER_REGISTRY``.

An adapter lives outside the audit's import wall: it may import whatever it needs to
read its declaration. The audit never imports an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass
class Declaration:
    """What an adapter declares for one entry.

    Attributes:
        values: ``{"id", "value"[, "channel"]}`` in catalogue order (``channel`` is the
            0-based channel index, only for a data set of more than one channel).
        omitted: ``(metric id, reason)`` for each value the declaration does not
            determine; the stage counts them per metric and reason.
    """

    values: List[Dict[str, Any]] = field(default_factory=list)
    omitted: List[Tuple[str, str]] = field(default_factory=list)


class DeclaredAdapter:
    """Base class of a declared adapter.

    A subclass sets ``name`` and ``version`` and implements :meth:`accepts`,
    :meth:`load`, :meth:`describe`, :meth:`channels`, :meth:`units`, :meth:`metrics`,
    :meth:`declare` and :meth:`bounds`. Its constructor arguments are its parameters;
    ``to_dict()`` returns them and ``from_config()`` rebuilds the adapter.

    Attributes:
        name: Registry name, recorded in ``declared.json``.
        version: Adapter version, recorded in ``declared.json``.
    """

    name: str = ""
    version: str = "1"

    @classmethod
    def from_config(
        cls, config: Optional[Mapping[str, Any]] = None
    ) -> "DeclaredAdapter":
        """Build the adapter from its parameters.

        Args:
            config: The adapter's parameters by name (``None`` for the defaults).

        Returns:
            An adapter instance.
        """
        return cls(**dict(config or {}))

    def to_dict(self) -> Dict[str, Any]:
        """The adapter's parameters, recorded under ``adapter.params``."""
        return {}

    @classmethod
    def get_param_spec(cls) -> List[Any]:
        """The adapter's parameters as ``ParamSpec`` objects."""
        return []

    @classmethod
    def accepts(cls, dataset_meta: Mapping[str, Any]) -> bool:
        """Whether this adapter can read the data set.

        Args:
            dataset_meta: The parsed ``dataset.json``.

        Returns:
            True when the data set carries a declaration this adapter understands.
        """
        raise NotImplementedError

    def load(self, dataset_meta: Mapping[str, Any]) -> Any:
        """Read the declaration out of ``dataset.json`` (the adapter's own object).

        Args:
            dataset_meta: The parsed ``dataset.json``.

        Returns:
            What the other methods take as ``world``.

        Raises:
            ValueError: If the declaration is missing or does not match the data set.
        """
        raise NotImplementedError

    def describe(self, dataset_meta: Mapping[str, Any], world: Any) -> Dict[str, str]:
        """The ``inputs`` object's identifiers.

        Args:
            dataset_meta: The parsed ``dataset.json``.
            world: The object :meth:`load` returned.

        Returns:
            ``{"dataset_id", "world_id", "world_format"}``.
        """
        raise NotImplementedError

    def channels(self, world: Any) -> int:
        """The number of channels the declaration spans (1 when it names none)."""
        return 1

    def units(self, world: Any) -> Dict[str, Optional[Dict[str, str]]]:
        """The unit of each quantity the declaration states.

        Args:
            world: The object :meth:`load` returned.

        Returns:
            ``{quantity: {"symbol", "origin", "source"} | None}`` for every quantity in
            ``audit.units`` (``None`` where the declaration states none).
        """
        raise NotImplementedError

    def metrics(
        self, units: Mapping[str, Optional[Mapping[str, str]]]
    ) -> List[Dict[str, Any]]:
        """The metrics the adapter can declare, in catalogue order.

        Args:
            units: The result of :meth:`units`.

        Returns:
            ``{"id", "unit", "basis", "source"}`` for each (the unit as the audit would
            report it for that id, ``None`` when unstated).
        """
        raise NotImplementedError

    def declare(self, row: Mapping[str, Any], world: Any) -> Declaration:
        """What the declaration implies for one manifest row.

        Args:
            row: A row of ``manifest.jsonl``.
            world: The object :meth:`load` returned.

        Returns:
            The values and the omissions, in catalogue order.
        """
        raise NotImplementedError

    def bounds(self, world: Any) -> List[Dict[str, Any]]:
        """The ranges the declaration allows, for display (never compared).

        Args:
            world: The object :meth:`load` returned.

        Returns:
            ``{"grouping", "label", "id", "lo", "hi", "source"}`` rows.
        """
        return []
