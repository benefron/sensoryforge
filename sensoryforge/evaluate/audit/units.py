"""Unit resolution (report contract, section 3).

Each quantity's unit is resolved once per data set and recorded with its origin:
``data`` (the data names it), ``user`` (given to the audit), ``pack:<name>`` (a metric
pack states it, with a source) or ``null`` (nobody states it). Units are composed as
symbols and never converted.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

#: Quantities whose unit is resolved per data set.
QUANTITIES: Tuple[str, ...] = ("time", "space", "frame", "drive", "filtered", "state")

#: ``{quantity: (symbol, source)}``.
UnitTable = Dict[str, Tuple[str, str]]


class UnitConflict(ValueError):
    """A user or pack unit disagrees with the unit the data names."""


def data_units(
    block: Optional[Mapping[str, Any]], spacing_key_present: bool, dt_key_present: bool
) -> UnitTable:
    """The units a bundle names itself.

    A ``units`` block in ``config.json`` outranks a key's suffix (``dt_ms`` gives time
    in
    ``ms``, ``spacing_mm`` gives space in ``mm``).

    Args:
        block: The ``units`` block of ``config.json``, or ``None``.
        spacing_key_present: True when ``grid.spacing_mm`` exists.
        dt_key_present: True when ``dt_ms`` exists.

    Returns:
        ``{quantity: (symbol, source)}`` for the quantities the bundle names.
    """
    table: UnitTable = {}
    if dt_key_present:
        table["time"] = ("ms", "data.h5:dt_ms")
    if spacing_key_present:
        table["space"] = ("mm", "config.json:grid.spacing_mm")
    for quantity, symbol in (block or {}).items():
        if quantity in QUANTITIES and isinstance(symbol, str) and symbol:
            table[quantity] = (symbol, f"config.json:units.{quantity}")
    return table


def resolve_units(
    data: Mapping[str, Tuple[str, str]],
    user: Optional[Mapping[str, str]] = None,
    packs: Sequence[Tuple[str, Mapping[str, Tuple[str, str]]]] = (),
) -> Dict[str, Optional[Dict[str, str]]]:
    """Resolve every quantity's unit.

    A ``data`` unit is final. A ``user`` or pack unit for a quantity the data names must
    equal it. For a quantity the data does not name, the user's unit applies first, then
    the first pack that states one, then ``None``.

    Args:
        data: Units the data names, from :func:`data_units`.
        user: Units given to the audit (``--units frame=kPa``), by quantity.
        packs: ``(pack name, {quantity: (symbol, source)})`` in the order the packs were
            given.

    Returns:
        ``{quantity: {"symbol", "origin", "source"} | None}`` for every quantity.

    Raises:
        UnitConflict: If a user or pack unit differs from the data's, naming both.
        ValueError: If ``user`` names a quantity this module does not know.
    """
    user = dict(user or {})
    for quantity in user:
        if quantity not in QUANTITIES:
            raise ValueError(
                f"unknown quantity {quantity!r} in user units; expected one of "
                f"{list(QUANTITIES)}"
            )
    out: Dict[str, Optional[Dict[str, str]]] = {}
    for quantity in QUANTITIES:
        stated = data.get(quantity)
        claims = []
        if quantity in user:
            claims.append((user[quantity], "user", "--units"))
        for name, table in packs:
            if quantity in table:
                symbol, source = table[quantity]
                claims.append((symbol, f"pack:{name}", source))
        if stated is not None:
            for symbol, origin, source in claims:
                if symbol != stated[0]:
                    raise UnitConflict(
                        f"unit of {quantity!r}: the data names {stated[0]!r} "
                        f"({stated[1]}) but {origin} gives {symbol!r} ({source})"
                    )
            out[quantity] = {"symbol": stated[0], "origin": "data", "source": stated[1]}
        elif claims:
            symbol, origin, source = claims[0]
            out[quantity] = {"symbol": symbol, "origin": origin, "source": source}
        else:
            out[quantity] = None
    return out


def compose(numerator: Optional[str], denominator: Optional[str]) -> Optional[str]:
    """Compose ``numerator/denominator`` as a symbol; ``None`` if a part is ``None``.

    Args:
        numerator: A unit symbol such as ``"mm"``, or ``"1"`` for dimensionless.
        denominator: A unit symbol such as ``"ms"``.

    Returns:
        ``"mm/ms"`` or ``"1/ms"``; just the numerator when the denominator is ``"1"``.
    """
    if numerator is None or denominator is None:
        return None
    return numerator if denominator == "1" else f"{numerator}/{denominator}"
