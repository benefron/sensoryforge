"""Pair one measured value with one declared value (report contract, section 9).

The comparison gives a raw difference ``measured - declared`` and says whether one could
be computed (``pairing``). It gives no verdict and converts no unit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from sensoryforge.evaluate.audit.summary import finite_or_none

#: The statistics of a summary that group and data-set rows pair (contract order).
STATS: Tuple[str, ...] = (
    "n",
    "mean",
    "std",
    "min",
    "q05",
    "q25",
    "q50",
    "q75",
    "q95",
    "max",
)

#: Every pairing state, in the order ``counts`` lists them.
PAIRINGS: Tuple[str, ...] = (
    "ok",
    "unit_unchecked",
    "unit_mismatch",
    "missing_measured",
    "missing_declared",
    "not_numeric",
)

#: Pairing states for which a difference is computed.
COMPUTED: Tuple[str, ...] = ("ok", "unit_unchecked")


def is_number(value: Any) -> bool:
    """True for an ``int`` or ``float`` (a ``bool`` is not a number here)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def pair_values(
    measured: Any,
    declared: Any,
    measured_unit: Optional[str],
    declared_unit: Optional[str],
) -> Dict[str, Any]:
    """Pair one measured value with the declared one.

    The first rule that applies decides the pairing: a measured value that is ``None``
    is ``missing_measured`` (also when the declared one is missing too), a declared
    value that is ``None`` is ``missing_declared``, a value that is not a number is
    ``not_numeric``, two known and different units are ``unit_mismatch``, a unit that
    nobody states is ``unit_unchecked``, otherwise ``ok``.

    Args:
        measured: The audit's value, or ``None``.
        declared: The declared value, or ``None``.
        measured_unit: The audit's unit symbol for the metric, or ``None``.
        declared_unit: The declared side's unit symbol, or ``None``.

    Returns:
        ``{measured, declared, diff, rel_diff, pairing}``. ``diff`` is
        ``measured - declared`` (exact for two integers) and ``rel_diff`` is
        ``diff / |declared|``, ``None`` when ``declared`` is 0; both are ``None``
        unless the pairing is ``ok`` or ``unit_unchecked``.
    """
    if measured is None:
        pairing = "missing_measured"
    elif declared is None:
        pairing = "missing_declared"
    elif not (is_number(measured) and is_number(declared)):
        pairing = "not_numeric"
    elif (
        measured_unit is not None
        and declared_unit is not None
        and measured_unit != declared_unit
    ):
        pairing = "unit_mismatch"
    elif measured_unit is None or declared_unit is None:
        pairing = "unit_unchecked"
    else:
        pairing = "ok"
    diff = rel_diff = None
    if pairing in COMPUTED:
        diff = finite_or_none(measured - declared)
        if diff is not None and declared != 0:
            rel_diff = finite_or_none(diff / abs(declared))
    return {
        "measured": measured,
        "declared": declared,
        "diff": diff,
        "rel_diff": rel_diff,
        "pairing": pairing,
    }
