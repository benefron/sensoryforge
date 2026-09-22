"""Parsing for one sweep row's values: a list, or a linear/log range.

The Batch screen's sweep table lets a user type either a comma-separated
list of values (``"10, 20, 40"``) or a ``start, stop, count`` range that is
expanded linearly (:func:`numpy.linspace`) or logarithmically
(:func:`numpy.geomspace`). :func:`parse_field_values` is the one place that
text becomes a value list (or an error string) -- it never raises, so a
malformed row shows its own error instead of taking the screen down (per the
brief: "parsed and validated inline ... show the parse error in the row,
never raise").

The parsed type follows the field: a ``ParamSpec`` with ``dtype == "int"``,
or (when there is no spec) the Python type of the field's current value in
the session, keeps every parsed value an ``int``; everything else is a
``float``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from sensoryforge.gui.session import Session
from sensoryforge.stimuli.base import ParamSpec

#: The row entry modes offered in the sweep table's mode combo.
FIELD_MODES = ("list", "linear", "log")


def infer_is_int(session: Session, path: str, spec: Optional[ParamSpec]) -> bool:
    """Whether values typed for ``path`` should be rounded to ``int``.

    Args:
        session: The session the path is read from when ``spec`` is absent.
        path: The dotted config path (see ``sensoryforge/gui/session.py``).
        spec: The path's ``ParamSpec``, when it came from a registry (see
            ``sweep_paths``); ``None`` for a plain schema field.

    Returns:
        ``True`` when the field is declared (or currently holds) an ``int``;
        ``False`` otherwise (``float`` is the default for any field whose
        current value cannot be read, e.g. ``None``).
    """
    if spec is not None:
        return spec.dtype == "int"
    try:
        current = session.get_by_path(path)
    except ValueError:
        return False
    return isinstance(current, int) and not isinstance(current, bool)


def _coerce(value: float, is_int: bool) -> Any:
    """Round to the nearest ``int`` when the field is integer-typed."""
    return int(round(value)) if is_int else float(value)


def parse_field_values(
    text: str, mode: str, is_int: bool
) -> Tuple[Optional[List[Any]], Optional[str]]:
    """Parse one row's raw text into a value list.

    Args:
        text: What the user typed. For ``mode="list"``, a comma-separated
            list of numbers. For ``"linear"``/``"log"``, exactly three
            comma-separated numbers: ``start, stop, count``.
        mode: One of :data:`FIELD_MODES`.
        is_int: Whether parsed values should be rounded to ``int`` (see
            :func:`infer_is_int`).

    Returns:
        ``(values, error)`` -- exactly one is not ``None``. ``values`` is
        never empty when returned without an error.
    """
    stripped = text.strip()
    if not stripped:
        return None, "enter values"

    if mode == "list":
        parts = [part.strip() for part in stripped.split(",") if part.strip()]
        if not parts:
            return None, "enter values"
        values: List[Any] = []
        for part in parts:
            try:
                values.append(_coerce(float(part), is_int))
            except ValueError:
                return None, f"{part!r} is not a number"
        return values, None

    if mode in ("linear", "log"):
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) != 3:
            return None, "expected start, stop, count"
        try:
            start = float(parts[0])
            stop = float(parts[1])
            count = int(float(parts[2]))
        except ValueError:
            return None, "start, stop, count must be numbers"
        if count < 1:
            return None, "count must be at least 1"
        if mode == "log":
            if start <= 0.0 or stop <= 0.0:
                return None, "a log range needs start and stop > 0"
            raw = np.geomspace(start, stop, count)
        else:
            raw = np.linspace(start, stop, count)
        return [_coerce(float(v), is_int) for v in raw], None

    return None, f"unknown entry mode {mode!r}"
