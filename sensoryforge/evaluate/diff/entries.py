"""Comparing one entry of two runs: stimulus frames and every population's arrays.

Both sides are read through the audit's blind reader (report contract, section 2), so a
diff sees exactly what an audit sees and never a declaration. Arrays are compared one by
one (:func:`~sensoryforge.evaluate.diff.arrays.compare_arrays`); the entry's *first
divergence* is the earliest frame at which any array differs beyond the tolerance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sensoryforge.evaluate.audit.reader import EntryView
from sensoryforge.evaluate.diff.arrays import (
    DIFFERENT,
    EQUIVALENT,
    IDENTICAL,
    compare_arrays,
)
from sensoryforge.testing.golden import FLOAT32_GOLDEN_ATOL


def entry_arrays(view: EntryView) -> Dict[str, Any]:
    """The arrays of an entry view by name: ``frames``, ``time``, ``<pop>.<array>``."""
    arrays: Dict[str, Any] = {"frames": view.frames, "time": view.time}
    for name, pop in view.populations.items():
        arrays[f"{name}.{pop.readout}"] = pop.data
        arrays[f"{name}.drive"] = pop.drive
        arrays[f"{name}.filtered"] = pop.filtered
    return arrays


def compare_entry(
    a: EntryView,
    b: EntryView,
    atol: float = FLOAT32_GOLDEN_ATOL,
    rtol: float = 0.0,
) -> Dict[str, Any]:
    """Compare one entry of the baseline run with the same entry of the new run.

    Args:
        a: The baseline entry.
        b: The new entry.
        atol: Absolute tolerance (default: the golden fixtures').
        rtol: Relative tolerance.

    Returns:
        ``status`` (``identical``, ``equivalent`` or ``different``), ``n_arrays``,
        ``arrays`` (only those that are not identical, each with its
        :func:`compare_arrays` result under ``name``), ``structure`` (differences in
        frame step, populations and readouts, as sentences), ``first_step`` and
        ``first_ms`` (the earliest divergence; ``None`` when none), and
        ``largest_abs_diff``.
    """
    structure: List[str] = []
    if a.dt != b.dt:
        structure.append(
            f"frame step {a.dt:g} ms in the baseline, {b.dt:g} ms in the new run"
        )
    for name in sorted(set(a.populations) - set(b.populations)):
        structure.append(f"population {name!r} is only in the baseline")
    for name in sorted(set(b.populations) - set(a.populations)):
        structure.append(f"population {name!r} is only in the new run")
    for name in sorted(set(a.populations) & set(b.populations)):
        if a.populations[name].readout != b.populations[name].readout:
            structure.append(
                f"population {name!r} reads out {a.populations[name].readout!r} in the "
                f"baseline, {b.populations[name].readout!r} in the new run"
            )
    left, right = entry_arrays(a), entry_arrays(b)
    shared = [k for k in left if k in right]
    results = []
    for name in shared:
        result = compare_arrays(left[name], right[name], atol, rtol)
        result["name"] = name
        results.append(result)
    differing = [r for r in results if r["status"] != IDENTICAL]
    steps = [r["first_step"] for r in results if r["first_step"] is not None]
    first = min(steps) if steps else None
    if structure or any(r["status"] == DIFFERENT for r in results):
        status = DIFFERENT
    elif differing:
        status = EQUIVALENT
    else:
        status = IDENTICAL
    largest = [r["max_abs_diff"] for r in differing if r["max_abs_diff"] is not None]
    first_ms: Optional[float] = None
    if first is not None and first < len(a.time):
        first_ms = float(a.time[first])
    elif first is not None:
        first_ms = float(first * a.dt)
    return {
        "status": status,
        "n_arrays": len(results),
        "arrays": differing,
        "structure": structure,
        "first_step": first,
        "first_ms": first_ms,
        "largest_abs_diff": max(largest) if largest else (0.0 if results else None),
    }
