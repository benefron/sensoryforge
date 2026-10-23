"""Comparing two arrays the way a golden fixture is compared (roadmap A10).

``sensoryforge.testing.golden`` holds the tolerance every golden comparison in
SensoryForge shares (:data:`~sensoryforge.testing.golden.FLOAT32_GOLDEN_ATOL`): a value
differs only when it is farther from the baseline than ``atol + rtol * |baseline|``,
so the last float32 step that another processor rounds differently is not flagged.
That absolute tolerance was set for values below 1; drives and filtered currents reach
tens to hundreds of mA, where one float32 step is larger (3.8e-6 at 40), so the diff
adds a relative term of two float32 steps (:data:`DIFF_RTOL`). Integer arrays (spike
and event counts) are compared exactly. This module applies the rule without
asserting, because a diff reports every array it finds different instead of stopping
at the first.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from sensoryforge.testing.golden import FLOAT32_GOLDEN_ATOL

#: Statuses of one array: bit-equal, equal within the tolerance, or different.
IDENTICAL, EQUIVALENT, DIFFERENT = "identical", "equivalent", "different"

#: Default relative tolerance: two float32 steps (one step is ``2**-23`` of a value).
DIFF_RTOL = 2.0**-22


def compare_arrays(
    a: np.ndarray,
    b: np.ndarray,
    atol: float = FLOAT32_GOLDEN_ATOL,
    rtol: float = DIFF_RTOL,
) -> Dict[str, Any]:
    """Compare a new array with the baseline, along the first (time) axis.

    Args:
        a: The baseline array ``[T, ...]``.
        b: The new array, same layout.
        atol: Absolute tolerance per element (default: the golden fixtures' one).
        rtol: Relative tolerance per element, applied to ``|a|`` (default: two float32
            steps). Neither applies when both arrays hold integers or booleans,
            which are compared exactly.

    Returns:
        ``status`` (``identical``, ``equivalent`` or ``different``), ``n_total`` values
        compared, ``n_differing`` beyond the tolerance, ``max_abs_diff`` (over the
        values both sides hold, ``None`` if there are none), ``first_step`` (the first
        index along axis 0 with a difference beyond the tolerance, else ``None``) and,
        for arrays whose shapes differ, ``shapes`` (``[a.shape, b.shape]``; the values
        are then compared over the common leading part).
    """
    out: Dict[str, Any] = {}
    run_out = None  # where one side's time axis ends, if the lengths differ
    if a.shape != b.shape:
        out["shapes"] = [list(a.shape), list(b.shape)]
        if a.ndim != b.ndim or a.ndim == 0:
            return {
                **out,
                "status": DIFFERENT,
                "n_total": 0,
                "n_differing": 0,
                "max_abs_diff": None,
                "first_step": 0,
            }
        if a.shape[0] != b.shape[0]:
            run_out = min(a.shape[0], b.shape[0])
        common = tuple(slice(0, min(x, y)) for x, y in zip(a.shape, b.shape))
        a, b = a[common], b[common]
    if _exact(a) and _exact(b):
        atol = rtol = 0.0
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    both_nan = np.isnan(a64) & np.isnan(b64)
    with np.errstate(invalid="ignore"):
        diff = np.abs(a64 - b64)
        beyond = diff > (atol + rtol * np.abs(a64))
    beyond |= np.isnan(a64) != np.isnan(b64)
    beyond &= ~both_nan
    finite = np.isfinite(diff)
    n_beyond = int(beyond.sum())
    first: Optional[int] = None
    if n_beyond:
        rows = beyond.reshape(beyond.shape[0], -1).any(axis=1)
        first = int(np.flatnonzero(rows)[0])
    if run_out is not None:
        first = run_out if first is None else min(first, run_out)
    if "shapes" in out:
        status = DIFFERENT
    elif np.array_equal(a, b, equal_nan=True):
        status = IDENTICAL
    else:
        status = DIFFERENT if n_beyond else EQUIVALENT
    out.update(
        status=status,
        n_total=int(a.size),
        n_differing=n_beyond,
        max_abs_diff=float(diff[finite].max()) if finite.any() else None,
        first_step=first,
    )
    return out


def _exact(x: np.ndarray) -> bool:
    return np.asarray(x).dtype.kind in "biu"
