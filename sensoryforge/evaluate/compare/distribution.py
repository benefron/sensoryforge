"""Distances between a metric's measured and declared values (contract, section 9).

Both distances are computed on the same entries (the pairs whose difference was
computed), so the two samples have equal size and each distance is exact. They are
numbers to read next to the raw difference, not thresholds.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def ks_distance(measured: np.ndarray, declared: np.ndarray) -> float:
    """The Kolmogorov-Smirnov distance between two samples.

    Args:
        measured: Measured values ``[n]``.
        declared: Declared values ``[n]``.

    Returns:
        ``max |F_measured - F_declared|`` over the pooled values, in ``[0, 1]``
        (unitless).

    Raises:
        ValueError: If a sample is empty.
    """
    x = np.sort(np.asarray(measured, dtype=np.float64))
    y = np.sort(np.asarray(declared, dtype=np.float64))
    if x.size == 0 or y.size == 0:
        raise ValueError("the KS distance needs at least one value on each side")
    grid = np.concatenate([x, y])
    fx = np.searchsorted(x, grid, side="right") / x.size
    fy = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(fx - fy)))


def wasserstein_distance(measured: np.ndarray, declared: np.ndarray) -> float:
    """The first Wasserstein distance between two samples of equal size.

    Args:
        measured: Measured values ``[n]``.
        declared: Declared values ``[n]``.

    Returns:
        The mean absolute difference of the sorted samples, in the metric's unit.

    Raises:
        ValueError: If the samples are empty or differ in size.
    """
    x = np.sort(np.asarray(measured, dtype=np.float64))
    y = np.sort(np.asarray(declared, dtype=np.float64))
    if x.size == 0 or x.size != y.size:
        raise ValueError(
            "the Wasserstein distance needs two non-empty samples of equal size, "
            f"got {x.size} and {y.size}"
        )
    return float(np.mean(np.abs(x - y)))


def pair_histogram(
    measured: np.ndarray, declared: np.ndarray, bins: int
) -> Dict[str, Any]:
    """The two samples' histograms on shared edges.

    Args:
        measured: Measured values ``[n]``.
        declared: Declared values ``[n]``.
        bins: Equal-width bins from the pooled minimum to the pooled maximum (one bin
            when they are equal).

    Returns:
        ``{edges, measured, declared}`` (counts per bin).
    """
    if bins < 1:
        raise ValueError(f"histogram_bins must be at least 1, got {bins}")
    x = np.asarray(measured, dtype=np.float64)
    y = np.asarray(declared, dtype=np.float64)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if lo == hi:
        return {
            "edges": [lo, hi],
            "measured": [int(x.size)],
            "declared": [int(y.size)],
        }
    edges = np.linspace(lo, hi, bins + 1)
    return {
        "edges": [float(e) for e in edges],
        "measured": [int(c) for c in np.histogram(x, bins=edges)[0]],
        "declared": [int(c) for c in np.histogram(y, bins=edges)[0]],
    }


def distribution_entry(
    measured: np.ndarray,
    declared: np.ndarray,
    bins: int,
    metric_id: str,
    channel: Optional[int] = None,
    population: Optional[str] = None,
) -> Dict[str, Any]:
    """One metric's distribution block for a group of entries.

    Args:
        measured: Measured values of the paired entries ``[n]``.
        declared: Their declared values ``[n]``.
        bins: Histogram bins.
        metric_id: The metric's id.
        channel: The channel index, when the value is per channel of several.
        population: The population name, when the value is per population.

    Returns:
        ``{id, channel?, population?, n_pairs, diff_mean, diff_abs_mean, ks,
        wasserstein, histogram}``; everything after ``n_pairs`` is ``None`` when
        ``n_pairs`` is 0. ``diff_mean`` and ``diff_abs_mean`` are the mean and the mean
        absolute value of ``measured - declared`` over the paired entries.
    """
    out: Dict[str, Any] = {"id": metric_id}
    if channel is not None:
        out["channel"] = int(channel)
    if population is not None:
        out["population"] = population
    n = int(np.asarray(measured).size)
    out["n_pairs"] = n
    if n == 0:
        out.update(
            {
                "diff_mean": None,
                "diff_abs_mean": None,
                "ks": None,
                "wasserstein": None,
                "histogram": None,
            }
        )
        return out
    diff = np.asarray(measured, dtype=np.float64) - np.asarray(
        declared, dtype=np.float64
    )
    out["diff_mean"] = float(diff.mean())
    out["diff_abs_mean"] = float(np.abs(diff).mean())
    out["ks"] = ks_distance(measured, declared)
    out["wasserstein"] = wasserstein_distance(measured, declared)
    out["histogram"] = pair_histogram(measured, declared, bins)
    return out
