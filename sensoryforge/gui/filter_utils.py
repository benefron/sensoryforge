"""Shared GUI helper utilities."""
from __future__ import annotations

from typing import Optional


def normalize_filter_method(
    method: Optional[object],
    neuron_type: Optional[str] = None,
) -> str:
    """Normalize legacy filter selections to the canonical option set.

    Parameters
    ----------
    method:
        Raw filter identifier provided by GUI state, config files, or legacy
        notebooks. Accepts ``None``, strings, or enum-like objects.
    neuron_type:
        Optional neuron population label (``"SA"``/``"RA"``). Used to map
        historical ``"multi_step"`` and ``"steady_state"`` selections onto the
        correct filter family.
    """

    if method is None:
        return "none"
    value = str(method).strip().lower()
    if value in {"", "none", "no filter"}:
        return "none"
    if value in {"sa", "sa filter", "sa_filter"}:
        return "sa"
    if value in {"ra", "ra filter", "ra_filter"}:
        return "ra"
    if value in {"multi_step", "steady_state"}:
        return "ra" if (neuron_type or "SA").upper() == "RA" else "sa"
    if value == "edge_response":
        return "ra"
    return "none"
