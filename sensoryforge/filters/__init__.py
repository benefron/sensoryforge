"""Temporal filtering module for SA/RA dual-pathway processing.

This module implements biologically-inspired temporal filters based on the
Parvizi-Fard SA (Slowly Adapting) and RA (Rapidly Adapting) differential equations.

Filters:
    SAFilterTorch: Slowly adapting (sustained response) filter
    RAFilterTorch: Rapidly adapting (transient response) filter
    CombinedSARAFilter: Dual-pathway filter combining SA and RA dynamics

Example:
    >>> from sensoryforge.filters import CombinedSARAFilter
    >>> filter = CombinedSARAFilter(num_neurons=100)
    >>> sa_currents, ra_currents = filter(input_current)
"""

from sensoryforge.filters.sa_ra import (
    SAFilterTorch,
    RAFilterTorch,
    CombinedSARAFilter,
)

__all__ = [
    "SAFilterTorch",
    "RAFilterTorch",
    "CombinedSARAFilter",
]
