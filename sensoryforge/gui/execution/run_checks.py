"""Checks on a finished run's results that the user should hear about.

Pure functions over the results dict
:meth:`~sensoryforge.core.simulation_engine.SimulationEngine.run` returns (or
a bundle's ``populations``, which has the same keys without the batch
dimension). No Qt and no engine here, so the CLI and the batch runner can
reuse them.

Ledger F-93b91b1: a population that fires no spikes looks like a working
result unless something says so. :func:`silent_populations` finds every such
population, with the peak current its neurons received, so the user can tell
a stimulus or input gain that is too weak (a peak far below what the neuron
needs to fire) from something else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

import torch


@dataclass(frozen=True)
class SilentPopulation:
    """A spiking population that fired no spikes in a run.

    Attributes:
        name: Population name, as keyed in the results dict.
        peak_input_ma: The largest value of the population's ``"filtered"``
            result in mA -- the current its neurons received, after the
            filter, ``input_gain`` and noise, before any ``input_floor`` --
            over every neuron and time step. ``None`` when the results did
            not keep ``"filtered"`` (a run without intermediates) or the
            population has no neurons.
    """

    name: str
    peak_input_ma: Optional[float]


def _peak(tensor: Any) -> Optional[float]:
    """The maximum of ``tensor``, or ``None`` if it is missing or empty."""
    if tensor is None:
        return None
    values = torch.as_tensor(tensor)
    if values.numel() == 0:
        return None
    return float(values.max())


def silent_populations(
    results: Mapping[str, Mapping[str, Any]],
) -> List[SilentPopulation]:
    """Every spiking population in ``results`` that fired no spikes.

    A population is spiking when its entry has a ``"spikes"`` tensor. An
    analog population (a DSL model with no threshold) carries ``"state"``
    and no ``"spikes"``; it cannot fire, so it is never reported. A value of
    ``None`` counts as missing.

    Args:
        results: Population name -> that population's results, as returned
            by ``SimulationEngine.run`` (tensors ``[batch, time, N]``) or
            stored in a bundle (``[time, N]``). ``"spikes"`` holds spike
            counts per bin; ``"filtered"`` (optional) the neuron input in mA.

    Returns:
        One :class:`SilentPopulation` per silent spiking population, in the
        order of ``results``. Empty when every spiking population fired.

    Example:
        >>> import torch
        >>> silent_populations({
        ...     "SA": {"spikes": torch.zeros(1, 10, 4),
        ...            "filtered": torch.full((1, 10, 4), 0.5)},
        ...     "RA": {"spikes": torch.ones(1, 10, 4)},
        ...     "Leaky": {"state": torch.zeros(1, 10, 4)},
        ... })
        [SilentPopulation(name='SA', peak_input_ma=0.5)]
    """
    silent: List[SilentPopulation] = []
    for name, pop_results in results.items():
        spikes = pop_results.get("spikes")
        if spikes is None:
            continue
        if bool(torch.as_tensor(spikes).any()):
            continue
        silent.append(
            SilentPopulation(
                name=str(name), peak_input_ma=_peak(pop_results.get("filtered"))
            )
        )
    return silent


def describe_silent(silent: Sequence[SilentPopulation]) -> str:
    """The warning text for ``silent``, one line per population.

    Args:
        silent: As returned by :func:`silent_populations`.

    Returns:
        ``""`` when ``silent`` is empty; otherwise a short paragraph naming
        each population and its peak neuron input in mA.

    Example:
        >>> print(describe_silent([SilentPopulation("RA", 0.0031)]))
        No spikes from 1 population in this run:
          RA: peak neuron input 0.0031 mA
        Far below what the neurons need to fire? Raise the stimulus or input gain.
    """
    if not silent:
        return ""
    noun = "population" if len(silent) == 1 else "populations"
    lines = [f"No spikes from {len(silent)} {noun} in this run:"]
    for pop in silent:
        peak = (
            f"peak neuron input {pop.peak_input_ma:.3g} mA"
            if pop.peak_input_ma is not None
            else "neuron input not recorded"
        )
        lines.append(f"  {pop.name}: {peak}")
    lines.append(
        "Far below what the neurons need to fire? Raise the stimulus or input gain."
    )
    return "\n".join(lines)
