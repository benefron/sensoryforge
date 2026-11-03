"""Which neurons' membrane voltage a bundle keeps (``voltage_probes``, roadmap A6p).

A run's bundle stores a population's readout (spikes, events or state) but not its
membrane voltage. ``voltage_probes`` is the opt-in key that keeps the voltage of a few
fixed neurons, so a re-simulation of the neuron stage can be checked against it. It is
set on a population (``PopulationConfig.voltage_probes``) or for every population
(``SimulationConfig.voltage_probes``; a population's own value wins).

The value is one of:

* a count ``K >= 1``: ``K`` evenly spaced neuron ids, each in the middle of its
  ``1/K`` share of the population (never an edge neuron);
* a list of neuron ids (non-negative integers, no repeats);
* the string ``"all"``: every neuron, for a run archived in full.
"""

from __future__ import annotations

from typing import Any, List, Union

#: The accepted forms of a ``voltage_probes`` value.
VoltageProbes = Union[int, str, List[int]]

ALL_NEURONS = "all"


def check_voltage_probes(spec: Any, *, where: str) -> None:
    """Raise ``ValueError`` if *spec* is not a valid ``voltage_probes`` value.

    Args:
        spec: The value to check; ``None`` (unset) is valid.
        where: What to name in the message (for example ``"Population 'SA'"``).
    """
    if spec is None or spec == ALL_NEURONS:
        return
    if isinstance(spec, bool):
        raise ValueError(f"{where}: voltage_probes must not be a bool, got {spec!r}")
    if isinstance(spec, int):
        if spec < 1:
            raise ValueError(f"{where}: voltage_probes count must be >= 1, got {spec}")
        return
    if isinstance(spec, (list, tuple)):
        ids = list(spec)
        if not ids:
            raise ValueError(f"{where}: voltage_probes id list is empty")
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
            raise ValueError(
                f"{where}: voltage_probes ids must be non-negative integers, "
                f"got {ids!r}"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(f"{where}: voltage_probes ids repeat: {ids!r}")
        return
    raise ValueError(
        f"{where}: voltage_probes must be a count, a list of neuron ids or "
        f"{ALL_NEURONS!r}, got {spec!r}"
    )


def resolve_voltage_probes(spec: VoltageProbes, num_neurons: int) -> List[int]:
    """The neuron ids a ``voltage_probes`` value names in a population.

    Args:
        spec: A valid ``voltage_probes`` value (see :func:`check_voltage_probes`).
        num_neurons: The population's neuron count ``N``.

    Returns:
        Increasing neuron ids in ``[0, N)``. A count above ``N`` keeps every neuron.

    Raises:
        ValueError: If an explicit id is not below *num_neurons*.
    """
    if num_neurons < 1:
        raise ValueError(f"num_neurons must be >= 1, got {num_neurons}")
    if spec == ALL_NEURONS:
        return list(range(num_neurons))
    if isinstance(spec, int):
        k = min(spec, num_neurons)
        return [(2 * i + 1) * num_neurons // (2 * k) for i in range(k)]
    ids = sorted(int(i) for i in spec)
    if ids[-1] >= num_neurons:
        raise ValueError(
            f"voltage_probes id {ids[-1]} is out of range for a population of "
            f"{num_neurons} neurons"
        )
    return ids
