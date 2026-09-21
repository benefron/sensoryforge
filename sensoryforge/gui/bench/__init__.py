"""Bench tests for the Populations screen (Task 2.3).

Each module here computes a small, cheap preview of one component of a
population's pipeline -- never a full simulation:

- :mod:`rf_footprint`: the population's :class:`~sensoryforge.core.rf_bank.ReceptiveFieldBank`,
  built exactly as :class:`~sensoryforge.core.simulation_engine.SimulationEngine` builds it.
- :mod:`filter_step`: the population's filter's response to a unit step and a ramp.
- :mod:`neuron_trace`: the population's neuron model's voltage trace and an f-I curve
  for a step current.

All three read the live :class:`~sensoryforge.gui.session.Session` config, recompute on
``configChanged``, and show errors as text rather than raising out of a Qt slot.
"""

from __future__ import annotations

from typing import Optional

from sensoryforge.config.schema import PopulationConfig, SensoryForgeConfig


def find_population(
    config: SensoryForgeConfig, name: Optional[str]
) -> Optional[PopulationConfig]:
    """The population named ``name`` in ``config``, or ``None``.

    A small shared lookup so every bench module (and the cards/screen) reads
    "the currently selected population" the same way, without assuming index
    stability across edits (populations may be reordered or removed).

    Args:
        config: The session's config.
        name: A population name, or ``None`` (nothing selected).

    Returns:
        The matching :class:`~sensoryforge.config.schema.PopulationConfig`,
        or ``None`` if ``name`` is ``None`` or no population has that name.
    """
    if name is None:
        return None
    for pop in config.populations:
        if pop.name == name:
            return pop
    return None


def population_index(config: SensoryForgeConfig, name: str) -> int:
    """The index of the population named ``name`` in ``config.populations``.

    Args:
        config: The session's config.
        name: A population name.

    Returns:
        The zero-based index.

    Raises:
        ValueError: If no population has that name.
    """
    for index, pop in enumerate(config.populations):
        if pop.name == name:
            return index
    known = [p.name for p in config.populations]
    raise ValueError(f"no population named {name!r}; the config has {known}")
