"""Config validity checks shared by every GUI v2 screen.

:func:`validate` is a pure function -- no Qt, no I/O -- that maps a
:class:`~sensoryforge.config.schema.SensoryForgeConfig` to a dict of dotted
path -> human-readable message for every problem it finds. An empty dict
means the config is clean. :mod:`sensoryforge.gui.widgets.pipeline_strip`
uses the paths as prefixes to colour each chip's status dot; screens later in
Phase 2 extend the set of checks this covers.
"""

from __future__ import annotations

from typing import Dict

from sensoryforge.config.schema import SensoryForgeConfig, validate_dt_ms


def validate(config: SensoryForgeConfig) -> Dict[str, str]:
    """Check ``config`` for the problems Phase 1 knows how to name.

    Runs the schema's own ``__post_init__`` validation (by round-tripping
    through ``to_dict``/``from_dict``, since a GUI edit made through
    ``Session.set_by_path`` writes fields directly and does not re-run
    dataclass validation), :func:`~sensoryforge.config.schema.validate_dt_ms`
    (F-042), that every population's grid reference names a grid that
    exists, that population names are unique, and that the config has at
    least one grid and one population.

    Args:
        config: The configuration to check.

    Returns:
        A dict mapping dotted path (as used by
        :meth:`~sensoryforge.gui.session.Session.set_by_path`) to an
        error message. ``""`` is used for problems that are not specific
        to one field. Empty when the config is clean.
    """
    errors: Dict[str, str] = {}

    try:
        SensoryForgeConfig.from_dict(config.to_dict())
    except ValueError as exc:
        errors[""] = str(exc)

    try:
        validate_dt_ms(config.simulation.dt_ms, config.simulation.integrate_dt_ms)
    except ValueError as exc:
        errors["simulation.dt_ms"] = str(exc)

    grid_names = {grid.name for grid in config.grids}
    for i, population in enumerate(config.populations):
        for population_input in population.effective_inputs():
            if population_input.grid not in grid_names:
                errors[f"populations.{i}.target_grid"] = (
                    f"population {population.name!r}: no grid named "
                    f"{population_input.grid!r}; known grids: "
                    f"{sorted(grid_names)}"
                )

    seen_names: set = set()
    for i, population in enumerate(config.populations):
        if population.name in seen_names:
            errors[f"populations.{i}.name"] = (
                f"duplicate population name {population.name!r}"
            )
        seen_names.add(population.name)

    if not config.grids or not config.populations:
        errors[""] = "config needs at least one grid and one population"

    return errors
