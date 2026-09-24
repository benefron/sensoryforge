"""Config validity checks shared by every GUI v2 screen.

:func:`validate` is a pure function -- no Qt, no I/O -- that maps a
:class:`~sensoryforge.config.schema.SensoryForgeConfig` to a dict of dotted
key -> human-readable message for every problem it finds. An empty dict means
the config will build. The keys name the pipeline stage a problem belongs to,
so the pipeline strip can colour the right chip and a screen can show the
problems that concern it:

==================================  ===========================================
Key                                 Covers
==================================  ===========================================
``""``                              the config as a whole
``"simulation"``/``"simulation.dt_ms"``  run settings (F-042 for the step)
``"stimulus"``                      the stimulus block
``"grids.<i>"``                     one sensor array
``"populations.<i>.name"``          duplicate names
``"populations.<i>.target_grid"``   an input naming a grid that does not exist
``"populations.<i>.rf"``            building the receptive fields
``"populations.<i>.combine"``       combining several inputs (F-062)
``"populations.<i>.filter"``        building the filter
``"populations.<i>.neuron"``        building the neuron model
``"populations.<i>.readout"``       a readout the neuron model cannot provide
``"populations.<i>"``               anything else about that population
==================================  ===========================================

Beyond the schema's own checks, each enabled population's receptive fields,
filter and neuron are built with the engine's own code
(:class:`~sensoryforge.core.simulation_engine.SimulationEngine`,
:func:`~sensoryforge.core.simulation_engine.build_filter`,
:func:`~sensoryforge.core.simulation_engine.build_neuron`), so a problem the
engine would raise when a run starts is reported while the config is edited.
Building receptive fields costs up to about 100 ms per population, so each
result is cached on the fields it depends on.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
    StimulusConfig,
    validate_dt_ms,
)

#: Exceptions the engine raises for a config it cannot build.
_BUILD_ERRORS = (ValueError, KeyError, TypeError, RuntimeError, IndexError, OSError)

#: PopulationConfig fields that do not affect the receptive fields; the
#: receptive-field check is cached on everything else.
_NON_RF_FIELDS = frozenset(
    {
        "name",
        "color",
        "visible",
        "enabled",
        "filter_method",
        "filter_params",
        "neuron_model",
        "neuron_type",
        "model_params",
        "dsl_config",
        "readout",
        "input_gain",
        "noise_std",
        "noise_mean",
        "noise_seed",
    }
)

_RF_CACHE: "OrderedDict[str, Tuple[Optional[str], Optional[str]]]" = OrderedDict()
_RF_CACHE_SIZE = 64


def validate(config: SensoryForgeConfig) -> Dict[str, str]:
    """Every problem that would stop ``config`` from building, by stage key.

    Args:
        config: The configuration to check.

    Returns:
        A dict mapping a stage key (see the module docstring) to an error
        message. Empty when the config is clean.
    """
    errors: Dict[str, str] = {}

    if not config.grids or not config.populations:
        errors[""] = "config needs at least one grid and one population"

    _check_schema(config, errors)

    try:
        validate_dt_ms(config.simulation.dt_ms, config.simulation.integrate_dt_ms)
    except ValueError as exc:
        errors["simulation.dt_ms"] = str(exc)

    grid_names = {grid.name for grid in config.grids}
    seen_names: set = set()
    for i, population in enumerate(config.populations):
        if population.name in seen_names:
            errors[f"populations.{i}.name"] = (
                f"duplicate population name {population.name!r}"
            )
        seen_names.add(population.name)

        missing = [
            population_input.grid
            for population_input in population.effective_inputs()
            if population_input.grid not in grid_names
        ]
        if missing and missing[0] is None:
            errors[f"populations.{i}.target_grid"] = (
                f"population {population.name!r} reads no sensor array; "
                "choose a grid for it on the Populations screen"
            )
        elif missing:
            errors[f"populations.{i}.target_grid"] = (
                f"population {population.name!r}: no grid named "
                f"{missing[0]!r}; known grids: {sorted(grid_names)}"
            )

    for i, grid in enumerate(config.grids):
        if f"grids.{i}" in errors:
            continue
        message = _grid_problem(grid)
        if message:
            errors[f"grids.{i}"] = message

    grids_ok = not any(key.startswith("grids.") for key in errors)
    if config.stimulus is not None and "stimulus" not in errors and grids_ok:
        message = _stimulus_problem(config)
        if message:
            errors["stimulus"] = message

    # A population is built only when what the build reads is valid: the run
    # settings, every grid, and the population's own block and grid names.
    shared_ok = not any(key.startswith(("grids.", "simulation")) for key in errors)
    if shared_ok:
        for i, population in enumerate(config.populations):
            own = (f"populations.{i}", f"populations.{i}.target_grid")
            if population.enabled and not any(key in errors for key in own):
                _check_population_build(config, i, errors)

    return errors


def errors_under(errors: Dict[str, str], prefix: str) -> List[str]:
    """The messages whose key is ``prefix`` or lies below it.

    Args:
        errors: A :func:`validate` result.
        prefix: A key such as ``"populations.1"`` or ``"grids"``; ``""``
            selects every message.

    Returns:
        The matching messages, in key order.
    """
    return [
        message
        for key, message in sorted(errors.items())
        if not prefix or key == prefix or key.startswith(prefix + ".")
    ]


# ----------------------------------------------------------------- schema


def _check_schema(config: SensoryForgeConfig, errors: Dict[str, str]) -> None:
    """Re-run each block's own ``__post_init__`` checks, keyed to the block.

    A GUI edit made through ``Session.set_by_path`` writes a field directly
    and does not re-run dataclass validation, so each block is rebuilt from
    its dict to find out whether it is still valid.
    """
    blocks = [("simulation", SimulationConfig, config.simulation)]
    if config.stimulus is not None:
        blocks.append(("stimulus", StimulusConfig, config.stimulus))
    blocks += [(f"grids.{i}", GridConfig, g) for i, g in enumerate(config.grids)]
    blocks += [
        (f"populations.{i}", PopulationConfig, p)
        for i, p in enumerate(config.populations)
    ]
    for key, cls, block in blocks:
        try:
            cls.from_dict(block.to_dict())
        except _BUILD_ERRORS as exc:
            errors[key] = str(exc)


# ------------------------------------------------------------ grids, stimulus


def _grid_problem(grid: GridConfig) -> Optional[str]:
    """A geometry the builder would accept but that cannot be meant."""
    if grid.arrangement != "composite" and not grid.coords_file:
        if grid.spacing is not None and grid.spacing <= 0:
            return (
                f"grid {grid.name!r}: spacing must be positive, got {grid.spacing} mm"
            )
        for name in ("rows", "cols"):
            value = getattr(grid, name)
            if value is not None and value < 1:
                return f"grid {grid.name!r}: {name} must be at least 1, got {value}"
    if grid.coords_file or (
        grid.arrangement != "composite" and grid.density is not None
    ):
        # Reading the file, or building with density set, is the only way to
        # know it is usable (D-88b4b41: density conflicts with 'grid' and
        # 'jittered_grid', and must be positive -- ReceptorGrid enforces both).
        from sensoryforge.core.simulation_engine import build_grid

        try:
            build_grid(grid, device="cpu")
        except _BUILD_ERRORS as exc:
            return f"grid {grid.name!r}: {exc}"
    return None


def _stimulus_problem(config: SensoryForgeConfig) -> Optional[str]:
    """Render one step of the stimulus the way a run does; the error, if any."""
    from sensoryforge.stimuli.render import render_for_config

    dt_ms = float(config.simulation.dt_ms)
    # Validation never touches the run device: probe on the CPU.
    probe = copy.copy(config)
    probe.simulation = copy.copy(config.simulation)
    probe.simulation.device = "cpu"
    try:
        render_for_config(probe, duration_ms=dt_ms, dt_ms=dt_ms)
    except _BUILD_ERRORS as exc:
        return f"stimulus {config.stimulus.type!r}: {exc}"
    return None


# ------------------------------------------------------------------ build


def _check_population_build(
    config: SensoryForgeConfig, index: int, errors: Dict[str, str]
) -> None:
    from sensoryforge.core.simulation_engine import build_filter, build_neuron

    population = config.populations[index]
    key = f"populations.{index}"

    rf_error, combine_error = _receptive_field_errors(config, population)
    if combine_error:
        errors[f"{key}.combine"] = combine_error
    elif rf_error:
        errors[f"{key}.rf"] = rf_error

    try:
        build_filter(population, config.simulation, device="cpu")
    except _BUILD_ERRORS as exc:
        errors[f"{key}.filter"] = f"filter {population.filter_method!r}: {exc}"

    try:
        build_neuron(population, config.simulation, device="cpu")
    except _BUILD_ERRORS as exc:
        message = str(exc)
        stage = "readout" if "readout" in message else "neuron"
        errors[f"{key}.{stage}"] = message


def _receptive_field_errors(
    config: SensoryForgeConfig, population: PopulationConfig
) -> Tuple[Optional[str], Optional[str]]:
    """``(rf_error, combine_error)`` for one population, cached.

    Builds the population alone through :class:`SimulationEngine`, with its
    filter and neuron replaced by defaults so only the receptive fields and
    their combination can fail.
    """
    rf_fields = {
        k: v for k, v in population.to_dict().items() if k not in _NON_RF_FIELDS
    }
    cache_key = repr(
        (
            rf_fields,
            [grid.to_dict() for grid in config.grids],
            config.simulation.dt_ms,
            config.simulation.integrate_dt_ms,
        )
    )
    if cache_key in _RF_CACHE:
        _RF_CACHE.move_to_end(cache_key)
        return _RF_CACHE[cache_key]

    from sensoryforge.core.simulation_engine import SimulationEngine

    snapshot = copy.deepcopy(config)
    probe = copy.deepcopy(population)
    probe.enabled = True
    probe.filter_method = "none"
    probe.filter_params = {}
    probe.neuron_model = "Izhikevich"
    probe.model_params = {}
    probe.dsl_config = None
    probe.readout = "auto"
    snapshot.populations = [probe]

    result: Tuple[Optional[str], Optional[str]] = (None, None)
    try:
        SimulationEngine(snapshot, device="cpu")
    except _BUILD_ERRORS as exc:
        message = str(exc)
        if "combine" in message:
            result = (None, message)
        else:
            result = (message, None)

    _RF_CACHE[cache_key] = result
    while len(_RF_CACHE) > _RF_CACHE_SIZE:
        _RF_CACHE.popitem(last=False)
    return result
