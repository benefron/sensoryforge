"""Load a pressure-simulation "design directory" into a SensoryForge config.

pressure-simulation designs an encoder analytically and hands it off as a
directory -- ``design.json`` plus one ``<population>.npz`` per population --
written by its own ``design.export.write_design`` (see that module's
docstring for the exact contract). This module is the SensoryForge side of
that hand-off boundary: :func:`load_design` turns the directory into a
:class:`~sensoryforge.config.schema.SensoryForgeConfig` whose populations use
the ``imported`` receptive-field builder
(:class:`sensoryforge.core.rf_builders.imported.ImportedRFBuilder`) to read
each population's ``.npz`` directly. This module does not touch ``H`` or
``centers`` itself -- ``ImportedRFBuilder`` performs the y-slow -> x-slow
column re-index and the ``[y, x]`` -> ``(x, y)`` centre flip when it reads
the ``.npz`` (Phase 2 guardrail 4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)

#: pressure-simulation's design.json `population.filter_method` -> the
#: SensoryForge `PopulationConfig.neuron_type` its resolvers
#: (`sensoryforge.config.defaults.resolve_neuron_params`) key the
#: Izhikevich/AdEx preset off of. `filter_method` on its own is not enough:
#: the resolvers read `neuron_type`, not `filter_method`. Anything other
#: than "sa"/"ra" is upper-cased as-is rather than dropped, so an unknown
#: filter method still gets a `neuron_type`.
_NEURON_TYPE_BY_FILTER_METHOD: Dict[str, str] = {"sa": "SA", "ra": "RA"}

#: Required keys of one `design.json["populations"][i]` record -- everything
#: `load_design` reads verbatim onto the emitted `PopulationConfig`.
_REQUIRED_POPULATION_KEYS = (
    "name",
    "rf_file",
    "N",
    "filter_method",
    "filter_params",
    "neuron_model",
    "model_params",
    "input_gain",
)


def read_manifest(design_dir: Union[str, Path]) -> Dict[str, Any]:
    """Read a design directory's ``design.json`` verbatim.

    Args:
        design_dir: Directory written by pressure-simulation's
            ``design.export.write_design`` (``design.json`` plus one
            ``<population>.npz`` per population).

    Returns:
        The parsed ``design.json`` as a plain dict (``design_id``,
        ``git_sha``, ``git_dirty``, ``decisions``, ``populations``),
        unmodified. Callers that must stamp the manifest verbatim into a
        downstream artifact (e.g. a data bundle) should use this instead of
        re-deriving it from the :class:`~sensoryforge.config.schema.SensoryForgeConfig`
        :func:`load_design` returns.

    Raises:
        FileNotFoundError: If ``design_dir/design.json`` does not exist.
    """
    manifest_path = Path(design_dir) / "design.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"load_design: {manifest_path} does not exist (expected a design "
            "directory written by pressure-simulation's "
            "design.export.write_design)"
        )
    with open(manifest_path) as fh:
        return json.load(fh)


def _require(mapping: Dict[str, Any], key: str, where: str) -> Any:
    """Return ``mapping[key]``, raising ``ValueError`` naming ``key`` if absent.

    Args:
        mapping: The dict to read from.
        key: The required key.
        where: Human-readable location of ``mapping``, for the error message.

    Returns:
        ``mapping[key]``.

    Raises:
        ValueError: If ``key`` is not in ``mapping``.
    """
    if key not in mapping:
        raise ValueError(f"load_design: {where} is missing required key {key!r}")
    return mapping[key]


def load_design(design_dir: Union[str, Path]) -> SensoryForgeConfig:
    """Load a pressure-simulation design directory into a SensoryForge config.

    Reads ``design_dir/design.json`` and builds:

    * one :class:`~sensoryforge.config.schema.GridConfig` (named ``"design"``)
      from ``decisions["grid"]`` (``[rows, cols]``) and
      ``decisions["spacing_mm"]`` (mm);
    * a :class:`~sensoryforge.config.schema.SimulationConfig` carrying
      ``decisions["dt_ms"]`` (ms, defaulting to 1.0 ms when absent -- not a
      required key) and ``decisions["seed"]`` (``None`` when absent);
    * one :class:`~sensoryforge.config.schema.PopulationConfig` per entry in
      ``design.json["populations"]``, with ``innervation_method="imported"``
      and ``innervation_params={"path": <absolute path to that population's
      .npz>}`` so
      :class:`sensoryforge.core.rf_builders.imported.ImportedRFBuilder` reads
      the receptive fields pressure-simulation designed. ``filter_method``,
      ``filter_params``, ``neuron_model``, ``model_params`` and
      ``input_gain`` are carried over verbatim (no remapping, no re-tuning).
      ``neuron_type`` is derived from ``filter_method`` via
      :data:`_NEURON_TYPE_BY_FILTER_METHOD` ("sa" -> "SA", "ra" -> "RA",
      anything else upper-cased as-is), because
      ``sensoryforge.config.defaults.resolve_neuron_params`` keys the
      Izhikevich preset off ``neuron_type``, not ``filter_method`` -- a
      designed population with no ``neuron_type`` would silently keep the
      dataclass default ("SA") for an RA population.

    Args:
        design_dir: Directory holding ``design.json`` and one
            ``<population>.npz`` per population, as written by
            pressure-simulation's ``design.export.write_design``.

    Returns:
        A :class:`~sensoryforge.config.schema.SensoryForgeConfig` with one
        grid and one population per design population, ready for
        :class:`~sensoryforge.core.simulation_engine.SimulationEngine`.

    Raises:
        FileNotFoundError: If ``design_dir/design.json`` or a population's
            ``.npz`` file does not exist.
        ValueError: If a required key is missing from ``decisions`` or from
            a population record -- the message names the missing key.
    """
    design_dir = Path(design_dir).expanduser().resolve()
    manifest = read_manifest(design_dir)

    decisions = _require(manifest, "decisions", "design.json")
    grid = _require(decisions, "grid", "decisions")
    if not isinstance(grid, (list, tuple)) or len(grid) != 2:
        raise ValueError(
            f"load_design: decisions['grid'] must be [rows, cols], got {grid!r}"
        )
    spacing_mm = _require(decisions, "spacing_mm", "decisions")
    rows, cols = int(grid[0]), int(grid[1])

    grid_config = GridConfig(
        name="design",
        arrangement="grid",
        rows=rows,
        cols=cols,
        spacing=float(spacing_mm),
    )

    populations_record = _require(manifest, "populations", "design.json")

    populations = []
    for index, prec in enumerate(populations_record):
        where = f"populations[{index}]"
        for key in _REQUIRED_POPULATION_KEYS:
            _require(prec, key, where)

        rf_file = prec["rf_file"]
        npz_path = design_dir / rf_file
        if not npz_path.exists():
            raise FileNotFoundError(
                f"load_design: {where} names rf_file {rf_file!r} but "
                f"{npz_path} does not exist"
            )

        filter_method = prec["filter_method"]
        neuron_type = _NEURON_TYPE_BY_FILTER_METHOD.get(
            str(filter_method).lower(), str(filter_method).upper()
        )

        populations.append(
            PopulationConfig(
                name=prec["name"],
                neuron_type=neuron_type,
                target_grid=grid_config.name,
                innervation_method="imported",
                innervation_params={"path": str(npz_path)},
                filter_method=filter_method,
                filter_params=dict(prec["filter_params"]),
                neuron_model=prec["neuron_model"],
                model_params=dict(prec["model_params"]),
                input_gain=float(prec["input_gain"]),
            )
        )

    simulation_kwargs: Dict[str, Any] = {}
    if "dt_ms" in decisions:
        simulation_kwargs["dt_ms"] = float(decisions["dt_ms"])
    if "seed" in decisions:
        simulation_kwargs["seed"] = decisions["seed"]

    return SensoryForgeConfig(
        grids=[grid_config],
        populations=populations,
        simulation=SimulationConfig(**simulation_kwargs),
    )


__all__ = ["load_design", "read_manifest"]
