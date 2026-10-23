"""Unit-free field names for a world file.

A world's space fields carry their original unit in the name (``sigma_mm``,
``x_mm``, ``speed_mm_per_ms``). Those names are part of ``world_id``, of every draw's
record and of pressure-simulation's contract, so they stay the canonical names. A world
file may write the same field without the suffix (``sigma``, ``x``, ``speed``); this
module rewrites such a name to its canonical one before the world is parsed, so the
parsed world, its ``world_id`` and its draws are identical to the canonical spelling's.

Which unit a world's space fields are in is declared by ``world.units.space``
(see :func:`sensoryforge.world.runner.declared_units`); the ``_mm`` in a canonical name
only says "the world's space unit" and is historical.

A name is rewritten only when it is not itself a field, the suffixed form is, and no two
fields would share it (``X_mm`` and ``X_mm_per_ms`` both existing leaves ``X`` alone).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Set

from sensoryforge.stimuli.layered import MOTIONS
from sensoryforge.world import kernel
from sensoryforge.world.kinds import EPISODE_FIELDS

_SUFFIXES = ("_mm_per_ms", "_mm")
_PARTS = ("shape", "pattern", "modulation", "motion")


def _known_fields() -> Set[str]:
    """Every field name any registered shape, pattern, modulation or motion has."""
    names: Set[str] = set(EPISODE_FIELDS) | {"amplitude", "background", "clamp_min"}
    for registry, specs_of in (
        (kernel.SHAPE_KINDS, kernel.shape_specs),
        (kernel.PATTERN_KINDS, kernel.pattern_specs),
        (kernel.MODULATION_KINDS, kernel.modulation_specs),
    ):
        for kind in registry:
            names.update(spec.name for spec in specs_of(kind))
    for specs in MOTIONS.values():
        names.update(spec.name for spec in specs)
    return names


def alias_table() -> Dict[str, str]:
    """``{unit-free name: canonical name}`` for the fields registered right now."""
    known = _known_fields()
    found: Dict[str, Set[str]] = {}
    for name in known:
        for suffix in _SUFFIXES:
            if name.endswith(suffix) and len(name) > len(suffix):
                found.setdefault(name[: -len(suffix)], set()).add(name)
                break
    return {
        alias: next(iter(targets))
        for alias, targets in found.items()
        if len(targets) == 1 and alias not in known
    }


def _rename(name: Any, table: Dict[str, str]) -> Any:
    """``name`` or its canonical spelling; dotted ``part.field`` names too."""
    if not isinstance(name, str):
        return name
    if "." in name:
        part, field = name.split(".", 1)
        return f"{part}.{table.get(field, field)}"
    return table.get(name, name)


def _rename_keys(mapping: Any, table: Dict[str, str], where: str) -> Any:
    """``mapping`` with each key renamed; two spellings of one field are an error."""
    if not isinstance(mapping, dict):
        return mapping
    out: Dict[Any, Any] = {}
    for key, value in mapping.items():
        new = _rename(key, table)
        if new in out:
            raise ValueError(
                f"{where}: {key!r} and {new!r} are the same field; write it once"
            )
        out[new] = value
    return out


def _rename_axis(spec: Any, table: Dict[str, str]) -> Any:
    """An axis spec; its ``same_as`` names an axis, so it is renamed too."""
    if isinstance(spec, dict) and isinstance(spec.get("same_as"), str):
        return {**spec, "same_as": _rename(spec["same_as"], table)}
    return spec


def _axes(mapping: Any, table: Dict[str, str], where: str) -> Any:
    renamed = _rename_keys(mapping, table, where)
    if not isinstance(renamed, dict):
        return renamed
    return {k: _rename_axis(v, table) for k, v in renamed.items()}


def _layer(layer: Any, table: Dict[str, str], where: str) -> Any:
    """A class's ``layer``: the field names in shape, pattern, modulation, motion."""
    if not isinstance(layer, dict):
        return layer
    out = dict(layer)
    for part in _PARTS:
        if isinstance(out.get(part), dict):
            out[part] = _rename_keys(out[part], table, f"{where}.{part}")
    return out


def _class(raw: Any, table: Dict[str, str], where: str) -> Any:
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    if "axes" in out:
        out["axes"] = _axes(out["axes"], table, f"{where}.axes")
    if "layer" in out:
        out["layer"] = _layer(out["layer"], table, f"{where}.layer")
    return out


def neutralise(
    raw: Dict[str, Any], table: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """The world mapping with every unit-free field name written canonically.

    Args:
        raw: The ``world:`` mapping (not modified).
        table: ``{alias: canonical}``; default :func:`alias_table`.

    Returns:
        A copy. A world that uses no alias comes back equal to ``raw``.

    Raises:
        ValueError: If a field is written twice, once in each spelling.
    """
    table = alias_table() if table is None else table
    out = copy.deepcopy(raw)
    if "defaults" in out:
        out["defaults"] = _axes(out["defaults"], table, "world.defaults")
    if isinstance(out.get("groups"), dict):
        out["groups"] = {
            name: _axes(group, table, f"world.groups.{name}")
            for name, group in out["groups"].items()
        }
    for section in ("classes", "held_out"):
        if isinstance(out.get(section), dict):
            out[section] = {
                name: _class(cls, table, f"world.{section}.{name}")
                for name, cls in out[section].items()
            }
    if isinstance(out.get("fixed_draws"), dict):
        out["fixed_draws"] = {
            name: _rename_keys(draw, table, f"world.fixed_draws.{name}")
            for name, draw in out["fixed_draws"].items()
        }
    return out
