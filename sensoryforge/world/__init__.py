"""Declared stimulus worlds: sample them, render them, build data sets on them.

A world (``world:`` YAML) declares classes of stimuli, each a layered layer
whose fields are drawn from axes. :func:`sample` gives deterministic draw
records, :func:`render` evaluates any draws at any times on any coordinates,
and :mod:`sensoryforge.world.dataset` builds data sets on a world. See
``docs/user_guide/worlds.md`` and ``docs/reference/world_contract.md``.
"""

from sensoryforge.world.distributions import AxisSpec, register_distribution
from sensoryforge.world.kernel import (
    register_modulation,
    register_pattern,
    register_shape,
)
from sensoryforge.world.kinds import ClassKind, register_class_kind
from sensoryforge.world.schema import ClassSpec, World, load_world
from sensoryforge.world.sampling import Draw, Session, fixed_draw, sample, session
from sensoryforge.world.dataset import (
    DatasetSpec,
    Entry,
    build_dataset,
    load_dataset,
    load_manifest,
    write_dataset,
)
from sensoryforge.world.render import Canvas, movie_times, render, render_movie

__all__ = [
    "AxisSpec",
    "Canvas",
    "ClassKind",
    "ClassSpec",
    "DatasetSpec",
    "Draw",
    "Entry",
    "Session",
    "World",
    "build_dataset",
    "fixed_draw",
    "load_dataset",
    "load_manifest",
    "load_world",
    "movie_times",
    "register_class_kind",
    "register_distribution",
    "register_modulation",
    "register_pattern",
    "register_shape",
    "render",
    "render_movie",
    "sample",
    "session",
    "write_dataset",
]
