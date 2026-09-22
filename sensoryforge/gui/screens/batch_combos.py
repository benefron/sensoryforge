"""Turning sweep rows into combinations, written configs, and previews.

The Batch screen's table produces ``(path, values)`` pairs; this module is
where those become the list of ``{path: value}`` combinations (full grid or
zipped), repetitions with distinct seeds, and the config files written to
disk. It builds on
:mod:`sensoryforge.gui.execution.sweep_controller` -- reusing
:class:`~sensoryforge.gui.execution.sweep_controller.SweepManifest`, its
directory-naming convention and its subprocess command -- rather than
duplicating them, but does not use
:func:`~sensoryforge.gui.execution.sweep_controller.write_sweep` itself:
that function's cartesian product does not have a "zipped" mode, and this
module's ``build_combinations`` covers both.
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.gui.execution.sweep_controller import (
    COMBO_DIR_FORMAT,
    MANIFEST_FILENAME,
    SweepManifest,
)
from sensoryforge.gui.session import resolve_parent
from sensoryforge.gui.session import set_by_path as session_set_by_path

#: The two ways a sweep table combines its rows into combinations.
COMBINE_MODES = ("full_grid", "zipped")

#: Where each repetition's seed is written.
SEED_PATH = "simulation.seed"

Combo = Dict[str, Any]


def build_combinations(
    fields: Sequence[Tuple[str, List[Any]]], mode: str
) -> List[Combo]:
    """Combine ``fields`` into ``{path: value}`` combinations.

    Args:
        fields: ``(dotted path, values)`` pairs, one per sweep row. Order is
            preserved so the first field varies slowest in ``"full_grid"``
            mode.
        mode: ``"full_grid"`` (cartesian product of every field's values) or
            ``"zipped"`` (values paired by position; every field must have
            the same number of values).

    Returns:
        One dict per combination, mapping every field's path to its value at
        that point. Empty when ``fields`` is empty.

    Raises:
        ValueError: If ``mode`` is not one of :data:`COMBINE_MODES`, or
            ``mode="zipped"`` and the fields' value lists are not all the
            same length.
    """
    if not fields:
        return []
    if mode not in COMBINE_MODES:
        raise ValueError(f"unknown combine mode {mode!r}; expected {COMBINE_MODES}")

    paths = [path for path, _values in fields]
    value_lists = [values for _path, values in fields]

    if mode == "zipped":
        lengths = sorted({len(values) for values in value_lists})
        if len(lengths) > 1:
            raise ValueError(
                "zipped mode requires every field to have the same number "
                f"of values; got lengths {lengths}"
            )
        return [dict(zip(paths, point)) for point in zip(*value_lists)]

    return [dict(zip(paths, point)) for point in itertools.product(*value_lists)]


def expand_repetitions(
    combos: Sequence[Combo],
    repetitions: int,
    base_seed: int,
    *,
    seed_path: str = SEED_PATH,
) -> List[Combo]:
    """Repeat each combination, giving each repeat a distinct seed.

    Args:
        combos: The base combinations (:func:`build_combinations`'s output).
        repetitions: How many times to repeat each combination (at least 1).
        base_seed: The first repetition's seed; repetition ``r`` (0-indexed)
            of any combination gets ``seed_path`` set to ``base_seed + r``,
            so noisy runs are reproducible and distinct (the brief).
        seed_path: The dotted path the seed is written to.

    Returns:
        ``len(combos) * repetitions`` combinations. With ``repetitions == 1``
        every combo is returned unchanged (no ``seed_path`` key added) --
        there is nothing to disambiguate, and a plain sweep with no
        repetitions must not silently overwrite the config's own seed.
        With ``repetitions > 1`` each combo's dict is extended with
        ``seed_path``, listed after the combo's own fields so
        ``dict.__repr__``/JSON output reads field values before the seed.

    Raises:
        ValueError: If ``repetitions`` is less than 1.
    """
    if repetitions < 1:
        raise ValueError(f"repetitions must be at least 1, got {repetitions}")
    if repetitions == 1:
        return [dict(combo) for combo in combos]
    expanded: List[Combo] = []
    for combo in combos:
        for r in range(repetitions):
            point = dict(combo)
            point[seed_path] = base_seed + r
            expanded.append(point)
    return expanded


def preview_config(config: SensoryForgeConfig, combo: Combo) -> SensoryForgeConfig:
    """The config one combination would run with, without writing anything.

    Args:
        config: The base experiment (never mutated).
        combo: A ``{path: value}`` combination, e.g. from
            :func:`expand_repetitions`.

    Returns:
        A deep copy of ``config`` with every path in ``combo`` set.

    Raises:
        ValueError: If a path in ``combo`` does not resolve against
            ``config``.
    """
    cfg = copy.deepcopy(config)
    for path, value in combo.items():
        session_set_by_path(cfg, path, value)
    return cfg


def build_preview_manifest(
    root: Union[str, Path], combos: Sequence[Combo]
) -> SweepManifest:
    """An in-memory :class:`SweepManifest` for ``combos``, writing nothing.

    Used to compute the exact ``sensoryforge.cli run`` command a written
    sweep would use (:func:`~sensoryforge.gui.execution.sweep_controller.sweep_command`
    only does path arithmetic) before anything is actually written to disk.

    Args:
        root: The sweep root the manifest would be written under.
        combos: The combinations, in write order.

    Returns:
        A manifest with the same ``combo_NNN`` naming
        :func:`write_combo_sweep` would use, but no files on disk.
    """
    root = Path(root)
    manifest_combos = [
        {"index": index, "dir": COMBO_DIR_FORMAT.format(index), "values": dict(combo)}
        for index, combo in enumerate(combos)
    ]
    return SweepManifest(root=root, combos=manifest_combos)


def write_combo_sweep(
    config: SensoryForgeConfig,
    combos: Sequence[Combo],
    *,
    root: Union[str, Path],
    duration_ms: float,
) -> SweepManifest:
    """Write one ``config.yml`` per combination, plus a ``manifest.json``.

    Mirrors
    :func:`~sensoryforge.gui.execution.sweep_controller.write_sweep`'s file
    layout exactly (so :class:`~sensoryforge.gui.execution.sweep_controller.SweepController`
    and :func:`~sensoryforge.gui.execution.sweep_controller.write_slurm_script`
    work unchanged on the result), but takes literal combinations instead of
    a cartesian :class:`~sensoryforge.gui.execution.sweep_controller.SweepSpec`
    -- the only way to support "zipped" mode and per-repetition seeds.

    Args:
        config: The base experiment. Deep-copied per combination; never
            mutated.
        combos: The combinations to write, in order (e.g.
            :func:`expand_repetitions`'s output).
        root: The sweep directory (created, with its parents).
        duration_ms: Written into every combination's
            ``simulation.duration_ms`` before that combination's own values
            are applied, so an explicit ``simulation.duration_ms`` in a combo
            (unusual, but not forbidden) wins.

    Returns:
        The :class:`SweepManifest`, also written to ``root/manifest.json``.

    Raises:
        ValueError: If ``combos`` is empty, or a path in some combination
            does not resolve against ``config`` -- named, before anything is
            written.
    """
    if not combos:
        raise ValueError("cannot write a sweep with no combinations")

    root = Path(root)
    base = copy.deepcopy(config)
    base.simulation.duration_ms = float(duration_ms)

    all_paths = sorted({path for combo in combos for path in combo})
    for path in all_paths:
        resolve_parent(base, path)

    root.mkdir(parents=True, exist_ok=True)
    manifest_combos: List[Dict[str, Any]] = []
    for index, combo in enumerate(combos):
        cfg = copy.deepcopy(base)
        for path, value in combo.items():
            session_set_by_path(cfg, path, value)
        combo_dir = root / COMBO_DIR_FORMAT.format(index)
        combo_dir.mkdir(exist_ok=True)
        (combo_dir / "config.yml").write_text(cfg.to_yaml(), encoding="utf-8")
        manifest_combos.append(
            {"index": index, "dir": combo_dir.name, "values": dict(combo)}
        )

    manifest = SweepManifest(root=root, combos=manifest_combos)
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "root": str(root),
                "duration_ms": float(duration_ms),
                "fields": all_paths,
                "combos": manifest_combos,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return manifest


def format_combo(combo: Combo) -> str:
    """One combination as a short human-readable line, e.g. for the summary.

    Args:
        combo: A ``{path: value}`` combination.

    Returns:
        ``"path=value, path=value"``, in the combo's own key order.
    """
    return ", ".join(f"{path}={value}" for path, value in combo.items())
