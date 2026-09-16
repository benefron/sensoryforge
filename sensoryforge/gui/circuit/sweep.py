"""Batch sweeps built from the live Circuit graph (Phase 3, Wave Q, Q1).

``BatchTab`` (``sensoryforge/gui/tabs/batch_tab.py``) sweeps a config *file*
today, expanded by :class:`~sensoryforge.core.batch_executor.BatchExecutor`.
This module adds the "sweep this graph" path the Q1 spec asks for: a
parameter is chosen by naming a node and one of its settings, a list of
values is supplied, and one bundle is written per value by mutating that one
node's stored config and re-running :func:`sensoryforge.gui.circuit.run.run_graph_once`
(the same helper the Run button uses) once per value.

Unlike :class:`~sensoryforge.core.batch_executor.BatchExecutor` (which sweeps
stimulus parameters via ``base_config`` + per-run overrides), a graph sweep
can target *any* node's setting -- a grid's row count, a filter's ``k1``, a
readout's ``model_params`` entry -- because :func:`set_node_param` dispatches
on the node's own ``to_config()``/``from_config()`` shape (see
``sensoryforge.gui.circuit.nodes`` module docstring for what each node owns).

Silent-wrongness note (the reason this module exists instead of a one-line
loop in the tab): a sweep that runs and writes N bundles looks successful
whether or not the parameter was actually varied -- a typo'd node name, a
param name the node ignores, or a value that gets silently clamped back to
one setting would all still produce N bundles. :func:`sweep_graph` guards
against exactly that: it re-reads the swept value back out of each bundle's
own written config (never trusts the value it *thinks* it set) and raises
if two bundles end up with the same value recorded, or if a param name does
not exist on the node at all (see :func:`set_node_param`).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, List, Sequence, Union

from pyqtgraph.flowchart import Flowchart

from sensoryforge.gui.circuit.nodes import (
    CombineNode,
    FilterNode,
    ProcessingNode,
    ReadoutNode,
    RFBankNode,
    RecordNode,
    SensorArrayNode,
    StimulusNode,
)
from sensoryforge.gui.circuit.run import run_graph_once


class SweepValidationError(ValueError):
    """Raised by :func:`set_node_param`/:func:`sweep_graph` when a node or
    parameter name does not resolve to something that can actually be set."""


def _set_dataclass_field(obj: Any, param_name: str, value: Any) -> None:
    field_names = {f.name for f in dataclasses.fields(obj)}
    if param_name not in field_names:
        raise SweepValidationError(
            f"{type(obj).__name__} has no field {param_name!r} "
            f"(known fields: {sorted(field_names)})."
        )
    setattr(obj, param_name, value)


def set_node_param(node, param_name: str, value: Any) -> None:
    """Set one setting on ``node`` by name, dispatching on its node type.

    Args:
        node: A live node from a Circuit tab's flowchart (one of the
            classes in :mod:`sensoryforge.gui.circuit.nodes`).
        param_name: The setting to change. For :class:`SensorArrayNode` and
            :class:`StimulusNode`, a field name of the underlying
            ``GridConfig``/``StimulusConfig`` dataclass. For
            :class:`RFBankNode`, an ``RFBuilderConfig`` field
            (``"method"``) or, failing that, a key written into
            ``rf.params``. For :class:`ProcessingNode`, ``"method"`` or a
            key in its ``params``. For :class:`CombineNode`, ``"combine"``.
            For :class:`FilterNode`, ``"filter_method"`` or a key in
            ``filter_params``. For :class:`ReadoutNode`, one of its
            top-level fields (``sensoryforge.gui.circuit.nodes.READOUT_FIELDS``)
            or, failing that, a key in its ``model_params`` dict.
        value: The new value.

    Raises:
        SweepValidationError: ``node`` is a type with nothing sweepable
            (:class:`RecordNode`), or ``param_name`` cannot be resolved
            against it.
    """
    if isinstance(node, SensorArrayNode):
        _set_dataclass_field(node.grid, param_name, value)
    elif isinstance(node, StimulusNode):
        _set_dataclass_field(node.stimulus, param_name, value)
    elif isinstance(node, RFBankNode):
        rf_field_names = {f.name for f in dataclasses.fields(node.pop_input.rf)}
        if param_name in rf_field_names and param_name != "params":
            setattr(node.pop_input.rf, param_name, value)
        elif param_name == "gain":
            node.pop_input.gain = value
        else:
            node.pop_input.rf.params[param_name] = value
    elif isinstance(node, ProcessingNode):
        if param_name == "method":
            node.spec["method"] = value
        else:
            node.spec.setdefault("params", {})[param_name] = value
    elif isinstance(node, CombineNode):
        if param_name != "combine":
            raise SweepValidationError(
                f"CombineNode has no field {param_name!r}; only 'combine'."
            )
        node.combine = value
    elif isinstance(node, FilterNode):
        if param_name == "filter_method":
            node.filter_method = value
        else:
            node.filter_params[param_name] = value
    elif isinstance(node, ReadoutNode):
        if param_name in node.fields:
            node.fields[param_name] = value
        else:
            model_params = node.fields.get("model_params")
            if model_params is None:
                model_params = {}
                node.fields["model_params"] = model_params
            model_params[param_name] = value
    elif isinstance(node, RecordNode):
        raise SweepValidationError(
            "RecordNode has nothing sweepable (it is a bundle destination)."
        )
    else:
        raise SweepValidationError(f"Cannot set a parameter on {node!r}.")


def _read_back(config_dict: dict, node_name: str, param_name: str) -> Any:
    """Best-effort read of the swept value back out of a written ``config.json``
    dict, for the caller to compare against what was intended.

    Looks in the grid/stimulus/population entries named ``node_name`` first
    (direct field), then in that population's ``filter_params``/
    ``model_params``/``inputs[].rf.params`` nested dicts, since a swept value
    may live in any of those depending on the node type (see
    :func:`set_node_param`). Returns ``None`` if nothing matches -- the
    caller decides whether that is fatal.
    """
    for grid in config_dict.get("grids", []):
        if grid.get("name") == node_name and param_name in grid:
            return grid[param_name]
    stim = config_dict.get("stimulus") or {}
    if stim.get("name") == node_name and param_name in stim:
        return stim[param_name]
    for pop in config_dict.get("populations", []):
        candidates = [pop]
        if isinstance(pop.get("model_params"), dict):
            candidates.append(pop["model_params"])
        if isinstance(pop.get("filter_params"), dict):
            candidates.append(pop["filter_params"])
        for inp in pop.get("inputs") or []:
            rf = inp.get("rf") or {}
            if param_name in rf:
                candidates.append(rf)
            if isinstance(rf.get("params"), dict):
                candidates.append(rf["params"])
        for candidate in candidates:
            if param_name in candidate:
                return candidate[param_name]
    return None


def sweep_graph(
    flowchart: Flowchart,
    node_name: str,
    param_name: str,
    values: Sequence[Any],
    output_dir: Union[str, Path],
    *,
    duration_ms: float = 200.0,
) -> List[Path]:
    """Run ``flowchart`` once per value in ``values``, writing one bundle each.

    For each value: set ``param_name`` on the node named ``node_name`` (see
    :func:`set_node_param`), rebuild the config from the whole graph, run it,
    and write the bundle to ``<output_dir>/run_%04d/``. Progress is reported
    per run by the caller iterating with ``enumerate`` over the return value
    is not possible (the whole sweep runs inside this call); GUI callers that
    want per-run progress should call this from a worker thread and poll
    ``output_dir`` or pass a smaller ``values`` list per call -- see
    ``BatchTab._on_run_graph_sweep`` for the threaded wrapper actually used
    by the tab.

    Args:
        flowchart: The Circuit tab's live flowchart. Mutated in place (the
            swept node's config is overwritten on every iteration) and left
            holding the last value in ``values`` when this returns.
        node_name: Name of the node (as it appears in ``flowchart.nodes()``)
            whose setting is swept.
        param_name: The setting name, per :func:`set_node_param`.
        values: The values to sweep over. Must be more than one distinct
            value -- a sweep of one repeated value cannot be distinguished
            from a parameter that silently failed to vary, so it is
            rejected up front.
        output_dir: Parent directory for the per-run bundle directories.
        duration_ms: Stimulus duration in ms, per run.

    Returns:
        The list of bundle directories written, one per value, in order.

    Raises:
        SweepValidationError: ``values`` has fewer than two distinct
            entries, ``node_name`` does not exist in the graph, or a run's
            own ``config.json`` does not record the value that was just set
            for it (the exact silent-pinning failure this function exists
            to catch), or two runs recorded the same value.
    """
    if len(set(_hashable(v) for v in values)) < 2:
        raise SweepValidationError(
            f"sweep_graph needs at least two distinct values, got {list(values)!r}."
        )

    nodes = flowchart.nodes()
    if node_name not in nodes:
        raise SweepValidationError(
            f"No node named {node_name!r} in the graph (have: {sorted(nodes)})."
        )
    node = nodes[node_name]

    output_dir = Path(output_dir)
    bundle_dirs: List[Path] = []
    recorded_values: List[Any] = []
    for i, value in enumerate(values):
        set_node_param(node, param_name, value)
        bundle_dir = output_dir / f"run_{i:04d}"
        config, _raw_results, _frames, _dt_ms = run_graph_once(
            flowchart, duration_ms=duration_ms, bundle_dir=str(bundle_dir)
        )

        written = _read_back(config.to_dict(), node_name, param_name)
        if written is None:
            raise SweepValidationError(
                f"Could not find {param_name!r} on node {node_name!r} in the "
                f"run's own config -- the sweep may not reach the "
                "simulation for this node/param combination."
            )
        if written != value:
            raise SweepValidationError(
                f"Set {node_name!r}.{param_name!r} = {value!r} but the run's "
                f"own config recorded {written!r} instead -- the sweep did "
                "not reach the simulation."
            )
        recorded_values.append(written)
        bundle_dirs.append(bundle_dir)

    if len(set(_hashable(v) for v in recorded_values)) != len(
        set(_hashable(v) for v in values)
    ):
        raise SweepValidationError(
            f"Swept values {list(values)!r} but the bundles' own configs "
            f"recorded {recorded_values!r} -- the parameter was pinned "
            "instead of varied."
        )
    return bundle_dirs


def _hashable(value: Any) -> Any:
    """Best-effort hashable key for de-duplicating swept values that may be
    lists (e.g. ``StimulusConfig.start``)."""
    if isinstance(value, list):
        return tuple(value)
    return value
