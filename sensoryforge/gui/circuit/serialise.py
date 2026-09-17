"""Graph <-> config conversion for the Circuit tab (Phase 3, Wave O, O3).

``graph_to_config`` walks a :class:`~pyqtgraph.flowchart.Flowchart` built from
``sensoryforge.gui.circuit.nodes`` node types and reconstructs a
:class:`~sensoryforge.config.schema.SensoryForgeConfig`; ``config_to_graph``
lays a config back out as a graph, deterministically, so loading the same
config twice gives the same picture.

Every :class:`~sensoryforge.config.schema.PopulationConfig` field is owned by
exactly one node (see ``sensoryforge.gui.circuit.nodes`` module docstring for
the table) with one exception, disclosed here: the schema has no node for
``SensoryForgeConfig.simulation``/``metadata``, so :class:`RecordNode` carries
them too (its docstring explains why).

``PopulationConfig.inputs``/``combine`` fidelity (M1's sugar-field collapse)
is preserved by round-tripping every reconstructed population through
``to_dict()``/``from_dict()`` once, the same normalisation a config loaded
from YAML already goes through -- this is what makes a config that started
in its pre-Wave-M flat/sugar form (``target_grid``, ``innervation_method``,
...) compare equal after a graph round trip instead of ending up with an
equivalent but differently-shaped ``inputs=[...]`` list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pyqtgraph.flowchart import Flowchart, Node

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    PopulationInput,
    SensoryForgeConfig,
    SimulationConfig,
    StimulusConfig,
    _input_is_sugar_shaped,
)
from sensoryforge.gui.circuit.nodes import (
    CombineNode,
    FilterNode,
    ProcessingNode,
    RFBankNode,
    ReadoutNode,
    RecordNode,
    SensorArrayNode,
    StimulusNode,
)


class GraphValidationError(ValueError):
    """Raised by :func:`graph_to_config` when the graph does not describe a
    valid config: a dangling terminal, a readout with no filter, a filter
    with no drive, or an RF bank with no channel source. Names the offending
    node and the problem, so the GUI can show a message instead of a
    traceback.
    """


def _connected_source(node: Node, terminal_name: str) -> Optional[Any]:
    """Return the single output ``Terminal`` feeding ``node``'s ``terminal_name``
    input, or ``None`` if it is unconnected."""
    term = node[terminal_name]
    connected = list(term.connections().keys())
    if not connected:
        return None
    return connected[0]


def _connected_sources(node: Node, terminal_name: str) -> List[Any]:
    """Return every output ``Terminal`` feeding a multi-input terminal."""
    term = node[terminal_name]
    return list(term.connections().keys())


def _trace_channel(rf_node: RFBankNode) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Walk backward from an ``RFBankNode``'s ``Channel`` input through zero or
    more ``ProcessingNode``s to the originating ``SensorArrayNode``.

    Returns:
        ``(grid_name, channel_name, processing_specs)`` -- ``processing_specs``
        ordered from the sensor array to the RF bank.
    """
    processing_specs: List[Dict[str, Any]] = []
    source = _connected_source(rf_node, "Channel")
    if source is None:
        raise GraphValidationError(
            f"RFBankNode {rf_node.name()!r} has a dangling 'Channel' input "
            "(connect it to a SensorArrayNode channel output)."
        )
    current_node = source.node()
    current_term = source
    while isinstance(current_node, ProcessingNode):
        processing_specs.insert(0, current_node.to_config())
        upstream = _connected_source(current_node, "Channel")
        if upstream is None:
            raise GraphValidationError(
                f"ProcessingNode {current_node.name()!r} has a dangling "
                "'Channel' input."
            )
        current_node = upstream.node()
        current_term = upstream
    if not isinstance(current_node, SensorArrayNode):
        raise GraphValidationError(
            f"RFBankNode {rf_node.name()!r}'s channel chain does not "
            f"originate from a SensorArrayNode (found {current_node!r})."
        )
    return current_node.name(), current_term.name(), processing_specs


def graph_to_config(flowchart: Flowchart) -> SensoryForgeConfig:
    """Reconstruct a :class:`SensoryForgeConfig` from a Circuit tab flowchart."""
    nodes = flowchart.nodes()

    grids: List[GridConfig] = []
    for node in nodes.values():
        if isinstance(node, SensorArrayNode):
            grids.append(node.to_config())

    stimulus = StimulusConfig()
    for node in nodes.values():
        if isinstance(node, StimulusNode):
            stimulus = node.to_config()
            break

    populations: List[PopulationConfig] = []
    for node in nodes.values():
        if not isinstance(node, ReadoutNode):
            continue

        filter_source = _connected_source(node, "Filtered")
        if filter_source is None:
            raise GraphValidationError(
                f"ReadoutNode {node.name()!r} has no connected FilterNode "
                "(a readout with no filter)."
            )
        filter_node = filter_source.node()
        if not isinstance(filter_node, FilterNode):
            raise GraphValidationError(
                f"ReadoutNode {node.name()!r}'s 'Filtered' input is not fed "
                f"by a FilterNode (found {filter_node!r})."
            )
        filter_cfg = filter_node.to_config()

        drive_source = _connected_source(filter_node, "Drive")
        if drive_source is None:
            raise GraphValidationError(
                f"FilterNode {filter_node.name()!r} has a dangling 'Drive' " "input."
            )
        drive_node = drive_source.node()

        if isinstance(drive_node, CombineNode):
            combine = drive_node.to_config()
            drive_sources = _connected_sources(drive_node, "Drives")
            if not drive_sources:
                raise GraphValidationError(
                    f"CombineNode {drive_node.name()!r} has no connected "
                    "RFBankNode inputs."
                )
            rf_nodes = [src.node() for src in drive_sources]
        elif isinstance(drive_node, RFBankNode):
            combine = "sum"
            rf_nodes = [drive_node]
        else:
            raise GraphValidationError(
                f"FilterNode {filter_node.name()!r}'s 'Drive' input is fed "
                f"by neither an RFBankNode nor a CombineNode (found "
                f"{drive_node!r})."
            )

        pop_inputs: List[PopulationInput] = []
        for rf_node in rf_nodes:
            if not isinstance(rf_node, RFBankNode):
                raise GraphValidationError(
                    f"Expected an RFBankNode feeding {filter_node.name()!r}, "
                    f"found {rf_node!r}."
                )
            grid_name, channel_name, processing_specs = _trace_channel(rf_node)
            base = rf_node.to_config()
            pop_inputs.append(
                PopulationInput(
                    grid=grid_name,
                    channel=channel_name,
                    rf=base.rf,
                    gain=base.gain,
                    layers=base.layers,
                    processing=processing_specs,
                )
            )

        readout_fields = node.to_config()

        # PopulationConfig.__post_init__ forbids a non-empty `inputs` list
        # together with a non-default legacy "sugar" builder knob (module
        # docstring). A single sugar-shaped RFBankNode is exactly the case
        # those knobs describe, so build the population in its flat/legacy
        # form (inputs=[], target_grid/target_layers/innervation_method set
        # directly) instead of passing both; any other shape (more than one
        # input, or a non-default channel/gain/processing) means those
        # legacy knobs must already be at their defaults on this population
        # (the graph could only have come from a config that itself passed
        # this same check), so passing them through readout_fields alongside
        # `inputs` is safe.
        if len(pop_inputs) == 1 and _input_is_sugar_shaped(pop_inputs[0]):
            single = pop_inputs[0]
            pop = PopulationConfig(
                **readout_fields,
                filter_method=filter_cfg["filter_method"],
                filter_params=filter_cfg["filter_params"],
                target_grid=single.grid,
                target_layers=single.layers,
                innervation_method=single.rf.method,
            )
        else:
            pop = PopulationConfig(
                **readout_fields,
                filter_method=filter_cfg["filter_method"],
                filter_params=filter_cfg["filter_params"],
                inputs=pop_inputs,
                combine=combine,
            )
        # Normalise through to_dict()/from_dict() once so a single
        # sugar-shaped input collapses to the flat/legacy fields exactly the
        # way a config loaded from YAML would (see module docstring).
        pop = PopulationConfig.from_dict(pop.to_dict())
        populations.append(pop)

    simulation = SimulationConfig()
    metadata: Dict[str, Any] = {}
    for node in nodes.values():
        if isinstance(node, RecordNode):
            record = node.to_config()
            if record["simulation"]:
                simulation = SimulationConfig.from_dict(record["simulation"])
            metadata = dict(record["metadata"])
            if record["output_dir"]:
                metadata["record_output_dir"] = record["output_dir"]
            break

    return SensoryForgeConfig(
        grids=grids,
        populations=populations,
        stimulus=stimulus,
        simulation=simulation,
        metadata=metadata,
    )


def config_to_graph(config: SensoryForgeConfig, flowchart: Flowchart) -> None:
    """Lay ``config`` out onto ``flowchart``, deterministically, replacing its
    current contents.

    Layout: sensor arrays in a left column (x=0), the stimulus above them
    (x=0, y above the first grid), then one row per population (RF bank(s),
    an optional combine, a filter, a readout) reading left to right starting
    at x=300, and a record node at the far right.
    """
    flowchart.clear()

    row_height = 120
    grid_nodes: Dict[str, SensorArrayNode] = {}
    for i, grid_cfg in enumerate(config.grids):
        node = flowchart.createNode(
            "SensorArray", name=grid_cfg.name, pos=(0, i * row_height)
        )
        node.from_config(grid_cfg)
        grid_nodes[grid_cfg.name] = node

    stim_node = flowchart.createNode(
        "Stimulus",
        name=config.stimulus.name or "Stimulus",
        pos=(0, -row_height),
    )
    stim_node.from_config(config.stimulus)

    pop_row_y = len(config.grids) * row_height + row_height
    for pop in config.populations:
        x = 300
        rf_nodes = []
        for j, pop_input in enumerate(pop.effective_inputs()):
            rf_name = f"{pop.name}__rf{j}"
            rf_node = flowchart.createNode("RFBank", name=rf_name, pos=(x, pop_row_y))
            rf_node.from_config(
                PopulationInput(
                    grid=pop_input.grid,
                    channel=pop_input.channel,
                    rf=pop_input.rf,
                    gain=pop_input.gain,
                    layers=pop_input.layers,
                    processing=[],
                )
            )
            # Wire the channel source: grid -> [processing chain] -> rf bank.
            grid_node = grid_nodes.get(pop_input.grid)
            if grid_node is not None:
                upstream_term = grid_node[pop_input.channel]
                for k, spec in enumerate(pop_input.processing):
                    proc_node = flowchart.createNode(
                        "Processing",
                        name=f"{rf_name}__proc{k}",
                        pos=(x - 100, pop_row_y),
                    )
                    proc_node.from_config(spec)
                    flowchart.connectTerminals(upstream_term, proc_node["Channel"])
                    upstream_term = proc_node["Out"]
                flowchart.connectTerminals(upstream_term, rf_node["Channel"])
            rf_nodes.append(rf_node)
            pop_row_y += row_height

        x += 200
        if len(rf_nodes) > 1:
            combine_node = flowchart.createNode(
                "Combine", name=f"{pop.name}__combine", pos=(x, pop_row_y)
            )
            combine_node.from_config(pop.combine)
            for rf_node in rf_nodes:
                flowchart.connectTerminals(rf_node["Drive"], combine_node["Drives"])
            drive_out = combine_node["Drive"]
            x += 200
        else:
            drive_out = rf_nodes[0]["Drive"]

        filter_node = flowchart.createNode(
            "Filter", name=f"{pop.name}__filter", pos=(x, pop_row_y)
        )
        filter_node.from_config(
            {"filter_method": pop.filter_method, "filter_params": pop.filter_params}
        )
        flowchart.connectTerminals(drive_out, filter_node["Drive"])

        x += 200
        readout_node = flowchart.createNode(
            "Readout", name=pop.name, pos=(x, pop_row_y)
        )
        from sensoryforge.gui.circuit.nodes import READOUT_FIELDS

        readout_node.from_config({f: getattr(pop, f) for f in READOUT_FIELDS})
        flowchart.connectTerminals(filter_node["Filtered"], readout_node["Filtered"])

        pop_row_y += row_height

    record_node = flowchart.createNode(
        "Record", name="Record", pos=(1200, len(config.grids) * row_height)
    )
    record_output_dir = config.metadata.get("record_output_dir")
    record_metadata = {
        k: v for k, v in config.metadata.items() if k != "record_output_dir"
    }
    record_node.from_config(
        {
            "output_dir": record_output_dir,
            "simulation": config.simulation.to_dict(),
            "metadata": record_metadata,
        }
    )


# ---------------------------------------------------------------------------
# View state (F-065)
#
# Node positions are not configuration: two people can arrange the same
# experiment differently and it is still the same experiment. They live in a
# sibling file so the config stays exactly what the CLI would run, and they
# are advisory -- a missing, stale or malformed layout must never stop a
# config from loading, because losing your arrangement is an annoyance and
# failing to open your experiment is not.
# ---------------------------------------------------------------------------

LAYOUT_SUFFIX = ".layout.json"


def layout_path_for(config_path: Union[str, Path]) -> Path:
    """The layout file that sits beside *config_path*.

    Args:
        config_path: Path to the config file, with or without its suffix.

    Returns:
        ``<config>.layout.json`` next to it.
    """
    config_path = Path(config_path)
    return config_path.with_suffix(config_path.suffix + LAYOUT_SUFFIX)


def save_layout(flowchart: Flowchart, config_path: Union[str, Path]) -> Path:
    """Write the graph's node positions beside *config_path*.

    Args:
        flowchart: The flowchart whose arrangement to record.
        config_path: The config file these positions belong to.

    Returns:
        The layout file written.
    """
    positions: Dict[str, List[float]] = {}
    for entry in flowchart.saveState().get("nodes", []):
        name = entry.get("name")
        pos = entry.get("pos")
        if name is None or pos is None:
            continue
        positions[str(name)] = [float(pos[0]), float(pos[1])]

    path = layout_path_for(config_path)
    path.write_text(
        json.dumps({"version": 1, "positions": positions}, indent=2, sort_keys=True)
    )
    return path


def apply_layout(flowchart: Flowchart, config_path: Union[str, Path]) -> int:
    """Restore node positions recorded beside *config_path*, if any.

    Advisory by contract: a missing file, unreadable JSON, an unexpected
    shape or a name that is no longer in the graph are all no-ops rather
    than errors. Nodes without a recorded position keep the deterministic
    placement :func:`config_to_graph` gave them.

    Args:
        flowchart: The flowchart to rearrange, already populated.
        config_path: The config file whose layout to look for.

    Returns:
        How many nodes were moved.
    """
    path = layout_path_for(config_path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return 0
    if not isinstance(payload, dict):
        return 0
    positions = payload.get("positions")
    if not isinstance(positions, dict):
        return 0

    nodes = flowchart.nodes()
    moved = 0
    for name, pos in positions.items():
        node = nodes.get(name)
        if node is None:
            continue
        try:
            x, y = float(pos[0]), float(pos[1])
        except (TypeError, ValueError, IndexError):
            continue
        item = node.graphicsItem()
        if item is None:
            continue
        item.setPos(x, y)
        moved += 1
    return moved
