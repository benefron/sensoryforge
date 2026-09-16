"""Node classes for the Circuit tab's node graph (Phase 3, Wave O).

Each node subclasses :class:`pyqtgraph.flowchart.Node` and owns a
``to_config()`` / ``from_config()`` pair that speaks one canonical dataclass
(or dataclass fragment) from :mod:`sensoryforge.config.schema` directly. This
module is the single place documenting how a connected chain of nodes maps
onto a :class:`~sensoryforge.config.schema.PopulationConfig` -- keep the two
in sync here, not by re-deriving the mapping elsewhere.

Graph -> config mapping
------------------------

* :class:`SensorArrayNode` -- one :class:`~sensoryforge.config.schema.GridConfig`.
  No inputs; one output terminal per named channel (``GridConfig.channels``).
* :class:`StimulusNode` -- the single
  :class:`~sensoryforge.config.schema.StimulusConfig` the config carries. No
  inputs; one output terminal (``"Out"``) that is channel-typed only in the
  sense that it is meant to connect to a :class:`SensorArrayNode`'s matching
  channel terminal for layout purposes -- the schema has one stimulus per
  config today, so only one ``StimulusNode`` is meaningful per graph.
* :class:`RFBankNode` -- one :class:`~sensoryforge.config.schema.PopulationInput`
  (via its nested :class:`~sensoryforge.config.schema.RFBuilderConfig`). One
  input terminal (``"Channel"``) that connects to a ``SensorArrayNode``
  output, one output terminal (``"Drive"``).
* :class:`ProcessingNode` -- one processing-layer spec
  (``{"method": ..., "params": {...}}``) that is inserted into a
  :class:`RFBankNode`'s owning ``PopulationInput.processing`` list when it
  sits between a ``SensorArrayNode`` and an ``RFBankNode``. One input, one
  output, both named ``"Channel"``.
* :class:`CombineNode` -- ``PopulationConfig.combine`` ("sum" or "concat").
  Many ``"Drive"`` inputs (dynamically named ``Drive0``, ``Drive1``, ...),
  one ``"Drive"`` output. A population with exactly one ``RFBankNode`` needs
  no ``CombineNode`` at all; ``combine`` then defaults to ``"sum"``.
* :class:`FilterNode` -- ``PopulationConfig.filter_method`` and
  ``filter_params``. One ``"Drive"`` input, one ``"Filtered"`` output.
* :class:`ReadoutNode` -- every remaining scalar/",dict field of
  :class:`~sensoryforge.config.schema.PopulationConfig` that is not owned by
  one of the nodes above (name, neuron_type, neuron_model, model_params,
  dsl_config, readout, noise_*, color, visible, enabled, input_gain, seed,
  neuron_arrangement/rows/cols/jitter, target_layers). One ``"Filtered"``
  input, one ``"Spikes"`` output. Exactly one per population -- it is the
  node ``serialise.graph_to_config`` walks backward from to reconstruct a
  population's chain.
* :class:`RecordNode` -- a bundle destination. The schema has no dedicated
  field for this, so it round-trips through
  ``SensoryForgeConfig.metadata["record_output_dir"]``. Any number of
  ``"In"`` inputs, no outputs.

A population in the config corresponds to a connected chain from one or more
``RFBankNode``s (each optionally preceded by ``ProcessingNode``s) through an
optional ``CombineNode`` to one ``FilterNode`` and one ``ReadoutNode``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pyqtgraph.flowchart import Node

from sensoryforge.config.schema import (
    GridConfig,
    PopulationInput,
    RFBuilderConfig,
    StimulusConfig,
)


def _channel_terminals(channels: List[str]) -> Dict[str, Dict[str, str]]:
    return {ch: {"io": "out"} for ch in channels}


class SensorArrayNode(Node):
    """Maps one :class:`GridConfig` onto the graph.

    No inputs. One output terminal per named channel in ``GridConfig.channels``
    (``["value"]`` by default -- a single, unnamed channel).
    """

    nodeName = "SensorArray"

    def __init__(self, name: str) -> None:
        self.grid = GridConfig(name=name)
        super().__init__(name, terminals=_channel_terminals(self.grid.channels))

    def _sync_terminals(self) -> None:
        wanted = set(self.grid.channels)
        have = set(self.outputs().keys())
        for extra in have - wanted:
            self.removeTerminal(extra)
        for missing in wanted - have:
            self.addOutput(missing)

    def to_config(self) -> GridConfig:
        """Return the :class:`GridConfig` this node represents."""
        return self.grid

    def from_config(self, grid: GridConfig) -> None:
        """Load this node's state from a :class:`GridConfig`."""
        self.grid = grid
        self._sync_terminals()

    def process(self, **kwargs):  # pragma: no cover - flowchart runtime hook
        return {ch: None for ch in self.grid.channels}


class StimulusNode(Node):
    """Maps the config's one :class:`StimulusConfig` onto the graph."""

    nodeName = "Stimulus"

    def __init__(self, name: str) -> None:
        self.stimulus = StimulusConfig(name=name)
        super().__init__(name, terminals={"Out": {"io": "out"}})

    def to_config(self) -> StimulusConfig:
        """Return the :class:`StimulusConfig` this node represents."""
        return self.stimulus

    def from_config(self, stimulus: StimulusConfig) -> None:
        """Load this node's state from a :class:`StimulusConfig`."""
        self.stimulus = stimulus

    def process(self, **kwargs):  # pragma: no cover - flowchart runtime hook
        return {"Out": None}


class RFBankNode(Node):
    """Maps one :class:`PopulationInput` (grid/channel + RF builder) onto the graph."""

    nodeName = "RFBank"

    def __init__(self, name: str) -> None:
        self.pop_input = PopulationInput(grid="")
        super().__init__(
            name, terminals={"Channel": {"io": "in"}, "Drive": {"io": "out"}}
        )

    def to_config(self) -> PopulationInput:
        """Return the :class:`PopulationInput` this node represents."""
        return self.pop_input

    def from_config(self, pop_input: PopulationInput) -> None:
        """Load this node's state from a :class:`PopulationInput`."""
        self.pop_input = pop_input

    def process(self, **kwargs):  # pragma: no cover - flowchart runtime hook
        return {"Drive": None}


class ProcessingNode(Node):
    """Maps one processing-layer spec (``PROCESSING_REGISTRY`` entry) onto the graph."""

    nodeName = "Processing"

    def __init__(self, name: str) -> None:
        self.spec: Dict[str, Any] = {"method": "identity", "params": {}}
        super().__init__(
            name, terminals={"Channel": {"io": "in"}, "Out": {"io": "out"}}
        )

    def to_config(self) -> Dict[str, Any]:
        """Return the processing-layer spec dict this node represents."""
        return dict(self.spec)

    def from_config(self, spec: Dict[str, Any]) -> None:
        """Load this node's state from a processing-layer spec dict."""
        self.spec = {
            "method": spec.get("method", "identity"),
            "params": dict(spec.get("params") or {}),
        }

    def process(self, **kwargs):  # pragma: no cover - flowchart runtime hook
        return {"Out": None}


class CombineNode(Node):
    """Maps ``PopulationConfig.combine`` onto the graph.

    ``Drives`` is a multi-connection input terminal -- any number of
    ``RFBankNode``s (or ``ProcessingNode`` chains) connect to it. The
    connection count is graph structure, not stored state, so
    ``to_config``/``from_config`` only carry the combine mode.
    """

    nodeName = "Combine"

    def __init__(self, name: str) -> None:
        self.combine: str = "sum"
        super().__init__(
            name,
            terminals={"Drives": {"io": "in", "multi": True}, "Drive": {"io": "out"}},
        )

    def to_config(self) -> str:
        """Return the ``combine`` mode string this node represents."""
        return self.combine

    def from_config(self, combine: str) -> None:
        """Load this node's state from a ``combine`` mode string."""
        self.combine = combine

    def process(self, **kwargs):  # pragma: no cover - flowchart runtime hook
        return {"Drive": None}


class FilterNode(Node):
    """Maps ``PopulationConfig.filter_method``/``filter_params`` onto the graph."""

    nodeName = "Filter"

    def __init__(self, name: str) -> None:
        self.filter_method: str = "none"
        self.filter_params: Dict[str, Any] = {}
        super().__init__(
            name, terminals={"Drive": {"io": "in"}, "Filtered": {"io": "out"}}
        )

    def to_config(self) -> Dict[str, Any]:
        """Return ``{"filter_method": ..., "filter_params": ...}``."""
        return {
            "filter_method": self.filter_method,
            "filter_params": dict(self.filter_params),
        }

    def from_config(self, data: Dict[str, Any]) -> None:
        """Load this node's state from ``{"filter_method": ..., "filter_params": ...}``."""
        self.filter_method = data.get("filter_method", "none")
        self.filter_params = dict(data.get("filter_params") or {})

    def process(self, **kwargs):  # pragma: no cover - flowchart runtime hook
        return {"Filtered": None}


# Fields of PopulationConfig owned by ReadoutNode -- every field NOT owned by
# an RFBankNode/ProcessingNode (rf inputs), a CombineNode (combine) or a
# FilterNode (filter_method/filter_params). Kept as an explicit tuple so a
# schema change that adds a field is a loud KeyError here, not silent data
# loss in serialise.py.
READOUT_FIELDS = (
    "name",
    "neuron_type",
    "neuron_arrangement",
    "neurons_per_row",
    "neuron_rows",
    "neuron_cols",
    "neuron_jitter_factor",
    "neuron_model",
    "model_params",
    "dsl_config",
    "readout",
    "solver_config",
    "noise_std",
    "noise_mean",
    "noise_seed",
    "color",
    "visible",
    "enabled",
    "input_gain",
    "seed",
    "target_layers",
)


class ReadoutNode(Node):
    """Maps the remaining :class:`PopulationConfig` fields onto the graph.

    Exactly one per population; the node ``serialise.graph_to_config`` walks
    backward from to reconstruct the whole chain.
    """

    nodeName = "Readout"

    def __init__(self, name: str) -> None:
        self.fields: Dict[str, Any] = {f: None for f in READOUT_FIELDS}
        self.fields["name"] = name
        super().__init__(
            name, terminals={"Filtered": {"io": "in"}, "Spikes": {"io": "out"}}
        )

    def to_config(self) -> Dict[str, Any]:
        """Return a dict of the population-level fields this node owns."""
        return dict(self.fields)

    def from_config(self, data: Dict[str, Any]) -> None:
        """Load this node's state from a dict of population-level fields."""
        self.fields = {f: data.get(f) for f in READOUT_FIELDS}

    def process(self, **kwargs):  # pragma: no cover - flowchart runtime hook
        return {"Spikes": None}


class RecordNode(Node):
    """A bundle destination. Round-trips through
    ``SensoryForgeConfig.metadata["record_output_dir"]`` since the schema
    has no dedicated field for it.
    """

    nodeName = "Record"

    def __init__(self, name: str) -> None:
        self.output_dir: Optional[str] = None
        super().__init__(name, terminals={"In": {"io": "in", "multi": True}})

    def to_config(self) -> Dict[str, Any]:
        """Return ``{"output_dir": ...}``."""
        return {"output_dir": self.output_dir}

    def from_config(self, data: Dict[str, Any]) -> None:
        """Load this node's state from ``{"output_dir": ...}``."""
        self.output_dir = data.get("output_dir")

    def process(self, **kwargs):  # pragma: no cover - flowchart runtime hook
        return {}


NODE_CLASSES = {
    cls.nodeName: cls
    for cls in (
        SensorArrayNode,
        StimulusNode,
        RFBankNode,
        ProcessingNode,
        CombineNode,
        FilterNode,
        ReadoutNode,
        RecordNode,
    )
}


def build_node_library():
    """Build a :class:`pyqtgraph.flowchart.NodeLibrary` with every Circuit node type."""
    from pyqtgraph.flowchart.NodeLibrary import NodeLibrary

    lib = NodeLibrary()
    for node_name, cls in NODE_CLASSES.items():
        lib.addNodeType(cls, [("Circuit",)])
    return lib
