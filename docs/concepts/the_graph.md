# The graph

Phase 3 added a node-graph editor, the **Circuit tab**, as the GUI's entry point to
building an experiment. This page is about what the graph *is*: a view onto a
`SensoryForgeConfig`, not a fourth representation alongside it. See
[the GUI walkthrough](../user_guide/gui_walkthrough.md) for how to build one, and
`sensoryforge/gui/circuit/nodes.py`'s module docstring for the authoritative
node-to-dataclass mapping table this page summarises.

## The config is still the source of truth

Phase 1 and 2 made `SensoryForgeConfig` the single thing the CLI, the batch executor
and `SimulationEngine` all read. The GUI's five original tabs pre-date that: each
owned a slice of the config and handed dictionaries to the others, so "which sensor
feeds which receptive field feeds which readout" existed only implicitly, split
across tabs.

The graph does not change what the source of truth is — it changes what editing it
looks like. `sensoryforge.gui.circuit.serialise` has exactly two functions:

```python
def graph_to_config(flowchart: Flowchart) -> SensoryForgeConfig: ...
def config_to_graph(config: SensoryForgeConfig, flowchart: Flowchart) -> None: ...
```

Everything the Circuit tab does — the Run button, the Batch tab's graph-sweep path
(Wave Q, Q1), the CLI round trip below — goes through a `SensoryForgeConfig` built by
`graph_to_config`. There is no direct graph-to-engine path and no separate
graph-shaped YAML: `graph_to_config(fc).to_yaml()` is byte-for-byte the same YAML
`sensoryforge run` accepts, because it is the same object every other entry point
produces.

**Guardrail (Phase 3 guardrail 1, still in force):** a graph feature that cannot be
expressed as a `SensoryForgeConfig` field is not a feature. Extend the schema first,
then teach a node about the new field — never the other way around.

## One node, one piece of config

Every node subclasses `pyqtgraph.flowchart.Node` and owns a `to_config()`/
`from_config()` pair speaking one dataclass (or dataclass fragment) from
`sensoryforge.config.schema` directly:

| Node | Config object | Notes |
|---|---|---|
| `SensorArrayNode` | `GridConfig` | one output terminal per named channel |
| `StimulusNode` | `StimulusConfig` | one per config today (the schema has one stimulus) |
| `RFBankNode` | `PopulationInput` (its `RFBuilderConfig`) | one `Channel` input, one `Drive` output |
| `ProcessingNode` | one processing-layer spec dict | optional, sits between a channel source and an `RFBankNode` |
| `CombineNode` | `PopulationConfig.combine` | only needed when a population has more than one input |
| `FilterNode` | `filter_method`/`filter_params` | one `Drive` input, one `Filtered` output |
| `ReadoutNode` | every remaining `PopulationConfig` field | see `READOUT_FIELDS` in `nodes.py` for the exact list |
| `RecordNode` | bundle destination, plus `SensoryForgeConfig.simulation`/`metadata` | see below |

A population is a connected **chain**: one or more `RFBankNode`s (each optionally
preceded by `ProcessingNode`s) through an optional `CombineNode`, into exactly one
`FilterNode`, into exactly one `ReadoutNode`. `graph_to_config` walks the graph
backward from each `ReadoutNode` to reconstruct that chain — a `ReadoutNode` with no
upstream `FilterNode`, or an `RFBankNode` whose `Channel` input does not trace back to
a `SensorArrayNode`, is a `GraphValidationError` naming the offending node, not a
silent gap in the exported config.

One honest exception: the schema has no node of its own for
`SensoryForgeConfig.simulation` (device, dt) or `.metadata`. Wave O's `RecordNode` —
already the graph's "where does the bundle go" node — carries those too, because a
lossless round trip is impossible without *some* node holding them and the spec only
ever described `RecordNode` loosely, as "a bundle destination." This is disclosed in
`RecordNode`'s own docstring, not hidden behind a plausible-looking but incomplete
mapping.

## What is graph structure, not config

Two things about a graph are true of the picture but not of the config it represents:

- **Connections that imply structure but hold no data of their own** — e.g. a
  `CombineNode`'s number of `Drives` inputs is read off the connection set, not stored
  as a field; `to_config()`/`from_config()` on a `CombineNode` only carry the
  `"sum"`/`"concat"` mode string.
- **Node positions.** Phase 3 guardrail 2: ephemeral view state (positions, zoom,
  which panels are collapsed) is not a config field. `config_to_graph` lays a config
  out deterministically (sensor arrays in a left column, the stimulus above them, one
  row per population left to right, `RecordNode` on the far right) so loading the
  same config twice gives the same picture, which is what makes the graph safe to
  treat as disposable. The Wave O spec additionally describes saving dragged-node
  positions into a sibling `<config>.layout.json` file; as recorded in the GUI
  walkthrough's discrepancy note, that persistence was never wired up in this
  checkout — positions are not preserved across a save/reload today, only the
  deterministic layout is.

## Round-tripping

Both directions are tested against every canonical config this repository ships —
`examples/*.yml` that are canonical-shaped, plus all three presets under
`sensoryforge/presets/`:

- `config -> graph -> config` gives back a `SensoryForgeConfig` equal to the original
  (dataclass `==`, which recurses into every nested list/dataclass/dict) and
  byte-identical YAML text.
- `graph -> config -> graph` gives back the same node/type set and the same
  connection set — not merely an equivalent-looking picture.

`tests/unit/test_circuit_roundtrip.py` (Wave O) and
`tests/gui/test_circuit_graph_end_to_end.py` (Wave R) both exercise this; see the
latter for
the additional GUI-export-through-the-CLI-and-back chain described in the
walkthrough.

## Why a graph, and why now

Phase 2 made a population able to read two channels from two grids through two
different RF builders (`PopulationConfig.inputs`, Wave M). A form-per-tab layout
cannot represent that shape at all — there is no single "the" population form once a
population has more than one input. A graph represents it directly: two `RFBankNode`s
feeding one `CombineNode`. The graph is not a nicer way to edit the old five-tab
model; it is the only GUI design that survives Phase 2's own generality.
