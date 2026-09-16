# Extending: when a plugin needs more than `get_param_spec()` gives it

`get_param_spec()` is the Phase 1g contract every component already implements
(CLAUDE.md, "ParamSpec and get_param_spec()"): a list of typed parameters a caller
can render however it wants. The Circuit tab's inspector
(`sensoryforge/gui/circuit/inspector.py`, Wave P, P1) renders that list as a form —
spin box, checkbox, combo box, grouped, with advanced params hidden unless Expert
mode is on. **A plugin filter, neuron model, stimulus, RF builder or processing layer
gets that form for free, with no GUI code of its own**, as long as its node type
already exists (`SensorArray`, `Stimulus`, `RFBank`, `Processing`, `Filter`,
`Readout` — Wave O's fixed structural set) and its component is registered in the
matching registry. `docs/examples/circuit_plugin_param_form.py` proves this, part 1
below.

This page is about the part that is **not** free today, disclosed rather than
glossed over: a **custom preview** beyond the generic per-node-type ones Circuit
ships (a receptor scatter for `SensorArray`, a single-frame image for `Stimulus`, a
label for `RFBank`), or an **entirely new structural node type**.

## What is and is not a plugin extension point

| Want | Mechanism | Plugin-only? |
|---|---|---|
| A settings form for a new filter/neuron/stimulus/RF builder/processing layer/grid arrangement | Register with the matching `*_REGISTRY` + implement `get_param_spec()` | **Yes** |
| A different preview for a new component fitting one of the six existing node types | — | **No, see below** |
| A wholly new node type (a new box shape on the canvas) | — | **No, see below** |

Checked directly against this checkout, not assumed from the spec:

- `sensoryforge.gui.circuit.nodes.NODE_CLASSES` is a fixed dict built from eight
  hard-coded classes (`nodes.py:377-389`) — there is no registry a plugin package
  registers a new *node type* into. Adding a node type means adding a class here, in
  this repository.
- `sensoryforge.gui.circuit.inspector._build_visualisation` dispatches on a
  hard-coded set of three strings (`"SensorArray"`, `"Stimulus"`, `"RFBank"`,
  `inspector.py:420-426`) — not on the registered component name. Two plugin RF
  builders both get the *same* generic `RFBank` label-only preview
  (`_rf_bank_preview`); there is no hook for a component to supply its own preview
  widget.
- `pyqtgraph.flowchart.Node.ctrlWidget()` exists as pyqtgraph's own generic
  per-node-control-widget hook (Fact P3-a in
  `docs/development/handover/phase3_tasks.md`) — but `build_node_inspector` never
  calls it. A node subclass that overrides `ctrlWidget()` gets nothing extra in the
  Circuit tab's inspector dock today; that override would only matter to code that
  calls `node.ctrlWidget()` directly.

So today, a custom preview or a new node type is an **in-repo** change (contribute to
`sensoryforge/gui/circuit/nodes.py` and `inspector.py` directly, the same
`--in-repo` route the component guide describes for contributing a new registered
component), not something `pip install`-ing a plugin package can add on its own. If
your plugin only needs a settings form, stop reading here — `get_param_spec()`
already does it. If it needs a genuinely different preview, the worked example below
is the smallest version of the in-repo change, exercised the same way
`_build_visualisation` is exercised for the shipped node types.

## Worked example

`docs/examples/circuit_plugin_param_form.py`, run by `tests/docs/test_docs_examples.py`
like every other example under `docs/examples/`, does both halves:

1. Registers a plugin RF builder (`DemoTemplateInnervation`, three parameters via
   `get_param_spec()`) with `INNERVATION_REGISTRY`, places it on an `RFBankNode`, and
   builds that node's real Circuit-tab inspector — asserting the spin box for one of
   the three parameters exists and is wired to the node's own `rf.params` dict with
   no GUI code written for this component. This is part of the *existing* extension
   point; no repository change was needed for it.
2. Demonstrates the **in-repo** path for a custom preview: a small
   `_demo_visualisation(node_type, node)` function shaped exactly like
   `inspector._build_visualisation`, dispatching on a node type string the way a real
   change to that function would, returning a `QLabel` derived from the node's own
   state (its registered component name) rather than the generic placeholder. It is
   called directly, standing in for what a contributor would splice into
   `inspector.py`'s real dispatch table — the example does not monkeypatch the
   shipped function, since a plugin is not able to alter another package's source
   either; it shows the shape the change takes.

Run it directly:

```bash
QT_QPA_PLATFORM=offscreen python docs/examples/circuit_plugin_param_form.py
```

## If you do need the in-repo change

1. New node type: add a `Node` subclass to `sensoryforge/gui/circuit/nodes.py`
   following the pattern every existing node uses (terminals in `__init__`, a
   `to_config()`/`from_config()` pair speaking one config dataclass or fragment,
   `process()` returning `None` placeholders — the flowchart runtime never actually
   executes node graphs the way pyqtgraph's own examples do; `graph_to_config`/
   `run_graph_once` do the real work), add it to `NODE_CLASSES`, and update the
   node-to-config mapping table in that module's docstring — the one place the graph
   and the dataclasses are kept in sync by hand.
2. New preview for an existing node type: extend
   `sensoryforge.gui.circuit.inspector._build_visualisation` — dispatch on
   `node_type` (or, for a genuinely per-component preview, also on the registered
   component name read off the node, e.g. `node.pop_input.rf.method`) and return a
   `QWidget`. Keep it defensive the way every shipped preview is
   (`_grid_preview`/`_stimulus_preview` both catch broadly and show
   `"Preview unavailable: ..."` rather than crashing the inspector on a bad
   parameter combination mid-edit).
3. Add a unit test alongside the existing `tests/unit/test_circuit_inspector*.py`
   files, following their pattern (build the node, call
   `build_node_inspector`/`build_param_form` directly, assert on the returned
   widget's children — no full GUI needed).

Whichever you add, `graph_to_config`/`config_to_graph` in
`sensoryforge/gui/circuit/serialise.py` do not need to change unless the new node
type owns a config field no existing node already owns — see
[The graph](../concepts/the_graph.md) for the full config-is-the-contract rule this
falls under.
