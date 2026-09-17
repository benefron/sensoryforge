# Phase 3 handover — the node-graph GUI

Prepared 2026-09-16. The approved plan is `docs/developer_guide/roadmap_v1.md` ("Phase 3 — node-graph
GUI"). Phase 3 starts only after Phase 2's exit criteria pass
(`docs/development/handover/phase2_tasks.md` section 6). Open findings are in `docs_root/LEDGER.md`;
the session-start hook injects a digest.

---

## How Phase 3 is run

Same orchestration as Phase 2: each wave goes to an implementation agent in its own git worktree, is
reviewed, then merged into the integration branch. Waves are O, P, Q, R and are **sequential** — they
all touch the GUI package and there is no clean file split between them.

Every agent reads this file's sections 1 to 3 and section 2 ("Guardrails") of
`docs/development/handover/phase1_tasks.md`, recreates the memory watchdog from that file's appendix,
commits one task per commit with single-line ledger trailers, never pushes, and reports each task's
commit hash with the output that proves its "Done when".

---

## 1. What Phase 3 delivers, and why a node graph

Phases 1 and 2 made the config the single source of truth: the CLI, the batch executor and the engine
all read one `SensoryForgeConfig`, and a run produces one bundle. The GUI is the part that has not
caught up. Its five tabs each own a slice of the config and hand dictionaries to each other, so the
shape of an experiment — which sensor feeds which receptive field feeds which readout — exists only
implicitly, scattered across three tabs.

A node graph makes that shape the thing you edit. It is also the only GUI design that survives Phase
2's generality: once a population can read two channels from two grids through two different
builders, a form-per-tab layout cannot express the experiment at all, while a graph expresses it
directly.

| Wave | Delivers |
|---|---|
| **O** | The Circuit tab: node classes mapping one-to-one onto the config dataclasses, and a lossless graph-to-config-to-graph round trip |
| **P** | The inspector: node parameters rendered from `get_param_spec()`, so a plugin appears in the GUI with no GUI code |
| **Q** | The graph drives the rest: batch sweeps built from the live graph, channel selectors in the Stimulus and Visualization tabs, and the dead GUI modules deleted (F-019) |
| **R** | Documentation, the GUI walkthrough, and the Phase 3 test suite |

---

## 2. Facts the agent must know before touching the GUI

**Fact P3-a (verified 2026-09-16).** `pyqtgraph` 0.14.0 is installed in the project environment and
`pyqtgraph.flowchart` imports cleanly. `Flowchart` and `Node` are both available;
`Flowchart.saveState()` and `Flowchart.restoreState()` exist, and `Node.ctrlWidget()` is the hook for
a per-node control panel. `Node.addInput(name="Input", **args)` is the terminal API. No new
dependency is needed for Phase 3.

**Fact P3-b (verified 2026-09-16).** The GUI already converts both ways between its own dictionaries
and the canonical schema: `SensoryForgeWindow._gui_config_to_canonical` (`gui/main.py:453-594`)
builds a `SensoryForgeConfig` from the three tabs' `get_config()` dictionaries, and
`_canonical_to_gui_config` (`:596`) goes back. Phase 3 adds a third representation (the graph) and
must route it through the *canonical* config, never directly to the GUI dictionaries — two
converters are maintainable, six are not.

**Fact P3-c (verified 2026-09-16, F-035).** With Python's cyclic garbage collector enabled,
`pytest -m gui` segfaults inside `pyqtgraph`'s `ScatterPlotItem.renderSymbol`. `tests/conftest.py`
disables GC for GUI sessions (`:54`) and exits with `os._exit` to skip interpreter teardown (`:57-80`).
Phase 3 adds many more pyqtgraph items; if a new segfault appears, it is F-035's family and the fix is
in the test harness, not in the GUI code. Do not re-enable GC to "fix" a failure.

**Fact P3-d (verified 2026-09-16, F-019).** `gui/protocol_suite_tab.py` (477 lines),
`gui/protocol_backend.py`, `gui/protocol_execution_controller.py` and `gui/neuron_explorer.py` are
imported by no tab — only by two tests. Roughly 3,500 lines. Wave Q deletes them unless a node in
Wave O genuinely needs one, in which case the decision is recorded and F-019 is closed with a note
rather than by deletion.

**Fact P3-e.** The GUI's largest modules are `stimulus_tab.py` (3,957 lines),
`mechanoreceptor_tab.py` (3,571) and `spiking_tab.py` (3,251). Phase 3 must not grow them. New code
goes in `gui/circuit/` as several small modules; existing widgets are *reused* by importing them, not
by copying them into the new tab and not by rewriting them.

---

## 3. Phase 3 guardrails

1. **The config is the contract.** Any graph must serialise to a `SensoryForgeConfig` that the CLI
   runs to the same result, and any config the CLI accepts must load into a graph. A graph feature
   that cannot be expressed in the config is not a feature; extend the schema first.
2. **No new GUI-only state.** If a node has a setting, it is a config field. Ephemeral view state
   (node positions, zoom, collapsed panels) lives in the flowchart's own saved state, kept beside the
   config, never inside it.
3. **Reuse, do not rewrite.** The Mechanoreceptor tab's receptive-field visualisation, the Stimulus
   Designer's preview and the Visualization tab's dock panels are the inspector's building blocks.
4. **Offscreen by default.** Every GUI test runs under `QT_QPA_PLATFORM=offscreen` and the `gui`
   marker. No test may require a visible display.
5. **The watchdog runs on every suite.** GUI sessions have been over 500 MB; a regression that
   doubles it is a finding, not a footnote.
6. **One task, one commit, single-line trailers.** A wrapped trailer is silently dropped by the sync
   hook.

---

## 4. Wave O — the Circuit tab

### O1. Node classes

New package `sensoryforge/gui/circuit/` with one module per node family. Each node subclasses
`pyqtgraph.flowchart.Node`, declares its terminals, and owns a `to_config()` / `from_config()` pair
that speaks the canonical dataclasses directly.

| Node | Config object | Inputs | Outputs |
|---|---|---|---|
| `SensorArrayNode` | `GridConfig` | none | one terminal per channel |
| `StimulusNode` | `StimulusConfig` | none | one channel-typed terminal |
| `RFBankNode` | `RFBuilderConfig` (a `PopulationInput`'s `rf`) | one channel | drive |
| `ProcessingNode` | a `processing` layer entry | one channel | one channel |
| `CombineNode` | `PopulationConfig.combine` | many drives | drive |
| `FilterNode` | `PopulationConfig.filter_method` and params | drive | filtered |
| `ReadoutNode` | `PopulationConfig` neuron fields | filtered | spikes or state |
| `RecordNode` | bundle destination | any | none |

A population in the config corresponds to a connected chain from one or more `RFBankNode`s through an
optional `CombineNode` to one `FilterNode` and one `ReadoutNode`. The mapping is explicit and
documented in the module docstring, because it is the one place where the graph and the dataclasses
could drift.

**Done when:** every node class instantiates headless, declares its terminals, and has a unit test
for `to_config()` and `from_config()` round-tripping one representative object.

### O2. The canvas

`gui/tabs/circuit_tab.py`: a `Flowchart` in a `QWidget`, a node palette on the left listing the node
types, the canvas in the middle and an empty inspector dock on the right (filled in Wave P). Adding,
connecting, moving and deleting nodes works. The tab is added to `SensoryForgeWindow` as the **first**
tab, since it is now the entry point to building an experiment.

**Done when:** a GUI test builds a three-node graph programmatically, connects it, and reads the
connections back.

### O3. Graph to config, config to graph

`gui/circuit/serialise.py` with two functions:

```python
def graph_to_config(flowchart: Flowchart) -> SensoryForgeConfig: ...
def config_to_graph(config: SensoryForgeConfig, flowchart: Flowchart) -> None: ...
```

`config_to_graph` lays out nodes deterministically (sensor arrays in a left column, stimuli above
them, then one row per population) so that loading the same config twice gives the same picture.
View state (positions after the user moves things) is saved via `Flowchart.saveState()` into a
sibling file, `<config>.layout.json`, and is advisory: a missing or stale layout file must never
prevent the config from loading.

Validation lives here: a dangling terminal, a readout with no filter, a combine whose inputs disagree
on neuron count. Each raises a `GraphValidationError` naming the node and the problem, shown in the
GUI as a message rather than a traceback.

**Done when:** a GUI test round-trips every config in `examples/` and both Wave K presets through
graph and back, asserting the resulting `SensoryForgeConfig` is equal to the original, and a second
test asserts graph-to-config-to-graph gives an identical node and edge set. Both must fail on the
Phase 2 exit commit.

### O4. Run from the graph

A Run button on the Circuit tab calls `graph_to_config`, runs `SimulationEngine`, writes a bundle via
the Wave J writer, and emits the same `simulation_finished` signal the Spiking tab emits, so the
Visualization tab receives graph runs with no changes.

**Done when:** a GUI test builds a small graph, runs it, and asserts a bundle directory appears that
`load_bundle` reads.

### Wave O exit

Both suites green under the watchdog, lint and strict docs clean, the round-trip tests passing and
proven to fail on the Phase 2 exit commit.

---

## 5. Wave P — the inspector

### P1. Parameters rendered from `get_param_spec()`

`gui/circuit/inspector.py` builds a widget for the selected node from its component's
`get_param_spec()` list: a spin box for float and int, a checkbox for bool, a combo box when
`choices` is set, the `unit` as a suffix, `tooltip` as hover text, `group` as a section header, and
`advanced` hidden unless Expert mode is on — the same `chk_expert_mode` convention the existing tabs
use. This is the payoff for the Phase 1g contract: a third-party plugin gets a GUI for free.

**Done when:** a test registers a plugin component with three parameter kinds at test time and
asserts the inspector renders the right widget for each, including the enum and the advanced flag.

### P2. Reusing the existing visualisations

The `SensorArrayNode` inspector embeds the Mechanoreceptor tab's grid view; the `RFBankNode`
inspector embeds its receptive-field weight display; the `StimulusNode` inspector embeds the Stimulus
Designer's preview. These are imported and embedded, not reimplemented. Where a widget is too
entangled with its tab to embed, extract it into `gui/widgets/` in its own commit, leaving the tab
using the extracted widget — a refactor with no behaviour change, provable by the existing tests
still passing untouched.

**Done when:** selecting each node type shows its visualisation, and the original tabs still pass
their existing tests with no edits to those tests.

### P3. The palette follows the registries

The node palette lists RF builders from `INNERVATION_REGISTRY`, filters from `FILTER_REGISTRY`,
neurons from `NEURON_REGISTRY`, stimuli from `STIMULUS_REGISTRY`, processing layers from
`PROCESSING_REGISTRY` and grid arrangements from `GRID_REGISTRY`. Nothing is hard-coded.

**Done when:** a plugin package installed into the environment appears in the palette and can be
placed, configured and run without any GUI code having been written for it. This is the same plugin
package Phase 1 used for its install proof.

### Wave P exit

Both suites green under the watchdog, lint and strict docs clean, and the plugin-in-the-palette proof
recorded with its output.

---

## 6. Wave Q — the graph drives the rest, and the dead code goes

### Q1. Batch sweeps from the live graph

`BatchTab` currently sweeps a config *file*. It gains a "sweep this graph" path: parameters are
chosen by clicking a node and picking one of its `ParamSpec`s, ranges are entered in the tab, and the
executor runs one bundle per combination. Progress is emitted per run, not per batch.

**Done when:** a GUI test sweeps one parameter over three values from a graph and asserts three
bundles, each with the swept value recorded in its `config.json`.

### Q2. Channel selectors

The Stimulus Designer gains a channel selector so a stimulus can be authored for a named channel of a
named grid, and the Visualization tab gains one so a multi-channel run can be inspected a channel at
a time. Nothing else in either tab changes.

**Done when:** a GUI test authors two stimuli on two channels and asserts the resulting
`StimulusConfig` objects carry the right channel names.

### Q3. Delete the unwired modules (F-019)

Delete `gui/protocol_suite_tab.py`, `gui/protocol_backend.py`, `gui/protocol_execution_controller.py`
and `gui/neuron_explorer.py`, and the two tests that import them. If Wave O found a genuine use for
any of them, keep that one, wire it, and say so in the commit message instead.

**Done when:** `grep -rn "protocol_suite\|neuron_explorer" sensoryforge tests` returns nothing, both
suites are green, and the commit carries `Closes: F-019`.

### Wave Q exit

Both suites green under the watchdog, lint and strict docs clean, F-019 closed, and the repository
smaller than it started.

---

## 7. Wave R — documentation and tests

### R1. The GUI walkthrough

`docs/user_guide/gui_walkthrough.md` rewritten around the Circuit tab: build the pressure-simulation
recipe as a graph, run it, inspect it, export the bundle. Screenshots are generated by a script under
`docs/scripts/` running offscreen, so they can be regenerated rather than going stale.

### R2. Concepts and extending

- `docs/concepts/the_graph.md` — how a graph maps onto the config, and why the config is still the
  source of truth.
- `docs/extending/add_gui_node.md` — when a plugin needs more than `get_param_spec()` gives it. The
  worked example is a node with a custom preview, executed in CI like the other extending guides.

### R3. The Phase 3 test suite

A `tests/gui/test_circuit_roundtrip.py` that round-trips every shipped example and preset, and a
smoke test that builds, runs and exports one graph end to end. Record the GUI suite's peak memory in
the report; if it has grown by more than half since Phase 2, open a finding.

### Wave R exit

`mkdocs build --strict` clean with no stub pages left for Phase 3, both suites green under the
watchdog, and the walkthrough's screenshots regenerable by one command.

---

## 8. Phase 3 exit criteria

- Waves O to R complete with their exit checks.
- A graph built in the GUI, exported to YAML, run by the CLI, and re-imported gives the same graph
  and the same results.
- A plugin package that this repository does not know about appears in the palette, is configurable
  through the inspector, and runs.
- All suites, `black`, the CI flake8 subset and `mkdocs build --strict` pass.
- Ledger: F-019 closed.
