"""Qt test: Circuit tab graph<->config round trip (Wave O, O3).

Marked gui; run alone (this repo's Qt test suite is order-dependent, F-016 --
see the appendix commands in docs/development/handover/phase1_tasks.md).

Per the Wave O spec, this is the heart of the wave, not paperwork: a graph
editor can lose a field on the way to config and still render a
perfectly good-looking diagram and still run. So these tests compare the
*whole* SensoryForgeConfig object (dataclass ``==``, which recurses into
every nested list/dataclass/dict -- not a handful of hand-picked
attributes), and separately compare the emitted YAML *text* the way Wave
M's byte-for-byte test does, for every canonical config this repository
ships: every ``examples/*.yml`` that is actually a canonical
(``SensoryForgeConfig``-shaped) config, plus all three presets under
``sensoryforge/presets/``.

``examples/example_config.yml`` and ``examples/batch_config.yml`` are
pre-canonical *legacy* pipeline-format configs (a top-level ``pipeline:``
key, no ``grids:``/``populations:``) -- there is no graph for them because
the Circuit tab's node classes map onto the canonical dataclasses, not the
legacy dict shape; they are skipped, not silently treated as passing.
``examples/canonical_batch_config.yml`` wraps a canonical config under
``base_config:`` inside a batch sweep spec -- that inner config is what is
round-tripped.
"""

import glob
import sys

import pytest
import yaml

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

_APP = None


def _ensure_app():
    global _APP
    from PyQt5 import QtWidgets

    _APP = QtWidgets.QApplication.instance()
    if _APP is None:
        _APP = QtWidgets.QApplication(sys.argv[:1])


def _canonical_configs():
    """Yield (label, SensoryForgeConfig) for every canonical config this repo ships."""
    from sensoryforge.config.schema import SensoryForgeConfig

    for path in sorted(glob.glob("examples/*.yml")):
        with open(path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict) or "grids" not in raw or "populations" not in raw:
            if "base_config" in (raw or {}):
                yield f"{path}::base_config", SensoryForgeConfig.from_dict(
                    raw["base_config"]
                )
            # else: legacy pipeline-format config, no canonical shape -- skip.
            continue
        yield path, SensoryForgeConfig.from_yaml(path)

    for path in sorted(glob.glob("sensoryforge/presets/*.yml")):
        yield path, SensoryForgeConfig.from_yaml(path)


def test_examples_and_presets_round_trip_through_the_graph_unchanged():
    """config -> graph -> config gives back an equal SensoryForgeConfig and
    identical YAML text, for every canonical example and preset."""
    _ensure_app()
    from pyqtgraph.flowchart import Flowchart

    from sensoryforge.gui.circuit.nodes import build_node_library
    from sensoryforge.gui.circuit.serialise import config_to_graph, graph_to_config

    covered = []
    for label, config in _canonical_configs():
        fc = Flowchart(terminals={}, name="Circuit")
        fc.library = build_node_library()
        config_to_graph(config, fc)
        round_tripped = graph_to_config(fc)

        assert round_tripped == config, f"{label}: config changed across the graph"
        assert (
            round_tripped.to_yaml() == config.to_yaml()
        ), f"{label}: YAML text changed across the graph"
        covered.append(label)

    # At least the five canonical files (canonical_config, canonical_template_config,
    # canonical_batch_config's base_config, and the three presets) must have been
    # exercised -- a glob regression that silently matched nothing must fail loudly.
    assert len(covered) >= 5, f"too few canonical configs covered: {covered}"


def test_graph_to_config_to_graph_gives_an_identical_node_and_edge_set():
    """graph -> config -> graph is not merely rebuilt into a different picture:
    the node name/type set and the connection set survive exactly."""
    _ensure_app()
    from pyqtgraph.flowchart import Flowchart

    from sensoryforge.gui.circuit.nodes import build_node_library
    from sensoryforge.gui.circuit.serialise import config_to_graph, graph_to_config

    _, config = next(iter(_canonical_configs()))

    fc1 = Flowchart(terminals={}, name="Circuit1")
    fc1.library = build_node_library()
    config_to_graph(config, fc1)

    round_tripped = graph_to_config(fc1)

    fc2 = Flowchart(terminals={}, name="Circuit2")
    fc2.library = build_node_library()
    config_to_graph(round_tripped, fc2)

    names1 = {(name, type(node).__name__) for name, node in fc1.nodes().items()}
    names2 = {(name, type(node).__name__) for name, node in fc2.nodes().items()}
    assert names1 == names2

    def _edge_set(fc):
        return {
            (out.node().name(), out.name(), in_.node().name(), in_.name())
            for out, in_ in fc.listConnections()
        }

    assert _edge_set(fc1) == _edge_set(fc2)


def test_round_trip_fails_to_detect_a_dropped_field_would_have_been_caught():
    """Sanity check on the test's own strength: dropping a field from one
    population before comparison must fail the whole-object assertion (proves
    the comparison is not shape/type-only)."""
    _ensure_app()
    from pyqtgraph.flowchart import Flowchart

    from sensoryforge.gui.circuit.nodes import build_node_library
    from sensoryforge.gui.circuit.serialise import config_to_graph, graph_to_config

    _, config = next(
        (label, cfg) for label, cfg in _canonical_configs() if cfg.populations
    )

    fc = Flowchart(terminals={}, name="Circuit")
    fc.library = build_node_library()
    config_to_graph(config, fc)
    round_tripped = graph_to_config(fc)

    import dataclasses

    mutated = dataclasses.replace(
        round_tripped.populations[0],
        input_gain=round_tripped.populations[0].input_gain + 1.0,
    )
    tampered = dataclasses.replace(
        round_tripped, populations=[mutated] + round_tripped.populations[1:]
    )
    assert tampered != config
