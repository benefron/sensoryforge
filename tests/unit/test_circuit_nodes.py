"""Qt test: Circuit tab node classes instantiate headless and round-trip config (Wave O, O1).

Marked gui; run alone (this repo's Qt test suite is order-dependent, F-016 --
see the appendix commands in docs/development/handover/phase1_tasks.md).
"""

import sys

import pytest

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

_APP = None


def _ensure_app():
    global _APP
    from PyQt5 import QtWidgets

    _APP = QtWidgets.QApplication.instance()
    if _APP is None:
        _APP = QtWidgets.QApplication(sys.argv[:1])


def test_every_node_instantiates_headless_and_declares_terminals():
    _ensure_app()
    from sensoryforge.gui.circuit.nodes import NODE_CLASSES

    expected_terminals = {
        "SensorArray": (set(), {"value"}),
        "Stimulus": (set(), {"Out"}),
        "RFBank": ({"Channel"}, {"Drive"}),
        "Processing": ({"Channel"}, {"Out"}),
        "Combine": ({"Drives"}, {"Drive"}),
        "Filter": ({"Drive"}, {"Filtered"}),
        "Readout": ({"Filtered"}, {"Spikes"}),
        "Record": ({"In"}, set()),
    }
    assert set(NODE_CLASSES) == set(expected_terminals)
    for node_name, cls in NODE_CLASSES.items():
        node = cls(node_name.lower())
        expected_in, expected_out = expected_terminals[node_name]
        actual_in = set(node.inputs().keys())
        actual_out = set(node.outputs().keys())
        if node_name == "SensorArray":
            assert actual_out == expected_out
        else:
            assert actual_in == expected_in
            assert actual_out == expected_out


def test_sensor_array_node_round_trips_grid_config():
    _ensure_app()
    from sensoryforge.gui.circuit.nodes import SensorArrayNode
    from sensoryforge.config.schema import GridConfig

    node = SensorArrayNode("grid1")
    grid = GridConfig(
        name="Main Grid",
        arrangement="grid",
        rows=40,
        cols=40,
        spacing=0.2,
        channels=["pressure", "shear"],
        coords_file="foo.csv",
    )
    node.from_config(grid)
    assert set(node.outputs().keys()) == {"pressure", "shear"}
    got = node.to_config()
    assert got == grid


def test_stimulus_node_round_trips_stimulus_config():
    _ensure_app()
    from sensoryforge.gui.circuit.nodes import StimulusNode
    from sensoryforge.config.schema import StimulusConfig

    node = StimulusNode("stim1")
    stim = StimulusConfig(
        name="Probe", type="moving", amplitude=42.0, channel="pressure"
    )
    node.from_config(stim)
    got = node.to_config()
    assert got == stim


def test_rf_bank_node_round_trips_population_input():
    _ensure_app()
    from sensoryforge.gui.circuit.nodes import RFBankNode
    from sensoryforge.config.schema import PopulationInput, RFBuilderConfig

    node = RFBankNode("rf1")
    pop_input = PopulationInput(
        grid="Main Grid",
        channel="pressure",
        rf=RFBuilderConfig(method="template", params={"resolvable_distance_mm": 1.0}),
        gain=2.0,
        layers=["skin"],
    )
    node.from_config(pop_input)
    got = node.to_config()
    assert got == pop_input


def test_processing_node_round_trips_spec():
    _ensure_app()
    from sensoryforge.gui.circuit.nodes import ProcessingNode

    node = ProcessingNode("proc1")
    spec = {"method": "rectify", "params": {"threshold": 0.1}}
    node.from_config(spec)
    assert node.to_config() == spec


def test_combine_node_round_trips_mode():
    _ensure_app()
    from sensoryforge.gui.circuit.nodes import CombineNode

    node = CombineNode("combine1")
    node.from_config("concat")
    assert node.to_config() == "concat"


def test_filter_node_round_trips():
    _ensure_app()
    from sensoryforge.gui.circuit.nodes import FilterNode

    node = FilterNode("filter1")
    data = {"filter_method": "sa", "filter_params": {"tau_r": 5.0}}
    node.from_config(data)
    assert node.to_config() == data


def test_readout_node_round_trips_population_fields():
    _ensure_app()
    from sensoryforge.gui.circuit.nodes import ReadoutNode, READOUT_FIELDS
    from sensoryforge.config.schema import PopulationConfig

    node = ReadoutNode("readout1")
    pop = PopulationConfig(
        name="SA Population", neuron_type="SA", neuron_model="izhikevich", seed=7
    )
    data = {f: getattr(pop, f) for f in READOUT_FIELDS}
    node.from_config(data)
    assert node.to_config() == data


def test_record_node_round_trips():
    """RecordNode also carries simulation/metadata (O3, disclosed): no node in
    the Wave O table owns SensoryForgeConfig.simulation or .metadata, and a
    lossless graph<->config round trip needs *some* node to hold them."""
    _ensure_app()
    from sensoryforge.gui.circuit.nodes import RecordNode

    node = RecordNode("record1")
    data = {
        "output_dir": "/tmp/bundle",
        "simulation": {"device": "cpu", "dt_ms": 2.0},
        "metadata": {"version": "1.0"},
    }
    node.from_config(data)
    assert node.to_config() == data


def test_build_node_library_registers_every_node():
    _ensure_app()
    from sensoryforge.gui.circuit.nodes import build_node_library, NODE_CLASSES

    lib = build_node_library()
    for node_name in NODE_CLASSES:
        assert lib.getNodeType(node_name) is NODE_CLASSES[node_name]
