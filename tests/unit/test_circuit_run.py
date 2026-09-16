"""Qt test: running a Circuit tab graph produces a bundle (Wave O, O4).

Marked gui; run alone (this repo's Qt test suite is order-dependent, F-016 --
see the appendix commands in docs/development/handover/phase1_tasks.md).
"""

import sys
import tempfile

import pytest

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

_APP = None


def _ensure_app():
    global _APP
    from PyQt5 import QtWidgets

    _APP = QtWidgets.QApplication.instance()
    if _APP is None:
        _APP = QtWidgets.QApplication(sys.argv[:1])


def _small_graph(circuit_tab, output_dir):
    """Build a minimal SensorArray -> RFBank -> Filter -> Readout chain,
    plus a Stimulus node and a Record node pointing at ``output_dir``."""
    from sensoryforge.config.schema import GridConfig, StimulusConfig

    grid_node = circuit_tab.add_node("SensorArray", "grid1")
    grid_node.from_config(
        GridConfig(name="grid1", arrangement="grid", rows=8, cols=8, spacing=0.2)
    )

    stim_node = circuit_tab.add_node("Stimulus", "stim1")
    stim_node.from_config(
        StimulusConfig(name="stim1", type="gaussian", amplitude=30.0, sigma=0.5)
    )

    rf_node = circuit_tab.add_node("RFBank", "rf1")
    filt_node = circuit_tab.add_node("Filter", "filter1")
    filt_node.from_config({"filter_method": "sa", "filter_params": {}})
    readout_node = circuit_tab.add_node("Readout", "SA Population")
    readout_node.from_config(
        {
            "name": "SA Population",
            "neuron_type": "SA",
            "neuron_model": "izhikevich",
            "neurons_per_row": 4,
        }
    )
    record_node = circuit_tab.add_node("Record", "Record")
    record_node.from_config({"output_dir": str(output_dir)})

    circuit_tab.connect_nodes(grid_node, "value", rf_node, "Channel")
    circuit_tab.connect_nodes(rf_node, "Drive", filt_node, "Drive")
    circuit_tab.connect_nodes(filt_node, "Filtered", readout_node, "Filtered")
    circuit_tab.connect_nodes(readout_node, "Spikes", record_node, "In")


def test_run_graph_writes_a_bundle_load_bundle_can_read():
    _ensure_app()
    from sensoryforge.gui.tabs.circuit_tab import CircuitTab
    from sensoryforge.io.bundle import load_bundle

    with tempfile.TemporaryDirectory() as tmp_dir:
        tab = CircuitTab()
        _small_graph(tab, tmp_dir)

        results = tab.run_graph(duration_ms=20.0)

        assert "SA Population" in results
        bundle = load_bundle(tmp_dir)
        assert bundle is not None


def test_run_graph_emits_simulation_finished_with_the_spiking_tab_shape():
    _ensure_app()
    from sensoryforge.gui.tabs.circuit_tab import CircuitTab
    from sensoryforge.gui.tabs.spiking_tab import SimulationResult

    with tempfile.TemporaryDirectory() as tmp_dir:
        tab = CircuitTab()
        _small_graph(tab, tmp_dir)

        received = []
        tab.simulation_finished.connect(lambda *args: received.append(args))
        tab.run_graph(duration_ms=20.0)

        assert len(received) == 1
        sim_results, frames, time_ms, dt_ms, xlim, ylim = received[0]
        assert isinstance(sim_results["SA Population"], SimulationResult)
        assert frames is not None
        assert len(time_ms) > 0
        assert dt_ms > 0
        assert len(xlim) == 2 and len(ylim) == 2
