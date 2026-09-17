"""Qt test: the Batch tab sweeps a parameter from the live Circuit graph (Wave Q, Q1).

A sweep that runs and writes N bundles looks successful whether or not the
parameter was actually varied. These tests do not settle for "three runs
produced three bundles": they read the swept value back out of each
bundle's own ``config.json`` and assert it differs run to run, and assert
the runs' own spike results differ too, so a sweep that silently pinned the
parameter (the exact failure shape every serious defect in this project has
had -- it ran, the output was plausible, nothing errored) fails loudly
instead of passing.

Marked gui; run alone (this repo's Qt test suite is order-dependent, F-016 --
see the appendix commands in docs/development/handover/phase1_tasks.md).
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

_APP = None


def _ensure_app():
    global _APP
    from PyQt5 import QtWidgets

    _APP = QtWidgets.QApplication.instance()
    if _APP is None:
        _APP = QtWidgets.QApplication(sys.argv[:1])


def _small_graph(circuit_tab):
    """Same minimal SensorArray -> RFBank -> Filter -> Readout chain as
    tests/unit/test_circuit_run.py, plus a Stimulus node, with no Record
    node -- the sweep supplies its own per-run bundle directory."""
    from sensoryforge.config.schema import GridConfig, StimulusConfig

    grid_node = circuit_tab.add_node("SensorArray", "grid1")
    grid_node.from_config(
        GridConfig(name="grid1", arrangement="grid", rows=8, cols=8, spacing=0.2)
    )

    stim_node = circuit_tab.add_node("Stimulus", "stim1")
    stim_node.from_config(
        StimulusConfig(name="stim1", type="gaussian", amplitude=10.0, sigma=0.5)
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

    circuit_tab.connect_nodes(grid_node, "value", rf_node, "Channel")
    circuit_tab.connect_nodes(rf_node, "Drive", filt_node, "Drive")
    circuit_tab.connect_nodes(filt_node, "Filtered", readout_node, "Filtered")
    return stim_node


@pytest.fixture
def wired_tabs():
    _ensure_app()
    from sensoryforge.gui.tabs.batch_tab import BatchTab
    from sensoryforge.gui.tabs.circuit_tab import CircuitTab

    circuit_tab = CircuitTab()
    _small_graph(circuit_tab)
    batch_tab = BatchTab()
    batch_tab.set_circuit_tab(circuit_tab)
    return batch_tab, circuit_tab


def test_sweep_writes_three_bundles_with_distinct_swept_values_and_results(
    wired_tabs,
):
    batch_tab, circuit_tab = wired_tabs

    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_dirs = batch_tab.run_graph_sweep(
            "stim1", "amplitude", [10.0, 40.0, 90.0], tmp_dir
        )

        assert len(bundle_dirs) == 3

        recorded_amplitudes = []
        recorded_drives = []
        for bundle_dir in bundle_dirs:
            config_path = Path(bundle_dir) / "config.json"
            assert config_path.exists()
            written = json.loads(config_path.read_text())
            amplitude = written["config"]["stimulus"]["amplitude"]
            recorded_amplitudes.append(amplitude)

            from sensoryforge.io.bundle import load_bundle

            bundle = load_bundle(bundle_dir)
            pop_data = bundle.populations["SA Population"]
            drive_key = "drive" if "drive" in pop_data else "spikes"
            recorded_drives.append(pop_data[drive_key].clone())

        # The exact failure shape this test guards against: three bundles
        # that all silently recorded the same value.
        assert recorded_amplitudes == [10.0, 40.0, 90.0]
        assert len(set(recorded_amplitudes)) == 3

        # And the swept value must actually have reached the simulation --
        # not merely been recorded in the config while the run used the
        # same drive/response regardless.
        assert not (recorded_drives[0] == recorded_drives[1]).all()
        assert not (recorded_drives[1] == recorded_drives[2]).all()


def test_sweep_rejects_a_parameter_that_does_not_vary(wired_tabs):
    from sensoryforge.gui.circuit.sweep import SweepValidationError

    batch_tab, circuit_tab = wired_tabs

    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(SweepValidationError):
            batch_tab.run_graph_sweep("stim1", "amplitude", [10.0, 10.0, 10.0], tmp_dir)


def test_sweep_param_combo_lists_the_stimulus_get_param_spec(wired_tabs):
    batch_tab, circuit_tab = wired_tabs

    batch_tab._sweep_node_combo.setCurrentText("stim1")
    names = [
        batch_tab._sweep_param_combo.itemText(i)
        for i in range(batch_tab._sweep_param_combo.count())
    ]
    assert "amplitude" in names
