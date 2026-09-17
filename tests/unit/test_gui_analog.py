"""The GUI shows analog populations (Phase 2, Wave N, N4).

Before this, ``SpikingNeuronTab._simulate_population`` unconditionally read
``backend["spikes"]``, which raises ``KeyError`` for an analog readout (N2:
the backend carries "state" instead, when the neuron model has no spike
condition -- N1). ``_update_raster_plot`` unconditionally scattered
``result.spikes``. Now a ``SimulationResult`` with ``is_analog=True`` plots
its state trace in the raster panel instead, with no raster points, and the
axis is labelled with the state variable's name.
"""

import sys

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`


_APP = None  # keep the QApplication alive for the module (F-016 pattern)


def _qt_app():
    global _APP
    try:
        from PyQt5 import QtWidgets

        _APP = QtWidgets.QApplication.instance()
        if _APP is None:
            _APP = QtWidgets.QApplication(sys.argv[:1])
        return _APP
    except ImportError:
        pytest.skip("PyQt5 not available")


def _tab_with_grid(rows=8, cols=8, spacing=0.15):
    _qt_app()
    from PyQt5 import QtGui
    from sensoryforge.gui.tabs.mechanoreceptor_tab import GridEntry, MechanoreceptorTab

    tab = MechanoreceptorTab()
    tab._add_grid_entry(
        GridEntry(
            name="G",
            rows=rows,
            cols=cols,
            spacing=spacing,
            color=QtGui.QColor(90, 90, 200),
        )
    )
    return tab


def _spiking_tab(mech_tab):
    from sensoryforge.gui.tabs.spiking_tab import SpikingNeuronTab
    from sensoryforge.gui.tabs.stimulus_tab import StimulusDesignerTab

    stim = StimulusDesignerTab(mechanoreceptor_tab=mech_tab)
    return SpikingNeuronTab(mechanoreceptor_tab=mech_tab, stimulus_tab=stim)


def _build_analog_population():
    from sensoryforge.gui.tabs.spiking_tab import PopulationConfig as SpikingConfig
    from sensoryforge.neurons.model_dsl import NeuronModel

    tab = _tab_with_grid(rows=8, cols=8)
    tab._on_add_population()
    tab._generate_populations()
    pop = tab.populations[-1]

    spiking = _spiking_tab(tab)
    # A thresholdless (analog) DSL model, bypassing the editor's own
    # non-empty-threshold check in _compile_dsl_model -- out of scope here
    # (only the result plotting path is touched, N4).
    spiking._compiled_dsl_model = NeuronModel(
        equations="dv/dt = (-(v - v_rest) + I) / tau_m",
        parameters={"v_rest": -65.0, "tau_m": 10.0},
        state_vars={"v": -65.0},
    )
    config = SpikingConfig(
        name=pop.name, neuron_type="SA", model="DSL (Custom)", filter_method="none"
    )
    config.input_gain = 1.0
    frames = torch.full((40, 8, 8), 5.0)
    return spiking, pop, config, frames


class TestGUIAnalogPopulation:
    def test_analog_population_simulates_and_returns_state(self):
        spiking, pop, config, frames = _build_analog_population()
        result = spiking._simulate_population(
            pop, config, frames, 1.0, torch.device("cpu")
        )
        assert result.is_analog is True
        assert result.state_var_name == "v"
        assert result.v_trace.shape[0] == 40
        assert result.v_trace.shape[1] == pop.num_neurons
        # spikes is an all-zero stand-in for analog results.
        assert np.all(result.spikes == 0)

    def test_analog_result_plots_trace_with_no_raster_points(self):
        spiking, pop, config, frames = _build_analog_population()
        result = spiking._simulate_population(
            pop, config, frames, 1.0, torch.device("cpu")
        )
        spiking.sim_results[pop.name] = result
        spiking.cmb_population.clear()
        spiking.cmb_population.addItem(pop.name)
        spiking.cmb_population.setCurrentText(pop.name)

        spiking._update_raster_plot()

        # The trace panel has data (one PlotDataItem per neuron)...
        data_items = [
            item
            for item in spiking.raster_plot.listDataItems()
            if hasattr(item, "getData")
        ]
        assert len(data_items) == pop.num_neurons
        for item in data_items:
            xdata, ydata = item.getData()
            assert xdata is not None and len(xdata) == 40

        # ...and no scatter (raster) points.
        scatter_items = [
            item
            for item in spiking.raster_plot.listDataItems()
            if item.__class__.__name__ == "ScatterPlotItem"
        ]
        assert scatter_items == []
        axis_label = spiking.raster_plot.getPlotItem().getAxis("left").labelText
        assert axis_label == "v"
