"""The GUI uses receptive-field banks (Phase 2, I7).

- Mechanoreceptor tab populations hold a ``ReceptiveFieldBank``
- an imported CSV population simulates through the Spiking tab and spikes
- a ``template`` population shows its derived neuron count
- the GUI's bank equals the engine's for the exported config (parity)
"""

import sys
from pathlib import Path

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


def _tab_with_grid(rows=16, cols=16, spacing=0.15):
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


def test_population_holds_a_bank():
    from sensoryforge.core.rf_bank import ReceptiveFieldBank

    tab = _tab_with_grid(rows=8, cols=8)
    tab._on_add_population()
    tab._generate_populations()
    pop = tab.populations[-1]
    assert isinstance(pop.bank, ReceptiveFieldBank)
    assert pop.grid_shape == (8, 8)
    assert pop.innervation_weights.shape == (pop.num_neurons, 8, 8)
    assert pop.bank.provenance["builder"] == "gaussian"
    assert not hasattr(pop, "module") and not hasattr(pop, "flat_module")


def test_template_population_shows_derived_neuron_count():
    tab = _tab_with_grid(rows=16, cols=16, spacing=0.15)
    tab._on_add_population()
    pop = tab.populations[-1]
    tab._selected_population = pop
    # The field's own hidden flag is what choosing a method controls. Checking
    # isVisibleTo(tab) also depended on whether the collapsible "Population
    # Settings" section was expanded -- collapsed by default, expanded only on
    # a machine where someone had saved it that way -- so this passed locally
    # and failed on a fresh CI runner.
    tab.cmb_innervation_method.setCurrentText("gaussian")
    assert tab.dbl_resolvable_distance.isHidden()
    tab.cmb_innervation_method.setCurrentText("template")
    tab.dbl_resolvable_distance.setValue(0.40)
    assert not tab.dbl_resolvable_distance.isHidden()
    assert not tab.spin_neurons_per_row.isEnabled()
    tab._generate_populations()
    tab._load_population_into_form(pop)
    assert pop.innervation_method == "template"
    assert pop.num_neurons == 36
    assert pop.bank.provenance["builder"] == "template"
    text = tab.lbl_population_info.text()
    assert "36" in text and "derived" in text

    cfg = tab.get_config()["populations"][-1]
    assert cfg["innervation_method"] == "template"
    assert cfg["resolvable_distance_mm"] == pytest.approx(0.40)

    # GUI-engine parity: the exported config builds the same bank
    from sensoryforge.config.schema import (
        GridConfig,
        PopulationConfig,
        SensoryForgeConfig,
    )
    from sensoryforge.core.simulation_engine import SimulationEngine

    grid_cfg = tab.get_config()["grids"][0]
    engine = SimulationEngine(
        SensoryForgeConfig(
            grids=[GridConfig.from_dict(grid_cfg)],
            populations=[PopulationConfig.from_dict(cfg)],
        )
    )
    assert torch.equal(engine.populations[0]["bank"].weights, pop.bank.weights)


def test_gaussian_population_matches_engine_for_exported_config():
    from sensoryforge.config.schema import (
        GridConfig,
        PopulationConfig,
        SensoryForgeConfig,
    )
    from sensoryforge.core.simulation_engine import SimulationEngine

    tab = _tab_with_grid(rows=10, cols=10)
    tab._on_add_population()
    tab._generate_populations()
    pop = tab.populations[-1]
    config = tab.get_config()
    engine = SimulationEngine(
        SensoryForgeConfig(
            grids=[GridConfig.from_dict(config["grids"][0])],
            populations=[PopulationConfig.from_dict(config["populations"][-1])],
        )
    )
    assert torch.equal(engine.populations[0]["bank"].weights, pop.bank.weights)


def _spiking_tab(mech_tab):
    from sensoryforge.gui.tabs.spiking_tab import SpikingNeuronTab
    from sensoryforge.gui.tabs.stimulus_tab import StimulusDesignerTab

    stim = StimulusDesignerTab(mechanoreceptor_tab=mech_tab)
    return SpikingNeuronTab(mechanoreceptor_tab=mech_tab, stimulus_tab=stim)


def test_imported_csv_population_simulates_and_spikes(tmp_path):
    from sensoryforge.gui.tabs.spiking_tab import PopulationConfig as SpikingConfig

    tab = _tab_with_grid(rows=8, cols=8)
    tab._on_add_population()
    tab._generate_populations()
    source = tab.populations[-1]
    tab.export_population_csv(source, tmp_path / "pop_csv")
    tab._on_add_population()
    pop = tab.populations[-1]
    tab.import_population_csv(pop, tmp_path / "pop_csv")
    assert pop.bank.provenance["builder"] == "imported"

    spiking = _spiking_tab(tab)
    config = SpikingConfig(
        name=pop.name, neuron_type="SA", model="Izhikevich", filter_method="sa"
    )
    config.input_gain = 50.0
    # 300 ms of a strong step over the whole grid at 1 ms
    frames = torch.full((300, 8, 8), 30.0)
    frames[:20] = 0.0
    result = spiking._simulate_population(pop, config, frames, 1.0, torch.device("cpu"))
    assert result.spikes.shape == (300, pop.num_neurons)
    assert result.raw_drive.shape == (300, pop.num_neurons)
    assert result.spikes.sum() > 0
    # raw drive is the bank applied to the flattened frames
    expected = frames.reshape(300, -1) @ pop.bank.weights.T
    assert np.allclose(result.raw_drive, expected.numpy(), atol=1e-5)
