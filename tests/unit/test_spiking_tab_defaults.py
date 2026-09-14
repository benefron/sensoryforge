"""Qt test: SpikingNeuronTab must build the resolver-owned defaults (F-026, task A4).

Marked gui; run alone (this repo's Qt test suite is order-dependent, F-016 --
see the appendix commands in docs/development/handover/phase1_tasks.md).
"""

import sys

import pytest

pytestmark = pytest.mark.gui  # F-016: Qt tests, run with `pytest -m gui`

_APP = None


def _ensure_app():
    global _APP
    try:
        from PyQt5 import QtWidgets

        _APP = QtWidgets.QApplication.instance()
        if _APP is None:
            _APP = QtWidgets.QApplication(sys.argv[:1])
    except ImportError:
        pytest.skip("PyQt5 not available")


@pytest.fixture(scope="module")
def mech_tab():
    _ensure_app()
    try:
        from sensoryforge.gui.tabs.mechanoreceptor_tab import MechanoreceptorTab
    except ImportError:
        pytest.skip("PyQt5 not available")
    return MechanoreceptorTab()


@pytest.fixture(scope="module")
def spiking_tab(mech_tab):
    try:
        from sensoryforge.gui.tabs.spiking_tab import SpikingNeuronTab
        from sensoryforge.gui.tabs.stimulus_tab import StimulusDesignerTab
    except ImportError:
        pytest.skip("GUI tabs not available")
    stim = StimulusDesignerTab(mechanoreceptor_tab=mech_tab)
    return SpikingNeuronTab(mechanoreceptor_tab=mech_tab, stimulus_tab=stim)


def test_ra_izhikevich_population_resolves_fast_spiking(spiking_tab):
    from sensoryforge.gui.tabs.spiking_tab import PopulationConfig
    import torch

    config = PopulationConfig(
        name="RA Pop", neuron_type="RA", model="Izhikevich", filter_method="ra"
    )
    neuron = spiking_tab._create_neuron_model(
        config, dt_ms=0.1, device=torch.device("cpu")
    )
    assert neuron.a == 0.1
    assert neuron.d == 2.0


def test_sa_izhikevich_population_resolves_regular_spiking(spiking_tab):
    from sensoryforge.gui.tabs.spiking_tab import PopulationConfig
    import torch

    config = PopulationConfig(
        name="SA Pop", neuron_type="SA", model="Izhikevich", filter_method="sa"
    )
    neuron = spiking_tab._create_neuron_model(
        config, dt_ms=0.1, device=torch.device("cpu")
    )
    assert neuron.a == 0.02
    assert neuron.d == 8.0


def test_ra_filter_resolves_tau_ra_8ms(spiking_tab):
    from sensoryforge.gui.tabs.spiking_tab import PopulationConfig

    config = PopulationConfig(
        name="RA Pop", neuron_type="RA", model="Izhikevich", filter_method="ra"
    )
    params = spiking_tab._gather_filter_parameters(config)
    assert params["tau_RA"] == 8.0


def test_create_neuron_model_uses_whatever_dt_it_is_given(spiking_tab):
    """_create_neuron_model itself is dt-agnostic (it just builds a neuron
    at the dt its caller passes); task E4 (F-008) is that the *caller*
    (_simulate_population) must pass DEFAULT_INTEGRATE_DT_MS, not the
    record step dt_ms. See test_simulate_population_constructs_neuron_at_
    integrate_dt below for that call-site check, and
    tests/integration/test_engine_parity.py::TestNeuronSubStepping::
    test_engine_constructs_neuron_at_integrate_dt_not_record_dt for the
    engine-side equivalent (both must agree on the same
    DEFAULT_INTEGRATE_DT_MS/SimulationConfig.integrate_dt_ms default,
    0.05 ms).
    """
    import torch

    from sensoryforge.gui.tabs.spiking_tab import PopulationConfig

    config = PopulationConfig(name="SA Pop", neuron_type="SA", model="Izhikevich")
    neuron = spiking_tab._create_neuron_model(
        config, dt_ms=0.05, device=torch.device("cpu")
    )
    assert neuron.dt == 0.05


def test_simulate_population_constructs_neuron_at_integrate_dt_not_record_dt():
    """_simulate_population must call _create_neuron_model with
    DEFAULT_INTEGRATE_DT_MS, not the record step dt_ms (F-008).

    Instantiating a full drivable population is out of scope for a fast
    unit test; instead inspect the source directly for the call-site
    contract, matching how this repo already checks for structural
    invariants it can't cheaply exercise end-to-end.
    """
    import inspect

    from sensoryforge.gui.tabs.spiking_tab import SpikingNeuronTab

    source = inspect.getsource(SpikingNeuronTab._simulate_population)
    assert "DEFAULT_INTEGRATE_DT_MS" in source, (
        "_simulate_population must construct the neuron with "
        "DEFAULT_INTEGRATE_DT_MS, not the record step dt_ms"
    )
