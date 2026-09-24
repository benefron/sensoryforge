"""Unit tests for sensoryforge.config.defaults (tasks A4/A8, F-026/F-030/F-031).

These tests are Qt-free: they cover the resolver functions directly, that
and that SimulationEngine builds exactly what the resolvers say. (The GUI v2
forms show these same resolved values; that is tested in tests/gui_v2.)
"""

import pytest

from sensoryforge.config.defaults import (
    resolve_filter_params,
    resolve_neuron_params,
)
from sensoryforge.neurons.adex import ADEX_PRESETS
from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
    StimulusConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine


class TestResolveFilterParams:
    def test_sa_empty_overrides(self):
        params = resolve_filter_params("sa", {})
        assert params == {"tau_r": 5.0, "tau_d": 30.0, "k1": 0.05, "k2": 8.0}

    def test_ra_empty_overrides_includes_k3(self):
        """D-Q1 decided: k3 = 2.0, resolver-owned like every other filter param."""
        params = resolve_filter_params("ra", {})
        assert params == {"tau_RA": 8.0, "k3": 2.0}

    def test_overrides_win(self):
        params = resolve_filter_params("sa", {"tau_r": 99.0})
        assert params["tau_r"] == 99.0
        assert params["tau_d"] == 30.0

    def test_case_insensitive(self):
        assert resolve_filter_params("SA", {}) == resolve_filter_params("sa", {})

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown filter method"):
            resolve_filter_params("fa", {})


class TestResolveNeuronParams:
    def test_ra_izhikevich_resolves_fast_spiking(self):
        params = resolve_neuron_params("Izhikevich", "RA", {})
        assert params["a"] == 0.1
        assert params["b"] == 0.2
        assert params["c"] == -65.0
        assert params["d"] == 2.0
        assert "preset" not in params

    def test_sa_izhikevich_resolves_regular_spiking(self):
        params = resolve_neuron_params("Izhikevich", "SA", {})
        assert params["a"] == 0.02
        assert params["d"] == 8.0

    def test_sa2_izhikevich_resolves_regular_spiking(self):
        params = resolve_neuron_params("Izhikevich", "SA2", {})
        assert params["a"] == 0.02

    def test_explicit_d_override_keeps_preset_as_base(self):
        """Regression for F-031: overriding one of a/b/c/d must NOT drop the
        neuron-type preset for the rest -- the preset is always the base,
        with overrides applied on top (matching
        IzhikevichNeuronTorch(preset=..., d=...) semantics). Fails on d194cd4,
        where an RA {"d": 99.0} override silently reverted a/b/c to RS.
        """
        params = resolve_neuron_params("Izhikevich", "RA", {"d": 99.0})
        assert params["a"] == 0.1  # RA/FS base, not suppressed
        assert params["b"] == 0.2
        assert params["c"] == -65.0
        assert params["d"] == 99.0  # override applied on top

        from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch

        neuron = IzhikevichNeuronTorch(**params)
        assert neuron.a == 0.1
        assert neuron.d == 99.0

    def test_explicit_preset_override_wins(self):
        params = resolve_neuron_params("Izhikevich", "SA", {"preset": "FS"})
        assert params["a"] == 0.1
        assert params["d"] == 2.0

    def test_unknown_preset_override_raises(self):
        with pytest.raises(ValueError, match="Unknown Izhikevich preset"):
            resolve_neuron_params("Izhikevich", "RA", {"preset": "nope"})

    def test_non_preset_model_passes_overrides_through(self):
        """MQIF (unlike Izhikevich and AdEx) has no preset table, so its
        defaults live only in the class constructor signature and the
        resolver must pass overrides straight through unchanged."""
        overrides = {"tau_m": 15.0}
        assert resolve_neuron_params("MQIF", "RA", overrides) == overrides

    def test_case_insensitive_model_name(self):
        assert resolve_neuron_params("izhikevich", "RA", {}) == resolve_neuron_params(
            "Izhikevich", "RA", {}
        )


class TestResolveNeuronParamsAdEx:
    def test_ra_adex_resolves_phasic(self):
        params = resolve_neuron_params("AdEx", "RA", None)
        assert params == ADEX_PRESETS["RA1_phasic"]
        assert "preset" not in params

    def test_sa_adex_resolves_tonic(self):
        params = resolve_neuron_params("AdEx", "SA", None)
        assert params == ADEX_PRESETS["SA1_tonic"]

    def test_sa2_adex_resolves_tonic(self):
        params = resolve_neuron_params("AdEx", "SA2", None)
        assert params == ADEX_PRESETS["SA1_tonic"]

    def test_single_override_keeps_preset_as_base(self):
        """F-031 for AdEx: overriding tau_w must not drop the rest of the
        SA1_tonic preset."""
        params = resolve_neuron_params("AdEx", "SA", {"tau_w": 300.0})
        expected = ADEX_PRESETS["SA1_tonic"]
        assert params["tau_w"] == 300.0
        for name, value in expected.items():
            if name == "tau_w":
                continue
            assert params[name] == value

    def test_explicit_preset_override_wins(self):
        params = resolve_neuron_params("AdEx", "SA", {"preset": "RA1_phasic"})
        assert params == ADEX_PRESETS["RA1_phasic"]

    def test_unknown_preset_override_raises(self):
        with pytest.raises(ValueError, match="Unknown AdEx preset"):
            resolve_neuron_params("AdEx", "RA", {"preset": "nope"})

    def test_case_insensitive_model_name(self):
        assert resolve_neuron_params("adex", "RA", {}) == resolve_neuron_params(
            "AdEx", "RA", {}
        )


def _engine_config(neuron_type: str) -> SensoryForgeConfig:
    return SensoryForgeConfig(
        grids=[
            GridConfig(name="grid", arrangement="grid", rows=10, cols=10, spacing=0.15)
        ],
        populations=[
            PopulationConfig(
                name="Pop",
                target_grid="grid",
                neuron_type=neuron_type,
                neuron_model="izhikevich",
                filter_method=neuron_type[:2].lower(),
                innervation_method="gaussian",
                neurons_per_row=3,
            )
        ],
        stimulus=StimulusConfig(type="gaussian", amplitude=10.0, sigma=1.0),
        simulation=SimulationConfig(device="cpu", dt_ms=0.1),
    )


class TestSimulationEngineMatchesResolvers:
    """A4 acceptance: engine construction must equal the resolver output."""

    def test_ra_population_builds_resolved_params(self):
        engine = SimulationEngine(_engine_config("RA"))
        neuron = engine.populations[0]["neuron"]
        filt = engine.populations[0]["filter"]
        expected_neuron = resolve_neuron_params("izhikevich", "RA", {})
        expected_filter = resolve_filter_params("ra", {})
        assert neuron.a == expected_neuron["a"]
        assert neuron.d == expected_neuron["d"]
        assert filt.tau_RA == expected_filter["tau_RA"]

    def test_sa_population_builds_resolved_params(self):
        engine = SimulationEngine(_engine_config("SA"))
        neuron = engine.populations[0]["neuron"]
        filt = engine.populations[0]["filter"]
        expected_neuron = resolve_neuron_params("izhikevich", "SA", {})
        expected_filter = resolve_filter_params("sa", {})
        assert neuron.a == expected_neuron["a"]
        assert neuron.d == expected_neuron["d"]
        assert filt.tau_r == expected_filter["tau_r"]
        assert filt.tau_d == expected_filter["tau_d"]
