"""Unit tests for sensoryforge.config.defaults (tasks A4/A8, F-026/F-030/F-031).

These tests are Qt-free: they cover the resolver functions directly, that
SimulationEngine builds exactly what the resolvers say, and that
gui/default_params.json does not silently drift from the resolver-owned
values (the GUI test covering SpikingNeuronTab itself lives in
tests/unit/test_spiking_tab_defaults.py, marked gui, run separately).
"""

import json
from pathlib import Path

import pytest

from sensoryforge.config.defaults import (
    FILTER_DEFAULTS,
    NEURON_PRESET_BY_TYPE,
    resolve_filter_params,
    resolve_neuron_params,
)
from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
    StimulusConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine

DEFAULT_PARAMS_PATH = (
    Path(__file__).resolve().parents[2] / "sensoryforge" / "gui" / "default_params.json"
)


class TestResolveFilterParams:
    def test_sa_empty_overrides(self):
        params = resolve_filter_params("sa", {})
        assert params == {"tau_r": 5.0, "tau_d": 30.0, "k1": 0.05, "k2": 3.0}

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

    def test_non_izhikevich_model_passes_overrides_through(self):
        overrides = {"tau_m": 15.0}
        assert resolve_neuron_params("AdEx", "RA", overrides) == overrides

    def test_case_insensitive_model_name(self):
        assert resolve_neuron_params("izhikevich", "RA", {}) == resolve_neuron_params(
            "Izhikevich", "RA", {}
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
        simulation=SimulationConfig(device="cpu", dt=0.1),
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


class TestDefaultParamsJsonDoesNotDrift:
    """Regression: gui/default_params.json must not silently diverge from the
    resolver for the keys the resolver owns (including k3, since D-Q1).
    """

    @pytest.fixture(autouse=True)
    def _load_json(self):
        with open(DEFAULT_PARAMS_PATH, "r", encoding="utf-8") as f:
            self.json_defaults = json.load(f)

    def test_json_izhikevich_matches_rs_resolver_default(self):
        """The JSON's flat 'Izhikevich' entry (used by the standalone
        neuron_explorer, which has no per-population neuron_type) must equal
        the RS preset the resolver would produce.
        """
        json_model = self.json_defaults["models"]["Izhikevich"]
        resolved_rs = resolve_neuron_params("Izhikevich", "SA", {})
        for key in ("a", "b", "c", "d", "threshold", "noise_std"):
            assert json_model[key] == resolved_rs[key], (
                f"default_params.json models.Izhikevich.{key}={json_model[key]!r} "
                f"has drifted from the resolver's RS value {resolved_rs[key]!r}"
            )

    def test_json_sa_filter_matches_resolver(self):
        json_sa = self.json_defaults["filters"]["SA"]
        resolved = resolve_filter_params("sa", {})
        for key, value in resolved.items():
            assert json_sa[key] == value, (
                f"default_params.json filters.SA.{key}={json_sa[key]!r} has "
                f"drifted from the resolver value {value!r}"
            )

    def test_json_ra_filter_matches_resolver(self):
        json_ra = self.json_defaults["filters"]["RA"]
        resolved = resolve_filter_params("ra", {})
        for key, value in resolved.items():
            assert json_ra[key] == value, (
                f"default_params.json filters.RA.{key}={json_ra[key]!r} has "
                f"drifted from the resolver value {value!r}"
            )
