"""Unit tests for case-insensitive component registry lookup (F-046).

Covers the ``ComponentRegistry`` contract directly and, end-to-end, a
third-party plugin neuron/filter registered from outside
``register_components.py`` that is referenced from a canonical config with
either casing.
"""

import torch
import pytest

from sensoryforge.registry import ComponentRegistry, NEURON_REGISTRY, FILTER_REGISTRY
from sensoryforge.register_components import register_all
from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
    StimulusConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine


class _DemoNeuron(torch.nn.Module):
    """Minimal plugin neuron: passes the drive through as voltage, never spikes."""

    def __init__(self, dt: float = 1.0, noise_std: float = 0.0):
        super().__init__()
        self.dt = dt
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor):
        # Matches the real neuron contract: output has steps+1 samples
        # (an initial state sample prepended), see IzhikevichNeuronTorch.
        batch, steps, features = x.shape
        pad = torch.zeros(batch, 1, features, dtype=x.dtype, device=x.device)
        v_trace = torch.cat([pad, x], dim=1)
        spikes = torch.zeros_like(v_trace)
        return v_trace, spikes

    def reset_state(self):
        pass


class _DemoGain(torch.nn.Module):
    """Minimal plugin filter: scales the drive by a fixed gain."""

    def __init__(self, dt: float = 1.0, gain: float = 1.0):
        super().__init__()
        self.dt = dt
        self.gain = gain

    def forward(self, x: torch.Tensor):
        return x * self.gain

    def reset_state(self):
        pass


def _demo_engine_config(neuron_model: str, filter_method: str) -> SensoryForgeConfig:
    return SensoryForgeConfig(
        grids=[
            GridConfig(name="grid", arrangement="grid", rows=5, cols=5, spacing=0.15)
        ],
        populations=[
            PopulationConfig(
                name="Pop",
                target_grid="grid",
                neuron_type="SA",
                neuron_model=neuron_model,
                filter_method=filter_method,
                innervation_method="gaussian",
                neurons_per_row=3,
            )
        ],
        stimulus=StimulusConfig(type="gaussian", amplitude=10.0, sigma=1.0),
        simulation=SimulationConfig(device="cpu", dt_ms=0.1),
    )


class TestPluginRegistrationCaseInsensitive:
    """A plugin registered outside register_components.py, referenced by
    either casing, must work identically end-to-end through SimulationEngine.
    """

    @classmethod
    def setup_class(cls):
        register_all()
        # Register the plugin components once, outside register_components.py,
        # exactly as a third-party plugin would.
        NEURON_REGISTRY.register("DemoNeuron", _DemoNeuron)
        FILTER_REGISTRY.register("DemoGain", _DemoGain)

    @pytest.mark.parametrize(
        "neuron_spelling,filter_spelling",
        [
            ("DemoNeuron", "DemoGain"),
            ("demoneuron", "demogain"),
        ],
    )
    def test_engine_builds_with_either_casing(self, neuron_spelling, filter_spelling):
        engine = SimulationEngine(
            _demo_engine_config(neuron_spelling, filter_spelling)
        )
        neuron = engine.populations[0]["neuron"]
        filt = engine.populations[0]["filter"]
        assert isinstance(neuron, _DemoNeuron)
        assert isinstance(filt, _DemoGain)

    def test_both_spellings_run_identically(self):
        torch.manual_seed(0)
        stim1 = torch.rand(1, 5, 5, 5)
        torch.manual_seed(0)
        stim2 = torch.rand(1, 5, 5, 5)

        engine_exact = SimulationEngine(_demo_engine_config("DemoNeuron", "DemoGain"))
        engine_lower = SimulationEngine(_demo_engine_config("demoneuron", "demogain"))

        result_exact = engine_exact.run(stim1)
        result_lower = engine_lower.run(stim2)

        assert torch.equal(
            result_exact["Pop"]["spikes"], result_lower["Pop"]["spikes"]
        )

    def test_registering_different_class_under_case_variant_raises(self):
        class _OtherNeuron(torch.nn.Module):
            def forward(self, x):
                return x

        with pytest.raises(ValueError):
            NEURON_REGISTRY.register("demoneuron", _OtherNeuron)

    def test_reregistering_same_class_under_case_variant_is_idempotent(self):
        # Should not raise, and display name stays the first-registered spelling.
        NEURON_REGISTRY.register("DEMONEURON", _DemoNeuron)
        names = [n.lower() for n in NEURON_REGISTRY.list_registered()]
        assert names.count("demoneuron") == 1


class TestComponentRegistryCaseFolding:
    """Direct unit tests on ComponentRegistry, independent of the built-in
    registries.
    """

    def test_lookup_is_case_insensitive(self):
        registry = ComponentRegistry("test")

        class Foo:
            pass

        registry.register("Foo", Foo)
        assert registry.is_registered("foo")
        assert registry.is_registered("FOO")
        assert registry.get_class("foo") is Foo
        assert registry.get_class("FOO") is Foo

    def test_create_is_case_insensitive(self):
        registry = ComponentRegistry("test")

        class Foo:
            def __init__(self, value=1):
                self.value = value

        registry.register("Foo", Foo)
        instance = registry.create("FOO", value=42)
        assert instance.value == 42

    def test_collision_with_different_class_raises(self):
        registry = ComponentRegistry("test")

        class A:
            pass

        class B:
            pass

        registry.register("thing", A)
        with pytest.raises(ValueError):
            registry.register("THING", B)

    def test_reregistering_same_class_is_idempotent_and_keeps_first_spelling(self):
        registry = ComponentRegistry("test")

        class A:
            pass

        registry.register("Thing", A)
        registry.register("THING", A)
        registry.register("thing", A)

        assert registry.list_registered() == ["Thing"]

    def test_list_registered_has_no_case_duplicates(self):
        registry = ComponentRegistry("test")

        class A:
            pass

        class B:
            pass

        registry.register("Alpha", A)
        registry.register("Beta", B)
        registry.register("alpha", A)  # idempotent, same class

        names = registry.list_registered()
        folded = [n.lower() for n in names]
        assert len(folded) == len(set(folded))
        assert "Alpha" in names
        assert "alpha" not in names


class TestExistingAliasesStillResolve:
    """Removing pure case-variant registrations from register_components.py
    must not break existing configs that spell components any which way.
    """

    @classmethod
    def setup_class(cls):
        register_all()

    @pytest.mark.parametrize("name", ["Izhikevich", "izhikevich", "IZHIKEVICH"])
    def test_izhikevich_resolves(self, name):
        assert NEURON_REGISTRY.is_registered(name)

    @pytest.mark.parametrize("name", ["SA", "sa", "Sa"])
    def test_sa_neuron_resolves(self, name):
        assert NEURON_REGISTRY.is_registered(name)

    @pytest.mark.parametrize("name", ["SA", "sa"])
    def test_sa_filter_resolves(self, name):
        assert FILTER_REGISTRY.is_registered(name)

    def test_dsl_custom_alias_still_distinct(self):
        assert NEURON_REGISTRY.is_registered("DSL (Custom)")
        assert NEURON_REGISTRY.is_registered("dsl")

    def test_safilter_alias_still_distinct(self):
        assert FILTER_REGISTRY.is_registered("safilter")


class TestListRegisteredNoDuplicatesGlobally:
    def test_neuron_registry_list_no_case_duplicates(self):
        register_all()
        names = NEURON_REGISTRY.list_registered()
        folded = [n.lower() for n in names]
        assert len(folded) == len(set(folded))

    def test_filter_registry_list_no_case_duplicates(self):
        register_all()
        names = FILTER_REGISTRY.list_registered()
        folded = [n.lower() for n in names]
        assert len(folded) == len(set(folded))
