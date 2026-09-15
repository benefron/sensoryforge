"""Tests for G1: ParamSpec extensibility and get_param_spec() on every base class.

Covers:
- ParamSpec gains choices/help/group/advanced without breaking existing
  positional/keyword call sites.
- BaseFilter, BaseNeuron, BaseSolver, BaseGrid, BaseInnervation all expose
  get_param_spec() (default []), matching BaseStimulus's existing contract.
"""

from sensoryforge.stimuli.base import ParamSpec


def test_param_spec_new_fields_default_safely():
    p = ParamSpec("amplitude", dtype="float", default=1.0)
    assert p.choices is None
    assert p.help == ""
    assert p.group == ""
    assert p.advanced is False


def test_param_spec_new_fields_are_keyword_only():
    p = ParamSpec(
        "mode",
        dtype="float",
        default=1.0,
        choices=["a", "b"],
        help="longer help text",
        group="Spatial",
        advanced=True,
    )
    assert p.choices == ["a", "b"]
    assert p.help == "longer help text"
    assert p.group == "Spatial"
    assert p.advanced is True


def test_param_spec_existing_call_site_unaffected():
    # Matches the exact call shape used throughout stimuli/*.py before G1.
    p = ParamSpec(
        "sigma",
        dtype="float",
        default=0.5,
        min_val=0.01,
        max_val=20.0,
        unit="mm",
    )
    assert p.name == "sigma"
    assert p.unit == "mm"


def test_base_filter_has_get_param_spec():
    from sensoryforge.filters.base import BaseFilter

    assert BaseFilter.get_param_spec() == []


def test_base_neuron_has_get_param_spec():
    from sensoryforge.neurons.base import BaseNeuron

    assert BaseNeuron.get_param_spec() == []


def test_base_solver_has_get_param_spec():
    from sensoryforge.solvers.base import BaseSolver

    assert BaseSolver.get_param_spec() == []


def test_base_grid_has_get_param_spec():
    from sensoryforge.core.grid_base import BaseGrid

    assert BaseGrid.get_param_spec() == []


def test_base_innervation_has_get_param_spec():
    from sensoryforge.core.innervation import BaseInnervation

    assert BaseInnervation.get_param_spec() == []


def test_concrete_neuron_models_expose_get_param_spec():
    """Concrete neuron classes must actually inherit BaseNeuron (not just
    plain nn.Module) so get_param_spec()/to_dict()/from_config() resolve."""
    from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
    from sensoryforge.neurons.adex import AdExNeuronTorch
    from sensoryforge.neurons.mqif import MQIFNeuronTorch
    from sensoryforge.neurons.fa import FANeuronTorch
    from sensoryforge.neurons.sa import SANeuronTorch
    from sensoryforge.neurons.base import BaseNeuron

    for cls in (
        IzhikevichNeuronTorch,
        AdExNeuronTorch,
        MQIFNeuronTorch,
        FANeuronTorch,
        SANeuronTorch,
    ):
        assert issubclass(cls, BaseNeuron)
        assert cls.get_param_spec() == []
        instance = cls()
        assert instance.to_dict() == {"dt": instance.dt}
