"""``check_component`` refuses an incomplete round trip, for every component kind.

F-049: the full ``to_dict()`` completeness check (every ``__init__`` parameter is
serialised, and ``from_config(to_dict())`` is a fixed point) used to run for
neurons only. Each kind now has to refuse a component that drops a parameter, and
one whose ``from_config`` ignores a key. A built-in that already passes is no proof
the check bites, so these tests break a built-in on purpose.
"""

import pytest
import torch

from sensoryforge.core.grid_arrangements import GridArrangement
from sensoryforge.core.innervation import GaussianInnervation
from sensoryforge.core.processing import OnOffLayer
from sensoryforge.filters.sa_ra import SAFilterTorch
from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch
from sensoryforge.solvers.euler import EulerSolver
from sensoryforge.stimuli.gaussian import GaussianStimulus
from sensoryforge.testing.contracts import available_kinds, check_component

_COORDS = torch.rand(20, 2, generator=torch.Generator().manual_seed(0))
_CENTRES = torch.rand(4, 2, generator=torch.Generator().manual_seed(1))
_ON_OFF_COORDS = torch.rand(8, 2, generator=torch.Generator().manual_seed(2))


#: kind -> (base class, factory(cls, **overrides), the parameter the tests break,
#: a non-default value for it, whether ``to_dict`` nests it under ``"params"``).
CASES = {
    "neuron": (
        IzhikevichNeuronTorch,
        lambda cls, **kw: cls(**kw),
        "threshold",
        25.0,
        False,
    ),
    "filter": (SAFilterTorch, lambda cls, **kw: cls(**kw), "k2", 0.123, False),
    "grid": (
        GridArrangement,
        lambda cls, **kw: cls(grid_size=4, **kw),
        "spacing",
        0.37,
        False,
    ),
    "solver": (EulerSolver, lambda cls, **kw: cls(**kw), "dt", 0.123, False),
    "innervation": (
        GaussianInnervation,
        lambda cls, **kw: cls(_COORDS, _CENTRES, device="cpu", **kw),
        "sigma_d_mm",
        0.9,
        False,
    ),
    "stimulus": (GaussianStimulus, lambda cls, **kw: cls(**kw), "sigma", 0.77, False),
    "processing": (
        OnOffLayer,
        lambda cls, **kw: cls(_ON_OFF_COORDS, **kw),
        "sigma_surround_mm",
        0.6,
        True,
    ),
}


def _dict_of(config, nested):
    return config["params"] if nested else config


def _dropping(cls, key, nested):
    """``cls`` with a ``to_dict`` that leaves out ``key``."""

    class Dropping(cls):
        def to_dict(self):
            out = super().to_dict()
            _dict_of(out, nested).pop(key)
            return out

    Dropping.__name__ = f"{cls.__name__}WithoutTheValueOf_{key}"
    return Dropping


def _ignoring(cls, key, nested):
    """``cls`` with a ``from_config`` that ignores ``key`` (``to_dict`` is whole)."""

    class Ignoring(cls):
        @classmethod
        def from_config(cls_, config, **kwargs):
            config = {**config}
            if nested:
                config["params"] = {**config["params"]}
            _dict_of(config, nested).pop(key, None)
            return super().from_config(config, **kwargs)

    Ignoring.__name__ = f"{cls.__name__}IgnoringTheKey_{key}"
    return Ignoring


def test_every_kind_has_a_case():
    """A kind added to ``check_component`` must be added here, with its case."""
    assert sorted(CASES) == available_kinds()


@pytest.mark.parametrize("kind", sorted(CASES))
def test_the_shipped_component_passes(kind):
    cls, make, _, _, _ = CASES[kind]
    check_component(kind, cls, instance=make(cls))


@pytest.mark.parametrize("kind", sorted(CASES))
def test_a_to_dict_that_drops_a_parameter_fails(kind):
    cls, make, key, _, nested = CASES[kind]
    broken = _dropping(cls, key, nested)
    with pytest.raises(AssertionError, match=f"missing constructor parameters.*{key}"):
        check_component(kind, broken, instance=make(broken))


@pytest.mark.parametrize("kind", sorted(CASES))
def test_a_from_config_that_ignores_a_key_fails_on_a_non_default_instance(kind):
    cls, make, key, odd, nested = CASES[kind]
    broken = _ignoring(cls, key, nested)
    # A default instance reaches the fixed point whatever from_config ignores...
    check_component(kind, broken, instance=make(broken))
    # ...so the check bites on an instance built with a non-default value.
    with pytest.raises(AssertionError, match="not a fixed point"):
        check_component(kind, broken, instance=make(broken, **{key: odd}))


def test_a_processing_layer_may_serialise_a_parameter_at_the_top_level():
    """``to_dict`` can put a parameter beside ``method`` or under ``params``."""

    class FlatOnOff(OnOffLayer):
        def to_dict(self):
            return {"method": "onoff", **super().to_dict()["params"]}

    check_component("processing", FlatOnOff, instance=FlatOnOff(_ON_OFF_COORDS))


def test_a_processing_layer_that_drops_a_parameter_names_it():
    class Dropping(OnOffLayer):
        def to_dict(self):
            return {"method": "onoff", "params": {"sigma_center_mm": 0.15}}

    with pytest.raises(AssertionError, match=r"missing.*sigma_surround_mm"):
        check_component("processing", Dropping, instance=Dropping(_ON_OFF_COORDS))
