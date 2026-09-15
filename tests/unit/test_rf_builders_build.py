"""Existing innervation methods build ``ReceptiveFieldBank``s (Phase 2, I3).

``tests/fixtures/rf_builder_golden_weights.pt`` records, from the a513dfc
code, the weights each of the four methods produced through
``FlatInnervationModule`` on a 12x12 grid at 0.15 mm with 3x3 neurons and
seed 5. ``BaseInnervation.build()`` must reproduce them exactly (Phase 2
guardrail 1: behaviour preservation is measured, not assumed).
"""

from pathlib import Path

import pytest
import torch

from sensoryforge.core.innervation import (
    BaseInnervation,
    DistanceWeightedInnervation,
    FlatInnervationModule,
    GaussianInnervation,
    OneToOneInnervation,
    UniformInnervation,
)
from sensoryforge.core.rf_bank import ReceptiveFieldBank
from sensoryforge.register_components import register_all
from sensoryforge.registry import INNERVATION_REGISTRY
from sensoryforge.testing.contracts import check_component

register_all()

GOLDEN = (
    Path(__file__).resolve().parents[1] / "fixtures" / "rf_builder_golden_weights.pt"
)
METHODS = ["gaussian", "uniform", "one_to_one", "distance_weighted"]
CLASSES = {
    "gaussian": GaussianInnervation,
    "uniform": UniformInnervation,
    "one_to_one": OneToOneInnervation,
    "distance_weighted": DistanceWeightedInnervation,
}


@pytest.fixture(scope="module")
def golden():
    return torch.load(GOLDEN, weights_only=False)


def _builder(method, golden):
    cls = CLASSES[method]
    entry = golden[method]
    return cls.from_config(
        {
            **cls.filter_params(entry["params"]),
            "receptor_coords": golden["receptor_coords"],
            "neuron_centers": entry["neuron_centers"],
            "device": "cpu",
        }
    )


def test_build_exists_on_base_class():
    assert callable(getattr(BaseInnervation, "build", None))


def test_filter_params_keeps_only_constructor_keys():
    params = {"sigma_d_mm": 0.2, "max_distance_mm": 0.5, "bogus": 1, "seed": 3}
    assert GaussianInnervation.filter_params(params) == {"sigma_d_mm": 0.2, "seed": 3}
    assert DistanceWeightedInnervation.filter_params(params) == {
        "sigma_d_mm": 0.2,
        "max_distance_mm": 0.5,
        "seed": 3,
    }


@pytest.mark.parametrize("method", METHODS)
def test_build_reproduces_recorded_flat_module_weights(method, golden):
    bank = _builder(method, golden).build()
    assert isinstance(bank, ReceptiveFieldBank)
    assert torch.equal(bank.weights, golden[method]["weights"])
    assert torch.equal(bank.neuron_centers, golden[method]["neuron_centers"])
    assert torch.equal(bank.receptor_coords, golden["receptor_coords"])


@pytest.mark.parametrize("method", METHODS)
def test_flat_module_still_matches_recorded_weights(method, golden):
    # The recorded file is only meaningful if today's FlatInnervationModule
    # still produces it; this pins the legacy path for I6's wrapper.
    module = FlatInnervationModule(
        neuron_type="SA",
        receptor_coords=golden["receptor_coords"],
        neurons_per_row=3,
        innervation_method=method,
        seed=golden["seed"],
        device="cpu",
    )
    assert torch.equal(module.innervation_weights, golden[method]["weights"])


@pytest.mark.parametrize("method", METHODS)
def test_build_provenance(method, golden):
    builder = _builder(method, golden)
    bank = builder.build()
    prov = bank.provenance
    assert prov["builder"] == method
    assert prov["seed"] == 5
    assert "sensoryforge_version" in prov
    cfg = prov["builder_config"]
    assert "receptor_coords" not in cfg and "neuron_centers" not in cfg
    for key in ("method", "num_neurons", "num_receptors"):
        assert key not in cfg
    rebuilt = CLASSES[method].from_config(
        {
            **cfg,
            "receptor_coords": golden["receptor_coords"],
            "neuron_centers": golden[method]["neuron_centers"],
        }
    )
    assert torch.equal(rebuilt.build().weights, bank.weights)


@pytest.mark.parametrize("method", METHODS)
def test_build_accepts_override_coordinates(method, golden):
    builder = _builder(method, golden)
    coords = golden["receptor_coords"][:100]
    centers = golden[method]["neuron_centers"][:4]
    bank = builder.build(coords, centers)
    assert bank.num_receptors == 100 and bank.num_neurons == 4
    direct = (
        CLASSES[method]
        .from_config(
            {
                **CLASSES[method].filter_params(golden[method]["params"]),
                "receptor_coords": coords,
                "neuron_centers": centers,
            }
        )
        .build()
    )
    assert torch.equal(bank.weights, direct.weights)


@pytest.mark.parametrize("method", METHODS)
def test_registry_maps_name_to_contract_checked_class(method, golden):
    cls = INNERVATION_REGISTRY.get_class(method)
    assert cls is CLASSES[method]
    instance = INNERVATION_REGISTRY.create(
        method,
        receptor_coords=torch.rand(20, 2),
        neuron_centers=torch.rand(4, 2),
        device="cpu",
    )
    assert isinstance(instance, cls)
    check_component("innervation", cls, instance)


def test_contract_check_rejects_builder_without_build():
    class Broken(GaussianInnervation):
        def build(self, *a, **k):
            return "not a bank"

    with pytest.raises(AssertionError, match="build"):
        check_component(
            "innervation",
            Broken,
            Broken(torch.rand(10, 2), torch.rand(2, 2), seed=1),
        )
