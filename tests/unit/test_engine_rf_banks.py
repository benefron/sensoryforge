"""The engine and pipelines build receptive fields as banks (Phase 2, I6; F-051).

Before I6, ``SimulationEngine`` built ``InnervationModule`` for every
non-composite grid without passing ``innervation_method``, so ``gaussian``,
``uniform``, ``one_to_one`` and ``distance_weighted`` produced bit-identical
weights (F-051). Now every population's receptive fields come from
``INNERVATION_REGISTRY.get_class(method)`` and ``build()``.

``tests/fixtures/rf_engine_golden_weights.pt`` records, from the a513dfc
code, the gaussian weights the engine's grid path and the legacy
``GeneralizedTactileEncodingPipeline`` produced for a 12x12 grid at 0.15 mm
with 3x3 neurons and seed 5; the bank-based paths must reproduce them
exactly (Phase 2 guardrail 1).
"""

import warnings
from pathlib import Path

import pytest
import torch

from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.core.generalized_pipeline import GeneralizedTactileEncodingPipeline
from sensoryforge.core.innervation import FlatInnervationModule, InnervationModule
from sensoryforge.core.rf_bank import ReceptiveFieldBank
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.registry import INNERVATION_REGISTRY

GOLDEN = (
    Path(__file__).resolve().parents[1] / "fixtures" / "rf_engine_golden_weights.pt"
)
METHODS = ["gaussian", "uniform", "one_to_one", "distance_weighted"]
ROWS, COLS = 12, 12


def _config(method="gaussian", **pop_kwargs):
    pop = dict(
        name="SA",
        neuron_type="SA",
        neurons_per_row=3,
        seed=5,
        innervation_method=method,
        filter_method="none",
    )
    pop.update(pop_kwargs)
    return SensoryForgeConfig(
        grids=[
            GridConfig(name="g", arrangement="grid", rows=ROWS, cols=COLS, spacing=0.15)
        ],
        populations=[PopulationConfig(**pop)],
        simulation=SimulationConfig(device="cpu", dt_ms=1.0),
    )


@pytest.fixture(scope="module")
def golden():
    return torch.load(GOLDEN, weights_only=False)


# ---------------------------------------------------------------------------
# F-051: the four methods now differ and each equals its builder's output
# ---------------------------------------------------------------------------


def test_engine_population_holds_a_bank():
    engine = SimulationEngine(_config())
    pop = engine.populations[0]
    assert isinstance(pop["innervation"], ReceptiveFieldBank)
    assert pop["bank"] is pop["innervation"]
    assert pop["innervation"].provenance["builder"] == "gaussian"
    assert pop["innervation"].provenance["seed"] == 5


def test_four_methods_give_four_different_weight_matrices():
    weights = {
        m: SimulationEngine(_config(m)).populations[0]["bank"].weights for m in METHODS
    }
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1 :]:
            assert not torch.equal(weights[a], weights[b]), f"{a} == {b} (F-051)"


@pytest.mark.parametrize("method", METHODS)
def test_engine_weights_equal_the_builders_own_build(method):
    engine = SimulationEngine(_config(method))
    pop = engine.populations[0]
    cls = INNERVATION_REGISTRY.get_class(method)
    params = SimulationEngine.builder_params(pop["config"], grid_path=True)
    builder = cls.from_config(
        {
            **cls.filter_params(params),
            "receptor_coords": engine.grids[0].get_receptor_coordinates(),
            "neuron_centers": pop["neuron_centers"],
            "device": "cpu",
        }
    )
    assert torch.equal(pop["bank"].weights, builder.build().weights)


def test_unknown_method_still_raises():
    with pytest.raises(ValueError, match="Unknown innervation method"):
        SimulationEngine(_config("no_such_method"))


# ---------------------------------------------------------------------------
# A single-pixel stimulus drives exactly the neurons wired to that receptor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("i,j", [(0, 0), (5, 7), (11, 3)])
def test_single_pixel_drives_exactly_the_wired_neurons(i, j):
    engine = SimulationEngine(_config("gaussian"))
    bank = engine.populations[0]["bank"]
    stim = torch.zeros(1, ROWS, COLS)
    stim[0, i, j] = 1.0
    out = engine.run(stim, return_intermediates=True)["SA"]
    drive = out["drive"][0, 0]  # [N]
    column = bank.weights[:, i * COLS + j]
    assert torch.equal(drive != 0, column != 0)
    assert torch.allclose(drive, column)


def test_receptor_count_mismatch_names_shapes():
    engine = SimulationEngine(_config("gaussian"))
    with pytest.raises(ValueError, match=r"144"):
        engine.run(torch.zeros(1, 10, 10))


# ---------------------------------------------------------------------------
# Behaviour preservation: gaussian on the grid path is bit-identical to a513dfc
# ---------------------------------------------------------------------------


def test_engine_gaussian_grid_path_matches_recorded_weights(golden):
    engine = SimulationEngine(_config("gaussian"))
    bank = engine.populations[0]["bank"]
    rec = golden["engine"]
    assert torch.equal(bank.weights, rec["weights"].reshape(9, ROWS * COLS))
    assert torch.equal(bank.neuron_centers, rec["neuron_centers"])
    assert torch.equal(bank.receptor_coords, rec["receptor_coords"])


def test_legacy_pipeline_matches_recorded_weights(golden):
    rec = golden["legacy_pipeline"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        p = GeneralizedTactileEncodingPipeline(config_dict=rec["config"])
    for key in ("sa", "ra", "sa2"):
        bank = getattr(p, f"{key}_innervation")
        assert isinstance(bank, ReceptiveFieldBank)
        n = rec[f"{key}_weights"].shape[0]
        assert torch.equal(bank.weights, rec[f"{key}_weights"].reshape(n, -1))
    assert torch.equal(p.sa_innervation.neuron_centers, rec["sa_centers"])
    out = p.forward(stimulus_type="gaussian", amplitude=30.0, sigma=0.5)
    assert out["sa_spikes"].shape[-1] == 9


# ---------------------------------------------------------------------------
# Deprecated wrappers: still work, warn, and hold a bank
# ---------------------------------------------------------------------------


def test_innervation_module_is_a_deprecated_wrapper_over_a_bank(golden):
    from sensoryforge.core.grid import GridManager

    gm = GridManager(grid_size=(ROWS, COLS), spacing=0.15)
    with pytest.warns(DeprecationWarning, match="ReceptiveFieldBank"):
        # weight_range matches PopulationConfig's default, which the engine
        # fixture was recorded with (the wrapper's own default is (0.1, 1.0)).
        mod = InnervationModule(
            neuron_type="SA",
            grid_manager=gm,
            neurons_per_row=3,
            seed=5,
            weight_range=(0.05, 1.0),
        )
    assert isinstance(mod.bank, ReceptiveFieldBank)
    assert tuple(mod.innervation_weights.shape) == (9, ROWS, COLS)
    assert torch.equal(mod.innervation_weights, golden["engine"]["weights"])
    assert mod.num_neurons == 9
    assert tuple(mod.neuron_centers.shape) == (9, 2)
    out = mod(torch.rand(2, 5, ROWS, COLS))
    assert tuple(out.shape) == (2, 5, 9)


def test_flat_innervation_module_is_a_deprecated_wrapper_over_a_bank():
    coords = torch.rand(40, 2)
    with pytest.warns(DeprecationWarning, match="ReceptiveFieldBank"):
        mod = FlatInnervationModule(
            neuron_type="RA", receptor_coords=coords, neurons_per_row=2, seed=1
        )
    assert isinstance(mod.bank, ReceptiveFieldBank)
    assert tuple(mod.innervation_weights.shape) == (4, 40)
    assert mod.num_neurons == 4 and mod.num_receptors == 40
    assert tuple(mod(torch.rand(3, 40)).shape) == (3, 4)


# ---------------------------------------------------------------------------
# Derived-lattice builders through the engine
# ---------------------------------------------------------------------------


def test_template_through_engine_derives_neuron_count_and_warns():
    cfg = _config("template", resolvable_distance_mm=0.40, neurons_per_row=7)
    with pytest.warns(UserWarning, match="neurons_per_row"):
        engine = SimulationEngine(cfg)
    bank = engine.populations[0]["bank"]
    # 12x12 at 0.15 mm: extended side 1.8 mm -> 4x4 neurons at pitch 0.40
    assert bank.num_neurons == 16
    assert bank.provenance["builder"] == "template"
    out = engine.run(torch.rand(1, 3, ROWS, COLS))["SA"]
    assert out["spikes"].shape[-1] == 16


def test_innervation_params_reach_the_builder():
    cfg = _config(
        "template",
        resolvable_distance_mm=0.40,
        innervation_params={"k": 5, "normalize": "sum"},
    )
    with pytest.warns(UserWarning):
        bank = SimulationEngine(cfg).populations[0]["bank"]
    assert torch.all((bank.weights != 0).sum(dim=1) == 5)
    assert torch.allclose(bank.weights.sum(dim=1), torch.ones(bank.num_neurons))


def test_population_config_new_fields_round_trip():
    pop = PopulationConfig(
        name="p",
        innervation_method="template",
        resolvable_distance_mm=0.4,
        innervation_params={"k": 3},
    )
    back = PopulationConfig.from_dict(pop.to_dict())
    assert back.resolvable_distance_mm == 0.4 and back.innervation_params == {"k": 3}
    assert PopulationConfig.from_dict({"name": "q"}).resolvable_distance_mm is None


def test_non_grid_arrangement_warns_naming_f010():
    cfg = SensoryForgeConfig(
        grids=[
            GridConfig(
                name="g", arrangement="poisson", rows=8, cols=8, spacing=0.2, seed=1
            )
        ],
        populations=[
            PopulationConfig(name="SA", neurons_per_row=2, seed=1, filter_method="none")
        ],
    )
    with pytest.warns(UserWarning, match="F-010"):
        SimulationEngine(cfg)
