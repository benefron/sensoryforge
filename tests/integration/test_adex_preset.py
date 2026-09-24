"""The ``tactile_sa1_ra1_adex`` preset loads and runs in the engine (Phase 2b).

``tactile_sa1_ra1_adex`` is ``tactile_sa1_ra1`` with ``neuron_model: AdEx``.
It ships as a separate file so the Izhikevich preset stays pinned by
``tests/integration/test_pressure_sim_parity.py``; these tests check that the
AdEx variant differs from it in exactly that one field, that the resolver
gives each population its AdEx preset by ``neuron_type`` (``SA1_tonic`` /
``RA1_phasic``), and that a shrunk copy (8x8 receptors, 20 ms) actually runs
through :class:`~sensoryforge.core.simulation_engine.SimulationEngine` and
produces spike tensors of the right shape.

Shapes and units: the stimulus is ``[batch, time, rows, cols]`` in mA, the
grid spacing is mm, durations are ms.
"""

from __future__ import annotations

import copy

import pytest
import torch

from sensoryforge.config.defaults import resolve_neuron_params
from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.neurons.adex import ADEX_PRESETS
from sensoryforge.presets import list_presets, load_preset, preset_description
from sensoryforge.core.simulation_engine import SimulationEngine

PRESET_NAME = "tactile_sa1_ra1_adex"


def test_preset_is_shipped_and_described():
    assert PRESET_NAME in list_presets()
    assert preset_description(PRESET_NAME)


def test_preset_loads_as_a_valid_canonical_config():
    config = SensoryForgeConfig.from_dict(load_preset(PRESET_NAME))
    assert len(config.grids) == 1
    assert config.grids[0].rows == 80
    assert config.grids[0].cols == 80
    assert {p.name for p in config.populations} == {"SA Population", "RA Population"}
    for pop in config.populations:
        assert pop.neuron_model.lower() == "adex"
        assert pop.innervation_method == "template"
        assert pop.resolvable_distance_mm == pytest.approx(0.40)


def test_differs_from_izhikevich_preset_only_in_neuron_model():
    """The AdEx preset is a copy: nothing but neuron_model, the
    human-readable metadata, each population's input_gain and its
    model_params may drift, or the two stop being comparable. Gains and
    neuron parameters differ because each recipe's are fitted for its own
    neuron model (D-ea0f017, D-f4d0967, D-d9bd411)."""
    base = load_preset("tactile_sa1_ra1")
    adex = load_preset(PRESET_NAME)
    base.pop("metadata", None)
    adex.pop("metadata", None)
    normalised = copy.deepcopy(adex)
    for pop in normalised["populations"]:
        pop["neuron_model"] = "izhikevich"
    for pops in (normalised["populations"], base["populations"]):
        for pop in pops:
            pop.pop("input_gain")
            pop.pop("model_params", None)
    assert normalised == base


def test_resolver_gives_each_population_its_adex_preset_by_type():
    config = SensoryForgeConfig.from_dict(load_preset(PRESET_NAME))
    expected = {"SA": "SA1_tonic", "RA": "RA1_phasic"}
    for pop in config.populations:
        params = resolve_neuron_params(
            pop.neuron_model, pop.neuron_type, pop.model_params
        )
        assert "preset" not in params
        for key, value in ADEX_PRESETS[expected[pop.neuron_type]].items():
            assert params[key] == pytest.approx(value)


def test_preset_runs_in_the_engine_on_an_8x8_grid_for_20_ms():
    """A shrunk copy of the preset runs end to end and spikes have the right
    shape. 80x80 for the full recipe is a benchmark, not a unit test."""
    data = load_preset(PRESET_NAME)
    data["grids"][0]["rows"] = 8
    data["grids"][0]["cols"] = 8
    config = SensoryForgeConfig.from_dict(data)
    config.simulation.dt_ms = 1.0

    n_bins = 20  # 20 ms at dt_ms = 1.0
    # A held ramp: 5 ms rise then a plateau, so SA sees a sustained drive and
    # RA sees an onset transient. Amplitude in mA.
    ramp = torch.linspace(0.0, 1.0, 5)
    stimulus = torch.ones(1, n_bins, 8, 8)
    stimulus[:, :5, :, :] = ramp.view(1, 5, 1, 1)

    engine = SimulationEngine(config)
    results = engine.run(stimulus, seed=0)

    assert set(results) == {"SA Population", "RA Population"}
    for pop_results in results.values():
        # `spikes` is the engine's per-bin sub-step spike *count*
        # (simulation_engine._run_pop_from_drive), not a boolean raster.
        spikes = pop_results["spikes"]
        assert spikes.shape[0] == 1
        assert spikes.shape[1] == n_bins
        assert spikes.shape[2] > 0
        assert torch.isfinite(spikes).all()
        assert float(spikes.min()) >= 0.0
