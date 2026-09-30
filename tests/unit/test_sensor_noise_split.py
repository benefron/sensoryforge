"""Sensor noise and membrane noise are separate parameters (C-130).

``PopulationConfig.sensor_noise_std`` is Gaussian noise (std, mA) added to each
neuron's injected current after the filter and the gain; ``membrane_noise_std``
is the neuron model's own Langevin noise. ``noise_std`` is the deprecated alias
that sets both. ``load_design`` simulates a design directory's exported
``noise_std`` as sensor noise and ``membrane_noise_std`` as membrane noise.

The preset / deprecated-alias bit-identity against the pre-change code is
``tests/integration/test_noise_split_golden.py``.
"""

from __future__ import annotations

import json
import shutil
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
from sensoryforge.core.simulation_engine import SimulationEngine
from sensoryforge.io.design import load_design

DESIGN_FIXTURE = Path(__file__).parent.parent / "fixtures" / "design_8x8"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_new_keys_default_unset_and_are_not_written():
    pop = PopulationConfig(name="SA")
    assert pop.sensor_noise_std is None and pop.membrane_noise_std is None
    assert pop.effective_sensor_noise_std() == 0.0
    assert pop.effective_membrane_noise_std() == 0.0
    written = pop.to_dict()
    assert "sensor_noise_std" not in written
    assert "membrane_noise_std" not in written


def test_new_keys_round_trip():
    pop = PopulationConfig(name="SA", sensor_noise_std=3.5, membrane_noise_std=0.0)
    back = PopulationConfig.from_dict(pop.to_dict())
    assert back.sensor_noise_std == 3.5
    assert back.membrane_noise_std == 0.0
    cfg = SensoryForgeConfig(
        grids=[GridConfig(name="g", rows=4, cols=4)],
        populations=[pop],
    )
    again = SensoryForgeConfig.from_yaml(cfg.to_yaml()).populations[0]
    assert (again.sensor_noise_std, again.membrane_noise_std) == (3.5, 0.0)


def test_deprecated_noise_std_warns_and_sets_both():
    with pytest.warns(FutureWarning, match="noise_std is deprecated"):
        pop = PopulationConfig(name="SA", noise_std=2.0)
    assert pop.effective_sensor_noise_std() == 2.0
    assert pop.effective_membrane_noise_std() == 2.0


def test_explicit_keys_override_the_alias():
    with pytest.warns(FutureWarning, match="membrane_noise_std set explicitly"):
        pop = PopulationConfig(name="SA", noise_std=2.0, membrane_noise_std=0.0)
    assert pop.effective_sensor_noise_std() == 2.0
    assert pop.effective_membrane_noise_std() == 0.0


def test_zero_noise_std_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PopulationConfig(name="SA", sensor_noise_std=1.0, membrane_noise_std=0.5)
        PopulationConfig.from_dict({"name": "SA", "noise_std": 0.0})


@pytest.mark.parametrize("key", ["sensor_noise_std", "membrane_noise_std", "noise_std"])
def test_negative_noise_raises(key):
    with pytest.raises(ValueError, match=key):
        PopulationConfig(name="SA", **{key: -1.0})


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _engine_config(**noise) -> SensoryForgeConfig:
    return SensoryForgeConfig(
        grids=[GridConfig(name="g", rows=8, cols=8, spacing=0.5)],
        populations=[
            PopulationConfig(
                name="SA",
                target_grid="g",
                neuron_type="SA",
                neurons_per_row=4,
                neuron_model="adex",
                filter_method="sa",
                input_gain=200.0,
                seed=1,
                **noise,
            )
        ],
        simulation=SimulationConfig(device="cpu", dt_ms=1.0, seed=0),
    )


def _stimulus(time_steps: int = 400, rows: int = 8, cols: int = 8) -> torch.Tensor:
    """A pressure blob that ramps up and holds, ``[1, T, rows, cols]``."""
    t = torch.linspace(0.0, 1.0, time_steps).clamp(max=0.5) * 2.0
    y, x = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    blob = torch.exp(-(((x - cols / 2) ** 2 + (y - rows / 2) ** 2) / 8.0))
    return (t[:, None, None] * blob[None]).unsqueeze(0).float()


def _run(config: SensoryForgeConfig, stimulus: torch.Tensor):
    engine = SimulationEngine(config)
    return engine, engine.run(stimulus, return_intermediates=True)


def test_alias_is_bit_identical_to_both_explicit_keys():
    stim = _stimulus()
    with pytest.warns(FutureWarning):
        alias_cfg = _engine_config(noise_std=2.0, noise_seed=11)
    _, alias = _run(alias_cfg, stim)
    _, explicit = _run(
        _engine_config(sensor_noise_std=2.0, membrane_noise_std=2.0, noise_seed=11),
        stim,
    )
    assert torch.equal(alias["SA"]["spikes"], explicit["SA"]["spikes"])
    assert torch.equal(alias["SA"]["filtered"], explicit["SA"]["filtered"])


def test_sensor_noise_only_reaches_the_current_not_the_membrane():
    stim = _stimulus(1000)
    _, clean = _run(_engine_config(), stim)
    engine, noisy = _run(
        _engine_config(sensor_noise_std=4.0, membrane_noise_std=0.0, noise_seed=5),
        stim,
    )
    assert engine.populations[0]["neuron"].noise_std == 0.0
    residual = noisy["SA"]["filtered"] - clean["SA"]["filtered"]
    assert float(residual.std()) == pytest.approx(4.0, rel=0.05)
    assert abs(float(residual.mean())) < 0.05 * 4.0


def test_membrane_noise_only_leaves_the_current_clean():
    stim = _stimulus()
    _, clean = _run(_engine_config(), stim)
    engine, noisy = _run(
        _engine_config(sensor_noise_std=0.0, membrane_noise_std=3.0, noise_seed=5),
        stim,
    )
    assert engine.populations[0]["neuron"].noise_std == 3.0
    assert torch.equal(noisy["SA"]["filtered"], clean["SA"]["filtered"])
    assert not torch.equal(noisy["SA"]["spikes"], clean["SA"]["spikes"])


# ---------------------------------------------------------------------------
# load_design
# ---------------------------------------------------------------------------


def _noisy_design(tmp_path: Path, sensor: dict, membrane: float) -> Path:
    """Copy the 8x8 design fixture, declaring per-population noise as
    pressure-simulation's ``design.export.write_design`` does."""
    design_dir = tmp_path / "design"
    shutil.copytree(DESIGN_FIXTURE, design_dir)
    manifest = json.loads((design_dir / "design.json").read_text())
    seeds = {"sa": 42, "ra": 43}
    for prec in manifest["populations"]:
        prec["noise_std"] = sensor[prec["name"]]
        prec["membrane_noise_std"] = membrane
        prec["noise_seed"] = seeds[prec["name"]]
    (design_dir / "design.json").write_text(json.dumps(manifest))
    return design_dir


def test_design_without_noise_keys_runs_noise_free():
    config = load_design(DESIGN_FIXTURE)
    for pop in config.populations:
        assert pop.effective_sensor_noise_std() == 0.0
        assert pop.effective_membrane_noise_std() == 0.0
        assert pop.noise_seed is None


def test_load_design_reads_declared_noise(tmp_path):
    design_dir = _noisy_design(tmp_path, {"sa": 3.59, "ra": 37.9}, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the new keys never warn
        config = load_design(design_dir)
    by_name = {p.name: p for p in config.populations}
    assert by_name["sa"].sensor_noise_std == pytest.approx(3.59)
    assert by_name["ra"].sensor_noise_std == pytest.approx(37.9)
    assert by_name["sa"].membrane_noise_std == 0.0
    assert by_name["ra"].membrane_noise_std == 0.0
    assert (by_name["sa"].noise_seed, by_name["ra"].noise_seed) == (42, 43)
    # survives the CLI's to_dict/from_dict round trip
    again = SensoryForgeConfig.from_dict(config.to_dict())
    assert {p.name: p.sensor_noise_std for p in again.populations} == {
        "sa": pytest.approx(3.59),
        "ra": pytest.approx(37.9),
    }


def test_load_design_rejects_negative_noise(tmp_path):
    design_dir = _noisy_design(tmp_path, {"sa": -1.0, "ra": 1.0}, 0.0)
    with pytest.raises(ValueError, match="noise_std"):
        load_design(design_dir)


def _design_run(design_dir: Path, run_seed: int = 0, time_steps: int = 2000):
    config = load_design(design_dir)
    config.simulation.seed = run_seed
    config.simulation.device = "cpu"
    grid = config.grids[0]
    return _run(config, _stimulus(time_steps, grid.rows, grid.cols))


def test_design_sensor_noise_is_simulated_with_the_declared_std(tmp_path):
    """Sensor noise v, membrane 0: the recorded current carries noise of std v
    (within 5 %), and the spikes are a noiseless function of that current."""
    declared = {"sa": 3.59, "ra": 37.9}
    clean_dir = _noisy_design(tmp_path / "clean", {"sa": 0.0, "ra": 0.0}, 0.0)
    noisy_dir = _noisy_design(tmp_path / "noisy", declared, 0.0)
    _, clean = _design_run(clean_dir)
    engine, noisy = _design_run(noisy_dir)

    for index, pop in enumerate(engine.populations):
        name = pop["config"].name
        residual = noisy[name]["filtered"] - clean[name]["filtered"]
        assert float(residual.std()) == pytest.approx(declared[name], rel=0.05)

        # Zero membrane noise: the neuron is built noiseless, and re-running it
        # on the recorded (noisy) current reproduces the spikes exactly.
        neuron = pop["neuron"]
        assert neuron.noise_std == 0.0
        replay = SimulationEngine._run_pop_from_drive(
            drive=noisy[name]["filtered"],
            filter_module=None,
            neuron_model=neuron,
            input_gain=1.0,
            noise_std=0.0,
            dt_ms=engine.config.simulation.dt_ms,
            integrate_dt_ms=engine.config.simulation.integrate_dt_ms,
            input_floor=0.0,
        )
        assert torch.equal(replay["spikes"], noisy[name]["spikes"])


def test_design_noise_is_seeded_per_population(tmp_path):
    """The declared noise_seed fixes the noise whatever the run seed."""
    noisy_dir = _noisy_design(tmp_path, {"sa": 3.59, "ra": 37.9}, 0.0)
    _, first = _design_run(noisy_dir, run_seed=0, time_steps=300)
    _, second = _design_run(noisy_dir, run_seed=99, time_steps=300)
    for name in ("sa", "ra"):
        assert torch.equal(first[name]["filtered"], second[name]["filtered"])
        assert torch.equal(first[name]["spikes"], second[name]["spikes"])


def test_design_membrane_noise_is_simulated_separately(tmp_path):
    """Sensor 0, membrane m: the current is clean and the neuron carries m."""
    clean_dir = _noisy_design(tmp_path / "clean", {"sa": 0.0, "ra": 0.0}, 0.0)
    membrane_dir = _noisy_design(tmp_path / "membrane", {"sa": 0.0, "ra": 0.0}, 2.0)
    _, clean = _design_run(clean_dir, time_steps=300)
    engine, noisy = _design_run(membrane_dir, time_steps=300)
    for pop in engine.populations:
        name = pop["config"].name
        assert pop["neuron"].noise_std == 2.0
        assert torch.equal(noisy[name]["filtered"], clean[name]["filtered"])
