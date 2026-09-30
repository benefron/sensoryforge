"""Receptor (sensor) noise lives in the sensors, before innervation.

``SimulationConfig.receptor_noise_std`` is one Gaussian source per receptor, in
the stimulus's own (pressure) units, added to the stimulus sampled on the
receptor grid before any population's receptive-field bank. It is drawn once
per run and shared by every population; pooling, filtering and gain then
propagate it, so a neuron pooling receptors with weights ``w`` receives input
noise of std ``sigma * ||w||_2``. ``PopulationConfig.sensor_noise_std`` is a
different thing -- the per-population neuron input (current) noise, added after
the filter and gain -- and is unchanged (``tests/unit/test_sensor_noise_split.py``).
Presets stay bit-identical (``tests/integration/test_noise_split_golden.py``).
"""

from __future__ import annotations

import json
import shutil
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


def _stimulus(time_steps: int, rows: int = 8, cols: int = 8) -> torch.Tensor:
    """A held pressure blob, ``[1, T, rows, cols]``."""
    y, x = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    blob = torch.exp(-(((x - cols / 2) ** 2 + (y - rows / 2) ** 2) / 8.0))
    return blob.expand(time_steps, rows, cols).unsqueeze(0).float().contiguous()


def _config(populations, **simulation) -> SensoryForgeConfig:
    simulation.setdefault("seed", 0)
    return SensoryForgeConfig(
        grids=[GridConfig(name="g", rows=8, cols=8, spacing=0.5)],
        populations=populations,
        # integrate at the record step: these tests read the current only
        simulation=SimulationConfig(
            device="cpu", dt_ms=1.0, integrate_dt_ms=1.0, **simulation
        ),
    )


def _one_to_one(name: str, gain: float = 3.0, **extra) -> PopulationConfig:
    return PopulationConfig(
        name=name,
        target_grid="g",
        innervation_method="one_to_one",
        connections_per_neuron=1,
        neurons_per_row=8,
        filter_method="none",
        input_gain=gain,
        seed=1,
        **extra,
    )


def _gaussian(name: str, seed: int, connections: int = 12) -> PopulationConfig:
    return PopulationConfig(
        name=name,
        target_grid="g",
        innervation_method="gaussian",
        connections_per_neuron=connections,
        sigma_d_mm=0.6,
        neurons_per_row=4,
        filter_method="none",
        input_gain=1.0,
        seed=seed,
    )


def _run(config: SensoryForgeConfig, stimulus: torch.Tensor):
    engine = SimulationEngine(config)
    return engine, engine.run(stimulus, return_intermediates=True)


def _residual(config_fn, sigma, stimulus, key="drive", **noise):
    """Run clean and noisy; return (engine, {pop: noisy - clean} for *key*)."""
    _, clean = _run(config_fn(), stimulus)
    engine, noisy = _run(config_fn(receptor_noise_std=sigma, **noise), stimulus)
    return engine, {name: (noisy[name][key] - clean[name][key])[0] for name in noisy}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_default_is_unset_and_not_written():
    sim = SimulationConfig()
    assert sim.receptor_noise_std is None and sim.receptor_noise_seed is None
    assert sim.effective_receptor_noise_std() == 0.0
    assert "receptor_noise_std" not in sim.to_dict()
    assert "receptor_noise_seed" not in sim.to_dict()


def test_round_trips_through_yaml():
    cfg = _config([_one_to_one("P")], receptor_noise_std=0.25, receptor_noise_seed=7)
    again = SensoryForgeConfig.from_yaml(cfg.to_yaml()).simulation
    assert (again.receptor_noise_std, again.receptor_noise_seed) == (0.25, 7)


def test_negative_std_raises():
    with pytest.raises(ValueError, match="receptor_noise_std"):
        SimulationConfig(receptor_noise_std=-0.1)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def test_zero_is_bit_identical_to_unset():
    stim = _stimulus(300)

    def cfg(**sim):
        return _config(
            [_one_to_one("A", sensor_noise_std=0.5, noise_seed=3), _gaussian("B", 2)],
            **sim,
        )

    _, unset = _run(cfg(), stim)
    _, zero = _run(cfg(receptor_noise_std=0.0, receptor_noise_seed=5), stim)
    for name in ("A", "B"):
        for key in ("spikes", "drive", "filtered", "voltages"):
            assert torch.equal(unset[name][key], zero[name][key]), (name, key)


def test_one_to_one_current_noise_is_sigma_times_gain():
    """One receptor per neuron, weight 1, no filter: the recorded current's
    noise std is sigma * gain (within 5 %)."""
    sigma, gain = 0.2, 3.0
    engine, residual = _residual(
        lambda **sim: _config([_one_to_one("P", gain=gain)], **sim),
        sigma,
        _stimulus(2000),
        key="filtered",
        receptor_noise_seed=11,
    )
    weights = engine.populations[0]["bank"].weights
    assert torch.equal(weights.norm(dim=1), torch.ones(weights.shape[0]))
    assert float(residual["P"].std()) == pytest.approx(sigma * gain, rel=0.05)


def test_gaussian_pooling_noise_is_sigma_times_weight_norm():
    """Per neuron, the pooled input noise std is sigma * ||w||_2."""
    sigma = 0.3
    engine, residual = _residual(
        lambda **sim: _config([_gaussian("P", seed=4)], **sim),
        sigma,
        _stimulus(6000),
        receptor_noise_seed=12,
    )
    weights = engine.populations[0]["bank"].weights
    predicted = sigma * weights.norm(dim=1)
    measured = residual["P"].std(dim=0)
    assert torch.allclose(measured, predicted, rtol=0.05)


def test_populations_share_one_realisation():
    """Two populations on one grid read the same noisy receptor frames: their
    input noise covariance is sigma^2 * W_A W_B^T, and identical wiring gives
    identical noise."""
    sigma = 0.3
    engine, residual = _residual(
        lambda **sim: _config(
            [
                _one_to_one("A1"),
                _one_to_one("A2", gain=7.0),
                _gaussian("B", seed=4),
                _gaussian("C", seed=9, connections=20),
            ],
            **sim,
        ),
        sigma,
        _stimulus(8000),
        receptor_noise_seed=13,
    )
    assert torch.equal(residual["A1"], residual["A2"])

    banks = {pop["config"].name: pop["bank"].weights for pop in engine.populations}
    n_b, n_c = residual["B"], residual["C"]
    cross = (n_b - n_b.mean(0)).T @ (n_c - n_c.mean(0)) / (n_b.shape[0] - 1)
    w_b, w_c = banks["B"], banks["C"]
    predicted = sigma**2 * w_b @ w_c.T
    corr_measured = cross / (n_b.std(0)[:, None] * n_c.std(0)[None, :])
    corr_predicted = predicted / (
        sigma**2 * w_b.norm(dim=1)[:, None] * w_c.norm(dim=1)[None, :]
    )
    assert float((corr_measured - corr_predicted).abs().max()) < 0.05


def test_seed_fixes_the_noise_whatever_the_run_seed():
    stim = _stimulus(200)

    def run(run_seed, noise_seed):
        cfg = _config(
            [_gaussian("P", seed=4)],
            seed=run_seed,
            receptor_noise_std=0.3,
            receptor_noise_seed=noise_seed,
        )
        return _run(cfg, stim)[1]["P"]["drive"]

    assert torch.equal(run(0, 21), run(99, 21))
    assert not torch.equal(run(0, 21), run(0, 22))


def test_unseeded_noise_follows_the_run_seed():
    stim = _stimulus(200)

    def run(run_seed):
        cfg = _config([_gaussian("P", seed=4)], seed=run_seed, receptor_noise_std=0.3)
        return _run(cfg, stim)[1]["P"]["drive"]

    assert torch.equal(run(3), run(3))
    assert not torch.equal(run(3), run(4))


def test_bundle_keeps_the_clean_stimulus_and_records_the_noise(tmp_path):
    h5py = pytest.importorskip("h5py")
    stim = _stimulus(100)
    cfg = _config([_one_to_one("P")], receptor_noise_std=0.3, receptor_noise_seed=8)
    SimulationEngine(cfg).run(stim, bundle_dir=tmp_path / "b")
    config_json = json.loads((tmp_path / "b" / "config.json").read_text())
    sim = config_json["config"]["simulation"]
    assert (sim["receptor_noise_std"], sim["receptor_noise_seed"]) == (0.3, 8)
    with h5py.File(tmp_path / "b" / "data.h5", "r") as f:
        stored = torch.from_numpy(f["stimulus"]["frames"][()])
    assert torch.equal(stored, stim[0])


# ---------------------------------------------------------------------------
# load_design
# ---------------------------------------------------------------------------


def _design_with_receptor_noise(tmp_path: Path, std, seed=None) -> Path:
    design_dir = tmp_path / "design"
    shutil.copytree(DESIGN_FIXTURE, design_dir)
    manifest = json.loads((design_dir / "design.json").read_text())
    manifest["receptor_noise_std"] = std
    if seed is not None:
        manifest["receptor_noise_seed"] = seed
    (design_dir / "design.json").write_text(json.dumps(manifest))
    return design_dir


def test_design_without_receptor_noise_leaves_it_unset():
    sim = load_design(DESIGN_FIXTURE).simulation
    assert sim.receptor_noise_std is None and sim.receptor_noise_seed is None


def test_load_design_reads_receptor_noise(tmp_path):
    sim = load_design(_design_with_receptor_noise(tmp_path, 0.05, seed=17)).simulation
    assert (sim.receptor_noise_std, sim.receptor_noise_seed) == (0.05, 17)


def test_load_design_rejects_negative_receptor_noise(tmp_path):
    with pytest.raises(ValueError, match="receptor_noise_std"):
        load_design(_design_with_receptor_noise(tmp_path, -1.0))


def test_imported_pooling_noise_is_sigma_times_weight_norm(tmp_path):
    """The imported (designed) banks: per-neuron input noise sigma * ||w||_2,
    and SA and RA see one realisation, correlated as their weights predict."""
    sigma = 0.05
    stim = _stimulus(8000)

    def run(design_dir):
        config = load_design(design_dir)
        config.simulation.integrate_dt_ms = 1.0
        return _run(config, stim)

    _, clean = run(_design_with_receptor_noise(tmp_path / "clean", None))
    engine, noisy = run(_design_with_receptor_noise(tmp_path / "noisy", sigma, seed=3))
    banks = {pop["config"].name: pop["bank"].weights for pop in engine.populations}
    residual = {n: (noisy[n]["drive"] - clean[n]["drive"])[0] for n in banks}
    for name, weights in banks.items():
        assert torch.allclose(
            residual[name].std(dim=0), sigma * weights.norm(dim=1), rtol=0.05
        ), name

    n_sa, n_ra = residual["sa"], residual["ra"]
    cross = (n_sa - n_sa.mean(0)).T @ (n_ra - n_ra.mean(0)) / (n_sa.shape[0] - 1)
    predicted = sigma**2 * banks["sa"] @ banks["ra"].T
    scale = (
        sigma**2 * banks["sa"].norm(dim=1)[:, None] * banks["ra"].norm(dim=1)[None, :]
    )
    assert float(((cross - predicted) / scale).abs().max()) < 0.05
