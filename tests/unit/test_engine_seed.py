"""Tests for Task 0.2: engine progress callback, run seed, per-population noise seed.

Covers:
  - `SimulationConfig.seed` round-trips through to_dict/from_dict and YAML.
  - `simulation.seed` makes a noisy run reproducible across two `SimulationEngine.run()`
    calls in the same process (spikes and filtered drive are bit-identical).
  - `PopulationConfig.noise_seed` makes a single population's noise reproducible
    independent of the run seed, and different seeds give different noise.
  - `progress_cb` fires once per population, in config order, before that
    population's results exist.
  - The `noise_generator=None` default path of `_run_pop_from_drive` is unchanged:
    bit-identical to seeding the global RNG directly (pins the pre-change behaviour).
  - CLI `--seed` exits 0 and sets `config.simulation.seed`.
"""

import subprocess
import sys
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

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    noise_std: float = 0.0,
    noise_seed=None,
    run_seed=None,
    n_populations: int = 1,
) -> SensoryForgeConfig:
    """Minimal canonical config: an 8x8 grid, N SA populations, noise_std configurable."""
    grid = GridConfig(
        name="test_grid",
        rows=8,
        cols=8,
        spacing=1.0,
        arrangement="grid",
    )
    pops = []
    for i in range(n_populations):
        pops.append(
            PopulationConfig(
                name=f"pop_{i}",
                target_grid="test_grid",
                neuron_type="SA",
                neurons_per_row=2,
                innervation_method="gaussian",
                connections_per_neuron=4,
                sigma_d_mm=2.0,
                filter_method="none",
                neuron_model="Izhikevich",
                input_gain=1.0,
                noise_std=noise_std,
                noise_seed=noise_seed,
                seed=1,
            )
        )
    sim = SimulationConfig(dt_ms=0.5, device="cpu", seed=run_seed)
    return SensoryForgeConfig(grids=[grid], populations=pops, simulation=sim)


def _make_stimulus(timesteps: int = 50) -> torch.Tensor:
    """Constant positive stimulus [time, H, W]."""
    return torch.ones(timesteps, 8, 8) * 5.0


# ---------------------------------------------------------------------------
# SimulationConfig.seed round-trip
# ---------------------------------------------------------------------------


def test_simulation_config_seed_round_trips_through_dict():
    cfg = SimulationConfig(seed=7)
    assert cfg.to_dict()["seed"] == 7
    restored = SimulationConfig.from_dict(cfg.to_dict())
    assert restored.seed == 7


def test_simulation_config_seed_round_trips_through_yaml():
    sf_config = SensoryForgeConfig(simulation=SimulationConfig(seed=7))
    yaml_text = sf_config.to_yaml()
    restored = SensoryForgeConfig.from_yaml(yaml_text)
    assert restored.simulation.seed == 7


def test_simulation_config_seed_defaults_to_none():
    cfg = SimulationConfig()
    assert cfg.seed is None
    assert "seed" not in cfg.to_dict()


# ---------------------------------------------------------------------------
# Run seed reproducibility
# ---------------------------------------------------------------------------


def test_run_seed_makes_noisy_run_reproducible():
    """Two runs with the same `simulation.seed` give bit-identical spikes and
    filtered drive (same process -- F-071 does not apply here)."""
    config = _make_config(noise_std=0.5, run_seed=1)
    stim = _make_stimulus()

    engine1 = SimulationEngine(config)
    result1 = engine1.run(stim, return_intermediates=True)

    engine2 = SimulationEngine(config)
    result2 = engine2.run(stim, return_intermediates=True)

    assert torch.equal(result1["pop_0"]["spikes"], result2["pop_0"]["spikes"])
    assert torch.equal(result1["pop_0"]["filtered"], result2["pop_0"]["filtered"])


def test_no_run_seed_no_noise_seed_asserts_nothing_about_equality():
    """With `seed=None` and no `noise_seed`, the noise draw consumes the
    ambient global RNG state, which is not controlled by the engine at all --
    two runs may coincidentally match (e.g. if the global RNG happens to be
    reseeded identically by test ordering) or differ. This test documents
    that no equality claim is made in that configuration; it only asserts
    the run completes and returns the expected shape.
    """
    config = _make_config(noise_std=0.5, run_seed=None)
    stim = _make_stimulus()
    engine = SimulationEngine(config)
    result = engine.run(stim, return_intermediates=True)
    assert result["pop_0"]["spikes"].shape[0] == 1


# ---------------------------------------------------------------------------
# Per-population noise seed
# ---------------------------------------------------------------------------


def test_noise_seed_makes_population_noise_reproducible_without_run_seed():
    config = _make_config(noise_std=0.5, noise_seed=3, run_seed=None)
    stim = _make_stimulus()

    engine1 = SimulationEngine(config)
    result1 = engine1.run(stim, return_intermediates=True)

    engine2 = SimulationEngine(config)
    result2 = engine2.run(stim, return_intermediates=True)

    assert torch.equal(result1["pop_0"]["filtered"], result2["pop_0"]["filtered"])
    assert torch.equal(result1["pop_0"]["spikes"], result2["pop_0"]["spikes"])


def test_different_noise_seeds_give_different_noise():
    stim = _make_stimulus()

    config3 = _make_config(noise_std=0.5, noise_seed=3, run_seed=None)
    engine3 = SimulationEngine(config3)
    result3 = engine3.run(stim, return_intermediates=True)

    config4 = _make_config(noise_std=0.5, noise_seed=4, run_seed=None)
    engine4 = SimulationEngine(config4)
    result4 = engine4.run(stim, return_intermediates=True)

    assert not torch.equal(result3["pop_0"]["filtered"], result4["pop_0"]["filtered"])


# ---------------------------------------------------------------------------
# progress_cb
# ---------------------------------------------------------------------------


def test_progress_cb_called_once_per_population_in_order():
    config = _make_config(n_populations=3)
    stim = _make_stimulus()
    engine = SimulationEngine(config)

    calls = []

    def progress_cb(index, n, name):
        calls.append((index, n, name))

    results = engine.run(stim, progress_cb=progress_cb)

    assert calls == [
        (0, 3, "pop_0"),
        (1, 3, "pop_1"),
        (2, 3, "pop_2"),
    ]
    assert set(results.keys()) == {"pop_0", "pop_1", "pop_2"}


def test_progress_cb_default_none_does_not_break_run():
    config = _make_config(n_populations=2)
    stim = _make_stimulus()
    engine = SimulationEngine(config)
    results = engine.run(stim)
    assert set(results.keys()) == {"pop_0", "pop_1"}


def test_progress_cb_called_before_that_populations_results_exist():
    """The callback fires immediately before the filter/neuron pass -- proven
    by patching `_run_pop_from_drive` (which produces a population's result)
    to record when each population is actually computed, and asserting that
    order interleaves as progress(i) before computed(i) for every i, never
    the reverse.
    """
    config = _make_config(n_populations=2)
    stim = _make_stimulus()
    engine = SimulationEngine(config)

    events = []
    original = SimulationEngine._run_pop_from_drive

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        events.append(("computed", len(events)))
        return result

    def progress_cb(index, n, name):
        events.append(("progress", index, n, name))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SimulationEngine, "_run_pop_from_drive", staticmethod(spy))
        engine.run(stim, progress_cb=progress_cb)

    kinds = [e[0] for e in events]
    assert kinds == ["progress", "computed", "progress", "computed"]
    assert events[0][1:] == (0, 2, "pop_0")
    assert events[2][1:] == (1, 2, "pop_1")


# ---------------------------------------------------------------------------
# noise_generator=None path is unchanged (pins pre-change behaviour)
# ---------------------------------------------------------------------------


def test_run_pop_from_drive_default_noise_path_unchanged():
    """With `noise_generator=None`, `_run_pop_from_drive` draws noise from
    `torch.randn_like` against the global RNG exactly as before this task --
    seeding the global RNG identically before two calls gives identical
    output.
    """
    from sensoryforge.filters.sa_ra import SAFilterTorch
    from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch

    def _build():
        filt = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=0.5)
        neuron = IzhikevichNeuronTorch(dt=0.05)
        drive = torch.ones(1, 20, 4) * 5.0
        return filt, neuron, drive

    torch.manual_seed(123)
    filt1, neuron1, drive1 = _build()
    out1 = SimulationEngine._run_pop_from_drive(
        drive=drive1,
        filter_module=filt1,
        neuron_model=neuron1,
        noise_std=0.5,
        return_intermediates=True,
        dt_ms=0.5,
        integrate_dt_ms=0.05,
    )

    torch.manual_seed(123)
    filt2, neuron2, drive2 = _build()
    out2 = SimulationEngine._run_pop_from_drive(
        drive=drive2,
        filter_module=filt2,
        neuron_model=neuron2,
        noise_std=0.5,
        return_intermediates=True,
        dt_ms=0.5,
        integrate_dt_ms=0.05,
    )

    assert torch.equal(out1["filtered"], out2["filtered"])
    assert torch.equal(out1["spikes"], out2["spikes"])


def test_run_pop_from_drive_noise_generator_overrides_default_path():
    """Passing an explicit `noise_generator` draws noise from that generator
    instead of the global RNG -- two calls with independently-seeded
    generators (same seed) give identical output regardless of global RNG
    state.
    """
    from sensoryforge.filters.sa_ra import SAFilterTorch
    from sensoryforge.neurons.izhikevich import IzhikevichNeuronTorch

    def _build():
        filt = SAFilterTorch(tau_r=5.0, tau_d=30.0, k1=0.05, k2=3.0, dt=0.5)
        neuron = IzhikevichNeuronTorch(dt=0.05)
        drive = torch.ones(1, 20, 4) * 5.0
        return filt, neuron, drive

    torch.manual_seed(1)  # different global state before each call
    filt1, neuron1, drive1 = _build()
    gen1 = torch.Generator(device="cpu").manual_seed(99)
    out1 = SimulationEngine._run_pop_from_drive(
        drive=drive1,
        filter_module=filt1,
        neuron_model=neuron1,
        noise_std=0.5,
        return_intermediates=True,
        dt_ms=0.5,
        integrate_dt_ms=0.05,
        noise_generator=gen1,
    )

    torch.manual_seed(999)  # different global state before this call
    filt2, neuron2, drive2 = _build()
    gen2 = torch.Generator(device="cpu").manual_seed(99)
    out2 = SimulationEngine._run_pop_from_drive(
        drive=drive2,
        filter_module=filt2,
        neuron_model=neuron2,
        noise_std=0.5,
        return_intermediates=True,
        dt_ms=0.5,
        integrate_dt_ms=0.05,
        noise_generator=gen2,
    )

    assert torch.equal(out1["filtered"], out2["filtered"])
    assert torch.equal(out1["spikes"], out2["spikes"])


# ---------------------------------------------------------------------------
# CLI --seed
# ---------------------------------------------------------------------------


def test_cli_run_accepts_seed_flag(tmp_path):
    config = _make_config(noise_std=0.0)
    yaml_path = tmp_path / "config.yml"
    yaml_path.write_text(config.to_yaml())

    env = _subprocess_env()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sensoryforge.cli",
            "run",
            str(yaml_path),
            "--seed",
            "5",
            "--duration",
            "20",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert result.returncode == 0, (
        f"CLI run --seed failed:\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "seed" in result.stdout.lower()


def _subprocess_env():
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env["PYTHONPATH"]]
        if env.get("PYTHONPATH")
        else [str(REPO_ROOT)]
    )
    return env
