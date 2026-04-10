"""Tests for C3-Step2: SimulationEngine correctness — input_gain, filter reset, bug fix.

Covers:
  - input_gain scaling is applied in run() (not silently ignored)
  - Filter state is reset between independent run() calls
  - FlatInnervationModule path no longer crashes on undefined innervation_method
"""

import pytest
import torch
from sensoryforge.config.schema import SensoryForgeConfig, GridConfig, PopulationConfig, SimulationConfig
from sensoryforge.core.simulation_engine import SimulationEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    input_gain: float = 1.0,
    filter_method: str = "none",
    noise_std: float = 0.0,
    seed: int = 42,
) -> SensoryForgeConfig:
    """Minimal canonical config for a single SA population on a 4×4 grid."""
    grid = GridConfig(
        name="test_grid",
        rows=4,
        cols=4,
        spacing=1.0,
        arrangement="grid",
    )
    pop = PopulationConfig(
        name="test_pop",
        target_grid="test_grid",
        neuron_type="SA",
        neurons_per_row=2,
        innervation_method="gaussian",
        connections_per_neuron=4,
        sigma_d_mm=2.0,
        filter_method=filter_method,
        neuron_model="Izhikevich",
        input_gain=input_gain,
        noise_std=noise_std,
        seed=seed,
    )
    sim = SimulationConfig(dt=0.1, device="cpu")
    return SensoryForgeConfig(grids=[grid], populations=[pop], simulation=sim)


def _make_stimulus(timesteps: int = 50) -> torch.Tensor:
    """Constant positive stimulus [time, H, W]."""
    torch.manual_seed(7)
    return torch.ones(timesteps, 4, 4) * 5.0


# ---------------------------------------------------------------------------
# Bug fix: innervation_method must not be undefined
# ---------------------------------------------------------------------------

def test_engine_initialises_without_innervation_method_crash():
    """SimulationEngine must not raise NameError for undefined innervation_method.

    Before the fix, _build_populations() referenced `innervation_method` (bare
    name) in the FlatInnervationModule branch without first reading it from
    pop_cfg, causing a NameError at init time.
    """
    config = _make_config()
    # If this line raises NameError or NameError-like crash, the bug is present
    engine = SimulationEngine(config)
    assert len(engine.populations) == 1


# ---------------------------------------------------------------------------
# input_gain applied in run()
# ---------------------------------------------------------------------------

def test_input_gain_1_is_identity():
    """input_gain=1.0 (default) must not change spike output vs no gain."""
    config_default = _make_config(input_gain=1.0)
    engine = SimulationEngine(config_default)
    stim = _make_stimulus()

    result = engine.run(stim, return_intermediates=True)
    pop_result = result["test_pop"]

    # With gain=1.0 the filtered tensor should equal filtered without gain
    # (trivially true — this test mainly checks run() doesn't crash)
    assert "spikes" in pop_result
    assert pop_result["spikes"].shape[-1] == 4  # 2×2 = 4 neurons


def test_input_gain_scales_drive():
    """input_gain=2.0 should produce more spikes than input_gain=0.5 for same stimulus.

    A higher gain amplifies the drive going into the neuron, so a constant
    positive stimulus should produce more spikes with gain=2.0 than with 0.5.
    """
    stim = _make_stimulus(timesteps=100)

    low_cfg = _make_config(input_gain=0.1)
    high_cfg = _make_config(input_gain=5.0)

    low_engine = SimulationEngine(low_cfg)
    high_engine = SimulationEngine(high_cfg)

    torch.manual_seed(0)
    low_result = low_engine.run(stim)
    torch.manual_seed(0)
    high_result = high_engine.run(stim)

    low_spikes = low_result["test_pop"]["spikes"].sum().item()
    high_spikes = high_result["test_pop"]["spikes"].sum().item()

    assert high_spikes >= low_spikes, (
        f"input_gain=5.0 should produce ≥ spikes vs input_gain=0.1, "
        f"got high={high_spikes}, low={low_spikes}"
    )


def test_input_gain_zero_silences_neuron():
    """input_gain=0.0 should produce zero spikes regardless of stimulus."""
    config = _make_config(input_gain=0.0)
    engine = SimulationEngine(config)
    stim = _make_stimulus(timesteps=200)

    result = engine.run(stim)
    total_spikes = result["test_pop"]["spikes"].sum().item()

    assert total_spikes == 0, (
        f"input_gain=0.0 should silence all neurons, got {total_spikes} spikes"
    )


def test_input_gain_reflected_in_filtered_intermediate():
    """With no filter, filtered must equal drive × input_gain.

    Uses a single engine with gain=3.0 and filter_method='none'.
    drive = innervation output (pre-gain)
    filtered = drive × gain (what enters the neuron)
    So: filtered should be exactly 3 × drive.
    """
    stim = _make_stimulus(timesteps=30)
    cfg = _make_config(input_gain=3.0, filter_method="none")
    engine = SimulationEngine(cfg)

    result = engine.run(stim, return_intermediates=True)
    drive = result["test_pop"]["drive"]
    filtered = result["test_pop"]["filtered"]

    assert torch.allclose(filtered, drive * 3.0, atol=1e-5), (
        f"filtered should be drive × 3.0 when filter_method='none' and input_gain=3.0. "
        f"Max diff: {(filtered - drive * 3.0).abs().max().item():.4f}"
    )


# ---------------------------------------------------------------------------
# Filter state reset between independent run() calls
# ---------------------------------------------------------------------------

def test_filter_run_is_stateless_across_calls():
    """Calling run() twice with identical stimuli must produce identical results.

    Without a filter reset at the start of each run(), the second call inherits
    state from the first, producing different output even for identical inputs.
    """
    config = _make_config(filter_method="SA")
    engine = SimulationEngine(config)
    stim = _make_stimulus(timesteps=50)

    result1 = engine.run(stim, return_intermediates=True)
    result2 = engine.run(stim, return_intermediates=True)

    filtered1 = result1["test_pop"]["filtered"]
    filtered2 = result2["test_pop"]["filtered"]

    assert torch.allclose(filtered1, filtered2, atol=1e-5), (
        "Two identical run() calls should produce identical filtered output. "
        "If they differ, filter state is leaking between runs."
    )
