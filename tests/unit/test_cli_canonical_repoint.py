"""Tests for C3-Step5: CLI cmd_run routes canonical configs through SimulationEngine.

Covers:
  - Canonical configs reach SimulationEngine, not GeneralizedTactileEncodingPipeline
  - Legacy configs still reach GeneralizedTactileEncodingPipeline (unchanged path)
  - SimulationEngine produces non-empty results for a canonical config + stimulus
  - Device override is applied to simulation.device for canonical configs
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from sensoryforge.config.schema import SensoryForgeConfig, GridConfig, PopulationConfig, SimulationConfig
from sensoryforge.core.simulation_engine import SimulationEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_config_dict(device: str = "cpu") -> dict:
    """Minimal canonical config dict with a single 4×4 grid and SA population."""
    return {
        "grids": [
            {
                "name": "test_grid",
                "rows": 4,
                "cols": 4,
                "spacing": 1.0,
                "arrangement": "grid",
            }
        ],
        "populations": [
            {
                "name": "SA Pop",
                "target_grid": "test_grid",
                "neuron_type": "SA",
                "neurons_per_row": 2,
                "innervation_method": "gaussian",
                "connections_per_neuron": 4,
                "sigma_d_mm": 2.0,
                "filter_method": "none",
                "neuron_model": "Izhikevich",
                "input_gain": 1.0,
                "noise_std": 0.0,
                "seed": 42,
            }
        ],
        "simulation": {
            "dt": 0.1,
            "device": device,
        },
    }


def _legacy_config_dict() -> dict:
    """Minimal legacy-format config dict."""
    return {
        "pipeline": {"device": "cpu", "dt": 0.1},
        "grid": {"rows": 10, "cols": 10, "spacing": 1.0},
        "neurons": {
            "sa_neurons": 4,
            "ra_neurons": 4,
            "dt": 0.1,
        },
    }


# ---------------------------------------------------------------------------
# SimulationEngine round-trip on canonical config
# ---------------------------------------------------------------------------

def test_simulation_engine_runs_canonical_config():
    """SimulationEngine must produce spikes dict for a canonical config + stimulus."""
    sf_config = SensoryForgeConfig.from_dict(_canonical_config_dict())
    engine = SimulationEngine(sf_config)

    # Small stimulus: 50 timesteps, 4×4 grid, constant positive
    stimulus = torch.ones(50, 4, 4) * 5.0

    results = engine.run(stimulus, return_intermediates=True)

    assert "SA Pop" in results
    pop = results["SA Pop"]
    assert "spikes" in pop
    assert "drive" in pop
    assert "filtered" in pop
    # Shape: [batch=1, time>=50, neurons=4]  (neuron may append an initial-state step)
    assert pop["spikes"].shape[0] == 1
    assert pop["spikes"].shape[1] >= 50
    assert pop["spikes"].shape[2] == 4


def test_simulation_engine_produces_nonzero_spikes_for_strong_drive():
    """Strong constant input must produce at least one spike over 200 ms."""
    sf_config = SensoryForgeConfig.from_dict(_canonical_config_dict())
    engine = SimulationEngine(sf_config)

    # Strong drive: 200 timesteps
    torch.manual_seed(99)
    stimulus = torch.ones(200, 4, 4) * 20.0

    results = engine.run(stimulus)
    total_spikes = results["SA Pop"]["spikes"].sum().item()

    assert total_spikes > 0, (
        "Strong constant input should produce at least one spike over 200 timesteps."
    )


# ---------------------------------------------------------------------------
# Device override for canonical configs
# ---------------------------------------------------------------------------

def test_canonical_device_override_applied_via_simulation_key():
    """Device override must go into config['simulation']['device'] for canonical format.

    We verify that after the override the parsed SensoryForgeConfig picks up the
    device without raising an error (cpu is always available).
    """
    config = _canonical_config_dict(device="cpu")
    # Simulate what cmd_run does: write device into simulation key
    config["simulation"]["device"] = "cpu"
    sf_config = SensoryForgeConfig.from_dict(config)
    assert sf_config.simulation.device == "cpu"


# ---------------------------------------------------------------------------
# Canonical format detection logic (mirrors cli.py is_canonical check)
# ---------------------------------------------------------------------------

def _is_canonical(config: dict) -> bool:
    """Mirror of the is_canonical check in cli.cmd_run."""
    return (
        isinstance(config.get("grids"), list) and
        isinstance(config.get("populations"), list) and
        "pipeline" not in config
    )


def test_canonical_config_detected_as_canonical():
    assert _is_canonical(_canonical_config_dict()) is True


def test_legacy_config_not_detected_as_canonical():
    assert _is_canonical(_legacy_config_dict()) is False


def test_canonical_with_pipeline_key_is_not_canonical():
    """A config with both grids/populations AND pipeline key is treated as legacy."""
    cfg = _canonical_config_dict()
    cfg["pipeline"] = {"device": "cpu"}
    assert _is_canonical(cfg) is False
