"""CLI integration tests for task E7 (F-039, F-040, F-024): --duration and
dt_ms must reach every stimulus type through the CLI's canonical path.

Before this fix, a canonical config's stimulus generators read the legacy
temporal.dt default (0.1 ms) regardless of simulation.dt_ms (F-039), and
--duration was explicitly excluded for the default "trapezoidal" stimulus
type (F-040) -- so `sensoryforge run config.yml --duration 100` on a
dt_ms: 1.0 config produced far more than 100 record bins.
"""

import sys

import torch
import yaml

from sensoryforge.cli import main as cli_main


def _canonical_config(stimuli: list | None = None) -> dict:
    config = {
        "grids": [
            {"name": "g", "rows": 4, "cols": 4, "spacing": 1.0, "arrangement": "grid"}
        ],
        "populations": [
            {
                "name": "SA Pop",
                "target_grid": "g",
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
        "simulation": {"dt_ms": 1.0, "device": "cpu"},
    }
    if stimuli is not None:
        config["stimuli"] = stimuli
    return config


def _run_cli(tmp_path, config: dict, duration: int):
    config_file = tmp_path / "config.yml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    output_file = tmp_path / "output.pt"

    sys.argv = [
        "sensoryforge",
        "run",
        str(config_file),
        "--duration",
        str(duration),
        "--output",
        str(output_file),
    ]
    exit_code = cli_main()
    assert exit_code == 0
    return torch.load(output_file, weights_only=False)


def test_gaussian_stimulus_gets_requested_bin_count(tmp_path):
    config = _canonical_config(
        stimuli=[{"type": "gaussian", "amplitude": 10.0, "sigma": 1.0}]
    )
    results = _run_cli(tmp_path, config, duration=100)
    spikes = results["results"]["SA Pop__spikes"]
    assert spikes.shape[1] == 100


def test_trapezoidal_stimulus_gets_requested_bin_count(tmp_path):
    """No 'stimuli' key -> defaults to type 'trapezoidal' (cli.py cmd_run)."""
    config = _canonical_config(stimuli=None)
    results = _run_cli(tmp_path, config, duration=100)
    spikes = results["results"]["SA Pop__spikes"]
    assert spikes.shape[1] == 100
