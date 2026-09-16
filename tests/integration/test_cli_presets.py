"""CLI integration tests for presets (Phase 2, Wave K, K4).

`sensoryforge list-presets` and `sensoryforge run --preset NAME [config.yml]`.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout

import pytest
import yaml

from sensoryforge.cli import main as cli_main


def test_list_presets_prints_both_names(capsys):
    sys.argv = ["sensoryforge", "list-presets"]
    exit_code = cli_main()
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "tactile_sa1_ra1" in out
    assert "tactile_stochastic_control" in out


def test_run_preset_with_no_config_file(capsys):
    sys.argv = [
        "sensoryforge",
        "run",
        "--preset",
        "tactile_sa1_ra1",
        "--duration",
        "5",
    ]
    exit_code = cli_main()
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "SA Population spikes" in out
    assert "RA Population spikes" in out


def test_run_preset_with_config_override(tmp_path, capsys):
    override = {
        "populations": [
            {"name": "SA Population", "noise_std": 0.0},
            {"name": "RA Population", "noise_std": 0.0},
        ]
    }
    config_file = tmp_path / "override.yml"
    with open(config_file, "w") as f:
        yaml.dump(override, f)

    sys.argv = [
        "sensoryforge",
        "run",
        "--preset",
        "tactile_sa1_ra1",
        str(config_file),
        "--duration",
        "5",
    ]
    exit_code = cli_main()
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Simulation completed successfully" in out


def test_run_with_no_config_and_no_preset_errors(capsys):
    sys.argv = ["sensoryforge", "run"]
    exit_code = cli_main()
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "no config file and no --preset" in err


def test_run_preset_argument_parses_before_positional_config():
    """--preset makes the positional config argument optional (K4)."""
    from sensoryforge.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(["run", "--preset", "tactile_sa1_ra1"])
    assert args.preset == "tactile_sa1_ra1"
    assert args.config is None
