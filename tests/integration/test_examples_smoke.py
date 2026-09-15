"""Smoke test for every shipped example config (F-043).

Before this fix, `examples/example_config.yml` and `examples/batch_config.yml` set
`sa_neurons: 100`, `ra_neurons: 196` as if they were totals; read per-row (the actual
semantics, see F-023), that is a 10,000 x 10,000 / 38,416 x 38,416 dense innervation
weight tensor, which fails the dense-weight cap in `InnervationModule`
(``ValueError``) or exhausts memory. This test runs `validate` (and `run`/
`batch --dry-run`) on every ``examples/*.yml`` file and asserts exit code 0, so a
broken example never ships silently again.
"""

import sys
from pathlib import Path

import pytest
import yaml

from sensoryforge.cli import main as cli_main

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE_CONFIGS = sorted(EXAMPLES_DIR.glob("*.yml"))


def _is_batch_config(path: Path) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return isinstance(data, dict) and "batch" in data


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_example_validates(config_path, monkeypatch):
    """`sensoryforge validate` must exit 0 for every shipped example."""
    monkeypatch.setattr(sys, "argv", ["sensoryforge", "validate", str(config_path)])
    assert cli_main() == 0


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_example_runs_or_dry_runs(config_path, tmp_path, monkeypatch):
    """Non-batch examples run for a short duration; batch examples dry-run."""
    if _is_batch_config(config_path):
        monkeypatch.setattr(
            sys,
            "argv",
            ["sensoryforge", "batch", str(config_path), "--dry-run"],
        )
    else:
        output_path = tmp_path / f"{config_path.stem}_output.pt"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "sensoryforge",
                "run",
                str(config_path),
                "--duration",
                "20",
                "--output",
                str(output_path),
            ],
        )
    assert cli_main() == 0
