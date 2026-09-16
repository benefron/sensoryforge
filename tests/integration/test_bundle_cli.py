"""`sensoryforge run --bundle` writes a bundle `load_bundle` can read (Wave J, J2)."""

import sys
from pathlib import Path

import torch

from sensoryforge.cli import main as cli_main
from sensoryforge.io.bundle import load_bundle

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
CANONICAL_CONFIG = EXAMPLES_DIR / "canonical_config.yml"


def test_run_bundle_matches_output_pt(tmp_path, monkeypatch):
    """--bundle and --output from the same invocation agree on spikes.

    cmd_run only runs the pipeline once, so both artefacts come from the
    same forward pass -- this mainly locks in that write_bundle's stored
    spikes are exactly engine.run()'s spikes, unchanged in transit.
    """
    output_path = tmp_path / "result.pt"
    bundle_dir = tmp_path / "bundle"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sensoryforge",
            "run",
            str(CANONICAL_CONFIG),
            "--duration",
            "20",
            "--output",
            str(output_path),
            "--bundle",
            str(bundle_dir),
        ],
    )
    assert cli_main() == 0

    assert output_path.exists()
    assert (bundle_dir / "config.json").exists()
    assert (bundle_dir / "data.h5").exists()

    saved = torch.load(output_path, weights_only=False)
    bundle = load_bundle(bundle_dir)

    populations = saved["populations"]
    assert set(populations) == set(bundle.populations)

    for pop_name in populations:
        output_spikes = saved["results"][f"{pop_name}__spikes"]
        bundle_spikes = bundle.populations[pop_name]["spikes"]
        assert torch.equal(
            output_spikes[0].cpu().to(torch.int16), bundle_spikes
        ), f"{pop_name}: --output spikes differ from --bundle spikes"


def test_run_without_bundle_flag_does_not_write_bundle(tmp_path, monkeypatch):
    output_path = tmp_path / "result.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sensoryforge",
            "run",
            str(CANONICAL_CONFIG),
            "--duration",
            "20",
            "--output",
            str(output_path),
        ],
    )
    assert cli_main() == 0
    assert not (tmp_path / "bundle").exists()
