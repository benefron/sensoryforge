"""`sensoryforge dataset build` and `sensoryforge world validate|sample`."""

import json
from pathlib import Path

from sensoryforge.cli import cmd_dataset, cmd_world, create_parser
from sensoryforge.world.dataset import load_dataset

WORLDS = Path(__file__).resolve().parents[1] / "fixtures" / "worlds"


def _cli(argv):
    args = create_parser().parse_args(argv)
    return {"dataset": cmd_dataset, "world": cmd_world}[args.command](args)


def test_dataset_build_writes_the_manifest(tmp_path, capsys):
    spec = WORLDS / "dataset_small.yml"
    assert _cli(["dataset", "build", str(spec), "--out", str(tmp_path / "ds")]) == 0
    assert (tmp_path / "ds" / "manifest.jsonl").exists()
    assert load_dataset(spec).dataset_id in capsys.readouterr().out


def test_world_validate_and_sample(capsys):
    world = str(WORLDS / "tactile_small.yml")
    assert _cli(["world", "validate", world]) == 0
    out = capsys.readouterr().out
    assert out.startswith("w-") and "class dots" in out and "held out gratings" in out
    assert _cli(["world", "sample", world, "-n", "3", "--seed", "5"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["index"] == 0


def test_a_bad_world_is_a_message_not_a_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.yml"
    bad.write_text("world: {classes: {}}\n")
    assert _cli(["world", "validate", str(bad)]) == 1
    assert "declare at least one class" in capsys.readouterr().err
