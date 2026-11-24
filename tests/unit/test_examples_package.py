"""The example configs that ship in the package, and ``sensoryforge examples`` (D4).

Each shipped example is copied out and run the way its own header says; the
slow test does the same from an unpacked wheel in a folder outside the
repository, because a user installs the package, not a clone.
"""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from sensoryforge.cli import main as cli_main
from sensoryforge.examples import (
    copy_example,
    example_description,
    example_text,
    list_examples,
)

REPO = Path(__file__).resolve().parents[2]


def _cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["sensoryforge", *argv])
    return cli_main()


def test_the_examples_ship_and_each_has_a_description():
    assert list_examples() == ["amplitude_sweep", "moving_edge"]
    for name in list_examples():
        assert example_description(name)
        assert yaml.safe_load(example_text(name))


def test_copy_writes_the_file_and_refuses_to_overwrite(tmp_path):
    path = copy_example("moving_edge", tmp_path / "new")
    assert path == tmp_path / "new" / "moving_edge.yml"
    assert path.read_text(encoding="utf-8") == example_text("moving_edge")
    with pytest.raises(FileExistsError):
        copy_example("moving_edge", tmp_path / "new")
    path.write_text("edited", encoding="utf-8")
    copy_example("moving_edge", tmp_path / "new", force=True)
    assert path.read_text(encoding="utf-8") == example_text("moving_edge")
    with pytest.raises(ValueError, match="amplitude_sweep"):
        copy_example("no_such_example", tmp_path)


def test_the_cli_lists_and_copies(tmp_path, monkeypatch, capsys):
    assert _cli(monkeypatch, "examples") == 0
    assert "amplitude_sweep" in capsys.readouterr().out
    assert _cli(monkeypatch, "examples", "list") == 0
    assert (
        _cli(monkeypatch, "examples", "copy", "moving_edge", "--to", str(tmp_path)) == 0
    )
    assert (tmp_path / "moving_edge.yml").exists()
    assert (
        _cli(monkeypatch, "examples", "copy", "moving_edge", "--to", str(tmp_path)) == 1
    )
    assert "--force" in capsys.readouterr().err
    assert _cli(monkeypatch, "examples", "copy", "nope", "--to", str(tmp_path)) == 1


def test_the_stimulus_block_runs_over_a_preset(tmp_path, monkeypatch):
    path = copy_example("moving_edge", tmp_path)
    out = tmp_path / "result.pt"
    code = _cli(
        monkeypatch,
        "run", "--preset", "tactile_sa1_ra1", str(path),
        "--duration", "20", "--output", str(out),
    )  # fmt: skip
    assert code == 0 and out.exists()


def test_the_sweep_validates_and_runs(tmp_path, monkeypatch):
    path = copy_example("amplitude_sweep", tmp_path)
    assert _cli(monkeypatch, "batch", str(path), "--dry-run") == 0
    # A shorter, smaller copy of the same sweep, run for real.
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["base_config"]["grids"][0].update(rows=12, cols=12)
    for population in config["base_config"]["populations"]:
        population["neurons_per_row"] = 3
    config["base_config"]["simulation"]["duration_ms"] = 20.0
    config["batch"]["stimuli"][0]["parameters"]["amplitude"] = [10.0]
    small = tmp_path / "small.yml"
    small.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert _cli(monkeypatch, "batch", str(small), "--output", str(tmp_path / "o")) == 0
    assert any((tmp_path / "o").rglob("*.pt"))


@pytest.mark.slow
def test_the_examples_run_from_an_unpacked_wheel(tmp_path):
    """Build the wheel, unpack it, and copy and check each example outside the repo."""
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    src = tmp_path / "src"
    shutil.copytree(
        REPO / "sensoryforge",
        src / "sensoryforge",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(REPO / name, src / name)
    built = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-q",
         "--no-build-isolation", "-w", str(tmp_path / "dist")],
        cwd=src, capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    assert built.returncode == 0, built.stderr[-1500:]
    unpacked = tmp_path / "site"
    with zipfile.ZipFile(next((tmp_path / "dist").glob("*.whl"))) as wheel:
        wheel.extractall(unpacked)
    work = tmp_path / "work"
    work.mkdir()
    env = dict(os.environ, PYTHONPATH=str(unpacked), OMP_NUM_THREADS="1")

    def run(*argv):
        done = subprocess.run(
            [sys.executable, "-m", "sensoryforge.cli", *argv],
            cwd=work, env=env, capture_output=True, text=True, timeout=600,
        )  # fmt: skip
        assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
        return done.stdout

    assert str(unpacked) in run_python(env, work)
    for name in list_examples():
        run("examples", "copy", name)
    run("batch", "amplitude_sweep.yml", "--dry-run")
    run("run", "--preset", "tactile_sa1_ra1", "moving_edge.yml", "--duration", "20")


def run_python(env, cwd):
    """Where the subprocess imports sensoryforge from (the wheel, not the clone)."""
    return subprocess.run(
        [sys.executable, "-c", "import sensoryforge; print(sensoryforge.__file__)"],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=120,
    ).stdout  # fmt: skip
