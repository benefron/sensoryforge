"""Tests for :mod:`sensoryforge.gui.execution.sweep_controller`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gui

from sensoryforge.config.schema import SensoryForgeConfig  # noqa: E402
from sensoryforge.gui.execution.sweep_controller import (  # noqa: E402
    SweepController,
    SweepSpec,
    sweep_command,
    sweep_paths,
    write_sweep,
    write_slurm_script,
)

PRESET = "sensoryforge/presets/tactile_sa1_ra1.yml"


def _config() -> SensoryForgeConfig:
    config = SensoryForgeConfig.from_yaml_file(PRESET)
    config.grids[0].rows = 10
    config.grids[0].cols = 10
    return config


# ---------------------------------------------------------------- the paths


def test_sweep_paths_covers_schema_fields_and_registry_parameters():
    paths = dict(sweep_paths(_config()))

    assert "grids.0.spacing" in paths
    assert paths["grids.0.spacing"] is None
    assert "populations.0.input_gain" in paths
    assert "stimulus.amplitude" in paths
    assert "simulation.dt_ms" in paths
    assert "simulation.duration_ms" in paths

    # From FILTER_REGISTRY.get_param_spec("sa") -- the population's filter,
    # with its ParamSpec so a UI can show the unit and the range.
    spec = paths["populations.0.filter_params.tau_r"]
    assert spec is not None and spec.name == "tau_r"
    # From NEURON_REGISTRY / INNERVATION_REGISTRY for the same population.
    assert "populations.0.model_params.a" in paths
    assert "populations.0.innervation_params.k" in paths


def test_sweep_paths_skips_non_numeric_and_unknown_components():
    config = _config()
    config.populations[0].filter_method = "no_such_filter_plugin"

    paths = dict(sweep_paths(config))

    assert "populations.0.filter_params.tau_r" not in paths
    # Booleans and strings are not sweep axes.
    assert "grids.0.visible" not in paths
    assert "populations.0.neuron_model" not in paths
    # The other population is unaffected.
    assert "populations.1.filter_params.tau_RA" in paths


# --------------------------------------------------------------- write_sweep


def test_write_sweep_writes_one_config_per_combination(tmp_path):
    spec = SweepSpec(
        fields=[
            ("populations.0.filter_params.tau_r", [3.0, 7.0]),
            ("populations.0.input_gain", [10.0, 20.0]),
        ]
    )
    assert spec.n_combinations == 4

    manifest = write_sweep(_config(), spec, root=tmp_path / "sweep", duration_ms=50.0)

    assert len(manifest.combos) == 4
    assert [c["dir"] for c in manifest.combos] == [
        "combo_000",
        "combo_001",
        "combo_002",
        "combo_003",
    ]
    expected = [(3.0, 10.0), (3.0, 20.0), (7.0, 10.0), (7.0, 20.0)]
    for index, (tau_r, gain) in enumerate(expected):
        written = SensoryForgeConfig.from_yaml_file(manifest.config_path(index))
        assert written.populations[0].filter_params["tau_r"] == tau_r
        assert written.populations[0].input_gain == gain
        assert written.simulation.duration_ms == 50.0
        assert manifest.combos[index]["values"] == {
            "populations.0.filter_params.tau_r": tau_r,
            "populations.0.input_gain": gain,
        }

    payload = json.loads((manifest.root / "manifest.json").read_text())
    assert payload["duration_ms"] == 50.0
    assert payload["fields"] == [
        "populations.0.filter_params.tau_r",
        "populations.0.input_gain",
    ]
    assert len(payload["combos"]) == 4


def test_write_sweep_leaves_the_base_config_untouched(tmp_path):
    config = _config()
    spec = SweepSpec(fields=[("populations.0.input_gain", [1.0, 2.0])])

    write_sweep(config, spec, root=tmp_path / "sweep", duration_ms=10.0)

    assert config.populations[0].input_gain == 50.0
    assert config.simulation.duration_ms is None


def test_write_sweep_rejects_a_path_that_does_not_resolve(tmp_path):
    spec = SweepSpec(fields=[("populations.9.input_gain", [1.0])])

    with pytest.raises(ValueError, match="out of range"):
        write_sweep(_config(), spec, root=tmp_path / "sweep", duration_ms=10.0)
    assert not (tmp_path / "sweep").exists()


@pytest.mark.parametrize(
    "fields", [[], [("populations.0.input_gain", [])]], ids=["no fields", "no values"]
)
def test_an_empty_sweep_spec_raises(fields):
    with pytest.raises(ValueError):
        SweepSpec(fields=fields)


# --------------------------------------------------------------------- SLURM


def test_write_slurm_script_is_an_array_job_over_the_combinations(tmp_path):
    spec = SweepSpec(
        fields=[
            ("populations.0.input_gain", [10.0, 20.0]),
            ("simulation.dt_ms", [0.5, 1.0]),
        ]
    )
    manifest = write_sweep(_config(), spec, root=tmp_path / "sweep", duration_ms=50.0)

    script_path = write_slurm_script(
        manifest,
        settings={
            "job_name": "sweepy",
            "partition": "short",
            "time": "00:30:00",
            "mem_gb": 8,
            "cpus_per_task": 2,
            "gpus": 0,
            "conda_env": "sensoryforge",
            "duration_ms": 50.0,
        },
    )
    text = script_path.read_text()

    assert script_path == manifest.root / "run_sweep.sh"
    assert "#SBATCH --array=0-3" in text
    assert "#SBATCH --job-name=sweepy" in text
    assert "#SBATCH --partition=short" in text
    assert "#SBATCH --cpus-per-task=2" in text
    assert "--gres=gpu" not in text  # gpus=0
    assert "conda activate sensoryforge" in text
    assert 'COMBO=$(printf "combo_%03d" "$SLURM_ARRAY_TASK_ID")' in text
    assert '"$SWEEP_ROOT/$COMBO/config.yml"' in text
    assert "--duration 50.0" in text
    assert '--bundle "$SWEEP_ROOT/$COMBO/bundle"' in text


# ---------------------------------------------------------------- subprocess


def test_sweep_command_uses_this_interpreter_and_the_module_entry_point(tmp_path):
    spec = SweepSpec(fields=[("populations.0.input_gain", [10.0])])
    manifest = write_sweep(_config(), spec, root=tmp_path / "sweep", duration_ms=5.0)

    argv = sweep_command(manifest, 0, 5.0)

    # Never a bare `sensoryforge`: that may resolve to a different checkout
    # or environment (F-053).
    assert argv[:4] == [sys.executable, "-m", "sensoryforge.cli", "run"]
    assert argv[4] == str(manifest.config_path(0))
    assert argv[5:7] == ["--duration", "5.0"]
    assert argv[7:9] == ["--bundle", str(manifest.bundle_path(0))]


def _can_run_subprocess() -> bool:
    """Whether a child interpreter can import the package under test."""
    import subprocess

    import sensoryforge

    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(Path(sensoryforge.__file__).resolve().parent.parent)
    completed = subprocess.run(
        [sys.executable, "-c", "import sensoryforge"],
        env=env,
        capture_output=True,
    )
    return completed.returncode == 0


@pytest.mark.slow
def test_sweep_controller_runs_every_combination_and_writes_bundles(qtbot, tmp_path):
    if not _can_run_subprocess():
        pytest.skip("a child interpreter cannot import sensoryforge here")

    spec = SweepSpec(fields=[("populations.0.input_gain", [10.0, 20.0])])
    manifest = write_sweep(_config(), spec, root=tmp_path / "sweep", duration_ms=5.0)

    controller = SweepController()
    lines: list[str] = []
    progressed: list[tuple] = []
    controller.log.connect(lines.append)
    controller.progress.connect(lambda done, total: progressed.append((done, total)))

    with qtbot.waitSignal(controller.finished, timeout=180000) as blocker:
        controller.start(manifest, duration_ms=5.0, parallel=2)

    assert blocker.args[0] == 0, "\n".join(lines)
    assert progressed[-1] == (2, 2)
    assert not controller.running
    for index in range(2):
        assert (manifest.bundle_path(index) / "config.json").is_file()
    assert any("Bundle written to" in line for line in lines)


def test_a_combination_that_cannot_start_still_finishes_the_sweep(
    qtbot, tmp_path, monkeypatch
):
    """A process that never starts must not leave the sweep waiting forever.

    ``QProcess`` does not reliably emit ``finished`` after ``FailedToStart``,
    so the controller has to reap the combination on ``errorOccurred`` itself.
    Without that, ``finished`` is never emitted and the Batch screen hangs.
    """
    import sensoryforge.gui.execution.sweep_controller as module

    spec = SweepSpec(fields=[("populations.0.input_gain", [10.0, 20.0])])
    manifest = write_sweep(_config(), spec, root=tmp_path / "sweep", duration_ms=5.0)
    monkeypatch.setattr(
        module,
        "sweep_command",
        lambda manifest, index, duration_ms: [
            str(tmp_path / "definitely-not-an-executable"),
            "run",
        ],
    )

    controller = SweepController()
    failures: list[str] = []
    controller.failed.connect(failures.append)

    with qtbot.waitSignal(controller.finished, timeout=30000) as blocker:
        controller.start(manifest, duration_ms=5.0, parallel=2)

    assert blocker.args[0] == 2, "both combinations must be counted as failed"
    assert len(failures) == 2
    assert not controller.running
