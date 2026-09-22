"""The GUI-vs-engine golden guard (GUI v2 task 0.4).

The whole premise of the GUI v2 rewrite is that the window runs the *same*
model as ``sensoryforge run``. The old Spiking Neurons tab did not (it had its
own stimulus renderer and always flat-reshaped the stimulus onto receptors, so
any arrangement but a resolution-matched regular grid diverged from the CLI).
This module pins the new path down:

* (a) the config run through :class:`~sensoryforge.gui.execution.run_controller.RunController`
* (b) the same config run through :meth:`SimulationEngine.run` directly
* (c) the same config written as YAML and run by ``sensoryforge run`` in a
  subprocess, compared through the bundles both wrote

(a) and (b) are computed in one process, so they are compared with
``torch.equal``: F-071's float32-step tolerance is for comparing against a
fixture recorded on another platform, not for two results from the same
process. (c) crosses a process boundary but not a platform one, so it is
compared with :func:`sensoryforge.testing.golden.assert_matches_golden`.

The comparisons cover ``noise_std > 0`` with a run seed and a non-grid (hex)
arrangement, the two cases the old tab got wrong.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gui

import sensoryforge  # noqa: E402
from sensoryforge.config.schema import SensoryForgeConfig  # noqa: E402
from sensoryforge.core.simulation_engine import SimulationEngine  # noqa: E402
from sensoryforge.gui.execution.render import render_for_config  # noqa: E402
from sensoryforge.gui.execution.run_controller import RunController  # noqa: E402
from sensoryforge.gui.project import ProjectHandle  # noqa: E402
from sensoryforge.gui.session import Session  # noqa: E402
from sensoryforge.io.bundle import load_bundle  # noqa: E402
from sensoryforge.testing.golden import assert_matches_golden  # noqa: E402

PRESET = Path("sensoryforge/presets/tactile_sa1_ra1.yml")

#: A small but structurally complete run: 20x20 receptors at 0.15 mm,
#: template receptive fields at d = 0.40 mm, 30 ms at 1 ms.
GRID_SIZE = 20
DURATION_MS = 30.0
SEED = 1
TIMEOUT_MS = 60000


def _config(*, arrangement: str = "grid", noise_std: float = 0.0) -> SensoryForgeConfig:
    """The ``tactile_sa1_ra1`` preset shrunk to something a test can run.

    Args:
        arrangement: The receptor grid arrangement. ``"hex"`` is the case the
            old GUI path got wrong.
        noise_std: Membrane noise standard deviation, in mA.

    Returns:
        A CPU config with ``simulation.seed`` set, so every run of it is
        reproducible.
    """
    config = SensoryForgeConfig.from_yaml_file(PRESET)
    grid = config.grids[0]
    grid.rows = GRID_SIZE
    grid.cols = GRID_SIZE
    grid.arrangement = arrangement
    if arrangement != "grid":
        grid.seed = 7
    config.simulation.device = "cpu"
    config.simulation.seed = SEED
    config.simulation.duration_ms = DURATION_MS
    for population in config.populations:
        population.resolvable_distance_mm = 0.40
        population.noise_std = noise_std
    return config


def _run_direct(config: SensoryForgeConfig) -> dict:
    """Run ``config`` through :class:`SimulationEngine` with no GUI involved."""
    snapshot = copy.deepcopy(config)
    rendered = render_for_config(snapshot, duration_ms=DURATION_MS)
    engine = SimulationEngine(snapshot, device=torch.device(snapshot.simulation.device))
    return engine.run(
        rendered.stimulus,
        return_intermediates=True,
        seed=snapshot.simulation.seed,
    )


def _run_controller(qtbot, config: SensoryForgeConfig, *, project_root=None):
    """Run ``config`` through :class:`RunController`, returning its ``RunResult``."""
    session = Session(copy.deepcopy(config))
    if project_root is not None:
        session.set_project(ProjectHandle.create(project_root, session.config))
    controller = RunController(session)
    with qtbot.waitSignal(controller.finished, timeout=TIMEOUT_MS) as blocker:
        controller.run(duration_ms=DURATION_MS, bundle=project_root is not None)
    return blocker.args[0]


def _assert_identical(left: dict, right: dict, *, what: str) -> None:
    """Every tensor of every population is bit-identical (same process)."""
    assert set(left) == set(right), what
    for population, tensors in left.items():
        assert set(tensors) == set(right[population]), f"{what}: {population}"
        for key, value in tensors.items():
            other = right[population][key]
            assert torch.equal(value, other), (
                f"{what}: {population}/{key} differs "
                f"(max |delta| = {(value - other).abs().max().item():g})"
            )


# ------------------------------------------------------- (a) == (b), in-process


@pytest.mark.parametrize(
    "arrangement,noise_std",
    [
        ("grid", 0.0),
        ("grid", 0.5),
        ("hex", 0.5),
    ],
    ids=["preset", "preset+noise", "hex+noise"],
)
def test_run_controller_equals_the_engine(qtbot, arrangement, noise_std):
    """Two controller runs and one direct engine run are all identical.

    Two controller runs, not one: a run whose noise is drawn from the ambient
    RNG would agree with the engine by luck the first time and diverge on the
    second. ``simulation.seed`` has to make the whole run reproducible.
    """
    config = _config(arrangement=arrangement, noise_std=noise_std)

    first = _run_controller(qtbot, config)
    second = _run_controller(qtbot, config)
    direct = _run_direct(config)

    _assert_identical(first.results, direct, what="controller vs engine")
    _assert_identical(second.results, direct, what="controller (2nd) vs engine")

    if noise_std > 0:
        # The noise really is in there: without it the traces would be
        # identical to the noiseless run, and this guard would be vacuous.
        noiseless = _run_direct(_config(arrangement=arrangement, noise_std=0.0))
        population = next(iter(direct))
        assert not torch.equal(
            direct[population]["filtered"], noiseless[population]["filtered"]
        )


def test_hex_arrangement_really_is_irregular():
    """Guard the guard: a hex config must not silently fall back to a lattice.

    If ``arrangement="hex"`` produced the same receptor coordinates as
    ``"grid"``, the hex case above would be testing nothing.
    """
    hex_engine = SimulationEngine(_config(arrangement="hex"))
    grid_engine = SimulationEngine(_config(arrangement="grid"))

    hex_coords = hex_engine.populations[0]["inputs"][0]["receptor_coords"]
    grid_coords = grid_engine.populations[0]["inputs"][0]["receptor_coords"]

    assert hex_coords.shape != grid_coords.shape or not torch.equal(
        hex_coords, grid_coords
    )


# ------------------------------------------------- (a) == (c), across processes


def _subprocess_env() -> dict:
    """The parent environment with this checkout pinned on ``PYTHONPATH``."""
    env = dict(os.environ)
    package_parent = str(Path(sensoryforge.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [package_parent] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


@pytest.mark.slow
@pytest.mark.parametrize("noise_std", [0.0, 0.5], ids=["no-noise", "seeded-noise"])
def test_run_controller_bundle_matches_the_cli_bundle(qtbot, tmp_path, noise_std):
    """A controller run and ``sensoryforge run`` write the same numbers.

    With membrane noise too: the run seed and each population's noise seed
    travel in the YAML, so the CLI draws the same noise the GUI did.
    """
    config = _config(noise_std=noise_std)

    result = _run_controller(qtbot, config, project_root=tmp_path / "project")
    assert result.bundle_dir is not None

    payload = config.to_dict()
    config_path = tmp_path / "cli_config.yml"
    import yaml

    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    cli_bundle = tmp_path / "cli_bundle"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sensoryforge.cli",
            "run",
            str(config_path),
            "--duration",
            str(DURATION_MS),
            "--bundle",
            str(cli_bundle),
        ],
        cwd=str(tmp_path),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    gui = load_bundle(result.bundle_dir)
    cli = load_bundle(cli_bundle)
    assert set(gui.populations) == set(cli.populations)
    for name, tensors in gui.populations.items():
        assert torch.equal(tensors["spikes"], cli.populations[name]["spikes"]), name
        for key in ("drive", "filtered"):
            assert_matches_golden(
                tensors[key],
                cli.populations[name][key],
                what=f"{name}/{key}: GUI bundle vs CLI bundle",
                rtol=1e-5,
                atol=1e-6,
            )


@pytest.mark.slow
def test_a_verbatim_yaml_export_runs_the_same_on_the_cli(qtbot, tmp_path):
    """Exporting a config verbatim is enough for the CLI (Task 0.6).

    This was a strict xfail while `sensoryforge run` ignored the `stimulus:`
    block and rendered a default trapezoid instead.
    """
    config = _config()

    result = _run_controller(qtbot, config, project_root=tmp_path / "project")
    config_path = tmp_path / "verbatim.yml"
    config_path.write_text(config.to_yaml(), encoding="utf-8")

    cli_bundle = tmp_path / "cli_bundle"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sensoryforge.cli",
            "run",
            str(config_path),
            "--duration",
            str(DURATION_MS),
            "--bundle",
            str(cli_bundle),
        ],
        cwd=str(tmp_path),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    gui = load_bundle(result.bundle_dir)
    cli = load_bundle(cli_bundle)
    for name, tensors in gui.populations.items():
        assert torch.equal(tensors["spikes"], cli.populations[name]["spikes"]), name
