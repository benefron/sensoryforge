"""`sensoryforge run` must render the canonical `stimulus:` block (Task 0.6, F-061).

Before this fix, `cmd_run` (`sensoryforge/cli.py`) chose a canonical config's
stimulus from a legacy top-level `stimuli:` list, falling back to a default
trapezoidal stimulus otherwise -- it never read `SensoryForgeConfig.stimulus`
(the canonical `stimulus:` block `SensoryForgeConfig.to_yaml()` writes and the
GUI exports). So a config exported from the GUI with, say, a moving edge ran a
default Gaussian trapezoid on the CLI instead.

`render_for_config` (`sensoryforge/stimuli/render.py`) is now the one renderer
both `sensoryforge run` and the GUI's Circuit tab
(`sensoryforge/gui/circuit/run.py::render_graph_stimulus`) call, so a config
run from either entry point renders byte-identical stimulus frames.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# A "moving_edge" stimulus (a registered [T, H, W] stimulus, see
# sensoryforge/stimuli/tactile.py::MovingEdgeStimulus) sweeps an oriented
# edge across the grid -- visibly different frames from the default
# trapezoidal stimulus, which is a static Gaussian blob held for a plateau
# (GeneralizedTactileEncodingPipeline's temporal defaults).
_GRID = {
    "name": "Skin",
    "arrangement": "grid",
    "rows": 8,
    "cols": 8,
    "spacing": 0.2,
    "center_x": 0.0,
    "center_y": 0.0,
}

_STIMULUS = {
    "type": "moving_edge",
    "start": [-3.0, 0.0],
    "end": [3.0, 0.0],
    "spread": 0.8,
    "orientation_deg": 25.0,
    "amplitude": 40.0,
    "ramp_up_ms": 5.0,
    "plateau_ms": 40.0,
    "ramp_down_ms": 5.0,
}

_POPULATION = {
    "name": "SA Pop",
    "target_grid": "Skin",
    "neuron_type": "SA",
    "neurons_per_row": 2,
    "innervation_method": "gaussian",
    "connections_per_neuron": 4,
    "sigma_d_mm": 0.3,
    "filter_method": "sa",
    "neuron_model": "Izhikevich",
    "input_gain": 1.0,
    "noise_std": 0.0,
    "seed": 42,
}

_SIMULATION = {"dt_ms": 1.0, "device": "cpu"}

_DURATION = 30


def _canonical_config(*, with_stimulus_block: bool) -> dict:
    config = {
        "grids": [dict(_GRID)],
        "populations": [dict(_POPULATION)],
        "simulation": dict(_SIMULATION),
    }
    if with_stimulus_block:
        config["stimulus"] = dict(_STIMULUS)
    return config


def _write_config(tmp_path: Path, config: dict, name: str = "config.yml") -> Path:
    path = tmp_path / name
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


def _run_cli(config_path: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sensoryforge.cli",
            "run",
            str(config_path),
            "--duration",
            str(_DURATION),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO_ROOT),
    )


def _default_trapezoid_frames() -> torch.Tensor:
    """The default trapezoidal stimulus for the same grid/duration/dt.

    Built the same way `cmd_run`'s legacy-default branch does, so the
    comparison below is against exactly what the CLI would have rendered
    before this fix (F-061) -- ignoring the `stimulus:` block entirely.
    """
    from sensoryforge.stimuli.canvas import stimulus_canvas
    from sensoryforge.stimuli.render import render_stimulus
    from sensoryforge.config.schema import GridConfig

    grid_cfg = GridConfig(**_GRID)
    canvas = stimulus_canvas(grid_cfg, device="cpu")
    frames, _ = render_stimulus(
        "trapezoidal",
        {},
        canvas.xx,
        canvas.yy,
        dt_ms=_SIMULATION["dt_ms"],
        duration_ms=float(_DURATION),
        device="cpu",
    )
    return frames


def test_stimulus_block_type_is_recorded_in_the_bundle(tmp_path):
    """The bundle's stimulus payload records the `stimulus:` block's own type."""
    config_path = _write_config(tmp_path, _canonical_config(with_stimulus_block=True))
    bundle_dir = tmp_path / "bundle"

    result = _run_cli(config_path, "--bundle", str(bundle_dir))
    assert result.returncode == 0, (
        f"exited {result.returncode}\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    # The deprecation notice must NOT fire: there is no legacy `stimuli:`
    # list here, only the canonical `stimulus:` block.
    assert "legacy top-level 'stimuli:' list" not in result.stdout
    assert "Using the canonical 'stimulus:' block" in result.stdout

    import json

    payload = json.loads((bundle_dir / "stimuli" / "stimulus.json").read_text())
    assert payload["type"] == "moving_edge", payload


def test_stimulus_block_frames_differ_from_default_trapezoid(tmp_path):
    """Rendered frames come from the declared `moving_edge`, not the default."""
    from sensoryforge.io.bundle import load_bundle

    config_path = _write_config(tmp_path, _canonical_config(with_stimulus_block=True))
    bundle_dir = tmp_path / "bundle"

    result = _run_cli(config_path, "--bundle", str(bundle_dir))
    assert result.returncode == 0, result.stdout + result.stderr

    bundle = load_bundle(bundle_dir)
    assert bundle.stimulus is not None

    default_frames = _default_trapezoid_frames()
    assert bundle.stimulus.shape == default_frames.shape
    assert not torch.allclose(bundle.stimulus, default_frames), (
        "sensoryforge run rendered the default trapezoidal stimulus instead "
        "of the canonical 'stimulus:' block"
    )


def test_default_stimulus_block_still_falls_back_to_trapezoid(tmp_path):
    """No `stimulus:` block at all -> the untouched schema default -> trapezoid."""
    from sensoryforge.io.bundle import load_bundle

    config_path = _write_config(tmp_path, _canonical_config(with_stimulus_block=False))
    bundle_dir = tmp_path / "bundle"

    result = _run_cli(config_path, "--bundle", str(bundle_dir))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "schema default" in result.stdout

    bundle = load_bundle(bundle_dir)
    default_frames = _default_trapezoid_frames()
    assert torch.allclose(bundle.stimulus, default_frames)


def test_legacy_stimuli_list_still_runs_with_a_deprecation_notice(tmp_path):
    """A legacy top-level `stimuli:` list still wins, with a printed notice."""
    from sensoryforge.io.bundle import load_bundle

    config = _canonical_config(with_stimulus_block=True)
    config["stimuli"] = [{"type": "gaussian", "amplitude": 10.0, "sigma": 1.0}]
    config_path = _write_config(tmp_path, config)
    bundle_dir = tmp_path / "bundle"

    result = _run_cli(config_path, "--bundle", str(bundle_dir))
    assert result.returncode == 0, (
        f"exited {result.returncode}\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "legacy top-level 'stimuli:' list" in result.stdout

    import json

    payload = json.loads((bundle_dir / "stimuli" / "stimulus.json").read_text())
    # "gaussian" is one of pressure-simulation's own reconstructible types
    # (see build_stimulus_payload), tagged "kind": "stimulus".
    assert payload["type"] == "gaussian"

    bundle = load_bundle(bundle_dir)
    assert bundle.stimulus is not None


def test_validate_reports_the_stimulus_source(tmp_path):
    """`sensoryforge validate` names which stimulus source `run` will use."""
    block_path = _write_config(
        tmp_path, _canonical_config(with_stimulus_block=True), "with_block.yml"
    )
    default_path = _write_config(
        tmp_path, _canonical_config(with_stimulus_block=False), "default.yml"
    )
    legacy_config = _canonical_config(with_stimulus_block=True)
    legacy_config["stimuli"] = [{"type": "gaussian"}]
    legacy_path = _write_config(tmp_path, legacy_config, "legacy.yml")

    def _validate(path: Path) -> str:
        result = subprocess.run(
            [sys.executable, "-m", "sensoryforge.cli", "validate", str(path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    assert "canonical 'stimulus:' block" in _validate(block_path)
    assert "default trapezoidal stimulus" in _validate(default_path)
    assert "legacy top-level 'stimuli:' list" in _validate(legacy_path)


@pytest.mark.gui
def test_run_graph_once_renders_the_same_stimulus_as_the_cli(tmp_path):
    """The Circuit tab's `run_graph_once` and `sensoryforge run` agree exactly.

    Both go through `render_for_config` now (F-061), so a config run from
    the CLI and the identical config built as a flowchart must render
    bit-identical stimulus frames.
    """
    import sys as _sys

    _app = None
    from PyQt5 import QtWidgets

    _app = QtWidgets.QApplication.instance()
    if _app is None:
        _app = QtWidgets.QApplication(_sys.argv[:1])

    from sensoryforge.gui.tabs.circuit_tab import CircuitTab
    from sensoryforge.gui.circuit.run import run_graph_once
    from sensoryforge.config.schema import GridConfig, StimulusConfig
    from sensoryforge.io.bundle import load_bundle

    # 1) Run the config through the CLI, exactly like the other tests here.
    config_path = _write_config(tmp_path, _canonical_config(with_stimulus_block=True))
    bundle_dir = tmp_path / "bundle"
    result = _run_cli(config_path, "--bundle", str(bundle_dir))
    assert result.returncode == 0, result.stdout + result.stderr
    cli_bundle = load_bundle(bundle_dir)

    # 2) Build the identical config as a Circuit tab flowchart.
    tab = CircuitTab()
    grid_node = tab.add_node("SensorArray", "grid1")
    grid_node.from_config(GridConfig(name="grid1", **_GRID_KW()))

    stim_node = tab.add_node("Stimulus", "stim1")
    stim_node.from_config(StimulusConfig(name="stim1", **_STIMULUS))

    rf_node = tab.add_node("RFBank", "rf1")
    filt_node = tab.add_node("Filter", "filter1")
    filt_node.from_config({"filter_method": "sa", "filter_params": {}})
    readout_node = tab.add_node("Readout", "SA Pop")
    readout_node.from_config(dict(_POPULATION, target_grid=None))

    tab.connect_nodes(grid_node, "value", rf_node, "Channel")
    tab.connect_nodes(rf_node, "Drive", filt_node, "Drive")
    tab.connect_nodes(filt_node, "Filtered", readout_node, "Filtered")

    _config, _raw_results, frames, _dt_ms = run_graph_once(
        tab.flowchart, duration_ms=float(_DURATION)
    )

    assert torch.allclose(frames, cli_bundle.stimulus)


def _GRID_KW() -> dict:
    return {k: v for k, v in _GRID.items() if k != "name"}
