"""hex/poisson arrangements run through every config-driven path (F-076).

Before ``sensoryforge/stimuli/canvas.py::stimulus_canvas`` existed, the CLI
(`sensoryforge.cli`), ``BatchExecutor`` and the Circuit tab
(`sensoryforge.gui.circuit.run.run_graph_once`) each built a throwaway
``ReceptorGrid(...)`` purely to call ``get_coordinates()`` for a render
canvas -- which raises ``ValueError`` for ``"poisson"``/``"hex"`` (no
lattice), so a config using either arrangement could not run through any of
them (F-076), even though ``SimulationEngine`` already samples any
arrangement correctly (Wave L3, F-010). This module runs a small hex config
and a small (seeded) poisson config through all three paths and checks the
receptor drive they compute agrees.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.core.batch_executor import BatchExecutor
from sensoryforge.io.bundle import load_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
POP_NAME = "SA Population"


def _canonical_config_dict(arrangement: str, seed: int) -> dict:
    """A small one-grid/one-population canonical config for ``arrangement``."""
    return {
        "metadata": {"name": f"{arrangement} nongrid test"},
        "grids": [
            {
                "name": "Main Grid",
                "arrangement": arrangement,
                "rows": 10,
                "cols": 10,
                "spacing": 0.15,
                "seed": seed,
            }
        ],
        "populations": [
            {
                "name": POP_NAME,
                "neuron_type": "SA",
                "target_grid": "Main Grid",
                "innervation_method": "gaussian",
                "connections_per_neuron": 8,
                "sigma_d_mm": 0.3,
                "neuron_arrangement": "grid",
                "neurons_per_row": 4,
                "neuron_model": "Izhikevich",
                "filter_method": "sa",
                "input_gain": 50.0,
                "noise_std": 0.0,
                "seed": seed,
            }
        ],
        # "stimulus" (singular) is the canonical field SimulationEngine's
        # callers use (SensoryForgeConfig.stimulus -- render_graph_stimulus,
        # BatchExecutor's base_config). "stimuli" (a legacy-shaped list) is
        # what `sensoryforge run`'s cmd_run reads instead (cli.py ~232-238);
        # both are given the same gaussian params so all three paths render
        # the identical stimulus.
        "stimulus": {
            "name": "stim1",
            "type": "gaussian",
            "amplitude": 30.0,
            "sigma": 0.5,
        },
        "stimuli": [
            {"type": "gaussian", "amplitude": 30.0, "sigma": 0.5},
        ],
        "simulation": {
            "device": "cpu",
            "dt_ms": 1.0,
            "integrate_dt_ms": 0.05,
        },
    }


def _batch_config_dict(config_dict: dict, output_dir: Path) -> dict:
    """A one-stimulus batch config wrapping ``config_dict`` (no top-level
    ``stimulus``/``stimuli`` keys -- the batch's own ``stimuli`` sweep spec
    supplies the stimulus instead)."""
    base_config = {
        k: v for k, v in config_dict.items() if k not in ("stimulus", "stimuli")
    }
    return {
        "metadata": {"batch_name": "nongrid_test"},
        "base_config": base_config,
        "batch": {
            "output_dir": str(output_dir),
            "save_intermediates": True,
            "stimuli": [
                {
                    "type": "gaussian",
                    "base_seed": 42,
                    "parameters": {
                        "amplitude": [30.0],
                        "sigma": [0.5],
                        "duration": [20.0],
                    },
                    "repetitions": 1,
                }
            ],
        },
    }


def _run_batch_executor(config_dict: dict, tmp_path: Path) -> torch.Tensor:
    """Run one stimulus through ``BatchExecutor`` and return its drive."""
    batch_output = tmp_path / "batch"
    batch_config = _batch_config_dict(config_dict, batch_output)
    executor = BatchExecutor(batch_config)
    result = executor.execute(task_index=0)
    bundle_dir = Path(result["output_path"])
    bundle = load_bundle(bundle_dir)
    return torch.as_tensor(bundle.populations[POP_NAME]["drive"])


def _run_cli_subprocess(config_dict: dict, tmp_path: Path) -> torch.Tensor:
    """Run ``sensoryforge run <yaml> --bundle <dir>`` as a real subprocess."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config_dict, f)
    bundle_dir = tmp_path / "cli_bundle"

    env = {"PYTHONPATH": str(REPO_ROOT)}
    import os

    env.update(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sensoryforge.cli",
            "run",
            str(config_path),
            "--duration",
            "20",
            "--bundle",
            str(bundle_dir),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"CLI subprocess failed (arrangement={config_dict['grids'][0]['arrangement']}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    bundle = load_bundle(bundle_dir)
    return torch.as_tensor(bundle.populations[POP_NAME]["drive"])


def _run_circuit_tab(config_dict: dict) -> torch.Tensor:
    """Run the same config through the Circuit tab's ``run_graph_once``."""
    from sensoryforge.gui.circuit.run import run_graph_once
    from sensoryforge.gui.circuit.serialise import config_to_graph
    from sensoryforge.gui.tabs.circuit_tab import CircuitTab

    config = SensoryForgeConfig.from_dict(config_dict)
    tab = CircuitTab()
    config_to_graph(config, tab.flowchart)
    _, raw_results, _, _ = run_graph_once(tab.flowchart, duration_ms=20.0)
    return raw_results[POP_NAME]["drive"]


@pytest.mark.parametrize("arrangement,seed", [("hex", 1), ("poisson", 1)])
def test_cli_and_batch_executor_agree(arrangement, seed, tmp_path):
    """The CLI (subprocess) and BatchExecutor produce the same drive for a
    hex/poisson config that ``ReceptorGrid.get_coordinates()`` alone cannot
    render (F-076)."""
    config_dict = _canonical_config_dict(arrangement, seed)

    batch_drive = _run_batch_executor(config_dict, tmp_path / "batch_run")
    cli_drive = _run_cli_subprocess(config_dict, tmp_path / "cli_run")

    assert cli_drive.shape == batch_drive.shape
    assert torch.allclose(
        cli_drive.float(), batch_drive.float(), rtol=1e-5, atol=1e-6
    ), f"{arrangement}: CLI bundle drive != BatchExecutor bundle drive"


@pytest.mark.gui
@pytest.mark.parametrize("arrangement,seed", [("hex", 1), ("poisson", 1)])
def test_circuit_tab_agrees_with_batch_executor(arrangement, seed, tmp_path):
    """The Circuit tab's ``run_graph_once`` agrees with ``BatchExecutor`` for
    hex/poisson too."""
    import sys as _sys

    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(_sys.argv[:1])

    config_dict = _canonical_config_dict(arrangement, seed)

    batch_drive = _run_batch_executor(config_dict, tmp_path / "batch_run")
    circuit_drive = _run_circuit_tab(config_dict)

    # run_graph_once's drive carries a batch dim; the bundle's does not.
    if circuit_drive.ndim == batch_drive.ndim + 1:
        circuit_drive = circuit_drive.squeeze(0)

    assert circuit_drive.shape == batch_drive.shape
    assert torch.allclose(
        circuit_drive.float(), batch_drive.float(), rtol=1e-5, atol=1e-6
    ), f"{arrangement}: Circuit tab drive != BatchExecutor drive"
