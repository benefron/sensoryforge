"""`sensoryforge run` handles a population that does not spike (F-060).

Wave N gave a neuron model with no spike condition a ``"state"`` trace and
no ``"spikes"`` key; Wave J wired the bundle into the CLI. Neither wave
tested the CLI with an analog population, so its summary loop still read
``pop_results["spikes"]`` unconditionally and raised ``KeyError: 'spikes'``
-- after the bundle had already been written, so the run had actually
succeeded and only the reporting failed.

Found by the Phase 2 exit check, running a config with an analog and a
spiking population together from a wheel installed outside the repo.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG = """
grids:
  - name: Skin
    arrangement: grid
    rows: 8
    cols: 8
    spacing: 0.3
    seed: 3

populations:
  - name: Analog
    neuron_type: SA
    neuron_model: dsl
    filter_method: sa
    neurons_per_row: 2
    readout: analog
    dsl_config:
      equations: "dv/dt = (-(v - v_rest) + R*I) / tau_m"
      parameters: {v_rest: -65.0, R: 1.0, tau_m: 10.0}
      state_vars: {v: -65.0}

  - name: Spiking
    neuron_type: RA
    neuron_model: izhikevich
    filter_method: ra
    neurons_per_row: 2

stimulus:
  type: gaussian
  amplitude: 30.0

simulation:
  device: cpu
  dt_ms: 1.0
  integrate_dt_ms: 0.5
"""


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "analog_config.yml"
    path.write_text(CONFIG)
    return path


def _run_cli(config_path, *extra):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sensoryforge.cli",
            "run",
            str(config_path),
            "--duration",
            "20",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO_ROOT),
    )


def test_cli_run_succeeds_with_an_analog_population(config_path):
    result = _run_cli(config_path)
    assert result.returncode == 0, (
        f"exited {result.returncode}\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "KeyError" not in result.stdout + result.stderr


def test_cli_reports_both_readout_kinds(config_path):
    result = _run_cli(config_path)
    out = result.stdout
    assert "Spiking spikes:" in out, out
    assert "Analog analog state:" in out, (
        "the analog population was not summarised; the CLI should report its "
        f"state rather than silently skipping it.\n{out}"
    )


def test_cli_writes_a_bundle_with_an_analog_population(config_path, tmp_path):
    bundle = tmp_path / "bundle"
    result = _run_cli(config_path, "--bundle", str(bundle))
    assert result.returncode == 0, result.stdout + result.stderr

    h5py = pytest.importorskip("h5py")
    with h5py.File(bundle / "data.h5", "r") as f:
        pops = f["populations"]
        assert "state" in pops["Analog"] and "spikes" not in pops["Analog"]
        assert "spikes" in pops["Spiking"] and "state" not in pops["Spiking"]
