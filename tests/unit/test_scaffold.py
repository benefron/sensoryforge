"""Fast in-repo tests for `sensoryforge new-component` (H2/F-047).

Covers the core safety fix and both scaffold modes end-to-end via the CLI
subprocess, without the cost of a real wheel build/install (that proof lives
in the task report, not here -- see task-H2-report.md).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sensoryforge.scaffold import ensure_not_installed_path, find_repo_root

# This worktree's checkout root, so the CLI subprocess below imports *this*
# sensoryforge (with our scaffold.py changes) rather than whatever
# `sensoryforge` an editable/dev install elsewhere on sys.path resolves to
# from an arbitrary cwd (e.g. the main checkout at /Users/.../sensoryforge).
_WORKTREE_ROOT = Path(__file__).resolve().parents[2]


def _run_cli(args, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_WORKTREE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "sensoryforge.cli", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
    )


def test_ensure_not_installed_path_refuses_site_packages(tmp_path):
    fake_site_packages = tmp_path / "lib" / "python3.11" / "site-packages" / "myproj"
    fake_site_packages.mkdir(parents=True)
    with pytest.raises(ValueError, match="site-packages"):
        ensure_not_installed_path(fake_site_packages)


def test_ensure_not_installed_path_refuses_dist_packages(tmp_path):
    fake_dist_packages = (
        tmp_path / "usr" / "lib" / "python3" / "dist-packages" / "myproj"
    )
    fake_dist_packages.mkdir(parents=True)
    with pytest.raises(ValueError, match="dist-packages"):
        ensure_not_installed_path(fake_dist_packages)


def test_ensure_not_installed_path_allows_ordinary_dir(tmp_path):
    ordinary = tmp_path / "projects" / "my-plugin"
    ordinary.mkdir(parents=True)
    ensure_not_installed_path(ordinary)  # must not raise


def test_find_repo_root_refuses_outside_checkout(tmp_path):
    bare = tmp_path / "nowhere"
    bare.mkdir()
    with pytest.raises(ValueError, match="SensoryForge git checkout"):
        find_repo_root(start=bare)


def test_find_repo_root_finds_this_checkout():
    # This test file lives inside the real worktree checkout, so searching
    # up from here must find the worktree root.
    root = find_repo_root(start=Path(__file__).resolve().parent)
    assert (root / ".git").exists()
    assert (root / "pyproject.toml").is_file()
    assert (root / "sensoryforge").is_dir()


def test_cli_default_mode_writes_installable_plugin_package(tmp_path):
    result = _run_cli(
        ["new-component", "filter", "DemoScaffoldFilter", "--dest", "."],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    package_root = tmp_path / "sensoryforge-demo-scaffold-filter"
    assert package_root.is_dir()

    pyproject = package_root / "pyproject.toml"
    assert pyproject.is_file()
    pyproject_text = pyproject.read_text()
    assert '[project.entry-points."sensoryforge.components"]' in pyproject_text
    assert "demo_scaffold_filter" in pyproject_text

    module_file = package_root / "sensoryforge_demo_scaffold_filter" / "component.py"
    assert module_file.is_file()
    module_text = module_file.read_text()
    assert "class DemoScaffoldFilterFilterTorch" in module_text
    assert "def register() -> None:" in module_text

    test_file = package_root / "tests" / "test_contract.py"
    assert test_file.is_file()
    assert "check_component" in test_file.read_text()

    readme_file = package_root / "README.md"
    assert readme_file.is_file()
    assert "pip install -e ." in readme_file.read_text()

    # Nothing should have been written anywhere under site-packages or
    # dist-packages, or outside the requested --dest.
    for path in package_root.rglob("*"):
        resolved_parts = {p.lower() for p in path.resolve().parts}
        assert "site-packages" not in resolved_parts
        assert "dist-packages" not in resolved_parts


def test_generated_neuron_template_accepts_noise_std(tmp_path):
    """I1 regression guard: `SimulationEngine` unconditionally injects
    ``noise_std`` into every neuron's constructor kwargs
    (`_build_populations` in `core/simulation_engine.py`), so the scaffold's
    generated neuron template must declare it or a plugin neuron generated
    by ``sensoryforge new-component neuron`` crashes with ``TypeError`` the
    first time it is used in a real simulation.
    """
    result = _run_cli(
        ["new-component", "neuron", "DemoScaffoldNeuron", "--dest", "."],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    package_root = tmp_path / "sensoryforge-demo-scaffold-neuron"
    module_file = package_root / "sensoryforge_demo_scaffold_neuron" / "component.py"
    module_text = module_file.read_text()
    assert "noise_std" in module_text

    # Import the generated module directly (without installing the package)
    # and instantiate it with noise_std=0.1, exactly as SimulationEngine's
    # _build_populations would call it.
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(_WORKTREE_ROOT)
        + os.pathsep
        + str(package_root)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sensoryforge_demo_scaffold_neuron.component import "
            "DemoScaffoldNeuronNeuronTorch as N; "
            "n = N(noise_std=0.1); "
            "assert n.noise_std == 0.1; "
            "assert n.to_dict()['noise_std'] == 0.1; "
            "print('OK')",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert check.returncode == 0, check.stdout + check.stderr
    assert "OK" in check.stdout


def test_cli_in_repo_mode_refuses_outside_checkout(tmp_path):
    result = _run_cli(
        ["new-component", "filter", "ShouldNotWrite", "--in-repo"],
        cwd=tmp_path,
    )
    assert result.returncode == 1
    assert "SensoryForge git checkout" in result.stderr
    # Nothing should have been written into the bare temp dir.
    assert list(tmp_path.iterdir()) == []
