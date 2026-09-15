"""Tests for `sensoryforge new-component --in-repo` scaffolding (G5, H2).

Covers:
- `generate_in_repo_component()` writes a module, test, and docs stub for
  every supported kind, with the right class name / registry key
  substitutions.
- The generated module satisfies the same contract as
  `tests/contract/test_component_contracts.py`: `get_param_spec()` returns
  `ParamSpec` objects, `from_config(to_dict())` round-trips, and a forward
  pass succeeds with the kind's canonical tensor shape.
- `generate_in_repo_component()` refuses to overwrite an existing module file.
- The CLI `new-component --in-repo` subcommand wires through to the scaffold.

The standalone-plugin-package default mode (the H2/F-047 fix) is covered by
`tests/unit/test_scaffold.py`.
"""

import importlib.util
import sys

import pytest
import torch

from sensoryforge.scaffold import available_kinds, generate_in_repo_component
from sensoryforge.stimuli.base import ParamSpec


def _load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scaffold_repo(tmp_path):
    (tmp_path / "sensoryforge").mkdir()
    for sub in ("neurons", "filters", "stimuli", "solvers", "core"):
        (tmp_path / "sensoryforge" / sub).mkdir()
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "docs" / "developer_guide").mkdir(parents=True)
    return tmp_path


def test_available_kinds():
    assert available_kinds() == ["filter", "grid", "neuron", "solver", "stimulus"]


@pytest.mark.parametrize("kind", available_kinds())
def test_generate_component_writes_three_files(scaffold_repo, kind):
    paths = generate_in_repo_component(kind, "MyScaffoldTest", repo_root=scaffold_repo)
    assert paths["module"].is_file()
    assert paths["test"].is_file()
    assert paths["docs"].is_file()
    assert "MyScaffoldTest" in paths["docs"].read_text()


def test_generate_component_refuses_overwrite(scaffold_repo):
    generate_in_repo_component("filter", "Dup", repo_root=scaffold_repo)
    with pytest.raises(FileExistsError):
        generate_in_repo_component("filter", "Dup", repo_root=scaffold_repo)


def test_generate_component_unknown_kind(scaffold_repo):
    with pytest.raises(ValueError):
        generate_in_repo_component("nonsense", "X", repo_root=scaffold_repo)


def test_scaffolded_neuron_satisfies_contract(scaffold_repo):
    paths = generate_in_repo_component(
        "neuron", "ContractCheck", repo_root=scaffold_repo
    )
    mod = _load_module(paths["module"], "sf_scaffold_test_neuron")
    cls = mod.ContractCheckNeuronTorch

    spec = cls.get_param_spec()
    assert isinstance(spec, list) and all(isinstance(p, ParamSpec) for p in spec)

    instance = cls()
    v_trace, spikes = instance(torch.randn(1, 5, 3))
    assert v_trace.shape == (1, 6, 3)
    assert spikes.shape == (1, 6, 3)

    reconstructed = cls.from_config(instance.to_dict())
    assert isinstance(reconstructed, cls)


def test_scaffolded_filter_satisfies_contract(scaffold_repo):
    paths = generate_in_repo_component(
        "filter", "ContractCheck", repo_root=scaffold_repo
    )
    mod = _load_module(paths["module"], "sf_scaffold_test_filter")
    cls = mod.ContractCheckFilterTorch

    instance = cls()
    out = instance(torch.randn(1, 5, 3))
    assert out.shape == (1, 5, 3)

    reconstructed = cls.from_config(instance.to_dict())
    assert isinstance(reconstructed, cls)


def test_scaffolded_stimulus_satisfies_contract(scaffold_repo):
    paths = generate_in_repo_component(
        "stimulus", "ContractCheck", repo_root=scaffold_repo
    )
    mod = _load_module(paths["module"], "sf_scaffold_test_stimulus")
    cls = mod.ContractCheckStimulus

    instance = cls()
    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 8), indexing="ij"
    )
    out = instance(xx, yy)
    assert out.shape == xx.shape

    reconstructed = cls.from_config(instance.to_dict())
    assert isinstance(reconstructed, cls)


def test_scaffolded_solver_satisfies_contract(scaffold_repo):
    paths = generate_in_repo_component(
        "solver", "ContractCheck", repo_root=scaffold_repo
    )
    mod = _load_module(paths["module"], "sf_scaffold_test_solver")
    cls = mod.ContractCheckSolver

    instance = cls()
    new_state = instance.step(
        lambda state, t: -0.1 * state, torch.randn(1, 3), t=0.0, dt=instance.dt
    )
    assert new_state.shape == (1, 3)

    reconstructed = cls.from_config(instance.to_dict())
    assert isinstance(reconstructed, cls)


def test_scaffolded_grid_satisfies_contract(scaffold_repo):
    paths = generate_in_repo_component("grid", "ContractCheck", repo_root=scaffold_repo)
    mod = _load_module(paths["module"], "sf_scaffold_test_grid")
    cls = mod.ContractCheckArrangement

    instance = cls(num_points=7)
    coords = instance.get_all_coordinates()
    assert coords.shape == (7, 2)

    reconstructed = cls.from_config(instance.to_dict())
    assert isinstance(reconstructed, cls)


def test_cli_new_component_in_repo_writes_files(scaffold_repo, monkeypatch):
    import sensoryforge.cli as cli_module
    from argparse import Namespace

    args = Namespace(kind="filter", name="CliCheck", dest=None, in_repo=True)
    monkeypatch.setattr("sensoryforge.scaffold.find_repo_root", lambda: scaffold_repo)
    monkeypatch.setattr(
        "sensoryforge.scaffold.generate_in_repo_component",
        lambda kind, name, repo_root: {
            "module": scaffold_repo / "module.py",
            "test": scaffold_repo / "test.py",
            "docs": scaffold_repo / "docs.md",
        },
    )
    assert cli_module.cmd_new_component(args) == 0


def test_cli_new_component_default_mode_writes_plugin_package(tmp_path, monkeypatch):
    import sensoryforge.cli as cli_module
    from argparse import Namespace

    args = Namespace(kind="filter", name="CliCheck", dest=str(tmp_path), in_repo=False)
    monkeypatch.setattr(
        "sensoryforge.scaffold.generate_plugin_package",
        lambda kind, name, dest: {
            "package_root": tmp_path / "sensoryforge-clicheck",
            "pyproject": tmp_path / "sensoryforge-clicheck" / "pyproject.toml",
            "module": tmp_path / "sensoryforge-clicheck" / "module.py",
            "test": tmp_path / "sensoryforge-clicheck" / "tests" / "test_contract.py",
            "readme": tmp_path / "sensoryforge-clicheck" / "README.md",
        },
    )
    assert cli_module.cmd_new_component(args) == 0


def test_cli_new_component_reports_unknown_kind_error(monkeypatch):
    import sensoryforge.cli as cli_module
    from argparse import Namespace

    args = Namespace(kind="not-a-kind", name="X", dest=None, in_repo=False)

    def _raise(kind, name, dest):
        raise ValueError(f"Unknown component kind {kind!r}")

    monkeypatch.setattr("sensoryforge.scaffold.generate_plugin_package", _raise)
    assert cli_module.cmd_new_component(args) == 1


def test_cli_new_component_in_repo_refuses_outside_checkout(tmp_path, monkeypatch):
    import sensoryforge.cli as cli_module
    from argparse import Namespace

    args = Namespace(kind="filter", name="X", dest=None, in_repo=True)

    def _raise():
        raise ValueError("no SensoryForge git checkout found")

    monkeypatch.setattr("sensoryforge.scaffold.find_repo_root", _raise)
    assert cli_module.cmd_new_component(args) == 1
