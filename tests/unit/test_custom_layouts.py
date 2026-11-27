"""A registered grid plugin can be named as a sensor layout or a neuron layout."""

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch
import yaml

from sensoryforge.cli import validate_config
from sensoryforge.config.schema import (
    GridConfig,
    PopulationConfig,
    SensoryForgeConfig,
    SimulationConfig,
)
from sensoryforge.core.simulation_engine import SimulationEngine, build_grid
from sensoryforge.io.bundle import load_bundle
from sensoryforge.plugins import load_plugin_import_paths
from sensoryforge.core.batch_executor import BatchExecutor
from sensoryforge.core.generalized_pipeline import (
    GeneralizedTactileEncodingPipeline,
)
from sensoryforge.core.grid_base import BaseGrid
from sensoryforge.core.layouts import (
    BUILTIN_LAYOUTS,
    build_layout_points,
    custom_layout_names,
    layout_problem,
)
from sensoryforge.register_components import register_all
from sensoryforge.registry import GRID_REGISTRY
from sensoryforge.stimuli.base import ParamSpec
from sensoryforge.testing.contracts import check_component


class _Layout(BaseGrid):
    """A test layout: ``self.points()`` is what it returns."""

    def __init__(self, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), device="cpu"):
        super().__init__(xlim, ylim, device)

    def get_all_coordinates(self):
        return self.points().to(self.device)

    @classmethod
    def from_config(cls, config):
        return cls(**{k: v for k, v in config.items() if k != "type"})

    @classmethod
    def get_param_spec(cls):
        return [ParamSpec("n", dtype="int", default=7)]


class Diagonal(_Layout):
    """``n`` points on the area's diagonal, both corners included, in float64."""

    def __init__(self, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), device="cpu", n=7):
        super().__init__(xlim, ylim, device)
        self.n = n

    def points(self):
        x = torch.linspace(*self.xlim, self.n, dtype=torch.float64)
        y = torch.linspace(*self.ylim, self.n, dtype=torch.float64)
        return torch.stack([x, y], dim=1)


class Scatter(_Layout):
    """Five random points drawn from ``seed``."""

    def __init__(self, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), device="cpu", seed=None):
        super().__init__(xlim, ylim, device)
        self.seed = seed

    def points(self):
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        unit = torch.rand(5, 2, generator=generator)
        low = torch.tensor([self.xlim[0], self.ylim[0]])
        high = torch.tensor([self.xlim[1], self.ylim[1]])
        return low + unit * (high - low)


class Lattice(_Layout):
    """The built-in grid's own points, ``nx`` along x by ``ny`` along y."""

    def __init__(self, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), device="cpu", nx=12, ny=8):
        super().__init__(xlim, ylim, device)
        self.nx, self.ny = nx, ny

    def points(self):
        xx, yy = torch.meshgrid(
            torch.linspace(*self.xlim, self.nx),
            torch.linspace(*self.ylim, self.ny),
            indexing="ij",
        )
        return torch.stack([xx.flatten(), yy.flatten()], dim=1)


class Broken(_Layout):
    """Returns whatever ``kind`` says is wrong."""

    def __init__(self, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), device="cpu", kind="shape"):
        super().__init__(xlim, ylim, device)
        self.kind = kind

    def points(self):
        if self.kind == "zero_division":
            return torch.zeros(int(1.0 / 0), 2)  # a plugin's own bug
        return {
            "shape": torch.zeros(4, 3),
            "empty": torch.zeros(0, 2),
            "nan": torch.tensor([[0.0, float("nan")]]),
            "outside": torch.tensor([[self.xlim[1] + 0.01, 0.0]]),
        }[self.kind]


class NeedsSetting(_Layout):
    def __init__(self, turns, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), device="cpu"):
        super().__init__(xlim, ylim, device)

    def points(self):
        return torch.zeros(1, 2)


class NoArea(BaseGrid):
    """An older grid plugin: it cannot be told the area to fill."""

    def __init__(self, num_points=3):
        super().__init__((-1.0, 1.0), (-1.0, 1.0))

    def get_all_coordinates(self):
        return torch.zeros(3, 2)

    @classmethod
    def from_config(cls, config):
        return cls()


_TEST_LAYOUTS = {
    "test_diagonal": Diagonal,
    "test_scatter": Scatter,
    "test_lattice": Lattice,
    "test_broken": Broken,
    "test_needs_setting": NeedsSetting,
    "test_no_area": NoArea,
}


@pytest.fixture(autouse=True, scope="module")
def _registered_layouts():
    """Register the test layouts for this module only.

    The registry is process-wide, and other tests list everything in it.
    """
    register_all()
    for name, cls in _TEST_LAYOUTS.items():
        GRID_REGISTRY.register(name, cls)
    yield
    for name in _TEST_LAYOUTS:
        GRID_REGISTRY._registry.pop(name.casefold(), None)


AREA = ((-2.0, 2.0), (-1.0, 1.0))


# --------------------------------------------------------------- the builder


def test_points_come_back_as_float32_including_the_area_edges():
    points = build_layout_points("test_diagonal", *AREA, params={"n": 3})

    assert points.dtype == torch.float32
    assert points.tolist() == [[-2.0, -1.0], [0.0, 0.0], [2.0, 1.0]]


def test_custom_names_are_the_plugins_that_accept_an_area():
    names = custom_layout_names()

    assert "test_diagonal" in names
    assert "test_no_area" not in names
    assert not set(BUILTIN_LAYOUTS) & set(names)


def test_a_seed_reaches_a_layout_that_takes_one():
    a = build_layout_points("test_scatter", *AREA, seed=3)
    b = build_layout_points("test_scatter", *AREA, seed=3)
    c = build_layout_points("test_scatter", *AREA, seed=4)

    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_a_seed_for_a_layout_without_one_is_ignored_unless_required():
    build_layout_points("test_diagonal", *AREA, seed=3)
    with pytest.raises(ValueError, match="test_diagonal.*seed"):
        build_layout_points("test_diagonal", *AREA, seed=3, require_seed_support=True)


@pytest.mark.parametrize(
    "name, params, expected",
    [
        ("demo_spiral", {}, "Unknown layout.*jittered_grid.*test_diagonal"),
        ("Test_Diagonal", {}, "Unknown layout"),
        ("hex", {}, "built-in"),
        ("test_no_area", {}, "test_no_area.*xlim"),
        ("test_diagonal", {"m": 1}, "test_diagonal.*'m'"),
        ("test_diagonal", {"xlim": [0, 1]}, "test_diagonal.*'xlim'"),
        ("test_diagonal", {"seed": 1}, "test_diagonal.*'seed'"),
        ("test_needs_setting", {}, "test_needs_setting.*'turns'"),
    ],
)
def test_a_layout_that_cannot_be_used_says_why(name, params, expected):
    assert layout_problem("test_diagonal", {"n": 3}) is None
    with pytest.raises(ValueError, match=expected):
        build_layout_points(name, *AREA, params=params)


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("shape", r"\[N, 2\]"),
        ("empty", r"\[N, 2\]"),
        ("nan", "finite"),
        ("outside", "outside"),
    ],
)
def test_bad_points_are_refused_and_never_moved(kind, expected):
    with pytest.raises(ValueError, match=f"test_broken.*{expected}"):
        build_layout_points("test_broken", *AREA, params={"kind": kind})


def test_a_setting_the_plugin_rejects_is_reported_with_its_name():
    with pytest.raises(ValueError, match="test_diagonal"):
        build_layout_points("test_diagonal", *AREA, params={"n": -1})


def test_any_error_raised_by_plugin_code_becomes_a_value_error():
    """The GUI validates on every edit and catches ValueError, not a plugin's bug."""
    with pytest.raises(ValueError, match="test_broken.*ZeroDivisionError"):
        build_layout_points("test_broken", *AREA, params={"kind": "zero_division"})


# --------------------------------------------------------------- config keys


def test_layout_settings_round_trip_and_are_left_out_when_empty():
    grid = GridConfig(name="g", arrangement="test_diagonal", layout_params={"n": 4})
    pop = PopulationConfig(
        name="p",
        neuron_arrangement="test_diagonal",
        neuron_layout_params={"n": 5},
    )

    assert GridConfig.from_dict(grid.to_dict()).layout_params == {"n": 4}
    assert PopulationConfig.from_dict(pop.to_dict()).neuron_layout_params == {"n": 5}
    assert "layout_params" not in GridConfig(name="g").to_dict()
    assert "neuron_layout_params" not in PopulationConfig(name="p").to_dict()


@pytest.mark.parametrize("arrangement", ["grid", "hex", "composite"])
def test_sensor_settings_with_a_built_in_name_are_an_error(arrangement):
    with pytest.raises(ValueError, match="layout_params.*built-in"):
        GridConfig(name="g", arrangement=arrangement, layout_params={"n": 4})


def test_neuron_settings_with_a_built_in_name_are_an_error():
    with pytest.raises(ValueError, match="neuron_layout_params.*built-in"):
        PopulationConfig(name="p", neuron_layout_params={"n": 4})


def test_the_schema_and_the_engine_agree_on_the_built_in_names():
    from sensoryforge.config import schema

    assert schema._BUILTIN_LAYOUT_NAMES == BUILTIN_LAYOUTS


# ------------------------------------------------------- neurons in the engine


def _config(grid=None, **population):
    population.setdefault("name", "p")
    population.setdefault("seed", 1)
    return SensoryForgeConfig(
        grids=[grid or GridConfig(name="g", rows=12, cols=8, spacing=0.2)],
        populations=[PopulationConfig(**population)],
        simulation=SimulationConfig(device="cpu", dt_ms=1.0),
    )


def _custom_neurons(**population):
    population.setdefault("neuron_arrangement", "test_diagonal")
    population.setdefault("neuron_layout_params", {"n": 6})
    return _config(**population)


def _blob():
    """A held Gaussian on the 12 x 8 canvas, wider in x than in y, off centre."""
    x = torch.linspace(-1.1, 1.1, 12)
    y = torch.linspace(-0.7, 0.7, 8)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    frame = torch.exp(-(((xx - 0.5) / 0.6) ** 2) - ((yy + 0.2) / 0.2) ** 2)
    stimulus = torch.zeros(1, 40, 12, 8)
    stimulus[:, 5:] = frame
    return stimulus


def _quiet_run(config, stimulus):
    return SimulationEngine(config).run(stimulus, return_intermediates=True)


def test_the_layout_places_the_neurons_and_decides_how_many():
    engine = SimulationEngine(_custom_neurons())

    # The grid spans x -1.1..1.1 (12 rows) and y -0.7..0.7 (8 cols).
    expected = build_layout_points(
        "test_diagonal", (-1.1, 1.1), (-0.7, 0.7), params={"n": 6}
    )
    centres = engine.populations[0]["bank"].neuron_centers
    assert torch.allclose(centres, expected, atol=1e-5)
    results = engine.run(_blob(), return_intermediates=True)
    assert results["p"]["spikes"].shape[-1] == 6


def test_the_neuron_area_is_the_grid_shrunk_by_the_edge_offset():
    engine = SimulationEngine(_custom_neurons(edge_offset=0.1))

    centres = engine.populations[0]["bank"].neuron_centers
    assert torch.allclose(centres[0], torch.tensor([-1.0, -0.6]), atol=1e-5)
    assert torch.allclose(centres[-1], torch.tensor([1.0, 0.6]), atol=1e-5)


@pytest.mark.parametrize(
    "field, value",
    [
        ("neurons_per_row", 4),
        ("neuron_rows", 3),
        ("neuron_cols", 3),
        ("neuron_jitter_factor", 0.5),
    ],
)
def test_a_neuron_setting_a_custom_layout_ignores_is_warned_about(field, value):
    with pytest.warns(UserWarning, match=f"test_diagonal.*{field}"):
        SimulationEngine(_custom_neurons(**{field: value}))


def test_no_warning_when_only_defaults_are_ignored():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        SimulationEngine(_custom_neurons())


def test_a_builder_with_its_own_lattice_says_the_custom_layout_is_ignored():
    config = _custom_neurons(innervation_method="template", resolvable_distance_mm=0.4)
    with pytest.warns(UserWarning, match="derives its own neuron lattice.*test_"):
        SimulationEngine(config)


def test_inputs_of_one_population_share_the_custom_centres():
    from sensoryforge.config.schema import PopulationInput

    config = _custom_neurons(
        inputs=[PopulationInput(grid="g"), PopulationInput(grid="g")]
    )
    pop = SimulationEngine(config).populations[0]

    first, second = (entry["bank"].neuron_centers for entry in pop["inputs"])
    assert torch.equal(first, second)


# ------------------------------------------------------- sensors in the engine


def _sensor_grid(name="test_lattice", **extra):
    extra.setdefault("rows", 12)
    extra.setdefault("cols", 8)
    return GridConfig(name="g", arrangement=name, spacing=0.2, **extra)


def test_sensors_are_the_layouts_points_inside_the_configured_area():
    grid = build_grid(_sensor_grid("test_diagonal", layout_params={"n": 4}))

    assert grid.xlim == pytest.approx((-1.1, 1.1))
    assert grid.ylim == pytest.approx((-0.7, 0.7))
    expected = build_layout_points(
        "test_diagonal", grid.xlim, grid.ylim, params={"n": 4}
    )
    assert torch.equal(grid.get_all_coordinates(), expected)
    assert grid.provenance["layout"] == "test_diagonal"


def test_sensors_on_the_grid_points_read_exactly_what_the_grid_reads():
    """x and y are not swapped: a lopsided stimulus on a 12 x 8 canvas.

    A plugin grid wires like a position-file grid, with a 3-sigma cut-off the
    built-in grid does not apply, so the cut-off is turned off here to compare
    the two outputs.
    """
    wiring = {"innervation_params": {"max_sigma_distance": 0.0}}
    engines = [
        SimulationEngine(_config(_sensor_grid(name), **wiring))
        for name in ("grid", "test_lattice")
    ]
    stimulus = _blob()
    readings = [
        e._stimulus_to_receptors(
            stimulus,
            e.grids[0],
            e.populations[0]["bank"].receptor_coords,
            e.grid_managers.get("g"),
        )
        for e in engines
    ]
    filtered = [e.run(stimulus, return_intermediates=True)["p"] for e in engines]

    assert readings[0].abs().sum() > 0
    assert torch.allclose(readings[1], readings[0], atol=1e-5)
    assert torch.allclose(filtered[1]["filtered"], filtered[0]["filtered"], atol=1e-3)


def test_a_plugin_sensor_grid_matches_the_same_points_from_a_file(tmp_path):
    points = build_grid(_sensor_grid()).get_all_coordinates()
    torch.save(points, tmp_path / "points.pt")
    from_file = GridConfig(
        name="g",
        rows=12,
        cols=8,
        spacing=0.2,
        coords_file=str(tmp_path / "points.pt"),
    )

    expected = _quiet_run(_config(from_file), _blob())["p"]
    actual = _quiet_run(_config(_sensor_grid()), _blob())["p"]

    assert expected["filtered"].abs().sum() > 0
    assert torch.allclose(actual["filtered"], expected["filtered"], atol=1e-3)


@pytest.mark.parametrize(
    "grid, expected",
    [
        (dict(density=10.0), "density"),
        (dict(coords_file="points.csv"), "coords_file"),
        (dict(rows=1), "width and a height"),
        (dict(seed=3), "takes no seed"),
    ],
)
def test_sensor_settings_that_conflict_with_a_layout_are_errors(grid, expected):
    with pytest.raises(ValueError, match=expected):
        build_grid(_sensor_grid(**grid))


def test_the_grid_seed_reaches_a_random_sensor_layout():
    a = build_grid(_sensor_grid("test_scatter", seed=5)).get_all_coordinates()
    b = build_grid(_sensor_grid("test_scatter", seed=5)).get_all_coordinates()

    assert torch.equal(a, b)


@pytest.mark.parametrize("use", ["sensor", "neuron"])
def test_a_layout_that_is_not_installed_is_named_in_the_error(use):
    config = (
        _config(_sensor_grid("gone_layout"))
        if use == "sensor"
        else _config(neuron_arrangement="gone_layout")
    )
    with pytest.raises(ValueError, match="gone_layout"):
        SimulationEngine(config)


# ------------------------------------------------- CLI, older pipeline, batch


def _raw(config):
    """A config as it is read from YAML."""
    return config.to_dict()


def test_validate_accepts_registered_layouts_for_neurons_and_sensors():
    config = _custom_neurons(grid=_sensor_grid())

    assert validate_config(_raw(config))


@pytest.mark.parametrize(
    "section, key, value, expected",
    [
        ("populations", "neuron_arrangement", "demo_spiral", "Unknown layout"),
        ("populations", "neuron_layout_params", {"m": 1}, "has no setting 'm'"),
        ("grids", "arrangement", "demo_spiral", "Unknown layout"),
        ("grids", "layout_params", {"m": 1}, "has no setting 'm'"),
    ],
)
def test_validate_rejects_a_layout_a_run_could_not_use(
    section, key, value, expected, capsys
):
    raw = _raw(_custom_neurons(grid=_sensor_grid()))
    raw[section][0][key] = value
    if key.endswith("arrangement"):
        raw[section][0].pop(
            "layout_params" if section == "grids" else "neuron_layout_params", None
        )

    assert not validate_config(raw)
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize("grid", [None, _sensor_grid()], ids=["built-in", "plugin"])
def test_validate_checks_every_dsl_population_whatever_the_grids(grid, capsys):
    raw = _raw(_config(grid))
    first = dict(raw["populations"][0], name="dsl_first", neuron_model="DSL (Custom)")
    last = dict(raw["populations"][0], name="last")
    raw["populations"] = [first, last]

    assert not validate_config(raw)
    err = capsys.readouterr().err
    assert err.count("DSL neuron requires dsl_config") == 1
    assert "'dsl_first'" in err


def test_validate_accepts_a_config_with_no_populations():
    raw = _raw(_config(_sensor_grid()))
    raw["populations"] = []

    validate_config(raw)  # must not raise NameError


@pytest.mark.parametrize("use", ["sensor", "neuron"])
def test_the_older_pipeline_refuses_a_plugin_layout_by_name(use):
    config = _config(_sensor_grid()) if use == "sensor" else _custom_neurons()
    with pytest.raises(ValueError, match="older pipeline.*test_"):
        GeneralizedTactileEncodingPipeline.from_config(_raw(config))


def _two_grid_batch(tmp_path, arrangement):
    second = GridConfig(name="b", arrangement=arrangement, rows=12, cols=8, spacing=0.2)
    config = _config(neurons_per_row=3)
    config.grids.append(second)
    return {
        "base_config": _raw(config),
        "batch": {
            "output_dir": str(tmp_path / arrangement),
            "stimuli": [
                {
                    "type": "gaussian",
                    "parameters": {"amplitude": [10.0]},
                    "repetitions": 1,
                }
            ],
        },
    }


def test_batch_runs_with_a_plugin_grid_beside_another_grid(tmp_path):
    frames = {}
    for arrangement in ("grid", "test_lattice"):
        executor = BatchExecutor(_two_grid_batch(tmp_path, arrangement))
        executor.execute()
        bundle = load_bundle(executor.batch_root / "stim_0000")
        frames[arrangement] = bundle.stimulus

    assert torch.equal(frames["test_lattice"], frames["grid"])


def test_batch_runs_two_grids_as_a_saved_config_writes_them(tmp_path):
    """A saved config says ``density: null``; ``batch`` used to crash on two grids."""
    raw = _two_grid_batch(tmp_path, "grid")
    assert [g["density"] for g in raw["base_config"]["grids"]] == [None, None]

    executor = BatchExecutor(raw)
    executor.execute()

    assert load_bundle(executor.batch_root / "stim_0000").stimulus.abs().sum() > 0


# ------------------------------------------- unset seeds and the plugin check


class _IntSeed(Scatter):
    """As the scaffold writes it: an integer ``seed`` with a default."""

    def __init__(self, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), device="cpu", seed=0):
        super().__init__(xlim, ylim, device, seed=int(seed))


class _IgnoresSeed(Scatter):
    def points(self):
        low = torch.tensor([self.xlim[0], self.ylim[0]])
        high = torch.tensor([self.xlim[1], self.ylim[1]])
        return low + torch.rand(5, 2) * (high - low)


class _IgnoresArea(Diagonal):
    """Always fills the default square, whatever area it is given."""

    def points(self):
        x = torch.linspace(-1.0, 1.0, self.n)
        return torch.stack([x, x], dim=1)


class _Contract(Diagonal):
    """A layout that passes the whole grid contract."""

    def to_dict(self):
        return {**super().to_dict(), "n": self.n}

    @classmethod
    def from_config(cls, config):
        keys = ("xlim", "ylim", "device", "n")
        return cls(**{k: v for k, v in config.items() if k in keys})


def test_an_unset_seed_leaves_the_layouts_own_default():
    GRID_REGISTRY.register("test_int_seed", _IntSeed)
    try:
        a = build_layout_points("test_int_seed", *AREA)
        b = build_layout_points("test_int_seed", *AREA, seed=0)
    finally:
        GRID_REGISTRY._registry.pop("test_int_seed")

    assert torch.equal(a, b)


def test_the_plugin_check_passes_a_layout_that_fills_any_area():
    check_component("grid", _Contract, _Contract(n=4))


@pytest.mark.parametrize(
    "broken, expected",
    [(_IgnoresArea, "outside the area"), (_IgnoresSeed, "same seed")],
)
def test_the_plugin_check_fails_a_layout_a_config_could_not_trust(broken, expected):
    class Checked(broken):
        def to_dict(self):
            own = {"n": self.n} if hasattr(self, "n") else {"seed": self.seed}
            return {**BaseGrid.to_dict(self), **own}

        @classmethod
        def from_config(cls, config):
            return cls(**{k: v for k, v in config.items() if k != "type"})

    Checked.__init__ = broken.__init__
    with pytest.raises(AssertionError, match=expected):
        check_component("grid", Checked)


def test_a_scaffolded_grid_can_be_named_in_a_config(tmp_path, monkeypatch):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    command = [sys.executable, "-m", "sensoryforge.cli"]
    subprocess.run(
        command + ["new-component", "grid", "demo_cfg", "--dest", str(tmp_path)],
        check=True,
        capture_output=True,
        env=env,
    )
    package = next(tmp_path.glob("*/pyproject.toml")).parent
    monkeypatch.syspath_prepend(str(package))
    load_plugin_import_paths(["sensoryforge_demo_cfg.component:register"])
    try:
        grid = build_grid(_sensor_grid("demo_cfg"))
        points = grid.get_all_coordinates()
        check_component("grid", GRID_REGISTRY.get_class("demo_cfg"))
    finally:
        GRID_REGISTRY._registry.pop("demo_cfg", None)

    assert points.shape == (100, 2)


# ---------------------------------------------------------------- end to end

REPO = Path(__file__).resolve().parents[2]

_WORLD = """
world:
  name: demo
  defaults:
    delay_ms:   {range: [0, 5]}
    touch_ms:   {range: [5, 10]}
    hold_ms:    {range: [10, 20]}
    release_ms: {range: [5, 10]}
    amplitude:  {range: [0.5, 1.0]}
    x_mm: {range: [-0.3, 0.3]}
    y_mm: {range: [-0.3, 0.3]}
  classes:
    dots:
      weight: 1.0
      layer: {shape: {kind: gaussian}}
      axes: {sigma_mm: {range: [0.15, 0.45]}}
"""


@pytest.fixture
def spiral_package(tmp_path):
    """The example spiral as an importable plugin outside SensoryForge."""
    package = tmp_path / "pkg" / "demo_spiral_plugin"
    package.mkdir(parents=True)
    example = (REPO / "docs/examples/grid_arrangement_plugin.py").read_text()
    (package / "__init__.py").write_text(
        example
        + "\n\ndef register():\n"
        + '    GRID_REGISTRY.register("demo_pkg_spiral", SpiralArrangement)\n'
    )
    return tmp_path / "pkg"


def _spiral_yaml(use):
    grid = {"name": "skin", "rows": 20, "cols": 20, "spacing": 0.1}
    population = {
        "name": "SA",
        "neurons_per_row": 10,
        "innervation_method": "gaussian",
        "filter_method": "sa",
        "input_gain": 50.0,
        "seed": 1,
    }
    if use in ("sensor", "both"):
        grid.update(arrangement="demo_pkg_spiral", layout_params={"pitch_mm": 0.08})
    if use in ("neuron", "both"):
        population.update(
            neuron_arrangement="demo_pkg_spiral",
            neuron_layout_params={"pitch_mm": 0.25},
        )
    return {
        "plugins": ["demo_spiral_plugin:register"],
        "grids": [grid],
        "populations": [population],
        "stimulus": {"type": "gaussian", "amplitude": 30.0, "sigma": 0.4},
        "simulation": {"device": "cpu", "dt_ms": 1.0},
    }


def _cli(args, cwd, plugin_dir):
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO), str(plugin_dir)]),
        "OMP_NUM_THREADS": "1",
    }
    done = subprocess.run(
        [sys.executable, "-m", "sensoryforge.cli", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return done


@pytest.mark.parametrize("use", ["sensor", "neuron", "both"])
def test_run_with_a_packaged_layout_and_rebuild_the_same_spikes(
    use, tmp_path, spiral_package, monkeypatch
):
    (tmp_path / "config.yml").write_text(yaml.safe_dump(_spiral_yaml(use)))
    _cli(["validate", "config.yml"], tmp_path, spiral_package)
    _cli(
        ["run", "config.yml", "--duration", "60", "--bundle", "out"],
        tmp_path,
        spiral_package,
    )

    bundle = load_bundle(tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "config.json").read_text())["config"]
    monkeypatch.syspath_prepend(str(spiral_package))
    load_plugin_import_paths(["demo_spiral_plugin:register"])
    try:
        engine = SimulationEngine(SensoryForgeConfig.from_dict(saved))
        again = engine.run(bundle.stimulus.unsqueeze(0), return_intermediates=True)
    finally:
        GRID_REGISTRY._registry.pop("demo_pkg_spiral", None)
        sys.modules.pop("demo_spiral_plugin", None)

    stored = bundle.populations["SA"]["spikes"]
    assert stored.sum() > 0
    assert torch.equal(again["SA"]["spikes"].reshape(stored.shape), stored)
    expected_neurons = 100 if use == "sensor" else stored.shape[-1]
    assert stored.shape[-1] == expected_neurons
    if use != "sensor":
        assert stored.shape[-1] != 100


def test_evaluate_checks_do_not_error_on_a_custom_layout(tmp_path, spiral_package):
    (tmp_path / "config.yml").write_text(yaml.safe_dump(_spiral_yaml("both")))
    (tmp_path / "world.yml").write_text(_WORLD)

    _cli(
        [
            "evaluate",
            "run",
            "world.yml",
            "--draws",
            "2",
            "--duration-ms",
            "50",
            "--seed",
            "7",
            "--config",
            "config.yml",
            "--out",
            "report",
        ],
        tmp_path,
        spiral_package,
    )

    found = next((tmp_path / "report").rglob("checks.json"))
    checks = json.loads(found.read_text())["checks"]
    assert checks
    errors = [(c["id"], c["note"]) for c in checks if c["status"] == "error"]
    assert not errors
    assert not [c["id"] for c in checks if c["status"] == "fail"]
