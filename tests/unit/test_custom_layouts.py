"""A registered grid plugin can be named as a sensor layout or a neuron layout."""

import pytest
import torch

from sensoryforge.core.grid_base import BaseGrid
from sensoryforge.core.layouts import (
    BUILTIN_LAYOUTS,
    build_layout_points,
    custom_layout_names,
    layout_problem,
)
from sensoryforge.registry import GRID_REGISTRY
from sensoryforge.stimuli.base import ParamSpec


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

    def __init__(
        self, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), device="cpu", nx=12, ny=8
    ):
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

    def __init__(
        self, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), device="cpu", kind="shape"
    ):
        super().__init__(xlim, ylim, device)
        self.kind = kind

    def points(self):
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


for _name, _cls in {
    "test_diagonal": Diagonal,
    "test_scatter": Scatter,
    "test_lattice": Lattice,
    "test_broken": Broken,
    "test_needs_setting": NeedsSetting,
    "test_no_area": NoArea,
}.items():
    GRID_REGISTRY.register(_name, _cls)

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
        build_layout_points(
            "test_diagonal", *AREA, seed=3, require_seed_support=True
        )


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
