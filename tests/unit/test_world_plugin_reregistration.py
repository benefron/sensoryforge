"""A world plugin whose ``register()`` runs more than once (F-5bd542f, C1).

Topic: plugin, world registries
Tests: this file
Pinned by: F-5bd542f
"""

import pytest

from sensoryforge.stimuli.base import ParamSpec
from sensoryforge.world import kernel
from sensoryforge.world.distributions import DISTRIBUTIONS, register_distribution
from sensoryforge.world.kinds import CLASS_KINDS, LayeredKind, register_class_kind


def _ring(x, y, params):
    return params["amplitude"] * (x * 0 + y * 0)


def _ring_other(x, y, params):
    return params["amplitude"] * (x * 0 + y * 0 + 1)


def _specs():
    return [ParamSpec("amplitude", dtype="float", default=1.0, unit="")]


def _sample(u, axis):
    return list(u)


def _sample_other(u, axis):
    return list(u)


def _plugin_register(shape_fn=_ring, sample=_sample, specs=None):
    """What a plugin's ``register()`` does: add one of each kind."""
    kernel.register_shape("c1_ring", shape_fn, specs or _specs())
    register_distribution("c1_dist", sample)


@pytest.fixture(autouse=True)
def _clean():
    yield
    kernel.SHAPE_KINDS.pop("c1_ring", None)
    DISTRIBUTIONS.pop("c1_dist", None)
    kernel.PATTERN_KINDS.pop("c1_pat", None)
    kernel.MODULATION_KINDS.pop("c1_mod", None)


def test_a_plugin_registering_twice_is_a_no_op():
    _plugin_register()
    shape = kernel.SHAPE_KINDS["c1_ring"]
    dist = DISTRIBUTIONS["c1_dist"]
    _plugin_register()
    assert kernel.SHAPE_KINDS["c1_ring"] is shape
    assert DISTRIBUTIONS["c1_dist"] is dist


def test_pattern_and_modulation_register_again_as_a_no_op():
    def pat(params):
        return [], []

    def mod(tc, params):
        return tc

    for _ in range(2):
        kernel.register_pattern("c1_pat", pat, _specs())
        kernel.register_modulation("c1_mod", mod, _specs())
    assert kernel.PATTERN_KINDS["c1_pat"].fn is pat
    assert kernel.MODULATION_KINDS["c1_mod"].fn is mod


def test_a_different_shape_function_still_conflicts():
    _plugin_register()
    with pytest.raises(ValueError, match="'c1_ring' is already registered"):
        _plugin_register(shape_fn=_ring_other)
    assert kernel.SHAPE_KINDS["c1_ring"].fn is _ring


def test_different_specs_or_flags_still_conflict():
    kernel.register_shape("c1_ring", _ring, _specs())
    other = _specs() + [ParamSpec("width_mm", dtype="float", default=1.0, unit="mm")]
    with pytest.raises(ValueError, match="already registered"):
        kernel.register_shape("c1_ring", _ring, other)
    with pytest.raises(ValueError, match="already registered"):
        kernel.register_shape("c1_ring", _ring, _specs(), unbounded=True)


def test_a_different_distribution_still_conflicts_and_replace_still_works():
    register_distribution("c1_dist", _sample)
    with pytest.raises(ValueError, match="'c1_dist' is already registered"):
        register_distribution("c1_dist", _sample_other)
    register_distribution("c1_dist", _sample_other, replace=True)
    assert DISTRIBUTIONS["c1_dist"].sample is _sample_other


def test_class_kind_registered_twice_by_a_plugin():
    before = CLASS_KINDS["layered"]
    register_class_kind(LayeredKind())
    assert CLASS_KINDS["layered"] is before


def test_the_plugin_loader_runs_a_world_plugin_twice_without_a_warning(
    tmp_path, monkeypatch, recwarn
):
    """The loader's ``module:register`` path, as a config's ``plugins:`` uses it."""
    (tmp_path / "c1_world_plugin.py").write_text(
        "from sensoryforge.stimuli.base import ParamSpec\n"
        "from sensoryforge.world.kernel import register_shape\n"
        "from sensoryforge.world.distributions import register_distribution\n"
        "def ring(x, y, params):\n"
        "    return params['amplitude'] * (x * 0 + y * 0)\n"
        "def sample(u, axis):\n"
        "    return list(u)\n"
        "def register():\n"
        "    register_shape('c1_ring', ring, [ParamSpec('amplitude', dtype='float',"
        " default=1.0, unit='')])\n"
        "    register_distribution('c1_dist', sample)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    from sensoryforge.plugins import load_plugin_import_paths

    path = "c1_world_plugin:register"
    assert load_plugin_import_paths([path]) == [path]
    assert load_plugin_import_paths([path]) == [path]
    assert not [w for w in recwarn if "already registered" in str(w.message)]
    assert "c1_ring" in kernel.SHAPE_KINDS
