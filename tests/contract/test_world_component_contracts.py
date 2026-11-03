"""Contract tests over every shipped world shape, pattern, modulation, distribution
and class kind (C4).

``tests/contract/test_component_contracts.py`` sweeps the component registries; this
sweeps the world registries (``sensoryforge/world/``) with
``sensoryforge.testing.world_contracts.check_world_component``. A built-in that cannot
pass is fixed, or recorded as an open ledger item -- the check is not weakened.

Only entries sensoryforge itself defines are swept, so a plugin or a test that
registered something else earlier in the session cannot fail this file; the guard
tests below fail instead when a built-in is added without its example.
"""

from pathlib import Path

import pytest
import yaml

from sensoryforge.testing.world_contracts import check_world_component
from sensoryforge.world import distributions, kernel, kinds

WORLD_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "worlds" / "tactile_small.yml"
)


def _shipped(obj) -> bool:
    module = getattr(obj, "__module__", None) or type(obj).__module__
    return module.startswith("sensoryforge.")


SHAPES = sorted(n for n, k in kernel.SHAPE_KINDS.items() if _shipped(k.fn))
PATTERNS = sorted(n for n, k in kernel.PATTERN_KINDS.items() if _shipped(k.fn))
MODULATIONS = sorted(
    n for n, k in kernel.MODULATION_KINDS.items() if k.fn is None or _shipped(k.fn)
)
DISTRIBUTIONS = sorted(
    n for n, d in distributions.DISTRIBUTIONS.items() if _shipped(d.sample)
)
CLASS_KINDS = sorted(n for n, k in kinds.CLASS_KINDS.items() if _shipped(k))

#: The options an axis needs for each shipped distribution (``None``: none).
DISTRIBUTION_EXAMPLES = {
    "braille_cells": None,
    "biased_direction": {"travel_ratio": 2.0},
    "letter_text": None,
}


def _world_example():
    """A world with a class of every shipped class kind."""
    with open(WORLD_FIXTURE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_the_sweep_sees_the_shipped_entries():
    """A filter that selected nothing would pass every test below."""
    assert {"gaussian", "disc", "bar", "grating", "gabor"} <= set(SHAPES)
    assert {"self_affine", "dot_array", "curved_contact", "step_edge"} <= set(SHAPES)
    assert {"single", "grid", "list", "random", "braille"} <= set(PATTERNS)
    assert {"none", "sine", "pulses"} <= set(MODULATIONS)
    assert {"braille_cells", "biased_direction", "letter_text"} <= set(DISTRIBUTIONS)
    assert {"layered", "quiet"} == set(CLASS_KINDS)


def test_every_shipped_distribution_has_an_example():
    """A built-in distribution that needs options must say which in this file."""
    assert sorted(DISTRIBUTION_EXAMPLES) == DISTRIBUTIONS


def test_the_example_world_has_a_class_of_every_shipped_class_kind():
    kinds_used = {
        spec.get("kind", "layered")
        for spec in _world_example()["world"]["classes"].values()
    }
    assert kinds_used >= set(CLASS_KINDS)


@pytest.mark.parametrize("name", SHAPES)
def test_shape_contract(name):
    check_world_component("shape", name)


@pytest.mark.parametrize("name", PATTERNS)
def test_pattern_contract(name):
    check_world_component("pattern", name)


@pytest.mark.parametrize("name", MODULATIONS)
def test_modulation_contract(name):
    check_world_component("modulation", name)


@pytest.mark.parametrize("name", DISTRIBUTIONS)
def test_distribution_contract(name):
    check_world_component("distribution", name, DISTRIBUTION_EXAMPLES[name])


@pytest.mark.parametrize("name", CLASS_KINDS)
def test_class_kind_contract(name):
    check_world_component("class_kind", name, _world_example())
