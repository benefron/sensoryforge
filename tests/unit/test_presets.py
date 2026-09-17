"""Tests for the shipped canonical-config presets (Phase 2, Wave K, K4).

A preset is data, not code -- a YAML fragment under sensoryforge/presets/,
read through importlib.resources (never a cwd-relative path). This file
loads every shipped preset and asserts it constructs a valid
SensoryForgeConfig, so a broken preset fails CI (per the K4 spec's own
"Done when").
"""

from __future__ import annotations

import pytest

from sensoryforge.config.schema import SensoryForgeConfig
from sensoryforge.presets import (
    PRESET_DIR,
    list_presets,
    load_preset,
    preset_description,
)


def test_list_presets_finds_both_shipped_presets():
    names = list_presets()
    assert "tactile_sa1_ra1" in names
    assert "tactile_stochastic_control" in names


def test_list_presets_is_sorted():
    names = list_presets()
    assert names == sorted(names)


@pytest.mark.parametrize("name", ["tactile_sa1_ra1", "tactile_stochastic_control"])
def test_preset_loads_and_builds_valid_config(name):
    data = load_preset(name)
    config = SensoryForgeConfig.from_dict(data)
    assert len(config.grids) == 1
    assert config.grids[0].rows == 80
    assert config.grids[0].cols == 80
    assert len(config.populations) == 2
    names = {p.name for p in config.populations}
    assert names == {"SA Population", "RA Population"}


def test_tactile_sa1_ra1_uses_template_builder_with_d_040():
    data = load_preset("tactile_sa1_ra1")
    config = SensoryForgeConfig.from_dict(data)
    for pop in config.populations:
        assert pop.innervation_method == "template"
        assert pop.resolvable_distance_mm == pytest.approx(0.40)


def test_tactile_stochastic_control_uses_gaussian_without_distance_weights():
    """D-019: the named control arm is gaussian/use_distance_weights=false,
    not a separate 'gaussian_stochastic' registered builder name."""
    data = load_preset("tactile_stochastic_control")
    config = SensoryForgeConfig.from_dict(data)
    for pop in config.populations:
        assert pop.innervation_method == "gaussian"
        assert pop.use_distance_weights is False


def test_ra_population_has_k3_2_0_via_filter_defaults():
    """Neither preset overrides k3; the resolver-owned default (D-Q1) is
    2.0. This is a smoke check that filter_params is left empty so the
    resolver applies it, not a re-test of the resolver itself."""
    for name in ("tactile_sa1_ra1", "tactile_stochastic_control"):
        data = load_preset(name)
        config = SensoryForgeConfig.from_dict(data)
        ra = next(p for p in config.populations if p.neuron_type == "RA")
        assert ra.filter_method == "ra"
        assert ra.filter_params in ({}, None)


def test_load_unknown_preset_raises_value_error_listing_names():
    with pytest.raises(ValueError, match="tactile_sa1_ra1"):
        load_preset("not_a_real_preset_name")


def test_preset_dir_survives_importlib_resources_and_lists_yml_files():
    entries = [e.name for e in PRESET_DIR.iterdir()]
    assert "tactile_sa1_ra1.yml" in entries
    assert "tactile_stochastic_control.yml" in entries


def test_preset_description_nonempty_for_shipped_presets():
    for name in list_presets():
        assert preset_description(name), f"{name} has no description"
