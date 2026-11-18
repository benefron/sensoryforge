"""The reference plugins in examples/plugins: each passes its own check, and all
three run in one data set (a ring shape in the world, a neuron and a filter in the
sensor), loaded as a plugin author's user would: by their register() functions."""

import sys
from pathlib import Path

import pytest

from sensoryforge.cli import cmd_batch, create_parser
from sensoryforge.io.bundle import load_bundle
from sensoryforge.plugins import load_plugin_import_paths
from sensoryforge.registry import FILTER_REGISTRY, NEURON_REGISTRY
from sensoryforge.world.kernel import SHAPE_KINDS
from sensoryforge.world.runner import read_batch_index

PLUGINS = Path(__file__).resolve().parents[2] / "examples" / "plugins"
PACKAGES = {
    "sensoryforge-ring-shape": "sensoryforge_ring_shape.shape:register",
    "sensoryforge-leaky-threshold": "sensoryforge_leaky_threshold.component:register",
    "sensoryforge-exp-smoother": "sensoryforge_exp_smoother.component:register",
}


@pytest.fixture
def loaded_plugins():
    paths = [str(PLUGINS / d) for d in PACKAGES]
    sys.path[:0] = paths
    try:
        assert load_plugin_import_paths(PACKAGES.values()) == list(PACKAGES.values())
        yield
    finally:
        for p in paths:
            sys.path.remove(p)


def test_the_three_plugins_run_in_a_data_set(loaded_plugins, tmp_path):
    assert "ring" in SHAPE_KINDS
    assert "leaky_threshold" in NEURON_REGISTRY.list_registered()
    assert "exp_smoother" in FILTER_REGISTRY.list_registered()
    out = tmp_path / "out"
    argv = [
        "batch",
        str(PLUGINS / "sensor.yml"),
        "--dataset",
        str(PLUGINS / "ring_dataset.yml"),
        "--output",
        str(out),
    ]
    assert cmd_batch(create_parser().parse_args(argv)) == 0
    rows = read_batch_index(out)
    assert len(rows) == 5 and {r["status"] for r in rows} == {"ok"}
    bundle = load_bundle(out / rows[0]["entry"])
    pop = bundle.meta["config_json"]["config"]["populations"][0]
    assert pop["neuron_model"] == "leaky_threshold"
    assert pop["filter_method"] == "exp_smoother"
    assert float(bundle.stimulus.abs().max()) > 0.0  # the ring drew something
