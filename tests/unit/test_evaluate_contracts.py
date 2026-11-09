"""The contract checks for the evaluation seams (roadmap C2d).

The built-in packs and the tactile adapter must pass; a pack or adapter that breaks a
rule must fail with a message that names the rule. The bad ones are put in the
registries for one test only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from sensoryforge.evaluate.declared import DeclaredAdapter
from sensoryforge.evaluate.registry import (
    DECLARED_ADAPTER_REGISTRY,
    METRIC_PACK_REGISTRY,
)
from sensoryforge.testing.evaluate_contracts import (
    available_evaluation_kinds,
    check_evaluation_component,
)
from tests.fixtures.eval_packs import Counts, Drift, Random, Raises, Stray, Undefined
from sensoryforge.world.dataset import build_dataset, load_dataset, write_dataset

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tactile_declared_only(tmp_path_factory):
    """A tactile data set with a manifest and ``dataset.json`` and no bundles."""
    root = tmp_path_factory.mktemp("tactile_decl")
    raw = {
        "dataset": {
            "name": "t",
            "world": str(REPO / "tests" / "fixtures" / "worlds" / "tactile_small.yml"),
            "seed": 3,
            "duration_ms": 150,
            "splits": {"train": {"n": 12}},
        }
    }
    (root / "spec.yml").write_text(yaml.safe_dump(raw))
    spec = load_dataset(root / "spec.yml")
    write_dataset(spec, build_dataset(spec), root / "data")
    return root / "data"


def _put(monkeypatch, registry, name, cls):
    monkeypatch.setitem(registry._registry, name.casefold(), (name, cls, None))


def test_the_kinds():
    assert available_evaluation_kinds() == ["declared_adapter", "metric_pack"]
    with pytest.raises(ValueError, match="Unknown evaluation kind"):
        check_evaluation_component("frame_renderer", "grid")


def test_built_in_packs_pass_on_the_toy_data_set(toy_dataset):
    check_evaluation_component("metric_pack", "core", example=toy_dataset.data)
    check_evaluation_component("metric_pack", "tactile", example=toy_dataset.data)
    check_evaluation_component("metric_pack", "CORE")  # no data: the static rules only


def test_the_tactile_adapter_passes(tactile_declared_only):
    check_evaluation_component("declared_adapter", "tactile")
    check_evaluation_component(
        "declared_adapter", "tactile", example=tactile_declared_only
    )


def test_a_good_third_party_pack_passes(monkeypatch, toy_dataset):
    _put(monkeypatch, METRIC_PACK_REGISTRY, "counts", Counts)
    check_evaluation_component("metric_pack", "counts", example=toy_dataset.data)


def test_unregistered_names_and_missing_folders_are_value_errors(monkeypatch):
    with pytest.raises(ValueError, match="not registered"):
        check_evaluation_component("metric_pack", "nope")
    with pytest.raises(ValueError, match="not registered"):
        check_evaluation_component("declared_adapter", "nope")
    _put(monkeypatch, METRIC_PACK_REGISTRY, "counts", Counts)
    with pytest.raises(ValueError, match="data-set folder"):
        check_evaluation_component("metric_pack", "counts", example="/no/such/dir")


def _bad(monkeypatch, toy_dataset, cls, name, match, example=True):
    _put(monkeypatch, METRIC_PACK_REGISTRY, name, cls)
    with pytest.raises(AssertionError, match=match):
        check_evaluation_component(
            "metric_pack", name, example=toy_dataset.data if example else None
        )


def test_a_pack_whose_name_differs_from_its_registration_fails(
    monkeypatch, toy_dataset
):
    _bad(monkeypatch, toy_dataset, Counts, "other", "not the registered", False)


def test_an_id_outside_the_packs_namespace_fails(monkeypatch, toy_dataset):
    _bad(monkeypatch, toy_dataset, Stray, "stray", "must be <pack>", False)


def test_a_value_for_an_undefined_id_fails(monkeypatch, toy_dataset):
    _bad(monkeypatch, toy_dataset, Undefined, "undefined", "undefined ids")


def test_a_pack_that_raises_fails(monkeypatch, toy_dataset):
    _bad(monkeypatch, toy_dataset, Raises, "raises", "measure\\(\\) failed")


def test_a_pack_that_is_not_deterministic_fails(monkeypatch, toy_dataset):
    _bad(monkeypatch, toy_dataset, Random, "random", "not deterministic")


def test_a_pack_that_changes_its_parameters_on_a_round_trip_fails(
    monkeypatch, toy_dataset
):
    _bad(monkeypatch, toy_dataset, Drift, "drift", "changed the parameters", False)


def test_a_pack_that_imports_the_world_fails(monkeypatch, toy_dataset, tmp_path):
    source = (
        "from sensoryforge.world import World  # noqa\n"
        "from sensoryforge.evaluate.pack import MetricPack, metric_definition\n\n\n"
        "class Peeks(MetricPack):\n"
        "    name = 'peeks'\n\n"
        "    def metrics(self, layout, units, dt=None):\n"
        "        return [metric_definition('peeks.frames.n', None, scope='entry')]\n"
    )
    path = tmp_path / "peeks_module.py"
    path.write_text(source)
    import importlib.util

    spec = importlib.util.spec_from_file_location("peeks_module", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "peeks_module", module)
    spec.loader.exec_module(module)
    _bad(monkeypatch, toy_dataset, module.Peeks, "peeks", "blind wall", False)


class _Adapter(DeclaredAdapter):
    name = "adapter"
    version = "1"

    @classmethod
    def accepts(cls, dataset_meta):
        return False


def test_an_adapter_that_does_not_accept_the_data_set_fails(
    monkeypatch, tactile_declared_only
):
    _put(monkeypatch, DECLARED_ADAPTER_REGISTRY, "adapter", _Adapter)
    with pytest.raises(AssertionError, match="accepts"):
        check_evaluation_component(
            "declared_adapter", "adapter", example=tactile_declared_only
        )


def test_an_adapter_with_a_wrong_name_fails(monkeypatch):
    _put(monkeypatch, DECLARED_ADAPTER_REGISTRY, "renamed", _Adapter)
    with pytest.raises(AssertionError, match="not the registered"):
        check_evaluation_component("declared_adapter", "renamed")


def test_an_adapter_that_declares_an_id_outside_its_catalogue_fails(
    monkeypatch, tactile_declared_only
):
    from sensoryforge.evaluate.declared import Declaration
    from sensoryforge.evaluate.tactile.declared import TactileAdapter

    class Extra(TactileAdapter):
        name = "extra"

        def declare(self, row, world):
            out = super().declare(row, world)
            return Declaration(
                out.values + [{"id": "core.frames.zzz", "value": 1.0}], out.omitted
            )

    _put(monkeypatch, DECLARED_ADAPTER_REGISTRY, "extra", Extra)
    with pytest.raises(AssertionError, match="not in the catalogue"):
        check_evaluation_component(
            "declared_adapter", "extra", example=tactile_declared_only
        )
