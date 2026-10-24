"""``sensoryforge describe``, the JSON Schemas and ``validate --json`` (roadmap B3).

The description must equal the registries it is generated from, so each test builds the
expected value from the registry itself, never from a list typed here.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sensoryforge import describe as dsc
from sensoryforge.registry import FILTER_REGISTRY
from sensoryforge.world import dataset, kernel, kinds
from sensoryforge.world import distributions as dist

REPO = Path(__file__).resolve().parents[2]
WORLDS = REPO / "tests" / "fixtures" / "worlds"


def _sf(*args, cwd=REPO):
    return subprocess.run(
        [sys.executable, "-m", "sensoryforge.cli", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
    )


# ------------------------------------------------------------------ describe


def test_components_match_the_registries():
    out = dsc.describe()["components"]
    registries = dsc._registries()
    assert set(out) == set(registries)
    for kind, registry in registries.items():
        assert list(out[kind]) == registry.list_registered()
        for name, entry in out[kind].items():
            expected = [dsc.spec_to_dict(s) for s in registry.get_param_spec(name)]
            assert entry["params"] == expected, (kind, name)
            cls = registry.get_class(name)
            assert entry["class"] == f"{cls.__module__}.{cls.__qualname__}"


def test_every_param_spec_field_is_carried():
    entry = dsc.describe()["components"]["neuron"]["AdEx"]
    first = entry["params"][0]
    assert set(first) == {
        "name",
        "label",
        "dtype",
        "default",
        "min",
        "max",
        "step",
        "unit",
        "choices",
        "tooltip",
        "help",
        "group",
        "advanced",
    }
    assert first["unit"] == "mV"


def test_component_without_a_param_spec_still_lists_its_constructor():
    out = dsc.describe()["components"]
    bare = [e for kind in out.values() for e in kind.values() if not e["params"]]
    assert bare, "the fixture assumes some built-in declares no ParamSpec"
    assert all(isinstance(e["init_params"], list) for e in bare)
    assert any(e["init_params"] for e in bare)


def test_a_registered_plugin_component_appears_and_a_removed_one_goes():
    class Ghost:
        """A throwaway filter."""

        def __init__(self, tau: float = 3.0, *, mode: str = "x"):
            self.tau = tau

    FILTER_REGISTRY.register("zz_ghost_filter", Ghost)
    try:
        entry = dsc.describe()["components"]["filter"]["zz_ghost_filter"]
        assert entry["summary"] == "A throwaway filter."
        assert entry["params"] == []
        assert entry["init_params"] == [
            {"name": "tau", "required": False, "default": 3.0},
            {"name": "mode", "required": False, "default": "x"},
        ]
    finally:
        FILTER_REGISTRY._registry.pop("zz_ghost_filter")
    assert "zz_ghost_filter" not in dsc.describe()["components"]["filter"]


def test_world_language_matches_the_world_registries():
    world = dsc.describe()["world"]
    assert list(world["shapes"]) == sorted(kernel.SHAPE_KINDS)
    assert list(world["patterns"]) == sorted(kernel.PATTERN_KINDS)
    assert list(world["modulations"]) == sorted(kernel.MODULATION_KINDS)
    assert list(world["distributions"]) == sorted(dist.DISTRIBUTIONS)
    assert list(world["class_kinds"]) == sorted(kinds.CLASS_KINDS)
    assert world["split_kinds"] == dataset.SPLIT_KINDS
    for name, shape in world["shapes"].items():
        specs = kernel.SHAPE_KINDS[name].specs
        assert [p["name"] for p in shape["params"]] == [s.name for s in specs]
        assert shape["indenter"] == kernel.SHAPE_KINDS[name].indenter


def test_a_registered_world_shape_appears():
    kernel.register_shape(
        "zz_ghost",
        lambda x, y, p: x * 0,
        [kernel.ParamSpec("amplitude", dtype="float", default=1.0)],
    )
    try:
        assert "zz_ghost" in dsc.describe()["world"]["shapes"]
        assert (
            "zz_ghost"
            in dsc.world_schema()["oneOf"][0]["properties"]["classes"][
                "additionalProperties"
            ]["properties"]["layer"]["properties"]["shape"]["properties"]["kind"][
                "examples"
            ]
        )
    finally:
        kernel.SHAPE_KINDS.pop("zz_ghost")


def test_recipes_match_the_presets():
    from sensoryforge.presets import list_presets, load_preset

    recipes = dsc.describe()["recipes"]
    assert list(recipes) == list_presets()
    for name, recipe in recipes.items():
        cfg = load_preset(name)
        assert len(recipe["grids"]) == len(cfg["grids"])
        assert [p["name"] for p in recipe["populations"]] == [
            p["name"] for p in cfg["populations"]
        ]


def test_evaluation_seams_match_their_registries():
    from sensoryforge.evaluate.checks.registry import CHECK_PACK_REGISTRY
    from sensoryforge.evaluate.registry import (
        DECLARED_ADAPTER_REGISTRY,
        METRIC_PACK_REGISTRY,
    )

    ev = dsc.describe()["evaluation"]
    assert list(ev["metric_packs"]) == METRIC_PACK_REGISTRY.list_registered()
    assert list(ev["check_packs"]) == CHECK_PACK_REGISTRY.list_registered()
    assert list(ev["declared_adapters"]) == DECLARED_ADAPTER_REGISTRY.list_registered()


def test_description_is_strict_json():
    data = dsc.describe()
    assert json.loads(json.dumps(data, allow_nan=False)) == data


def test_cli_describe_json_section():
    done = _sf("describe", "--json", "--section", "recipes")
    assert done.returncode == 0, done.stderr
    data = json.loads(done.stdout)
    assert set(data) == {"format", "recipes"}
    bad = _sf("describe", "--section", "nope")
    assert bad.returncode == 1 and "unknown section" in bad.stderr


# ------------------------------------------------------------------- schemas


@pytest.mark.parametrize("name", sorted(dsc.SCHEMAS))
def test_committed_schema_is_current(name):
    path = REPO / "schemas" / f"{name}.schema.json"
    assert path.read_text(encoding="utf-8") == dsc.schema_text(
        name
    ), "run: python scripts/build_schemas.py"


@pytest.mark.parametrize("name", sorted(dsc.SCHEMAS))
def test_schemas_are_valid_json_schema(name):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(dsc.SCHEMAS[name]())


def test_shipped_configs_pass_the_config_schema():
    jsonschema = pytest.importorskip("jsonschema")
    from sensoryforge.presets import list_presets, load_preset

    validator = jsonschema.Draft202012Validator(dsc.config_schema())
    docs = {n: load_preset(n) for n in list_presets()}
    docs["examples/canonical_config.yml"] = yaml.safe_load(
        (REPO / "examples" / "canonical_config.yml").read_text(encoding="utf-8")
    )
    for name, doc in docs.items():
        errors = [e.message for e in validator.iter_errors(doc)]
        assert not errors, (name, errors)


def test_config_schema_rejects_a_wrong_type():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(dsc.config_schema())
    bad = {"grids": [{"name": "g", "rows": "ten"}], "populations": []}
    assert [e.message for e in validator.iter_errors(bad)]


def test_fixture_worlds_pass_the_world_schema():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(dsc.world_schema())
    for name in ("tactile_small", "every_element_v1_1", "elements_v1_2"):
        doc = yaml.safe_load((WORLDS / f"{name}.yml").read_text(encoding="utf-8"))
        errors = [e.message for e in validator.iter_errors(doc)]
        assert not errors, (name, errors[:3])


def test_fixture_datasets_pass_the_dataset_schema():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(dsc.dataset_schema())
    for name in ("dataset_small", "dataset_v1_2"):
        doc = yaml.safe_load((WORLDS / f"{name}.yml").read_text(encoding="utf-8"))
        errors = [e.message for e in validator.iter_errors(doc)]
        assert not errors, (name, errors[:3])


def test_world_schema_rejects_an_unknown_top_key():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(dsc.world_schema())
    assert [e for e in validator.iter_errors({"world": {"clases": {}}})]


# ------------------------------------------------------------------ validate


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_valid_config_reports_valid():
    report = dsc.validate_file(str(REPO / "examples" / "canonical_config.yml"))
    assert report["kind"] == "config" and report["valid"] is True
    assert report["errors"] == [] and report["format"] == dsc.REPORT_FORMAT


def test_config_typo_has_a_location_and_a_fix(tmp_path):
    path = _write(
        tmp_path,
        "c.yml",
        "grids:\n  - {name: g, arrangment: grid, rows: 10, cols: 10}\n"
        "populations:\n  - {name: p, neuron_model: Izhikevic, target_grid: g}\n",
    )
    report = dsc.validate_file(path)
    assert not report["valid"]
    (error,) = [e for e in report["errors"] if e["path"] == "populations.0.neuron"]
    assert "Izhikevich" in error["fix"]
    (warning,) = report["warnings"]
    assert warning["path"] == "grids.0.arrangment"
    assert warning["fix"] == "did you mean 'arrangement'?"


def test_world_typo_is_located_and_suggested(tmp_path):
    path = _write(
        tmp_path,
        "w.yml",
        "world:\n  classes:\n    a:\n      layer:\n        shape: {kind: gausian}\n",
    )
    report = dsc.validate_file(path)
    assert report["kind"] == "world" and not report["valid"]
    (error,) = report["errors"]
    assert error["path"] == "world.classes.a.layer.shape"
    assert "did you mean 'gaussian'" in error["fix"]


def test_valid_world_and_dataset_fixtures():
    assert dsc.validate_file(str(WORLDS / "tactile_small.yml"))["valid"]
    report = dsc.validate_file(str(WORLDS / "dataset_small.yml"))
    assert report["kind"] == "dataset" and report["valid"], report["errors"]


def test_yaml_error_names_the_line(tmp_path):
    report = dsc.validate_file(_write(tmp_path, "y.yml", "grids: [unclosed\n"))
    (error,) = report["errors"]
    assert error["source"] == "yaml" and error["line"] == 2


def test_unreadable_and_unrecognised_files(tmp_path):
    missing = dsc.validate_file(str(tmp_path / "none.yml"))
    assert not missing["valid"] and "cannot read" in missing["errors"][0]["message"]
    odd = dsc.validate_file(_write(tmp_path, "o.yml", "colour: blue\n"))
    assert odd["kind"] == "unknown" and "--kind" in odd["errors"][0]["fix"]
    forced = dsc.validate_file(_write(tmp_path, "o2.yml", "colour: blue\n"), "world")
    assert forced["kind"] == "world" and not forced["valid"]


def test_text_number_hint_becomes_the_fix(tmp_path):
    assert "3.0e-1" in dsc.suggest_fix(
        "x: must be numbers, got '3e-1' (YAML reads a number with an exponent "
        "but no decimal point as text: write 3.0e-1, not 3e-1)"
    )


def test_no_fix_is_invented():
    assert dsc.suggest_fix("the tensor has the wrong shape") is None


def test_cli_validate_json_exit_codes(tmp_path):
    good = _sf("validate", "--json", "examples/canonical_config.yml")
    assert good.returncode == 0 and json.loads(good.stdout)["valid"] is True
    bad = _write(tmp_path, "b.yml", "world:\n  clases: {}\n")
    done = _sf("validate", "--json", bad)
    assert done.returncode == 1
    assert json.loads(done.stdout)["errors"][0]["path"] == "world"


def test_validate_without_json_is_unchanged():
    done = _sf("validate", "examples/canonical_config.yml")
    assert done.returncode == 0 and "Configuration is valid" in done.stdout
