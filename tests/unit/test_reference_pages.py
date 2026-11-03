"""The generated reference pages in ``docs/reference/`` (roadmap D5).

``scripts/build_reference.py`` renders the pages from ``sensoryforge describe --json``.
These tests fail when a committed page differs from what the script writes (stale),
when something the registries hold is missing from the pages (incomplete), when a page
carries a ledger id (D-41f73ea), and when a link into a page names an anchor it lacks.

The registries are read in a fresh interpreter, once: another test that registers a
component and forgets to remove it cannot change what the pages must hold. The list of
what is registered is taken from the registry objects, not from ``describe``'s own
dict, so a registry that ``describe`` forgets is still caught.
"""

import ast
import copy
import dataclasses
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sensoryforge.testing import file_headers

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "build_reference.py"
DOCS = REPO / "docs"
REGENERATE = "python scripts/build_reference.py"

#: The registry objects in ``sensoryforge.registry`` and the section id of each.
REGISTRY_KINDS = {
    "NEURON_REGISTRY": "neuron",
    "FILTER_REGISTRY": "filter",
    "STIMULUS_REGISTRY": "stimulus",
    "SOLVER_REGISTRY": "solver",
    "GRID_REGISTRY": "grid",
    "INNERVATION_REGISTRY": "innervation",
    "PROCESSING_REGISTRY": "processing",
}

#: Run in a fresh interpreter: the description, and what is registered, read from the
#: registry objects themselves.
_PROBE = """
import json
import sensoryforge.describe as dsc
import sensoryforge.registry as registry
from sensoryforge.register_components import register_all

register_all()
data = dsc.describe()
import sensoryforge.evaluate.declared  # noqa: F401  (the adapters register on import)
from sensoryforge.evaluate.checks.registry import CHECK_PACK_REGISTRY
from sensoryforge.evaluate.registry import (
    DECLARED_ADAPTER_REGISTRY,
    METRIC_PACK_REGISTRY,
)
from sensoryforge.presets import list_presets
from sensoryforge.world import distributions, kernel, kinds

registered = {
    "registries": {
        name: obj.list_registered()
        for name, obj in vars(registry).items()
        if isinstance(obj, registry.ComponentRegistry)
    },
    "shape": sorted(kernel.SHAPE_KINDS),
    "pattern": sorted(kernel.PATTERN_KINDS),
    "modulation": sorted(kernel.MODULATION_KINDS),
    "distribution": sorted(distributions.DISTRIBUTIONS),
    "class-kind": sorted(kinds.CLASS_KINDS),
    "metric-pack": METRIC_PACK_REGISTRY.list_registered(),
    "check-pack": CHECK_PACK_REGISTRY.list_registered(),
    "declared-adapter": DECLARED_ADAPTER_REGISTRY.list_registered(),
    "recipe": list(list_presets()),
}
print(json.dumps({"describe": data, "registered": registered}))
"""


def _load_script():
    spec = importlib.util.spec_from_file_location("build_reference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_script()


@pytest.fixture(scope="module")
def probe():
    done = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True
    )
    return json.loads(done.stdout)


@pytest.fixture(scope="module")
def described(probe):
    return probe["describe"]


@pytest.fixture(scope="module")
def pages(described):
    return build.render(described)


@pytest.fixture(scope="module")
def committed():
    out = {}
    for name in build.PAGES:
        path = DOCS / "reference" / name
        assert path.is_file(), f"{path} is missing; run `{REGENERATE}`"
        out[name] = path.read_text(encoding="utf-8")
    return out


def _registered(probe):
    """``{(section id, name)}`` of everything the registry objects hold."""
    expected = set()
    for attr, names in probe["registered"]["registries"].items():
        assert attr in REGISTRY_KINDS, (
            f"sensoryforge.registry has a new registry, {attr}: add it to describe.py, "
            f"to scripts/build_reference.py and to REGISTRY_KINDS in this test"
        )
        expected |= {(REGISTRY_KINDS[attr], n) for n in names}
    for kind in (
        "shape",
        "pattern",
        "modulation",
        "distribution",
        "class-kind",
        "metric-pack",
        "check-pack",
        "declared-adapter",
        "recipe",
    ):
        expected |= {(kind, n) for n in probe["registered"][kind]}
    return expected


# ----------------------------------------------------------------------- stale pages


@pytest.mark.parametrize("name", sorted(build.PAGES))
def test_a_committed_page_equals_what_the_script_writes(name, pages, committed):
    assert committed[name] == pages[name], (
        f"docs/reference/{name} is behind the code; run `{REGENERATE}` and commit "
        "the pages"
    )


def test_the_script_writes_exactly_the_pages_it_names(pages):
    assert set(pages) == set(build.PAGES)


# ------------------------------------------------------------------ missing entries


def test_every_registered_component_is_in_the_committed_reference(probe, committed):
    absent = build.missing(committed, _registered(probe))
    assert not absent, (
        "registered but not in docs/reference/: "
        + "; ".join(absent)
        + f". Run `{REGENERATE}`; if it does not list them, describe.py or the "
        "script drops them"
    )


def test_the_pages_list_what_describe_lists_and_nothing_else(described, pages):
    listed = set()
    for text in pages.values():
        listed |= build.entries(text)
    assert listed == build.expected_entries(described)


def test_a_component_missing_from_the_pages_is_reported(described, pages):
    expected = build.expected_entries(described)
    assert build.missing(pages, expected) == []
    assert ("processing", "onoff") in expected
    without = {
        name: text.replace("### `onoff`", "### `renamed`")
        for name, text in pages.items()
    }
    assert build.missing(without, expected) == ["processing: onoff"]
    ghost = expected | {("neuron", "ghost_neuron")}
    assert build.missing(pages, ghost) == ["neuron: ghost_neuron"]


def test_a_component_added_to_a_registry_appears_in_the_rendered_pages(described):
    data = copy.deepcopy(described)
    data["components"]["neuron"]["ghost_neuron"] = {
        "class": "sensoryforge.neurons.ghost.GhostNeuron",
        "summary": "A neuron that is not there.",
        "init_params": [],
        "params": [],
    }
    assert ("neuron", "ghost_neuron") in build.entries(
        build.render(data)["components.md"]
    )


# ------------------------------------------------------------------------ rendering


def _with_components(described, components):
    data = copy.deepcopy(described)
    data["components"] = components
    return data


def _entry(**changes):
    entry = {
        "class": "sensoryforge.neurons.toy.Toy",
        "summary": "A toy.",
        "init_params": [],
        "params": [],
    }
    entry.update(changes)
    return entry


def _param(name, **changes):
    param = {
        "name": name,
        "label": name,
        "dtype": "float",
        "default": 1.0,
        "min": None,
        "max": None,
        "step": None,
        "unit": "",
        "choices": None,
        "tooltip": "",
        "help": "",
        "group": "",
        "advanced": False,
    }
    param.update(changes)
    return param


def test_names_that_register_one_class_share_a_section(described):
    other = _entry(**{"class": "sensoryforge.neurons.toy.Other"})
    data = _with_components(
        described, {"neuron": {"Toy": _entry(), "toy_alias": _entry(), "other": other}}
    )
    page = build.render(data)["components.md"]
    assert "### `Toy` / `toy_alias` {" in page
    assert page.count("\n### ") == 2
    assert build.entries(page) == {
        ("neuron", "Toy"),
        ("neuron", "toy_alias"),
        ("neuron", "other"),
    }
    assert "Registered under more than one name: `Toy`, `toy_alias`." in page


def test_a_parameter_row_carries_default_range_unit_and_meaning(described):
    params = [
        _param("theta", min=1e-6, max=1e6, unit="mA", tooltip="Per event"),
        _param("mode", dtype="str", default="zero", choices=["zero", "first"]),
        _param("leak", default=None, min=0.001, advanced=True, help="Long help."),
        _param("on", dtype="bool", default=False, min=0, max=1),
    ]
    page = build.render(
        _with_components(described, {"neuron": {"T": _entry(params=params)}})
    )["components.md"]
    assert "| `theta` | float | `1` | 1e-06 to 1000000 | `mA` | Per event |" in page
    assert '| `mode` | str | `"zero"` | one of `zero`, `first` |  |  |' in page
    assert "| `leak` *(advanced)* | float | `null` | >= 0.001 |  | Long help. |" in page
    assert "| `on` | bool | `false` |  |  |  |" in page


def test_a_longer_help_text_becomes_a_note_under_the_table(described):
    params = [
        _param("theta", tooltip="Per event", help="The reference moves by theta.")
    ]
    page = build.render(
        _with_components(described, {"neuron": {"T": _entry(params=params)}})
    )["components.md"]
    assert "Notes:\n\n- `theta`: The reference moves by theta." in page


def test_a_component_without_a_param_spec_lists_its_constructor(described):
    init = [
        {"name": "dt", "required": False, "default": 0.05},
        {"name": "base", "required": True},
        {"name": "items", "required": False},
    ]
    page = build.render(
        _with_components(described, {"solver": {"s": _entry(init_params=init)}})
    )["components.md"]
    assert "It declares no `ParamSpec`; these are its constructor arguments." in page
    assert "| `dt` |  | `0.05` |" in page
    assert "| `base` | yes |  |" in page
    assert "| `items` |  | *not shown* |" in page


def test_a_name_that_selects_nothing_is_said_so(described):
    none = _entry(**{"class": "builtins.NoneType"})
    none["summary"] = "The type of the None singleton."
    data = _with_components(
        described, {"filter": {"none": none, "identity": dict(none)}}
    )
    page = build.render(data)["components.md"]
    assert "No filter: the registry holds `None` under these names." in page
    assert "None singleton" not in page


def test_text_is_escaped_for_markdown(described):
    params = [_param("p", tooltip="a | b <c> *d* $e$ `x | y`")]
    page = build.render(
        _with_components(described, {"neuron": {"T": _entry(params=params)}})
    )["components.md"]
    assert r"a \| b &lt;c&gt; \*d\* \$e\$ `x | y`" in page


def test_a_plugin_component_is_refused_unless_allowed(described):
    plugin = _entry(**{"class": "my_package.plugin.Mine"})
    data = _with_components(described, {"neuron": {"mine": plugin}})
    with pytest.raises(ValueError, match="my_package.plugin.Mine"):
        build.render(data)
    assert "mine" in build.render(data, allow_plugins=True)["components.md"]


def test_a_description_in_another_format_or_without_a_part_is_refused(described):
    with pytest.raises(ValueError, match="format"):
        build.render({**described, "format": "sensoryforge-describe/2"})
    cut = {k: v for k, v in described.items() if k != "recipes"}
    with pytest.raises(ValueError, match="recipes"):
        build.render(cut)


# --------------------------------------------------------------------------- clean


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Unset by default (D-ce22df3). Set it", "Unset by default. Set it"),
        ("8 ms follows Kandel Ch. 21 (D-015).", "8 ms follows Kandel Ch. 21."),
        ("receptor coordinates (Wave M3).", "receptor coordinates."),
        ("Wave M4 generality demo: one grid", "Generality demo: one grid"),
        ("poisson (F-050); None draws", "poisson; None draws"),
        ("seed (D-015, F-050) and", "seed and"),
        ("Named control arm (D-019): identical", "Named control arm: identical"),
        ("see :class:`~a.B` and ``c``", "see `a.B` and `c`"),
        ("two\n   lines", "two lines"),
        ("A-1 and F-1234 are not ids", "A-1 and F-1234 are not ids"),
    ],
)
def test_clean_removes_ledger_ids_and_labels(text, expected):
    assert build.clean(text) == expected


def test_clean_refuses_an_id_it_cannot_remove():
    with pytest.raises(ValueError, match=r"components\.neuron\.T\.help.*F-050"):
        build.clean("see F-050 for why", "components.neuron.T.help")


def test_no_committed_page_carries_a_ledger_id_or_roadmap_label(committed):
    pattern = re.compile(
        r"\b[FDCRA]-(?:[0-9a-f]{7}|\d{2,3})\b|\bWave [A-Z]|\bPhase \d|docs_root"
    )
    for name, text in committed.items():
        found = pattern.search(text)
        assert (
            not found
        ), f"docs/reference/{name} carries {found.group(0)!r} (D-41f73ea)"


# ------------------------------------------------------------------ links and anchors


def _anchors(text):
    return set(re.findall(r"\{ #([\w-]+) \}", text))


def test_links_into_the_reference_pages_name_anchors_that_exist(committed):
    anchors = {name: _anchors(text) for name, text in committed.items()}
    reference = (DOCS / "reference").resolve()
    bad = []
    for path in DOCS.rglob("*.md"):
        rel = path.relative_to(DOCS)
        if rel.parts[0] == "development":
            continue
        text = path.read_text(encoding="utf-8")
        for page, anchor in re.findall(r"\]\(([^)\s#]*)#([\w-]+)\)", text):
            if page:
                target = (path.parent / page).resolve()
                name = target.name if target.parent == reference else None
            else:  # an anchor in the same page
                name = path.name if path.parent.resolve() == reference else None
            if name in anchors and anchor not in anchors[name]:
                bad.append(f"{rel}: {page}#{anchor}")
    assert not bad, "links to anchors the generated pages lack: " + "; ".join(bad)


def test_the_pages_link_to_pages_that_exist(committed):
    bad = []
    for name, text in committed.items():
        for target in re.findall(r"\]\(([^)#\s]+\.md)(?:#[\w-]+)?\)", text):
            if not (DOCS / "reference" / target).resolve().is_file():
                bad.append(f"{name}: {target}")
    assert not bad, "links to missing pages: " + "; ".join(bad)


def test_the_pages_are_in_the_navigation():
    nav = (REPO / "mkdocs.yml").read_text(encoding="utf-8")
    for name in build.PAGES:
        assert f": reference/{name}\n" in nav, name


def test_the_config_keys_the_pages_name_exist():
    from sensoryforge.config import schema

    fields = {
        f.name
        for obj in vars(schema).values()
        if isinstance(obj, type) and dataclasses.is_dataclass(obj)
        for f in dataclasses.fields(obj)
    }
    fields |= {"type"}  # a key of the solver and stimulus dicts
    named = set()
    for _, used in build.KINDS.values():
        named |= set(re.findall(r"`(\w+)`", used))
    assert named <= fields, sorted(named - fields)


# ------------------------------------------------------------------- the script itself


def test_the_script_needs_only_the_standard_library_and_the_installed_package():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names), imported
    source = SCRIPT.read_text(encoding="utf-8")
    assert "sys.path" not in source and "sensoryforge.cli" in source


def test_the_script_header_is_a_true_one():
    assert file_headers.check_module(SCRIPT, REPO) == []


def test_check_passes_and_names_a_stale_page(tmp_path, described):
    saved = tmp_path / "describe.json"
    saved.write_text(json.dumps(described), encoding="utf-8")
    out = tmp_path / "pages"
    args = [sys.executable, str(SCRIPT), "--from-json", str(saved), "--out", str(out)]
    assert (
        subprocess.run(args, capture_output=True, text=True, cwd=tmp_path).returncode
        == 0
    )
    ok = subprocess.run(
        args + ["--check"], capture_output=True, text=True, cwd=tmp_path
    )
    assert ok.returncode == 0 and ok.stdout == ""
    (out / "recipes.md").write_text("old\n", encoding="utf-8")
    (out / "formats.md").unlink()
    stale = subprocess.run(
        args + ["--check"], capture_output=True, text=True, cwd=tmp_path
    )
    assert stale.returncode == 1
    assert "recipes.md" in stale.stdout and "formats.md" in stale.stdout
    assert REGENERATE in stale.stdout
    for name in ("components.md", "world_language.md", "evaluation.md"):
        assert name not in stale.stdout


def test_the_script_writes_the_committed_pages_from_any_folder(tmp_path, committed):
    """No cwd, no repository layout: the package is found as an installed one."""
    out = tmp_path / "pages"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    for name, text in committed.items():
        assert (out / name).read_text(encoding="utf-8") == text, name
