#!/usr/bin/env python
"""Write (or check) the generated reference pages in ``docs/reference/``.

The component lists and parameter tables of the docs come from the code, so they cannot
disagree with it. The pages are rendered from ``sensoryforge describe --json`` (the
registries and each component's ``ParamSpec`` list), and
``tests/unit/test_reference_pages.py`` fails when a committed page differs from what
this script writes, or when a registered component is missing from a page.

Usage:
    python scripts/build_reference.py             # rewrite docs/reference/*.md
    python scripts/build_reference.py --check     # exit 1 if a page is stale or missing
    python scripts/build_reference.py --out DIR   # write (or check) in DIR instead
    sensoryforge describe --json > d.json
    python scripts/build_reference.py --from-json d.json   # render a saved description

The script needs only the installed ``sensoryforge`` package and the standard library:
it runs ``python -m sensoryforge.cli describe --json`` in a fresh interpreter (so a
registry another process changed cannot leak in) and reads nothing else from a checkout
except where it writes. Run it in an environment with no plugin packages installed:
``describe`` lists a plugin's components too, and a committed page lists the built-ins
only, so a component whose class lives outside ``sensoryforge`` is refused
(``--allow-plugins`` renders it anyway, for a look). World shapes, patterns and
modulations carry no class in the description, so a plugin's cannot be told apart.

Text that comes from a docstring or a ``ParamSpec`` is cleaned before it is written: a
ledger id or a roadmap label (``D-015``, ``Wave M3``) is removed, because public pages
carry none (D-41f73ea), and one the cleaner cannot remove stops the build with the field
named.

Topic: registry, cli
Tests: tests/unit/test_reference_pages.py
Entry points: render, load_description, entries, expected_entries, missing, clean, main,
    PAGES
Pinned by: docs/reference/components.md, docs/reference/world_language.md,
    docs/reference/evaluation.md, docs/reference/recipes.md,
    docs/reference/formats.md, D-41f73ea
Do not change without: regenerating the pages in the same commit (the test fails
    otherwise); the script never holds its own copy of a parameter list, a default, a
    unit or a component name, only what ``describe`` returns.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "reference"

#: The ``format`` tag of the description this script reads.
DESCRIBE_FORMAT = "sensoryforge-describe/1"
REGENERATE = "python scripts/build_reference.py"

#: Page file -> its title; the order is the order the pages are written in.
PAGES = {
    "components.md": "Component Reference",
    "world_language.md": "World Language Reference",
    "evaluation.md": "Evaluation Packs Reference",
    "recipes.md": "Recipe Reference",
    "formats.md": "File Format Tags",
}

#: Component kind -> (section title, where a name of this kind is written in a config).
#: The config keys are checked against the config dataclasses by the test.
KINDS = {
    "neuron": ("Neuron models", "Written as the `neuron_model` of a population."),
    "filter": ("Filters", "Written as the `filter_method` of a population."),
    "stimulus": ("Stimuli", "Written as the `type` of the `stimulus`."),
    "solver": (
        "Solvers",
        "Written as the `type` of a population's `solver_config`, or of the "
        "`solver` of the `simulation`.",
    ),
    "grid": ("Grid arrangements", "Written as the `arrangement` of a grid."),
    "innervation": (
        "Receptive-field builders",
        "Written as the `innervation_method` of a population, or the `method` of "
        "an input's `rf`.",
    ),
    "processing": (
        "Processing layers",
        "Written as the `method` of an entry in an input's `processing` list.",
    ),
}

#: Extra sentence under a kind's heading (what the reader must know to read its tables).
KIND_NOTES = {
    "neuron": (
        "A default below is the model's own. For a model that has type presets, a "
        "population starts from the preset its `neuron_type` selects and applies "
        "`model_params` on top, so the values a population runs can differ from "
        "these defaults (`resolve_neuron_params` in "
        "`sensoryforge/config/defaults.py`)."
    ),
    "stimulus": (
        "The `layered` stimulus is built from the shapes, patterns, motions and "
        "modulations in the [World Language Reference](world_language.md); "
        "[Designing stimuli](../user_guide/designing_stimuli.md) explains how."
    ),
}

#: Section title, ``id`` and the key of the description for each part of the world page.
WORLD_PARTS = (
    ("shape", "Shapes", "shapes"),
    ("pattern", "Patterns", "patterns"),
    ("modulation", "Modulations", "modulations"),
    ("distribution", "Distributions", "distributions"),
    ("class-kind", "Class kinds", "class_kinds"),
)

EVALUATION_PARTS = (
    ("metric-pack", "Metric packs", "metric_packs"),
    ("check-pack", "Check packs", "check_packs"),
    ("declared-adapter", "Declared adapters", "declared_adapters"),
)

#: What each format tag in the description tags.
FORMAT_LABELS = {
    "world": "A world file",
    "dataset": "A data-set file",
    "bundle": "A data bundle without stored voltage probes or a sparse population file",
    "bundle_with_voltage_probes": "A data bundle with stored voltage probes",
    "bundle_with_sparse_weights": (
        "A data bundle with a sparse population file (it also covers stored "
        "voltage probes)"
    ),
    "describe": "The description printed by `sensoryforge describe --json`",
}

# ------------------------------------------------------------------ text cleaning

_LEDGER_ID = r"[FDCRA]-(?:[0-9a-f]{7}|\d{2,3})"
_LABEL = r"(?:Wave [A-Z]\d*|Phase [A-Z0-9]+)"
_REF = rf"(?:{_LEDGER_ID}|{_LABEL})"
_PAREN_REF = re.compile(rf"\s*\(\s*{_REF}(?:\s*[,;/]\s*{_REF})*\s*\)")
_LEAD_LABEL = re.compile(rf"^{_LABEL}\s+")
_STRAY_REF = re.compile(rf"\b{_REF}\b")
_ROLE = re.compile(r":[a-z]+:`~?([^`]+)`")
_RST_CODE = re.compile(r"``([^`]+)``")


def clean(text: str, where: str = "text") -> str:
    """Prose from a docstring or ``ParamSpec`` as one line, fit for a public page.

    Removes a ledger id or roadmap label in parentheses (``(D-015)``, ``(Wave M3)``) or
    leading a sentence, turns ``:role:`x``` and ````x```` into ```x```, and joins the
    lines.

    Args:
        text: The source text.
        where: Where it came from, named in the error.

    Returns:
        The cleaned text.

    Raises:
        ValueError: If a ledger id or label is left that the rules above cannot remove.
    """
    out = _RST_CODE.sub(r"`\1`", _ROLE.sub(r"`\1`", text))
    out = _PAREN_REF.sub("", out)
    lead = _LEAD_LABEL.sub("", out)
    if lead != out:
        out = lead[:1].upper() + lead[1:]
    out = " ".join(out.split())
    found = _STRAY_REF.search(out)
    if found:
        raise ValueError(
            f"{where}: {found.group(0)!r} is a ledger id or roadmap label, and public "
            "pages carry none (D-41f73ea); remove it from the source text, or teach "
            "scripts/build_reference.py `clean` to strip this shape"
        )
    return out


def _prose(text: str, where: str) -> str:
    """:func:`clean` text, then escape what Markdown would read as markup."""
    parts = clean(text, where).split("`")
    for i in range(0, len(parts), 2):  # outside the code spans
        parts[i] = (
            parts[i]
            .replace("|", "\\|")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("*", "\\*")
            .replace("$", "\\$")
        )
    return "`".join(parts)


def _num(value: float) -> str:
    """A number as short text: ``1e-06``, ``0.05``, ``1000000``, ``-65``."""
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value in (float("inf"), float("-inf")):
            return "inf" if value > 0 else "-inf"
        text = repr(value)
        return text[:-2] if text.endswith(".0") else text
    return str(value)


def _value(value: Any) -> str:
    """A default as YAML/JSON spells it: ``null``, ``true``, ``"zero"``, ``[0, 1]``."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return _num(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(v) for v in value) + "]"
    return json.dumps(value, sort_keys=True)


def _code(text: str) -> str:
    """``text`` as an inline code span (a backtick inside it gets a longer fence)."""
    fence = "``" if "`" in text else "`"
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """A Markdown table; cells are already escaped."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _anchor(kind: str, name: str) -> str:
    """The ``id`` of an entry's heading: ``neuron-level-crossing``."""
    return f"{kind}-{_slug(name)}"


# --------------------------------------------------------------- parameter tables


def _range(param: Dict[str, Any]) -> str:
    """The allowed values of a parameter: ``one of ...``, ``0 to 1``, ``>= 0``."""
    if param.get("choices"):
        return "one of " + ", ".join(
            _code(c if isinstance(c, str) else _value(c)) for c in param["choices"]
        )
    if param["dtype"] == "bool":
        return ""
    lo, hi = param["min"], param["max"]
    if lo is not None and hi is not None:
        return f"{_num(lo)} to {_num(hi)}"
    if lo is not None:
        return f">= {_num(lo)}"
    if hi is not None:
        return f"<= {_num(hi)}"
    return ""


def _param_table(params: Sequence[Dict[str, Any]], where: str) -> List[str]:
    """The table of a component's ``ParamSpec`` list, then its longer notes."""
    rows = []
    for p in params:
        name = _code(p["name"]) + (" *(advanced)*" if p["advanced"] else "")
        rows.append(
            [
                name,
                p["dtype"],
                _code(_value(p["default"])),
                _range(p),
                _code(p["unit"]) if p["unit"] else "",
                _prose(p["tooltip"] or p["help"], f"{where}.{p['name']}.meaning"),
            ]
        )
    lines = _table(["Parameter", "Type", "Default", "Range", "Unit", "Meaning"], rows)
    notes = []
    for p in params:
        # The tooltip is the meaning; a longer help, when there is both, is a note.
        if p["tooltip"] and p["help"]:
            help_ = clean(p["help"], f"{where}.{p['name']}.help")
            if help_ != clean(p["tooltip"], f"{where}.{p['name']}.tooltip"):
                notes.append(f"- {_code(p['name'])}: {_prose(help_, where)}")
    if notes:
        lines += ["", "Notes:", ""] + notes
    return lines


def _init_table(init_params: Sequence[Dict[str, Any]]) -> List[str]:
    """The table of a constructor's arguments, for a component with no ``ParamSpec``."""
    rows = []
    for p in init_params:
        if "default" in p:
            default = _code(_value(p["default"]))
        else:
            default = "" if p["required"] else "*not shown*"
        rows.append([_code(p["name"]), "yes" if p["required"] else "", default])
    return _table(["Argument", "Required", "Default"], rows)


def _heading(level: int, names: Sequence[str], anchor: str) -> str:
    text = " / ".join(_code(n) for n in names)
    return f"{'#' * level} {text} {{ #{anchor} }}"


def _section(level: int, title: str, anchor: str) -> str:
    return f"{'#' * level} {title} {{ #{anchor} }}"


def _groups(entries: Dict[str, Dict[str, Any]]) -> List[List[str]]:
    """Names that register the same class with the same parameters, grouped.

    Within a group the names that are plain identifiers come first, then by name
    ignoring case; the groups are in the order of their first name.
    """
    by_key: Dict[str, List[str]] = {}
    for name, entry in entries.items():
        key = json.dumps(
            [
                entry["class"],
                entry["params"],
                entry["init_params"],
                entry.get("params_error"),
            ],
            sort_keys=True,
        )
        by_key.setdefault(key, []).append(name)
    groups = [
        sorted(names, key=lambda n: (not n.isidentifier(), n.casefold()))
        for names in by_key.values()
    ]
    return sorted(groups, key=lambda g: g[0].casefold())


def _component_entry(kind: str, names: List[str], entry: Dict[str, Any]) -> List[str]:
    """One component's section: what it is, its class and its parameters."""
    where = f"components.{kind}.{names[0]}"
    lines = [_heading(3, names, _anchor(kind, names[0])), ""]
    if entry["class"] == "builtins.NoneType":
        lines += [f"No {kind}: the registry holds `None` under these names.", ""]
    else:
        if entry["summary"]:
            lines += [_prose(entry["summary"], f"{where}.summary"), ""]
        lines += [f"Class: {_code(entry['class'])}", ""]
    if len(names) > 1 and entry["class"] != "builtins.NoneType":
        lines += [
            "Registered under more than one name: "
            + ", ".join(_code(n) for n in names)
            + ". They select the same class.",
            "",
        ]
    if entry.get("params_error"):
        lines += [
            "Its parameters could not be listed: "
            + _prose(entry["params_error"], where),
            "",
        ]
    elif entry["params"]:
        lines += _param_table(entry["params"], where) + [""]
    elif entry["init_params"]:
        lines += (
            [
                "It declares no `ParamSpec`; these are its constructor arguments.",
                "",
            ]
            + _init_table(entry["init_params"])
            + [""]
        )
    elif entry["class"] != "builtins.NoneType":
        lines += ["It takes no parameters.", ""]
    return lines


# ----------------------------------------------------------------------- the pages

_GENERATED = (
    f"<!-- Generated by `{REGENERATE}` from `sensoryforge describe --json`. "
    "Do not edit: change the code,\nthen run the command again. -->"
)


def _page(title: str, intro: Sequence[str], body: Sequence[str]) -> str:
    out = [f"# {title}", "", _GENERATED, "", *intro, "", *body]
    return "\n".join(out).rstrip("\n") + "\n"


def _is_plugin(class_path: str) -> bool:
    return class_path != "builtins.NoneType" and not class_path.startswith(
        "sensoryforge."
    )


def _check_builtin(data: Dict[str, Any]) -> None:
    """Refuse a component whose class lives outside ``sensoryforge``.

    Raises:
        ValueError: Naming the first such component.
    """
    found = []
    for kind, entries in data["components"].items():
        found += [(f"{kind} {n!r}", e["class"]) for n, e in entries.items()]
    for _, _, key in EVALUATION_PARTS:
        found += [
            (f"{key} {n!r}", e["class"]) for n, e in data["evaluation"][key].items()
        ]
    found += [
        (f"class kind {n!r}", e["class"])
        for n, e in data["world"]["class_kinds"].items()
    ]
    for what, class_path in found:
        if _is_plugin(class_path):
            raise ValueError(
                f"{what} is registered by {class_path}, not by sensoryforge: a "
                "committed page lists the built-ins only. Run in an environment "
                "with no plugin packages installed, or pass --allow-plugins to "
                "render it anyway"
            )


def _components_page(data: Dict[str, Any]) -> str:
    intro = [
        "Every neuron model, filter, stimulus, solver, grid arrangement, "
        "receptive-field builder and processing layer that SensoryForge registers, "
        "with its parameters, defaults, ranges and units.",
        "",
        "This page is generated from the component registries and from each "
        "component's `ParamSpec` list, so a name, default, range or unit here is the "
        "component's own. `sensoryforge describe --json` prints the same data for a "
        "program ([Describing SensoryForge to a program]"
        "(../user_guide/describe.md)). The page lists the components that ship with "
        "SensoryForge; a plugin's components appear in `sensoryforge describe` once "
        "the plugin is installed. A component you add to SensoryForge appears here "
        f"after `{REGENERATE}`, and a test fails until it does.",
        "",
        "## Reading the tables { #reading }",
        "",
        "- **Names.** A name is matched without regard to case. Names under one "
        "heading are the same component.",
        "- **Default.** Written as YAML writes it: `null` is unset, a string is "
        "quoted.",
        "- **Range.** The smallest and largest value the parameter allows, or the "
        "choices it allows.",
        "- **Unit.** Empty means the parameter is dimensionless.",
        "- **Advanced.** The GUI shows an advanced parameter only when its "
        "**Advanced** toggle is on.",
        "- **No `ParamSpec`.** A component that declares none has a table of its "
        "constructor arguments instead. That is not a component with no arguments.",
    ]
    body: List[str] = []
    known = [k for k in KINDS if k in data["components"]]
    others = sorted(k for k in data["components"] if k not in KINDS)
    for kind in known + others:
        title, used = KINDS.get(kind, (kind.capitalize(), ""))
        entries = data["components"][kind]
        groups = _groups(entries)
        body += [_section(2, title, kind), ""]
        if used:
            body += [used, ""]
        if kind in KIND_NOTES:
            body += [KIND_NOTES[kind], ""]
        glance = []
        for names in groups:
            entry = entries[names[0]]
            what = (
                f"No {kind}"
                if entry["class"] == "builtins.NoneType"
                else _prose(entry["summary"], f"components.{kind}.{names[0]}.summary")
            )
            link = f"[{_code(names[0])}](#{_anchor(kind, names[0])})"
            others = "".join(f", {_code(n)}" for n in names[1:])
            glance.append([link + others, what])
        body += _table(["Name", "What it is"], glance) + [""]
        for names in groups:
            body += _component_entry(kind, names, entries[names[0]])
    return _page(PAGES["components.md"], intro, body)


def _value_list(values: Sequence[Any]) -> str:
    return ", ".join(_code(str(v)) for v in values)


def _world_page(data: Dict[str, Any]) -> str:
    world = data["world"]
    intro = [
        "What a `world:` file and a data-set file may name: the shapes, patterns, "
        "modulations, distributions and class kinds, each with its parameters, and "
        "the keys a world, class, layer, axis and split may carry. "
        "[Worlds and Data Sets](../user_guide/worlds.md) explains how a world is "
        "written; [World Engine Contract](world_contract.md) states what is "
        "guaranteed.",
        "",
        "This page is generated from the world registries by "
        f"`{REGENERATE}`, so it lists what the loader accepts. Layered stimuli use "
        "the same shapes, patterns and modulations. A parameter table reads as on "
        "the [Component Reference](components.md#reading).",
        "",
        f"The world format is {_code(world['format'])} and the data-set format "
        f"{_code(world['dataset_format'])}.",
    ]
    body: List[str] = [_section(2, "File keys", "keys"), ""]
    body += _table(
        ["Where", "Keys"],
        [
            ["A world file", _value_list(world["top_keys"])],
            [
                "A class (under `classes` or `held_out`)",
                _value_list(world["class_keys"]),
            ],
            ["The `sessions` block of a world", _value_list(world["session_keys"])],
            ["A class's `layer`", _value_list(world["layer_keys"])],
            [
                "An axis (a value under `axes`, `defaults` or a group)",
                _value_list(world["axis_keys"]),
            ],
            ["The `kind` of a layer's `motion`", _value_list(world["motion_kinds"])],
            [
                "The `dist` of an axis with a `range`",
                _value_list(world["numeric_distributions"]),
            ],
        ],
    )
    body += ["", _section(3, "Splits of a data set", "splits"), ""]
    body += [
        "A split with no `kind:` takes the kind its name gives. Each kind takes "
        "the keys below besides `kind`, `repeats` and `noise_repeats`.",
        "",
    ]
    body += _table(
        ["Split", "Kind"],
        [[_code(n), _code(k)] for n, k in world["split_kinds"].items()],
    )
    body += [""]
    body += _table(
        ["Kind", "Keys"],
        [[_code(k), _value_list(v)] for k, v in world["split_keys"].items()],
    )
    body += [""]
    for kind, title, key in WORLD_PARTS:
        body += [_section(2, title, kind), ""]
        for name, entry in world[key].items():
            where = f"world.{key}.{name}"
            body += [_heading(3, [name], _anchor(kind, name)), ""]
            body += _world_facts(key, entry, where)
            if "params" in entry:
                if entry["params"]:
                    body += _param_table(entry["params"], where) + [""]
                else:
                    body += ["It takes no parameters.", ""]
    return _page(PAGES["world_language.md"], intro, body)


def _world_facts(key: str, entry: Dict[str, Any], where: str) -> List[str]:
    """The sentences about one shape, pattern, distribution or class kind."""
    facts: List[str] = []
    if entry.get("summary"):
        facts.append(_prose(entry["summary"], f"{where}.summary"))
    if entry.get("unbounded"):
        facts.append("Drawn once over the plane, not at each pattern position.")
    if entry.get("indenter"):
        facts.append("A rigid indenter: its `amplitude` is the peak depth.")
    if key == "patterns" and entry.get("placement"):
        facts.append("`x_mm` and `y_mm` place the pattern.")
    if key == "distributions":
        if entry["quantile"]:
            facts.append(
                "A quantile function: an axis that uses it can be stratified into "
                "equal-probability bins."
            )
        if entry["finite_support"]:
            facts.append("It has a finite support, which stratifying needs.")
        if entry["declares_bounds"]:
            facts.append(
                "It declares bounds, a closed interval holding every number it "
                "draws, which the loader checks against the field the axis binds."
            )
    if key == "class_kinds":
        facts.append(f"Class: {_code(entry['class'])}")
    return [f for fact in facts for f in (fact, "")]


def _evaluation_page(data: Dict[str, Any]) -> str:
    intro = [
        "The metric packs (`--packs`), check packs (`--check-packs`) and declared "
        "adapters (`--adapter`) that `sensoryforge evaluate run` can use. "
        "[Evaluating a World](../user_guide/evaluate_pipeline.md) explains the "
        "options, and [Evaluation Report Contract](report_contract.md) states what a "
        "report holds.",
        "",
        f"This page is generated from the evaluation registries by `{REGENERATE}`.",
    ]
    body: List[str] = []
    for kind, title, key in EVALUATION_PARTS:
        body += [_section(2, title, kind), ""]
        for name, entry in data["evaluation"][key].items():
            where = f"evaluation.{key}.{name}"
            body += [_heading(3, [name], _anchor(kind, name)), ""]
            if entry["summary"]:
                body += [_prose(entry["summary"], f"{where}.summary"), ""]
            body += [f"Class: {_code(entry['class'])}", ""]
    return _page(PAGES["evaluation.md"], intro, body)


def _recipe_rows(items: Sequence[Dict[str, Any]]) -> List[str]:
    columns: List[str] = []
    for item in items:
        columns += [k for k in item if k not in columns]
    rows = [
        [_code(_value(item[c])) if c in item else "" for c in columns] for item in items
    ]
    return _table([_code(c) for c in columns], rows)


def _recipes_page(data: Dict[str, Any]) -> str:
    intro = [
        "The recipes that ship with SensoryForge: complete canonical configs you "
        "can run with `sensoryforge run --preset NAME`. Each entry is a digest of "
        "the recipe's grids and populations; [Presets](../user_guide/presets.md) "
        "explains how to run one, override it and add one, and [Units and gains]"
        "(../user_guide/units_and_gains.md) says how its gains were set.",
        "",
        "This page is generated from the shipped presets by " f"`{REGENERATE}`.",
    ]
    body: List[str] = []
    for name, recipe in data["recipes"].items():
        where = f"recipes.{name}"
        body += [_heading(2, [name], _anchor("recipe", name)), ""]
        if recipe["description"]:
            body += [_prose(recipe["description"], f"{where}.description"), ""]
        if recipe["grids"]:
            body += ["Grids:", ""] + _recipe_rows(recipe["grids"]) + [""]
        if recipe["populations"]:
            body += ["Populations:", ""] + _recipe_rows(recipe["populations"]) + [""]
    return _page(PAGES["recipes.md"], intro, body)


def _formats_page(data: Dict[str, Any]) -> str:
    intro = [
        "The version tag of each file format SensoryForge reads or writes. The "
        "[World Engine Contract](world_contract.md) states what the world, data-set "
        "and bundle formats guarantee and when their tags change.",
        "",
        f"This page is generated by `{REGENERATE}`.",
    ]
    rows = [
        [FORMAT_LABELS.get(key, _prose(key, "formats")), _code(str(tag))]
        for key, tag in data["formats"].items()
        if key != "sensoryforge"
    ]
    body = _table(["Format", "Tag"], rows) + [""]
    return _page(PAGES["formats.md"], intro, body)


# ----------------------------------------------------------------------- the API


def render(data: Dict[str, Any], allow_plugins: bool = False) -> Dict[str, str]:
    """Every generated page, from a ``sensoryforge describe --json`` description.

    Args:
        data: The description (``sensoryforge.describe.describe()``, or its JSON).
        allow_plugins: Render components that a plugin registers too.

    Returns:
        ``{file name: page text}``, one entry per name in :data:`PAGES`.

    Raises:
        ValueError: If the description is not format ``sensoryforge-describe/1``,
            lacks a part, holds a plugin's component (unless ``allow_plugins``) or
            holds text with a ledger id the cleaner cannot remove.
    """
    if data.get("format") != DESCRIBE_FORMAT:
        raise ValueError(
            f"the description has format {data.get('format')!r}; this script reads "
            f"{DESCRIBE_FORMAT!r}"
        )
    for part in ("formats", "components", "world", "evaluation", "recipes"):
        if part not in data:
            raise ValueError(f"the description has no {part!r} part")
    if not allow_plugins:
        _check_builtin(data)
    return {
        "components.md": _components_page(data),
        "world_language.md": _world_page(data),
        "evaluation.md": _evaluation_page(data),
        "recipes.md": _recipes_page(data),
        "formats.md": _formats_page(data),
    }


def load_description(source: Optional[str] = None) -> Dict[str, Any]:
    """The description: from ``sensoryforge describe --json``, a file or stdin.

    Args:
        source: ``None`` runs ``python -m sensoryforge.cli describe --json`` in a new
            interpreter; ``"-"`` reads standard input; anything else is a JSON file.

    Returns:
        The parsed description.

    Raises:
        RuntimeError: If the command fails.
        ValueError: If the text is not JSON.
    """
    if source is None:
        done = subprocess.run(
            [sys.executable, "-m", "sensoryforge.cli", "describe", "--json"],
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            raise RuntimeError(
                f"sensoryforge describe --json failed ({done.returncode}): "
                f"{done.stderr.strip()[-500:]}"
            )
        text = done.stdout
    elif source == "-":
        text = sys.stdin.read()
    else:
        text = Path(source).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the description is not JSON: {exc}") from None


_SECTION = re.compile(r"^## .*\{ #([\w-]+) \}\s*$")
_ENTRY = re.compile(r"^### (.*?)\s*\{ #[\w-]+ \}\s*$")
_RECIPE = re.compile(r"^## (.*?)\s*\{ #recipe-[\w-]+ \}\s*$")


def entries(page: str) -> Set[Tuple[str, str]]:
    """The ``(kind, name)`` of every entry a page lists, read from its headings.

    Args:
        page: The text of a generated page.

    Returns:
        For each ``###`` heading, the section's ``id`` (``neuron``, ``shape``) and each
        name in the heading; for a recipe, ``("recipe", name)``.
    """
    found: Set[Tuple[str, str]] = set()
    kind = ""
    for line in page.splitlines():
        recipe = _RECIPE.match(line)
        if recipe:
            found |= {("recipe", n) for n in re.findall(r"`([^`]+)`", recipe.group(1))}
            continue
        section = _SECTION.match(line)
        if section:
            kind = section.group(1)
            continue
        entry = _ENTRY.match(line)
        if entry:
            found |= {(kind, n) for n in re.findall(r"`([^`]+)`", entry.group(1))}
    return found


def expected_entries(data: Dict[str, Any]) -> Set[Tuple[str, str]]:
    """The ``(kind, name)`` of everything a description registers, as pages list it."""
    expected: Set[Tuple[str, str]] = set()
    for kind, comps in data["components"].items():
        expected |= {(kind, n) for n in comps}
    for kind, _, key in WORLD_PARTS:
        expected |= {(kind, n) for n in data["world"][key]}
    for kind, _, key in EVALUATION_PARTS:
        expected |= {(kind, n) for n in data["evaluation"][key]}
    expected |= {("recipe", n) for n in data["recipes"]}
    return expected


def missing(pages: Dict[str, str], expected: Set[Tuple[str, str]]) -> List[str]:
    """What ``expected`` names that no page lists, as sorted ``kind: name`` lines."""
    listed: Set[Tuple[str, str]] = set()
    for text in pages.values():
        listed |= entries(text)
    return [f"{kind}: {name}" for kind, name in sorted(expected - listed)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Write or check the pages; the exit code is 1 when a page is stale."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if out of date")
    parser.add_argument("--out", type=Path, default=OUT, help="the pages' folder")
    parser.add_argument(
        "--from-json", metavar="FILE", help="a saved description ('-' for stdin)"
    )
    parser.add_argument(
        "--allow-plugins", action="store_true", help="also render a plugin's components"
    )
    args = parser.parse_args(argv)
    pages = render(load_description(args.from_json), args.allow_plugins)
    if args.check:
        stale = [
            name
            for name, text in pages.items()
            if not (args.out / name).is_file()
            or (args.out / name).read_text(encoding="utf-8") != text
        ]
        if stale:
            print(f"stale or missing: {', '.join(stale)}; run {REGENERATE}")
            return 1
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in pages.items():
        (args.out / name).write_text(text, encoding="utf-8")
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
