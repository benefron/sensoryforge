"""Check the agent header on a module's docstring (roadmap B5).

A header is a block of ``Tag: value`` lines at the end of a module docstring. It tells
an agent that opens the file what the file is for, which tests pin it and what not to
change without. Every claim a program can check is checked here: tests and contract
pages exist, named entry points are defined, topics come from one closed list. The
convention, with the tags and topics, is ``docs/developer_guide/file_headers.md``.

Topic: testing
Tests: tests/unit/test_file_headers.py
Entry points: TAGS, TOPICS, parse_header, check_module, needs_header, source_modules,
    docstring_of
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

#: The tag lines a header may carry, in the order the page documents them.
TAGS = (
    "Topic",
    "Tests",
    "Entry points",
    "Pinned by",
    "Do not change without",
    "Shapes",
    "Units",
)

#: Tags every header has; the others appear where they apply.
REQUIRED_TAGS = ("Topic", "Tests", "Entry points")

#: A module shorter than this (lines), a package ``__init__`` or a ``__main__`` is glue.
MIN_LINES = 40

#: The closed list of topics; one search for ``Topic: <word>`` lists a topic's files.
TOPICS = {
    "engine": "running a config: the simulation engine and its pipelines",
    "batch": "running many stimuli or a data set, and writing results",
    "config": "the config schema, its defaults and their resolvers",
    "registry": "component registries and registration",
    "plugin": "plugin discovery and scaffolding",
    "cli": "the `sensoryforge` command line",
    "receptive-field": "receptive-field banks and their builders",
    "neuron": "neuron models and their solvers",
    "filter": "temporal filters",
    "stimulus": "stimuli and their rendering",
    "grid": "sensor arrays and their arrangements",
    "event-converter": "level-crossing and sigma-delta converters",
    "world": "worlds, sampling and rendering draws",
    "dataset": "data sets: splits, probes and entries",
    "bundle": "the data bundle a run writes",
    "evaluate": "the evaluation pipeline as a whole",
    "audit": "the blind audit of generated data",
    "compare": "comparing audited data with the declaration",
    "checks": "reference-free checks",
    "server": "the local evaluation server and its exports",
    "gui": "the Qt application",
    "agents": "agent skills, setup and instruction files",
    "recipe": "shipped recipes, presets and their gains",
    "testing": "test helpers shipped in the package",
    "legacy": "kept for old configs; new work uses the canonical path",
}

_TAG_LINE = re.compile(r"^(%s): +(\S.*)$" % "|".join(re.escape(t) for t in TAGS))
_LEDGER_ID = re.compile(r"^[A-Z]-[0-9A-Za-z]{2,8}$")


def docstring_of(path: Path) -> Optional[str]:
    """Return a module's docstring, or ``None`` if it has none or does not parse."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except SyntaxError:
        return None
    return ast.get_docstring(tree)


def _tag_pairs(docstring: str) -> List[Tuple[str, str]]:
    """Return ``(tag, value)`` for each tag line, with its indented continuations."""
    pairs: List[Tuple[str, str]] = []
    open_tag = False
    for line in docstring.splitlines():
        match = _TAG_LINE.match(line)
        if match:
            pairs.append((match.group(1), match.group(2).strip()))
            open_tag = True
        elif open_tag and line[:1] in (" ", "\t") and line.strip():
            tag, value = pairs[-1]
            pairs[-1] = (tag, f"{value} {line.strip()}")
        else:
            open_tag = False
    return pairs


def parse_header(docstring: str) -> Dict[str, str]:
    """Read the header tags out of a module docstring.

    Args:
        docstring: The module's docstring text.

    Returns:
        ``{tag: value}`` for each tag line present; an indented line after a tag
        continues it. Empty when the docstring has no header.
    """
    return dict(_tag_pairs(docstring))


def _items(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _definitions(body: List[ast.stmt]) -> Dict[str, ast.AST]:
    """Names a module or class body defines (also under ``if`` and ``try``)."""
    found: Dict[str, ast.AST] = {}
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found[node.target.id] = node
        elif isinstance(node, ast.If):
            found.update(_definitions(node.body + node.orelse))
        elif isinstance(node, ast.Try):
            inner = node.body + node.orelse + node.finalbody
            for handler in node.handlers:
                inner += handler.body
            found.update(_definitions(inner))
    return found


def _entry_point_problems(tree: ast.Module, names: List[str]) -> List[str]:
    top = _definitions(tree.body)
    problems = []
    for name in names:
        owner, _, member = name.partition(".")
        node = top.get(owner)
        if node is None:
            problems.append(f"Entry points: {name!r} is not defined in the module")
        elif member and not (
            isinstance(node, ast.ClassDef) and member in _definitions(node.body)
        ):
            problems.append(f"Entry points: {name!r} is not a member of {owner}")
    return problems


def check_module(path: Path, root: Path) -> List[str]:
    """Check one module's header against the tree.

    Args:
        path: The module file.
        root: The repository root that the header's paths are relative to.

    Returns:
        A list of problems; empty when the header is present and true.
    """
    path, root = Path(path), Path(root)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"does not parse: {exc}"]
    docstring = ast.get_docstring(tree)
    if not docstring:
        return ["no module docstring"]
    pairs = _tag_pairs(docstring)
    seen = [tag for tag, _ in pairs]
    problems = [
        f"{tag}: given twice" for tag in sorted(set(seen)) if seen.count(tag) > 1
    ]
    header = dict(pairs)
    problems += [f"missing tag {tag}:" for tag in REQUIRED_TAGS if tag not in header]

    for topic in _items(header.get("Topic", "")):
        if topic not in TOPICS:
            problems.append(f"Topic: unknown topic {topic!r}")
    for item in _items(header.get("Tests", "")):
        if not (root / item).exists():
            problems.append(f"Tests: {item} does not exist")
    for item in _items(header.get("Pinned by", "")):
        if not (root / item).exists() and not _LEDGER_ID.match(item):
            problems.append(f"Pinned by: {item!r} is neither a path nor a ledger id")
    problems += _entry_point_problems(tree, _items(header.get("Entry points", "")))
    return problems


def needs_header(path: Path) -> bool:
    """Return whether a module is big enough, and not glue, to carry a header."""
    path = Path(path)
    if path.name in ("__init__.py", "__main__.py"):
        return False
    return len(path.read_text(encoding="utf-8").splitlines()) >= MIN_LINES


def source_modules(root: Path) -> List[Path]:
    """Return the modules under ``<root>/sensoryforge`` that need a header, sorted."""
    skip: Set[str] = {"__pycache__"}
    return sorted(
        p
        for p in (Path(root) / "sensoryforge").rglob("*.py")
        if not skip.intersection(p.parts) and needs_header(p)
    )
